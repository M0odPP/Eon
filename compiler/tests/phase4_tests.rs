use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use eonc::cfg::{CfgBuiltinOp, CfgConst};
use eonc::codegen_x64::{X64Options, fn_sym_name};
use eonc::diag::{DiagBag, ErrorCode};
use eonc::eir::{EirCallee, EirFunction, EirInstKind, EirProgram, EirValue};
use eonc::elf64::ElfMode;
use eonc::span::{Interner, SourceMap};
use eonc::{
    EmitNativeResult, check_files_in_mem, compile_files_to_x64_elf, interpret_files,
    load_eon_sources, run_compiled_binary, write_executable_elf,
};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempElf {
    bin_path: PathBuf,
}

impl TempElf {
    fn new(tag: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        Self {
            bin_path: std::env::temp_dir().join(format!("eon_p4_{tag}_{pid}_{id}.out")),
        }
    }
}

impl Drop for TempElf {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.bin_path);
    }
}

const RT_HAL_SRC: &str = r#"
module rt.hal @hal;

extern "C" fn eon_rt_print_str(s: *str);
extern "C" fn eon_rt_print_u64(v: u64);
extern "C" fn eon_rt_print_i64(v: i64);
extern "C" fn eon_rt_print_nl();

pub fn print_str(s: *str) { eon_rt_print_str(s); }
pub fn print_u64(v: u64) { eon_rt_print_u64(v); }
pub fn print_i64(v: i64) { eon_rt_print_i64(v); }
pub fn print_nl() { eon_rt_print_nl(); }
"#;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn core_file(name: &str) -> (String, String) {
    let p = repo_root().join("core").join(name);
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("failed to read core file `{}`: {e}", p.display()));
    (p.to_string_lossy().into_owned(), txt)
}

struct ParityResult {
    res: EmitNativeResult,
    interner: Interner,
    code: i32,
    stdout: String,
    stderr: String,
}

fn compile_and_run_native(
    tag: &str,
    owned: &[(String, Vec<u8>)],
    sanitize: bool,
) -> (EmitNativeResult, Interner, i32, String, String) {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize,
        needed_libs: Vec::new(),
    };
    let res = compile_files_to_x64_elf(owned, &mut sm, &mut interner, &mut diags, &opts, None)
        .unwrap_or_else(|e| {
            panic!("unexpected x86_64 compiler errors for {tag} (sanitize={sanitize}):\n{e}");
        });
    let tmp = TempElf::new(tag);
    write_executable_elf(&res.elf.bytes, &tmp.bin_path).unwrap_or_else(|e| {
        panic!("write ELF failed for {tag} (sanitize={sanitize}): {e}");
    });
    let (code, stdout, stderr) = run_compiled_binary(&tmp.bin_path)
        .unwrap_or_else(|e| panic!("run binary failed for {tag} (sanitize={sanitize}): {e}"));
    (res, interner, code, stdout, stderr)
}

fn compile_and_run_parity_bytes(
    name: &str,
    owned: &[(String, Vec<u8>)],
    normalize: fn(&str) -> String,
) -> ParityResult {
    let mut sm_i = SourceMap::new();
    let mut int_i = Interner::new();
    let mut diags_i = DiagBag::new();
    let interp = interpret_files(owned, &mut sm_i, &mut int_i, &mut diags_i)
        .unwrap_or_else(|e| panic!("EIR-Interp failed for {name}:\n{e}"));

    let (res_n, int_n, n_code, n_out, n_err) =
        compile_and_run_native(&format!("{name}_pie"), owned, false);
    let (_res_s, _int_s, s_code, s_out, s_err) =
        compile_and_run_native(&format!("{name}_san"), owned, true);

    assert_eq!(
        (interp.exit_code, n_code, s_code),
        (n_code, n_code, n_code),
        "exit code mismatch across Interp ({}) / PIE ({}) / Sanitize ({}) for {name}\nPIE stderr:\n{n_err}\nSanitize stderr:\n{s_err}",
        interp.exit_code,
        n_code,
        s_code
    );
    assert_eq!(
        normalize(&interp.stdout),
        normalize(&n_out),
        "stdout mismatch between EIR-Interp and Native PIE for {name}\n--- INTERP ---\n{}\n--- PIE ---\n{n_out}",
        interp.stdout
    );
    assert_eq!(
        normalize(&n_out),
        normalize(&s_out),
        "stdout mismatch between Native PIE and Native --sanitize for {name}\n--- PIE ---\n{n_out}\n--- SANITIZE ---\n{s_out}"
    );
    assert!(s_err.is_empty(), "unexpected --sanitize stderr for {name}:\n{s_err}");

    ParityResult {
        res: res_n,
        interner: int_n,
        code: n_code,
        stdout: n_out,
        stderr: n_err,
    }
}

fn identity_out(s: &str) -> String {
    s.to_owned()
}

fn strip_elapsed_ns(s: &str) -> String {
    s.lines()
        .map(|l| l.split(" elapsed_ns=").next().unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

fn compile_and_run_inline(name: &str, files: &[(String, String)]) -> ParityResult {
    let owned: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|(p, s)| (p.clone(), s.as_bytes().to_vec()))
        .collect();
    compile_and_run_parity_bytes(name, &owned, identity_out)
}

fn check_inline_diags(files: &[(String, String)]) -> (Vec<ErrorCode>, String) {
    let owned: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|(p, s)| (p.clone(), s.as_bytes().to_vec()))
        .collect();
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let _ = check_files_in_mem(&owned, &mut sm, &mut interner, &mut diags);
    let codes = diags.diagnostics().iter().map(|d| d.code).collect();
    let rendered = diags.render_all(&sm);
    (codes, rendered)
}

fn find_eir_fn<'a>(eir: &'a EirProgram, interner: &Interner, name: &str) -> &'a EirFunction {
    eir.functions
        .iter()
        .find(|f| interner.get(f.name) == name)
        .unwrap_or_else(|| panic!("could not find EIR function `{name}`"))
}

fn find_section<'a>(elf: &'a [u8], sec_name: &str) -> Option<(u64, u64, &'a [u8])> {
    let shoff = u64::from_le_bytes(elf[40..48].try_into().ok()?) as usize;
    let shentsize = u16::from_le_bytes(elf[58..60].try_into().ok()?) as usize;
    let shnum = u16::from_le_bytes(elf[60..62].try_into().ok()?) as usize;
    let shstrndx = u16::from_le_bytes(elf[62..64].try_into().ok()?) as usize;
    let shstr_hdr = shoff + shstrndx * shentsize;
    let shstr_off = u64::from_le_bytes(elf[shstr_hdr + 24..shstr_hdr + 32].try_into().ok()?) as usize;
    let shstr_sz = u64::from_le_bytes(elf[shstr_hdr + 32..shstr_hdr + 40].try_into().ok()?) as usize;
    let shstr = &elf[shstr_off..shstr_off + shstr_sz];

    for i in 0..shnum {
        let h = shoff + i * shentsize;
        let name_off = u32::from_le_bytes(elf[h..h + 4].try_into().ok()?) as usize;
        let end = shstr[name_off..].iter().position(|&b| b == 0).unwrap_or(0);
        let name = std::str::from_utf8(&shstr[name_off..name_off + end]).ok()?;
        if name == sec_name {
            let addr = u64::from_le_bytes(elf[h + 16..h + 24].try_into().ok()?);
            let off = u64::from_le_bytes(elf[h + 24..h + 32].try_into().ok()?) as usize;
            let sz = u64::from_le_bytes(elf[h + 32..h + 40].try_into().ok()?) as usize;
            return Some((addr, addr + sz as u64, &elf[off..off + sz]));
        }
    }
    None
}

fn resolve_const_tags_in_fn(f: &EirFunction) -> BTreeMap<EirValue, i128> {
    let mut tags = BTreeMap::new();
    for _pass in 0..2 {
        for blk in &f.blocks {
            for inst in &blk.insts {
                match &inst.kind {
                    EirInstKind::Const {
                        dst,
                        konst: CfgConst::EnumTag { tag_val, .. },
                        ..
                    } => {
                        tags.insert(*dst, *tag_val);
                    }
                    EirInstKind::CopyVal { dst, src } => {
                        if let Some(&tv) = tags.get(src) {
                            tags.insert(*dst, tv);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    tags
}

#[test]
fn exit_gate_4_layout_proto_uses_core_mem_pools() {
    let proto_dir = repo_root().join("examples").join("layout_proto");
    let main_txt = std::fs::read_to_string(proto_dir.join("main.eon")).unwrap();
    assert!(
        main_txt.contains("import core.mem;")
            && main_txt.contains("mem.init_slot_pool")
            && main_txt.contains("mem.init_frame_pool"),
        "layout_proto must use core.mem library pool implementations"
    );

    let loaded = load_eon_sources(&[proto_dir]).expect("load layout_proto + core.mem");
    let parity = compile_and_run_parity_bytes("layout_proto_core", &loaded, strip_elapsed_ns);
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert!(parity.stdout.contains("box tag=1 pos=(0,0) size=24x8"));
    assert!(parity.stdout.contains(".=*********=++++++++++=."));
    assert!(parity.stdout.contains("err_cleanup_leaks=0 final_live_allocs=0"));

    let (ro_lo, ro_hi, rodata_bytes) = find_section(&parity.res.elf.bytes, ".rodata")
        .expect("expected `.rodata` section in layout_proto ELF");
    assert!(
        rodata_bytes.windows(b"box tag=".len()).any(|w| w == b"box tag="),
        "expected module string constant bytes in `.rodata`"
    );
    assert!(
        parity
            .res
            .elf
            .symbol_vaddrs
            .values()
            .any(|&va| va >= ro_lo && va < ro_hi),
        "expected at least one symbol inside `.rodata` [0x{ro_lo:x}..0x{ro_hi:x})"
    );
}

#[test]
fn exit_gate_4_json_parser_e2e() {
    let json_dir = repo_root().join("examples").join("json_parser");
    let loaded = load_eon_sources(&[json_dir]).expect("load json_parser + core.mem + core.str");
    let parity = compile_and_run_parity_bytes("json_parser", &loaded, identity_out);
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert_eq!(
        parity.stdout.trim(),
        "fields=4 nodes=9 str0_len=3 arr_len=4 int_sum=-8 bad_rejected=1 live=0"
    );
}

#[test]
fn exit_gate_4_multithreaded_par_for_on_slot_pool() {
    let main_src = r#"
module app.par_slot_pool;

import core.mem;
import rt.hal;

pub pool Particle {
    pub x: u64;
    pub vx: u64;
    pub out_x: u64;
}

pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | StaleId {
    var particles = try mem.init_slot_pool.[Particle](alloc, 1200_usize);
    errdefer particles.destroy(alloc);

    var live_count: usize = 0_usize;
    var expected_sum: u64 = 0_u64;

    // Insert 1024 particles, removing every 8th to exercise generational tombstones in parallel!
    for i in 0_usize..1024_usize {
        let iu = i.widen.[u64]();
        let id = try particles.insert(alloc, .{
            x: iu *% 3_u64,
            vx: (iu & 15_u64) +% 1_u64,
            out_x: 0_u64,
        });
        if (i & 7_usize) == 0_usize {
            let _removed = try particles.remove(id);
        } else {
            live_count = live_count +% 1_usize;
            let expected = (iu *% 3_u64) +% (((iu & 15_u64) +% 1_u64) *% 10_u64);
            expected_sum = expected_sum +% expected;
        }
    }

    // Real multi-threaded `par for` over `SlotPool` (> 256 threshold -> spawns pthread workers)!
    par for id in particles.each_mut() {
        let px = particles.x[id];
        let pvx = particles.vx[id];
        particles.out_x[id] = px +% (pvx *% 10_u64);
    }

    var actual_sum: u64 = 0_u64;
    var seen_live: usize = 0_usize;
    for i in 0_usize..particles.cap() {
        if let id = particles.id_at(i) {
            actual_sum = actual_sum +% particles.out_x[id];
            seen_live = seen_live +% 1_usize;
        }
    }

    particles.destroy(alloc);

    hal.print_str("live=");
    hal.print_u64(seen_live.widen.[u64]());
    hal.print_str(" match=");
    hal.print_u64(if actual_sum == expected_sum && seen_live == live_count { 1_u64 } else { 0_u64 });
    hal.print_str(" sum=");
    hal.print_u64(actual_sum);
    hal.print_str(" leaks=");
    hal.print_u64(alloc.live_allocs().widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "par_slot_pool",
        &[
            core_file("mem.eon"),
            ("rt.hal.eon".into(), RT_HAL_SRC.into()),
            ("main.eon".into(), main_src.into()),
        ],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    let main_fn = find_eir_fn(&parity.res.eir, &parity.interner, "main");
    assert!(
        !main_fn.par_for_regions.is_empty(),
        "expected non-empty `par_for_regions` on `main` in EIR"
    );
    assert!(
        parity.res.elf.symbol_vaddrs.contains_key("eon_par_for"),
        "expected native ELF symbol `eon_par_for` for parallel dispatch"
    );
    assert!(
        parity.stdout.trim().starts_with("live=896 match=1 sum=")
            && parity.stdout.trim().ends_with(" leaks=0"),
        "unexpected output: {}",
        parity.stdout
    );
}

#[test]
fn exit_gate_4_generics_layout_sharing_and_definition_site_check() {
    let main_src = r#"
module app.layout_sharing;

import core.mem;
import rt.hal;

@hot
fn pick_first[T](a: T, _b: T) -> T {
    return a;
}

pub fn main() -> i32 {
    var ua: u64 = 10_u64;
    var ub: u64 = 20_u64;
    mem.swap(&mut ua, &mut ub);

    var ia: i64 = -5_i64;
    var ib: i64 = 42_i64;
    mem.swap(&mut ia, &mut ib);

    var fa: f64 = 1.5_f64;
    var fb: f64 = 9.25_f64;
    mem.swap(&mut fa, &mut fb);

    let p1 = pick_first(100_u64, 200_u64);
    let p2 = pick_first(300_i64, 400_i64);

    hal.print_u64(ua);
    hal.print_str(",");
    hal.print_i64(ia);
    hal.print_str(",");
    hal.print_u64(fa.trunc_to_int.[i64]().bitcast.[u64]());
    hal.print_str(",");
    hal.print_u64(p1);
    hal.print_str(",");
    hal.print_i64(p2);
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "layout_sharing",
        &[
            core_file("mem.eon"),
            ("rt.hal.eon".into(), RT_HAL_SRC.into()),
            ("main.eon".into(), main_src.into()),
        ],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert_eq!(parity.stdout.trim(), "20,42,9,100,300");

    // Verify Layout Sharing deduplicated swap[u64], swap[i64], swap[f64] into ONE shared function!
    assert!(
        parity
            .res
            .eir
            .layout_shared_groups
            .iter()
            .any(|g| g.layout_tag == "ls_8_8"
                && g.member_insts.len() == 3
                && parity
                    .res
                    .eir
                    .by_key
                    .get(&g.canonical_key)
                    .map(|&idx| parity.interner.get(parity.res.eir.functions[idx].name) == "swap")
                    .unwrap_or(false)),
        "expected 3-way Layout Sharing group `ls_8_8` for swap[u64/i64/f64], got: {:?}",
        parity.res.eir.layout_shared_groups
    );
    assert!(
        parity.res.elf.symbol_vaddrs.contains_key("eon_fn_swap__ls_8_8"),
        "expected single Layout-Shared ELF symbol `eon_fn_swap__ls_8_8`, got: {:?}",
        parity.res.elf.symbol_vaddrs.keys().collect::<Vec<_>>()
    );
    let swap_eir_fns: Vec<&EirFunction> = parity
        .res
        .eir
        .functions
        .iter()
        .filter(|f| parity.interner.get(f.name) == "swap")
        .collect();
    assert_eq!(
        swap_eir_fns.len(),
        1,
        "expected duplicate swap monomorphizations to be merged into 1 EIR function"
    );
    assert_eq!(
        swap_eir_fns[0].layout_shared_tag.as_deref(),
        Some("ls_8_8")
    );
    assert!(
        !parity
            .res
            .elf
            .symbol_vaddrs
            .keys()
            .any(|k| k != "eon_fn_swap__ls_8_8" && k.contains("swap")),
        "unexpected unmerged swap symbol in ELF symbol_vaddrs"
    );

    // Verify `@hot fn pick_first[T]` was specialized per type (NOT layout-shared)!
    let pick_fns: Vec<&EirFunction> = parity
        .res
        .eir
        .functions
        .iter()
        .filter(|f| parity.interner.get(f.name) == "pick_first")
        .collect();
    assert_eq!(
        pick_fns.len(),
        2,
        "expected `@hot` generic `pick_first` to be specialized per type (2 EIR functions)"
    );
    for pf in &pick_fns {
        assert!(
            pf.layout_shared_tag.is_none(),
            "`@hot` generic `pick_first` must not be layout-shared"
        );
        let sym = fn_sym_name(pf.key, &parity.res.eir, &parity.interner);
        assert!(
            parity.res.elf.symbol_vaddrs.contains_key(&sym),
            "expected specialized ELF symbol `{sym}` for `@hot` pick_first"
        );
    }

    // Negative definition-site generic checking: error caught WITHOUT instantiating `broken`!
    let (codes, out) = check_inline_diags(&[(
        "def_site_fail.eon".into(),
        r#"
module test.def_site;
pub fn broken[T](x: *T) -> u64 {
    return x.nonexistent_method();
}
"#
        .into(),
    )]);
    assert!(
        codes.contains(&ErrorCode::E0501),
        "expected E0501 definition-site generic error, got {codes:?}\n{out}"
    );

    // Negative call-site bound check: concrete type missing required interface!
    let (codes, out) = check_inline_diags(&[(
        "bound_fail.eon".into(),
        r#"
module test.bound_fail;
pub iface Tag {
    fn tag(self: *Self) -> u32 @pure @noalloc @stack(32);
}
pub struct Plain { pub x: u32; }
pub fn get_tag[T: Tag](v: *T) -> u32 {
    return v.tag();
}
pub fn caller() -> u32 {
    let p = Plain.{ x: 1_u32 };
    return get_tag(&p);
}
"#
        .into(),
    )]);
    assert!(
        codes.contains(&ErrorCode::E0501),
        "expected E0501 unsatisfied bound error, got {codes:?}\n{out}"
    );
}

#[test]
fn exit_gate_4_iface_dyn_vtable_stack_and_object_safety() {
    let main_src = r#"
module app.dyn_dispatch;

import rt.hal;

pub iface Hasher {
    @noalloc @stack(64)
    fn write_u64(self: *mut Self, v: u64);

    @pure @noalloc @stack(64)
    fn finish(self: *Self) -> u64;
}

pub struct AddHasher {
    pub acc: u64;
}

impl Hasher for AddHasher {
    @noalloc @stack(64)
    fn write_u64(self: *mut Self, v: u64) {
        self.acc = self.acc +% v;
    }

    @pure @noalloc @stack(64)
    fn finish(self: *Self) -> u64 {
        return self.acc;
    }
}

pub struct XorHasher {
    pub acc: u64;
}

impl Hasher for XorHasher {
    @noalloc @stack(64)
    fn write_u64(self: *mut Self, v: u64) {
        self.acc = self.acc ^ v;
    }

    @pure @noalloc @stack(64)
    fn finish(self: *Self) -> u64 {
        return self.acc;
    }
}

@noalloc @stack(256)
fn feed_dyn(h: *mut dyn Hasher) -> u64 {
    h.write_u64(10_u64);
    h.write_u64(3_u64);
    return h.finish();
}

pub fn main() -> i32 {
    var a = AddHasher.{ acc: 0_u64 };
    var x = XorHasher.{ acc: 0_u64 };
    let ra = feed_dyn(&mut a);
    let rx = feed_dyn(&mut x);
    hal.print_u64(ra);
    hal.print_str(",");
    hal.print_u64(rx);
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "dyn_dispatch",
        &[
            ("rt.hal.eon".into(), RT_HAL_SRC.into()),
            ("main.eon".into(), main_src.into()),
        ],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    // Add: 10 + 3 = 13; Xor: 10 ^ 3 = 9
    assert_eq!(parity.stdout.trim(), "13,9");

    let feed_fn = find_eir_fn(&parity.res.eir, &parity.interner, "feed_dyn");
    let dyn_calls = feed_fn
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|inst| {
            matches!(
                &inst.kind,
                EirInstKind::Call {
                    callee: EirCallee::DynCall { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        dyn_calls, 3,
        "expected 3 `EirCallee::DynCall` instructions in `feed_dyn`"
    );

    let (dr_lo, dr_hi, _) = find_section(&parity.res.elf.bytes, ".data.rel.ro")
        .expect("expected `.data.rel.ro` section for dyn vtables");
    let vtable_syms: Vec<(&String, &u64)> = parity
        .res
        .elf
        .symbol_vaddrs
        .iter()
        .filter(|(name, _)| name.starts_with("eon_vtable_"))
        .collect();
    assert!(
        vtable_syms.len() >= 2,
        "expected at least 2 `eon_vtable_*` symbols in ELF, got {vtable_syms:?}"
    );
    for (name, vaddr) in &vtable_syms {
        let vaddr = **vaddr;
        assert!(
            vaddr >= dr_lo && vaddr < dr_hi,
            "expected vtable `{name}` (0x{vaddr:x}) to live in `.data.rel.ro` [0x{dr_lo:x}..0x{dr_hi:x})"
        );
    }

    // Negative @stack(N) check through `dyn`: caller budget 64 < own(16) + dyn ceiling(128) -> E0701!
    let (codes, out) = check_inline_diags(&[(
        "dyn_stack_fail.eon".into(),
        r#"
module test.dyn_stack;
pub iface Heavy {
    @noalloc @stack(128)
    fn work(self: *Self) -> u64;
}
@noalloc @stack(64)
pub fn too_tight(h: *dyn Heavy) -> u64 {
    return h.work();
}
"#
        .into(),
    )]);
    assert!(
        codes.contains(&ErrorCode::E0701),
        "expected E0701 stack overflow across `dyn` call, got {codes:?}\n{out}"
    );

    // Negative E0512: non-object-safe interface (returns `Self` by value) used in `*dyn`!
    let (codes, out) = check_inline_diags(&[(
        "dyn_unsafe.eon".into(),
        r#"
module test.dyn_unsafe;
pub iface Cloneable {
    fn clone_val(self: *Self) -> Self;
}
pub fn bad_dyn(c: *dyn Cloneable) {
    let _ = c;
}
"#
        .into(),
    )]);
    assert!(
        codes.contains(&ErrorCode::E0512),
        "expected E0512 non-dyn-compatible interface error, got {codes:?}\n{out}"
    );
}

#[test]
fn exit_gate_4_core_sync_str_io_hal_atomics_and_restrict() {
    let main_src = r#"
module app.core_all;

import core.str;
import core.sync;
import core.io;
import core.hal;
import rt.out;

fn bump_both(a: *mut u64, b: *u64) {
    a.* = a.* +% b.*;
}

pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | ArithError | Utf8Error {
    // 1. core.str: UTF-8 validation (ASCII + valid 2/3-byte UTF-8 + reject invalid) + StrBuf + parse_u64
    let good_utf8: [5]u8 = [b'E', 0xC3_u8, 0xA9_u8, b'!', b'1'];
    try str.validate_utf8(&good_utf8);
    let bad_utf8: [2]u8 = [0xFF_u8, 0xFE_u8];
    var bad_utf8_rejected: u64 = 0_u64;
    str.validate_utf8(&bad_utf8) catch {
        bad_utf8_rejected = 1_u64;
    };
    var sb = try str.StrBuf.init(alloc, 64_usize);
    errdefer sb.destroy(alloc);
    try sb.push_str("v=");
    try sb.push_u64(1234_u64);
    let sb_len = sb.len();
    let parsed_n = try str.parse_u64("420".as_bytes());
    sb.destroy(alloc);

    // 2. core.sync: AtomicCounter + SpinLock[u64] + linear SpinGuard + Chase-Lev WorkDeque
    var ctr = sync.AtomicCounter.init(10_u64);
    let prev = ctr.fetch_add(5_u64, .AcqRel);
    let cas_ok = ctr.compare_exchange(15_u64, 99_u64);
    let after = ctr.load(.Acquire);

    var sl = sync.SpinLock.[u64].init(7_u64);
    let guard = sl.lock();
    sl.set_val(77_u64);
    let locked_v = sl.get_val();
    guard.unlock(&mut sl);

    var dq = sync.WorkDeque.init();
    let _p1 = dq.push_bottom(101_u64);
    let _p2 = dq.push_bottom(202_u64);
    let stolen = dq.steal_top() orelse 0_u64;
    let popped = dq.pop_bottom() orelse 0_u64;

    // 3. core.io: IoRing SQE/CQE + FiberContext
    var ring = io.IoRing.init();
    let _sub = ring.submit(io.IoSqe.{
        op: .Read,
        flags: 0_u8,
        pad: 0_u16,
        fd: 3_u32,
        offset: 0_u64,
        buf_addr: 0x1000_u64,
        buf_len: 512_u32,
        op_flags: 0_u32,
        user_data: 88_u64,
        ext: [0_u8; 24],
    });
    let _proc = ring.process_one();
    let cqe_tag: u64 = if let cqe = ring.poll_cqe() { cqe.user_data } else { 0_u64 };
    let fctx = io.FiberContext.init(0x8000_u64, 4096_u64);
    let fregs = fctx.saved_reg_count();

    // 4. core.hal: PhysAddr alignment + DMA barriers
    let pa = PhysAddr.from_usize(0x1005_usize);
    let aligned = hal.phys_align_up(pa, 4096_u64).as_u64();
    var dma = hal.DmaBuf.{
        phys: pa,
        virt: VirtAddr.from_usize(0x2005_usize),
        dma: DmaAddr.from_usize(0x3005_usize),
        len: 64_usize,
    };
    hal.dma_barrier_to_device(&dma);
    hal.dma_barrier_from_device(&mut dma);
    hal.mmio_barrier();

    var x_val: u64 = 5_u64;
    let y_val: u64 = 6_u64;
    bump_both(&mut x_val, &y_val);

    out.print_u64(sb_len.widen.[u64]());
    out.print_str(",");
    out.print_u64(bad_utf8_rejected);
    out.print_str(",");
    out.print_u64(parsed_n);
    out.print_str(",");
    out.print_u64(prev);
    out.print_str(",");
    out.print_u64(if cas_ok { after } else { 0_u64 });
    out.print_str(",");
    out.print_u64(locked_v);
    out.print_str(",");
    out.print_u64(stolen);
    out.print_str(",");
    out.print_u64(popped);
    out.print_str(",");
    out.print_u64(cqe_tag);
    out.print_str(",");
    out.print_u64(fregs);
    out.print_str(",");
    out.print_u64(aligned);
    out.print_str(",");
    out.print_u64(x_val);
    out.print_str(",");
    out.print_u64(alloc.live_allocs().widen.[u64]());
    out.print_nl();
    return 0_i32;
}
"#;
    let rt_src = RT_HAL_SRC.replace("module rt.hal @hal;", "module rt.out @hal;");

    let parity = compile_and_run_inline(
        "core_all",
        &[
            core_file("str.eon"),
            core_file("sync.hal.eon"),
            core_file("io.eon"),
            core_file("hal.hal.eon"),
            ("rt.out.hal.eon".into(), rt_src),
            ("main.eon".into(), main_src.into()),
        ],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    // sb_len=6, bad_utf8=1, parsed_n=420, prev=10, after=99, locked_v=77, stolen=101, popped=202, cqe_tag=88, fregs=7, aligned=8192, x_val=11, live=0
    assert_eq!(
        parity.stdout.trim(),
        "6,1,420,10,99,77,101,202,88,7,8192,11,0"
    );

    // Verify EIR Atomic builtin ops and memory orderings (`Relaxed`=0, `Acquire`=1, `Release`=2, `AcqRel`=3, `SeqCst`=4)
    let mut seen_atomic_load = false;
    let mut seen_atomic_store = false;
    let mut seen_atomic_fetch_add = false;
    let mut seen_atomic_cas = false;
    let mut seen_orderings: BTreeSet<i128> = BTreeSet::new();

    for f in &parity.res.eir.functions {
        let val_tags = resolve_const_tags_in_fn(f);
        for blk in &f.blocks {
            for inst in &blk.insts {
                match &inst.kind {
                    EirInstKind::BuiltinCall { op, args, .. } => match op {
                        CfgBuiltinOp::AtomicLoad(_) => {
                            seen_atomic_load = true;
                            if let Some(arg) = args.get(1)
                                && let Some(&tv) = val_tags.get(arg)
                            {
                                seen_orderings.insert(tv);
                            }
                        }
                        CfgBuiltinOp::AtomicStore(_) => {
                            seen_atomic_store = true;
                            if let Some(arg) = args.get(2)
                                && let Some(&tv) = val_tags.get(arg)
                            {
                                seen_orderings.insert(tv);
                            }
                        }
                        CfgBuiltinOp::AtomicFetchAdd(_) => {
                            seen_atomic_fetch_add = true;
                            if let Some(arg) = args.get(2)
                                && let Some(&tv) = val_tags.get(arg)
                            {
                                seen_orderings.insert(tv);
                            }
                        }
                        CfgBuiltinOp::AtomicCas(_) => {
                            seen_atomic_cas = true;
                            for idx in [3usize, 4usize] {
                                if let Some(arg) = args.get(idx)
                                    && let Some(&tv) = val_tags.get(arg)
                                {
                                    seen_orderings.insert(tv);
                                }
                            }
                        }
                        _ => {}
                    },
                    EirInstKind::Call { args, .. } => {
                        for arg in args {
                            if let Some(&tv) = val_tags.get(arg)
                                && matches!(
                                    parity.res.check.env.get(f.value_tys[arg.0 as usize]),
                                    eonc::types::Ty::Builtin(eonc::resolve::BuiltinKind::Ordering, _)
                                )
                            {
                                seen_orderings.insert(tv);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        seen_atomic_load && seen_atomic_store && seen_atomic_fetch_add && seen_atomic_cas,
        "expected AtomicLoad, AtomicStore, AtomicFetchAdd, and AtomicCas BuiltinCalls in EIR"
    );
    for (ord_val, ord_name) in [
        (0i128, "Relaxed"),
        (1i128, "Acquire"),
        (2i128, "Release"),
        (3i128, "AcqRel"),
        (4i128, "SeqCst"),
    ] {
        assert!(
            seen_orderings.contains(&ord_val),
            "expected Ordering.{ord_name} ({ord_val}) in EIR atomic calls, got {seen_orderings:?}"
        );
    }

    // Verify `noalias_params` on `bump_both(a: *mut u64, b: *u64)`!
    let bump_fn = find_eir_fn(&parity.res.eir, &parity.interner, "bump_both");
    assert!(
        !bump_fn.noalias_params.is_empty(),
        "expected non-empty `noalias_params` on `bump_both`, got {:?}",
        bump_fn.noalias_params
    );

    // Negative E0610: non-lock-free Atomic[BigStruct]
    let (codes, out) = check_inline_diags(&[(
        "bad_atomic.eon".into(),
        r#"
module test.bad_atomic;
pub struct Big { pub a: u64; pub b: u64; pub c: u64; }
pub fn f(x: *Atomic[Big]) { let _ = x; }
"#
        .into(),
    )]);
    assert!(
        codes.contains(&ErrorCode::E0610),
        "expected E0610 for non-lock-free Atomic[Big], got {codes:?}\n{out}"
    );

    // Negative E0805: MmioReg access violation (.WriteOnly read, .ReadOnly write)
    let (codes, out) = check_inline_diags(&[(
        "bad_mmio.hal.eon".into(),
        r#"
module test.bad_mmio @hal;
pub struct Dev {
    pub ro: MmioReg[u32, .ReadOnly];
    pub wo: MmioReg[u32, .WriteOnly];
}
pub fn bad_access(d: *mut Dev) -> u32 {
    raw {
        d.ro.write(1_u32);
        return d.wo.read();
    }
}
"#
        .into(),
    )]);
    assert_eq!(
        codes.iter().filter(|&&c| c == ErrorCode::E0805).count(),
        2,
        "expected two E0805 MmioReg access violations, got {codes:?}\n{out}"
    );

    // Negative E0109: forgotten linear SpinGuard!
    let (codes, out) = check_inline_diags(&[
        core_file("sync.hal.eon"),
        (
            "leak_guard.eon".into(),
            r#"
module test.leak_guard;
import core.sync;
pub fn bad_lock() {
    var sl = sync.SpinLock.[u64].init(1_u64);
    let _g = sl.lock();
}
"#
            .into(),
        ),
    ]);
    assert!(
        codes.contains(&ErrorCode::E0109),
        "expected E0109 for leaked linear SpinGuard, got {codes:?}\n{out}"
    );
}
