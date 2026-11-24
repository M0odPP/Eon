use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use eonc::ast::BinaryOp;
use eonc::cfg::CfgBuiltinOp;
use eonc::codegen_x64::X64Options;
use eonc::diag::{DiagBag, ErrorCode};
use eonc::eir::{EirFunction, EirInstKind, EirPlace, EirProgram, EirProj};
use eonc::elf64::ElfMode;
use eonc::span::{Interner, SourceMap};
use eonc::types::TypeEnv;
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
            bin_path: std::env::temp_dir().join(format!("eon_p3b_{tag}_{pid}_{id}.out")),
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
extern "C" fn eon_rt_print_nl();
extern "C" fn eon_rt_now_ns() -> u64;

pub fn print_str(s: *str) { eon_rt_print_str(s); }
pub fn print_u64(v: u64) { eon_rt_print_u64(v); }
pub fn print_nl() { eon_rt_print_nl(); }
pub fn now_ns() -> u64 { return eon_rt_now_ns(); }
"#;

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

fn compile_and_run_inline(name: &str, files: &[(&str, &str)]) -> ParityResult {
    let owned: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|(p, s)| ((*p).to_owned(), s.as_bytes().to_vec()))
        .collect();
    compile_and_run_parity_bytes(name, &owned, identity_out)
}

fn check_inline_diags(files: &[(&str, &str)]) -> (Vec<ErrorCode>, String) {
    let owned: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|(p, s)| ((*p).to_owned(), s.as_bytes().to_vec()))
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

fn place_has_index(place: &EirPlace) -> bool {
    place.projs.iter().any(|p| matches!(p, EirProj::Index { .. }))
}

fn assert_all_indexed_and_arith_proven(
    f: &EirFunction,
    env: &TypeEnv,
    fn_label: &str,
) -> (usize, usize) {
    let mut proven_idx = 0usize;
    let mut proven_arith = 0usize;
    for blk in &f.blocks {
        for (ii, inst) in blk.insts.iter().enumerate() {
            match &inst.kind {
                EirInstKind::LoadPlace { place, .. }
                | EirInstKind::StorePlace { place, .. }
                | EirInstKind::PlaceAddr { place, .. }
                    if place_has_index(place) =>
                {
                    assert!(
                        f.proven_safe_insts.contains(&(blk.id, ii)),
                        "{fn_label}: expected place index at ({:?}, {ii}) to be in proven_safe_insts",
                        blk.id
                    );
                    proven_idx += 1;
                }
                EirInstKind::Binary { op, op_ty, .. }
                    if env.is_integer(*op_ty)
                        && matches!(
                            op,
                            BinaryOp::Add
                                | BinaryOp::Sub
                                | BinaryOp::Mul
                                | BinaryOp::Div
                                | BinaryOp::Mod
                                | BinaryOp::Shl
                                | BinaryOp::Shr
                        ) =>
                {
                    assert!(
                        f.proven_safe_insts.contains(&(blk.id, ii)),
                        "{fn_label}: expected integer op {op:?} at ({:?}, {ii}) to be in proven_safe_insts",
                        blk.id
                    );
                    proven_arith += 1;
                }
                _ => {}
            }
        }
    }
    (proven_idx, proven_arith)
}

fn count_checked_binary(f: &EirFunction, target: BinaryOp) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|inst| matches!(&inst.kind, EirInstKind::Binary { op, .. } if *op == target))
        .count()
}

fn count_builtin_calls(f: &EirFunction, pred: impl Fn(&CfgBuiltinOp) -> bool) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|inst| matches!(&inst.kind, EirInstKind::BuiltinCall { op, .. } if pred(op)))
        .count()
}

fn find_section_in_elf<'a>(elf: &'a [u8], sec_name: &str) -> Option<&'a [u8]> {
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
            let off = u64::from_le_bytes(elf[h + 24..h + 32].try_into().ok()?) as usize;
            let sz = u64::from_le_bytes(elf[h + 32..h + 40].try_into().ok()?) as usize;
            return Some(&elf[off..off + sz]);
        }
    }
    None
}

#[test]
fn exit_gate_3b_copy_prefix_and_sum_zero_check() {
    let main_src = r#"
module app.zero_check;

import rt.hal;

pub fn copy_prefix(dst: *mut [u8], src: *[u8], n: usize)
    where n <= dst.len && n <= src.len
{
    for i in 0_usize..n {
        dst[i] = src[i];
    }
}

pub fn sum(xs: *[u32]) -> u64 ! ArithError {
    var acc: u64 = 0_u64;
    var i: usize = 0_usize;
    while i < xs.len {
        acc = try (acc +? xs[i].widen.[u64]());
        i = i + 1_usize;
    }
    return acc;
}

pub fn sum_u64(xs: *[u64]) -> u64 ! ArithError {
    var acc: u64 = 0_u64;
    var i: usize = 0_usize;
    while i < xs.len {
        acc = try (acc +? xs[i]);
        i = i + 1_usize;
    }
    return acc;
}

pub fn main() -> i32 ! ArithError {
    let src: [5]u8 = [10_u8, 20_u8, 30_u8, 40_u8, 50_u8];
    var dst: [5]u8 = [0_u8; 5];
    copy_prefix(&mut dst, &src, 3_usize);

    let nums: [4]u32 = [100_u32, 200_u32, 300_u32, 400_u32];
    let s = try sum(&nums);

    let big: [2]u64 = [18446744073709551615_u64, 5_u64];
    let ovf_hit: u64 = sum_u64(&big) catch 777_u64;

    hal.print_str("dst=");
    hal.print_u64(dst[0].widen.[u64]());
    hal.print_str(",");
    hal.print_u64(dst[1].widen.[u64]());
    hal.print_str(",");
    hal.print_u64(dst[2].widen.[u64]());
    hal.print_str(",");
    hal.print_u64(dst[3].widen.[u64]());
    hal.print_str(" sum=");
    hal.print_u64(s);
    hal.print_str(" ovf=");
    hal.print_u64(ovf_hit);
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "copy_prefix_and_sum",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert_eq!(parity.stdout.trim(), "dst=10,20,30,0 sum=1000 ovf=777");

    // Inspect EIR: copy_prefix must have ZERO bounds checks and ZERO overflow checks.
    let copy_fn = find_eir_fn(&parity.res.eir, &parity.interner, "copy_prefix");
    let (copy_idx, _copy_arith) =
        assert_all_indexed_and_arith_proven(copy_fn, &parity.res.check.env, "copy_prefix");
    assert!(
        copy_idx >= 2,
        "copy_prefix should have at least 2 proven-safe slice index ops (src[i] and dst[i]), got {copy_idx}"
    );
    assert_eq!(
        count_checked_binary(copy_fn, BinaryOp::AddCheck)
            + count_checked_binary(copy_fn, BinaryOp::SubCheck)
            + count_checked_binary(copy_fn, BinaryOp::MulCheck),
        0,
        "copy_prefix should have zero checked arithmetic ops"
    );
    assert_eq!(
        count_builtin_calls(copy_fn, |op| matches!(
            op,
            CfgBuiltinOp::SliceAt { .. }
                | CfgBuiltinOp::ArrayAt { .. }
                | CfgBuiltinOp::SliceGet { .. }
                | CfgBuiltinOp::ArrayGet { .. }
        )),
        0,
        "copy_prefix should have zero runtime bounds-checked slice calls"
    );

    // Inspect EIR: sum must have ZERO bounds checks on xs[i] and ZERO overflow check on i + 1,
    // while retaining explicit `BinaryOp::AddCheck` for `+?`!
    let sum_fn = find_eir_fn(&parity.res.eir, &parity.interner, "sum");
    let (sum_idx, sum_arith) =
        assert_all_indexed_and_arith_proven(sum_fn, &parity.res.check.env, "sum");
    assert!(
        sum_idx >= 1,
        "sum should have at least 1 proven-safe slice index op (xs[i]), got {sum_idx}"
    );
    assert!(
        sum_arith >= 1,
        "sum should have at least 1 proven-safe integer op (i + 1_usize), got {sum_arith}"
    );
    assert_eq!(
        count_checked_binary(sum_fn, BinaryOp::AddCheck),
        1,
        "sum must preserve exactly 1 explicit checked `+?` (and not use AddCheck for proven `i + 1_usize`)"
    );
    assert_eq!(
        count_builtin_calls(sum_fn, |op| matches!(
            op,
            CfgBuiltinOp::SliceAt { .. } | CfgBuiltinOp::ArrayAt { .. }
        )),
        0,
        "sum should have zero runtime bounds-checked slice calls"
    );
}

#[test]
fn exit_gate_3b_contracts_masks_and_safe_div() {
    let main_src = r#"
module app.relational;

import rt.hal;

fn clamp_u32(x: u32, hi: u32) -> u32
    ensures return <= hi
{
    if x > hi {
        return hi;
    }
    return x;
}

fn safe_div(a: u32, b: u32) -> u32 ! ArithError {
    if b == 0_u32 {
        return .DivByZero;
    }
    return a / b;
}

fn lookup_masked(tbl: *[u32], x: u32) -> u32
    where tbl.len >= 8_usize
{
    let idx = (x & 7_u32).widen.[usize]();
    return tbl[idx];
}

pub fn main() -> i32 ! ArithError {
    let arr: [8]u32 = [11_u32, 22_u32, 33_u32, 44_u32, 55_u32, 66_u32, 77_u32, 88_u32];
    let c = clamp_u32(99_u32, 7_u32);
    let v1 = arr[c.widen.[usize]()];
    let v2 = lookup_masked(&arr, 13_u32); // 13 & 7 == 5 -> 66
    let d_ok = try safe_div(84_u32, 2_u32);
    let d_err = safe_div(10_u32, 0_u32) catch 999_u32;

    hal.print_u64(v1.widen.[u64]());
    hal.print_str(",");
    hal.print_u64(v2.widen.[u64]());
    hal.print_str(",");
    hal.print_u64(d_ok.widen.[u64]());
    hal.print_str(",");
    hal.print_u64(d_err.widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "contracts_masks_div",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert_eq!(parity.stdout.trim(), "88,66,42,999");

    let masked_fn = find_eir_fn(&parity.res.eir, &parity.interner, "lookup_masked");
    let (masked_idx, _) =
        assert_all_indexed_and_arith_proven(masked_fn, &parity.res.check.env, "lookup_masked");
    assert!(
        masked_idx >= 1,
        "lookup_masked should have proven-safe indexing for tbl[idx], got {masked_idx}"
    );
    assert_eq!(
        count_checked_binary(masked_fn, BinaryOp::AddCheck)
            + count_checked_binary(masked_fn, BinaryOp::SubCheck)
            + count_checked_binary(masked_fn, BinaryOp::MulCheck),
        0,
        "lookup_masked should have zero checked arithmetic ops"
    );
    assert_eq!(
        count_builtin_calls(masked_fn, |op| matches!(
            op,
            CfgBuiltinOp::SliceAt { .. } | CfgBuiltinOp::ArrayAt { .. }
        )),
        0,
        "lookup_masked should have zero runtime bounds-checked slice calls"
    );
}

#[test]
fn exit_gate_3b_comptime_vm_crc32_type_info_and_comptime_if() {
    let main_src = r#"
module app.comptime_vm;

import rt.hal;

pub struct Packet {
    pub tag: u8;
    pub len: u32;
    pub flags: u16;
}

const CRC32_TABLE: [256]u32 = comptime {
    var tbl: [256]u32 = [0_u32; 256];
    var i: u32 = 0_u32;
    while i < 256_u32 {
        var c: u32 = i;
        var j: u32 = 0_u32;
        while j < 8_u32 {
            if (c & 1_u32) != 0_u32 {
                c = (c >> 1_u32) ^ 0xEDB88320_u32;
            } else {
                c = c >> 1_u32;
            }
            j = j + 1_u32;
        }
        tbl[i] = c;
        i = i + 1_u32;
    }
    tbl
};

fn crc32(data: *[u8]) -> u32 {
    var crc: u32 = 0xFFFFFFFF_u32;
    var i: usize = 0_usize;
    while i < data.len {
        let byte_u32 = data[i].widen.[u32]();
        let idx = ((crc ^ byte_u32) & 0xFF_u32).widen.[usize]();
        crc = CRC32_TABLE[idx] ^ (crc >> 8_u32);
        i = i + 1_usize;
    }
    return crc ^ 0xFFFFFFFF_u32;
}

pub fn main() -> i32 {
    let arch_ok: u64 = comptime if @target_arch() == .X86_64 {
        1_u64
    } else {
        @compile_error("unsupported architecture");
        0_u64
    };

    let info = @type_info(Packet);
    let sz = @size_of(Packet);
    let al = @align_of(Packet);

    let msg: [9]u8 = [b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'];
    let check = crc32(&msg);

    hal.print_str("arch=");
    hal.print_u64(arch_ok);
    hal.print_str(" sz=");
    hal.print_u64(sz.widen.[u64]());
    hal.print_str(" al=");
    hal.print_u64(al.widen.[u64]());
    hal.print_str(" fields=");
    hal.print_u64(info.field_count.widen.[u64]());
    hal.print_str(" t1=");
    hal.print_u64(CRC32_TABLE[1].widen.[u64]());
    hal.print_str(" crc=");
    hal.print_u64(check.widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let parity = compile_and_run_inline(
        "comptime_crc32",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    // IEEE CRC32("123456789") = 0xCBF43926 = 3421780262
    // CRC32_TABLE[1] = 0x77073096 = 1996959894
    // Packet layout (reordered): u32 (4) + u16 (2) + u8 (1) + 1 pad = size 8, align 4, fields 3.
    assert_eq!(
        parity.stdout.trim(),
        "arch=1 sz=8 al=4 fields=3 t1=1996959894 crc=3421780262"
    );

    let rodata = find_section_in_elf(&parity.res.elf.bytes, ".rodata")
        .expect("expected .rodata section in native ELF");
    let crc1_le = 1996959894_u32.to_le_bytes();
    assert!(
        rodata.len() >= 1024 && rodata.windows(4).any(|w| w == crc1_le),
        "expected precomputed CRC32_TABLE constant bytes (1996959894 = 0x77073096) in native ELF .rodata (len={})",
        rodata.len()
    );

    let crc_fn = find_eir_fn(&parity.res.eir, &parity.interner, "crc32");
    let (crc_idx, crc_arith) =
        assert_all_indexed_and_arith_proven(crc_fn, &parity.res.check.env, "crc32");
    assert!(
        crc_idx >= 2,
        "crc32 should have both data[i] and CRC32_TABLE[idx] proven-safe, got {crc_idx}"
    );
    assert!(
        crc_arith >= 1,
        "crc32 should have proven-safe integer arithmetic (i + 1_usize), got {crc_arith}"
    );
    assert_eq!(
        count_checked_binary(crc_fn, BinaryOp::AddCheck)
            + count_checked_binary(crc_fn, BinaryOp::SubCheck)
            + count_checked_binary(crc_fn, BinaryOp::MulCheck),
        0,
        "crc32 should have zero checked arithmetic ops"
    );
}

#[test]
fn exit_gate_3b_layout_proto_zero_check_and_checked_preserved() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let proto_dir = root.join("examples").join("layout_proto");
    let loaded = load_eon_sources(&[proto_dir]).expect("load layout_proto");
    let parity = compile_and_run_parity_bytes("layout_proto_p3b", &loaded, strip_elapsed_ns);
    assert_eq!(parity.code, 0, "stderr: {}", parity.stderr);
    assert!(parity.stdout.contains("box tag=1 pos=(0,0) size=24x8"));
    assert!(parity.stdout.contains("final_live_allocs=0"));

    let fill_fn = find_eir_fn(&parity.res.eir, &parity.interner, "fill_rect");
    assert_eq!(
        count_builtin_calls(fill_fn, |op| matches!(op, CfgBuiltinOp::SliceAt { is_mut: true, .. })),
        1,
        "fill_rect must keep explicit checked `grid.at_mut(idx)`"
    );
    assert_eq!(
        count_checked_binary(fill_fn, BinaryOp::MulCheck),
        1,
        "fill_rect must keep explicit checked `py *? stride`"
    );
    // `px + 1_u32` and `py + 1_u32` are proven zero-check -> only 3 checked additions (`gx0 +? gw`, `gy0 +? gh`, `off_y +? px`)!
    assert_eq!(
        count_checked_binary(fill_fn, BinaryOp::AddCheck),
        3,
        "proven `px + 1_u32` and `py + 1_u32` should not emit BinaryOp::AddCheck"
    );
    let (_fill_idx, fill_arith) =
        assert_all_indexed_and_arith_proven(fill_fn, &parity.res.check.env, "fill_rect");
    assert!(
        fill_arith >= 2,
        "fill_rect should prove `px + 1_u32` and `py + 1_u32` safe, got {fill_arith}"
    );

    let main_fn = find_eir_fn(&parity.res.eir, &parity.interner, "main");
    let (main_idx, _) =
        assert_all_indexed_and_arith_proven(main_fn, &parity.res.check.env, "main");
    assert!(
        main_idx >= 1,
        "main should have proven zero-check `grid[idx]`, got {main_idx}"
    );
    assert_eq!(
        count_builtin_calls(main_fn, |op| matches!(op, CfgBuiltinOp::ArenaPoolIdAt(_))),
        1,
        "main must keep explicit checked `boxes.id_at(bi)`"
    );
    assert_eq!(
        count_builtin_calls(main_fn, |op| matches!(
            op,
            CfgBuiltinOp::SliceAt { .. } | CfgBuiltinOp::ArrayAt { .. }
        )),
        0,
        "main should not use checked SliceAt/ArrayAt for proven `grid[idx]`"
    );
}

#[test]
fn exit_gate_3b_negative_verifier_with_known_facts() {
    // 1. Unproven index bounds -> E0107 with `known:` facts
    let (codes, out) = check_inline_diags(&[(
        "oob.eon",
        r#"
module test.oob;
pub fn bad_index(buf: *[u8], i: usize) -> u8 {
    return buf[i];
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0107), "expected E0107, got {codes:?}\n{out}");
    assert!(out.contains("known:"), "expected known facts in E0107:\n{out}");

    // 2. Unproven arithmetic overflow, div-by-zero, shift -> E0108 with `known:` facts
    let (codes, out) = check_inline_diags(&[(
        "arith.eon",
        r#"
module test.arith;
pub fn bad_arith(a: u32, b: u32, s: u32) -> u32 {
    let x = a + b;
    let y = x / b;
    return y << s;
}
"#,
    )]);
    assert_eq!(
        codes.iter().filter(|&&c| c == ErrorCode::E0108).count(),
        3,
        "expected three E0108 diagnostics, got {codes:?}\n{out}"
    );
    assert!(out.contains("known:"), "expected known facts in E0108:\n{out}");

    // 3. Dead user `if` branch -> E0114
    let (codes, out) = check_inline_diags(&[(
        "dead_if.eon",
        r#"
module test.dead_if;
pub fn bad_if(x: u32) -> u32 {
    if x < 0_u32 {
        return 1_u32;
    }
    return x;
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0114), "expected E0114, got {codes:?}\n{out}");

    // 4. Reachable `unreachable;` -> E0115
    let (codes, out) = check_inline_diags(&[(
        "reach.eon",
        r#"
module test.reach;
pub fn bad_unreach(x: u32) -> u32 {
    if x > 10_u32 {
        unreachable;
    }
    return x;
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0115), "expected E0115, got {codes:?}\n{out}");

    // 5. Comptime fuel exhaustion -> E0520
    let (codes, out) = check_inline_diags(&[(
        "fuel.eon",
        r#"
module test.fuel;
const SPIN: u32 = comptime {
    var x: u32 = 0_u32;
    @comptime_budget(100);
    while true {
        x = x +% 1_u32;
    }
    x
};
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0520), "expected E0520, got {codes:?}\n{out}");
}

#[test]
fn exit_gate_3b_negative_effects_stack_sys_and_par_for() {
    // 1. Effects: @pure mutation (E0112), @noalloc allocation (E0112), @interrupt missing @stack (E0112), recursion without @recursive (E0113)
    let (codes, out) = check_inline_diags(&[(
        "effects.eon",
        r#"
module test.effects;

@pure
pub fn bad_pure(p: *mut u32) {
    p.* = 1_u32;
}

@noalloc
pub fn bad_noalloc(alloc: *mut AllocCap) -> !MemError {
    let a = try Arena.init(alloc, 64_usize);
    a.destroy(alloc);
    return;
}

@interrupt @noalloc
pub fn bad_isr() {
    return;
}

pub fn bad_rec(n: u32) -> u32 {
    if n == 0_u32 {
        return 0_u32;
    }
    return bad_rec(n - 1_u32);
}
"#,
    )]);
    assert!(
        codes.iter().filter(|&&c| c == ErrorCode::E0112).count() >= 3,
        "expected >= 3 E0112 effect errors, got {codes:?}\n{out}"
    );
    assert!(
        codes.contains(&ErrorCode::E0113),
        "expected E0113 recursion error, got {codes:?}\n{out}"
    );

    // 2. @stack(N) exceeded directly and via call chain -> E0701, and impl Iface -> E0510
    let (codes, out) = check_inline_diags(&[(
        "stack.eon",
        r#"
module test.stack;

pub iface Handler {
    @stack(64)
    fn handle(self: *Self) -> u32;
}

pub struct BigHandler {
    pub x: u32;
}

impl Handler for BigHandler {
    @stack(64)
    fn handle(self: *Self) -> u32 {
        let buf: [128]u8 = [0_u8; 128];
        return self.x +% buf[0].widen.[u32]();
    }
}

fn leaf() -> u32 {
    let a: [96]u8 = [0_u8; 96];
    return a[0].widen.[u32]();
}

@stack(64)
pub fn too_deep() -> u32 {
    let b: [32]u8 = [0_u8; 32];
    return leaf() +% b[0].widen.[u32]();
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0701), "expected E0701 stack overflow, got {codes:?}\n{out}");
    assert!(codes.contains(&ErrorCode::E0510), "expected E0510 iface stack overflow, got {codes:?}\n{out}");

    // 3. sys column conflict -> E0601
    let (codes, out) = check_inline_diags(&[(
        "sys_conflict.eon",
        r#"
module test.sys_conflict;

pub pool Widget {
    pub pos: u32;
    pub color: u32;
}

pub sys write_pos(w: write ArenaPool[Widget].pos) {
    let _ = w;
}

pub sys read_pos(r: read ArenaPool[Widget].pos) {
    let _ = r;
}

pub fn build_sched() {
    let _s = Schedule.build(.{ write_pos, read_pos });
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0601), "expected E0601 sys conflict, got {codes:?}\n{out}");

    // 4. par for mutating outer var -> E0602, and non-deterministic par_reduce(.Add) -> E0603
    let (codes, out) = check_inline_diags(&[(
        "par_for.eon",
        r#"
module test.par_for;

pub fn bad_par(xs: *mut [u32], fs: *[f32]) -> f32 {
    var total: u32 = 0_u32;
    par for x in xs.each_mut() {
        total = total +% x.*;
    }
    let _ = total;
    return fs.par_reduce(.Add, 0.0_f32);
}
"#,
    )]);
    assert!(codes.contains(&ErrorCode::E0602), "expected E0602 par for mutation, got {codes:?}\n{out}");
    assert!(codes.contains(&ErrorCode::E0603), "expected E0603 par_reduce .Add, got {codes:?}\n{out}");
}
