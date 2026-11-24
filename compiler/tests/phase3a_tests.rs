use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use eonc::codegen_x64::X64Options;
use eonc::diag::{DiagBag, ErrorCode};
use eonc::elf64::ElfMode;
use eonc::span::{Interner, SourceMap};
use eonc::{
    compile_files_to_x64_elf, interpret_files, load_eon_sources, run_compiled_binary,
    write_executable_elf,
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
            bin_path: std::env::temp_dir().join(format!("eon_p3a_{tag}_{pid}_{id}.out")),
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

fn run_native_variant(
    tag: &str,
    inputs: &[(String, Vec<u8>)],
    sanitize: bool,
) -> (i32, String, String) {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize,
        needed_libs: Vec::new(),
    };
    let res = compile_files_to_x64_elf(inputs, &mut sm, &mut interner, &mut diags, &opts, None)
        .unwrap_or_else(|e| panic!("unexpected x86_64 compiler errors for {tag} (sanitize={sanitize}):\n{e}"));
    let tmp = TempElf::new(tag);
    write_executable_elf(&res.elf.bytes, &tmp.bin_path).unwrap_or_else(|e| {
        panic!("failed to write ELF for {tag} (sanitize={sanitize}): {e}");
    });
    run_compiled_binary(&tmp.bin_path)
        .unwrap_or_else(|e| panic!("failed to run compiled binary for {tag} (sanitize={sanitize}): {e}"))
}

fn compile_and_run_parity_bytes(
    tag: &str,
    inputs: &[(String, Vec<u8>)],
    normalize: fn(&str) -> String,
) -> (i32, String, String) {
    let mut sm_i = SourceMap::new();
    let mut int_i = Interner::new();
    let mut diags_i = DiagBag::new();
    let interp = interpret_files(inputs, &mut sm_i, &mut int_i, &mut diags_i)
        .unwrap_or_else(|e| panic!("EIR-Interp failed for {tag}:\n{e}"));

    let (n_code, n_out, n_err) = run_native_variant(&format!("{tag}_pie"), inputs, false);
    let (s_code, s_out, s_err) = run_native_variant(&format!("{tag}_san"), inputs, true);

    assert_eq!(
        (interp.exit_code, n_code, s_code),
        (n_code, n_code, n_code),
        "exit code mismatch across Interp ({}) / PIE ({}) / Sanitize ({}) for {tag}\nPIE stderr:\n{n_err}\nSanitize stderr:\n{s_err}",
        interp.exit_code,
        n_code,
        s_code
    );
    assert_eq!(
        normalize(&interp.stdout),
        normalize(&n_out),
        "stdout mismatch between EIR-Interp and Native PIE for {tag}\n--- INTERP ---\n{}\n--- PIE ---\n{n_out}",
        interp.stdout
    );
    assert_eq!(
        normalize(&n_out),
        normalize(&s_out),
        "stdout mismatch between Native PIE and Native --sanitize for {tag}\n--- PIE ---\n{n_out}\n--- SANITIZE ---\n{s_out}"
    );
    assert!(s_err.is_empty(), "unexpected --sanitize stderr for {tag}:\n{s_err}");

    (n_code, n_out, n_err)
}

fn identity_out(s: &str) -> String {
    s.to_owned()
}

fn compile_and_run_inline(tag: &str, files: &[(&str, &str)]) -> (i32, String, String) {
    let inputs: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|(n, s)| ((*n).to_owned(), s.as_bytes().to_vec()))
        .collect();
    compile_and_run_parity_bytes(tag, &inputs, identity_out)
}

#[test]
fn e2e_try_linear_errdefer_both_outcomes() {
    let main_src = r#"
module app.errdefer_test;

import rt.hal;

fn step(fail: bool) -> !MemError {
    if fail {
        return .AllocFailed;
    }
    return;
}

fn run_case(alloc: *mut AllocCap, fail: bool) -> !MemError {
    let a1 = try Arena.init(alloc, 128_usize);
    errdefer {
        hal.print_str("err1;");
        a1.destroy(alloc);
    }
    defer hal.print_str("def1;");

    let a2 = try Arena.init(alloc, 256_usize);
    errdefer {
        hal.print_str("err2;");
        a2.destroy(alloc);
    }
    defer hal.print_str("def2;");

    try step(fail);

    a2.destroy(alloc);
    a1.destroy(alloc);
    hal.print_str("ok_body;");
    return;
}

pub fn main(alloc: *mut AllocCap) -> i32 ! MemError {
    hal.print_str("OK:");
    try run_case(alloc, false);
    hal.print_str("live=");
    hal.print_u64(alloc.live_allocs().widen.[u64]());
    hal.print_nl();

    hal.print_str("ERR:");
    let res = run_case(alloc, true);
    res catch {
        hal.print_str("caught;");
    };
    hal.print_str("live=");
    hal.print_u64(alloc.live_allocs().widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let (code, stdout, stderr) = compile_and_run_inline(
        "try_errdefer",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines,
        vec![
            "OK:ok_body;def2;def1;live=0",
            "ERR:def2;err2;def1;err1;caught;live=0",
        ],
        "unexpected LIFO defer/errdefer trace:\n{stdout}"
    );
}

#[test]
fn e2e_slot_pool_genid_aba_and_tree_links() {
    let main_src = r#"
module app.slot_aba;

import rt.hal;

pub pool Node {
    pub val: u32;
    pub parent: ?Link[Self];
    pub child: ?Link[Self];
}

pub fn main(alloc: *mut AllocCap) -> i32 ! MemError | OutOfMemory | StaleId {
    var p = try SlotPool.[Node].init(alloc, 4_usize);
    errdefer p.destroy(alloc);

    let a = try p.insert(alloc, .{ val: 11_u32, parent: null, child: null });
    let removed = try p.remove(a);
    let b = try p.insert(alloc, .{ val: 22_u32, parent: null, child: null });
    let c = try p.insert(alloc, .{ val: 33_u32, parent: b, child: null });
    p.child[b] = c;

    let aba_ok: u64 = if let _r = p.get(a) { 0_u64 } else { 1_u64 };
    let child_val: u64 = if let ch = p.child[b] {
        p.val[ch].widen.[u64]()
    } else {
        0_u64
    };
    let parent_val: u64 = if let parent_id = p.parent[c] {
        p.val[parent_id].widen.[u64]()
    } else {
        0_u64
    };

    p.destroy(alloc);

    hal.print_str("removed=");
    hal.print_u64(removed.val.widen.[u64]());
    hal.print_str(" aba_ok=");
    hal.print_u64(aba_ok);
    hal.print_str(" child=");
    hal.print_u64(child_val);
    hal.print_str(" parent=");
    hal.print_u64(parent_val);
    hal.print_str(" live=");
    hal.print_u64(alloc.live_allocs().widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let (code, stdout, stderr) = compile_and_run_inline(
        "slot_aba",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_eq!(
        stdout.trim(),
        "removed=11 aba_ok=1 child=33 parent=22 live=0"
    );
}

#[test]
fn e2e_arena_pool_reset_and_checked_grid() {
    let main_src = r#"
module app.arena_checked;

import rt.hal;

pub pool Item {
    pub weight: u32;
    pub next: ?Link[Self];
}

fn checked_cell(y: u32, stride: u32, x: u32) -> u32 ! ArithError {
    let row = try (y *? stride);
    return try (row +? x);
}

fn read_at_or(buf: *[u8], idx: usize, fallback: u8) -> u8 {
    let ptr_res = buf.at(idx);
    let p = ptr_res catch {
        return fallback;
    };
    return p.*;
}

pub fn main(alloc: *mut AllocCap) -> i32 ! MemError | OutOfMemory | BoundsError | ArithError {
    var arena = try Arena.init(alloc, 2048_usize);
    errdefer arena.destroy(alloc);
    var items = try ArenaPool.[Item].init(&mut arena, 8_usize);

    var total: u64 = 0_u64;
    for _frame in 0_usize..5_usize {
        items = items.reset();
        let a = try items.push(.{ weight: 10_u32, next: null });
        let b = try items.push(.{ weight: 20_u32, next: a });
        let prev_w = if let prev = items.next[b] { items.weight[prev] } else { 0_u32 };
        let sum_u32 = try (items.weight[b] +? prev_w);
        total = try (total +? sum_u32.widen.[u64]());
    }

    let ovf = checked_cell(100000_u32, 100000_u32, 1_u32) catch 999_u32;
    let buf: [4]u8 = [10_u8, 20_u8, 30_u8, 40_u8];
    let ok_v = read_at_or(&buf, 2_usize, 77_u8);
    let oob_v = read_at_or(&buf, 9_usize, 77_u8);

    let used_before = arena.used_bytes();
    arena.reset();
    let used_after = arena.used_bytes();
    arena.destroy(alloc);

    hal.print_str("total=");
    hal.print_u64(total);
    hal.print_str(" ovf=");
    hal.print_u64(ovf.widen.[u64]());
    hal.print_str(" ok_v=");
    hal.print_u64(ok_v.widen.[u64]());
    hal.print_str(" oob_v=");
    hal.print_u64(oob_v.widen.[u64]());
    hal.print_str(" reset=");
    hal.print_u64(used_after.widen.[u64]());
    hal.print_str(" used_pos=");
    hal.print_u64(if used_before > 0_usize { 1_u64 } else { 0_u64 });
    hal.print_str(" live=");
    hal.print_u64(alloc.live_allocs().widen.[u64]());
    hal.print_nl();
    return 0_i32;
}
"#;

    let (code, stdout, stderr) = compile_and_run_inline(
        "arena_checked",
        &[("rt.hal.eon", RT_HAL_SRC), ("main.eon", main_src)],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_eq!(
        stdout.trim(),
        "total=150 ovf=999 ok_v=30 oob_v=77 reset=0 used_pos=1 live=0"
    );
}

#[test]
fn e2e_unsupported_unchecked_op_diagnosed() {
    let bad_arith = r#"
module app.bad_arith;

pub fn add_unchecked(a: u32, b: u32) -> u32 {
    return a + b;
}
"#;
    let mut sm1 = SourceMap::new();
    let mut int1 = Interner::new();
    let mut d1 = DiagBag::new();
    let _ = compile_files_to_x64_elf(
        &[("bad_arith.eon".into(), bad_arith.as_bytes().to_vec())],
        &mut sm1,
        &mut int1,
        &mut d1,
        &X64Options::default(),
        None,
    );
    assert!(
        d1.diagnostics().iter().any(|d| d.code == ErrorCode::E0108),
        "expected E0108 on unchecked non-const addition, got:\n{}",
        d1.render_all(&sm1)
    );

    let bad_idx = r#"
module app.bad_idx;

pub fn get_unchecked(arr: *[u8], i: usize) -> u8 {
    return arr[i];
}
"#;
    let mut sm2 = SourceMap::new();
    let mut int2 = Interner::new();
    let mut d2 = DiagBag::new();
    let _ = compile_files_to_x64_elf(
        &[("bad_idx.eon".into(), bad_idx.as_bytes().to_vec())],
        &mut sm2,
        &mut int2,
        &mut d2,
        &X64Options::default(),
        None,
    );
    assert!(
        d2.diagnostics().iter().any(|d| d.code == ErrorCode::E0107),
        "expected E0107 on unchecked dynamic indexing, got:\n{}",
        d2.render_all(&sm2)
    );

    let good_checked = r#"
module app.good_checked;

pub fn add_and_get(arr: *[u8], a: usize, b: usize) -> u8 ! ArithError | BoundsError {
    let idx = try (a +? b);
    let p = try arr.at(idx);
    return p.*;
}
"#;
    let mut sm3 = SourceMap::new();
    let mut int3 = Interner::new();
    let mut d3 = DiagBag::new();
    let shared_opts = X64Options {
        mode: ElfMode::LinuxShared { soname: None },
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let res_good = compile_files_to_x64_elf(
        &[("good.eon".into(), good_checked.as_bytes().to_vec())],
        &mut sm3,
        &mut int3,
        &mut d3,
        &shared_opts,
        None,
    );
    assert!(
        res_good.is_ok() && !d3.has_errors(),
        "expected valid checked code to pass verify_eir and emit native ELF, got:\n{}\n{:?}",
        d3.render_all(&sm3),
        res_good.err()
    );
}

fn strip_elapsed_ns(s: &str) -> String {
    s.lines()
        .map(|l| l.split(" elapsed_ns=").next().unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn e2e_layout_proto_matches_golden_and_metrics() {
    let proto_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("examples")
        .join("layout_proto");
    let sources = load_eon_sources(&[proto_dir]).expect("load layout_proto sources");
    let (code, stdout, stderr) =
        compile_and_run_parity_bytes("layout_proto", &sources, strip_elapsed_ns);
    assert_eq!(code, 0, "stderr: {stderr}");

    let lines: Vec<&str> = stdout.lines().collect();
    let expected_prefix = vec![
        "box tag=1 pos=(0,0) size=24x8",
        "box tag=2 pos=(1,1) size=22x2",
        "box tag=3 pos=(1,3) size=22x4",
        "box tag=4 pos=(2,4) size=9x2",
        "box tag=5 pos=(12,4) size=10x2",
        "........................",
        ".######################.",
        ".######################.",
        ".======================.",
        ".=*********=++++++++++=.",
        ".=*********=++++++++++=.",
        ".======================.",
        "........................",
    ];
    assert_eq!(lines.len(), expected_prefix.len() + 1, "stdout:\n{stdout}");
    assert_eq!(&lines[..expected_prefix.len()], expected_prefix.as_slice());

    let metrics = lines[expected_prefix.len()];
    assert!(metrics.starts_with("metrics: "), "metrics line: {metrics}");
    for needle in [
        "frames=100",
        "boxes=5",
        "aba_rejected=1",
        "arena_cap=4096",
        "arena_after_reset=0",
        "err_cleanup_leaks=0",
        "final_live_allocs=0",
    ] {
        assert!(
            metrics.contains(needle),
            "expected `{needle}` in metrics line `{metrics}`"
        );
    }
}
