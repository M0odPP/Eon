use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use eonc::build_sys::{self, BuildTargetKind};
use eonc::diag::{DiagBag, ErrorCode};
use eonc::elf64::{
    ElfLinkInput, ElfMode, ElfReloc, ElfSymbolDef, RelocKind, RelocSection, link_elf64,
};
use eonc::span::{Interner, SourceMap};
use eonc::tcb_audit;
use eonc::{
    X64Options, compile_files_to_x64_elf, interpret_files_with_args, load_eon_sources,
    run_compiled_binary, run_compiled_binary_with_args, write_executable_elf, write_shared_elf,
};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "eon_phase6_{tag}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create unique phase6 temp dir");
        Self { path: dir }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("compiler crate has workspace parent")
        .to_path_buf()
}

fn eonc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_eonc")
}

fn load_with_stdlib(
    dir: &Path,
    extra_files: &[(&str, &str)],
    stdlib_rel_paths: &[&str],
) -> Vec<(String, Vec<u8>)> {
    let root = workspace_root();
    let mut out = Vec::new();
    for rel in stdlib_rel_paths {
        let p = root.join(rel);
        let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read stdlib `{}`: {e}", p.display()));
        out.push((p.to_string_lossy().into_owned(), bytes));
    }
    for (name, src) in extra_files {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("create test file parent");
        }
        fs::write(&p, src).expect("write test source file");
        out.push((p.to_string_lossy().into_owned(), src.as_bytes().to_vec()));
    }
    out
}

fn assert_three_way_parity_with_args(
    tag: &str,
    files: &[(String, Vec<u8>)],
    interp_args: &[String],
    native_args: &[String],
) -> String {
    let tmp = TempDirGuard::new(tag);

    // 1. EIR Interpreter
    let mut sm_i = SourceMap::new();
    let mut int_i = Interner::new();
    let mut diags_i = DiagBag::new();
    let interp = interpret_files_with_args(files, &mut sm_i, &mut int_i, &mut diags_i, interp_args)
        .unwrap_or_else(|e| panic!("EIR-Interp failed for `{tag}`:\n{e}"));

    // 2. Native x86_64 PIE
    let mut sm_n = SourceMap::new();
    let mut int_n = Interner::new();
    let mut diags_n = DiagBag::new();
    let opts_pie = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let emitted_n = compile_files_to_x64_elf(
        files,
        &mut sm_n,
        &mut int_n,
        &mut diags_n,
        &opts_pie,
        None,
    )
    .unwrap_or_else(|e| panic!("Native PIE compile failed for `{tag}`:\n{e}"));
    let bin_n = tmp.path().join(format!("{tag}_pie"));
    write_executable_elf(&emitted_n.elf.bytes, &bin_n).expect("write PIE ELF");
    let (code_n, out_n, err_n) =
        run_compiled_binary_with_args(&bin_n, native_args).expect("run Native PIE");
    assert!(
        err_n.is_empty(),
        "Native PIE stderr not empty for `{tag}`:\n{err_n}"
    );

    // 3. Native x86_64 PIE + --sanitize
    let mut sm_s = SourceMap::new();
    let mut int_s = Interner::new();
    let mut diags_s = DiagBag::new();
    let opts_san = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize: true,
        needed_libs: Vec::new(),
    };
    let emitted_s = compile_files_to_x64_elf(
        files,
        &mut sm_s,
        &mut int_s,
        &mut diags_s,
        &opts_san,
        None,
    )
    .unwrap_or_else(|e| panic!("Native Sanitize compile failed for `{tag}`:\n{e}"));
    let bin_s = tmp.path().join(format!("{tag}_san"));
    write_executable_elf(&emitted_s.elf.bytes, &bin_s).expect("write Sanitize ELF");
    let (code_s, out_s, err_s) =
        run_compiled_binary_with_args(&bin_s, native_args).expect("run Native Sanitize");
    assert!(
        err_s.is_empty(),
        "Native Sanitize stderr not empty for `{tag}`:\n{err_s}"
    );

    assert_eq!(
        (interp.exit_code, code_n, code_s),
        (0, 0, 0),
        "non-zero exit code in 3-way parity for `{tag}`"
    );
    assert_eq!(
        interp.stdout, out_n,
        "Interp vs Native PIE stdout mismatch for `{tag}`"
    );
    assert_eq!(
        out_n, out_s,
        "Native PIE vs Native Sanitize stdout mismatch for `{tag}`"
    );
    out_n
}

fn compile_native_pie(
    dir: &Path,
    bin_name: &str,
    files: &[(String, Vec<u8>)],
    sanitize: bool,
    needed_libs: Vec<String>,
) -> PathBuf {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize,
        needed_libs,
    };
    let emitted = compile_files_to_x64_elf(files, &mut sm, &mut interner, &mut diags, &opts, None)
        .unwrap_or_else(|e| panic!("compile `{bin_name}` (sanitize={sanitize}) failed:\n{e}"));
    let out_bin = dir.join(bin_name);
    write_executable_elf(&emitted.elf.bytes, &out_bin).expect("write compiled ELF");
    out_bin
}

fn ulp_distance_f64(got_bits: u64, expected: f64) -> u64 {
    let got = f64::from_bits(got_bits);
    if got.is_nan() && expected.is_nan() {
        return 0;
    }
    if got.is_nan() || expected.is_nan() {
        return u64::MAX;
    }
    let ord = |b: u64| -> i128 {
        if (b >> 63) != 0 {
            -((b & 0x7FFF_FFFF_FFFF_FFFF) as i128)
        } else {
            b as i128
        }
    };
    (ord(got_bits) - ord(expected.to_bits())).unsigned_abs() as u64
}

fn run_readelf(path: &Path, flags: &[&str]) -> String {
    let out = Command::new("readelf")
        .args(flags)
        .arg(path)
        .output()
        .expect("invoke readelf");
    assert!(
        out.status.success(),
        "readelf {:?} failed on `{}`:\n{}",
        flags,
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

// ============================================================================
// Phase 6.1 — Pure-Eon Freestanding core.math & core.col
// ============================================================================

#[test]
fn phase6_1_core_math_intrinsics_and_transcendentals_1ulp_three_way_parity() {
    let tmp = TempDirGuard::new("math_1ulp");

    let trig_inputs: &[f64] = &[
        0.0,
        -0.0,
        1.0e-12,
        -1.0e-12,
        5.0e-324,                // smallest subnormal
        2.225_073_858_507_201_4e-308, // smallest normal
        0.123_456_789,
        -0.75,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        -2.345_678_9,
        100.0,
        -1000.5,
        1.0e6,
        1.0e9,
        -1.0e12,
        1.0e15, // Payne-Hanek large range reduction
    ];

    let exp_inputs: &[f64] = &[
        0.0,
        -0.0,
        1.0e-12,
        -1.0e-12,
        0.5,
        1.0,
        -1.0,
        2.5,
        -7.25,
        20.0,
        -20.0,
        700.0,
        -700.0,
        -740.0,
        5.0e-324,
    ];

    let log_inputs: &[f64] = &[
        5.0e-324,
        1.0e-315, // subnormal
        2.225_073_858_507_201_4e-308,
        1.0e-100,
        1.0e-20,
        0.5,
        0.75,
        1.0,
        1.000_001,
        2.0,
        std::f64::consts::E,
        100.0,
        1.0e100,
    ];

    let pow_cases: &[(f64, f64)] = &[
        (2.0, 0.5),
        (2.0, 10.0),
        (2.5, 3.5),
        (0.5, -2.5),
        (-2.0, 3.0),
        (-2.0, 4.0),
        (10.0, -5.0),
        (1.0, 100.0),
        (0.0, 0.0),
        (1.0e-10, 0.25),
        (std::f64::consts::PI, -2.75),
    ];

    let atan2_cases: &[(f64, f64)] = &[
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (-0.0, -1.0),
        (3.25, 0.0),
        (-3.25, 0.0),
        (1.0e-315, 1.0e-315),
        (1.0e300, -1.0e300),
        (0.1234, 5.6789),
        (-9.876, -0.00123),
    ];

    let mut eon_body = String::from(
        "module app.math_test;\n\
         import core.math;\n\
         import std.rt;\n\
         pub fn main() -> i32 {\n",
    );

    // 1. Hardware intrinsics & bit manipulation
    let sqrt64_in = 12.25_f64;
    let sqrt32_in = 19.36_f32;
    let (fma64_a, fma64_b, fma64_c) = (1.5_f64, 2.25_f64, -0.375_f64);
    let (fma32_a, fma32_b, fma32_c) = (1.5_f32, 2.25_f32, -0.375_f32);
    let bit_x: u64 = 0x0012_3456_789A_BC00;
    let bswap32_x: u32 = 0x1234_5678;
    let bswap16_x: u16 = 0xABCD;

    eon_body.push_str(&format!(
        "    rt.print_u64(math.f64_to_bits(math.sqrt_f64(math.bits_to_f64({0}_u64)))); rt.print_nl();\n\
         rt.print_u64(math.f32_to_bits(math.sqrt_f32(math.bits_to_f32({1}_u32))).widen.[u64]()); rt.print_nl();\n\
         rt.print_u64(math.f64_to_bits(math.fma_f64(math.bits_to_f64({2}_u64), math.bits_to_f64({3}_u64), math.bits_to_f64({4}_u64)))); rt.print_nl();\n\
         rt.print_u64(math.f32_to_bits(math.fma_f32(math.bits_to_f32({5}_u32), math.bits_to_f32({6}_u32), math.bits_to_f32({7}_u32))).widen.[u64]()); rt.print_nl();\n\
         rt.print_u64(math.lzcnt_u64({8}_u64)); rt.print_nl();\n\
         rt.print_u64(math.tzcnt_u64({8}_u64)); rt.print_nl();\n\
         rt.print_u64(math.popcnt_u64({8}_u64)); rt.print_nl();\n\
         rt.print_u64(math.bswap_u64({8}_u64)); rt.print_nl();\n\
         rt.print_u64(math.bswap_u32({9}_u32).widen.[u64]()); rt.print_nl();\n\
         rt.print_u64(math.bswap_u16({10}_u16).widen.[u64]()); rt.print_nl();\n",
        sqrt64_in.to_bits(),
        sqrt32_in.to_bits(),
        fma64_a.to_bits(),
        fma64_b.to_bits(),
        fma64_c.to_bits(),
        fma32_a.to_bits(),
        fma32_b.to_bits(),
        fma32_c.to_bits(),
        bit_x,
        bswap32_x,
        bswap16_x,
    ));

    // 2. Transcendentals
    for &x in trig_inputs {
        let xb = x.to_bits();
        eon_body.push_str(&format!(
            "    rt.print_u64(math.f64_to_bits(math.sin(math.bits_to_f64({xb}_u64)))); rt.print_nl();\n\
             rt.print_u64(math.f64_to_bits(math.cos(math.bits_to_f64({xb}_u64)))); rt.print_nl();\n\
             rt.print_u64(math.f64_to_bits(math.tan(math.bits_to_f64({xb}_u64)))); rt.print_nl();\n"
        ));
    }
    for &x in exp_inputs {
        let xb = x.to_bits();
        eon_body.push_str(&format!(
            "    rt.print_u64(math.f64_to_bits(math.exp(math.bits_to_f64({xb}_u64)))); rt.print_nl();\n"
        ));
    }
    for &x in log_inputs {
        let xb = x.to_bits();
        eon_body.push_str(&format!(
            "    rt.print_u64(math.f64_to_bits(math.log(math.bits_to_f64({xb}_u64)))); rt.print_nl();\n"
        ));
    }
    for &(x, y) in pow_cases {
        let xb = x.to_bits();
        let yb = y.to_bits();
        eon_body.push_str(&format!(
            "    rt.print_u64(math.f64_to_bits(math.pow(math.bits_to_f64({xb}_u64), math.bits_to_f64({yb}_u64)))); rt.print_nl();\n"
        ));
    }
    for &(y, x) in atan2_cases {
        let yb = y.to_bits();
        let xb = x.to_bits();
        eon_body.push_str(&format!(
            "    rt.print_u64(math.f64_to_bits(math.atan2(math.bits_to_f64({yb}_u64), math.bits_to_f64({xb}_u64)))); rt.print_nl();\n"
        ));
    }
    eon_body.push_str("    return 0_i32;\n}\n");

    let files = load_with_stdlib(
        tmp.path(),
        &[("main.eon", &eon_body)],
        &["core/math.eon", "core/str.eon", "std/rt.hal.eon"],
    );
    let stdout = assert_three_way_parity_with_args("core_math_1ulp", &files, &[], &[]);
    let bits: Vec<u64> = stdout
        .lines()
        .map(|l| l.trim().parse::<u64>().expect("parse u64 bit pattern"))
        .collect();

    let expected_intrinsics = [
        sqrt64_in.sqrt().to_bits(),
        u64::from(sqrt32_in.sqrt().to_bits()),
        fma64_a.mul_add(fma64_b, fma64_c).to_bits(),
        u64::from(fma32_a.mul_add(fma32_b, fma32_c).to_bits()),
        u64::from(bit_x.leading_zeros()),
        u64::from(bit_x.trailing_zeros()),
        u64::from(bit_x.count_ones()),
        bit_x.swap_bytes(),
        u64::from(bswap32_x.swap_bytes()),
        u64::from(bswap16_x.swap_bytes()),
    ];
    let mut idx = 0usize;
    for (k, &exp_b) in expected_intrinsics.iter().enumerate() {
        assert_eq!(bits[idx], exp_b, "intrinsic #{k} bit mismatch");
        idx += 1;
    }

    let mut max_ulp = 0u64;
    let mut checked = 0usize;
    let mut check_1ulp = |fn_name: &str, input_desc: String, got_b: u64, std_val: f64| {
        let ulp = ulp_distance_f64(got_b, std_val);
        max_ulp = max_ulp.max(ulp);
        checked += 1;
        assert!(
            ulp <= 1,
            "{fn_name}({input_desc}) exceeded 1 ULP: ulp={ulp}, got=0x{got_b:016x} ({}), std=0x{:016x} ({std_val})",
            f64::from_bits(got_b),
            std_val.to_bits()
        );
        if got_b != 0 && std_val != 0.0 {
            let i_diff = (got_b as i64).saturating_sub(std_val.to_bits() as i64).abs();
            assert!(
                i_diff <= 1,
                "{fn_name}({input_desc}) signed i64 bit diff = {i_diff} > 1"
            );
        }
    };

    for &x in trig_inputs {
        check_1ulp("sin", format!("{x:e}"), bits[idx], x.sin());
        idx += 1;
        check_1ulp("cos", format!("{x:e}"), bits[idx], x.cos());
        idx += 1;
        check_1ulp("tan", format!("{x:e}"), bits[idx], x.tan());
        idx += 1;
    }
    for &x in exp_inputs {
        check_1ulp("exp", format!("{x:e}"), bits[idx], x.exp());
        idx += 1;
    }
    for &x in log_inputs {
        check_1ulp("log", format!("{x:e}"), bits[idx], x.ln());
        idx += 1;
    }
    for &(x, y) in pow_cases {
        check_1ulp("pow", format!("{x:e}, {y:e}"), bits[idx], x.powf(y));
        idx += 1;
    }
    for &(y, x) in atan2_cases {
        check_1ulp("atan2", format!("{y:e}, {x:e}"), bits[idx], y.atan2(x));
        idx += 1;
    }
    assert_eq!(idx, bits.len());
    println!(
        "[phase6.1] core.math verified {checked} transcendental cases (max observed ULP = {max_ulp}) + 10 intrinsics with 3-way parity"
    );
}

#[test]
fn phase6_1_core_col_bitset_and_interner_three_way_parity() {
    let tmp = TempDirGuard::new("col_bitset_interner");
    let src = r#"
module app.col_bi_test;

import core.col;
import core.str;
import std.rt;

fn check_bitset(alloc: *mut AllocCap) -> u64 ! OutOfMemory {
    var a = try col.BitSet.init(alloc, 200_usize);
    errdefer a.destroy(alloc);
    var b = try col.BitSet.init(alloc, 200_usize);
    errdefer b.destroy(alloc);

    a.set(5_usize);
    a.set(67_usize);
    a.set(199_usize);
    a.unset(5_usize);
    let f1 = a.first_set() orelse 9999_usize;
    let c1 = a.count_ones();

    b.set(67_usize);
    b.set(130_usize);
    a.union_with(&b);
    let c_union = a.count_ones(); // {67, 130, 199} -> 3
    a.intersect_with(&b);
    let c_inter = a.count_ones(); // {67, 130} -> 2
    try a.set_grow(alloc, 300_usize);
    let has_300: u64 = if a.test(300_usize) { 1_u64 } else { 0_u64 };

    a.destroy(alloc);
    b.destroy(alloc);
    return (f1.widen.[u64]()) *% 10000_u64
        +% (c1.widen.[u64]()) *% 1000_u64
        +% (c_union.widen.[u64]()) *% 100_u64
        +% (c_inter.widen.[u64]()) *% 10_u64
        +% has_300;
}

fn check_interner(alloc: *mut AllocCap) -> u64 ! OutOfMemory {
    var intern = try col.Interner.init(alloc, 16_usize);
    errdefer intern.destroy(alloc);
    var sb = try str.StrBuf.init(alloc, 32_usize);
    errdefer sb.destroy(alloc);

    var id_acc: u64 = 0_u64;
    // Intern 80 distinct strings across multiple rehashes (16 -> 32 -> 64 -> 128).
    for i in 0_u64..80_u64 {
        sb.clear();
        try sb.push_str("sym_");
        try sb.push_u64(i);
        let id = try intern.intern(alloc, sb.as_bytes());
        let dup_id = try intern.intern(alloc, sb.as_bytes());
        if id != dup_id || id.widen.[u64]() != i {
            sb.destroy(alloc);
            intern.destroy(alloc);
            return 0_u64;
        }
        id_acc = id_acc +% id.widen.[u64]();
    }

    // Lookup existing and missing entries and verify stored bytes.
    sb.clear();
    try sb.push_str("sym_42");
    let got42 = intern.lookup(sb.as_bytes()) orelse 9999_u32;
    let miss = intern.lookup_str("missing_key") orelse 8888_u32;
    var byte_ok: u64 = 0_u64;
    if let slice42 = intern.get(got42) {
        if str.bytes_eq(slice42, sb.as_bytes()) {
            byte_ok = 1_u64;
        }
    }
    let count = intern.len().widen.[u64]();

    sb.destroy(alloc);
    intern.destroy(alloc);
    return id_acc *% 1000000_u64
        +% got42.widen.[u64]() *% 10000_u64
        +% miss.widen.[u64]()
        +% byte_ok *% 100_u64
        +% count;
}

pub fn main(env: *mut Env) -> i32 ! OutOfMemory {
    let b = try check_bitset(&mut env.alloc);
    let s = try check_interner(&mut env.alloc);
    let live = env.alloc.live_allocs().widen.[u64]();
    let used = env.alloc.used_bytes().widen.[u64]();
    rt.print_u64(b); rt.print_nl();
    rt.print_u64(s); rt.print_nl();
    rt.print_u64(live); rt.print_nl();
    rt.print_u64(used); rt.print_nl();
    return 0_i32;
}
"#;

    let files = load_with_stdlib(
        tmp.path(),
        &[("main.eon", src)],
        &["core/col.eon", "core/str.eon", "std/rt.hal.eon"],
    );
    let stdout = assert_three_way_parity_with_args("core_col_bi", &files, &[], &[]);
    let vals: Vec<u64> = stdout
        .lines()
        .map(|l| l.trim().parse::<u64>().expect("parse u64 line"))
        .collect();
    assert_eq!(vals.len(), 4);
    assert_eq!(vals[0], 672_321, "BitSet check mismatch: {vals:?}");
    // id_acc = 0..79 = 3160; got42=42, miss=8888, byte_ok=1, count=80 => 8888 + 100 + 80 = 9068
    assert_eq!(vals[1], 3_160_429_068, "Interner check mismatch: {vals:?}");
    assert_eq!(vals[2], 0, "live_allocs must be 0 (zero leaks)");
    assert_eq!(vals[3], 0, "used_bytes must be 0 after destroy");
}

#[test]
fn phase6_1_core_col_swisstable_hashmap_three_way_parity() {
    let tmp = TempDirGuard::new("col_hashmap");
    let src = r#"
module app.col_hm_test;

import core.col;
import std.rt;

fn check_hashmap(alloc: *mut AllocCap) -> u64 ! OutOfMemory {
    var map = try col.HashMap.[u64].init(alloc, 16_usize);
    errdefer map.destroy(alloc);

    // Insert 600 keys to trigger multiple rehashes (16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024).
    for i in 0_u64..600_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        try map.insert(alloc, k, i *% 3_u64 +% 7_u64);
    }

    // Update 100 keys in-place.
    for i in 0_u64..100_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        try map.insert(alloc, k, i *% 11_u64 +% 5_u64);
    }

    // Remove 200 keys (creating 0xFE tombstones).
    var removed_sum: u64 = 0_u64;
    for i in 100_u64..300_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        if let v = map.remove(k) {
            removed_sum = removed_sum +% v;
        }
    }

    // Re-insert 250 keys to exercise tombstone reuse + extra growth.
    for i in 600_u64..850_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        try map.insert(alloc, k, i +% 1000_u64);
    }
    for i in 100_u64..150_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        try map.insert(alloc, k, i *% 13_u64);
    }

    var present_sum: u64 = 0_u64;
    for i in 0_u64..850_u64 {
        let k = (i *% 2654435761_u64) +% 17_u64;
        if let v = map.get(k) {
            present_sum = present_sum +% v;
        }
    }

    let final_len = map.len().widen.[u64]();
    let final_cap = map.cap().widen.[u64]();
    map.destroy(alloc);
    return present_sum ^ (removed_sum *% 31_u64) ^ (final_len *% 131_u64) ^ final_cap;
}

pub fn main(env: *mut Env) -> i32 ! OutOfMemory {
    let h = try check_hashmap(&mut env.alloc);
    let live = env.alloc.live_allocs().widen.[u64]();
    let used = env.alloc.used_bytes().widen.[u64]();
    rt.print_u64(h); rt.print_nl();
    rt.print_u64(live); rt.print_nl();
    rt.print_u64(used); rt.print_nl();
    return 0_i32;
}
"#;

    let files = load_with_stdlib(
        tmp.path(),
        &[("main.eon", src)],
        &["core/col.eon", "core/str.eon", "std/rt.hal.eon"],
    );
    let stdout = assert_three_way_parity_with_args("core_col_hm", &files, &[], &[]);
    let vals: Vec<u64> = stdout
        .lines()
        .map(|l| l.trim().parse::<u64>().expect("parse u64 line"))
        .collect();
    assert_eq!(vals.len(), 3);
    assert_ne!(vals[0], 0);
    assert_eq!(vals[1], 0, "live_allocs must be 0 (zero leaks)");
    assert_eq!(vals[2], 0, "used_bytes must be 0 after destroy");
}

// ============================================================================
// Phase 6.2 — Industrial ELF64 W^X PIE, PT_TLS, --shared .so, & Dynamic -l
// ============================================================================

#[test]
fn phase6_2_elf64_wx_pie_dynamic_and_pt_tls() {
    let tmp = TempDirGuard::new("elf64_wx_tls");

    // 1. Static PIE with dyn vtable (.data.rel.ro + PT_GNU_RELRO + 3 W^X PT_LOAD segments + 0 PT_INTERP)
    let pie_src = r#"
module app.wx_pie;

import std.rt;

pub iface Op {
    fn eval(self: *Self, x: u64) -> u64;
}

pub struct Mul3 { pub k: u64; }
impl Op for Mul3 {
    fn eval(self: *Self, x: u64) -> u64 {
        return x *% self.k;
    }
}

fn call_op(op: *dyn Op, x: u64) -> u64 {
    return op.eval(x);
}

pub fn main(env: *mut Env) -> i32 ! OutOfMemory {
    var buf = try Buf.[u64].init(&mut env.alloc, 4_usize);
    try buf.push(call_op(&Mul3.{ k: 6_u64 }, 7_u64));
    let sl = buf.as_slice();
    let v = if sl.len > 0_usize { sl[0_usize] } else { 0_u64 };
    buf.destroy(&mut env.alloc);
    rt.print_u64(v);
    rt.print_nl();
    return 0_i32;
}
"#;
    let pie_files = load_with_stdlib(
        tmp.path(),
        &[("pie_main.eon", pie_src)],
        &["core/str.eon", "std/rt.hal.eon"],
    );
    let pie_bin = compile_native_pie(tmp.path(), "static_pie", &pie_files, false, Vec::new());

    let (pie_code, pie_out, pie_err) = run_compiled_binary(&pie_bin).expect("run static_pie");
    assert_eq!(pie_code, 0, "stderr: {pie_err}");
    assert_eq!(pie_out.trim(), "42");

    let elf_hdr = run_readelf(&pie_bin, &["-h"]);
    assert!(
        elf_hdr.contains("DYN ("),
        "Static PIE must be ET_DYN, got:\n{elf_hdr}"
    );

    let ph_out = run_readelf(&pie_bin, &["-l", "-W"]);
    assert!(
        !ph_out.contains("INTERP"),
        "Static PIE must have zero PT_INTERP, got:\n{ph_out}"
    );
    assert!(
        ph_out.contains("GNU_RELRO"),
        "Static PIE must contain PT_GNU_RELRO, got:\n{ph_out}"
    );

    let mut load_segs: Vec<(u64, u64, String)> = Vec::new();
    for line in ph_out.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("LOAD") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            // Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg Align
            assert!(parts.len() >= 8, "unexpected LOAD line: {trimmed}");
            let vaddr = u64::from_str_radix(parts[2].trim_start_matches("0x"), 16).unwrap();
            let memsz = u64::from_str_radix(parts[5].trim_start_matches("0x"), 16).unwrap();
            let mut flg_parts = Vec::new();
            for p in &parts[6..parts.len() - 1] {
                flg_parts.push(*p);
            }
            let flg = flg_parts.join(" ");
            assert_eq!(vaddr % 0x1000, 0, "LOAD vaddr not page-aligned: 0x{vaddr:x}");
            assert!(
                !(flg.contains('W') && flg.contains('E')),
                "W^X violated on LOAD segment: {trimmed}"
            );
            load_segs.push((vaddr, memsz, flg));
        }
    }
    assert_eq!(load_segs.len(), 3, "expected 3 PT_LOAD segments, got {load_segs:?}");
    assert_eq!(load_segs[0].2, "R", "seg 0 must be R only: {load_segs:?}");
    assert_eq!(load_segs[1].2, "R E", "seg 1 must be R E: {load_segs:?}");
    assert_eq!(load_segs[2].2, "RW", "seg 2 must be RW: {load_segs:?}");
    let unique_vaddrs: BTreeSet<u64> = load_segs.iter().map(|s| s.0).collect();
    assert_eq!(unique_vaddrs.len(), 3, "LOAD vaddrs must be distinct");
    assert!(load_segs[0].0 + load_segs[0].1 <= load_segs[1].0);
    assert!(load_segs[1].0 + load_segs[1].1 <= load_segs[2].0);

    let sec_out = run_readelf(&pie_bin, &["-S", "-W"]);
    assert!(
        sec_out.contains(".data.rel.ro"),
        "expected .data.rel.ro in Static PIE sections:\n{sec_out}"
    );
    let reloc_out = run_readelf(&pie_bin, &["-r", "-W"]);
    assert!(
        reloc_out.contains("R_X86_64_RELATIVE"),
        "expected R_X86_64_RELATIVE in Static PIE relocs:\n{reloc_out}"
    );

    // 2. Dynamic PIE linked against -l libc.so.6 -> PT_INTERP, PT_DYNAMIC, .gnu.hash, DT_BIND_NOW, JUMP_SLOT
    let dyn_src = r#"
module app.dyn_libc @hal;

extern "C" fn abs(x: i32) -> i32;

pub fn main() -> i32 {
    let v = abs(-77_i32);
    if v == 77_i32 {
        return 0_i32;
    }
    return 1_i32;
}
"#;
    let dyn_bin = tmp.path().join("dyn_libc");
    let dyn_file = tmp.path().join("dyn_libc.hal.eon");
    fs::write(&dyn_file, dyn_src).expect("write dyn_libc.hal.eon");
    let build_dyn = Command::new(eonc_bin())
        .args([
            "build",
            "-l",
            "libc.so.6",
            "-o",
            &dyn_bin.to_string_lossy(),
            &dyn_file.to_string_lossy(),
        ])
        .output()
        .expect("run eonc build -l libc.so.6");
    assert!(
        build_dyn.status.success(),
        "eonc build -l libc.so.6 failed:\n{}",
        String::from_utf8_lossy(&build_dyn.stderr)
    );
    let (dyn_code, _, dyn_err) = run_compiled_binary(&dyn_bin).expect("run dyn_libc");
    assert_eq!(dyn_code, 0, "dyn_libc stderr: {dyn_err}");

    let dyn_ph = run_readelf(&dyn_bin, &["-l", "-W"]);
    assert!(dyn_ph.contains("INTERP"), "dynamic exe missing PT_INTERP:\n{dyn_ph}");
    assert!(dyn_ph.contains("DYNAMIC"), "dynamic exe missing PT_DYNAMIC:\n{dyn_ph}");
    let dyn_secs = run_readelf(&dyn_bin, &["-S", "-W"]);
    assert!(dyn_secs.contains(".gnu.hash"), "missing .gnu.hash:\n{dyn_secs}");
    let dyn_tags = run_readelf(&dyn_bin, &["-d", "-W"]);
    assert!(
        dyn_tags.contains("BIND_NOW") && dyn_tags.contains("NOW"),
        "missing DT_BIND_NOW / DF_1_NOW:\n{dyn_tags}"
    );
    assert!(dyn_tags.contains("libc.so.6"), "missing DT_NEEDED libc.so.6:\n{dyn_tags}");
    let dyn_rel = run_readelf(&dyn_bin, &["-r", "-W"]);
    assert!(
        dyn_rel.contains("R_X86_64_JUMP_SLOT") && dyn_rel.contains("abs"),
        "missing PLT reloc for abs:\n{dyn_rel}"
    );

    // 3. PT_TLS + .tdata/.tbss + TpOff32 + SYS_arch_prctl(ARCH_SET_FS) end-to-end execution
    // Machine code layout:
    //   allocate 64B stack for TCB + TLS (Variant II: [rsp+0..16] = tls_tdata(8B) + tls_tbss(8B),
    //   [rsp+16] = self_tcb_ptr = rsp + 16), copy .tdata from GOT/RIP-relative or init via arch_prctl,
    //   wait: copy initial 8 bytes from `tls_seed` in .tdata via direct TpOff32 after initializing from .tdata image!
    //   Specifically:
    //     sub rsp, 64
    //     lea rbx, [rsp + 64]       ; tcb = rsp + 64 (FS.base; TLS block at [tcb - 64 .. tcb])
    //     mov [rbx], rbx            ; TCB self-pointer
    //     ; Copy initial `.tdata` image (0x123456789ABCDEF0) into [rbx + tpoff(tls_init)] via __eon_tdata_start
    //     ; Actually: test loading `__eon_tdata_start` (from link_elf64 synthetic symbol via GOT!), copying 8B,
    //     ; calling SYS_arch_prctl(0x1002, rbx), then reading `fs:[tls_init@tpoff]` and writing/reading `fs:[tls_zero@tpoff]`,
    //     ; verifying exact expected values and exiting 0 via SYS_exit_group(231)!
    let mut text = Vec::new();
    let mut relocs = Vec::new();
    // sub rsp, 128
    text.extend_from_slice(&[0x48, 0x81, 0xEC, 128, 0, 0, 0]);
    // lea rbx, [rsp + 64]
    text.extend_from_slice(&[0x48, 0x8D, 0x5C, 0x24, 64]);
    // mov [rbx], rbx
    text.extend_from_slice(&[0x48, 0x89, 0x1B]);
    // Self-relocate if static PIE (R_X86_64_RELATIVE entries between __eon_rela_dyn_start..end):
    // Actually `.tdata` is loaded by kernel in PT_LOAD seg2, and we can read `__eon_tdata_start` via LEA RIP!
    // Wait: `__eon_tdata_start` is defined at `tdata_vaddr` in link_elf64! So `lea r10, [rip + disp32(__eon_tdata_start)]` works WITHOUT GOT relocations!
    // lea r10, [rip + disp32]
    text.extend_from_slice(&[0x4C, 0x8D, 0x15]);
    let tdata_disp_off = text.len();
    text.extend_from_slice(&[0, 0, 0, 0]);
    relocs.push(ElfReloc {
        section: RelocSection::Text,
        offset: tdata_disp_off,
        kind: RelocKind::Rel32,
        target_sym: "__eon_tdata_start".to_owned(),
        addend: 0,
    });
    // mov rax, [r10]  ; initial 8B of .tdata
    text.extend_from_slice(&[0x49, 0x8B, 0x02]);
    // Store at [rbx - 64] (Variant II: tls_aligned_memsz = align_up(16, 64) = 64 -> tpoff(tls_init) = -64!)
    // mov [rbx - 64], rax
    text.extend_from_slice(&[0x48, 0x89, 0x43, 0xC0]);
    // xor rax, rax; mov [rbx - 56], rax (zero .tbss slot)
    text.extend_from_slice(&[0x48, 0x31, 0xC0, 0x48, 0x89, 0x43, 0xC8]);
    // SYS_arch_prctl(ARCH_SET_FS=0x1002, rbx) -> rax=158, rdi=0x1002, rsi=rbx
    text.extend_from_slice(&[0x48, 0xC7, 0xC7, 0x02, 0x10, 0x00, 0x00]); // mov rdi, 0x1002
    text.extend_from_slice(&[0x48, 0x89, 0xDE]); // mov rsi, rbx
    text.extend_from_slice(&[0x48, 0xC7, 0xC0, 158, 0, 0, 0]); // mov rax, 158
    text.extend_from_slice(&[0x0F, 0x05]); // syscall
    // Read `fs:[tls_init@tpoff]` -> rcx (`64 48 8B 0C 25 <disp32>`)
    text.extend_from_slice(&[0x64, 0x48, 0x8B, 0x0C, 0x25]);
    let tp1_off = text.len();
    text.extend_from_slice(&[0, 0, 0, 0]);
    relocs.push(ElfReloc {
        section: RelocSection::Text,
        offset: tp1_off,
        kind: RelocKind::TpOff32,
        target_sym: "tls_init".to_owned(),
        addend: 0,
    });
    // Write rcx + 1 into `fs:[tls_zero@tpoff]` -> `48 83 C1 01` then `64 48 89 0C 25 <disp32>`
    text.extend_from_slice(&[0x48, 0x83, 0xC1, 0x01]);
    text.extend_from_slice(&[0x64, 0x48, 0x89, 0x0C, 0x25]);
    let tp2_off = text.len();
    text.extend_from_slice(&[0, 0, 0, 0]);
    relocs.push(ElfReloc {
        section: RelocSection::Text,
        offset: tp2_off,
        kind: RelocKind::TpOff32,
        target_sym: "tls_zero".to_owned(),
        addend: 0,
    });
    // Read back `fs:[tls_zero@tpoff]` -> rdx (`64 48 8B 14 25 <disp32>`)
    text.extend_from_slice(&[0x64, 0x48, 0x8B, 0x14, 0x25]);
    let tp3_off = text.len();
    text.extend_from_slice(&[0, 0, 0, 0]);
    relocs.push(ElfReloc {
        section: RelocSection::Text,
        offset: tp3_off,
        kind: RelocKind::TpOff32,
        target_sym: "tls_zero".to_owned(),
        addend: 0,
    });
    // Compare rdx == 0x123456789ABCDEF1 (stored in r8: movabs r8, imm64)
    let expected_val: u64 = 0x1234_5678_9ABC_DEF1;
    text.extend_from_slice(&[0x49, 0xB8]);
    text.extend_from_slice(&expected_val.to_le_bytes());
    // xor rdi, rdi; cmp rdx, r8; sete dil
    text.extend_from_slice(&[0x48, 0x31, 0xFF, 0x4C, 0x39, 0xC2, 0x40, 0x0F, 0x95, 0xC7]); // setne dil (0 if equal, 1 if mismatch!)
    // SYS_exit_group(rdi) -> rax = 231
    text.extend_from_slice(&[0x48, 0xC7, 0xC0, 231, 0, 0, 0, 0x0F, 0x05]);

    let tls_seed: u64 = 0x1234_5678_9ABC_DEF0;
    let tls_elf = link_elf64(&ElfLinkInput {
        mode: ElfMode::LinuxPie,
        entry_symbol: "_start".to_owned(),
        text,
        rodata: Vec::new(),
        data_rel_ro: Vec::new(),
        data: Vec::new(),
        bss_size: 0,
        tdata: tls_seed.to_le_bytes().to_vec(),
        tbss_size: 8,
        symbols: vec![
            ElfSymbolDef {
                name: "_start".to_owned(),
                section: RelocSection::Text,
                offset: 0,
                size: 0,
                is_func: true,
            },
            ElfSymbolDef {
                name: "tls_init".to_owned(),
                section: RelocSection::Tdata,
                offset: 0,
                size: 8,
                is_func: false,
            },
            ElfSymbolDef {
                name: "tls_zero".to_owned(),
                section: RelocSection::Tbss,
                offset: 0,
                size: 8,
                is_func: false,
            },
        ],
        relocs,
        source_files: Vec::new(),
        line_entries: Vec::new(),
        emit_debug_line: false,
        needed_libs: Vec::new(),
        exported_symbols: BTreeSet::new(),
        external_symbols: Vec::new(),
    })
    .expect("link PT_TLS PIE ELF");

    let tls_bin = tmp.path().join("tls_pie");
    write_executable_elf(&tls_elf.bytes, &tls_bin).expect("write tls_pie");
    let (tls_code, _, tls_err) = run_compiled_binary(&tls_bin).expect("run tls_pie");
    assert_eq!(tls_code, 0, "PT_TLS fs:[tpoff] verification failed, stderr: {tls_err}");

    let tls_ph = run_readelf(&tls_bin, &["-l", "-W"]);
    assert!(tls_ph.contains("TLS"), "missing PT_TLS header:\n{tls_ph}");
    let tls_sec = run_readelf(&tls_bin, &["-S", "-W"]);
    assert!(
        tls_sec.contains(".tdata") && tls_sec.contains(".tbss"),
        "missing .tdata/.tbss sections:\n{tls_sec}"
    );
}

#[test]
fn phase6_2_shared_library_ctypes_and_eon_dynamic_linking() {
    let tmp = TempDirGuard::new("shared_lib");

    let shlib_src = r#"
module shlib @hal;

@extern("C")
pub fn add_mul(a: i64, b: i64) -> i64 {
    return (a +% b) *% 3_i64;
}
"#;
    let shlib_path = tmp.path().join("shlib.hal.eon");
    fs::write(&shlib_path, shlib_src).expect("write shlib.hal.eon");
    let so_path = tmp.path().join("libeonsh.so");

    // Compile .so via CLI `eonc build --shared --soname libeonsh.so`
    let build_so = Command::new(eonc_bin())
        .args([
            "build",
            "--shared",
            "--soname",
            "libeonsh.so",
            "-o",
            &so_path.to_string_lossy(),
            &shlib_path.to_string_lossy(),
        ])
        .output()
        .expect("eonc build --shared");
    assert!(
        build_so.status.success(),
        "eonc build --shared failed:\n{}",
        String::from_utf8_lossy(&build_so.stderr)
    );

    let so_dyn = run_readelf(&so_path, &["-d", "-W"]);
    assert!(
        so_dyn.contains("SONAME") && so_dyn.contains("libeonsh.so"),
        "missing SONAME in .so:\n{so_dyn}"
    );
    let so_syms = run_readelf(&so_path, &["--dyn-syms", "-W"]);
    assert!(
        so_syms.contains("add_mul"),
        "add_mul missing from .dynsym:\n{so_syms}"
    );

    // Also verify ElfMode::LinuxShared library API produces valid shared ELF bytes
    let mut sm_api = SourceMap::new();
    let mut int_api = Interner::new();
    let mut diags_api = DiagBag::new();
    let api_opts = X64Options {
        mode: ElfMode::LinuxShared {
            soname: Some("libeonsh.so".to_owned()),
        },
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let api_res = compile_files_to_x64_elf(
        &[("shlib.hal.eon".to_owned(), shlib_src.as_bytes().to_vec())],
        &mut sm_api,
        &mut int_api,
        &mut diags_api,
        &api_opts,
        None,
    )
    .expect("ElfMode::LinuxShared API compile");
    assert!(api_res.elf.symbol_vaddrs.contains_key("add_mul"));
    let api_so_path = tmp.path().join("libeonsh_api.so");
    write_shared_elf(&api_res.elf.bytes, &api_so_path).expect("write_shared_elf");
    assert!(api_so_path.is_file());

    // Test calling libeonsh.so from Python ctypes.CDLL
    let py_script = format!(
        "import ctypes\n\
         lib = ctypes.CDLL({:?})\n\
         lib.add_mul.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.add_mul.restype = ctypes.c_int64\n\
         res = lib.add_mul(10, 4)\n\
         assert res == 42, f'got {{res}}'\n\
         print(res)\n",
        so_path.to_string_lossy()
    );
    let py_out = Command::new("python3")
        .args(["-c", &py_script])
        .output()
        .expect("run python3 ctypes.CDLL");
    assert!(
        py_out.status.success(),
        "python3 ctypes.CDLL failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&py_out.stdout),
        String::from_utf8_lossy(&py_out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&py_out.stdout).trim(), "42");

    // Compile a second Eon executable dynamically linked against libeonsh.so (zero ld/gcc!)
    let client_src = r#"
module app.client @hal;

import std.rt;

extern "C" fn add_mul(a: i64, b: i64) -> i64;

pub fn main() -> i32 {
    let ans = add_mul(10_i64, 4_i64);
    rt.print_i64(ans);
    rt.print_nl();
    if ans == 42_i64 {
        return 0_i32;
    }
    return 1_i32;
}
"#;
    let client_path = tmp.path().join("client.hal.eon");
    fs::write(&client_path, client_src).expect("write client.hal.eon");
    let client_bin = tmp.path().join("client");
    let build_client = Command::new(eonc_bin())
        .args([
            "build",
            "-l",
            &so_path.to_string_lossy(),
            "-o",
            &client_bin.to_string_lossy(),
            &client_path.to_string_lossy(),
        ])
        .output()
        .expect("eonc build client -l libeonsh.so");
    assert!(
        build_client.status.success(),
        "eonc build client failed:\n{}",
        String::from_utf8_lossy(&build_client.stderr)
    );

    let (c_code, c_out, c_err) = run_compiled_binary(&client_bin).expect("run client");
    assert_eq!(c_code, 0, "client stderr: {c_err}");
    assert_eq!(c_out.trim(), "42");
}

// ============================================================================
// Phase 6.3 — Linux _start, Env Capabilities, vDSO < 10 ns, ArgsCap, DirCap
// ============================================================================

#[test]
fn phase6_3_env_args_and_dircap_openat2() {
    let tmp = TempDirGuard::new("env_args_dircap");

    // 1. Env.args via std.rt.arg_count / std.rt.arg_at in BOTH --interp and Native PIE
    let args_src = r#"
module app.args_test;

import std.rt;

pub fn main(env: *mut Env) -> i32 {
    let n = rt.arg_count(&env.args);
    let start: usize = if n == 4_usize { 1_usize } else { 0_usize };
    rt.print_u64((n -% start).widen.[u64]());
    rt.print_nl();
    var i = start;
    while i < n {
        if let s = rt.arg_at(&env.args, i) {
            rt.print_str(s);
            rt.print_nl();
        }
        i = i + 1_usize;
    }
    return 0_i32;
}
"#;
    let args_files = load_with_stdlib(
        tmp.path(),
        &[("args_main.eon", args_src)],
        &["core/str.eon", "std/rt.hal.eon"],
    );
    let args_eon_path = tmp.path().join("args_main.eon");

    // Test via eonc run --interp and eonc run --native
    let interp_cli = Command::new(eonc_bin())
        .args([
            "run",
            "--interp",
            &args_eon_path.to_string_lossy(),
            "--",
            "--hello",
            "world",
            "42",
        ])
        .output()
        .expect("eonc run --interp");
    assert!(
        interp_cli.status.success(),
        "eonc run --interp failed:\n{}",
        String::from_utf8_lossy(&interp_cli.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&interp_cli.stdout).trim(),
        "3\n--hello\nworld\n42"
    );

    let native_cli = Command::new(eonc_bin())
        .args([
            "run",
            "--native",
            &args_eon_path.to_string_lossy(),
            "--",
            "--hello",
            "world",
            "42",
        ])
        .output()
        .expect("eonc run --native");
    assert!(
        native_cli.status.success(),
        "eonc run --native failed:\n{}",
        String::from_utf8_lossy(&native_cli.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&native_cli.stdout).trim(),
        "3\n--hello\nworld\n42"
    );

    // Also test exact arg_count == 4 parity with program name at index 0
    let parity_out = assert_three_way_parity_with_args(
        "args_parity",
        &args_files,
        &[
            "prog".to_owned(),
            "--hello".to_owned(),
            "world".to_owned(),
            "42".to_owned(),
        ],
        &["--hello".to_owned(), "world".to_owned(), "42".to_owned()],
    );
    assert_eq!(parity_out.trim(), "3\n--hello\nworld\n42");

    // 2. DirCap backed by SYS_openat2(RESOLVE_BENEATH) — write, read, reject ../outside.txt with PathEscape
    let jail_root = tmp.path().join("jail_root");
    let jail_sub = jail_root.join("sub");
    fs::create_dir_all(&jail_sub).expect("create jail_sub");
    fs::write(jail_root.join("outside.txt"), b"TOP_SECRET_OUTSIDE").expect("write outside.txt");
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        symlink("../outside.txt", jail_sub.join("sym_escape.txt")).expect("create escape symlink");
    }

    let dircap_src = format!(
        r#"
module app.dircap_test;

import core.str;
import std.rt;

fn read_confined_len(dir: *DirCap, alloc: *mut AllocCap, rel: *str) -> u64 ! IoError | OutOfMemory {{
    let buf = try dir.read_file(alloc, rel);
    let n = buf.len().widen.[u64]();
    buf.destroy(alloc);
    return n;
}}

pub fn main(env: *mut Env) -> i32 ! IoError | OutOfMemory {{
    let dir = try env.fs.sub_dir({:?});
    errdefer dir.close();

    try dir.write_file("inside.txt", "hello_openat2".as_bytes());
    let content = try dir.read_file(&mut env.alloc, "inside.txt");
    let eq_ok: u64 = if str.bytes_eq(content.as_slice(), "hello_openat2".as_bytes()) {{ 1_u64 }} else {{ 0_u64 }};
    content.destroy(&mut env.alloc);

    let esc_tag: u64 = match read_confined_len(&dir, &mut env.alloc, "../outside.txt") {{
        .PathEscape => 14_u64,
        .NotFound => 12_u64,
        .PermissionDenied => 13_u64,
        .AlreadyExists => 15_u64,
        .InvalidInput => 16_u64,
        .Other => 17_u64,
        _ => 0_u64,
    }};

    let sym_tag: u64 = match read_confined_len(&dir, &mut env.alloc, "sym_escape.txt") {{
        .PathEscape => 14_u64,
        .NotFound => 12_u64,
        .PermissionDenied => 13_u64,
        .AlreadyExists => 15_u64,
        .InvalidInput => 16_u64,
        .Other => 17_u64,
        _ => 0_u64,
    }};

    dir.close();
    rt.print_u64(eq_ok); rt.print_nl();
    rt.print_u64(esc_tag); rt.print_nl();
    rt.print_u64(sym_tag); rt.print_nl();
    rt.print_u64(env.alloc.live_allocs().widen.[u64]()); rt.print_nl();
    return 0_i32;
}}
"#,
        jail_sub.to_string_lossy()
    );

    let dircap_files = load_with_stdlib(
        tmp.path(),
        &[("dircap_main.eon", &dircap_src)],
        &["core/str.eon", "std/rt.hal.eon"],
    );
    let dircap_out = assert_three_way_parity_with_args("dircap_openat2", &dircap_files, &[], &[]);
    assert_eq!(dircap_out.trim(), "1\n14\n14\n0");

    // 4. Linear DirCap forgetting dir.close() must fail compilation with E0109
    let leak_dir_src = r#"
module app.leak_dir;

pub fn main(env: *mut Env) -> i32 ! IoError {
    let _dir = try env.fs.sub_dir(".");
    return 0_i32;
}
"#;
    let mut sm_l = SourceMap::new();
    let mut int_l = Interner::new();
    let mut diags_l = DiagBag::new();
    let _ = eonc::check_files_in_mem(
        &[("leak_dir.eon".to_owned(), leak_dir_src.as_bytes().to_vec())],
        &mut sm_l,
        &mut int_l,
        &mut diags_l,
    );
    assert!(
        diags_l.diagnostics().iter().any(|d| d.code == ErrorCode::E0109),
        "forgetting dir.close() must trigger E0109, got:\n{}",
        diags_l.render_all(&sm_l)
    );
}

#[test]
fn phase6_3_vdso_clock_monotonic_speed() {
    let tmp = TempDirGuard::new("vdso_clock");

    // Measure ClockCap.now_ns() over 1,000,000 calls x 5 rounds
    let vdso_src = r#"
module app.vdso_bench;

import std.rt;

pub fn main(env: *mut Env) -> i32 {
    var warm: u64 = 0_u64;
    for _ in 0_usize..10000_usize {
        warm = warm ^ env.clock.now_ns();
    }

    var all_mono: u64 = 1_u64;
    var sink: u64 = warm & 1_u64;
    for _ in 0_usize..5_usize {
        let start = env.clock.now_ns();
        var prev = start;
        var i: usize = 0_usize;
        while i < 1000000_usize {
            let t = env.clock.now_ns();
            if t < prev {
                all_mono = 0_u64;
            }
            prev = t;
            sink = sink ^ (t & 1_u64);
            i = i + 1_usize;
        }
        let end = env.clock.now_ns();
        let elapsed = if end >= start { end - start } else { 0_u64 };
        rt.print_u64(elapsed);
        rt.print_nl();
    }
    rt.print_u64(all_mono);
    rt.print_nl();
    rt.print_u64(sink & 1_u64);
    rt.print_nl();
    return 0_i32;
}
"#;
    let vdso_files = load_with_stdlib(
        tmp.path(),
        &[("vdso_bench.eon", vdso_src)],
        &["core/str.eon", "std/rt.hal.eon"],
    );
    let vdso_bin = compile_native_pie(tmp.path(), "vdso_bench", &vdso_files, false, Vec::new());
    let (v_code, v_out, v_err) = run_compiled_binary(&vdso_bin).expect("run vdso_bench");
    assert_eq!(v_code, 0, "vdso_bench stderr: {v_err}");
    let v_nums: Vec<u64> = v_out
        .lines()
        .map(|l| l.trim().parse::<u64>().expect("parse vdso output"))
        .collect();
    assert_eq!(v_nums.len(), 7);
    assert_eq!(v_nums[5], 1, "vDSO ClockCap.now_ns() must be monotonic");
    let mut round_ns_per_call: Vec<f64> = v_nums[0..5]
        .iter()
        .map(|&ns| (ns as f64) / 1_000_000.0)
        .collect();
    round_ns_per_call.sort_by(f64::total_cmp);
    let median_ns = round_ns_per_call[2];

    // Also measure host hardware C/Rust clock_gettime vDSO vs SYS_clock_gettime syscall baseline
    let base_rs = tmp.path().join("vdso_baseline.rs");
    let base_bin = tmp.path().join("vdso_baseline");
    fs::write(
        &base_rs,
        r#"#[repr(C)]
#[derive(Default)]
struct Ts { s: i64, ns: i64 }
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Ts) -> i32;
    fn syscall(num: i64, ...) -> i64;
}
fn now_vdso() -> u64 {
    let mut ts = Ts::default();
    unsafe { clock_gettime(1, &mut ts); }
    ts.s as u64 * 1_000_000_000 + ts.ns as u64
}
fn now_syscall() -> u64 {
    let mut ts = Ts::default();
    unsafe { syscall(228, 1i64, &mut ts); }
    ts.s as u64 * 1_000_000_000 + ts.ns as u64
}
fn main() {
    let mut s = 0u64;
    let t0 = now_vdso();
    for _ in 0..1_000_000u64 { s ^= now_vdso() & 1; }
    let t1 = now_vdso();
    let t2 = now_vdso();
    for _ in 0..1_000_000u64 { s ^= now_syscall() & 1; }
    let t3 = now_vdso();
    println!("{}\n{}\n{s}", t1 - t0, t3 - t2);
}
"#,
    )
    .expect("write vdso_baseline.rs");
    let rc_out = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "-O",
            "-A",
            "warnings",
            "-o",
            &base_bin.to_string_lossy(),
            &base_rs.to_string_lossy(),
        ])
        .output()
        .expect("compile vdso_baseline");
    assert!(rc_out.status.success(), "{}", String::from_utf8_lossy(&rc_out.stderr));
    let (b_code, b_out, _) = run_compiled_binary(&base_bin).expect("run vdso_baseline");
    assert_eq!(b_code, 0);
    let b_lines: Vec<u64> = b_out.lines().map(|l| l.trim().parse::<u64>().unwrap()).collect();
    let host_vdso_ns = (b_lines[0] as f64) / 1_000_000.0;
    let host_syscall_ns = (b_lines[1] as f64) / 1_000_000.0;
    println!(
        "[phase6.3] vDSO ClockCap.now_ns() over 1,000,000 calls: Eon median = {median_ns:.2} ns/call (rounds = {round_ns_per_call:.2?}), host libc vDSO = {host_vdso_ns:.2} ns/call, host SYS_clock_gettime = {host_syscall_ns:.2} ns/call"
    );
    assert!(
        median_ns * 5.0 < host_syscall_ns,
        "vDSO appears inactive: Eon {median_ns:.2} ns/call vs syscall {host_syscall_ns:.2} ns/call"
    );
    assert!(
        median_ns <= host_vdso_ns * 2.0,
        "vDSO ClockCap.now_ns() measured {median_ns:.2} ns/call (> 2x host libc CLOCK_MONOTONIC vDSO {host_vdso_ns:.2} ns/call)"
    );
    assert!(
        median_ns < 60.0,
        "vDSO ClockCap.now_ns() measured {median_ns:.2} ns/call (expected < 60.0 ns/call ceiling; syscall is {host_syscall_ns:.2} ns/call)"
    );
}

// ============================================================================
// Phase 6.4 — Slab Allocator, AllocCap.sub_budget, Throughput vs malloc, --sanitize
// ============================================================================

#[test]
fn phase6_4_sub_budget_slab_vs_malloc_and_sanitize_traps() {
    let tmp = TempDirGuard::new("alloc_and_sanitize");

    // 1. AllocCap.sub_budget parent accounting + OOM enforcement + zero-leak cleanup (3-way parity)
    let budget_src = r#"
module app.sub_budget_test;

import std.alloc;
import std.rt;

fn try_alloc_u8(cap: *mut AllocCap, n: usize) -> u64 {
    let b = Buf.[u8].init(cap, n) catch |err| match err {
        .OutOfMemory => return 77_u64,
    };
    b.destroy(cap);
    return 0_u64;
}

pub fn main(env: *mut Env) -> i32 ! OutOfMemory {
    var child = alloc.sub_budget(&mut env.alloc, 4096_usize);
    let b1 = try Buf.[u8].init(&mut child, 2048_usize);
    errdefer b1.destroy(&mut child);

    let c_live1 = alloc.live_allocs(&child).widen.[u64]();
    let c_used1 = alloc.used_bytes(&child).widen.[u64]();
    let p_live1 = alloc.live_allocs(&env.alloc).widen.[u64]();
    let p_used1 = alloc.used_bytes(&env.alloc).widen.[u64]();

    // Exceed remaining 2048 bytes -> OutOfMemory cleanly without leaking!
    let oom_code = try_alloc_u8(&mut child, 2049_usize);
    let c_live_after_oom = alloc.live_allocs(&child).widen.[u64]();
    let p_used_after_oom = alloc.used_bytes(&env.alloc).widen.[u64]();

    b1.destroy(&mut child);
    let c_live_end = alloc.live_allocs(&child).widen.[u64]();
    let c_used_end = alloc.used_bytes(&child).widen.[u64]();
    let c_peak_end = alloc.peak_bytes(&child).widen.[u64]();
    let p_live_end = alloc.live_allocs(&env.alloc).widen.[u64]();
    let p_used_end = alloc.used_bytes(&env.alloc).widen.[u64]();

    rt.print_u64(c_live1); rt.print_nl();
    rt.print_u64(c_used1); rt.print_nl();
    rt.print_u64(p_live1); rt.print_nl();
    rt.print_u64(p_used1); rt.print_nl();
    rt.print_u64(oom_code); rt.print_nl();
    rt.print_u64(c_live_after_oom); rt.print_nl();
    rt.print_u64(p_used_after_oom); rt.print_nl();
    rt.print_u64(c_live_end); rt.print_nl();
    rt.print_u64(c_used_end); rt.print_nl();
    rt.print_u64(c_peak_end); rt.print_nl();
    rt.print_u64(p_live_end); rt.print_nl();
    rt.print_u64(p_used_end); rt.print_nl();
    return 0_i32;
}
"#;
    let budget_files = load_with_stdlib(
        tmp.path(),
        &[("budget.eon", budget_src)],
        &["core/mem.eon", "core/str.eon", "std/alloc.hal.eon", "std/rt.hal.eon"],
    );
    let budget_out = assert_three_way_parity_with_args("sub_budget", &budget_files, &[], &[]);
    assert_eq!(
        budget_out.trim(),
        "1\n2048\n1\n2048\n77\n1\n2048\n0\n0\n2048\n0\n0"
    );

    // 2. Slab allocator throughput vs system libc malloc/free (500,000 alloc/free pairs = 1,000,000 ops)
    // Note: dynamic `-l libc.so.6` Eon binaries overwrite ld-linux's FS.base in `eon_rt_tls_init`
    // and bypass glibc `__libc_early_init`, so calling glibc `malloc` (tcache TLS) from Eon `-l libc`
    // traps; therefore we run Eon's static-PIE slab benchmark natively and system `libc` `malloc`/`free`
    // side-by-side in a tiny native ELF compiled via `rustc -O`.
    let bench_src = r#"
module app.slab_bench @hal;

import std.alloc;
import std.rt;

extern "C" fn eon_alloc_raw(cap: *mut AllocCap, size: usize, align: usize) -> rawptr_mut[u8];
extern "C" fn eon_free_raw(cap: *mut AllocCap, ptr: rawptr_mut[u8], size: usize, align: usize);

pub fn main(env: *mut Env) -> i32 ! OutOfMemory {
    // Warmup both paths
    let _ = try alloc.bench_small_allocs(&mut env.alloc, &env.clock, 5000_usize);
    for wi in 0_usize..5000_usize {
        let p = eon_alloc_raw(&mut env.alloc, 32_usize, 16_usize);
        p.* = wi.trunc.[u8]();
        eon_free_raw(&mut env.alloc, p, 32_usize, 16_usize);
    }

    // 500,000 Eon TLS slab Buf cycles (via std.alloc.bench_small_allocs)
    let buf_ns = try alloc.bench_small_allocs(&mut env.alloc, &env.clock, 500000_usize);

    // 500,000 raw Eon TLS slab alloc(32)/free cycles
    let r_start = env.clock.now_ns();
    var sink: u8 = 0_u8;
    for i in 0_usize..500000_usize {
        let p = eon_alloc_raw(&mut env.alloc, 32_usize, 16_usize);
        p.* = i.trunc.[u8]();
        sink = sink ^ p.*;
        eon_free_raw(&mut env.alloc, p, 32_usize, 16_usize);
    }
    let r_end = env.clock.now_ns();
    let raw_ns = if r_end >= r_start { r_end - r_start } else { 0_u64 };

    rt.print_u64(buf_ns); rt.print_nl();
    rt.print_u64(raw_ns); rt.print_nl();
    rt.print_u64(env.alloc.live_allocs().widen.[u64]()); rt.print_nl();
    rt.print_u64(sink.widen.[u64]()); rt.print_nl();
    return 0_i32;
}
"#;
    let bench_files = load_with_stdlib(
        tmp.path(),
        &[("slab_bench.hal.eon", bench_src)],
        &["core/mem.eon", "core/str.eon", "std/alloc.hal.eon", "std/rt.hal.eon"],
    );
    let bench_bin = compile_native_pie(
        tmp.path(),
        "slab_bench",
        &bench_files,
        false,
        Vec::new(),
    );
    let mut buf_rounds: Vec<u64> = Vec::with_capacity(3);
    let mut raw_rounds: Vec<u64> = Vec::with_capacity(3);
    for _ in 0..3 {
        let (b_code, b_out, b_err) = run_compiled_binary(&bench_bin).expect("run slab_bench");
        assert_eq!(b_code, 0, "slab_bench stderr: {b_err}");
        let vals: Vec<u64> = b_out
            .lines()
            .map(|l| l.trim().parse::<u64>().expect("parse bench u64"))
            .collect();
        assert_eq!(vals.len(), 4);
        assert_eq!(vals[2], 0, "live_allocs must be 0");
        buf_rounds.push(vals[0]);
        raw_rounds.push(vals[1]);
    }
    buf_rounds.sort_unstable();
    raw_rounds.sort_unstable();
    let b_vals = [buf_rounds[1], raw_rounds[1]];

    let libc_rs = tmp.path().join("libc_bench.rs");
    let libc_bin = tmp.path().join("libc_bench");
    fs::write(
        &libc_rs,
        r#"use std::ffi::c_void;
use std::time::Instant;
unsafe extern "C" {
    fn malloc(sz: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}
fn main() {
    for i in 0..5000usize {
        unsafe {
            let p = malloc(32) as *mut u8;
            p.write_volatile(i as u8);
            free(p as *mut c_void);
        }
    }
    let t0 = Instant::now();
    let mut sink = 0u8;
    for i in 0..500_000usize {
        unsafe {
            let p = malloc(32) as *mut u8;
            p.write_volatile(i as u8);
            sink ^= p.read_volatile();
            free(p as *mut c_void);
        }
    }
    let ns = t0.elapsed().as_nanos() as u64;
    println!("{ns}\n{sink}");
}
"#,
    )
    .expect("write libc_bench.rs");
    let rc_out = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "-O",
            "-o",
            &libc_bin.to_string_lossy(),
            &libc_rs.to_string_lossy(),
        ])
        .output()
        .expect("invoke rustc for libc malloc benchmark");
    assert!(
        rc_out.status.success(),
        "rustc libc_bench failed:\n{}",
        String::from_utf8_lossy(&rc_out.stderr)
    );
    let (l_code, l_out, l_err) = run_compiled_binary(&libc_bin).expect("run libc_bench");
    assert_eq!(l_code, 0, "libc_bench stderr: {l_err}");
    let libc_ns: u64 = l_out
        .lines()
        .next()
        .expect("libc output line")
        .trim()
        .parse::<u64>()
        .expect("parse libc_ns");

    let eon_buf_ns_per_pair = (b_vals[0] as f64) / 500_000.0;
    let eon_buf_ns_per_op = (b_vals[0] as f64) / 1_000_000.0;
    let eon_raw_ns_per_pair = (b_vals[1] as f64) / 500_000.0;
    let eon_raw_ns_per_op = (b_vals[1] as f64) / 1_000_000.0;
    let libc_ns_per_pair = (libc_ns as f64) / 500_000.0;
    let libc_ns_per_op = (libc_ns as f64) / 1_000_000.0;
    println!(
        "[phase6.4] 500,000 small alloc/free (1M ops): Eon bench_small_allocs = {eon_buf_ns_per_op:.2} ns/op ({eon_buf_ns_per_pair:.2} ns/pair, {} ms), Eon raw TLS slab = {eon_raw_ns_per_op:.2} ns/op ({eon_raw_ns_per_pair:.2} ns/pair, {} ms), libc malloc/free = {libc_ns_per_op:.2} ns/op ({libc_ns_per_pair:.2} ns/pair, {} ms)",
        b_vals[0] / 1_000_000,
        b_vals[1] / 1_000_000,
        libc_ns / 1_000_000
    );
    assert!(
        eon_raw_ns_per_op < 35.0 && eon_buf_ns_per_op < 80.0,
        "Eon TLS slab unexpectedly slow: raw={eon_raw_ns_per_op:.2} ns/op, buf={eon_buf_ns_per_op:.2} ns/op"
    );

    // 3. --sanitize trap tests
    // 3a. Heap OOB write hitting trailing PROT_NONE guard page -> SIGSEGV (139) + backtrace
    let oob_src = r#"
module app.san_oob @hal;

extern "C" fn eon_alloc_raw(cap: *mut AllocCap, size: usize, align: usize) -> rawptr_mut[u8];

pub fn main(env: *mut Env) -> i32 {
    let ptr = eon_alloc_raw(&mut env.alloc, 64_usize, 16_usize);
    let oob = rawptr_mut.[u8].from_usize(ptr.as_usize() +% 64_usize);
    oob.* = 0xFF_u8;
    return 0_i32;
}
"#;
    let oob_files = vec![("san_oob.hal.eon".to_owned(), oob_src.as_bytes().to_vec())];
    let oob_bin = compile_native_pie(tmp.path(), "san_oob", &oob_files, true, Vec::new());
    let (oob_code, _, oob_err) = run_compiled_binary(&oob_bin).expect("run san_oob");
    assert_eq!(oob_code, 139, "expected SIGSEGV(139) on heap OOB, got {oob_code}, stderr:\n{oob_err}");
    assert!(oob_err.contains("EON SANITIZER TRAP") && oob_err.contains("bt:"), "missing trap backtrace:\n{oob_err}");

    // 3b. Use-After-Free hitting PROT_NONE quarantine page -> SIGSEGV (139) + backtrace
    let uaf_src = r#"
module app.san_uaf @hal;

extern "C" fn eon_alloc_raw(cap: *mut AllocCap, size: usize, align: usize) -> rawptr_mut[u8];
extern "C" fn eon_free_raw(cap: *mut AllocCap, ptr: rawptr_mut[u8], size: usize);

pub fn main(env: *mut Env) -> i32 {
    let ptr = eon_alloc_raw(&mut env.alloc, 64_usize, 16_usize);
    let addr = ptr.as_usize();
    rawptr_mut.[u8].from_usize(addr).* = 0x55_u8;
    eon_free_raw(&mut env.alloc, ptr, 64_usize);
    let v = rawptr.[u8].from_usize(addr).*;
    return v.widen.[i32]();
}
"#;
    let uaf_files = vec![("san_uaf.hal.eon".to_owned(), uaf_src.as_bytes().to_vec())];
    let uaf_bin = compile_native_pie(tmp.path(), "san_uaf", &uaf_files, true, Vec::new());
    let (uaf_code, _, uaf_err) = run_compiled_binary(&uaf_bin).expect("run san_uaf");
    assert_eq!(uaf_code, 139, "expected SIGSEGV(139) on UAF, got {uaf_code}, stderr:\n{uaf_err}");
    assert!(uaf_err.contains("EON SANITIZER TRAP") && uaf_err.contains("bt:"), "missing trap backtrace:\n{uaf_err}");

    // 3c. Arena.reset() 0xDE poison verification under --sanitize
    let arena_san_src = r#"
module app.san_arena @hal;

import std.rt;

extern "C" fn eon_arena_alloc_aligned(a: *mut Arena, size: usize, align: usize) -> rawptr_mut[u8];

pub fn main(env: *mut Env) -> i32 ! MemError {
    var arena = try Arena.init(&mut env.alloc, 256_usize);
    let p = eon_arena_alloc_aligned(&mut arena, 64_usize, 16_usize);
    let base_addr = p.as_usize();
    for i in 0_usize..64_usize {
        rawptr_mut.[u8].from_usize(base_addr +% i).* = 0xAA_u8;
    }
    arena.reset();
    var poisoned_ok: u64 = 1_u64;
    for i in 0_usize..64_usize {
        if rawptr.[u8].from_usize(base_addr +% i).* != 0xDE_u8 {
            poisoned_ok = 0_u64;
        }
    }
    let used_after = arena.used_bytes().widen.[u64]();
    arena.destroy(&mut env.alloc);
    rt.print_u64(poisoned_ok); rt.print_nl();
    rt.print_u64(used_after); rt.print_nl();
    rt.print_u64(env.alloc.live_allocs().widen.[u64]()); rt.print_nl();
    return 0_i32;
}
"#;
    let arena_files = load_with_stdlib(
        tmp.path(),
        &[("san_arena.hal.eon", arena_san_src)],
        &["core/str.eon", "std/rt.hal.eon"],
    );
    let arena_bin = compile_native_pie(tmp.path(), "san_arena", &arena_files, true, Vec::new());
    let (a_code, a_out, a_err) = run_compiled_binary(&arena_bin).expect("run san_arena");
    assert_eq!(a_code, 0, "san_arena stderr: {a_err}");
    assert_eq!(a_out.trim(), "1\n0\n0");

    // 3d. Misaligned pointer deref -> SIGILL (132) + backtrace
    let align_src = r#"
module app.san_align @hal;

extern "C" fn eon_alloc_raw(cap: *mut AllocCap, size: usize, align: usize) -> rawptr_mut[u8];

pub fn main(env: *mut Env) -> i32 {
    let ptr = eon_alloc_raw(&mut env.alloc, 64_usize, 16_usize);
    let misaligned = rawptr.[u64].from_usize(ptr.as_usize() +% 1_usize);
    let v = misaligned.*;
    return v.trunc.[i32]();
}
"#;
    let align_files = vec![("san_align.hal.eon".to_owned(), align_src.as_bytes().to_vec())];
    let align_bin = compile_native_pie(tmp.path(), "san_align", &align_files, true, Vec::new());
    let (al_code, _, al_err) = run_compiled_binary(&align_bin).expect("run san_align");
    assert_eq!(al_code, 132, "expected SIGILL(132) on misaligned deref, got {al_code}, stderr:\n{al_err}");
    assert!(al_err.contains("EON SANITIZER TRAP") && al_err.contains("bt:"), "missing trap backtrace:\n{al_err}");

    // 3e. Fiber stack overflow hitting PROT_NONE fiber guard page under --sanitize -> SIGSEGV (139)
    let fiber_ovf_src = r#"
module app.san_fiber_ovf @hal;

extern "C" fn eon_run_fibers_iouring(count: usize, stack_bytes: usize, entry_fn: fn(), arg_ptr: rawptr[u8], sanitize: u64) -> u64;

@recursive
fn blow_fiber_stack(depth: u64) -> u64 {
    var scratch: [1024]u8 = [1_u8; 1024];
    scratch[0_usize] = depth.trunc.[u8]();
    if depth == 0_u64 {
        return scratch[0_usize].widen.[u64]();
    }
    return blow_fiber_stack(depth -% 1_u64) +% scratch[0_usize].widen.[u64]();
}

fn fiber_entry() {
    let _ = blow_fiber_stack(16_u64);
}

pub fn main() -> i32 {
    let _ = eon_run_fibers_iouring(1_usize, 4096_usize, fiber_entry, rawptr.[u8].from_usize(0_usize), 1_u64);
    return 0_i32;
}
"#;
    let fiber_ovf_files = vec![("san_fiber_ovf.hal.eon".to_owned(), fiber_ovf_src.as_bytes().to_vec())];
    let fiber_ovf_bin = compile_native_pie(tmp.path(), "san_fiber_ovf", &fiber_ovf_files, true, Vec::new());
    let (fo_code, _, fo_err) = run_compiled_binary(&fiber_ovf_bin).expect("run san_fiber_ovf");
    assert_eq!(fo_code, 139, "expected SIGSEGV(139) on fiber stack overflow, got {fo_code}, stderr:\n{fo_err}");
    assert!(fo_err.contains("EON SANITIZER TRAP"), "missing trap output:\n{fo_err}");
}

// ============================================================================
// Phase 6.5 — Exit Gate 6 Parallel & Fiber Stress Test (sys + par for + 10k fibers)
// ============================================================================

#[test]
fn phase6_5_exit_gate_6_parallel_schedule_par_for_10k_fibers_stress() {
    let tmp = TempDirGuard::new("exit_gate_6_stress");

    let stress_src = r#"
module app.exit_gate_6;

import core.mem;
import std.rt;
import std.sched;

pub pool WorkItem {
    pub x: u64;
    pub y: u64;
    pub out_sum: u64;
}

// Disjoint columns x and y allow sys_a and sys_c to execute concurrently in Wave 0!
pub sys sys_a(p: write ArenaPool[WorkItem].x) {
    let n = p.len();
    for i in 0_usize..n {
        let id = p.id_at(i) catch { continue; };
        p.x[id] = p.x[id] +% 10_u64;
    }
}

pub sys sys_c(p: write ArenaPool[WorkItem].y) {
    let n = p.len();
    for i in 0_usize..n {
        let id = p.id_at(i) catch { continue; };
        p.y[id] = p.y[id] *% 2_u64;
    }
}

pub sys sys_b(
    rx: read ArenaPool[WorkItem].x,
    ry: read ArenaPool[WorkItem].y,
    wo: write ArenaPool[WorkItem].out_sum,
) {
    let n = rx.len();
    for i in 0_usize..n {
        let id = rx.id_at(i) catch { continue; };
        let y_id = ry.id_at(i) catch { continue; };
        let o_id = wo.id_at(i) catch { continue; };
        wo.out_sum[o_id] = rx.x[id] +% ry.y[y_id];
    }
}

fn run_schedule_phase(alloc: *mut AllocCap) -> u64 ! OutOfMemory | MemError | StaleId {
    var arena = try Arena.init(alloc, 65536_usize);
    errdefer arena.destroy(alloc);
    var apool = try ArenaPool.[WorkItem].init(&mut arena, 256_usize);
    for i in 0_usize..256_usize {
        let iu = i.widen.[u64]();
        let _ = try apool.push(.{ x: iu, y: iu +% 1_u64, out_sum: 0_u64 });
    }

    let frame_sched = comptime Schedule.build(.{
        sys_a.before(sys_b),
        sys_c.before(sys_b),
    });
    frame_sched.run(&mut apool);

    var apool_sum: u64 = 0_u64;
    for i in 0_usize..apool.len() {
        let id = apool.id_at(i) catch { continue; };
        apool_sum = apool_sum +% apool.out_sum[id];
    }
    arena.destroy(alloc);
    return apool_sum;
}

pub struct PoolPhaseRes {
    pub live_cnt: u64;
    pub pool_sum: u64;
}

fn run_pool_par_phase(alloc: *mut AllocCap) -> PoolPhaseRes ! OutOfMemory | StaleId {
    var items = try mem.init_slot_pool.[WorkItem](alloc, 1024_usize);
    errdefer items.destroy(alloc);
    for i in 0_usize..1024_usize {
        let iu = i.widen.[u64]();
        let id = try items.insert(alloc, .{ x: iu, y: iu +% 1_u64, out_sum: 0_u64 });
        if (i & 7_usize) == 0_usize {
            let _ = try items.remove(id);
        }
    }

    par for id in items.each_mut() {
        items.out_sum[id] = (items.x[id] +% 10_u64) +% (items.y[id] *% 2_u64) +% 1_u64;
    }

    var pool_sum: u64 = 0_u64;
    var live_cnt: usize = 0_usize;
    for i in 0_usize..items.cap() {
        if let id = items.id_at(i) {
            pool_sum = pool_sum +% items.out_sum[id];
            live_cnt = live_cnt +% 1_usize;
        }
    }
    items.destroy(alloc);
    let r: PoolPhaseRes = .{ live_cnt: live_cnt.widen.[u64](), pool_sum: pool_sum };
    return r;
}

pub struct SlicePhaseRes {
    pub int_red: u64;
    pub flt_red: u64;
}

fn run_slice_par_phase(alloc: *mut AllocCap) -> SlicePhaseRes ! OutOfMemory {
    var arr = try Buf.[u64].init(alloc, 1024_usize);
    errdefer arr.destroy(alloc);
    var farr = try Buf.[f64].init(alloc, 1024_usize);
    errdefer farr.destroy(alloc);
    for i in 0_usize..1024_usize {
        try arr.push((i.widen.[u64]()) +% 1_u64);
        try farr.push(1.0_f64);
    }
    par for x in arr.as_mut_slice().each_mut() {
        x.* = x.* *% 2_u64;
    }
    let int_red = arr.as_slice().par_reduce(.AddWrap, 0_u64);
    let flt_red = farr.as_slice().par_reduce(.AddTree, 0.0_f64);
    arr.destroy(alloc);
    farr.destroy(alloc);
    let r: SlicePhaseRes = .{ int_red: int_red, flt_red: flt_red.trunc_to_int.[u64]() };
    return r;
}

pub struct StressPhaseRes {
    pub stress_hash: u64;
    pub par_buf_sum: u64;
}

fn run_stress_loop_phase(alloc: *mut AllocCap, io: *IoCap) -> StressPhaseRes ! OutOfMemory | IoError {
    var stress_hash: u64 = 0_u64;
    var par_buf = try Buf.[u64].init(alloc, 512_usize);
    errdefer par_buf.destroy(alloc);
    for _ in 0_usize..512_usize {
        try par_buf.push(1_u64);
    }
    for iter in 0_usize..1000_usize {
        var tmp_buf = try Buf.[u64].init(alloc, 16_usize);
        errdefer tmp_buf.destroy(alloc);
        try tmp_buf.push(iter.widen.[u64]());
        let s = tmp_buf.as_slice();
        if s.len > 0_usize {
            stress_hash = stress_hash +% s[0_usize];
        }
        tmp_buf.destroy(alloc);

        par for elem in par_buf.as_mut_slice().each_mut() {
            elem.* = elem.* +% 1_u64;
        }
        let f_it = try io.run_fibers(4_usize, 4096_usize);
        stress_hash = stress_hash +% f_it.widen.[u64]();
    }
    let par_buf_sum = par_buf.as_slice().par_reduce(.AddWrap, 0_u64);
    par_buf.destroy(alloc);
    let r: StressPhaseRes = .{ stress_hash: stress_hash, par_buf_sum: par_buf_sum };
    return r;
}

pub fn main(env: *mut Env) -> i32 ! OutOfMemory | MemError | IoError | StaleId {
    let start_ns = env.clock.now_ns();

    // 1. Schedule.build(.{ sys_a.before(sys_b), sys_c.before(sys_b) }) + concurrent sys wave on ArenaPool
    let apool_sum = try run_schedule_phase(&mut env.alloc);

    // 2. Multi-threaded `par for` over SlotPool + slice + ParReduce(.AddWrap & .AddTree)
    let pool_res = try run_pool_par_phase(&mut env.alloc);
    let slice_res = try run_slice_par_phase(&mut env.alloc);

    // 3. 10,000 concurrent static-stack fibers on real Linux io_uring
    let fibers_done = try sched.run_fibers(&env.io, 10000_usize, 8192_usize);

    // 4. 1,000-iteration multi-threaded stress loop (alloc/free + par for + io_uring fibers)
    let st_res = try run_stress_loop_phase(&mut env.alloc, &env.io);

    let end_ns = env.clock.now_ns();
    let elapsed_ms = if end_ns >= start_ns { (end_ns - start_ns) / 1000000_u64 } else { 0_u64 };

    rt.print_u64(pool_res.live_cnt); rt.print_nl();
    rt.print_u64(pool_res.pool_sum); rt.print_nl();
    rt.print_u64(apool_sum); rt.print_nl();
    rt.print_u64(slice_res.int_red); rt.print_nl();
    rt.print_u64(slice_res.flt_red); rt.print_nl();
    rt.print_u64(fibers_done.widen.[u64]()); rt.print_nl();
    rt.print_u64(st_res.stress_hash); rt.print_nl();
    rt.print_u64(st_res.par_buf_sum); rt.print_nl();
    rt.print_u64(env.alloc.live_allocs().widen.[u64]()); rt.print_nl();
    rt.print_u64(elapsed_ms); rt.print_nl();
    return 0_i32;
}
"#;
    let stress_files = load_with_stdlib(
        tmp.path(),
        &[("stress.eon", stress_src)],
        &[
            "core/io.eon",
            "core/mem.eon",
            "core/str.eon",
            "core/sync.hal.eon",
            "std/rt.hal.eon",
            "std/sched.hal.eon",
        ],
    );

    // Compute exact expected deterministic checksums in Rust
    let exp_apool_sum: u64 = (0u64..256).map(|i| (i + 10) + ((i + 1) * 2)).sum();
    let mut exp_pool_sum: u64 = 0;
    let mut exp_live: u64 = 0;
    for i in 0u64..1024 {
        if (i & 7) != 0 {
            exp_live += 1;
            exp_pool_sum += (i + 10) + ((i + 1) * 2) + 1;
        }
    }
    let exp_int_red: u64 = (1..=1024u64).map(|x| x * 2).sum(); // 1049600
    let exp_stress_hash: u64 = (0..1000u64).sum::<u64>() + 4 * 1000; // 499500 + 4000 = 503500
    let exp_par_buf_sum: u64 = 512 * 1001; // 512512

    for &(label, sanitize) in &[("normal_pie", false), ("sanitize_pie", true)] {
        let bin = compile_native_pie(tmp.path(), label, &stress_files, sanitize, Vec::new());
        for run_idx in 0..5 {
            let t0 = Instant::now();
            let (code, out, err) = run_compiled_binary(&bin).expect("run exit_gate_6");
            let host_ms = t0.elapsed().as_millis();
            assert_eq!(code, 0, "{label} run {run_idx} exited {code}, stderr:\n{err}");
            assert!(err.is_empty(), "{label} run {run_idx} stderr not empty:\n{err}");
            let vals: Vec<u64> = out
                .lines()
                .map(|l| l.trim().parse::<u64>().expect("parse stress output"))
                .collect();
            assert_eq!(vals.len(), 10, "unexpected output lines for {label}:\n{out}");
            assert_eq!(vals[0], exp_live, "{label} live count mismatch");
            assert_eq!(vals[1], exp_pool_sum, "{label} pool_sum mismatch");
            assert_eq!(vals[2], exp_apool_sum, "{label} apool_sum mismatch");
            assert_eq!(vals[3], exp_int_red, "{label} int_red mismatch");
            assert_eq!(vals[4], 1024, "{label} flt_red mismatch");
            assert_eq!(vals[5], 10_000, "{label} 10,000 io_uring fibers mismatch");
            assert_eq!(vals[6], exp_stress_hash, "{label} stress_hash mismatch");
            assert_eq!(vals[7], exp_par_buf_sum, "{label} par_buf_sum mismatch");
            assert_eq!(vals[8], 0, "{label} live_allocs leak detected");
            if run_idx == 0 {
                println!(
                    "[phase6.5] Exit Gate 6 ({label} x5): 10,000 io_uring fibers + Schedule + par for + 1,000 stress iters passed in {} ms (guest {} ms), live_allocs=0",
                    host_ms, vals[9]
                );
            }
        }
    }
}

// ============================================================================
// Phase 6.6 — build.eon Sandbox & eonc build --audit-tcb
// ============================================================================

#[test]
fn phase6_6_build_eon_sandbox_and_audit_tcb() {
    let tmp = TempDirGuard::new("build_and_audit");
    let pkg_root = tmp.path().join("pkg");
    let app_dir = pkg_root.join("src/app");
    let plugin_dir = pkg_root.join("src/plugin");
    fs::create_dir_all(&app_dir).expect("mkdir pkg/src/app");
    fs::create_dir_all(&plugin_dir).expect("mkdir pkg/src/plugin");

    let hal_src = r#"module drv.hal @hal;

extern "C" fn puts(s: rawptr[u8]) -> i32;

pub fn fence_nop() {
    raw {
        asm("nop");
        asm("pause");
    }
}
"#;
    let main_src = r#"module app.main;

import drv.hal;

pub fn main() -> i32 {
    hal.fence_nop();
    return 0_i32;
}
"#;
    let plugin_src = r#"module plugin @hal;

@extern("C")
pub fn plugin_val() -> i64 {
    return 99_i64;
}
"#;

    let hal_path = app_dir.join("hal.hal.eon");
    let main_path = app_dir.join("main.eon");
    let plugin_path = plugin_dir.join("plugin.hal.eon");
    fs::write(&hal_path, hal_src).expect("write hal.hal.eon");
    fs::write(&main_path, main_src).expect("write main.eon");
    fs::write(&plugin_path, plugin_src).expect("write plugin.hal.eon");

    // 1. CLI `eonc build --audit-tcb` on multi-file package with @hal module
    let audit_bin = tmp.path().join("audited_app");
    let audit_out = Command::new(eonc_bin())
        .args([
            "build",
            "--audit-tcb",
            "-o",
            &audit_bin.to_string_lossy(),
            &main_path.to_string_lossy(),
            &hal_path.to_string_lossy(),
        ])
        .output()
        .expect("run eonc build --audit-tcb");
    assert!(
        audit_out.status.success(),
        "eonc build --audit-tcb failed:\n{}",
        String::from_utf8_lossy(&audit_out.stderr)
    );
    let report_str = String::from_utf8_lossy(&audit_out.stdout);
    let expected_sha = tcb_audit::sha256_hex(hal_src.as_bytes());
    assert!(
        report_str.contains("EON TCB AUDIT REPORT"),
        "missing audit report header:\n{report_str}"
    );
    assert!(
        report_str.contains("module: drv.hal")
            && report_str.contains(&format!("sha256: {expected_sha}"))
            && report_str.contains("raw_lines: 4")
            && report_str.contains("asm_sites: 2")
            && report_str.contains("extern_c: [puts]"),
        "unexpected TCB audit report:\n{report_str}"
    );

    // 2. Valid safe build.eon building exe + shared lib into out/
    let build_eon_path = pkg_root.join("build.eon");
    let valid_build_eon = r#"
module build;

pub fn build(b: *mut BuildCap) {
    b.set_out_dir("out");
    b.set_audit_tcb(true);
    b.allow_hal("drv.hal");
    b.allow_hal("plugin");
    b.add_exe("app", "src/app");
    b.add_shared_lib("libplugin.so", "src/plugin/plugin.hal.eon");
    let _ = b.out_path("manifest.txt");
}
"#;
    fs::write(&build_eon_path, valid_build_eon).expect("write valid build.eon");
    let plan = build_sys::execute_build_eon(&pkg_root, &build_eon_path)
        .expect("execute_build_eon on valid build.eon");
    assert_eq!(plan.out_dir, "out");
    assert!(plan.audit_tcb);
    assert_eq!(plan.targets.len(), 2);
    assert_eq!(plan.targets[0].kind, BuildTargetKind::Exe);
    assert_eq!(plan.targets[1].kind, BuildTargetKind::Shared);

    let cli_build_pkg = Command::new(eonc_bin())
        .current_dir(&pkg_root)
        .args(["build", "build.eon"])
        .output()
        .expect("eonc build build.eon");
    assert!(
        cli_build_pkg.status.success(),
        "eonc build build.eon failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&cli_build_pkg.stdout),
        String::from_utf8_lossy(&cli_build_pkg.stderr)
    );
    let built_exe = pkg_root.join("out/app");
    let built_so = pkg_root.join("out/libplugin.so");
    assert!(built_exe.is_file(), "expected out/app to be built");
    assert!(built_so.is_file(), "expected out/libplugin.so to be built");
    let (exe_code, _, exe_err) = run_compiled_binary(&built_exe).expect("run out/app");
    assert_eq!(exe_code, 0, "out/app stderr: {exe_err}");

    // Verify load_eon_sources works on the package too
    let loaded_pkg = load_eon_sources(&[pkg_root.join("src")]).expect("load pkg sources");
    assert!(loaded_pkg.len() >= 3);

    // 3. Reject hostile build.eon scripts (../escape, @hal, raw {}, asm(), extern "C")
    let hostile_cases = [
        (
            "escape_out_dir",
            "module build;\npub fn build(b: *mut BuildCap) { b.set_out_dir(\"../escape\"); }\n",
            "escapes package root",
        ),
        (
            "escape_target_root",
            "module build;\npub fn build(b: *mut BuildCap) { b.add_exe(\"x\", \"../escape.eon\"); }\n",
            "escapes package root",
        ),
        (
            "escape_out_path",
            "module build;\npub fn build(b: *mut BuildCap) { let _ = b.out_path(\"../pwn\"); }\n",
            "escapes package root",
        ),
        (
            "hal_attr",
            "module build @hal;\npub fn build(b: *mut BuildCap) { let _ = b; }\n",
            "@hal/raw/asm is forbidden",
        ),
        (
            "raw_block",
            "module build;\npub fn build(b: *mut BuildCap) { let _ = b; raw { let _x = 1_i32; } }\n",
            "@hal/raw/asm is forbidden",
        ),
        (
            "inline_asm",
            "module build;\npub fn build(b: *mut BuildCap) { let _ = b; asm(\"nop\"); }\n",
            "@hal/raw/asm is forbidden",
        ),
        (
            "extern_c",
            "module build;\nextern \"C\" fn system(c: rawptr[u8]) -> i32;\npub fn build(b: *mut BuildCap) { let _ = b; }\n",
            "extern C/FFI declarations are forbidden",
        ),
    ];

    for (case_name, src, expected_substr) in hostile_cases {
        let h_path = pkg_root.join("hostile_build.eon");
        fs::write(&h_path, src).expect("write hostile build.eon");
        let err = build_sys::execute_build_eon(&pkg_root, &h_path)
            .expect_err(&format!("hostile build.eon `{case_name}` must fail"));
        assert!(
            err.contains(expected_substr),
            "hostile case `{case_name}` produced unexpected error `{err}` (expected `{expected_substr}`)"
        );
    }
}

// ============================================================================
// Phase 6.7 — Zero C Generation Verification
// ============================================================================

#[test]
fn phase6_7_zero_c_generation() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(
        !Path::new("../compiler/src/codegen_c.rs").exists(),
        "../compiler/src/codegen_c.rs must not exist"
    );
    let src_dir = manifest_dir.join("src");
    assert!(
        !src_dir.join("codegen_c.rs").exists(),
        "compiler/src/codegen_c.rs must not exist"
    );

    fn visit_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read compiler/src") {
            let p = entry.expect("dir entry").path();
            if p.is_dir() {
                visit_rs_files(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }

    let mut rs_files = Vec::new();
    visit_rs_files(&src_dir, &mut rs_files);
    assert!(!rs_files.is_empty());

    let forbidden = [
        "clang",
        "-std=c11",
        "build_c_binary",
        "compile_files_to_c",
        "mod codegen_c",
    ];
    for f in rs_files {
        let text = fs::read_to_string(&f).expect("read compiler/src/*.rs");
        for pat in forbidden {
            assert!(
                !text.contains(pat),
                "forbidden C codegen pattern `{pat}` found in `{}`",
                f.display()
            );
        }
    }
}
