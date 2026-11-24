use std::fs;
use std::path::{Path, PathBuf};

use eonc::cfg::LocalId;
use eonc::codegen_x64::compile_and_link_x64;
use eonc::diag::{DiagBag, ErrorCode};
use eonc::eir::{lint_eir, EirInst, EirInstKind, EirTerm, EirValue};
use eonc::elf64::ElfMode;
use eonc::span::{Interner, SourceMap, Span};
use eonc::types::TyId;
use eonc::{
    check_files_in_mem, compile_files_to_x64_elf, interpret_files, load_eon_sources,
    run_compiled_binary, write_executable_elf, X64Options,
};

fn test_tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("eon_phase5_5_tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create phase5_5 tmp dir");
    dir
}

const TEST_RT_HAL: &str = r#"
module test.rt @hal;

extern "C" fn eon_rt_print_u64(v: u64);
extern "C" fn eon_rt_print_i64(v: i64);
extern "C" fn eon_rt_print_str(s: *str);
extern "C" fn eon_rt_print_nl();

pub fn put_u64(v: u64) { eon_rt_print_u64(v); eon_rt_print_nl(); }
pub fn put_i64(v: i64) { eon_rt_print_i64(v); eon_rt_print_nl(); }
pub fn put_str(s: *str) { eon_rt_print_str(s); }
"#;

fn with_test_rt(main_src: &str) -> Vec<(String, Vec<u8>)> {
    vec![
        ("rt.hal.eon".to_owned(), TEST_RT_HAL.as_bytes().to_vec()),
        ("main.eon".to_owned(), main_src.as_bytes().to_vec()),
    ]
}

fn check_codes(files: &[(String, Vec<u8>)]) -> Vec<ErrorCode> {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let _ = check_files_in_mem(files, &mut sm, &mut interner, &mut diags);
    diags.diagnostics().iter().map(|d| d.code).collect()
}

fn assert_three_way_parity(name: &str, files: &[(String, Vec<u8>)], expected_out: &str) {
    let dir = test_tmp_dir(name);

    // 1. Reference EIR Interpreter
    let mut sm_i = SourceMap::new();
    let mut int_i = Interner::new();
    let mut diags_i = DiagBag::new();
    let interp = interpret_files(files, &mut sm_i, &mut int_i, &mut diags_i)
        .unwrap_or_else(|e| panic!("EIR-Interp failed for `{name}`:\n{e}"));

    // 2. Native x86_64 ELF64 PIE
    let mut sm_n = SourceMap::new();
    let mut int_n = Interner::new();
    let mut diags_n = DiagBag::new();
    let pie_opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let emitted_n = compile_files_to_x64_elf(
        files,
        &mut sm_n,
        &mut int_n,
        &mut diags_n,
        &pie_opts,
        None,
    )
    .unwrap_or_else(|e| panic!("native x86_64 PIE compile failed for `{name}`:\n{e}"));
    let n_bin = dir.join(format!("{name}_pie"));
    write_executable_elf(&emitted_n.elf.bytes, &n_bin).expect("write PIE ELF");
    let (n_code, n_out, n_err) = run_compiled_binary(&n_bin).expect("run native PIE");
    assert!(n_err.is_empty(), "native PIE stderr for `{name}`:\n{n_err}");

    // 3. Native x86_64 ELF64 PIE with `--sanitize`
    let mut sm_s = SourceMap::new();
    let mut int_s = Interner::new();
    let mut diags_s = DiagBag::new();
    let san_opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize: true,
        needed_libs: Vec::new(),
    };
    let emitted_s = compile_files_to_x64_elf(
        files,
        &mut sm_s,
        &mut int_s,
        &mut diags_s,
        &san_opts,
        None,
    )
    .unwrap_or_else(|e| panic!("native x86_64 --sanitize compile failed for `{name}`:\n{e}"));
    let s_bin = dir.join(format!("{name}_pie_san"));
    write_executable_elf(&emitted_s.elf.bytes, &s_bin).expect("write sanitized PIE ELF");
    let (s_code, s_out, s_err) = run_compiled_binary(&s_bin).expect("run native --sanitize");
    assert!(
        s_err.is_empty(),
        "native --sanitize stderr for `{name}`:\n{s_err}"
    );

    assert_eq!(
        (interp.exit_code, n_code, s_code),
        (0, 0, 0),
        "exit code mismatch for `{name}`"
    );
    assert_eq!(
        interp.stdout, n_out,
        "Interp vs Native PIE stdout mismatch for `{name}`"
    );
    assert_eq!(
        n_out, s_out,
        "Native PIE vs Native --sanitize stdout mismatch for `{name}`"
    );
    assert_eq!(
        s_out.trim(),
        expected_out.trim(),
        "unexpected stdout for `{name}`"
    );
}

#[test]
fn phase5_5_eir_lint_and_x64_explicit_ice() {
    let src = r#"
module app.lint;

pub fn main() -> i32 {
    return 0_i32;
}
"#;
    let files = vec![("main.eon".to_owned(), src.as_bytes().to_vec())];
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let chk = check_files_in_mem(&files, &mut sm, &mut interner, &mut diags);
    assert!(!diags.has_errors());
    assert!(lint_eir(&chk.eir, &chk.env).is_ok());

    // 1. Corrupt an EIR value type to TyId::ERROR -> lint_eir must reject!
    let mut bad1 = chk.eir.clone();
    bad1.functions[0].value_tys.push(TyId::ERROR);
    let err1 = lint_eir(&bad1, &chk.env).expect_err("expected EIR-Lint error on TyId::ERROR");
    assert!(err1.contains("EIR-Lint"), "unexpected error: {err1}");

    // 2. Corrupt branch condition to non-bool -> lint_eir must reject!
    let mut bad2 = chk.eir.clone();
    let non_bool = EirValue(bad2.functions[0].value_tys.len() as u32);
    bad2.functions[0].value_tys.push(TyId::I32);
    bad2.functions[0].blocks[0].insts.push(EirInst {
        span: Span::DUMMY,
        kind: EirInstKind::Const {
            dst: non_bool,
            ty: TyId::I32,
            konst: eonc::cfg::CfgConst::Int(1, TyId::I32),
        },
    });
    bad2.functions[0].blocks[0].term = EirTerm::Branch {
        cond: non_bool,
        then_bb: eonc::cfg::BlockId(0),
        else_bb: eonc::cfg::BlockId(0),
        span: Span::DUMMY,
        is_user_if: false,
    };
    let err2 = lint_eir(&bad2, &chk.env).expect_err("expected EIR-Lint error on non-bool branch");
    assert!(err2.contains("EIR-Lint"), "unexpected error: {err2}");

    // 3. Inject unlowered GetVar -> codegen_x64 must emit explicit ICE error (not silently skip)!
    let mut bad3 = chk.eir.clone();
    let gv = EirValue(bad3.functions[0].value_tys.len() as u32);
    bad3.functions[0].value_tys.push(TyId::I32);
    bad3.functions[0].blocks[0].insts.push(EirInst {
        span: Span::DUMMY,
        kind: EirInstKind::GetVar {
            dst: gv,
            local: LocalId(0),
        },
    });
    let pie_opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let err3 = compile_and_link_x64(&bad3, &chk.resolved, &chk.env, &sm, &interner, &pie_opts)
        .expect_err("expected explicit ICE on unlowered GetVar");
    assert!(err3.contains("ICE"), "expected ICE message, got: {err3}");
}

#[test]
fn phase5_5_dbm_verifier_upgrades_positive_and_negative() {
    // Positive: all 5 relational DBM upgrades working together with 3-way parity!
    let pos_src = r#"
module app.dbm_pos;

import test.rt;

pub view struct Reader {
    pub data: *[u8];
    pub pos: usize;
}

impl Reader {
    fn peek_at_pos(self: *Self) -> u8 {
        if self.pos < self.data.len {
            return self.data[self.pos];
        }
        return 0_u8;
    }

    fn scan_word_len(self: *mut Self) -> usize {
        if self.pos > self.data.len {
            return 0_usize;
        }
        let start = self.pos;
        var p = start;
        while p < self.data.len {
            if self.data[p] == b' ' { break; }
            p = p + 1_usize;
        }
        // Upgrade 1: monotonic loop preserves `start <= p <= self.data.len`!
        let word = self.data[start..p];
        self.pos = p;
        return word.len + (p - start);
    }
}

fn diff_algebra_third(s: *[u8], start: usize) -> u8 {
    if start <= s.len && s.len - start >= 3_usize {
        // Upgrade 3: `s.len - start >= 3` => `start + 2 < s.len`!
        return s[start + 2_usize];
    }
    return 0_u8;
}

fn checked_postcond(s: *[u8], base: usize, n: usize) -> usize ! ArithError {
    let end = try (base +? n);
    // Upgrade 4: `try (base +? n)` proves `base <= end`!
    if end <= s.len {
        let sub = s[base..end];
        return sub.len + (end - base);
    }
    return 0_usize;
}

fn known_bits_table(x: usize) -> u64 {
    let tbl: [8]u64 = [10_u64, 20_u64, 30_u64, 40_u64, 50_u64, 60_u64, 70_u64, 80_u64];
    // Upgrade 5: `x & 7` has known upper bound `<= 7 < 8`!
    let idx = x & 7_usize;
    return tbl[idx];
}

pub fn main() -> i32 ! ArithError {
    let bytes = "hello world".as_bytes();
    var r = Reader.{ data: bytes, pos: 0_usize };
    rt.put_u64(r.peek_at_pos().widen.[u64]());
    rt.put_u64(r.scan_word_len().widen.[u64]());
    rt.put_u64(diff_algebra_third(bytes, 1_usize).widen.[u64]());
    rt.put_u64((try checked_postcond(bytes, 2_usize, 4_usize)).widen.[u64]());
    rt.put_u64(known_bits_table(12345_usize));
    return 0_i32;
}
"#;
    assert_three_way_parity("dbm_pos", &with_test_rt(pos_src), "104\n10\n108\n8\n20");

    // Negative 1: `len - start >= 2` is NOT enough for `s[start + 2]` -> E0107!
    let neg_diff = r#"
module app.neg_diff;
pub fn bad(s: *[u8], start: usize) -> u8 {
    if start <= s.len && s.len - start >= 2_usize {
        return s[start + 2_usize];
    }
    return 0_u8;
}
pub fn main() -> i32 { return 0_i32; }
"#;
    let codes1 = check_codes(&[("main.eon".to_owned(), neg_diff.as_bytes().to_vec())]);
    assert!(codes1.contains(&ErrorCode::E0107), "expected E0107, got {codes1:?}");

    // Negative 2: `x & 15` is NOT `< 8` -> E0107!
    let neg_kb = r#"
module app.neg_kb;
pub fn bad(x: usize) -> u64 {
    let tbl: [8]u64 = [0_u64; 8];
    let idx = x & 15_usize;
    return tbl[idx];
}
pub fn main() -> i32 { return 0_i32; }
"#;
    let codes2 = check_codes(&[("main.eon".to_owned(), neg_kb.as_bytes().to_vec())]);
    assert!(codes2.contains(&ErrorCode::E0107), "expected E0107, got {codes2:?}");

    // Negative 3: decrementing loop does NOT preserve `start <= p` -> E0108!
    let neg_mono = r#"
module app.neg_mono;
pub fn bad(n: usize) -> usize {
    let start = n;
    var p = start;
    while p > 0_usize {
        p = p - 1_usize;
    }
    return p - start;
}
pub fn main() -> i32 { return 0_i32; }
"#;
    let codes3 = check_codes(&[("main.eon".to_owned(), neg_mono.as_bytes().to_vec())]);
    assert!(codes3.contains(&ErrorCode::E0108), "expected E0108, got {codes3:?}");
}

#[test]
fn phase5_5_view_struct_ergonomics_and_soundness() {
    let pos_src = r#"
module app.view_erg;

import test.rt;

pub error ParseErr { Bad }

pub enum Kind : u8 { A = 1, B = 2, C = 3 }

pub union Val {
    Num: u64,
    Empty,
}

fn opt_kind(x: u64) -> ?Kind {
    if x == 1_u64 { return .A; }
    if x == 2_u64 { return .B; }
    return null;
}

fn res_val(x: u64) -> Val ! ParseErr {
    if x == 0_u64 { return .Bad; }
    if x == 1_u64 { return .Empty; }
    return .Num(x *% 10_u64);
}

pub view struct SlicePair {
    pub a: *[u64];
    pub b: *mut Buf[u64];
}

impl SlicePair {
    fn copy_mid(self: *mut Self) -> u64 ! OutOfMemory | BoundsError {
        let mid = try self.a.sub(1_usize, 3_usize);
        try self.b.push_slice(mid);
        let p = self.b.get(1_usize) orelse {
            return 999_u64;
        };
        return p.*;
    }
}

pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | BoundsError | ParseErr {
    let arr: [4]u64 = [10_u64, 20_u64, 30_u64, 40_u64];
    var buf = try Buf.[u64].init(alloc, 4_usize);
    errdefer buf.destroy(alloc);
    var sp = SlicePair.{ a: &arr, b: &mut buf };
    let v = try sp.copy_mid();
    rt.put_u64(v);

    // `orelse { ... }` block with side effect & fallback value
    let k = opt_kind(9_u64) orelse {
        rt.put_u64(777_u64);
        .C
    };
    match k {
        .A => { rt.put_u64(1_u64); }
        .B => { rt.put_u64(2_u64); }
        .C => { rt.put_u64(3_u64); }
    }

    // `.Variant(payload)` inference in `Union ! E`
    let rv = try res_val(5_u64);
    match rv {
        .Num(n) => { rt.put_u64(n); }
        .Empty => { rt.put_u64(0_u64); }
    }

    // `sub_mut` + `get_mut` on array
    var mut_arr: [4]u64 = [1_u64, 2_u64, 3_u64, 4_u64];
    let sm = try mut_arr.sub_mut(1_usize, 3_usize);
    if let cell = sm.get_mut(1_usize) {
        cell.* = 99_u64;
    }
    rt.put_u64(mut_arr[2_usize]);

    buf.destroy(alloc);
    return 0_i32;
}
"#;
    assert_three_way_parity("view_erg", &with_test_rt(pos_src), "30\n777\n3\n50\n99");

    // Negative 1: regular struct cannot contain a `view struct` -> E0310
    let neg_store = r#"
module app.neg_view_store;
pub view struct V { pub s: *[u8]; }
pub struct Bad { pub v: V; }
pub fn main() -> i32 { return 0_i32; }
"#;
    let c1 = check_codes(&[("main.eon".to_owned(), neg_store.as_bytes().to_vec())]);
    assert!(c1.contains(&ErrorCode::E0310), "expected E0310, got {c1:?}");

    // Negative 2: non-`view fn` cannot return a `view struct` -> E0310
    let neg_ret = r#"
module app.neg_view_ret;
pub view struct V { pub s: *[u8]; }
fn bad(s: *[u8]) -> V { return V.{ s: s }; }
pub fn main() -> i32 { return 0_i32; }
"#;
    let c2 = check_codes(&[("main.eon".to_owned(), neg_ret.as_bytes().to_vec())]);
    assert!(c2.contains(&ErrorCode::E0310), "expected E0310, got {c2:?}");

    // Negative 3: `view struct` literal aliasing `&mut x` and `&x` -> E0118
    let neg_alias = r#"
module app.neg_view_alias;
pub view struct Pair { pub a: *mut u64; pub b: *u64; }
pub fn main() -> i32 {
    var x: u64 = 1_u64;
    let p = Pair.{ a: &mut x, b: &x };
    let _ = p.b.*;
    return 0_i32;
}
"#;
    let c3 = check_codes(&[("main.eon".to_owned(), neg_alias.as_bytes().to_vec())]);
    assert!(c3.contains(&ErrorCode::E0118), "expected E0118, got {c3:?}");
}

#[test]
fn phase5_5_full_niche_optimization_sizes_and_parity() {
    let src = r#"
module app.niche;

import test.rt;

pub enum Color : u8 { Red = 0, Green = 1, Blue = 2 }

pub error MyErr { A, B }

pub pool Item {
    pub val: u64;
}

pub iface Measurable {
    fn measure(self: *Self) -> u64;
}

pub struct Boxed {
    pub n: u64;
}

impl Measurable for Boxed {
    fn measure(self: *Self) -> u64 {
        return self.n;
    }
}

fn id_fn(x: u64) -> u64 { return x +% 1_u64; }

pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | MemError {
    // 1. Verify exact niche sizes via `@size_of`!
    rt.put_u64(@size_of(?*u64).widen.[u64]());
    rt.put_u64(@size_of(?*[u8]).widen.[u64]());
    rt.put_u64(@size_of(?bool).widen.[u64]());
    rt.put_u64(@size_of(?char).widen.[u64]());
    rt.put_u64(@size_of(?Color).widen.[u64]());
    rt.put_u64(@size_of(?u5).widen.[u64]());
    rt.put_u64(@size_of(?i12).widen.[u64]());
    rt.put_u64(@size_of(?GenId[Item]).widen.[u64]());
    rt.put_u64(@size_of(?*dyn Measurable).widen.[u64]());
    rt.put_u64(@size_of(?fn(u64) -> u64).widen.[u64]());

    // 2. Runtime niche tests across all categories!
    // ?bool: false, true, null
    let b_null: ?bool = null;
    let b_false: ?bool = false;
    let b_true: ?bool = true;
    var acc: u64 = 0_u64;
    if let _ = b_null { acc = acc +% 1000_u64; }
    if let bf = b_false { if !bf { acc = acc +% 1_u64; } }
    if let bt = b_true { if bt { acc = acc +% 2_u64; } }

    // ?*[u8]: null vs valid empty slice (ptr may be NULL, len == 0!) vs non-empty
    let s_null: ?*[u8] = null;
    let empty_s: *[u8] = "".as_bytes();
    let s_empty: ?*[u8] = empty_s;
    if let _ = s_null { acc = acc +% 1000_u64; }
    if let se = s_empty { if se.len == 0_usize { acc = acc +% 4_u64; } }

    // ?u5 and ?i12 at boundary values
    let u5_null: ?u5 = null;
    let u5_max: ?u5 = 31_u5;
    let i12_null: ?i12 = null;
    let i12_neg: ?i12 = -2048_i12;
    if let _ = u5_null { acc = acc +% 1000_u64; }
    if let uv = u5_max { if uv == 31_u5 { acc = acc +% 8_u64; } }
    if let _ = i12_null { acc = acc +% 1000_u64; }
    if let iv = i12_neg { if iv == -2048_i12 { acc = acc +% 16_u64; } }

    // ?Color
    let c_null: ?Color = null;
    let c_red: ?Color = .Red;
    if let _ = c_null { acc = acc +% 1000_u64; }
    if let cr = c_red { if cr == .Red { acc = acc +% 32_u64; } }

    // ?*dyn Measurable
    let bx = Boxed.{ n: 64_u64 };
    let d_null: ?*dyn Measurable = null;
    let d_some: ?*dyn Measurable = &bx;
    if let _ = d_null { acc = acc +% 1000_u64; }
    if let ds = d_some { acc = acc +% ds.measure(); }

    // ?fn(u64) -> u64
    let f_null: ?fn(u64) -> u64 = null;
    let f_some: ?fn(u64) -> u64 = id_fn;
    if let _ = f_null { acc = acc +% 1000_u64; }
    if let fs = f_some { acc = acc +% fs(127_u64); }

    // SlotPool.get -> ?*Item (FatPtrNullFirst niche)
    var sp = try SlotPool.[Item].init(alloc, 4_usize);
    errdefer sp.destroy(alloc);
    let gid = try sp.insert(alloc, .{ val: 256_u64 });
    if let row = sp.get(gid) { acc = acc +% row.val; }
    let _ = sp.remove(gid);
    if let _ = sp.get(gid) { acc = acc +% 1000_u64; }
    sp.destroy(alloc);

    rt.put_u64(acc);
    return 0_i32;
}
"#;
    // Expected sizes: 8, 16, 1, 4, 1, 1, 2, 8, 16, 16; acc = 1+2+4+8+16+32+64+128+256 = 511
    assert_three_way_parity("niche_full", &with_test_rt(src), "8\n16\n1\n4\n1\n1\n2\n8\n16\n16\n511");
}

#[test]
fn phase5_5_packed_bitfield_subword_f16_vec_parity() {
    let src = r#"
module app.lowlevel;

import test.rt;

@packed
pub struct WireHdr {
    pub tag: u8;
    pub seq: u32;
    pub flags: u16;
}

pub bitfield CtrlReg : u32 {
    en: u1 @bits(0..1);
    mode: u3 @bits(1..4);
    irq: u12 @bits(4..16);
    ver: u16 @bits(16..32);
}

pub fn main() -> i32 {
    // 1. @packed struct
    rt.put_u64(@size_of(WireHdr).widen.[u64]());
    var hdr = WireHdr.{ tag: 7_u8, seq: 0x12345678_u32, flags: 0xABCD_u16 };
    hdr.seq = hdr.seq +% 2_u32;
    rt.put_u64(hdr.tag.widen.[u64]() +% hdr.seq.widen.[u64]() +% hdr.flags.widen.[u64]());

    // 2. bitfield
    var reg = CtrlReg.{ en: 1_u1, mode: 5_u3, irq: 300_u12, ver: 42_u16 };
    reg.mode = 6_u3;
    reg.irq = 400_u12;
    rt.put_u64(
        reg.en.widen.[u64]()
        +% reg.mode.widen.[u64]() *% 10_u64
        +% reg.irq.widen.[u64]() *% 100_u64
        +% reg.ver.widen.[u64]() *% 100000_u64
    );

    // 3. Subword arbitrary-width integers (u5, i7, i12)
    let a5: u5 = 25_u5;
    let b5: u5 = 10_u5;
    let w5: u5 = a5 +% b5; // (25 + 10) mod 32 = 3
    let s5: u5 = a5 +| b5; // sat to 31
    let c5: u5 = 200_u32.clamp.[u5](); // clamped to 31
    rt.put_u64(w5.widen.[u64]() *% 10000_u64 +% s5.widen.[u64]() *% 100_u64 +% c5.widen.[u64]());

    let n7: i7 = -40_i7;
    let p7: i7 = 15_i7;
    let sum7: i7 = n7 + p7; // -25
    let shr7: i7 = n7 >> 2_u8; // -10 (arithmetic right shift!)
    rt.put_i64(sum7.widen.[i64]() *% 100_i64 +% shr7.widen.[i64]());

    // 4. f16 (2-byte IEEE 754 binary16)
    rt.put_u64(@size_of(f16).widen.[u64]());
    let h1: f16 = 3.5_f16;
    let h2: f16 = 2.0_f16;
    let h3: f16 = (h1 * h2) + 1.5_f16; // 7.0 + 1.5 = 8.5
    let h4: f16 = -h3; // -8.5
    let h_int: i32 = (h4 * 2.0_f16).trunc_to_int.[i32](); // -17
    rt.put_i64(h_int.widen.[i64]());

    // 5. vec[T, N] (SSE2 packed f32x4 + integer vec[i32, 4])
    let vf1: vec[f32, 4] = .{ 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32 };
    let vf2: vec[f32, 4] = .{ 10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32 };
    let vf_scale: vec[f32, 4] = .{ 2.0_f32, 2.0_f32, 2.0_f32, 2.0_f32 };
    let vf3: vec[f32, 4] = (vf1 + vf2) * vf_scale;
    let vf_sum: i32 = (vf3[0_usize] + vf3[1_usize] + vf3[2_usize] + vf3[3_usize]).trunc_to_int.[i32]();
    // (11 + 22 + 33 + 44) * 2 = 220
    rt.put_i64(vf_sum.widen.[i64]());

    let vi1: vec[i32, 4] = .{ 10_i32, -20_i32, 30_i32, -40_i32 };
    let vi2: vec[i32, 4] = .{ 5_i32, 8_i32, -10_i32, 12_i32 };
    let vi3: vec[i32, 4] = vi1 + vi2;
    let vi4: vec[i32, 4] = -vi3;
    // vi3 = [15, -12, 20, -28], vi4 = [-15, 12, -20, 28], sum = 5
    rt.put_i64((vi4[0_usize] +% vi4[1_usize] +% vi4[2_usize] +% vi4[3_usize]).widen.[i64]());
    let vi_exp: vec[i32, 4] = .{ 15_i32, -12_i32, 20_i32, -28_i32 };
    let veq: u64 = if vi3 == vi_exp { 1_u64 } else { 0_u64 };
    rt.put_u64(veq);
    return 0_i32;
}
"#;
    // WireHdr: size=7; 7 + 0x1234567A (305419898) + 0xABCD (43981) = 305463886
    // CtrlReg: 1 + 60 + 40000 + 4200000 = 4240061
    // u5: 3*10000 + 31*100 + 31 = 33131
    // i7: -25*100 + (-10) = -2510
    // f16: size=2, h_int = -17
    // vec: vf_sum = 220, vi4_sum = 5, veq = 1
    assert_three_way_parity(
        "lowlevel_parity",
        &with_test_rt(src),
        "7\n305463886\n4240061\n33131\n-2510\n2\n-17\n220\n5\n1",
    );
}

#[test]
fn phase5_5_second_class_closures_and_fnref_parity() {
    let src = r#"
module app.closures;

import test.rt;

fn apply_twice(f: fn(u64) -> u64, x: u64) -> u64 {
    return f(f(x));
}

fn for_each_u64(s: *[u64], f: fn(u64)) {
    for i in 0_usize..s.len {
        f(s[i]);
    }
}

fn triple(x: u64) -> u64 {
    return x *% 3_u64;
}

pub struct Adder {
    pub base: u64;
}

impl Adder {
    fn run(self: *Self, x: u64) -> u64 {
        return apply_twice(|v: u64| { v +% self.base }, x);
    }
}

pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory {
    // 1. Top-level fn passed as fn(...) (FnRef thunk)
    rt.put_u64(apply_twice(triple, 2_u64)); // 18

    // 2. Non-capturing closure
    rt.put_u64(apply_twice(|x: u64| { x +% 5_u64 }, 10_u64)); // 20

    // 3. Read-only capturing closure
    let bias: u64 = 7_u64;
    rt.put_u64(apply_twice(|x: u64| { x +% bias }, 1_u64)); // 15

    // 4. Mutating capturing closure invoked across slice
    let nums: [5]u64 = [1_u64, 2_u64, 3_u64, 4_u64, 5_u64];
    var total: u64 = 0_u64;
    for_each_u64(&nums, |n: u64| {
        total = total +% n;
    });
    rt.put_u64(total); // 15

    // 5. Closure capturing `self` inside impl method
    let ad = Adder.{ base: 100_u64 };
    rt.put_u64(ad.run(5_u64)); // 205

    // 6. Buf.drain executing closure at runtime
    var buf = try Buf.[u64].init(alloc, 4_usize);
    errdefer buf.destroy(alloc);
    try buf.push(10_u64);
    try buf.push(20_u64);
    try buf.push(30_u64);
    var drained: u64 = 0_u64;
    buf.drain(alloc, |item| {
        drained = drained +% item;
    });
    rt.put_u64(drained); // 60

    return 0_i32;
}
"#;
    assert_three_way_parity("closures_parity", &with_test_rt(src), "18\n20\n15\n15\n205\n60");

    // Negative: mutating a captured variable while a closure holding a borrow of it is still live -> E0118!
    let neg_capture_alias = r#"
module app.neg_cl;
pub fn main() -> i32 {
    var x: u64 = 1_u64;
    let f: fn(u64) -> u64 = |a: u64| { a +% x };
    x = 2_u64;
    let _ = f(3_u64);
    return 0_i32;
}
"#;
    let c = check_codes(&[("main.eon".to_owned(), neg_capture_alias.as_bytes().to_vec())]);
    assert!(c.contains(&ErrorCode::E0118), "expected E0118 for mutating captured var while closure is live, got {c:?}");
}

#[test]
fn phase5_5_compiler_spike_three_way_parity_and_zero_friction() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let spike_dir = root.join("examples").join("compiler_spike");
    let main_src = fs::read_to_string(spike_dir.join("main.eon")).expect("read compiler_spike/main.eon");
    assert!(!main_src.contains("FRICTION"), "compiler_spike/main.eon must have zero FRICTION comments");
    assert!(!main_src.contains("+%"), "compiler_spike/main.eon must have zero forced `+%`");
    assert!(!main_src.contains("-%"), "compiler_spike/main.eon must have zero forced `-%`");

    let files = load_eon_sources(&[spike_dir]).expect("load compiler_spike");
    assert_three_way_parity("compiler_spike_5_5", &files, "22\n199\nsymbols=3 ast_nodes=30");
}

#[test]
fn phase5_5_simd_masks_reductions_shuffle_swizzle_and_shift_mask() {
    let src = r#"
module app.simd_full;

import test.rt;

pub fn main() -> i32 {
    // 1. f32x4 SSE2 arithmetic + splat + min/max/fma/dot/reduce + swizzle
    let a: f32x4 = .{ 1.0_f32, 4.0_f32, 2.0_f32, 8.0_f32 };
    let b: f32x4 = f32x4.splat(3.0_f32);
    let mn = a.min(b); // {1, 3, 2, 3} -> sum 9
    let mx = a.max(b); // {3, 4, 3, 8} -> sum 18
    let fm = a.fma(b, f32x4.splat(1.0_f32)); // {4, 13, 7, 25} -> sum 49
    rt.put_i64(mn.reduce_add().trunc_to_int.[i32]().widen.[i64]()); // 9
    rt.put_i64(mx.reduce_add().trunc_to_int.[i32]().widen.[i64]()); // 18
    rt.put_i64(fm.reduce_add().trunc_to_int.[i32]().widen.[i64]()); // 49
    rt.put_i64(a.dot(b).trunc_to_int.[i32]().widen.[i64]()); // 45
    rt.put_i64(a.reduce_min().trunc_to_int.[i32]().widen.[i64]()); // 1
    rt.put_i64(a.reduce_max().trunc_to_int.[i32]().widen.[i64]()); // 8

    // Multi-lane swizzle `.yxwz` and `.rgba`
    let sw = a.yxwz; // {4, 1, 8, 2}
    let rg = a.bg;   // {2, 4}
    rt.put_i64((sw.x + sw.z + rg.x + rg.y).trunc_to_int.[i32]().widen.[i64]()); // 4 + 8 + 2 + 4 = 18

    // 2. f64x2 SSE2 packed arithmetic + min/max/dot
    let d1: f64x2 = .{ 2.5_f64, 7.5_f64 };
    let d2: f64x2 = vec.[f64, 2].splat(4.0_f64);
    let dsum = (d1 + d2).min(d1 * d2).reduce_add(); // min({6.5, 11.5}, {10, 30}) -> 6.5 + 11.5 = 18
    rt.put_i64(dsum.trunc_to_int.[i64]()); // 18

    // 3. u8x16 SSE2 SwissTable tag lookup (`pcmpeqb` + `psllw 7` + `pmovmskb`)!
    let tags: u8x16 = .{
        0x10_u8, 0x7F_u8, 0x22_u8, 0x7F_u8,
        0x00_u8, 0x01_u8, 0x02_u8, 0x7F_u8,
        0xAA_u8, 0xBB_u8, 0xCC_u8, 0xDD_u8,
        0xEE_u8, 0x7F_u8, 0x11_u8, 0x7F_u8
    };
    let needle = u8x16.splat(0x7F_u8);
    let m16 = tags.cmpeq(needle);
    // Bits set at lanes 1, 3, 7, 13, 15 => 2 + 8 + 128 + 8192 + 32768 = 41098
    rt.put_u64(m16.bitmask()); // 41098
    rt.put_u64(if m16.any() { 1_u64 } else { 0_u64 }); // 1
    rt.put_u64(if m16.all() { 1_u64 } else { 0_u64 }); // 0

    // 4. i32x4 SSE2 packed ops + cmplt + select + 1-src & 2-src shuffle
    let iv1: i32x4 = .{ -10_i32, 20_i32, -30_i32, 40_i32 };
    let iv2: i32x4 = .{ 5_i32, 15_i32, 25_i32, 35_i32 };
    let imask = iv1.cmplt(iv2); // {true, false, true, false} -> bitmask 0b0101 = 5
    let sel = imask.select(iv2, iv1); // {5, 20, 25, 40} -> sum 90
    rt.put_u64(imask.bitmask()); // 5
    rt.put_i64(sel.reduce_add().widen.[i64]()); // 90

    let idx1: u32x4 = .{ 3_u32, 1_u32, 2_u32, 0_u32 };
    let sh1 = iv1.shuffle(idx1); // {40, 20, -30, -10} -> dot with {1,2,3,4} = 40+40-90-40 = -50
    let w_dot: i32x4 = .{ 1_i32, 2_i32, 3_i32, 4_i32 };
    rt.put_i64(sh1.dot(w_dot).widen.[i64]()); // -50

    let idx2: u32x4 = .{ 0_u32, 5_u32, 3_u32, 6_u32 };
    let sh2 = iv1.shuffle(iv2, idx2); // {-10, 15, 40, 25} -> sum 70
    rt.put_i64(sh2.reduce_add().widen.[i64]()); // 70

    // 5. `<<%` shift-count masking on sub-64 and arbitrary subword integers (`u8`, `u12`)
    let s8: u8 = 3_u8 <<% 10_u32;   // 10 & 7 = 2 => 3 << 2 = 12
    let s12: u12 = 5_u12 <<% 15_u32; // 15 & 11 = 11 => (5 << 11) & 0xFFF = 2048
    rt.put_u64(s8.widen.[u64]()); // 12
    rt.put_u64(s12.widen.[u64]()); // 2048
    return 0_i32;
}
"#;
    assert_three_way_parity(
        "simd_full",
        &with_test_rt(src),
        "9\n18\n49\n45\n1\n8\n18\n18\n41098\n1\n0\n5\n90\n-50\n70\n12\n2048",
    );
}
