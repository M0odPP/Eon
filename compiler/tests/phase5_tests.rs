use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eonc::diag::{DiagBag, ErrorCode};
use eonc::elf64::ElfMode;
use eonc::emi::{EMI_HEADER_SIZE, EMI_MAGIC, read_emi};
use eonc::span::{Interner, SourceMap};
use eonc::x64_asm::emit_win64_to_eon_trampoline;
use eonc::{
    compile_files_to_spirv, compile_files_to_x64_elf, interpret_files, load_eon_sources,
    run_compiled_binary, write_executable_elf, X64Options,
};

fn test_tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("eon_phase5_tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create phase5 tmp dir");
    dir
}

fn pie_opts(sanitize: bool) -> X64Options {
    X64Options {
        mode: ElfMode::LinuxPie,
        sanitize,
        needed_libs: Vec::new(),
    }
}

struct ThreeWayRun {
    interp: (i32, String),
    pie: (i32, String),
    sanitized: (i32, String),
}

fn compile_run_three_way(name: &str, files: &[(String, Vec<u8>)]) -> ThreeWayRun {
    let dir = test_tmp_dir(name);

    // 1. Reference EIR interpreter
    let mut sm_i = SourceMap::new();
    let mut int_i = Interner::new();
    let mut diags_i = DiagBag::new();
    let interp = interpret_files(files, &mut sm_i, &mut int_i, &mut diags_i)
        .unwrap_or_else(|e| panic!("EIR-Interp failed for `{name}`:\n{e}"));

    // 2. Direct native x86_64 ELF64 PIE (zero C compiler, zero external linker!)
    let mut sm_n = SourceMap::new();
    let mut int_n = Interner::new();
    let mut diags_n = DiagBag::new();
    let emitted_n = compile_files_to_x64_elf(
        files,
        &mut sm_n,
        &mut int_n,
        &mut diags_n,
        &pie_opts(false),
        None,
    )
    .unwrap_or_else(|e| panic!("native x86_64 PIE compile failed for `{name}`:\n{e}"));
    let n_bin = dir.join(format!("{name}_pie"));
    write_executable_elf(&emitted_n.elf.bytes, &n_bin)
        .unwrap_or_else(|e| panic!("write PIE ELF failed for `{name}`: {e}"));
    let (n_code, n_out, n_err) = run_compiled_binary(&n_bin)
        .unwrap_or_else(|e| panic!("native x86_64 PIE run failed for `{name}`: {e}"));
    assert!(n_err.is_empty(), "native PIE stderr for `{name}`:\n{n_err}");

    // 3. Direct native x86_64 ELF64 PIE with `--sanitize`
    let mut sm_s = SourceMap::new();
    let mut int_s = Interner::new();
    let mut diags_s = DiagBag::new();
    let emitted_s = compile_files_to_x64_elf(
        files,
        &mut sm_s,
        &mut int_s,
        &mut diags_s,
        &pie_opts(true),
        None,
    )
    .unwrap_or_else(|e| panic!("native x86_64 --sanitize compile failed for `{name}`:\n{e}"));
    let s_bin = dir.join(format!("{name}_pie_san"));
    write_executable_elf(&emitted_s.elf.bytes, &s_bin)
        .unwrap_or_else(|e| panic!("write sanitized ELF failed for `{name}`: {e}"));
    let (s_code, s_out, s_err) = run_compiled_binary(&s_bin)
        .unwrap_or_else(|e| panic!("native x86_64 --sanitize run failed for `{name}`: {e}"));
    assert!(
        s_err.is_empty(),
        "native --sanitize stderr for `{name}`:\n{s_err}"
    );

    ThreeWayRun {
        interp: (interp.exit_code, interp.stdout),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    }
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
fn exit_gate_5_empty_binary_pie_wx_and_freestanding_under_8kb() {
    let dir = test_tmp_dir("empty_elf");
    let src = r#"
module app.empty;

pub fn main() -> i32 {
    return 0_i32;
}
"#;
    let files = vec![("empty.eon".to_owned(), src.as_bytes().to_vec())];
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let res = compile_files_to_x64_elf(
        &files,
        &mut sm,
        &mut interner,
        &mut diags,
        &pie_opts(false),
        None,
    )
    .expect("native PIE compile of empty main");

    let elf = &res.elf.bytes;
    assert_eq!(&elf[0..4], b"\x7fELF", "must have ELF magic");
    assert_eq!(elf[4], 2, "must be ELFCLASS64");
    assert_eq!(elf[5], 1, "must be ELFDATA2LSB");
    assert_eq!(
        u16::from_le_bytes([elf[16], elf[17]]),
        3,
        "LinuxPie must be ET_DYN (e_type == 3)"
    );
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 62, "must be EM_X86_64");

    // Verify 3 page-aligned PT_LOAD W^X segments: R (4), R|X (5), R|W (6).
    let phoff = u64::from_le_bytes(elf[32..40].try_into().unwrap()) as usize;
    let phentsize = u16::from_le_bytes(elf[54..56].try_into().unwrap()) as usize;
    let phnum = u16::from_le_bytes(elf[56..58].try_into().unwrap()) as usize;
    assert!(phentsize >= 56);
    let mut pt_loads = Vec::new();
    for i in 0..phnum {
        let p = phoff + i * phentsize;
        let p_type = u32::from_le_bytes(elf[p..p + 4].try_into().unwrap());
        if p_type == 1 {
            let p_flags = u32::from_le_bytes(elf[p + 4..p + 8].try_into().unwrap());
            let p_offset = u64::from_le_bytes(elf[p + 8..p + 16].try_into().unwrap());
            let p_vaddr = u64::from_le_bytes(elf[p + 16..p + 24].try_into().unwrap());
            let p_align = u64::from_le_bytes(elf[p + 48..p + 56].try_into().unwrap());
            pt_loads.push((p_flags, p_offset, p_vaddr, p_align));
        }
    }
    assert_eq!(pt_loads.len(), 3, "LinuxPie must have 3 PT_LOAD segments, got {pt_loads:?}");
    let flags: Vec<u32> = pt_loads.iter().map(|(f, _, _, _)| *f).collect();
    assert_eq!(
        flags,
        vec![4, 5, 6],
        "PT_LOAD W^X flags must be [PF_R, PF_R|PF_X, PF_R|PF_W]"
    );
    for (p_flags, p_offset, p_vaddr, p_align) in pt_loads {
        assert_ne!(p_flags & 3, 3, "W^X violation: segment has both PF_W and PF_X");
        assert_eq!(p_align, 0x1000, "PT_LOAD p_align must be 4 KiB page size");
        assert_eq!(p_offset % 0x1000, 0, "PT_LOAD p_offset must be page-aligned");
        assert_eq!(p_vaddr % 0x1000, 0, "PT_LOAD p_vaddr must be page-aligned");
    }

    // Verify DWARF5 `.debug_line` section is present and valid DWARF version 5!
    let debug_line = find_section_in_elf(elf, ".debug_line")
        .expect("DWARF5 .debug_line section must be emitted");
    assert!(debug_line.len() >= 12);
    let dwarf_ver = u16::from_le_bytes([debug_line[4], debug_line[5]]);
    let addr_sz = debug_line[6];
    assert_eq!(dwarf_ver, 5, "DWARF .debug_line must be version 5");
    assert_eq!(addr_sz, 8, "DWARF5 address_size must be 8 for x86_64");

    // Execute native PIE ELF64 binary directly on Linux x86_64 (zero C compiler / linker)!
    let bin_path = dir.join("empty_pie");
    write_executable_elf(elf, &bin_path).expect("write executable PIE ELF64");
    let (code, stdout, stderr) = run_compiled_binary(&bin_path).expect("run empty native PIE");
    assert_eq!(code, 0, "empty main must exit 0");
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());

    // Also verify `--sanitize` on empty PIE binary!
    let mut sm_san = SourceMap::new();
    let mut int_san = Interner::new();
    let mut diags_san = DiagBag::new();
    let res_san = compile_files_to_x64_elf(
        &files,
        &mut sm_san,
        &mut int_san,
        &mut diags_san,
        &pie_opts(true),
        None,
    )
    .expect("native PIE --sanitize compile of empty main");
    let san_path = dir.join("empty_pie_san");
    write_executable_elf(&res_san.elf.bytes, &san_path).expect("write sanitized PIE ELF64");
    let (code_san, out_san, err_san) =
        run_compiled_binary(&san_path).expect("run empty sanitized PIE");
    assert_eq!(code_san, 0);
    assert!(out_san.is_empty());
    assert!(err_san.is_empty());

    // Explicit < 8 KB gate for Freestanding ELF64 (`ET_EXEC`, `e_type == 2`).
    let mut sm_fs = SourceMap::new();
    let mut int_fs = Interner::new();
    let mut diags_fs = DiagBag::new();
    let fs_opts = X64Options {
        mode: ElfMode::Freestanding {
            load_addr: 0x0010_0000,
            multiboot2: true,
        },
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let res_fs = compile_files_to_x64_elf(
        &files,
        &mut sm_fs,
        &mut int_fs,
        &mut diags_fs,
        &fs_opts,
        None,
    )
    .expect("freestanding compile of empty main");
    let fs_elf = &res_fs.elf.bytes;
    assert_eq!(&fs_elf[0..4], b"\x7fELF", "freestanding must have ELF magic");
    assert_eq!(
        u16::from_le_bytes([fs_elf[16], fs_elf[17]]),
        2,
        "Freestanding must be ET_EXEC (e_type == 2)"
    );
    assert_eq!(u16::from_le_bytes([fs_elf[18], fs_elf[19]]), 62, "must be EM_X86_64");
    assert!(
        fs_elf.len() < 8192,
        "empty freestanding ELF64 binary must be < 8 KB, got {} bytes",
        fs_elf.len()
    );

    // Also test non-zero exit code (42) on LinuxPie!
    let src42 = r#"
module app.ret42;
pub fn main() -> i32 { return 42_i32; }
"#;
    let mut sm2 = SourceMap::new();
    let mut int2 = Interner::new();
    let mut diags2 = DiagBag::new();
    let res42 = compile_files_to_x64_elf(
        &[("ret42.eon".into(), src42.as_bytes().to_vec())],
        &mut sm2,
        &mut int2,
        &mut diags2,
        &pie_opts(false),
        None,
    )
    .expect("compile ret42");
    let bin42 = dir.join("ret42_pie");
    write_executable_elf(&res42.elf.bytes, &bin42).expect("write ret42");
    let (code42, _, _) = run_compiled_binary(&bin42).expect("run ret42");
    assert_eq!(code42, 42);
}

#[test]
fn exit_gate_5_freestanding_bare_metal_elf_naked_interrupt_and_asm() {
    let kernel_src = r#"
module kernel.entry @hal;

@naked
pub fn _start() {
    raw {
        asm("cli; nop; hlt");
    }
}

@interrupt @noalloc @stack(128)
pub fn timer_isr() {
    raw {
        asm("nop; mfence");
    }
}

@extern("C")
pub fn c_add(a: i64, b: i64) -> i64 {
    return a +% b;
}
"#;
    let files = vec![("entry.hal.eon".to_owned(), kernel_src.as_bytes().to_vec())];
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let opts = X64Options {
        mode: ElfMode::Freestanding {
            load_addr: 0x0010_0000,
            multiboot2: true,
        },
        sanitize: false,
        needed_libs: Vec::new(),
    };
    let res = compile_files_to_x64_elf(&files, &mut sm, &mut interner, &mut diags, &opts, None)
        .expect("compile freestanding bare-metal ELF64");

    let elf = &res.elf.bytes;
    assert_eq!(&elf[0..4], b"\x7fELF");
    assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), 2, "must be ET_EXEC");
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 62); // EM_X86_64
    assert!(elf.len() < 8192, "freestanding image must be < 8 KB, got {}", elf.len());
    // Program header PT_LOAD maps directly at physical/virtual 0x100000!
    let p_vaddr = u64::from_le_bytes(elf[64 + 16..64 + 24].try_into().unwrap());
    let p_paddr = u64::from_le_bytes(elf[64 + 24..64 + 32].try_into().unwrap());
    assert_eq!(p_vaddr, 0x0010_0000);
    assert_eq!(p_paddr, 0x0010_0000);

    // Verify valid Multiboot2 header magic (0xE85250D6) and zero checksum within first 32 KiB!
    let mb2_pos = elf
        .windows(4)
        .position(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == 0xE852_50D6)
        .expect("Multiboot2 magic must be present");
    assert!(mb2_pos < 32768 && mb2_pos % 8 == 0);
    let mb_magic = u32::from_le_bytes(elf[mb2_pos..mb2_pos + 4].try_into().unwrap());
    let mb_arch = u32::from_le_bytes(elf[mb2_pos + 4..mb2_pos + 8].try_into().unwrap());
    let mb_len = u32::from_le_bytes(elf[mb2_pos + 8..mb2_pos + 12].try_into().unwrap());
    let mb_chk = u32::from_le_bytes(elf[mb2_pos + 12..mb2_pos + 16].try_into().unwrap());
    assert_eq!(
        mb_magic.wrapping_add(mb_arch).wrapping_add(mb_len).wrapping_add(mb_chk),
        0,
        "Multiboot2 header checksum must sum to 0"
    );

    // Verify `.text` contains `@naked` `cli; nop; hlt` (0xFA, 0x90, 0xF4) and `@interrupt` `iretq` (0x48, 0xCF)!
    let text = find_section_in_elf(elf, ".text").expect(".text section");
    assert!(
        text.windows(3).any(|w| w == [0xFA, 0x90, 0xF4]),
        "@naked _start must contain exact `cli; nop; hlt` bytes"
    );
    assert!(
        text.windows(2).any(|w| w == [0x48, 0xCF]),
        "@interrupt timer_isr must return via `iretq` (0x48 0xCF)"
    );
    assert!(res.elf.symbol_vaddrs.contains_key("c_add"), "@extern(\"C\") symbol must be exported unmangled");

    // Verify Win64 ABI adapter trampoline: arg shuffling, Win64-only callee-saved regs, stack args.
    let (w64_code, w64_relocs) = emit_win64_to_eon_trampoline("c_add", 6).expect("win64 trampoline");
    assert_eq!(w64_relocs.len(), 1);
    let has = |pat: &[u8]| w64_code.windows(pat.len()).any(|w| w == pat);
    assert!(has(&[0x57]) && has(&[0x56]), "must save Win64 callee-saved rdi/rsi");
    assert!(has(&[0x41, 0x57]) && has(&[0x41, 0x5F]), "must save/restore r15 (EonCall error register)");
    // movdqu [rsp+0x90], xmm15 / movdqu xmm15, [rsp+0x90]  (REX.R for xmm8..15)
    assert!(has(&[0xF3, 0x44, 0x0F, 0x7F, 0xBC, 0x24, 0x90, 0, 0, 0]), "must spill xmm15");
    assert!(has(&[0xF3, 0x44, 0x0F, 0x6F, 0xBC, 0x24, 0x90, 0, 0, 0]), "must reload xmm15");
    // mov r8, [rbp+48] / mov r9, [rbp+56]: Win64 stack args 5 and 6.
    assert!(has(&[0x4C, 0x8B, 0x45, 0x30]) && has(&[0x4C, 0x8B, 0x4D, 0x38]), "must load Win64 stack args");
    assert!(emit_win64_to_eon_trampoline("c_add", 9).is_err(), ">8 args must be rejected");

    // Negative test: `@extern("C")` returning ErrorUnion must emit E0810!
    let bad_c_ffi = r#"
module kernel.bad_ffi @hal;

error BadErr { Boom }

@extern("C")
pub fn bad_export() -> i32 ! BadErr {
    return 0_i32;
}
"#;
    let mut sm_b = SourceMap::new();
    let mut int_b = Interner::new();
    let mut diags_b = DiagBag::new();
    let _ = eonc::check_files_in_mem(
        &[("bad_ffi.hal.eon".into(), bad_c_ffi.as_bytes().to_vec())],
        &mut sm_b,
        &mut int_b,
        &mut diags_b,
    );
    assert!(
        diags_b.diagnostics().iter().any(|d| d.code == ErrorCode::E0810),
        "expected E0810 for @extern(\"C\") returning ErrorUnion, got: {}",
        diags_b.render_all(&sm_b)
    );
}

#[test]
fn exit_gate_5_parity_core_features_three_way() {
    let rt_src = r#"
module test.rt @hal;

extern "C" fn eon_rt_print_str(s: *str);
extern "C" fn eon_rt_print_u64(v: u64);
extern "C" fn eon_rt_print_i64(v: i64);
extern "C" fn eon_rt_print_f32(v: f32);
extern "C" fn eon_rt_print_nl();

pub fn pstr(s: *str) { raw { eon_rt_print_str(s); } }
pub fn pu64(v: u64) { raw { eon_rt_print_u64(v); } }
pub fn pi64(v: i64) { raw { eon_rt_print_i64(v); } }
pub fn pf32(v: f32) { raw { eon_rt_print_f32(v); } }
pub fn pnl() { raw { eon_rt_print_nl(); } }
"#;
    let main_src = r#"
module test.parity;

import test.rt;

error MathErr { Neg, TooBig }

pub struct BigRes {
    pub a: u64;
    pub b: u64;
    pub c: u64;
}

pub iface Scorer {
    fn score(self: *Self, factor: u64) -> u64;
}

pub struct Flat { pub v: u64; }
pub struct Boost { pub v: u64; pub k: u64; }

impl Scorer for Flat {
    fn score(self: *Self, factor: u64) -> u64 { return self.v *% factor; }
}
impl Scorer for Boost {
    fn score(self: *Self, factor: u64) -> u64 { return (self.v +% self.k) *% factor; }
}

fn eval_dyn(s: *dyn Scorer, f: u64) -> u64 {
    return s.score(f);
}

fn swap[T](a: *mut T, b: *mut T) {
    let t = a.*;
    a.* = b.*;
    b.* = t;
}

@recursive
fn fib(n: u64) -> u64 {
    if n <= 1_u64 { return n; }
    return fib(n -% 1_u64) +% fib(n -% 2_u64);
}

fn make_big(x: i64) -> BigRes ! MathErr {
    if x < 0_i64 { return .Neg; }
    if x > 100_i64 { return .TooBig; }
    let u = x.trunc.[u64]();
    return BigRes.{ a: u, b: u *% 2_u64, c: u *% 3_u64 };
}

fn try_chain(x: i64) -> u64 ! MathErr {
    let r = try make_big(x);
    return r.a +% r.b +% r.c;
}

pub fn main(alloc: *mut AllocCap) -> i32 ! MemError {
    // 1. Recursion + arithmetic + saturating + float
    let f10 = fib(10_u64);
    let sat_hi = 250_u8 +| 30_u8;
    let sat_lo = 5_u8 -| 20_u8;
    let flt = (3.25_f32 + 1.50_f32) * 2.0_f32;

    // 2. Generics with Layout Sharing (3 distinct 8B types)
    var x1: u64 = 10_u64;
    var y1: u64 = 20_u64;
    swap.[u64](&mut x1, &mut y1);

    var x2: i64 = -5_i64;
    var y2: i64 = 99_i64;
    swap.[i64](&mut x2, &mut y2);

    // 3. dyn vtable dispatch
    let fl = Flat.{ v: 7_u64 };
    let bo = Boost.{ v: 4_u64, k: 6_u64 };
    let d1 = eval_dyn(&fl, 3_u64);
    let d2 = eval_dyn(&bo, 5_u64);

    // 4. EonCall r15+CF ErrorUnion + >16B sret return
    let ok_sum = try_chain(10_i64) catch { 999_u64 };
    let err_sum = try_chain(-3_i64) catch { 777_u64 };

    // 5. Atomic + Buf + checked arithmetic
    let at = Atomic.[u64].new(100_u64);
    let prev = at.fetch_add(23_u64, .AcqRel);
    let cas_ok = at.compare_exchange(123_u64, 200_u64, .AcqRel, .Acquire);
    let after = at.load(.Acquire);

    var buf = try Buf.[u64].init(alloc, 4_usize);
    try buf.push(11_u64);
    try buf.push(22_u64);
    let sl = buf.as_slice();
    let bsum = if sl.len >= 2_usize { sl[0_usize] +% sl[1_usize] } else { 0_u64 };
    buf.destroy(alloc);

    let chk_div = (10_u64 /? 0_u64) catch { 555_u64 };

    rt.pstr("fib="); rt.pu64(f10);
    rt.pstr(" sat="); rt.pu64(sat_hi.widen.[u64]()); rt.pstr(","); rt.pu64(sat_lo.widen.[u64]());
    rt.pstr(" flt="); rt.pf32(flt);
    rt.pstr(" sw="); rt.pu64(x1); rt.pstr(","); rt.pi64(x2);
    rt.pstr(" dyn="); rt.pu64(d1); rt.pstr(","); rt.pu64(d2);
    rt.pstr(" eu="); rt.pu64(ok_sum); rt.pstr(","); rt.pu64(err_sum);
    rt.pstr(" at="); rt.pu64(prev); rt.pstr(","); rt.pu64(after);
    rt.pstr(" cas="); rt.pu64(if cas_ok { 1_u64 } else { 0_u64 });
    rt.pstr(" buf="); rt.pu64(bsum);
    rt.pstr(" divz="); rt.pu64(chk_div);
    rt.pstr(" live="); rt.pu64(alloc.live_allocs().widen.[u64]());
    rt.pnl();
    return 0_i32;
}
"#;
    let files = vec![
        ("rt.hal.eon".into(), rt_src.as_bytes().to_vec()),
        ("main.eon".into(), main_src.as_bytes().to_vec()),
    ];
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("parity_core", &files);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0), "exit code mismatch");
    assert_eq!(i_out, n_out, "Interp vs Native PIE stdout mismatch");
    assert_eq!(n_out, s_out, "Native PIE vs --sanitize stdout mismatch");
    assert_eq!(
        n_out.trim(),
        "fib=55 sat=255,0 flt=9.50 sw=20,99 dyn=21,50 eu=60,777 at=100,200 cas=1 buf=33 divz=555 live=0"
    );
}

#[test]
fn exit_gate_5_parity_layout_proto_three_way() {
    let example_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/layout_proto");
    let loaded = load_eon_sources(&[example_dir]).expect("load layout_proto + core.mem");
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("layout_proto", &loaded);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));

    let strip_ns = |s: &str| -> String {
        s.lines()
            .map(|l| l.split(" elapsed_ns=").next().unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(
        strip_ns(&i_out),
        strip_ns(&n_out),
        "layout_proto Interp vs Native PIE output mismatch!"
    );
    assert_eq!(
        strip_ns(&n_out),
        strip_ns(&s_out),
        "layout_proto Native PIE vs --sanitize output mismatch!"
    );
    assert!(n_out.contains("aba_rejected=1"));
    assert!(n_out.contains("final_live_allocs=0"));
}

#[test]
fn exit_gate_5_parity_json_parser_three_way() {
    let example_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/json_parser");
    let loaded = load_eon_sources(&[example_dir]).expect("load json_parser + core");
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("json_parser", &loaded);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));
    assert_eq!(i_out, n_out, "json_parser Interp vs Native PIE mismatch!");
    assert_eq!(n_out, s_out, "json_parser Native PIE vs --sanitize mismatch!");
    assert_eq!(
        n_out.trim(),
        "fields=4 nodes=9 str0_len=3 arr_len=4 int_sum=-8 bad_rejected=1 live=0"
    );
}

#[test]
fn exit_gate_5_gpu_spirv_and_validation_with_spirv_val() {
    let dir = test_tmp_dir("gpu_spirv");
    let shader_src = r#"
module gpu.scale;

@layout(std430)
pub struct Item {
    pub x: f32;
    pub y: f32;
    pub z: f32;
    pub w: f32;
}

@gpu(.Compute, local_size = (8, 8, 1))
pub fn scale_items(
    @binding(0, 0) src: read GpuBuf[Item],
    @binding(0, 1) dst: write GpuBuf[Item],
) {
    let gid = @gpu_global_id().x;
    @gpu_workgroup_barrier();
    let v = src[gid].x * 2.0_f32 + src[gid].y;
    dst[gid].x = v;
    dst[gid].w = 1.0_f32;
}
"#;
    let files = vec![("scale.eon".into(), shader_src.as_bytes().to_vec())];
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let spvs = compile_files_to_spirv(&files, &mut sm, &mut interner, &mut diags)
        .unwrap_or_else(|e| panic!("SPIR-V compile failed:\n{e}"));
    assert_eq!(spvs.len(), 1);
    assert_eq!(spvs[0].entry_name, "scale_items");
    assert_eq!(spvs[0].words[0], 0x0723_0203, "SPIR-V magic");

    let spv_path = dir.join("scale_items.spv");
    fs::write(&spv_path, &spvs[0].bytes).expect("write .spv");

    // Validate generated binary SPIR-V directly with official `spirv-val`!
    let val_out = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1", &spv_path.to_string_lossy()])
        .output()
        .expect("invoke spirv-val");
    assert!(
        val_out.status.success(),
        "spirv-val rejected generated SPIR-V:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&val_out.stdout),
        String::from_utf8_lossy(&val_out.stderr)
    );

    // Negative 1: missing `@layout(std430)` -> E0820
    let bad_std430 = r#"
module gpu.bad_layout;

pub struct NotStd430 {
    pub x: f32;
    pub y: f32;
}

@gpu(.Compute, local_size = (8, 1, 1))
pub fn bad_kernel(@binding(0, 0) buf: read GpuBuf[NotStd430]) {
    let gid = @gpu_global_id().x;
    let _v = buf[gid].x;
}
"#;
    let mut sm1 = SourceMap::new();
    let mut int1 = Interner::new();
    let mut d1 = DiagBag::new();
    let _ = eonc::check_files_in_mem(
        &[("bad_layout.eon".into(), bad_std430.as_bytes().to_vec())],
        &mut sm1,
        &mut int1,
        &mut d1,
    );
    assert!(
        d1.diagnostics().iter().any(|d| d.code == ErrorCode::E0820),
        "expected E0820 for non-std430 struct in @gpu, got:\n{}",
        d1.render_all(&sm1)
    );

    // Negative 2: CPU-only `AllocCap` in `@gpu` -> E0821
    let bad_cap = r#"
module gpu.bad_cap;

@gpu(.Compute, local_size = (8, 1, 1))
pub fn bad_alloc(_alloc: *mut AllocCap) {
}
"#;
    let mut sm2 = SourceMap::new();
    let mut int2 = Interner::new();
    let mut d2 = DiagBag::new();
    let _ = eonc::check_files_in_mem(
        &[("bad_cap.eon".into(), bad_cap.as_bytes().to_vec())],
        &mut sm2,
        &mut int2,
        &mut d2,
    );
    assert!(
        d2.diagnostics().iter().any(|d| d.code == ErrorCode::E0821),
        "expected E0821 for CPU allocator in @gpu, got:\n{}",
        d2.render_all(&sm2)
    );

    // Negative 3: Non-uniform `@gpu_workgroup_barrier()` inside `@gpu_local_id()` branch -> E0822
    let bad_barrier = r#"
module gpu.bad_barrier;

@gpu(.Compute, local_size = (8, 1, 1))
pub fn bad_sync(@binding(0, 0) out_buf: write GpuBuf[u32]) {
    let lid = @gpu_local_id().x;
    if lid < 4_u32 {
        @gpu_workgroup_barrier();
        out_buf[lid] = 1_u32;
    }
}
"#;
    let mut sm3 = SourceMap::new();
    let mut int3 = Interner::new();
    let mut d3 = DiagBag::new();
    let _ = eonc::check_files_in_mem(
        &[("bad_barrier.eon".into(), bad_barrier.as_bytes().to_vec())],
        &mut sm3,
        &mut int3,
        &mut d3,
    );
    assert!(
        d3.diagnostics().iter().any(|d| d.code == ErrorCode::E0822),
        "expected E0822 for non-uniform @gpu_workgroup_barrier, got:\n{}",
        d3.render_all(&sm3)
    );
}

#[test]
fn exit_gate_5_emi_cache_and_per_function_incremental_rebuild() {
    let dir = test_tmp_dir("emi_incremental");
    let cache_dir = dir.join("emi_cache");

    let math_v1 = r#"
module inc.math;

pub fn add_one(x: i64) -> i64 {
    return x +% 1_i64;
}

pub fn mul_two(x: i64) -> i64 {
    return x *% 2_i64;
}

pub fn compute(x: i64) -> i64 {
    return add_one(mul_two(x)) +% 10_i64;
}
"#;
    let app_src = r#"
module inc.app;

import inc.math;

pub fn main() -> i32 {
    let v = math.compute(5_i64);
    return v.trunc.[i32]();
}
"#;

    // Build 1: Cold `.emi` cache -> compiles all 4 functions and writes `.emi` files.
    let files_v1 = vec![
        ("math.eon".into(), math_v1.as_bytes().to_vec()),
        ("app.eon".into(), app_src.as_bytes().to_vec()),
    ];
    let mut sm1 = SourceMap::new();
    let mut int1 = Interner::new();
    let mut d1 = DiagBag::new();
    let r1 = compile_files_to_x64_elf(
        &files_v1,
        &mut sm1,
        &mut int1,
        &mut d1,
        &pie_opts(false),
        Some(&cache_dir),
    )
    .expect("incremental build 1");
    let st1 = r1.stats.expect("incremental stats");
    assert_eq!(st1.fns_total, 4);
    assert_eq!(st1.fns_cached, 0);
    assert_eq!(st1.fns_recompiled, 4);

    // Verify `.emi` binary header magic & deserialization!
    let math_emi_bytes = fs::read(cache_dir.join("inc.math.emi")).expect("inc.math.emi exists");
    assert!(math_emi_bytes.len() >= EMI_HEADER_SIZE);
    assert_eq!(&math_emi_bytes[0..4], &EMI_MAGIC);
    let parsed_emi = read_emi(&math_emi_bytes).expect("parse inc.math.emi");
    assert_eq!(parsed_emi.module_name, "inc.math");
    assert_eq!(parsed_emi.fn_hashes.len(), 3);

    let bin1 = dir.join("app_v1");
    write_executable_elf(&r1.elf.bytes, &bin1).expect("write app_v1");
    let (c1, _, _) = run_compiled_binary(&bin1).expect("run app_v1");
    assert_eq!(c1, 21); // (5 * 2 + 1) + 10 = 21

    // Build 2: Unchanged sources -> 100% cache hit & Early Cutoff on both modules!
    let mut sm2 = SourceMap::new();
    let mut int2 = Interner::new();
    let mut d2 = DiagBag::new();
    let r2 = compile_files_to_x64_elf(
        &files_v1,
        &mut sm2,
        &mut int2,
        &mut d2,
        &pie_opts(false),
        Some(&cache_dir),
    )
    .expect("incremental build 2");
    let st2 = r2.stats.expect("incremental stats 2");
    assert_eq!(st2.modules_early_cutoff, 2);
    assert_eq!(st2.fns_cached, 4);
    assert_eq!(st2.fns_recompiled, 0);

    // Build 3: Change ONLY the body of `compute` (signature unchanged!) ->
    // Early Cutoff on BOTH modules + ONLY 1 function recompiled (`3` reused from `.emi` cache)!
    let math_v2 = r#"
module inc.math;

pub fn add_one(x: i64) -> i64 {
    return x +% 1_i64;
}

pub fn mul_two(x: i64) -> i64 {
    return x *% 2_i64;
}

pub fn compute(x: i64) -> i64 {
    return add_one(mul_two(x)) +% 30_i64;
}
"#;
    let files_v2 = vec![
        ("math.eon".into(), math_v2.as_bytes().to_vec()),
        ("app.eon".into(), app_src.as_bytes().to_vec()),
    ];
    let mut sm3 = SourceMap::new();
    let mut int3 = Interner::new();
    let mut d3 = DiagBag::new();
    let r3 = compile_files_to_x64_elf(
        &files_v2,
        &mut sm3,
        &mut int3,
        &mut d3,
        &pie_opts(false),
        Some(&cache_dir),
    )
    .expect("incremental build 3");
    let st3 = r3.stats.expect("incremental stats 3");
    assert_eq!(st3.modules_early_cutoff, 2, "pub signatures unchanged => Early Cutoff!");
    assert_eq!(st3.fns_cached, 3, "3 unchanged functions must be reused from .emi cache");
    assert_eq!(st3.fns_recompiled, 1, "ONLY the 1 edited function must be recompiled");

    let bin3 = dir.join("app_v2");
    write_executable_elf(&r3.elf.bytes, &bin3).expect("write app_v2");
    let (c3, _, _) = run_compiled_binary(&bin3).expect("run app_v2");
    assert_eq!(c3, 41); // (5 * 2 + 1) + 30 = 41

    // Build 4: Switch `sanitize: false -> true` on same cache dir -> full invalidation!
    let mut sm4 = SourceMap::new();
    let mut int4 = Interner::new();
    let mut d4 = DiagBag::new();
    let r4 = compile_files_to_x64_elf(
        &files_v2,
        &mut sm4,
        &mut int4,
        &mut d4,
        &pie_opts(true),
        Some(&cache_dir),
    )
    .expect("incremental build 4 (sanitize: true)");
    let st4 = r4.stats.expect("incremental stats 4");
    assert_eq!(st4.modules_early_cutoff, 0, "sanitize option change must invalidate Early Cutoff");
    assert_eq!(st4.fns_cached, 0, "sanitize option change must invalidate cached machine blobs");
    assert_eq!(st4.fns_recompiled, 4, "all 4 functions must be recompiled with sanitizer");

    let bin4 = dir.join("app_v2_san");
    write_executable_elf(&r4.elf.bytes, &bin4).expect("write app_v2_san");
    let (c4, _, _) = run_compiled_binary(&bin4).expect("run app_v2_san");
    assert_eq!(c4, 41);

    // Build 5: Rebuild unchanged with `sanitize: true` -> 100% cache hit!
    let mut sm5 = SourceMap::new();
    let mut int5 = Interner::new();
    let mut d5 = DiagBag::new();
    let r5 = compile_files_to_x64_elf(
        &files_v2,
        &mut sm5,
        &mut int5,
        &mut d5,
        &pie_opts(true),
        Some(&cache_dir),
    )
    .expect("incremental build 5 (sanitize: true cache hit)");
    let st5 = r5.stats.expect("incremental stats 5");
    assert_eq!(st5.modules_early_cutoff, 2);
    assert_eq!(st5.fns_cached, 4);
    assert_eq!(st5.fns_recompiled, 0);
}

const PRINT_RT_SRC: &str = r#"
module test.rt @hal;

extern "C" fn eon_rt_print_str(s: *str);
extern "C" fn eon_rt_print_u64(v: u64);
extern "C" fn eon_rt_print_i64(v: i64);
extern "C" fn eon_rt_print_nl();

pub fn pstr(s: *str) { raw { eon_rt_print_str(s); } }
pub fn pu64(v: u64) { raw { eon_rt_print_u64(v); } }
pub fn pi64(v: i64) { raw { eon_rt_print_i64(v); } }
pub fn pnl() { raw { eon_rt_print_nl(); } }
"#;

/// Regression: signed right shift `iN >> k` must be an arithmetic shift across all backends.
#[test]
fn regression_signed_shr_is_arithmetic_on_all_backends() {
    let main_src = r#"
module test.shr;

import test.rt;

fn sh64(x: i64, k: u32) -> i64
    where k < 64_u32
{
    return x >> k;
}
fn sh32(x: i32, k: u32) -> i32
    where k < 32_u32
{
    return x >> k;
}
fn sh16(x: i16, k: u32) -> i16
    where k < 16_u32
{
    return x >> k;
}
fn sh8(x: i8, k: u32) -> i8
    where k < 8_u32
{
    return x >> k;
}
fn shu(x: u64, k: u32) -> u64
    where k < 64_u32
{
    return x >> k;
}

pub fn main() -> i32 {
    var acc: i64 = -1000_i64;
    acc >>= 3_u32;
    rt.pi64(sh64(-1000_i64, 2_u32)); rt.pstr(" ");
    rt.pi64(sh64(-1_i64, 63_u32)); rt.pstr(" ");
    rt.pi64(sh64(-9223372036854775807_i64 -% 1_i64, 62_u32)); rt.pstr(" ");
    rt.pi64(sh64(1000_i64, 2_u32)); rt.pstr(" ");
    rt.pi64(sh32(-7_i32, 1_u32).widen.[i64]()); rt.pstr(" ");
    rt.pi64(sh16(-32768_i16, 15_u32).widen.[i64]()); rt.pstr(" ");
    rt.pi64(sh8(-128_i8, 3_u32).widen.[i64]()); rt.pstr(" ");
    rt.pi64(sh8(127_i8, 3_u32).widen.[i64]()); rt.pstr(" ");
    rt.pu64(shu(0xFFFFFFFFFFFFFFFF_u64, 60_u32)); rt.pstr(" ");
    rt.pi64(acc);
    rt.pnl();
    return 0_i32;
}
"#;
    let files = vec![
        ("rt.hal.eon".into(), PRINT_RT_SRC.as_bytes().to_vec()),
        ("main.eon".into(), main_src.as_bytes().to_vec()),
    ];
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("signed_shr", &files);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));
    assert_eq!(i_out, n_out, "Interp vs Native PIE mismatch on signed `>>`");
    assert_eq!(n_out, s_out, "Native PIE vs --sanitize mismatch on signed `>>`");
    assert_eq!(i_out.trim(), "-250 -1 -2 250 -4 -1 -16 15 15 -125");
}

/// Regression: DWARF5 starts the line-program `file` register at 1, so rows for file 0 were
/// attributed to the second source file; and without `.debug_info` debuggers ignored the table.
#[test]
fn regression_dwarf_line_table_file_attribution_and_compile_unit() {
    let main_src = r#"
module test.dbg;

import test.rt;

fn twice(x: u64) -> u64 {
    let y = x +% x;
    return y;
}

pub fn main() -> i32 {
    rt.pu64(twice(21_u64));
    rt.pnl();
    return 0_i32;
}
"#;
    // `main.eon` deliberately first so it becomes file 0 (the case that used to break).
    let files = vec![
        ("main.eon".into(), main_src.as_bytes().to_vec()),
        ("rt.hal.eon".into(), PRINT_RT_SRC.as_bytes().to_vec()),
    ];
    let mut sm = SourceMap::new();
    let mut int = Interner::new();
    let mut diags = DiagBag::new();
    let res = compile_files_to_x64_elf(
        &files,
        &mut sm,
        &mut int,
        &mut diags,
        &pie_opts(false),
        None,
    )
    .unwrap_or_else(|e| panic!("native PIE compile failed:\n{e}"));
    let elf = &res.elf.bytes;
    assert!(find_section_in_elf(elf, ".debug_info").is_some(), "missing .debug_info compile unit");
    assert!(find_section_in_elf(elf, ".debug_abbrev").is_some(), "missing .debug_abbrev");

    let dir = test_tmp_dir("dwarf_lines");
    let bin = dir.join("dbg");
    write_executable_elf(elf, &bin).expect("write ELF");
    let (code, out, _) = run_compiled_binary(&bin).expect("run");
    assert_eq!((code, out.trim()), (0, "42"));

    let Ok(dump) = Command::new("readelf").arg("--debug-dump=decodedline").arg(&bin).output() else {
        eprintln!("readelf not available; skipping decoded line-table check");
        return;
    };
    let dump = String::from_utf8_lossy(&dump.stdout);
    // `twice` body (lines 7-8 of main.eon) must be attributed to main.eon, never to rt.hal.eon.
    let rows: Vec<(&str, u32)> = dump
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let file = it.next()?;
            let line = it.next()?.parse::<u32>().ok()?;
            file.ends_with(".eon").then_some((file, line))
        })
        .collect();
    assert!(rows.iter().any(|(f, l)| f.ends_with("main.eon") && (7..=8).contains(l)), "rows: {rows:?}");
    let rt_lines = PRINT_RT_SRC.lines().count() as u32;
    assert!(
        rows.iter().all(|(f, l)| !f.ends_with("rt.hal.eon") || *l <= rt_lines),
        "rows attributed to rt.hal.eon beyond its {rt_lines} lines: {rows:?}"
    );
    assert!(!dump.contains("././"), "paths must not be doubled with the `.` directory");
}

/// Regression (found by `tests/fuzz/diff_backends.py`): `i64::MIN` boundary constants and
/// `i64` saturating ops (`+|`, `-|`, `*|`) must match across Interp, Native PIE, and `--sanitize`.
#[test]
fn regression_int64_min_literal_and_i64_saturation() {
    let main_src = r#"
module test.sat64;

import test.rt;

fn sat(a: i64, b: i64) -> i64 {
    let x = a +| b;
    let y = a -| b;
    let z = a *| b;
    return (x ^ y) ^ z;
}

pub fn main() -> i32 {
    let lo = -9223372036854775807_i64 -% 1_i64;
    rt.pi64(sat(lo, -1_i64)); rt.pstr(" ");
    rt.pi64(sat(9223372036854775807_i64, 5_i64)); rt.pstr(" ");
    rt.pi64(-4611686018427387904_i64 *| 4_i64); rt.pstr(" ");
    rt.pi64(lo);
    rt.pnl();
    return 0_i32;
}
"#;
    let files = vec![
        ("rt.hal.eon".into(), PRINT_RT_SRC.as_bytes().to_vec()),
        ("main.eon".into(), main_src.as_bytes().to_vec()),
    ];
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("sat64", &files);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));
    assert_eq!(i_out, n_out, "Interp vs Native PIE mismatch on i64 saturation");
    assert_eq!(n_out, s_out, "Native PIE vs --sanitize mismatch on i64 saturation");
}

/// Regressions found by writing a compiler-shaped program (`examples/compiler_spike`):
/// * `x.try_into.[T]()` (mandated by ZM-05) must lower identically across all execution modes.
/// * `return E.V;` in a `-> ?E` function must wrap the enum tag into the niche optional.
#[test]
fn regression_try_into_and_enum_path_into_optional() {
    let main_src = r#"
module test.conv;

import test.rt;

enum K : u8 { A = 0, B = 1, C = 2 }

fn pick(b: u8) -> ?K {
    if b == 1_u8 { return K.B; }
    if b == 2_u8 { return K.C; }
    return null;
}

fn k_code(o: ?K) -> u64 {
    if let k = o {
        match k {
            .A => { return 10_u64; }
            .B => { return 11_u64; }
            .C => { return 12_u64; }
        }
    }
    return 99_u64;
}

fn to_u32(x: usize) -> u32 ! ArithError { return x.try_into.[u32](); }
fn to_i8(x: i64) -> i8 ! ArithError { return x.try_into.[i8](); }
fn to_u64(x: i32) -> u64 ! ArithError { return x.try_into.[u64](); }
fn to_i64(x: u64) -> i64 ! ArithError { return x.try_into.[i64](); }
fn widen_ok(x: u8) -> u32 ! ArithError { return x.try_into.[u32](); }

pub fn main() -> i32 {
    rt.pu64(k_code(pick(1_u8))); rt.pstr(" ");
    rt.pu64(k_code(pick(2_u8))); rt.pstr(" ");
    rt.pu64(k_code(pick(7_u8))); rt.pstr(" ");
    let a = to_u32(4294967295_usize) catch { 1_u32 };
    let b = to_u32(4294967296_usize) catch { 1_u32 };
    let c = to_i8(-128_i64) catch { 0_i8 };
    let d = to_i8(128_i64) catch { 0_i8 };
    let e = to_u64(-1_i32) catch { 5_u64 };
    let f = to_i64(9223372036854775808_u64) catch { -7_i64 };
    let g = widen_ok(255_u8) catch { 0_u32 };
    rt.pu64(a.widen.[u64]()); rt.pstr(" ");
    rt.pu64(b.widen.[u64]()); rt.pstr(" ");
    rt.pi64(c.widen.[i64]()); rt.pstr(" ");
    rt.pi64(d.widen.[i64]()); rt.pstr(" ");
    rt.pu64(e); rt.pstr(" ");
    rt.pi64(f); rt.pstr(" ");
    rt.pu64(g.widen.[u64]());
    rt.pnl();
    return 0_i32;
}
"#;
    let files = vec![
        ("rt.hal.eon".into(), PRINT_RT_SRC.as_bytes().to_vec()),
        ("main.eon".into(), main_src.as_bytes().to_vec()),
    ];
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("try_into_opt_enum", &files);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));
    assert_eq!(i_out, n_out, "Interp vs Native PIE mismatch");
    assert_eq!(n_out, s_out, "Native PIE vs --sanitize mismatch");
    assert_eq!(i_out.trim(), "11 12 99 4294967295 1 -128 0 5 -7 255");
}

/// The compiler-shaped spike must keep building and producing identical output across all modes.
#[test]
fn compiler_spike_example_three_way_parity() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples").join("compiler_spike");
    let files = load_eon_sources(&[dir]).expect("load compiler_spike + core");
    let ThreeWayRun {
        interp: (i_code, i_out),
        pie: (n_code, n_out),
        sanitized: (s_code, s_out),
    } = compile_run_three_way("compiler_spike", &files);
    assert_eq!((i_code, n_code, s_code), (0, 0, 0));
    assert_eq!(i_out, n_out);
    assert_eq!(n_out, s_out);
    assert_eq!(i_out.trim(), "22\n199\nsymbols=3 ast_nodes=30");
}
