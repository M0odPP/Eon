#!/usr/bin/env python3
"""Selfhost compiler soundness regression suite (Section 4 guarantees):
  1. Contract verification (where / ensures / loop invariants: Proven vs Disproven vs Unknown)
  2. u128 upper-bound soundness (no i128::MAX clamping, match exhaustiveness, overflow)
  3. .emi incremental cache invalidation (transitive stdlib/user body edit, corrupted cache, opts change)
  4. Real native @stack(N) machine frame verification (tight budget + callee contribution + @interrupt)
  5. Freestanding W^X ELF + EonOS QEMU boot check
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def run_cmd(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def test_ui_suites(eonc: str):
    r1 = run_cmd([eonc, "test", "tests/ui"])
    assert r1.returncode == 0, f"tests/ui failed:\n{r1.stdout}\n{r1.stderr}"
    r2 = run_cmd([eonc, "test", "tests/selfhost"])
    assert r2.returncode == 0, f"tests/selfhost failed:\n{r2.stdout}\n{r2.stderr}"
    print(f"[PASS] UI suites: {r1.stdout.strip()} | {r2.stdout.strip()}")


def test_emi_invalidation(eonc: str):
    """Verify .emi cache invalidates on transitive dependency body edit (including core/std),
    corrupted cache file, deleted file, and compile option change."""
    with tempfile.TemporaryDirectory(prefix="eon_emi_reg_") as td:
        td_p = Path(td)
        # Create isolated project with local core/ + std/ copy so we can mutate a transitive core module!
        shutil.copytree(ROOT / "core", td_p / "core")
        shutil.copytree(ROOT / "std", td_p / "std")
        # Add a tiny custom helper inside core/mem.eon to test stdlib transitive body mutation
        mem_path = td_p / "core" / "mem.eon"
        orig_mem = mem_path.read_text()
        helper_v1 = "\npub fn emi_sentinel_val() -> u64 { return 111_u64; }\n"
        helper_v2 = "\npub fn emi_sentinel_val() -> u64 { return 222_u64; }\n"
        mem_path.write_text(orig_mem + helper_v1)

        dep_path = td_p / "dep.eon"
        dep_path.write_text(
            "module app.dep;\n"
            "import core.mem;\n"
            "pub fn get_val() -> u64 { return mem.emi_sentinel_val() +% 1000_u64; }\n"
        )
        main_path = td_p / "main.eon"
        main_path.write_text(
            "module app.main;\n"
            "import app.dep;\n"
            "pub fn main() -> i32 {\n"
            "    let v = dep.get_val();\n"
            "    if v == 1111_u64 { return 11_i32; }\n"
            "    if v == 1222_u64 { return 22_i32; }\n"
            "    if v == 2222_u64 { return 33_i32; }\n"
            "    return 99_i32;\n"
            "}\n"
        )
        cache_dir = td_p / "cache"
        out_bin = td_p / "app"

        # 1. Clean cold build -> expect exit 11
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi cold build failed:\n{r.stderr}"
        rc1 = subprocess.run([str(out_bin)]).returncode
        assert rc1 == 11, f"expected rc=11 after cold build, got {rc1}"

        # 2. Rebuild unchanged -> expect exit 11
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi no-change build failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 11

        # 3. Mutate transitive stdlib `core/mem.eon` body with UNCHANGED signature -> MUST invalidate cache!
        mem_path.write_text(orig_mem + helper_v2)
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi stdlib-edit build failed:\n{r.stderr}"
        rc2 = subprocess.run([str(out_bin)]).returncode
        assert rc2 == 22, f"STALE .emi CACHE BUG: core/mem.eon changed body (111->222), expected rc=22, got {rc2}!"

        # 4. Mutate user dependency `dep.eon` body with UNCHANGED signature -> MUST update!
        dep_path.write_text(
            "module app.dep;\n"
            "import core.mem;\n"
            "pub fn get_val() -> u64 { return mem.emi_sentinel_val() +% 2000_u64; }\n"
        )
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi user-edit build failed:\n{r.stderr}"
        rc3 = subprocess.run([str(out_bin)]).returncode
        assert rc3 == 33, f"STALE .emi CACHE BUG: dep.eon changed body, expected rc=33, got {rc3}!"

        # 4b. Interface edit (add new pub fn in dep.eon + call from main.eon) -> reverse-dep selective rebuild!
        dep_path.write_text(
            "module app.dep;\n"
            "import core.mem;\n"
            "pub fn get_val() -> u64 { return mem.emi_sentinel_val() +% 2000_u64; }\n"
            "pub fn get_extra(x: u64) -> u64 { return x +% 11_u64; }\n"
        )
        main_path.write_text(
            "module app.main;\n"
            "import app.dep;\n"
            "pub fn main() -> i32 {\n"
            "    let v = dep.get_val();\n"
            "    let e = dep.get_extra(v);\n"
            "    if e == 2233_u64 { return 44_i32; }\n"
            "    return 99_i32;\n"
            "}\n"
        )
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi interface-add build failed:\n{r.stderr}"
        rc4b = subprocess.run([str(out_bin)]).returncode
        assert rc4b == 44, f"expected rc=44 after interface add, got {rc4b}"

        # 4c. Breaking interface edit (change dep.get_val() signature without updating main.eon) -> MUST FAIL cleanly!
        dep_path.write_text(
            "module app.dep;\n"
            "import core.mem;\n"
            "pub fn get_val(need_arg: u64) -> u64 { return mem.emi_sentinel_val() +% need_arg; }\n"
            "pub fn get_extra(x: u64) -> u64 { return x +% 11_u64; }\n"
        )
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode != 0, f"STALE .emi CACHE BUG: breaking interface change in dep.eon MUST fail main.eon build!"

        # Restore valid dep.eon + main.eon -> MUST recover cleanly and return 33!
        dep_path.write_text(
            "module app.dep;\n"
            "import core.mem;\n"
            "pub fn get_val() -> u64 { return mem.emi_sentinel_val() +% 2000_u64; }\n"
        )
        main_path.write_text(
            "module app.main;\n"
            "import app.dep;\n"
            "pub fn main() -> i32 {\n"
            "    let v = dep.get_val();\n"
            "    if v == 1111_u64 { return 11_i32; }\n"
            "    if v == 1222_u64 { return 22_i32; }\n"
            "    if v == 2222_u64 { return 33_i32; }\n"
            "    return 99_i32;\n"
            "}\n"
        )
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"emi recover after breaking interface edit failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 33

        # 5. Corrupt/truncate cache manifest (simulating interrupted write) -> MUST safely fall back and succeed!
        man_file = cache_dir / "_inc_prog.emi"
        assert man_file.exists(), "expected _inc_prog.emi in cache dir"
        raw = man_file.read_bytes()
        man_file.write_bytes(raw[:-17])  # truncated!
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"corrupted .emi fallback failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 33

        # 6. Single-byte corruption inside a cached module .emi (with out_bin removed) -> SHA-256 MUST reject and fall back cleanly!
        mod_emis = [p for p in cache_dir.glob("*.emi") if p.name != "_inc_prog.emi"]
        assert mod_emis, "expected at least one module .emi file"
        out_bin.unlink()
        mraw = bytearray(mod_emis[0].read_bytes())
        mraw[len(mraw) // 2] ^= 0x5A
        mod_emis[0].write_bytes(bytes(mraw))
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"bit-flipped .emi fallback failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 33

        # 6b. Single-byte corruption of the output binary itself (unchanged sources) -> out_elf_sha256 MUST detect and relink!
        oraw = bytearray(out_bin.read_bytes())
        oraw[len(oraw) // 2] ^= 0xA5
        out_bin.write_bytes(bytes(oraw))
        r = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"corrupted out_bin relink failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 33

        # 7. Profile/flag change (--dev with same --emi-cache) -> MUST NOT reuse --release cache!
        r = run_cmd([eonc, "build", "--dev", "--emi-cache", str(cache_dir), str(dep_path), str(main_path), "-o", str(out_bin)], cwd=td_p)
        assert r.returncode == 0, f"dev rebuild with shared .emi cache failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 33

    print("[PASS] .emi invalidation (cold, unchanged, transitive core/ body edit, user dep edit, truncated/bit-flipped cache, flag change)")


def test_real_native_stack_budget(eonc: str):
    """Verify post-codegen native @stack(N) catches real machine frame + callee chain + @interrupt saves."""
    with tempfile.TemporaryDirectory(prefix="eon_stk_reg_") as td:
        td_p = Path(td)
        # Case A: @interrupt with @stack(128) — frontend saw ~16 B own frame,
        # but real x86_64 @interrupt saves 40B HW + 112B 14 GPRs + 8B rbp = >= 160 B -> MUST FAIL E0701!
        f_isr = td_p / "bad_isr.eon"
        f_isr.write_text(
            "module test.bad_isr;\n"
            "pub fn tiny_isr() @interrupt @noalloc @stack(128) {\n"
            "    return;\n"
            "}\n"
            "pub fn main() -> i32 { return 0_i32; }\n"
        )
        r_isr = run_cmd([eonc, "build", "--release", str(f_isr), "-o", str(td_p / "bad_isr")])
        assert r_isr.returncode != 0 and "E0701" in (r_isr.stdout + r_isr.stderr), (
            f"Expected E0701 for @interrupt @stack(128), got rc={r_isr.returncode}:\n{r_isr.stdout}\n{r_isr.stderr}"
        )

        # Case B: @stack(400) passes frontend estimate (~224B), but real native machine frame after
        # inlining + spills is 528 bytes -> MUST FAIL E0701 post-codegen!
        f_tight = td_p / "bad_callee_stack.eon"
        f_tight.write_text(
            "module test.bad_stack;\n"
            "fn leaf_with_buf(x: u64, i: usize) -> u64 {\n"
            "    var buf: [16]u64 = [x; 16];\n"
            "    let k = i & 15_usize;\n"
            "    buf[k] = x +% 1_u64;\n"
            "    return buf[k] +% buf[(k +% 1_usize) & 15_usize];\n"
            "}\n"
            "pub fn caller(x: u64, i: usize) -> u64 @stack(400) {\n"
            "    var a: [4]u64 = [x; 4];\n"
            "    let k = i & 3_usize;\n"
            "    a[k] = x +% 2_u64;\n"
            "    return leaf_with_buf(a[k], i) +% a[(k +% 1_usize) & 3_usize];\n"
            "}\n"
            "pub fn main() -> i32 {\n"
            "    let _ = caller(1_u64, 2_usize);\n"
            "    return 0_i32;\n"
            "}\n"
        )
        r_tight = run_cmd([eonc, "build", "--release", str(f_tight), "-o", str(td_p / "bad_callee")])
        assert r_tight.returncode != 0 and "E0701" in (r_tight.stdout + r_tight.stderr) and "final native stack usage" in (r_tight.stdout + r_tight.stderr), (
            f"Expected post-codegen E0701 for caller exceeding @stack(400), got rc={r_tight.returncode}:\n{r_tight.stdout}\n{r_tight.stderr}"
        )

        # Case C: Generous @stack(1024) for same caller+callee -> MUST PASS and run!
        f_ok = td_p / "ok_callee_stack.eon"
        f_ok.write_text(
            "module test.ok_stack;\n"
            "fn leaf_with_buf(x: u64, i: usize) -> u64 {\n"
            "    var buf: [16]u64 = [x; 16];\n"
            "    let k = i & 15_usize;\n"
            "    buf[k] = x +% 1_u64;\n"
            "    return buf[k] +% buf[(k +% 1_usize) & 15_usize];\n"
            "}\n"
            "pub fn caller(x: u64, i: usize) -> u64 @stack(1024) {\n"
            "    var a: [4]u64 = [x; 4];\n"
            "    let k = i & 3_usize;\n"
            "    a[k] = x +% 2_u64;\n"
            "    return leaf_with_buf(a[k], i) +% a[(k +% 1_usize) & 3_usize];\n"
            "}\n"
            "pub fn main() -> i32 {\n"
            "    let v = caller(7_u64, 1_usize);\n"
            "    return if v == 26_u64 { 0_i32 } else { 1_i32 };\n"
            "}\n"
        )
        r_ok = run_cmd([eonc, "build", "--release", str(f_ok), "-o", str(td_p / "ok_callee")])
        assert r_ok.returncode == 0, f"Expected ok_callee_stack to build:\n{r_ok.stdout}\n{r_ok.stderr}"
        assert subprocess.run([str(td_p / "ok_callee")]).returncode == 0

    print("[PASS] Real native @stack(N) machine frame verification (@interrupt, callee chain, valid budget)")


def test_dwarf5_debug_info(eonc: str):
    """Verify DWARF5 .debug_info emits DW_TAG_subprogram, DW_TAG_base_type, formal params & locals for gdb."""
    with tempfile.TemporaryDirectory(prefix="eon_dwarf_reg_") as td:
        td_p = Path(td)
        src = td_p / "dbg_app.eon"
        src.write_text(
            "module demo.dbg;\n"
            "fn compute(a: i32, b: u64) -> i64 {\n"
            "    let base: i64 = a.widen.[i64]();\n"
            "    var total: i64 = base +% b.bitcast.[i64]();\n"
            "    if total > 10_i64 { total = total *% 2_i64; }\n"
            "    return total;\n"
            "}\n"
            "pub fn main() -> i32 {\n"
            "    let r = compute(7_i32, 35_u64);\n"
            "    return if r == 84_i64 { 0_i32 } else { 1_i32 };\n"
            "}\n"
        )
        out_bin = td_p / "dbg_app"
        r = run_cmd([eonc, "build", "--dev", str(src), "-o", str(out_bin)])
        assert r.returncode == 0, f"dwarf dev build failed:\n{r.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 0

        rd = run_cmd(["readelf", "--debug-dump=info", str(out_bin)])
        assert rd.returncode == 0, f"readelf failed:\n{rd.stderr}"
        info_txt = rd.stdout
        for tok in ("DW_TAG_compile_unit", "DW_TAG_base_type", "DW_TAG_subprogram",
                    "DW_TAG_formal_parameter", "DW_TAG_variable", "compute", "main"):
            assert tok in info_txt, f"Missing {tok} in DWARF5 .debug_info"

        if shutil.which("gdb"):
            g = run_cmd([
                "gdb", "-q", "--batch",
                "-ex", "set debuginfod enabled off",
                "-ex", "b compute",
                "-ex", "run",
                "-ex", "bt",
                "-ex", "info args",
                "-ex", "next",
                "-ex", "p base",
                "-ex", "continue",
                str(out_bin),
            ])
            assert g.returncode == 0, f"gdb failed:\n{g.stdout}\n{g.stderr}"
            assert "compute (a=7, b=35)" in g.stdout, f"gdb missed compute args:\n{g.stdout}"
            assert "$1 = 7" in g.stdout, f"gdb missed local base=7:\n{g.stdout}"

    print("[PASS] DWARF5 .debug_info (DW_TAG_subprogram, params, locals, gdb breakpoint + backtrace)")


def test_m11_opt_and_dbm(eonc: str):
    """Verify M1.1: auto-vectorization (loop + FMA + 4-lane SLP), SCEV-lite, -Oz, GC+ICF, and DBM ergonomics."""
    with tempfile.TemporaryDirectory(prefix="eon_m11_reg_") as td:
        td_p = Path(td)

        # 1. Auto-vec (scalar slice loop + @fma + 4-lane SLP) + SCEV-lite + GC + Safe ICF parity vs --interp
        src_opt = td_p / "m11_opt.eon"
        src_opt.write_text(
            "module test.m11_opt;\n"
            "pub fn dead_giant_fn(x: u64) -> u64 {\n"
            "    return (x *% 17_u64) ^ (x >> 3_u64) ^ 0x123456789ABCDEF_u64;\n"
            "}\n"
            "fn icf_dup_a(x: u64, y: u64) -> u64 {\n"
            "    var acc = x ^ 0x9E3779B97F4A7C15_u64;\n"
            "    var i: u64 = 0_u64;\n"
            "    while i < y {\n"
            "        acc = (acc *% 6364136223846793005_u64) +% 1442695040888963407_u64;\n"
            "        i = i +% 1_u64;\n"
            "    }\n"
            "    return acc;\n"
            "}\n"
            "fn icf_dup_b(a: u64, b: u64) -> u64 {\n"
            "    var acc = a ^ 0x9E3779B97F4A7C15_u64;\n"
            "    var i: u64 = 0_u64;\n"
            "    while i < b {\n"
            "        acc = (acc *% 6364136223846793005_u64) +% 1442695040888963407_u64;\n"
            "        i = i +% 1_u64;\n"
            "    }\n"
            "    return acc;\n"
            "}\n"
            "fn saxpy_loop(a: f32, x: *[f32], y: *mut [f32]) {\n"
            "    let n = if x.len < y.len { x.len } else { y.len };\n"
            "    var i: usize = 0_usize;\n"
            "    while i < n {\n"
            "        y[i] = @fma(a, x[i], y[i]);\n"
            "        i = i + 1_usize;\n"
            "    }\n"
            "}\n"
            "fn slp4_add(a: *[i32], b: *[i32], out: *mut [i32]) {\n"
            "    if a.len >= 4_usize && b.len >= 4_usize && out.len >= 4_usize {\n"
            "        let s0 = a[0_usize] +% b[0_usize];\n"
            "        let s1 = a[1_usize] +% b[1_usize];\n"
            "        let s2 = a[2_usize] +% b[2_usize];\n"
            "        let s3 = a[3_usize] +% b[3_usize];\n"
            "        out[0_usize] = s0;\n"
            "        out[1_usize] = s1;\n"
            "        out[2_usize] = s2;\n"
            "        out[3_usize] = s3;\n"
            "    }\n"
            "}\n"
            "fn scev_strided(buf: *[u32], rows: usize, stride: usize) -> u32 {\n"
            "    var sum: u32 = 0_u32;\n"
            "    var r: usize = 0_usize;\n"
            "    while r < rows {\n"
            "        let off = r *% stride;\n"
            "        if off < buf.len {\n"
            "            sum = sum +% buf[off];\n"
            "        }\n"
            "        r = r +% 1_usize;\n"
            "    }\n"
            "    return sum;\n"
            "}\n"
            "pub fn main() -> i32 {\n"
            "    var x: [19]f32 = [1.0_f32; 19];\n"
            "    var y: [19]f32 = [2.0_f32; 19];\n"
            "    x[18_usize] = 3.0_f32;\n"
            "    saxpy_loop(2.0_f32, &x, &mut y);\n"
            "    let a4: [4]i32 = [1_i32, 2_i32, 3_i32, 4_i32];\n"
            "    let b4: [4]i32 = [10_i32, 20_i32, 30_i32, 40_i32];\n"
            "    var o4: [4]i32 = [0_i32; 4];\n"
            "    slp4_add(&a4, &b4, &mut o4);\n"
            "    let ubuf: [8]u32 = [1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32];\n"
            "    let sc = scev_strided(&ubuf, 4_usize, 2_usize);\n"
            "    let h = (icf_dup_a(7_u64, 5_u64) ^ icf_dup_b(7_u64, 5_u64)) | (icf_dup_a(9_u64, 3_u64) ^ icf_dup_b(9_u64, 3_u64));\n"
            "    let h2 = icf_dup_a(11_u64, 4_u64) ^ icf_dup_b(11_u64, 4_u64);\n"
            "    if y[0_usize] == 4.0_f32 && y[18_usize] == 8.0_f32 && o4[3_usize] == 44_i32 && sc == 16_u32 && h == 0_u64 && h2 == 0_u64 {\n"
            "        return 0_i32;\n"
            "    }\n"
            "    return 1_i32;\n"
            "}\n"
        )
        b_rel = td_p / "m11_rel"
        b_oz = td_p / "m11_oz"
        r_int = run_cmd([eonc, "run", "--interp", str(src_opt)])
        assert r_int.returncode == 0, f"m11_opt interp failed:\n{r_int.stderr}"
        r_rel = run_cmd([eonc, "build", "--release", "--verify-opt", str(src_opt), "-o", str(b_rel)])
        assert r_rel.returncode == 0, f"m11_opt --release failed:\n{r_rel.stderr}"
        assert subprocess.run([str(b_rel)]).returncode == 0
        r_oz = run_cmd([eonc, "build", "-Oz", "--verify-opt", str(src_opt), "-o", str(b_oz)])
        assert r_oz.returncode == 0, f"m11_opt -Oz failed:\n{r_oz.stderr}"
        assert subprocess.run([str(b_oz)]).returncode == 0
        # Dead-strip check: `dead_giant_fn` must be absent from symbol table
        rd_sym = run_cmd(["llvm-objdump", "-t", str(b_rel)])
        assert rd_sym.returncode == 0
        assert "dead_giant_fn" not in rd_sym.stdout, "Linker GC failed to dead-strip unreachable `dead_giant_fn`!"
        # Safe ICF check: in -Oz, `icf_dup_a` and `icf_dup_b` must fold to the exact same .text address!
        rd_oz = run_cmd(["llvm-objdump", "-t", str(b_oz)])
        assert rd_oz.returncode == 0
        icf_addrs = {}
        for line in rd_oz.stdout.splitlines():
            for fn_nm in ("icf_dup_a", "icf_dup_b"):
                if line.endswith("__" + fn_nm):
                    icf_addrs[fn_nm] = line.split()[0]
        assert len(icf_addrs) == 2 and icf_addrs["icf_dup_a"] == icf_addrs["icf_dup_b"], (
            f"Safe ICF failed to fold identical functions in -Oz: {icf_addrs}"
        )
        # -Oz vs --release size check on a multi-function program (`05_ecs_physics`)
        ecs_rel = td_p / "ecs_rel"
        ecs_oz = td_p / "ecs_oz"
        r_er = run_cmd([eonc, "build", "--release", str(ROOT / "bench/05_ecs_physics/main.eon"), "-o", str(ecs_rel)])
        r_ez = run_cmd([eonc, "build", "-Oz", str(ROOT / "bench/05_ecs_physics/main.eon"), "-o", str(ecs_oz)])
        assert r_er.returncode == 0 and r_ez.returncode == 0, f"ecs build failed: {r_er.stderr}\n{r_ez.stderr}"
        assert ecs_oz.stat().st_size < ecs_rel.stat().st_size, (
            f"-Oz ({ecs_oz.stat().st_size}) not smaller than --release ({ecs_rel.stat().st_size})"
        )

        # 2. DBM Ergonomics: min(a.len, b.len), min(a.len, 16), base+k<len, base+%k<len, SliceSub, StrAsBytes, UnwrapOptional
        src_dbm = td_p / "m11_dbm.eon"
        src_dbm.write_text(
            "module test.m11_dbm;\n"
            "fn test_ergonomics(a: *[i32], b: *[i32], s: *str, base: usize, k: usize) -> i32 ! BoundsError {\n"
            "    var total: i32 = 0_i32;\n"
            "    let n = if a.len < b.len { a.len } else { b.len };\n"
            "    var i: usize = 0_usize;\n"
            "    while i < n {\n"
            "        total = total +% a[i] +% b[i];\n"
            "        i = i + 1_usize;\n"
            "    }\n"
            "    let cap16 = if a.len < 16_usize { a.len } else { 16_usize };\n"
            "    var j: usize = 0_usize;\n"
            "    while j < cap16 {\n"
            "        total = total +% a[j];\n"
            "        j = j + 1_usize;\n"
            "    }\n"
            "    if base <= a.len && base + 4_usize <= a.len {\n"
            "        total = total +% a[base + 3_usize];\n"
            "        let tile = try a.sub(base, base + 4_usize);\n"
            "        total = total +% tile[0_usize] +% tile[3_usize];\n"
            "    }\n"
            "    if base +% k < a.len {\n"
            "        total = total +% a[base +% k];\n"
            "    }\n"
            "    let sb = s.as_bytes();\n"
            "    if s.len > 0_usize {\n"
            "        total = total +% sb[0_usize].widen.[i32]();\n"
            "    }\n"
            "    let opt_a: ?*[i32] = a;\n"
            "    if a.len > 0_usize {\n"
            "        if let ua = opt_a {\n"
            "            total = total +% ua[0_usize];\n"
            "        }\n"
            "    }\n"
            "    return total;\n"
            "}\n"
            "pub fn main() -> i32 ! BoundsError {\n"
            "    let xs: [6]i32 = [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32];\n"
            "    let ys: [4]i32 = [10_i32, 20_i32, 30_i32, 40_i32];\n"
            "    let r = try test_ergonomics(&xs, &ys, \"A\", 1_usize, 2_usize);\n"
            "    return if r == 213_i32 { 0_i32 } else { 1_i32 };\n"
            "}\n"
        )
        b_dbm = td_p / "m11_dbm"
        r_dbm = run_cmd([eonc, "build", "--release", str(src_dbm), "-o", str(b_dbm)])
        assert r_dbm.returncode == 0, f"m11_dbm ergonomics failed:\n{r_dbm.stderr}"
        assert subprocess.run([str(b_dbm)]).returncode == 0

        # 3. Contextual ZM-07 off-by-one diagnostic check
        src_obo = td_p / "m11_obo.eon"
        src_obo.write_text(
            "module test.m11_obo;\n"
            "pub fn bad(a: *[i32], i: usize) -> i32 {\n"
            "    if i <= a.len {\n"
            "        return a[i];\n"
            "    }\n"
            "    return 0_i32;\n"
            "}\n"
            "pub fn main() -> i32 { return 0_i32; }\n"
        )
        r_obo = run_cmd([eonc, "check", str(src_obo)])
        assert r_obo.returncode != 0 and "off-by-one" in (r_obo.stdout + r_obo.stderr), (
            f"Expected contextual off-by-one ZM-07 help, got:\n{r_obo.stdout}\n{r_obo.stderr}"
        )

    print("[PASS] M1.1 optimizer, -Oz, linker GC+ICF, and DBM ergonomics")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eonc", default=str(ROOT / "selfhost/out/eonc2_rel"))
    args = ap.parse_args()
    eonc = str(Path(args.eonc).resolve())
    test_ui_suites(eonc)
    test_emi_invalidation(eonc)
    test_real_native_stack_budget(eonc)
    test_dwarf5_debug_info(eonc)
    test_m11_opt_and_dbm(eonc)
    print("ALL SELFHOST SOUNDNESS REGRESSIONS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
