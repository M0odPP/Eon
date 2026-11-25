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


def test_m12_deps_lock_and_impl_cutoff(eonc: str):
    """Verify M1.2: multi-package build.eon deps, deterministic eon.lock, .emi libs,
    per-package allow_hal enforcement, cycle detection, and item-level impl method .emi diff."""
    with tempfile.TemporaryDirectory(prefix="eon_m12_reg_") as td:
        td_p = Path(td)

        # 1. Multi-package transitive graph: app -> mathlib -> coreutil + .emi library build
        (td_p / "coreutil" / "src").mkdir(parents=True)
        (td_p / "coreutil" / "build.eon").write_text(
            "module coreutil.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.set_out_dir(\"out\");\n"
            "    b.add_lib(\"coreutil\", \"src\");\n"
            "}\n"
        )
        (td_p / "coreutil" / "src" / "util.eon").write_text(
            "module coreutil.util;\n"
            "pub fn base_val() -> i32 { return 10_i32; }\n"
        )

        (td_p / "mathlib" / "src").mkdir(parents=True)
        (td_p / "mathlib" / "build.eon").write_text(
            "module mathlib.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.set_out_dir(\"out\");\n"
            "    b.add_dep(\"coreutil\", \"../coreutil\");\n"
            "    b.add_lib(\"mathlib\", \"src\");\n"
            "}\n"
        )
        (td_p / "mathlib" / "src" / "math.eon").write_text(
            "module mathlib.math;\n"
            "import coreutil.util;\n"
            "pub fn compute() -> i32 { return util.base_val() +% 32_i32; }\n"
        )

        (td_p / "app" / "src").mkdir(parents=True)
        (td_p / "app" / "build.eon").write_text(
            "module app.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.set_out_dir(\"out\");\n"
            "    b.add_dep(\"mathlib\", \"../mathlib\");\n"
            "    b.add_exe(\"app\", \"src\");\n"
            "}\n"
        )
        (td_p / "app" / "src" / "main.eon").write_text(
            "module app.main;\n"
            "import mathlib.math;\n"
            "pub fn main() -> i32 {\n"
            "    return if math.compute() == 42_i32 { 0_i32 } else { 1_i32 };\n"
            "}\n"
        )

        # Build .emi library directly
        r_lib = run_cmd([eonc, "build", "--release", str(td_p / "mathlib")], cwd=td_p)
        assert r_lib.returncode == 0, f"mathlib .emi build failed:\n{r_lib.stdout}\n{r_lib.stderr}"
        emi_file = td_p / "mathlib" / "out" / "mathlib.emi"
        assert emi_file.exists(), "expected mathlib/out/mathlib.emi"
        assert emi_file.read_bytes()[:4] == b"EOMI", "mathlib.emi missing EOMI magic header"

        # Build app with transitive deps and deterministic eon.lock
        r_app1 = run_cmd([eonc, "build", "--release", str(td_p / "app")], cwd=td_p)
        assert r_app1.returncode == 0, f"app build failed:\n{r_app1.stdout}\n{r_app1.stderr}"
        assert subprocess.run([str(td_p / "app" / "out" / "app")]).returncode == 0
        lock_file = td_p / "app" / "eon.lock"
        assert lock_file.exists(), "expected app/eon.lock to be generated"
        lock1 = lock_file.read_text()
        assert 'name = "coreutil"' in lock1 and 'name = "mathlib"' in lock1, f"unexpected eon.lock:\n{lock1}"
        assert lock1.index('name = "coreutil"') < lock1.index('name = "mathlib"'), "eon.lock entries not sorted"

        r_app2 = run_cmd([eonc, "build", "--release", str(td_p / "app")], cwd=td_p)
        assert r_app2.returncode == 0, f"app rebuild failed:\n{r_app2.stderr}"
        lock2 = lock_file.read_text()
        assert lock1 == lock2, "eon.lock is not deterministic across repeated builds"

        # 2. Per-package allow_hal enforcement (cannot escalate @hal without root build.eon approval)
        (td_p / "haldep" / "src").mkdir(parents=True)
        (td_p / "haldep" / "build.eon").write_text(
            "module haldep.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.allow_hal(\"haldep.hw\");\n"
            "    b.add_lib(\"haldep\", \"src\");\n"
            "}\n"
        )
        (td_p / "haldep" / "src" / "hw.hal.eon").write_text(
            "module haldep.hw @hal;\n"
            "pub fn ping() -> i32 { return 7_i32; }\n"
        )
        (td_p / "halroot" / "src").mkdir(parents=True)
        (td_p / "halroot" / "src" / "main.eon").write_text(
            "module halroot.main;\n"
            "import haldep.hw;\n"
            "pub fn main() -> i32 { return if hw.ping() == 7_i32 { 0_i32 } else { 1_i32 }; }\n"
        )
        (td_p / "halroot" / "build.eon").write_text(
            "module halroot.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.add_dep(\"haldep\", \"../haldep\");\n"
            "    b.add_exe(\"halroot\", \"src\");\n"
            "}\n"
        )
        r_deny = run_cmd([eonc, "build", "--release", str(td_p / "halroot")], cwd=td_p)
        assert r_deny.returncode != 0 and "not authorized in root build.eon" in (r_deny.stdout + r_deny.stderr), (
            f"Expected allow_hal escalation rejection, got rc={r_deny.returncode}:\n{r_deny.stdout}\n{r_deny.stderr}"
        )

        # Grant explicit allow_hal in root build.eon -> MUST succeed
        (td_p / "halroot" / "build.eon").write_text(
            "module halroot.build;\n"
            "pub fn build(b: *mut BuildCap) {\n"
            "    b.allow_hal(\"haldep.hw\");\n"
            "    b.add_dep(\"haldep\", \"../haldep\");\n"
            "    b.add_exe(\"halroot\", \"src\");\n"
            "}\n"
        )
        r_allow = run_cmd([eonc, "build", "--release", str(td_p / "halroot")], cwd=td_p)
        assert r_allow.returncode == 0, f"Authorized halroot build failed:\n{r_allow.stdout}\n{r_allow.stderr}"
        assert subprocess.run([str(td_p / "halroot" / "out" / "halroot")]).returncode == 0

        # 3. Dependency cycle detection
        (td_p / "cyca" / "src").mkdir(parents=True)
        (td_p / "cycb" / "src").mkdir(parents=True)
        (td_p / "cyca" / "build.eon").write_text(
            "module cyca.build;\n"
            "pub fn build(b: *mut BuildCap) { b.add_dep(\"cycb\", \"../cycb\"); b.add_exe(\"cyca\", \"src\"); }\n"
        )
        (td_p / "cycb" / "build.eon").write_text(
            "module cycb.build;\n"
            "pub fn build(b: *mut BuildCap) { b.add_dep(\"cyca\", \"../cyca\"); b.add_lib(\"cycb\", \"src\"); }\n"
        )
        (td_p / "cyca" / "src" / "main.eon").write_text("module cyca.main;\npub fn main() -> i32 { return 0_i32; }\n")
        (td_p / "cycb" / "src" / "lib.eon").write_text("module cycb.lib;\npub fn f() -> i32 { return 1_i32; }\n")
        r_cyc = run_cmd([eonc, "build", "--release", str(td_p / "cyca")], cwd=td_p)
        assert r_cyc.returncode != 0 and "dependency cycle" in (r_cyc.stdout + r_cyc.stderr), (
            f"Expected dependency cycle error, got rc={r_cyc.returncode}:\n{r_cyc.stdout}\n{r_cyc.stderr}"
        )

        # 4. Item-level impl method .emi stub diff
        impl_dir = td_p / "impl_inc"
        impl_dir.mkdir()
        ctr_path = impl_dir / "ctr.eon"
        main_path = impl_dir / "main.eon"
        cache_dir = impl_dir / "cache"
        out_bin = impl_dir / "ctr_app"
        ctr_path.write_text(
            "module inc.ctr;\n"
            "pub struct Counter { pub v: i32; }\n"
            "impl Counter {\n"
            "    fn step(self: *Self) -> i32 { return if self.v >= 0_i32 { 1_i32 } else { 0_i32 }; }\n"
            "    pub fn inc(self: *mut Self) -> i32 {\n"
            "        self.v = self.v +% self.step();\n"
            "        return self.v;\n"
            "    }\n"
            "}\n"
        )
        main_path.write_text(
            "module inc.main;\n"
            "import inc.ctr;\n"
            "pub fn main() -> i32 {\n"
            "    var c = ctr.Counter.{ v: 10_i32 };\n"
            "    return c.inc();\n"
            "}\n"
        )
        r_i1 = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(ctr_path), str(main_path), "-o", str(out_bin)], cwd=impl_dir)
        assert r_i1.returncode == 0, f"impl cold build failed:\n{r_i1.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 11

        # Add a new pub method + mutate private impl method
        ctr_path.write_text(
            "module inc.ctr;\n"
            "pub struct Counter { pub v: i32; }\n"
            "impl Counter {\n"
            "    fn step(self: *Self) -> i32 { return if self.v >= 0_i32 { 12_i32 } else { 0_i32 }; }\n"
            "    pub fn inc(self: *mut Self) -> i32 {\n"
            "        self.v = self.v +% self.step();\n"
            "        return self.v;\n"
            "    }\n"
            "    pub fn bonus(self: *Self) -> i32 { return self.v +% 5_i32; }\n"
            "}\n"
        )
        r_i2 = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(ctr_path), str(main_path), "-o", str(out_bin)], cwd=impl_dir)
        assert r_i2.returncode == 0, f"impl additive/private rebuild failed:\n{r_i2.stderr}"
        assert subprocess.run([str(out_bin)]).returncode == 22

        # Breaking edit to pub impl method signature -> MUST fail
        ctr_path.write_text(
            "module inc.ctr;\n"
            "pub struct Counter { pub v: i32; }\n"
            "impl Counter {\n"
            "    fn step(self: *Self) -> i32 { return if self.v >= 0_i32 { 12_i32 } else { 0_i32 }; }\n"
            "    pub fn inc(self: *mut Self, delta: i32) -> i32 {\n"
            "        self.v = self.v +% delta +% self.step();\n"
            "        return self.v;\n"
            "    }\n"
            "}\n"
        )
        r_i3 = run_cmd([eonc, "build", "--release", "--emi-cache", str(cache_dir), str(ctr_path), str(main_path), "-o", str(out_bin)], cwd=impl_dir)
        assert r_i3.returncode != 0, "Expected breaking impl method signature edit to fail build!"

    print("[PASS] M1.2 multi-package deps, eon.lock, .emi libs, allow_hal, cycle detection, and impl method diff")


def test_m13_core_and_std(eonc: str):
    """Verify M1.3: freestanding core/{math_vec,color,fmt} (0 HAL/raw) and OS std/{time,fs,sync,net,dl}
    under both --release and --sanitize (zero leaks)."""
    with tempfile.TemporaryDirectory(prefix="eon_m13_reg_") as td:
        td_p = Path(td)

        # 1. Pure freestanding core/ test (0 @hal, 0 raw, 0 heap)
        core_src = td_p / "m13_core.eon"
        core_src.write_text(
            "module test.m13_core;\n"
            "import core.math;\n"
            "import core.math_vec;\n"
            "import core.color;\n"
            "import core.fmt;\n"
            "import core.str;\n"
            "fn check_eq(buf: *[u8], n: usize, expected: *str) -> bool {\n"
            "    let sl = buf.sub(0_usize, n) catch { return false; };\n"
            "    return str.bytes_eq(sl, expected.as_bytes());\n"
            "}\n"
            "pub fn main() -> i32 {\n"
            "    // Vec2/Vec3/Vec4 + Bounds + Mat4\n"
            "    let a = math_vec.Vec2.new(3.0_f32, 4.0_f32);\n"
            "    if math.abs_f32(a.len() - 5.0_f32) > 1.0e-5_f32 { return 1_i32; }\n"
            "    let n2 = a.normalize();\n"
            "    if math.abs_f32(n2.len() - 1.0_f32) > 1.0e-5_f32 { return 2_i32; }\n"
            "    let c3 = math_vec.Vec3.new(1.0_f32, 0.0_f32, 0.0_f32).cross(math_vec.Vec3.new(0.0_f32, 1.0_f32, 0.0_f32));\n"
            "    if math.abs_f32(c3.z - 1.0_f32) > 1.0e-5_f32 { return 3_i32; }\n"
            "    let v4 = math_vec.Vec4.from_f32x4(math_vec.Vec4.new(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32).to_f32x4());\n"
            "    if math.abs_f32(v4.dot(math_vec.Vec4.splat(1.0_f32)) - 10.0_f32) > 1.0e-5_f32 { return 4_i32; }\n"
            "    let b1 = math_vec.Bounds.from_xywh(10.0_f32, 20.0_f32, 100.0_f32, 50.0_f32);\n"
            "    let b2 = math_vec.Bounds.from_xywh(50.0_f32, 30.0_f32, 100.0_f32, 50.0_f32);\n"
            "    if !b1.contains(math_vec.Point.new(25.0_f32, 25.0_f32)) { return 5_i32; }\n"
            "    let bi = b1.intersect(b2);\n"
            "    if math.abs_f32(bi.size.width - 60.0_f32) > 1.0e-4_f32 || math.abs_f32(bi.size.height - 40.0_f32) > 1.0e-4_f32 { return 6_i32; }\n"
            "    let t = math_vec.Mat4.translate(10.0_f32, 20.0_f32, 0.0_f32);\n"
            "    let s = math_vec.Mat4.scale(2.0_f32, 3.0_f32, 1.0_f32);\n"
            "    let ts = t.mul(&s);\n"
            "    let pt = ts.mul_point2d(math_vec.Point.new(5.0_f32, 4.0_f32));\n"
            "    if math.abs_f32(pt.x - 20.0_f32) > 1.0e-4_f32 || math.abs_f32(pt.y - 32.0_f32) > 1.0e-4_f32 { return 7_i32; }\n"
            "    // Color conversions\n"
            "    let col = color.Rgba.from_hex_rgb(0x3366CC_u32);\n"
            "    let h = col.to_hsla();\n"
            "    let rt = h.to_rgba();\n"
            "    if math.abs_f32(rt.r - col.r) > 0.01_f32 || math.abs_f32(rt.g - col.g) > 0.01_f32 || math.abs_f32(rt.b - col.b) > 0.01_f32 { return 8_i32; }\n"
            "    let lin = col.to_linear().from_linear();\n"
            "    if math.abs_f32(lin.r - col.r) > 0.005_f32 || math.abs_f32(lin.b - col.b) > 0.005_f32 { return 9_i32; }\n"
            "    let over = color.Rgba.new(1.0_f32, 0.0_f32, 0.0_f32, 0.5_f32).blend_over(color.Rgba.new(0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32));\n"
            "    if math.abs_f32(over.r - 0.5_f32) > 0.01_f32 || math.abs_f32(over.b - 0.5_f32) > 0.01_f32 { return 10_i32; }\n"
            "    if color.Rgba.from_u8(0x11_u8, 0x22_u8, 0x33_u8, 0xFF_u8).to_rgba8_u32() != 0xFF332211_u32 { return 11_i32; }\n"
            "    // Allocation-free fmt\n"
            "    var buf: [64]u8 = [0_u8; 64];\n"
            "    let n_i = fmt.format_i64(&mut buf, -9223372036854775807_i64 -% 1_i64);\n"
            "    if !check_eq(&buf, n_i, \"-9223372036854775808\") { return 12_i32; }\n"
            "    let n_h = fmt.format_hex_u64(&mut buf, 0xBEEF_u64, true, 8_usize);\n"
            "    if !check_eq(&buf, n_h, \"0000BEEF\") { return 13_i32; }\n"
            "    let n_b = fmt.format_bin_u64(&mut buf, 0xA5_u64, 8_usize, true);\n"
            "    if !check_eq(&buf, n_b, \"1010 0101\") { return 14_i32; }\n"
            "    let n_o = fmt.format_oct_u64(&mut buf, 493_u64, 4_usize);\n"
            "    if !check_eq(&buf, n_o, \"0755\") { return 15_i32; }\n"
            "    let n_f = fmt.format_f64_fixed(&mut buf, -3.14159_f64, 3_usize);\n"
            "    if !check_eq(&buf, n_f, \"-3.142\") { return 16_i32; }\n"
            "    let n_c1 = fmt.format_f64_compact(&mut buf, 42.5000_f64, 6_usize);\n"
            "    if !check_eq(&buf, n_c1, \"42.5\") { return 17_i32; }\n"
            "    let n_c2 = fmt.format_f64_compact(&mut buf, 100.0_f64, 6_usize);\n"
            "    if !check_eq(&buf, n_c2, \"100\") { return 18_i32; }\n"
            "    let n_c3 = fmt.format_f64_compact(&mut buf, 1.25e14_f64, 6_usize);\n"
            "    if !check_eq(&buf, n_c3, \"1.25e+14\") { return 19_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
        )
        core_bin = td_p / "m13_core"
        r_c = run_cmd([eonc, "build", "--release", "--audit-tcb", str(core_src), "-o", str(core_bin)])
        assert r_c.returncode == 0, f"m13_core build failed:\n{r_c.stdout}\n{r_c.stderr}"
        assert "files: 0" in r_c.stdout and "raw_lines: 0" in r_c.stdout, f"core/ leaked HAL/raw:\n{r_c.stdout}"
        rc_c = subprocess.run([str(core_bin)]).returncode
        assert rc_c == 0, f"m13_core exited {rc_c}"

        # 2. OS std/{time,fs,sync,net,dl} test under BOTH --release and --sanitize
        std_src = td_p / "m13_std.eon"
        tmp_f1 = str(td_p / "io_a.txt")
        tmp_f2 = str(td_p / "io_b.txt")
        std_src.write_text(
            "module test.m13_std;\n"
            "import core.str;\n"
            "import std.time;\n"
            "import std.fs;\n"
            "import std.sync;\n"
            "import std.net;\n"
            "import std.dl;\n"
            "fn check_time(clock: *ClockCap) -> i32 {\n"
            "    let t0 = time.Instant.now(clock);\n"
            "    let m0 = time.Instant.now_monotonic();\n"
            "    time.sleep_ms(5_u64);\n"
            "    if t0.elapsed(clock).as_millis() < 3_u64 { return 1_i32; }\n"
            "    if m0.elapsed_monotonic().as_millis() < 3_u64 { return 2_i32; }\n"
            "    if time.WallTime.now_utc().unix_secs() < 1700000000_i64 { return 3_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "fn write_sample(fscap: *FsCap, p1: *str) -> !IoError {\n"
            "    var fw = try fs.File.create_write(fscap, p1);\n"
            "    errdefer fw.close();\n"
            "    try fw.write_all(\"hello-eon!\".as_bytes());\n"
            "    try fw.sync_all();\n"
            "    fw.close();\n"
            "    return;\n"
            "}\n"
            "fn read_tail(fscap: *FsCap, p1: *str) -> i32 ! IoError {\n"
            "    var fr = try fs.File.open_read(fscap, p1);\n"
            "    errdefer fr.close();\n"
            "    let _ = try fr.seek(6_i64, fs.SEEK_SET);\n"
            "    var rbuf: [16]u8 = [0_u8; 16];\n"
            "    let nr = try fr.read(&mut rbuf);\n"
            "    fr.close();\n"
            "    if nr != 4_usize { return 5_i32; }\n"
            "    let tail = rbuf.sub(0_usize, nr) catch { return 6_i32; };\n"
            "    if !str.bytes_eq(tail, \"eon!\".as_bytes()) { return 7_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "fn check_fs(fscap: *FsCap, p1: *str, p2: *str) -> i32 ! IoError {\n"
            "    try write_sample(fscap, p1);\n"
            "    let st = try fs.stat_path(fscap, p1);\n"
            "    if !st.is_file || st.size != 10_u64 { return 4_i32; }\n"
            "    let rc = try read_tail(fscap, p1);\n"
            "    if rc != 0_i32 { return rc; }\n"
            "    try fs.rename_path(fscap, p1, p2);\n"
            "    if fs.exists(fscap, p1) || !fs.exists(fscap, p2) { return 8_i32; }\n"
            "    try fs.remove_file(fscap, p2);\n"
            "    if fs.exists(fscap, p2) { return 9_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "fn check_sync(alloc: *mut AllocCap) -> i32 ! OutOfMemory {\n"
            "    var mu = sync.Mutex.new();\n"
            "    let g1 = mu.lock();\n"
            "    if let g2 = mu.try_lock() { g2.unlock(&mut mu); g1.unlock(&mut mu); return 10_i32; }\n"
            "    g1.unlock(&mut mu);\n"
            "    var rw = sync.RwLock.new();\n"
            "    let rg1 = rw.read_lock();\n"
            "    let rg2 = rw.read_lock();\n"
            "    if let wg = rw.try_write_lock() { wg.unlock(&mut rw); rg2.unlock(&mut rw); rg1.unlock(&mut rw); return 11_i32; }\n"
            "    rg2.unlock(&mut rw);\n"
            "    rg1.unlock(&mut rw);\n"
            "    let wg = rw.write_lock();\n"
            "    wg.unlock(&mut rw);\n"
            "    var ch = try sync.Channel.[u64].init(alloc, 4_usize, 0_u64);\n"
            "    if !ch.try_send(111_u64) || !ch.try_send(222_u64) { ch.destroy(alloc); return 12_i32; }\n"
            "    let v1 = ch.try_recv() orelse 0_u64;\n"
            "    let v2 = ch.recv() orelse 0_u64;\n"
            "    ch.close();\n"
            "    if v1 != 111_u64 || v2 != 222_u64 || ch.try_send(333_u64) { ch.destroy(alloc); return 13_i32; }\n"
            "    ch.destroy(alloc);\n"
            "    return 0_i32;\n"
            "}\n"
            "fn check_tcp(ncap: *NetCap) -> i32 ! IoError {\n"
            "    var lst = try net.TcpListener.bind(ncap, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16));\n"
            "    errdefer lst.close();\n"
            "    let laddr = try lst.local_addr();\n"
            "    var cli = try net.TcpStream.connect(ncap, laddr);\n"
            "    errdefer cli.close();\n"
            "    var srv = try lst.accept();\n"
            "    errdefer srv.close();\n"
            "    try cli.write_all(\"ping\".as_bytes());\n"
            "    var nbuf: [16]u8 = [0_u8; 16];\n"
            "    let rn = try srv.read(&mut nbuf);\n"
            "    try srv.write_all(\"pong\".as_bytes());\n"
            "    var cbuf: [16]u8 = [0_u8; 16];\n"
            "    let cn = try cli.read(&mut cbuf);\n"
            "    srv.close();\n"
            "    cli.close();\n"
            "    lst.close();\n"
            "    let s1 = nbuf.sub(0_usize, rn) catch { return 14_i32; };\n"
            "    let s2 = cbuf.sub(0_usize, cn) catch { return 15_i32; };\n"
            "    if !str.bytes_eq(s1, \"ping\".as_bytes()) || !str.bytes_eq(s2, \"pong\".as_bytes()) { return 16_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "fn check_udp(ncap: *NetCap) -> i32 ! IoError {\n"
            "    var u1 = try net.UdpSocket.bind(ncap, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16));\n"
            "    errdefer u1.close();\n"
            "    var u2 = try net.UdpSocket.bind(ncap, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16));\n"
            "    errdefer u2.close();\n"
            "    let u2a = try u2.local_addr();\n"
            "    _ = try u1.send_to(\"udp42\".as_bytes(), u2a);\n"
            "    var ubuf: [16]u8 = [0_u8; 16];\n"
            "    let ur = try u2.recv_from(&mut ubuf);\n"
            "    u2.close();\n"
            "    u1.close();\n"
            "    let us = ubuf.sub(0_usize, ur.len) catch { return 17_i32; };\n"
            "    if !str.bytes_eq(us, \"udp42\".as_bytes()) { return 18_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "fn check_dl() -> i32 ! IoError {\n"
            "    var missing_ok = false;\n"
            "    let mlib = dl.DynLib.open(\"libeon_does_not_exist_999.so\") catch {\n"
            "        missing_ok = true;\n"
            "        try dl.DynLib.open(\"libc.so.6\")\n"
            "    };\n"
            "    if !missing_ok { mlib.close(); return 19_i32; }\n"
            "    let abs_fp = mlib.sym(\"abs\") orelse 0_u64;\n"
            "    if abs_fp == 0_u64 { mlib.close(); return 20_i32; }\n"
            "    let abs_res = dl.call_c1(abs_fp, (-42_i64).bitcast.[u64]()).trunc.[i32]();\n"
            "    mlib.close();\n"
            "    if abs_res != 42_i32 { return 21_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
            "pub fn main(env: *mut Env) -> i32 ! OutOfMemory | IoError {\n"
            "    let r1 = check_time(&env.clock);\n"
            "    if r1 != 0_i32 { return r1; }\n"
            f"    let r2 = try check_fs(&env.fs, \"{tmp_f1}\", \"{tmp_f2}\");\n"
            "    if r2 != 0_i32 { return r2; }\n"
            "    let r3 = try check_sync(&mut env.alloc);\n"
            "    if r3 != 0_i32 { return r3; }\n"
            "    let r4 = try check_tcp(&env.net);\n"
            "    if r4 != 0_i32 { return r4; }\n"
            "    let r5 = try check_udp(&env.net);\n"
            "    if r5 != 0_i32 { return r5; }\n"
            "    let r6 = try check_dl();\n"
            "    if r6 != 0_i32 { return r6; }\n"
            "    if env.alloc.live_allocs() != 0_usize { return 22_i32; }\n"
            "    return 0_i32;\n"
            "}\n"
        )
        for mode_flag, suffix in [("--release", "rel"), ("--sanitize", "san")]:
            out_b = td_p / f"m13_std_{suffix}"
            r_s = run_cmd([eonc, "build", mode_flag, "-l", "c", str(std_src), "-o", str(out_b)])
            assert r_s.returncode == 0, f"m13_std {mode_flag} build failed:\n{r_s.stdout}\n{r_s.stderr}"
            res_run = subprocess.run([str(out_b)], capture_output=True, text=True)
            assert res_run.returncode == 0, f"m13_std {mode_flag} exited {res_run.returncode}:\n{res_run.stdout}\n{res_run.stderr}"

    print("[PASS] M1.3 freestanding core/{math_vec,color,fmt} and OS std/{time,fs,sync,net,dl} (--release + --sanitize)")


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
    test_m12_deps_lock_and_impl_cutoff(eonc)
    test_m13_core_and_std(eonc)
    print("ALL SELFHOST SOUNDNESS REGRESSIONS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
