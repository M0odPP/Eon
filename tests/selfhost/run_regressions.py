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


def _lsp_send(proc: subprocess.Popen, obj: dict):
    import json
    body = json.dumps(obj).encode("utf-8")
    hdr = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    proc.stdin.write(hdr + body)
    proc.stdin.flush()


def _lsp_recv(proc: subprocess.Popen) -> dict:
    import json
    content_len = None
    while True:
        line = proc.stdout.readline()
        assert line, "LSP server closed stdout unexpectedly"
        s = line.decode("ascii", errors="replace").strip()
        if s == "":
            if content_len is not None:
                break
            continue
        if s.lower().startswith("content-length:"):
            content_len = int(s.split(":", 1)[1].strip())
    data = proc.stdout.read(content_len)
    assert len(data) == content_len, f"Short LSP read: {len(data)} != {content_len}"
    return json.loads(data.decode("utf-8"))


def test_m14_fmt_and_lsp(eonc: str):
    """Verify M1.4: canonical idempotent `eonc fmt [--check]` and full `eonc lsp` JSON-RPC lifecycle
    (diagnostics, hover with DBM proven intervals & @stack(N), definition, completion, formatting, quick-fixes, 0 leaks)."""
    with tempfile.TemporaryDirectory(prefix="eon_m14_reg_") as td:
        td_p = Path(td)

        # 1. `eonc fmt` & `eonc fmt --check` idempotency + comment preservation + valid code
        messy = td_p / "messy.eon"
        messy.write_text(
            "module demo.fmt;\n"
            "/// Doc comment preserved\n"
            "pub fn add_pair( a:i32,b:i32 )->i32{\n"
            "\n\n\n"
            "  // line comment preserved\n"
            "  let s=a+%b;\n"
            "  /* block comment preserved */\n"
            "  return s;\n"
            "}\n"
            "pub fn main()->i32{\n"
            "    let r=add_pair(20_i32,22_i32);\n"
            "    if r==42_i32{return 0_i32;}\n"
            "    return 1_i32;\n"
            "}\n"
        )
        r_chk1 = run_cmd([eonc, "fmt", "--check", str(messy)])
        assert r_chk1.returncode != 0 and "would reformat" in r_chk1.stderr, (
            f"Expected fmt --check to flag messy.eon: rc={r_chk1.returncode}\n{r_chk1.stderr}"
        )
        r_fmt = run_cmd([eonc, "fmt", str(messy)])
        assert r_fmt.returncode == 0, f"eonc fmt failed:\n{r_fmt.stderr}"
        fmt_txt = messy.read_text()
        for keep in ("/// Doc comment preserved", "// line comment preserved", "/* block comment preserved */"):
            assert keep in fmt_txt, f"Formatter lost comment `{keep}`:\n{fmt_txt}"
        assert "let s = a +% b;" in fmt_txt, f"Expected canonical spacing in:\n{fmt_txt}"
        # Second pass must be 100% idempotent
        r_chk2 = run_cmd([eonc, "fmt", "--check", str(messy)])
        assert r_chk2.returncode == 0, f"eonc fmt is not idempotent:\n{r_chk2.stderr}\n{fmt_txt}"
        # Formatted file must still compile and run
        fmt_bin = td_p / "messy_bin"
        r_b = run_cmd([eonc, "build", "--release", str(messy), "-o", str(fmt_bin)])
        assert r_b.returncode == 0, f"Formatted file failed to build:\n{r_b.stderr}\n{fmt_txt}"
        assert subprocess.run([str(fmt_bin)]).returncode == 0

        # Real-world idempotency check on copies of core/{math_vec,color,fmt}.eon
        core_copy = td_p / "core_fmt_test"
        core_copy.mkdir()
        for nm in ("math_vec.eon", "color.eon", "fmt.eon"):
            shutil.copy(ROOT / "core" / nm, core_copy / nm)
        assert run_cmd([eonc, "fmt", str(core_copy)]).returncode == 0
        r_core_chk = run_cmd([eonc, "fmt", "--check", str(core_copy)])
        assert r_core_chk.returncode == 0, f"core/ fmt not idempotent:\n{r_core_chk.stderr}"

        # 2. Live `eonc lsp` JSON-RPC 2.0 session
        lsp_file = td_p / "sample.eon"
        lsp_uri = lsp_file.as_uri()
        sample_src = (
            "module demo.lsp;\n"
            "\n"
            "pub fn clamp_inc(x: usize) -> usize {\n"
            "    if x < 10_usize {\n"
            "        let y = x + 1_usize;\n"
            "        return y;\n"
            "    }\n"
            "    return 10_usize;\n"
            "}\n"
            "\n"
            "pub fn main() -> i32 {\n"
            "    let r = clamp_inc(4_usize);\n"
            "    return if r == 5_usize { 0_i32 } else { 1_i32 };\n"
            "}\n"
        )
        lsp_file.write_text(sample_src)

        proc = subprocess.Popen(
            [eonc, "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # initialize
            _lsp_send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}})
            init_resp = _lsp_recv(proc)
            assert init_resp["id"] == 1
            caps = init_resp["result"]["capabilities"]
            for cap_k in ("hoverProvider", "definitionProvider", "completionProvider", "documentFormattingProvider", "codeActionProvider"):
                assert caps.get(cap_k), f"Missing LSP capability {cap_k}: {caps}"

            _lsp_send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

            # didOpen -> clean publishDiagnostics
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {"textDocument": {"uri": lsp_uri, "languageId": "eon", "version": 1, "text": sample_src}},
            })
            diag1 = _lsp_recv(proc)
            assert diag1["method"] == "textDocument/publishDiagnostics"
            assert diag1["params"]["diagnostics"] == [], f"Expected 0 diagnostics on valid file, got {diag1}"

            # hover on function `clamp_inc` (line 2, col 8) -> must show @stack(...) & DBM checks
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/hover",
                "params": {"textDocument": {"uri": lsp_uri}, "position": {"line": 2, "character": 8}},
            })
            hov_fn = _lsp_recv(proc)
            hv_fn = hov_fn["result"]["contents"]["value"]
            assert "@stack(" in hv_fn and "DBM checks:" in hv_fn and "clamp_inc" in hv_fn, f"Unexpected fn hover:\n{hv_fn}"

            # hover on local `y` (line 4, col 12) -> must show type AND proven DBM interval [1..=10]
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "textDocument/hover",
                "params": {"textDocument": {"uri": lsp_uri}, "position": {"line": 4, "character": 12}},
            })
            hov_var = _lsp_recv(proc)
            hv_var = hov_var["result"]["contents"]["value"]
            assert "let y: usize" in hv_var and "DBM proven interval:" in hv_var and "[1..=10]" in hv_var, (
                f"Expected DBM proven interval [1..=10] in var hover:\n{hv_var}"
            )

            # definition on call `clamp_inc` in main (line 11, col 15) -> points to line 2
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "textDocument/definition",
                "params": {"textDocument": {"uri": lsp_uri}, "position": {"line": 11, "character": 15}},
            })
            def_resp = _lsp_recv(proc)
            assert def_resp["result"]["range"]["start"]["line"] == 2, f"Unexpected definition: {def_resp}"

            # completion at `clamp_` (line 11, col 16) -> includes `clamp_inc`
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "textDocument/completion",
                "params": {"textDocument": {"uri": lsp_uri}, "position": {"line": 11, "character": 16}},
            })
            comp_resp = _lsp_recv(proc)
            labels = [it["label"] for it in comp_resp["result"]]
            assert "clamp_inc" in labels, f"Expected clamp_inc in completions: {labels}"

            # formatting
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "textDocument/formatting",
                "params": {"textDocument": {"uri": lsp_uri}, "options": {"tabSize": 4, "insertSpaces": True}},
            })
            fmt_resp = _lsp_recv(proc)
            assert len(fmt_resp["result"]) == 1 and "pub fn clamp_inc" in fmt_resp["result"][0]["newText"]

            # didChange with ZM-01 & ZM-02 errors -> publishDiagnostics + codeAction quick-fixes
            bad_src = (
                "module demo.lsp;\n"
                "pub fn helper() -> i32 { return 7_i32; }\n"
                "pub fn main() -> i32 {\n"
                "    let unused_v = 1_i32;\n"
                "    helper();\n"
                "    return 0_i32;\n"
                "}\n"
            )
            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": lsp_uri, "version": 2},
                    "contentChanges": [{"text": bad_src}],
                },
            })
            diag2 = _lsp_recv(proc)
            codes = {d["code"] for d in diag2["params"]["diagnostics"]}
            assert "E0101" in codes and "E0102" in codes, f"Expected E0101 & E0102 in diagnostics: {diag2}"

            _lsp_send(proc, {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "textDocument/codeAction",
                "params": {
                    "textDocument": {"uri": lsp_uri},
                    "range": {"start": {"line": 3, "character": 0}, "end": {"line": 5, "character": 0}},
                    "context": {"diagnostics": diag2["params"]["diagnostics"]},
                },
            })
            ca_resp = _lsp_recv(proc)
            titles = [a["title"] for a in ca_resp["result"]]
            assert any("ZM-01" in t for t in titles) and any("ZM-02" in t for t in titles), (
                f"Expected ZM-01 and ZM-02 quick-fixes: {ca_resp}"
            )

            # shutdown + exit -> rc=0 (proving zero heap leaks!)
            _lsp_send(proc, {"jsonrpc": "2.0", "id": 8, "method": "shutdown"})
            sd_resp = _lsp_recv(proc)
            assert sd_resp["id"] == 8 and sd_resp["result"] is None
            _lsp_send(proc, {"jsonrpc": "2.0", "method": "exit"})
            rc_lsp = proc.wait(timeout=10)
            err_lsp = proc.stderr.read().decode("utf-8", errors="replace")
            assert rc_lsp == 0, f"eonc lsp exited {rc_lsp} (possible leak!):\n{err_lsp}"
        finally:
            if proc.poll() is None:
                proc.kill()

    print("[PASS] M1.4 eonc fmt [--check] (idempotent, comment-safe) and eonc lsp (diagnostics, DBM+@stack hover, definition, completion, formatting, quick-fixes, 0 leaks)")


def test_m21_target_spec_and_comptime(eonc: str) -> None:
    print("[TEST] M2.1 TargetSpec (--target=<arch>-<os>), @target_arch/@target_os/@target_is_nostd, dead comptime branch pruning...")
    with tempfile.TemporaryDirectory(prefix="eon_m21_") as td:
        p = Path(td)
        src = p / "target_matrix.eon"
        src.write_text(
            "module demo.target_matrix;\n"
            "\n"
            "fn arch_code() -> i32 {\n"
            "    comptime if @target_arch() == .X86_64 {\n"
            "        return 10_i32;\n"
            "    } else {\n"
            "        comptime if @target_arch() == .AArch64 {\n"
            "            return 20_i32;\n"
            "        } else {\n"
            "            comptime if @target_arch() == .RiscV64 {\n"
            "                return 30_i32;\n"
            "            } else {\n"
            "                @compile_error(\"unsupported arch\");\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "fn os_code() -> i32 {\n"
            "    comptime if @target_os() == .Linux {\n"
            "        return 1_i32;\n"
            "    } else {\n"
            "        comptime if @target_os() == .Windows {\n"
            "            return 2_i32;\n"
            "        } else {\n"
            "            comptime if @target_os() == .MacOS {\n"
            "                return 3_i32;\n"
            "            } else {\n"
            "                comptime if @target_os() == .Android {\n"
            "                    return 4_i32;\n"
            "                } else {\n"
            "                    comptime if @target_os() == .IOS {\n"
            "                        return 5_i32;\n"
            "                    } else {\n"
            "                        comptime if @target_os() == .NoStd {\n"
            "                            return 6_i32;\n"
            "                        } else {\n"
            "                            return nonexistent_symbol_pruned();\n"
            "                        }\n"
            "                    }\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
            "fn nostd_bonus(b: bool) -> i32 {\n"
            "    return if b { 100_i32 } else { 0_i32 };\n"
            "}\n"
            "\n"
            "pub fn main() -> i32 {\n"
            "    let a = arch_code();\n"
            "    let o = os_code();\n"
            "    let n: i32 = comptime if @target_is_nostd() { 100_i32 } else { 0_i32 };\n"
            "    let rt_n = nostd_bonus(@target_is_nostd());\n"
            "    if n != rt_n {\n"
            "        return 255_i32;\n"
            "    }\n"
            "    return a +% o +% n;\n"
            "}\n",
            encoding="utf-8",
        )

        # Verify dead-branch pruning with an explicit unresolved symbol in the non-taken arm
        prune_src = p / "prune.eon"
        prune_src.write_text(
            "module demo.prune;\n"
            "pub fn main() -> i32 {\n"
            "    comptime if @target_os() == .Windows {\n"
            "        return 7_i32;\n"
            "    } else {\n"
            "        return missing_windows_only_api();\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        cp_prune_ok = run_cmd([eonc, "check", "--no-stdlib", "--target=x86_64-windows", str(prune_src)])
        assert cp_prune_ok.returncode == 0, f"dead branch was not pruned:\n{cp_prune_ok.stderr}"
        cp_prune_err = run_cmd([eonc, "check", "--no-stdlib", "--target=x86_64-linux", str(prune_src)])
        assert cp_prune_err.returncode != 0, "expected unresolved symbol error on non-Windows target"

        # Verify all 10 triples through both `check` and `run --interp`
        matrix = [
            ("x86_64-linux", 11),
            ("aarch64-linux", 21),
            ("riscv64-linux", 31),
            ("x86_64-windows", 12),
            ("aarch64-macos", 23),
            ("aarch64-android", 24),
            ("aarch64-ios", 25),
            ("x86_64-nostd", 116),
            ("aarch64-nostd", 126),
            ("riscv64-nostd", 136),
        ]
        for triple, expected_rc in matrix:
            cp_c = run_cmd([eonc, "check", "--no-stdlib", f"--target={triple}", str(src)])
            assert cp_c.returncode == 0, f"check --target={triple} failed:\n{cp_c.stderr}"
            cp_r = run_cmd([eonc, "run", "--interp", f"--target={triple}", str(src)])
            assert cp_r.returncode == expected_rc, (
                f"run --interp --target={triple}: expected rc={expected_rc}, got {cp_r.returncode}:\n{cp_r.stderr}"
            )

        # Verify invalid target triple is cleanly rejected
        cp_bad = run_cmd([eonc, "check", "--no-stdlib", "--target=mips-plan9", str(src)])
        assert cp_bad.returncode != 0 and "unknown target triple" in cp_bad.stderr, (
            f"Expected clean unknown target triple error, got rc={cp_bad.returncode}:\n{cp_bad.stderr}"
        )

    print("[PASS] M2.1 TargetSpec (--target=<arch>-<os> across all 10 triples, @target_arch/@target_os/@target_is_nostd, dead comptime pruning)")


def test_m22_aarch64_backend(eonc: str) -> None:
    print("[TEST] M2.2 Native AArch64 Backend (AAPCS64/NEON, ELF64 EM_AARCH64, std/core on qemu-aarch64, shared + nostd)...")
    qemu = shutil.which("qemu-aarch64")
    assert qemu is not None, "qemu-aarch64 not found in PATH"
    with tempfile.TemporaryDirectory(prefix="eon_m22_") as td:
        p = Path(td)
        f_tmp = str(p / "a64_io.txt")
        src = p / "a64_suite.eon"
        src.write_text(
            "module demo.a64_suite;\n"
            "import core.math_vec;\n"
            "import core.color;\n"
            "import core.fmt;\n"
            "import core.sync;\n"
            "import std.rt;\n"
            "import std.time;\n"
            "import std.fs;\n"
            "import std.sync as os_sync;\n"
            "import std.net;\n"
            "\n"
            "iface Compute { fn eval(self: *Self, x: i64) -> i64; }\n"
            "struct Affine { m: i64; b: i64; }\n"
            "impl Compute for Affine {\n"
            "    fn eval(self: *Self, x: i64) -> i64 { return self.m *% x +% self.b; }\n"
            "}\n"
            "fn run_dyn(c: *dyn Compute, x: i64) -> i64 { return c.eval(x); }\n"
            "\n"
            "@recursive\n"
            "fn gcd(a: u64, b: u64) -> u64 {\n"
            "    if b == 0_u64 { return a; }\n"
            "    return gcd(b, a % b);\n"
            "}\n"
            "\n"
            "fn check_arch() -> i32 {\n"
            "    comptime if @target_arch() == .AArch64 { return 183_i32; }\n"
            "    else { return 0_i32; }\n"
            "}\n"
            "\n"
            "pub fn main(env: *mut Env) -> i32 {\n"
            "    if check_arch() != 183_i32 { return 1_i32; }\n"
            "    if gcd(1071_u64, 462_u64) != 21_u64 { return 2_i32; }\n"
            "    // NEON + core.math_vec + core.color + core.fmt\n"
            "    let a = math_vec.Vec4.new(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32);\n"
            "    let b = math_vec.Vec4.new(4.0_f32, 3.0_f32, 2.0_f32, 1.0_f32);\n"
            "    if a.dot(b).trunc_to_int.[i32]() != 20_i32 { return 3_i32; }\n"
            "    let c = color.Srgb8.new(0x33_u8, 0x66_u8, 0x99_u8, 0xFF_u8);\n"
            "    if c.pack_rgba_u32() != 0xFF996633_u32 { return 4_i32; }\n"
            "    var fbuf: [64]u8 = [0_u8; 64];\n"
            "    var fc = fmt.FmtCursor.new();\n"
            "    fc.write_str(&mut fbuf, \"a64=\");\n"
            "    fc.write_u64(&mut fbuf, 42_u64);\n"
            "    if fc.len() != 6_usize || fc.overflowed || fbuf[4_usize] != b'4' || fbuf[5_usize] != b'2' { return 6_i32; }\n"
            "    // dyn + atomics + futex Mutex + clock\n"
            "    let af = Affine.{ m: 3_i64, b: 11_i64 };\n"
            "    if run_dyn(&af, 9_i64) != 38_i64 { return 7_i32; }\n"
            "    var ac = sync.AtomicCounter.init(100_u64);\n"
            "    if ac.fetch_add(25_u64, .SeqCst) != 100_u64 || ac.load(.Acquire) != 125_u64 { return 8_i32; }\n"
            "    var mu = os_sync.Mutex.new();\n"
            "    let g = mu.lock();\n"
            "    g.unlock(&mut mu);\n"
            "    if time.Instant.now(&env.clock).nanos == 0_u64 { return 9_i32; }\n"
            "    // std.fs file I/O + stat\n"
            f"    var wf = fs.File.create_write(&env.fs, \"{f_tmp}\") catch {{ return 10_i32; }};\n"
            "    wf.write_all(\"eon-aarch64\".as_bytes()) catch { wf.close(); return 11_i32; };\n"
            "    wf.close();\n"
            f"    let st = fs.stat_path(&env.fs, \"{f_tmp}\") catch {{ return 12_i32; }};\n"
            "    if !st.is_file || st.is_dir || st.size != 11_u64 { return 13_i32; }\n"
            f"    fs.remove_file(&env.fs, \"{f_tmp}\") catch {{ return 14_i32; }};\n"
            "    // std.net UDP loopback over AArch64 asm-generic syscalls\n"
            "    var srv = net.UdpSocket.bind(&env.net, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16)) catch { return 15_i32; };\n"
            "    let saddr = srv.local_addr() catch { srv.close(); return 16_i32; };\n"
            "    var cli = net.UdpSocket.bind(&env.net, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16)) catch { srv.close(); return 17_i32; };\n"
            "    let sent = cli.send_to(\"ping\".as_bytes(), saddr) catch { cli.close(); srv.close(); return 18_i32; };\n"
            "    var rbuf: [16]u8 = [0_u8; 16];\n"
            "    let rec = srv.recv_from(&mut rbuf) catch { cli.close(); srv.close(); return 19_i32; };\n"
            "    cli.close();\n"
            "    srv.close();\n"
            "    if sent != 4_usize || rec.len != 4_usize || rbuf[0_usize] != b'p' { return 20_i32; }\n"
            "    rt.print_str(\"M22_AARCH64_ALL_OK\\n\");\n"
            "    return 0_i32;\n"
            "}\n",
            encoding="utf-8",
        )
        for mode in ("--dev", "--release"):
            out_bin = p / f"a64_suite_{mode[2:]}"
            cp_b = run_cmd([eonc, "build", mode, "--target=aarch64-linux", str(src), "-o", str(out_bin)])
            assert cp_b.returncode == 0, f"AArch64 {mode} build failed:\n{cp_b.stderr}"
            cp_h = run_cmd(["llvm-readelf", "-h", str(out_bin)])
            assert "AArch64" in cp_h.stdout, f"Expected EM_AARCH64:\n{cp_h.stdout}"
            cp_r = run_cmd([qemu, str(out_bin)])
            assert cp_r.returncode == 0 and "M22_AARCH64_ALL_OK" in cp_r.stdout, (
                f"qemu-aarch64 {mode} failed rc={cp_r.returncode} out={cp_r.stdout!r} err={cp_r.stderr!r}"
            )

        # Shared library (--shared) + freestanding (--target=aarch64-nostd) ELF64 checks
        lib_src = p / "libmath.eon"
        lib_src.write_text(
            "module demo.libmath;\n"
            "@export(\"C\")\n"
            "pub fn a64_add_mul(a: i64, b: i64, c: i64) -> i64 { return a *% b +% c; }\n",
            encoding="utf-8",
        )
        so_path = p / "libmath_a64.so"
        cp_so = run_cmd([eonc, "build", "--release", "--shared", "--target=aarch64-linux", str(lib_src), "-o", str(so_path)])
        assert cp_so.returncode == 0, f"AArch64 --shared build failed:\n{cp_so.stderr}"
        cp_so_h = run_cmd(["llvm-readelf", "-h", "--dyn-syms", str(so_path)])
        assert "AArch64" in cp_so_h.stdout and "DYN" in cp_so_h.stdout and "a64_add_mul" in cp_so_h.stdout, cp_so_h.stdout

        nostd_src = p / "kern_a64.eon"
        nostd_src.write_text(
            "module demo.kern_a64;\n"
            "pub fn kmain() -> i64 {\n"
            "    comptime if @target_arch() == .AArch64 && @target_is_nostd() { return 42_i64; }\n"
            "    else { @compile_error(\"expected aarch64-nostd\"); }\n"
            "}\n",
            encoding="utf-8",
        )
        kern_bin = p / "kern_a64.elf"
        cp_k = run_cmd([eonc, "build", "--release", "--target=aarch64-nostd", str(nostd_src), "-o", str(kern_bin)])
        assert cp_k.returncode == 0, f"AArch64 nostd build failed:\n{cp_k.stderr}"
        cp_k_h = run_cmd(["llvm-readelf", "-h", str(kern_bin)])
        assert "AArch64" in cp_k_h.stdout and "EXEC" in cp_k_h.stdout, cp_k_h.stdout

    print("[PASS] M2.2 Native AArch64 Backend (AAPCS64/NEON, ELF64 EM_AARCH64, std/core live on qemu-aarch64, --shared & nostd)")


def test_m23_riscv64_backend(eonc: str) -> None:
    print("[TEST] M2.3 Native RISC-V 64 Backend (RV64IMAFD/LP64D, ELF64 EM_RISCV, std/core on qemu-riscv64, shared + nostd)...")
    qemu = shutil.which("qemu-riscv64")
    assert qemu is not None, "qemu-riscv64 must be installed to verify M2.3"
    with tempfile.TemporaryDirectory(prefix="eon_m23_") as td:
        p = Path(td)
        f_tmp = str(p / "rv_io.txt")
        src = p / "rv64_suite.eon"
        src.write_text(
            "module demo.rv64_suite;\n"
            "import core.math_vec;\n"
            "import core.color;\n"
            "import core.fmt;\n"
            "import core.sync;\n"
            "import std.rt;\n"
            "import std.time;\n"
            "import std.fs;\n"
            "import std.sync as os_sync;\n"
            "import std.net;\n"
            "\n"
            "iface Eval { fn eval(self: *Self, x: i64) -> i64; }\n"
            "struct Affine { m: i64; b: i64; }\n"
            "impl Eval for Affine { fn eval(self: *Self, x: i64) -> i64 { return self.m *% x +% self.b; } }\n"
            "fn run_dyn(e: *dyn Eval, x: i64) -> i64 { return e.eval(x); }\n"
            "\n"
            "@recursive\n"
            "fn gcd(a: u64, b: u64) -> u64 {\n"
            "    if b == 0_u64 { return a; }\n"
            "    return gcd(b, a % b);\n"
            "}\n"
            "\n"
            "fn check_arch() -> i32 {\n"
            "    comptime if @target_arch() == .RiscV64 { return 243_i32; }\n"
            "    else { return 0_i32; }\n"
            "}\n"
            "\n"
            "pub fn main(env: *mut Env) -> i32 {\n"
            "    if check_arch() != 243_i32 { return 1_i32; }\n"
            "    if gcd(1071_u64, 462_u64) != 21_u64 { return 2_i32; }\n"
            "    let a = math_vec.Vec4.new(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32);\n"
            "    let b = math_vec.Vec4.new(4.0_f32, 3.0_f32, 2.0_f32, 1.0_f32);\n"
            "    if a.dot(b).trunc_to_int.[i32]() != 20_i32 { return 3_i32; }\n"
            "    let c = color.Srgb8.new(0x33_u8, 0x66_u8, 0x99_u8, 0xFF_u8);\n"
            "    if c.pack_rgba_u32() != 0xFF996633_u32 { return 4_i32; }\n"
            "    var fbuf: [64]u8 = [0_u8; 64];\n"
            "    var fc = fmt.FmtCursor.new();\n"
            "    fc.write_str(&mut fbuf, \"rv=\");\n"
            "    fc.write_u64(&mut fbuf, 243_u64);\n"
            "    if fc.len() != 6_usize || fc.overflowed || fbuf[3_usize] != b'2' || fbuf[5_usize] != b'3' { return 6_i32; }\n"
            "    let af = Affine.{ m: 3_i64, b: 11_i64 };\n"
            "    if run_dyn(&af, 9_i64) != 38_i64 { return 7_i32; }\n"
            "    var ac = sync.AtomicCounter.init(100_u64);\n"
            "    if ac.fetch_add(25_u64, .SeqCst) != 100_u64 || ac.load(.Acquire) != 125_u64 { return 8_i32; }\n"
            "    var mu = os_sync.Mutex.new();\n"
            "    let g = mu.lock();\n"
            "    g.unlock(&mut mu);\n"
            "    if time.Instant.now(&env.clock).nanos == 0_u64 { return 9_i32; }\n"
            f"    var wf = fs.File.create_write(&env.fs, \"{f_tmp}\") catch {{ return 10_i32; }};\n"
            "    wf.write_all(\"eon-riscv64\".as_bytes()) catch { wf.close(); return 11_i32; };\n"
            "    wf.close();\n"
            f"    let st = fs.stat_path(&env.fs, \"{f_tmp}\") catch {{ return 12_i32; }};\n"
            "    if !st.is_file || st.is_dir || st.size != 11_u64 { return 13_i32; }\n"
            f"    fs.remove_file(&env.fs, \"{f_tmp}\") catch {{ return 14_i32; }};\n"
            "    var srv = net.UdpSocket.bind(&env.net, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16)) catch { return 15_i32; };\n"
            "    let saddr = srv.local_addr() catch { srv.close(); return 16_i32; };\n"
            "    var cli = net.UdpSocket.bind(&env.net, net.SocketAddrV4.new(net.Ipv4Addr.localhost(), 0_u16)) catch { srv.close(); return 17_i32; };\n"
            "    let sent = cli.send_to(\"pong\".as_bytes(), saddr) catch { cli.close(); srv.close(); return 18_i32; };\n"
            "    var rbuf: [16]u8 = [0_u8; 16];\n"
            "    let rec = srv.recv_from(&mut rbuf) catch { cli.close(); srv.close(); return 19_i32; };\n"
            "    cli.close();\n"
            "    srv.close();\n"
            "    if sent != 4_usize || rec.len != 4_usize || rbuf[0_usize] != b'p' { return 20_i32; }\n"
            "    rt.print_str(\"M23_RISCV64_ALL_OK\\n\");\n"
            "    return 0_i32;\n"
            "}\n",
            encoding="utf-8",
        )
        for mode in ("--dev", "--release"):
            out_bin = p / f"rv64_suite_{mode[2:]}"
            cp_b = run_cmd([eonc, "build", mode, "--target=riscv64-linux", str(src), "-o", str(out_bin)])
            assert cp_b.returncode == 0, f"RISC-V 64 {mode} build failed:\n{cp_b.stderr}"
            cp_h = run_cmd(["llvm-readelf", "-h", str(out_bin)])
            assert "RISC-V" in cp_h.stdout, f"Expected EM_RISCV:\n{cp_h.stdout}"
            cp_r = run_cmd([qemu, str(out_bin)])
            assert cp_r.returncode == 0 and "M23_RISCV64_ALL_OK" in cp_r.stdout, (
                f"qemu-riscv64 {mode} failed rc={cp_r.returncode} out={cp_r.stdout!r} err={cp_r.stderr!r}"
            )

        lib_src = p / "libmath_rv.eon"
        lib_src.write_text(
            "module demo.libmath_rv;\n"
            "@export(\"C\")\n"
            "pub fn rv64_add_mul(a: i64, b: i64, c: i64) -> i64 { return a *% b +% c; }\n",
            encoding="utf-8",
        )
        so_path = p / "libmath_rv64.so"
        cp_so = run_cmd([eonc, "build", "--release", "--shared", "--target=riscv64-linux", str(lib_src), "-o", str(so_path)])
        assert cp_so.returncode == 0, f"RISC-V 64 --shared build failed:\n{cp_so.stderr}"
        cp_so_h = run_cmd(["llvm-readelf", "-h", "--dyn-syms", str(so_path)])
        assert "RISC-V" in cp_so_h.stdout and "DYN" in cp_so_h.stdout and "rv64_add_mul" in cp_so_h.stdout, cp_so_h.stdout

        nostd_src = p / "kern_rv64.eon"
        nostd_src.write_text(
            "module demo.kern_rv64;\n"
            "pub fn kmain() -> i64 {\n"
            "    comptime if @target_arch() == .RiscV64 && @target_is_nostd() { return 243_i64; }\n"
            "    else { @compile_error(\"expected riscv64-nostd\"); }\n"
            "}\n",
            encoding="utf-8",
        )
        kern_bin = p / "kern_rv64.elf"
        cp_k = run_cmd([eonc, "build", "--release", "--target=riscv64-nostd", str(nostd_src), "-o", str(kern_bin)])
        assert cp_k.returncode == 0, f"RISC-V 64 nostd build failed:\n{cp_k.stderr}"
        cp_k_h = run_cmd(["llvm-readelf", "-h", str(kern_bin)])
        assert "RISC-V" in cp_k_h.stdout and "EXEC" in cp_k_h.stdout, cp_k_h.stdout

    print("[PASS] M2.3 Native RISC-V 64 Backend (RV64IMAFD/LP64D, ELF64 EM_RISCV, std/core live on qemu-riscv64, --shared & nostd)")


def _find_qemu_sys(name: str) -> tuple[str | None, dict[str, str]]:
    env = os.environ.copy()
    local_bin = Path.home() / ".local/usr/bin" / name
    if local_bin.exists():
        lib_dir = str(Path.home() / ".local/usr/lib")
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
        return str(local_bin), env
    found = shutil.which(name)
    return found, env


def test_m24_linkers_and_nostd_fb(eonc: str):
    """Verify M2.4: Zero-C PE32+ (.exe/--gui/.dll) live in Wine, Mach-O 64 + SHA-256 LC_CODE_SIGNATURE
    + .app bundle, Android ET_DYN .so + zipalign + AXML + OpenSSL-verified ECDSA-P256 APK v2,
    and all 3 bare-metal nostd framebuffers (x86_64, AArch64, RISC-V 64) live in QEMU."""
    import hashlib
    import struct
    import zipfile

    with tempfile.TemporaryDirectory(prefix="eon_m24_reg_") as td:
        p = Path(td)

        # 1. Windows PE32+ (--dev + --release + --gui + --shared .dll) live under Wine
        win_src = p / "win_m24.eon"
        win_src.write_text(
            "module demo.win_m24;\n"
            "import core.col;\n"
            "import core.str;\n"
            "import std.rt;\n"
            "import std.os.windows;\n"
            "@extern(\"C\")\n"
            "fn win_cb(a: i64, b: i64, c: i64, d: i64, e: i64) -> i64 {\n"
            "    return a *% 10000_i64 +% b *% 1000_i64 +% c *% 100_i64 +% d *% 10_i64 +% e;\n"
            "}\n"
            "fn sum_sq(alloc: *mut AllocCap) -> i64 ! OutOfMemory {\n"
            "    var buf = try col.DynBuf.[i64].init(alloc, 16_usize);\n"
            "    errdefer buf.destroy(alloc);\n"
            "    var i: i64 = 1_i64;\n"
            "    while i <= 10_i64 { try buf.push(alloc, i *% i); i = i +% 1_i64; }\n"
            "    var sum: i64 = 0_i64;\n"
            "    let s = buf.as_slice();\n"
            "    var k: usize = 0_usize;\n"
            "    while k < s.len { sum = sum +% s[k]; k = k +% 1_usize; }\n"
            "    buf.destroy(alloc);\n"
            "    return sum;\n"
            "}\n"
            "pub fn main(env: *mut Env) -> i32 ! OutOfMemory {\n"
            "    rt.print_str(\"WIN_M24_OK\\n\");\n"
            "    let sum = try sum_sq(&mut env.alloc);\n"
            "    let qpc_ok: u64 = if windows.qpc_freq() > 0_u64 && windows.qpc_now() > 0_u64 { 1_u64 } else { 0_u64 };\n"
            "    let cb = win_cb(1_i64, 2_i64, 3_i64, 4_i64, 5_i64);\n"
            "    var out = try str.StrBuf.init(&mut env.alloc, 64_usize);\n"
            "    errdefer out.destroy(&mut env.alloc);\n"
            "    try out.push_str_grow(&mut env.alloc, \"SUM=\");\n"
            "    try out.push_u64_grow(&mut env.alloc, sum.bitcast.[u64]());\n"
            "    try out.push_str_grow(&mut env.alloc, \" CB=\");\n"
            "    try out.push_u64_grow(&mut env.alloc, cb.bitcast.[u64]());\n"
            "    try out.push_str_grow(&mut env.alloc, \" QPC=\");\n"
            "    try out.push_u64_grow(&mut env.alloc, qpc_ok);\n"
            "    try out.push_byte_grow(&mut env.alloc, b'\\n');\n"
            "    rt.print_bytes(out.as_bytes());\n"
            "    out.destroy(&mut env.alloc);\n"
            "    return 42_i32;\n"
            "}\n",
            encoding="utf-8",
        )
        wine = shutil.which("wine")
        for mode in ("--dev", "--release"):
            exe = p / f"win_{mode[2:]}.exe"
            r = run_cmd([eonc, "build", mode, "--target=x86_64-windows", str(win_src), "-o", str(exe)])
            assert r.returncode == 0, f"Windows PE32+ {mode} build failed:\n{r.stderr}"
            b = exe.read_bytes()
            assert b[:2] == b"MZ", "Missing MZ DOS stub"
            if wine:
                wenv = os.environ.copy()
                wenv["WINEDEBUG"] = "-all"
                w = subprocess.run([wine, str(exe)], env=wenv, capture_output=True, text=True, timeout=30)
                assert w.returncode == 42 and "WIN_M24_OK" in w.stdout and "SUM=385 CB=12345 QPC=1" in w.stdout, (
                    f"Wine {mode} failed: rc={w.returncode} out={w.stdout!r} err={w.stderr[:300]!r}"
                )
        gui_exe = p / "win_gui.exe"
        assert run_cmd([eonc, "build", "--release", "--gui", "--target=x86_64-windows", str(win_src), "-o", str(gui_exe)]).returncode == 0
        gb = gui_exe.read_bytes()
        g_pe = struct.unpack_from("<I", gb, 0x3C)[0]
        assert struct.unpack_from("<H", gb, g_pe + 24 + 68)[0] == 2, "Expected IMAGE_SUBSYSTEM_WINDOWS_GUI (2)"
        dll_p = p / "win_lib.dll"
        assert run_cmd([eonc, "build", "--release", "--shared", "--target=x86_64-windows", str(win_src), "-o", str(dll_p)]).returncode == 0
        db = dll_p.read_bytes()
        d_pe = struct.unpack_from("<I", db, 0x3C)[0]
        assert (struct.unpack_from("<H", db, d_pe + 22)[0] & 0x2000) != 0, "Expected IMAGE_FILE_DLL"

        # 2. Darwin Mach-O 64 + SHA-256 LC_CODE_SIGNATURE + .app bundle
        dar_src = p / "dar_m24.eon"
        dar_src.write_text(
            "module demo.dar_m24;\n"
            "import std.rt;\n"
            "import std.os.darwin;\n"
            "pub fn main(env: *mut Env) -> i32 {\n"
            "    _ = env;\n"
            "    let ptr = darwin.page_alloc(4096_usize);\n"
            "    if ptr != 0_u64 { _ = darwin.page_free(ptr, 4096_usize); }\n"
            "    rt.print_str(\"DARWIN_M24_OK\\n\");\n"
            "    return 0_i32;\n"
            "}\n",
            encoding="utf-8",
        )

        def check_macho(path: Path, exp_cpu: int, exp_plat: int):
            data = path.read_bytes()
            magic, cputype, _, _, ncmds, sizeofcmds, _, _ = struct.unpack_from("<8I", data, 0)
            assert magic == 0xFEEDFACF and cputype == exp_cpu
            off = 32
            cs_off = cs_sz = plat = 0
            seen = set()
            for _ in range(ncmds):
                cmd, cmdsize = struct.unpack_from("<II", data, off)
                seen.add(cmd)
                if cmd == 0x1D:
                    cs_off, cs_sz = struct.unpack_from("<II", data, off + 8)
                elif cmd == 0x32:
                    plat = struct.unpack_from("<I", data, off + 8)[0]
                off += cmdsize
            assert off == 32 + sizeofcmds and plat == exp_plat
            assert {0x80000034, 0x80000033, 0x80000028, 0x1D}.issubset(seen)
            sb_magic, _, sb_cnt, slot_ty, cd_rel = struct.unpack_from(">5I", data, cs_off)
            assert sb_magic == 0xFADE0CC0 and sb_cnt == 1 and slot_ty == 0
            cd_off = cs_off + cd_rel
            cd_magic, _, _, _, hash_off, _, _, n_code, code_limit = struct.unpack_from(">9I", data, cd_off)
            hsz, hty, _, psh = struct.unpack_from(">4B", data, cd_off + 36)
            assert cd_magic == 0xFADE0C02 and hsz == 32 and hty == 2 and psh == 12 and code_limit == cs_off
            pg_sz = 1 << psh
            assert n_code == (code_limit + pg_sz - 1) // pg_sz
            for idx in range(n_code):
                want = hashlib.sha256(data[idx * pg_sz : min((idx + 1) * pg_sz, code_limit)]).digest()
                got = data[cd_off + hash_off + idx * 32 : cd_off + hash_off + (idx + 1) * 32]
                assert want == got, f"Mach-O LC_CODE_SIGNATURE SHA-256 mismatch on page {idx}"

        for triple, cput, plat, nm in [
            ("aarch64-macos", 0x0100000C, 1, "mac_arm64"),
            ("x86_64-macos", 0x01000007, 1, "mac_x64"),
            ("aarch64-ios", 0x0100000C, 2, "ios_arm64"),
        ]:
            outp = p / nm
            r = run_cmd([eonc, "build", "--release", f"--target={triple}", str(dar_src), "-o", str(outp)])
            assert r.returncode == 0, f"Mach-O {triple} build failed:\n{r.stderr}"
            check_macho(outp, cput, plat)
        app_dir = p / "CalcTest.app"
        assert run_cmd([eonc, "build", "--release", "--target=aarch64-macos", str(dar_src), "-o", str(app_dir)]).returncode == 0
        assert (app_dir / "Contents/Info.plist").is_file()
        check_macho(app_dir / "Contents/MacOS/CalcTest", 0x0100000C, 1)

        # 3. Android ET_DYN .so + Signed APK v2 (zipalign + AXML + OpenSSL ECDSA-P256 verify)
        and_src = p / "and_m24.eon"
        and_src.write_text(
            "module demo.and_m24;\n"
            "import std.rt;\n"
            "import std.os.android;\n"
            "@export\n"
            "@extern(\"C\")\n"
            "pub fn ANativeActivity_onCreate(act: u64, st: u64, sz: usize) {\n"
            "    _ = act; _ = st; _ = sz;\n"
            "    _ = android.log_info(\"EonM24\\0\".as_bytes(), \"OK\\0\".as_bytes());\n"
            "}\n"
            "pub fn main(env: *mut Env) -> i32 { _ = env; rt.print_str(\"AND\\n\"); return 0_i32; }\n",
            encoding="utf-8",
        )
        openssl = shutil.which("openssl")
        for triple, abi, em in [("aarch64-android", "arm64-v8a", 183), ("x86_64-android", "x86_64", 62)]:
            apkp = p / f"app_{abi}.apk"
            r = run_cmd([eonc, "build", "--release", f"--target={triple}", str(and_src), "-o", str(apkp)])
            assert r.returncode == 0, f"Android APK {triple} build failed:\n{r.stderr}"
            raw_apk = apkp.read_bytes()
            with zipfile.ZipFile(apkp, "r") as zf:
                assert zf.testzip() is None
                axml = zf.read("AndroidManifest.xml")
                assert struct.unpack_from("<I", axml, 0)[0] == 0x00080003
                so_name = f"lib/{abi}/libmain.so"
                so_b = zf.read(so_name)
                assert so_b[:4] == b"\x7fELF" and struct.unpack_from("<HH", so_b, 16) == (3, em)
                lh = zf.getinfo(so_name).header_offset
                fn_l, ex_l = struct.unpack_from("<HH", raw_apk, lh + 26)
                assert (lh + 30 + fn_l + ex_l) % 4096 == 0, "libmain.so not 4096-byte zipaligned"
            eocd_off = raw_apk.rfind(b"PK\x05\x06")
            _, cd_off = struct.unpack_from("<II", raw_apk, eocd_off + 12)
            assert raw_apk[cd_off - 16 : cd_off] == b"APK Sig Block 42"
            blk_sz = struct.unpack_from("<Q", raw_apk, cd_off - 24)[0]
            blk_start = cd_off - blk_sz - 8
            pos = blk_start + 20 + 8
            sd_len = struct.unpack_from("<I", raw_apk, pos)[0]; pos += 4
            sd = raw_apk[pos : pos + sd_len]; pos += sd_len
            pos += 8
            sig_alg, sig_len = struct.unpack_from("<II", raw_apk, pos); pos += 8
            sig_der = raw_apk[pos : pos + sig_len]
            assert sig_alg == 0x0201
            sec1, sec3, sec4 = raw_apk[:blk_start], raw_apk[cd_off:eocd_off], bytearray(raw_apk[eocd_off:])
            struct.pack_into("<I", sec4, 16, blk_start)
            chunks = []
            for sec in (sec1, sec3, bytes(sec4)):
                if not sec:
                    chunks.append(hashlib.sha256(b"\xa5\x00\x00\x00\x00").digest())
                else:
                    for off_c in range(0, len(sec), 1048576):
                        part = sec[off_c : off_c + 1048576]
                        chunks.append(hashlib.sha256(b"\xa5" + struct.pack("<I", len(part)) + part).digest())
            want_d = hashlib.sha256(b"\x5a" + struct.pack("<I", len(chunks)) + b"".join(chunks)).digest()
            assert want_d == sd[16:48], "APK v2 Merkle digest mismatch"
            if openssl:
                cert_len = struct.unpack_from("<I", sd, 52)[0]
                cf, pf, sdf, sigf = p / f"c_{abi}.der", p / f"p_{abi}.pem", p / f"s_{abi}.bin", p / f"g_{abi}.der"
                cf.write_bytes(sd[56 : 56 + cert_len])
                sdf.write_bytes(sd)
                sigf.write_bytes(sig_der)
                pf.write_bytes(subprocess.check_output([openssl, "x509", "-inform", "DER", "-in", str(cf), "-pubkey", "-noout"]))
                vr = subprocess.run([openssl, "dgst", "-sha256", "-verify", str(pf), "-signature", str(sigf), str(sdf)], capture_output=True, text=True)
                assert vr.returncode == 0 and "Verified OK" in vr.stdout, f"OpenSSL ECDSA-P256 failed: {vr.stdout} {vr.stderr}"

        # 4. All 3 bare-metal nostd framebuffers live in QEMU (x86_64, AArch64, RISC-V 64)
        nostd_cases = [
            ("x86_64-nostd", "x86_64", "qemu-system-x86_64", "0x00400000_u64", True),
            ("aarch64-nostd", "aarch64", "qemu-system-aarch64", "0x45000000_u64", False),
            ("riscv64-nostd", "riscv64", "qemu-system-riscv64", "0x85000000_u64", False),
        ]
        for triple, arch, qname, heap_base, is_x64 in nostd_cases:
            ksrc = p / f"fb_{arch}.eon"
            init_ser = "    fb.serial_init();\n" if is_x64 else ""
            exit_ok = "        fb.qemu_exit(0x10_u8);\n" if is_x64 else "        fb.qemu_exit();\n"
            exit_fl = "        fb.qemu_exit(0x20_u8);\n" if is_x64 else "        fb.qemu_exit();\n"
            ksrc.write_text(
                f"module demo.fb_{arch};\n"
                f"import kernel.arch.{arch}.fb;\n"
                "pub fn kmain() {\n"
                f"{init_ser}"
                "    let info = fb.init_framebuffer(800_u32, 600_u32);\n"
                "    fb.put_pixel(&info, 10_u32, 5_u32, 0xFF112233_u32);\n"
                "    let px = fb.read_pixel(&info, 10_u32, 5_u32);\n"
                f"    var hp = fb.StaticHeap.init({heap_base}, 0x01000000_u64);\n"
                "    let blk = hp.alloc_aligned(4096_u64, 64_u64);\n"
                "    let fp: f32 = 1.5_f32 * 4.0_f32;\n"
                f"    if px == 0xFF112233_u32 && blk == {heap_base} && fp > 5.9_f32 && fp < 6.1_f32 {{\n"
                f"        fb.serial_write(\"[Eon-nostd-fb] {arch} 800x600 OK\\n\");\n"
                f"{exit_ok}"
                "    } else {\n"
                f"        fb.serial_write(\"[Eon-nostd-fb] {arch} FAIL\\n\");\n"
                f"{exit_fl}"
                "    }\n"
                "}\n",
                encoding="utf-8",
            )
            kelf = p / f"fb_{arch}.elf"
            rb = run_cmd([eonc, "build", "--release", f"--target={triple}", str(ksrc), str(ROOT / f"kernel/arch/{arch}/fb.hal.eon"), "-o", str(kelf)])
            assert rb.returncode == 0, f"{triple} build failed:\n{rb.stderr}"
            qbin, qenv = _find_qemu_sys(qname)
            if qbin:
                if arch == "x86_64":
                    qcmd = [qbin, "-nographic", "-m", "128M", "-vga", "std", "-no-reboot", "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04", "-kernel", str(kelf)]
                    exp_rc = 33
                elif arch == "aarch64":
                    qcmd = [qbin, "-M", "virt", "-cpu", "cortex-a57", "-m", "256M", "-device", "ramfb", "-nographic", "-kernel", str(kelf)]
                    exp_rc = 0
                else:
                    qcmd = [qbin, "-M", "virt", "-bios", str(kelf), "-m", "256M", "-device", "ramfb", "-nographic"]
                    exp_rc = 0
                qr = subprocess.run(qcmd, env=qenv, capture_output=True, text=True, timeout=20)
                assert qr.returncode == exp_rc and f"[Eon-nostd-fb] {arch} 800x600 OK" in qr.stdout, (
                    f"{qname} failed rc={qr.returncode} out={qr.stdout!r} err={qr.stderr[:300]!r}"
                )

    print("[PASS] M2.4 Zero-C Linkers (PE32+ live in Wine, Mach-O SHA-256 + .app, Android APK OpenSSL ECDSA-P256) & 3x QEMU nostd Framebuffers")


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
    test_m14_fmt_and_lsp(eonc)
    test_m21_target_spec_and_comptime(eonc)
    test_m22_aarch64_backend(eonc)
    test_m23_riscv64_backend(eonc)
    test_m24_linkers_and_nostd_fb(eonc)
    print("ALL SELFHOST SOUNDNESS REGRESSIONS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
