#!/usr/bin/env python3
"""Soundness Fuzzer for `verify.rs`, `borrow.rs`, `cfg_analysis.rs`, and `effects.rs`.

Generates deliberately invalid / mutated Eon programs across 7 soundness families and asserts
that `eonc check` rejects EVERY SINGLE ONE with the expected Zero-Mercy / safety diagnostic
(0 missed out of N).

Soundness families covered:
  1. Bounds off-by-one & slice bounds (`E0107`)
  2. Integer overflow / underflow / div0 / bad shift / signed MIN neg (`E0108`)
  3. Law of Exclusivity & borrow/closure/view-struct aliasing (`E0118`)
  4. Linear resource leaks, branch leaks, try leaks, double-consume, discard (`E0109`, `E0323`, `E0324`)
  5. Second-class escape: `view struct` / `*T` / `fn` stored in struct or returned (`E0310`)
  6. Stale `ArenaPool` `Id[p]` used after `reset()` (`E0331`)
  7. Unannotated recursion & effect violations (`E0112`, `E0113`)

Usage:
    python3 tests/fuzz/soundness_fuzz.py [-n 3000] [--seed 1] [--jobs N] [--eonc PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE_EONC = REPO / "compiler" / "target" / "release" / "eonc"
DEBUG_EONC = REPO / "compiler" / "target" / "debug" / "eonc"
DEFAULT_EONC = RELEASE_EONC if RELEASE_EONC.exists() else DEBUG_EONC


def gen_bounds_mutation(r: random.Random) -> tuple[str, set[str], str]:
    n = r.randint(2, 64)
    k = r.randint(0, 8)
    kind = r.randint(0, 5)
    if kind == 0:
        # Direct off-by-one array index `arr[n + k]`
        src = f"""
module sf.bounds;
pub fn bad() -> u64 {{
    let arr: [{n}]u64 = [0_u64; {n}];
    return arr[{n + k}_usize];
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0107"}, "arr_const_oob"
    if kind == 1:
        # `<=` instead of `<` on slice index
        src = f"""
module sf.bounds;
pub fn bad(s: *[u8], i: usize) -> u8 {{
    if i <= s.len {{
        return s[i];
    }}
    return 0_u8;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0107"}, "slice_le_instead_of_lt"
    if kind == 2:
        # Difference algebra off-by-one: `len - start >= d` indexing `start + d`
        d = r.randint(1, 6)
        src = f"""
module sf.bounds;
pub fn bad(s: *[u8], start: usize) -> u8 {{
    if start <= s.len && s.len - start >= {d}_usize {{
        return s[start + {d}_usize];
    }}
    return 0_u8;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0107"}, "diff_algebra_off_by_one"
    if kind == 3:
        # KnownBits mask too wide: `x & (2*p - 1)` into `[p]u64`
        p = r.choice([4, 8, 16, 32])
        mask = (p * 2) - 1
        src = f"""
module sf.bounds;
pub fn bad(x: usize) -> u64 {{
    let tbl: [{p}]u64 = [0_u64; {p}];
    let idx = x & {mask}_usize;
    return tbl[idx];
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0107"}, "known_bits_mask_too_wide"
    if kind == 4:
        # Slice lo > hi not ruled out
        src = f"""
module sf.bounds;
pub fn bad(s: *[u8], lo: usize, hi: usize) -> usize {{
    if lo <= s.len && hi <= s.len {{
        let sub = s[lo..hi];
        return sub.len;
    }}
    return 0_usize;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0107"}, "slice_lo_gt_hi"
    # Field index after mutating field
    src = f"""
module sf.bounds;
pub view struct Cur {{ pub s: *[u8]; pub pos: usize; }}
impl Cur {{
    fn bad(self: *mut Self) -> u8 {{
        if self.pos < self.s.len {{
            self.pos = self.pos + 1_usize;
            return self.s[self.pos];
        }}
        return 0_u8;
    }}
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
    return src, {"E0107"}, "view_field_mutated_before_index"


def gen_arith_mutation(r: random.Random) -> tuple[str, set[str], str]:
    t = r.choice(["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "u5", "i7"])
    kind = r.randint(0, 5)
    if kind == 0:
        # Unchecked addition overflow
        src = f"""
module sf.arith;
pub fn bad(a: {t}, b: {t}) -> {t} {{
    return a + b;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0108"}, f"add_overflow_{t}"
    if kind == 1:
        # Unchecked subtraction underflow
        src = f"""
module sf.arith;
pub fn bad(a: {t}, b: {t}) -> {t} {{
    return a - b;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0108"}, f"sub_underflow_{t}"
    if kind == 2:
        # Unchecked division by zero
        src = f"""
module sf.arith;
pub fn bad(a: {t}, b: {t}) -> {t} {{
    return a / b;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0108"}, f"div_zero_{t}"
    if kind == 3:
        # Signed MIN / -1 overflow even with `b != 0`
        st = r.choice(["i8", "i16", "i32", "i64"])
        src = f"""
module sf.arith;
pub fn bad(a: {st}, b: {st}) -> {st} {{
    if b != 0_{st} {{
        return a / b;
    }}
    return 0_{st};
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0108"}, f"signed_min_div_neg1_{st}"
    if kind == 4:
        # Unary `-` on possibly MIN signed integer
        st = r.choice(["i8", "i16", "i32", "i64", "i7", "i12"])
        src = f"""
module sf.arith;
pub fn bad(a: {st}) -> {st} {{
    return -a;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0108"}, f"neg_min_{st}"
    # Decrementing loop loses `start <= p`
    src = f"""
module sf.arith;
pub fn bad(n: usize) -> usize {{
    let start = n;
    var p = start;
    while p > 0_usize {{
        p = p - 1_usize;
    }}
    return p - start;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
    return src, {"E0108"}, "decrement_loop_sub"


def gen_alias_mutation(r: random.Random) -> tuple[str, set[str], str]:
    v = r.randint(1, 99)
    kind = r.randint(0, 3)
    if kind == 0:
        # Two `&mut` to same var in call
        src = f"""
module sf.alias;
fn touch(a: *mut u64, b: *mut u64) {{ a.* = b.* +% 1_u64; }}
pub fn main() -> i32 {{
    var x: u64 = {v}_u64;
    touch(&mut x, &mut x);
    return 0_i32;
}}
"""
        return src, {"E0118"}, "double_mut_call"
    if kind == 1:
        # Mutate var while shared borrow is live
        src = f"""
module sf.alias;
pub fn main() -> i32 {{
    var x: u64 = {v}_u64;
    let r: *u64 = &x;
    x = {v + 1}_u64;
    let _ = r.*;
    return 0_i32;
}}
"""
        return src, {"E0118"}, "mutate_while_shared_live"
    if kind == 2:
        # `view struct` literal aliasing `&mut x` and `&x`
        src = f"""
module sf.alias;
pub view struct AliasPair {{ pub a: *mut u64; pub b: *u64; }}
pub fn main() -> i32 {{
    var x: u64 = {v}_u64;
    let p = AliasPair.{{ a: &mut x, b: &x }};
    let _ = p.b.*;
    return 0_i32;
}}
"""
        return src, {"E0118"}, "view_struct_alias"
    # Mutate captured variable while closure borrowing it is live
    src = f"""
module sf.alias;
pub fn main() -> i32 {{
    var x: u64 = {v}_u64;
    let f: fn(u64) -> u64 = |a: u64| {{ a +% x }};
    x = {v + 2}_u64;
    let _ = f(1_u64);
    return 0_i32;
}}
"""
    return src, {"E0118"}, "mutate_while_closure_captures"


def gen_linear_mutation(r: random.Random) -> tuple[str, set[str], str]:
    v = r.randint(1, 50)
    kind = r.randint(0, 4)
    if kind == 0:
        # Drop linear struct without consuming
        src = f"""
module sf.lin;
pub linear struct Handle {{ pub fd: u32; }}
pub fn main() -> i32 {{
    let h = Handle.{{ fd: {v}_u32 }};
    let _ = h.fd;
    return 0_i32;
}}
"""
        return src, {"E0109"}, "linear_leak_scope_exit"
    if kind == 1:
        # Leak on one branch of `if`
        src = f"""
module sf.lin;
pub linear struct Handle {{ pub fd: u32; }}
fn close_h(h: Handle) {{ let Handle.{{ fd: _ }} = h; }}
pub fn bad(cond: bool) {{
    let h = Handle.{{ fd: {v}_u32 }};
    if cond {{
        close_h(h);
    }}
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0109", "E0323"}, "linear_branch_leak"
    if kind == 2:
        # Leak on `try` early return without `errdefer`
        src = f"""
module sf.lin;
pub error IoErr {{ Fail }}
pub linear struct Handle {{ pub fd: u32; }}
fn close_h(h: Handle) {{ let Handle.{{ fd: _ }} = h; }}
fn might_fail(b: bool) -> !IoErr {{
    if b {{ return .Fail; }}
    return;
}}
pub fn bad(b: bool) -> !IoErr {{
    let h = Handle.{{ fd: {v}_u32 }};
    try might_fail(b);
    close_h(h);
    return;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0109", "E0323"}, "linear_try_leak"
    if kind == 3:
        # Double consume of linear value
        src = f"""
module sf.lin;
pub linear struct Handle {{ pub fd: u32; }}
fn close_h(h: Handle) {{ let Handle.{{ fd: _ }} = h; }}
pub fn main() -> i32 {{
    let h = Handle.{{ fd: {v}_u32 }};
    close_h(h);
    close_h(h);
    return 0_i32;
}}
"""
        return src, {"E0109", "E0321", "E0324"}, "linear_double_consume"
    # Discard linear value with `_ = h;`
    src = f"""
module sf.lin;
pub linear struct Handle {{ pub fd: u32; }}
pub fn main() -> i32 {{
    let h = Handle.{{ fd: {v}_u32 }};
    _ = h;
    return 0_i32;
}}
"""
    return src, {"E0109", "E0324"}, "linear_discard"


def gen_escape_mutation(r: random.Random) -> tuple[str, set[str], str]:
    v = r.randint(1, 50)
    kind = r.randint(0, 3)
    if kind == 0:
        # Regular struct storing `view struct`
        src = """
module sf.esc;
pub view struct V { pub s: *[u8]; }
pub struct Bad { pub v: V; }
pub fn main() -> i32 { return 0_i32; }
"""
        return src, {"E0310"}, "view_in_regular_struct"
    if kind == 1:
        # Non-view fn returning `view struct`
        src = """
module sf.esc;
pub view struct V { pub s: *[u8]; }
fn bad(s: *[u8]) -> V { return V.{ s: s }; }
pub fn main() -> i32 { return 0_i32; }
"""
        return src, {"E0310"}, "return_view_non_view_fn"
    if kind == 2:
        # Returning local borrow from `view fn`
        src = f"""
module sf.esc;
view fn bad(x: *u64) -> *u64 {{
    let local: u64 = x.* +% {v}_u64;
    return &local;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
        return src, {"E0310"}, "return_local_borrow_view_fn"
    # Regular struct storing closure `fn(...)`
    src = """
module sf.esc;
pub struct Bad { pub cb: fn(u64) -> u64; }
pub fn main() -> i32 { return 0_i32; }
"""
    return src, {"E0310"}, "fnptr_in_regular_struct"


def gen_stale_pool_mutation(r: random.Random) -> tuple[str, set[str], str]:
    v = r.randint(1, 100)
    src = f"""
module sf.arenapool;
pub pool Row {{ pub x: u64; }}
pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | MemError {{
    var arena = try Arena.init(alloc, 4096_usize);
    errdefer arena.destroy(alloc);
    var p = try ArenaPool.[Row].init(&mut arena, 8_usize);
    let id = try p.push(.{{ x: {v}_u64 }});
    p = p.reset();
    let _ = p.x[id];
    arena.destroy(alloc);
    return 0_i32;
}}
"""
    return src, {"E0331"}, "stale_id_after_reset"


def gen_effects_mutation(r: random.Random) -> tuple[str, set[str], str]:
    v = r.randint(1, 20)
    kind = r.randint(0, 2)
    if kind == 0:
        # Unannotated direct recursion
        src = f"""
module sf.eff;
fn bad(n: u64) -> u64 {{
    if n == 0_u64 {{ return {v}_u64; }}
    return bad(n - 1_u64);
}}
pub fn main() -> i32 {{ let _ = bad(3_u64); return 0_i32; }}
"""
        return src, {"E0113"}, "unannotated_direct_recursion"
    if kind == 1:
        # Unannotated mutual recursion
        src = f"""
module sf.eff;
fn fa(n: u64) -> u64 {{
    if n == 0_u64 {{ return {v}_u64; }}
    return fb(n - 1_u64);
}}
fn fb(n: u64) -> u64 {{
    if n == 0_u64 {{ return 1_u64; }}
    return fa(n - 1_u64);
}}
pub fn main() -> i32 {{ let _ = fa(3_u64); return 0_i32; }}
"""
        return src, {"E0113"}, "unannotated_mutual_recursion"
    # `@pure` function mutating through `*mut`
    src = f"""
module sf.eff;
@pure
pub fn bad(p: *mut u64) {{
    p.* = {v}_u64;
}}
pub fn main() -> i32 {{ return 0_i32; }}
"""
    return src, {"E0112"}, "pure_mutates_mem"


GENERATORS = [
    gen_bounds_mutation,
    gen_arith_mutation,
    gen_alias_mutation,
    gen_linear_mutation,
    gen_escape_mutation,
    gen_stale_pool_mutation,
    gen_effects_mutation,
]


def run_one(args_tuple: tuple[int, int, str, Path, Path]) -> tuple[bool, int, str, str]:
    idx, seed, eonc, work, out_dir = args_tuple
    rng = random.Random(seed)
    gen_fn = GENERATORS[idx % len(GENERATORS)]
    src, expected_codes, tag = gen_fn(rng)

    f = work / f"s{idx}.eon"
    f.write_text(src)
    try:
        p = subprocess.run([eonc, "check", str(f)], capture_output=True, text=True, timeout=30)
        combined = p.stdout + p.stderr
        caught = p.returncode != 0 and any(f"error[{c}]" in combined for c in expected_codes)
        if not caught:
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"missed_{tag}_seed_{seed}.eon"
            dst.write_text(src)
            (out_dir / f"missed_{tag}_seed_{seed}.log").write_text(
                f"expected={sorted(expected_codes)} rc={p.returncode}\n{combined}"
            )
            return (False, seed, tag, str(dst))
        return (True, seed, tag, "")
    finally:
        try:
            f.unlink()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 4, 16)))
    ap.add_argument("--eonc", default=str(DEFAULT_EONC))
    ap.add_argument("--out", default=str(REPO / "tests" / "fuzz" / "soundness_failures"))
    args = ap.parse_args()

    t0 = time.monotonic()
    eonc_abs = str(Path(args.eonc).resolve())
    work = Path(tempfile.mkdtemp(prefix="eon_soundness_"))
    out_dir = Path(args.out)
    caught = missed = 0
    try:
        tasks = [
            (i, args.seed * 1_000_003 + i, eonc_abs, work, out_dir)
            for i in range(args.n)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for ok, seed, tag, info in ex.map(run_one, tasks):
                if ok:
                    caught += 1
                else:
                    missed += 1
                    print(f"MISSED tag={tag} seed={seed} -> {info}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    dt = time.monotonic() - t0
    print(f"checks={args.n} caught={caught} missed={missed} elapsed={dt:.1f}s (jobs={args.jobs})")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
