#!/usr/bin/env python3
"""3-way Zero-C Differential Fuzzer: Reference EIR-Interp == Native x86_64 PIE == Native x86_64 --sanitize.

Generates random, valid Eon programs exercising:
- integer arithmetic over standard (i8..i64, u8..u64) and arbitrary subword widths (u5, u12, i7, i12)
- wrapping, saturating, checked (`+?`, `-?`, `*?`, `/?`, `%?`) + `catch`, shifts, bitwise, widen/trunc/clamp
- structs, `@packed` structs, `bitfield`, and `view struct` with methods
- tagged `union` + `match` + `.Variant` inference
- niche-optimized `?T` (`?*u64`, `?*[u8]`, `?bool`, `?u5`, `?i12`, `?Enum`, `?GenId`, `?fn`) + `if let` + `orelse { ... }`
- slices, arrays, `sub` / `get` / `get_mut`, `Buf` + `push_slice`
- `T ! E` error unions + `try` + `catch |e|` + `errdefer`
- `ArenaPool` (`Id[p]`) and `SlotPool` (`GenId`, `get`, `remove`)
- second-class closures (`|x| { ... }`, read-only & mutating captures) + top-level `FnRef`
- `vec[f32, 4]` (SSE2), `vec[i32, 4]`, and `f16`
- bounded `@recursive` functions

Usage:
    python3 tests/fuzz/diff_backends.py [-n 3000] [--seed 1] [--jobs N] [--eonc PATH] [--out DIR]
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

TYPES = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"]
SUBWORD_TYPES = ["u5", "u12", "i7", "i12"]
BITS = {t: int(t[1:]) for t in TYPES + SUBWORD_TYPES}


def signed(t: str) -> bool:
    return t[0] == "i"


def trange(t: str) -> tuple[int, int]:
    b = BITS[t]
    if signed(t):
        return -(1 << (b - 1)), (1 << (b - 1)) - 1
    return 0, (1 << b) - 1


RT_SRC = """module fz.rt @hal;

extern "C" fn eon_rt_print_str(s: *str);
extern "C" fn eon_rt_print_u64(v: u64);
extern "C" fn eon_rt_print_i64(v: i64);
extern "C" fn eon_rt_print_nl();

pub fn ps(s: *str) { raw { eon_rt_print_str(s); } }
pub fn pu(v: u64) { raw { eon_rt_print_u64(v); } }
pub fn pi(v: i64) { raw { eon_rt_print_i64(v); } }
pub fn nl() { raw { eon_rt_print_nl(); } }
"""


class Gen:
    def __init__(self, rng: random.Random):
        self.r = rng
        self.funcs: list[tuple[str, list[str], str]] = []

    def lit(self, t: str, avoid_extremes: bool = False) -> str:
        lo, hi = trange(t)
        if t == "i64":
            lo, hi = -(1 << 62), 1 << 62
        pick = self.r.random()
        if avoid_extremes:
            lo, hi = lo + 1, hi - 1
            v = self.r.randint(lo, hi)
        elif pick < 0.3:
            v = self.r.choice([lo, hi, 0, 1, -1 if signed(t) else 2, lo + 1, hi - 1])
        elif pick < 0.6:
            v = self.r.randint(-20 if signed(t) else 0, 20)
        else:
            v = self.r.randint(lo, hi)
        v = max(lo, min(hi, v))
        return f"{v}_{t}"

    @staticmethod
    def to_i64(e: str, t: str) -> str:
        if t == "i64":
            return e
        if t == "u64":
            return f"({e}).trunc.[i64]()"
        return f"({e}).widen.[i64]()"

    @staticmethod
    def from_i64(e: str, t: str) -> str:
        return e if t == "i64" else f"({e}).trunc.[{t}]()"

    def convert(self, e: str, src: str, dst: str) -> str:
        return e if src == dst else self.from_i64(self.to_i64(e, src), dst)

    def leaf(self, t: str, env: list[tuple[str, str]]) -> str:
        same = [n for n, ty in env if ty == t]
        if same and self.r.random() < 0.6:
            return self.r.choice(same)
        if env and self.r.random() < 0.5:
            n, ty = self.r.choice(env)
            return self.convert(n, ty, t)
        return self.lit(t)

    def expr(self, t: str, env: list[tuple[str, str]], depth: int) -> str:
        if depth <= 0 or self.r.random() < 0.25:
            return self.leaf(t, env)
        a = self.expr(t, env, depth - 1)
        k = self.r.random()
        if k < 0.45:
            op = self.r.choice(["+%", "-%", "*%", "+|", "-|", "*|"])
            return f"({a} {op} {self.expr(t, env, depth - 1)})"
        if k < 0.70:
            op = self.r.choice(["&", "|", "^"])
            return f"({a} {op} {self.expr(t, env, depth - 1)})"
        if k < 0.85:
            amt = self.r.randint(0, BITS[t] - 1)
            return f"({a} >> {amt}_u32)"
        amt = self.r.randint(0, BITS[t] - 1)
        return f"({a} <<% {amt}_u32)"

    def body(self, params: list[tuple[str, str]], ret: str) -> list[str]:
        env = list(params)
        lines: list[str] = []
        n_locals = 0

        def fresh() -> str:
            nonlocal n_locals
            n_locals += 1
            return f"t{n_locals}"

        for _ in range(self.r.randint(2, 5)):
            k = self.r.random()
            t = self.r.choice(TYPES)
            name = fresh()
            if k < 0.35:
                lines.append(f"    let {name}: {t} = {self.expr(t, env, 3)};")
            elif k < 0.55:
                op = self.r.choice(["/?", "%?", "+?", "-?", "*?"])
                a, b = self.expr(t, env, 2), self.expr(t, env, 2)
                lines.append(f"    let {name}: {t} = ({a} {op} {b}) catch {{ {self.lit(t)} }};")
            elif k < 0.70 and self.funcs:
                fname, ptys, fret = self.r.choice(self.funcs)
                args = ", ".join(self.expr(pt, env, 2) for pt in ptys)
                lines.append(f"    let {name}: {fret} = {fname}({args});")
                t = fret
            elif k < 0.85:
                cands = [(p, pt) for p, pt in params]
                pairs = [(x, y) for x in cands for y in cands if x[1] == y[1] and x[0] != y[0]]
                if pairs:
                    (x, _), (y, _) = self.r.choice(pairs)
                    cmp = self.r.choice(["<", "<=", ">", ">=", "==", "!="])
                    lines.append(f"    var {name}: {t} = {self.lit(t)};")
                    lines.append(f"    if {x} {cmp} {y} {{")
                    lines.append(f"        {name} = {self.expr(t, env, 2)};")
                    lines.append("    } else {")
                    lines.append(f"        {name} = {self.expr(t, env, 2)};")
                    lines.append("    }")
                else:
                    lines.append(f"    let {name}: {t} = {self.expr(t, env, 2)};")
            else:
                n_iter = self.r.randint(1, 5)
                lines.append(f"    var {name}: {t} = {self.lit(t)};")
                lines.append(f"    for _i in 0_usize..{n_iter}_usize {{")
                lines.append(f"        {name} = {self.expr(t, env + [(name, t)], 2)};")
                lines.append("    }")
            env.append((name, t))

        parts = [self.convert(n, ty, ret) for n, ty in env]
        acc = parts[0]
        for p in parts[1:]:
            acc = f"({acc} {self.r.choice(['^', '+%', '-%'])} {p})"
        lines.append(f"    return {acc};")
        return lines

    def function(self, idx: int) -> str:
        ret = self.r.choice(TYPES)
        ptys = [self.r.choice(TYPES) for _ in range(self.r.randint(1, 5))]
        ptys.append(ptys[0])
        params = [(f"p{i}", t) for i, t in enumerate(ptys)]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        src = [f"fn f{idx}({sig}) -> {ret} {{"] + self.body(params, ret) + ["}", ""]
        self.funcs.append((f"f{idx}", ptys, ret))
        return "\n".join(src)

    # ---- Phase 5.5 feature snippets (randomly composed into every generated program) ----

    def snippet_struct_union_niche(self, out_decls: list[str], main_lines: list[str]) -> None:
        a = self.r.randint(1, 100)
        b = self.r.randint(-50, 50)
        flag = self.r.choice(["true", "false"])
        u5v = self.r.randint(0, 31)
        i12v = self.r.randint(-2048, 2047)
        pick = self.r.randint(0, 3)
        out_decls.append("""
pub enum FzTag : u8 { Z = 0, One = 1, Two = 2 }
pub error FzErr { Boom, Empty }
pub struct FzPair { pub x: i64; pub y: u32; pub t: FzTag; }
pub union FzVal {
    Num: i64,
    Pair: FzPair,
    None,
}
fn fz_make_tag(k: u32) -> ?FzTag {
    if k == 0_u32 { return .Z; }
    if k == 1_u32 { return .One; }
    if k == 2_u32 { return .Two; }
    return null;
}
fn fz_make_val(k: u32, a: i64, b: u32) -> FzVal ! FzErr {
    if k == 0_u32 { return .Boom; }
    if k == 1_u32 { return .None; }
    if k == 2_u32 { return .Num(a); }
    return .Pair(FzPair.{ x: a, y: b, t: .Two });
}
""")
        main_lines.append(f"""
    let opt_b: ?bool = {flag};
    let bv: u64 = if let x = opt_b {{ if x {{ 11_u64 }} else {{ 22_u64 }} }} else {{ 33_u64 }};
    let opt_u5: ?u5 = {u5v}_u5;
    let opt_i12: ?i12 = {i12v}_i12;
    let u5_out: u64 = (opt_u5 orelse {{ 7_u5 }}).widen.[u64]();
    let i12_out: i64 = (opt_i12 orelse {{ -3_i12 }}).widen.[i64]();
    let tg: FzTag = fz_make_tag({pick}_u32) orelse {{ .One }};
    let tnum: u64 = match tg {{ .Z => 1_u64, .One => 2_u64, .Two => 3_u64 }};
    let fv: FzVal = fz_make_val({pick}_u32, {b}_i64, {a}_u32) catch |e| {{
        match e {{ .Boom => .Num(77_i64), .Empty => .None }}
    }};
    let fnum: i64 = match fv {{
        .Num(n) => n,
        .Pair(p) => p.x +% p.y.widen.[i64](),
        .None => -9_i64,
    }};
    rt.pu(bv +% u5_out +% tnum); rt.nl();
    rt.pi(i12_out +% fnum); rt.nl();""")

    def snippet_view_slice_buf(self, out_decls: list[str], main_lines: list[str]) -> None:
        arr = [self.r.randint(0, 500) for _ in range(6)]
        arr_lit = ", ".join(f"{x}_u64" for x in arr)
        lo = self.r.randint(0, 3)
        hi = self.r.randint(lo, 6)
        idx = self.r.randint(0, 7)
        out_decls.append(f"""
pub view struct FzScanner {{
    pub data: *[u64];
    pub pos: usize;
}}
impl FzScanner {{
    fn sum_rest(self: *mut Self) -> u64 {{
        var acc: u64 = 0_u64;
        while self.pos < self.data.len {{
            acc = acc +% self.data[self.pos];
            self.pos = self.pos + 1_usize;
        }}
        return acc;
    }}
}}
fn fz_run_view_buf(alloc: *mut AllocCap) -> u64 ! OutOfMemory | BoundsError {{
    let fz_arr: [6]u64 = [{arr_lit}];
    var fz_buf = try Buf.[u64].init(alloc, 8_usize);
    errdefer fz_buf.destroy(alloc);
    let fz_sub = try fz_arr.sub({lo}_usize, {hi}_usize);
    try fz_buf.push_slice(fz_sub);
    if let cell = fz_buf.get_mut(0_usize) {{
        cell.* = cell.* +% 5_u64;
    }}
    var sc = FzScanner.{{ data: fz_buf.as_slice(), pos: 0_usize }};
    let sc_sum = sc.sum_rest();
    let g_val: u64 = if let p = fz_arr.get({idx}_usize) {{ p.* }} else {{ 999_u64 }};
    fz_buf.destroy(alloc);
    return sc_sum +% g_val;
}}
""")
        main_lines.append("    rt.pu(try fz_run_view_buf(alloc)); rt.nl();")

    def snippet_pools(self, out_decls: list[str], main_lines: list[str]) -> None:
        v1 = self.r.randint(1, 500)
        v2 = self.r.randint(1, 500)
        v3 = self.r.randint(1, 500)
        out_decls.append(f"""
pub pool FzNode {{
    pub val: u64;
    pub next: ?Link[Self];
}}
fn fz_run_pools(alloc: *mut AllocCap) -> u64 ! OutOfMemory | MemError {{
    var fz_arena = try Arena.init(alloc, 4096_usize);
    errdefer fz_arena.destroy(alloc);
    var ap = try ArenaPool.[FzNode].init(&mut fz_arena, 16_usize);
    let n0 = try ap.push(.{{ val: {v1}_u64, next: null }});
    let n1 = try ap.push(.{{ val: {v2}_u64, next: n0 }});
    var ap_acc: u64 = ap.val[n1];
    if let prev = ap.next[n1] {{
        ap_acc = ap_acc +% ap.val[prev];
    }}

    var sp = try SlotPool.[FzNode].init(alloc, 8_usize);
    errdefer sp.destroy(alloc);
    let g0 = try sp.insert(alloc, .{{ val: {v3}_u64, next: null }});
    let g1 = try sp.insert(alloc, .{{ val: {v1}_u64, next: g0 }});
    var sp_acc: u64 = 0_u64;
    if let r = sp.get(g1) {{ sp_acc = sp_acc +% r.val; }}
    let _ = sp.remove(g0);
    if let r = sp.get(g0) {{ sp_acc = sp_acc +% r.val; }} else {{ sp_acc = sp_acc +% 13_u64; }}
    sp.destroy(alloc);
    fz_arena.destroy(alloc);
    return ap_acc +% sp_acc;
}}
""")
        main_lines.append("    rt.pu(try fz_run_pools(alloc)); rt.nl();")

    def snippet_closures_and_recursion(self, out_decls: list[str], main_lines: list[str]) -> None:
        k = self.r.randint(1, 12)
        bias = self.r.randint(1, 50)
        seed_x = self.r.randint(1, 30)
        out_decls.append("""
fn fz_apply2(f: fn(u64) -> u64, x: u64) -> u64 {
    return f(f(x));
}
fn fz_inc(x: u64) -> u64 { return x +% 3_u64; }
@recursive
fn fz_rec(n: u64, acc: u64) -> u64 {
    if n == 0_u64 { return acc; }
    return fz_rec(n - 1_u64, acc +% (n *% 2_u64));
}
""")
        main_lines.append(f"""
    let cl_bias: u64 = {bias}_u64;
    var cl_sum: u64 = 0_u64;
    let c1 = fz_apply2(fz_inc, {seed_x}_u64);
    let c2 = fz_apply2(|v: u64| {{ cl_sum = cl_sum +% v; v +% cl_bias }}, {seed_x}_u64);
    let r_out = fz_rec({k}_u64, 1_u64);
    rt.pu(c1 +% c2 +% cl_sum +% r_out); rt.nl();""")

    def snippet_packed_bitfield_f16_vec(self, out_decls: list[str], main_lines: list[str]) -> None:
        tag = self.r.randint(0, 255)
        seq = self.r.randint(0, 100000)
        m = self.r.randint(0, 7)
        irq = self.r.randint(0, 4095)
        h1 = self.r.choice(["1.5", "2.5", "3.25", "-4.5", "0.5", "6.0"])
        h2 = self.r.choice(["2.0", "0.5", "-1.5", "4.0"])
        vf = [self.r.randint(-20, 20) for _ in range(4)]
        vi = [self.r.randint(-100, 100) for _ in range(4)]
        u8s = [self.r.randint(0, 7) for _ in range(16)]
        needle = self.r.randint(0, 7)
        shuf = [self.r.randint(0, 7) for _ in range(4)]
        out_decls.append("""
@packed
pub struct FzPack {
    pub tag: u8;
    pub seq: u32;
    pub tail: u16;
}
pub bitfield FzBf : u32 {
    en: u1 @bits(0..1);
    mode: u3 @bits(1..4);
    irq: u12 @bits(4..16);
    hi: u16 @bits(16..32);
}
""")
        vf_lit = ", ".join(f"{x}.5_f32" for x in vf)
        vi_lit = ", ".join(f"{x}_i32" for x in vi)
        u8_lit = ", ".join(f"{x}_u8" for x in u8s)
        shuf_lit = ", ".join(f"{x}_u32" for x in shuf)
        main_lines.append(f"""
    var pk = FzPack.{{ tag: {tag}_u8, seq: {seq}_u32, tail: 0x1234_u16 }};
    pk.seq = pk.seq +% 7_u32;
    var bf = FzBf.{{ en: 1_u1, mode: {m}_u3, irq: {irq}_u12, hi: 9_u16 }};
    bf.mode = bf.mode ^ 3_u3;
    let fh: f16 = ({h1}_f16 * {h2}_f16) + 1.0_f16;
    let fhi: i64 = (fh * 4.0_f16).trunc_to_int.[i32]().widen.[i64]();
    let v1: f32x4 = .{{ {vf_lit} }};
    let v2: f32x4 = .{{ 1.0_f32, -2.0_f32, 3.0_f32, -4.0_f32 }};
    let vsc: f32x4 = f32x4.splat(2.0_f32);
    let v3: f32x4 = v1.fma(vsc, v2).min(v1.yxwz.max(v2));
    let vfs: i64 = (v3.reduce_add() + v1.dot(v2)).trunc_to_int.[i32]().widen.[i64]();
    let iv1: i32x4 = .{{ {vi_lit} }};
    let iv2: i32x4 = .{{ 3_i32, -5_i32, 7_i32, -11_i32 }};
    let ivm = iv1.cmplt(iv2);
    let iv_idx: u32x4 = .{{ {shuf_lit} }};
    let iv_sh = iv1.shuffle(iv2, iv_idx);
    let iv3: i32x4 = ivm.select(-(iv1 + iv2), iv_sh);
    let ivs: i64 = (iv3.reduce_add() +% iv3.reduce_min() +% iv3.reduce_max()).widen.[i64]();
    let tags: u8x16 = .{{ {u8_lit} }};
    let bm: u64 = tags.cmpeq(u8x16.splat({needle}_u8)).bitmask();
    rt.pu(pk.tag.widen.[u64]() +% pk.seq.widen.[u64]() +% bf.mode.widen.[u64]() +% bf.irq.widen.[u64]() +% bm +% ivm.bitmask()); rt.nl();
    rt.pi(fhi +% vfs +% ivs); rt.nl();""")

    def snippet_loops_soa_and_floats(self, out_decls: list[str], main_lines: list[str]) -> None:
        n = self.r.randint(12, 48)
        scale = self.r.choice(["0.25", "0.5", "1.25", "2.0", "-0.75"])
        bias = self.r.choice(["1.0", "-2.5", "0.125", "3.5"])
        div_c = self.r.choice([3, 5, 7, 9, 10, 13, 17, 100])
        seed_u = self.r.randint(1, 500)
        out_decls.append(f"""
pub pool FzBody {{
    pub x: f32;
    pub vx: f32;
    pub y: f32;
}}
fn fz_helper_mix(a: u64, b: u64) -> u64 {{
    return ((a *% 31_u64) ^ (b +% 17_u64)) +% (a / {div_c}_u64) +% (b % {div_c}_u64);
}}
fn fz_run_loops_soa(alloc: *mut AllocCap) -> u64 ! OutOfMemory | MemError {{
    var ar = try Arena.init(alloc, 16384_usize);
    errdefer ar.destroy(alloc);
    var bodies = try ArenaPool.[FzBody].init(&mut ar, {n}_usize);
    var i: usize = 0_usize;
    while i < {n}_usize {{
        let fi: f32 = i.trunc.[u32]().to_float.[f32]();
        let _ = try bodies.push(.{{ x: fi * 0.5_f32, vx: fi * {scale}_f32 + {bias}_f32, y: fi * 0.25_f32 }});
        i = i +% 1_usize;
    }}
    let dt: f32 = 0.5_f32;
    let cnt = bodies.len();
    for bi in 0_usize..cnt {{
        let b = bodies.id_at(bi) catch {{ continue; }};
        bodies.vx[b] = bodies.vx[b] + 0.25_f32;
        bodies.x[b] = bodies.x[b] + bodies.vx[b] * dt;
    }}
    for bi in 0_usize..cnt {{
        let b = bodies.id_at(bi) catch {{ continue; }};
        bodies.y[b] = bodies.y[b] + bodies.x[b] * 0.5_f32;
        if bodies.x[b] > 10.0_f32 {{
            bodies.x[b] = 10.0_f32;
            bodies.vx[b] = -bodies.vx[b];
        }}
    }}
    var acc: u64 = {seed_u}_u64;
    var k: usize = 0_usize;
    for bi in 0_usize..cnt {{
        let b = bodies.id_at(bi) catch {{ continue; }};
        let xi: i32 = (bodies.x[b] * 16.0_f32).trunc_to_int.[i32]();
        let yi: i32 = (bodies.y[b] * 16.0_f32).trunc_to_int.[i32]();
        acc = fz_helper_mix(acc, xi.widen.[i64]().bitcast.[u64]() ^ yi.widen.[i64]().bitcast.[u64]());
        k = k +% 1_usize;
    }}
    if k != cnt {{
        acc = acc ^ 0xDEAD_u64;
    }}
    ar.destroy(alloc);
    return acc;
}}
""")
        main_lines.append("    rt.pu(try fz_run_loops_soa(alloc)); rt.nl();")

    def program(self) -> str:
        out = ["module fz.main;", "", "import fz.rt;", ""]
        feature_decls: list[str] = []
        main_extra: list[str] = []

        # Pick 2..4 distinct feature blocks per program to keep compile fast while
        # exercising every single combination thoroughly across thousands of seeds!
        snippets = [
            self.snippet_struct_union_niche,
            self.snippet_view_slice_buf,
            self.snippet_pools,
            self.snippet_closures_and_recursion,
            self.snippet_packed_bitfield_f16_vec,
            self.snippet_loops_soa_and_floats,
        ]
        chosen = self.r.sample(snippets, k=self.r.randint(2, 4))
        uses_alloc = (
            (self.snippet_view_slice_buf in chosen)
            or (self.snippet_pools in chosen)
            or (self.snippet_loops_soa_and_floats in chosen)
        )
        uses_err = uses_alloc

        for fn in chosen:
            fn(feature_decls, main_extra)

        out.extend(feature_decls)
        for i in range(self.r.randint(1, 3)):
            out.append(self.function(i))

        if uses_alloc:
            out.append("pub fn main(alloc: *mut AllocCap) -> i32 ! OutOfMemory | BoundsError | MemError {")
        else:
            out.append("pub fn main() -> i32 {")
        for fname, ptys, ret in self.funcs:
            args = ", ".join(self.lit(t) for t in ptys)
            call = f"{fname}({args})"
            if signed(ret):
                out.append(f"    rt.pi({self.to_i64(call, ret)}); rt.nl();")
            elif ret == "u64":
                out.append(f"    rt.pu({call}); rt.nl();")
            else:
                out.append(f"    rt.pu(({call}).widen.[u64]()); rt.nl();")
        for chunk in main_extra:
            out.extend(chunk.splitlines())
        out.append("    return 0_i32;")
        out.append("}")
        return "\n".join(out) + "\n"


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def build_and_run(
    eonc: str,
    src_dir: Path,
    bin_path: Path,
    extra_build_flags: list[str],
    runner: list[str] | None = None,
) -> tuple[bool, int, str]:
    b_code, b_out = run_cmd(
        [eonc, "build", *extra_build_flags, "-o", str(bin_path), "."],
        src_dir,
    )
    if b_code != 0:
        return False, b_code, f"[build failed]\n{b_out}"
    exec_cmd = [*(runner or []), str(bin_path)]
    r_code, r_out = run_cmd(exec_cmd, src_dir)
    return True, r_code, r_out


def run_one(args_tuple: tuple[int, int, str, Path, Path, bool, bool, bool, bool, bool]) -> tuple[str, int, str | None]:
    idx, seed, eonc, work, out_dir, test_release, verify_opt, test_sanitize, test_aarch64, test_riscv64 = args_tuple
    src = Gen(random.Random(seed)).program()
    d = work / f"p{idx}"
    d.mkdir()
    (d / "rt.hal.eon").write_text(RT_SRC)
    (d / "main.eon").write_text(src)

    try:
        i_code, i_out = run_cmd([eonc, "run", "--interp", "."], d)
        if i_code != 0 and ("error[E" in i_out):
            return ("rejected", seed, "\n".join(i_out.strip().splitlines()[:6]))
        n_ok, n_code, n_out = build_and_run(eonc, d, d / "prog_pie", [])
        bad = (not n_ok) or ((i_code, i_out) != (n_code, n_out))

        r_ok, r_code, r_out = True, 0, ""
        if test_release:
            rel_flags = ["--release"]
            if verify_opt:
                rel_flags.append("--verify-opt")
            r_ok, r_code, r_out = build_and_run(eonc, d, d / "prog_rel", rel_flags)
            if (not r_ok) or ((i_code, i_out) != (r_code, r_out)):
                bad = True

        s_ok, s_code, s_out = True, 0, ""
        if test_sanitize:
            s_ok, s_code, s_out = build_and_run(eonc, d, d / "prog_san", ["--sanitize"])
            if (not s_ok) or ((n_code, n_out) != (s_code, s_out)):
                bad = True

        a64_dev_res = (True, 0, "")
        a64_rel_res = (True, 0, "")
        if test_aarch64:
            a64_dev_res = build_and_run(
                eonc, d, d / "prog_a64_dev", ["--dev", "--target=aarch64-linux"], ["/usr/bin/qemu-aarch64"]
            )
            if (not a64_dev_res[0]) or ((i_code, i_out) != (a64_dev_res[1], a64_dev_res[2])):
                bad = True
            if test_release:
                a64_rel_flags = ["--release", "--target=aarch64-linux"]
                if verify_opt:
                    a64_rel_flags.append("--verify-opt")
                a64_rel_res = build_and_run(
                    eonc, d, d / "prog_a64_rel", a64_rel_flags, ["/usr/bin/qemu-aarch64"]
                )
                if (not a64_rel_res[0]) or ((i_code, i_out) != (a64_rel_res[1], a64_rel_res[2])):
                    bad = True

        rv64_dev_res = (True, 0, "")
        rv64_rel_res = (True, 0, "")
        if test_riscv64:
            rv64_dev_res = build_and_run(
                eonc, d, d / "prog_rv64_dev", ["--dev", "--target=riscv64-linux"], ["/usr/bin/qemu-riscv64"]
            )
            if (not rv64_dev_res[0]) or ((i_code, i_out) != (rv64_dev_res[1], rv64_dev_res[2])):
                bad = True
            if test_release:
                rv64_rel_flags = ["--release", "--target=riscv64-linux"]
                if verify_opt:
                    rv64_rel_flags.append("--verify-opt")
                rv64_rel_res = build_and_run(
                    eonc, d, d / "prog_rv64_rel", rv64_rel_flags, ["/usr/bin/qemu-riscv64"]
                )
                if (not rv64_rel_res[0]) or ((i_code, i_out) != (rv64_rel_res[1], rv64_rel_res[2])):
                    bad = True

        if bad:
            dst = out_dir / f"seed_{seed}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(d / "main.eon", dst / "main.eon")
            shutil.copy(d / "rt.hal.eon", dst / "rt.hal.eon")
            (dst / "interp.out").write_text(f"exit={i_code}\n{i_out}")
            (dst / "native_pie.out").write_text(f"exit={n_code}\n{n_out}")
            if test_release:
                (dst / "native_rel.out").write_text(f"exit={r_code}\n{r_out}")
            if test_sanitize:
                (dst / "native_san.out").write_text(f"exit={s_code}\n{s_out}")
            if test_aarch64:
                (dst / "aarch64_dev.out").write_text(f"exit={a64_dev_res[1]}\n{a64_dev_res[2]}")
                if test_release:
                    (dst / "aarch64_rel.out").write_text(f"exit={a64_rel_res[1]}\n{a64_rel_res[2]}")
            if test_riscv64:
                (dst / "riscv64_dev.out").write_text(f"exit={rv64_dev_res[1]}\n{rv64_dev_res[2]}")
                if test_release:
                    (dst / "riscv64_rel.out").write_text(f"exit={rv64_rel_res[1]}\n{rv64_rel_res[2]}")
            return ("mismatch", seed, str(dst))
        return ("ok", seed, None)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 4, 16)))
    ap.add_argument("--eonc", default=str(DEFAULT_EONC))
    ap.add_argument("--out", default=str(REPO / "tests" / "fuzz" / "failures"))
    ap.add_argument("--release", action="store_true", help="Include --release optimization leg")
    ap.add_argument("--verify-opt", action="store_true", help="Pass --verify-opt when building --release")
    ap.add_argument("--no-sanitize", action="store_true", help="Skip --sanitize leg")
    ap.add_argument("--aarch64", action="store_true", help="Include AArch64 cross-compile + qemu-aarch64 leg")
    ap.add_argument("--riscv64", action="store_true", help="Include RISC-V 64 cross-compile + qemu-riscv64 leg")
    args = ap.parse_args()

    test_sanitize = not args.no_sanitize
    if (args.release or args.aarch64 or args.riscv64) and not args.no_sanitize:
        test_sanitize = False

    t0 = time.monotonic()
    eonc_abs = str(Path(args.eonc).resolve())
    work = Path(tempfile.mkdtemp(prefix="eon_fuzz3_"))
    out_dir = Path(args.out)
    ok = mismatches = rejected = 0
    first_reject = None
    try:
        tasks = [
            (
                i,
                args.seed * 1_000_003 + i,
                eonc_abs,
                work,
                out_dir,
                args.release,
                args.verify_opt,
                test_sanitize,
                args.aarch64,
                args.riscv64,
            )
            for i in range(args.n)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for status, seed, info in ex.map(run_one, tasks):
                if status == "ok":
                    ok += 1
                elif status == "rejected":
                    rejected += 1
                    if first_reject is None:
                        first_reject = (seed, info)
                else:
                    mismatches += 1
                    print(f"MISMATCH seed={seed} -> {info}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    dt = time.monotonic() - t0
    legs = ["interp", "x64_dev"]
    if args.release:
        legs.append("x64_rel")
    if test_sanitize:
        legs.append("x64_san")
    if args.aarch64:
        legs.append("a64_dev+rel" if args.release else "a64_dev")
    if args.riscv64:
        legs.append("rv64_dev+rel" if args.release else "rv64_dev")
    mode_tag = "==".join(legs)
    print(
        f"programs={args.n} mode={mode_tag} agree={ok} mismatch={mismatches} rejected={rejected} elapsed={dt:.1f}s (jobs={args.jobs})"
    )
    if first_reject:
        print(f"first rejection (seed {first_reject[0]}):\n{first_reject[1]}")
    return 1 if (mismatches or rejected) else 0


if __name__ == "__main__":
    sys.exit(main())
