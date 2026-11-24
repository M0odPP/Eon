#!/usr/bin/env python3
"""Deterministic 30,000+ LOC multi-module codebase generator & compile-speed benchmark
for Eon vs safe Rust (`bench/12_compile_speed`).

Generates 20 interconnected modules (`mod00`..`mod19`) + `main` (~31,000 LOC each)
in both Eon (`gen_eon/`) and safe Rust (`gen_rs/`) with identical structs, enums,
arithmetic, loops, state machines, cross-module imports, and deterministic checksums.

Usage:
  python3 gen_codebase.py                       # Generate gen_eon/ and gen_rs/
  python3 gen_codebase.py --incremental         # Benchmark cold + 1-fn incremental rebuilds
  python3 gen_codebase.py --mutate-one --delta 1
  python3 gen_codebase.py --clean               # Remove generated trees & caches
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

NUM_MODULES = 20
TARGET_MOD_IDX = 10  # Middle of the 20-module DAG (imported by mod11, mod12, main)
MASK64 = (1 << 64) - 1
RESULT_RE = re.compile(r"^RESULT\s+(\S+)\s+checksum=(\d+)\s+elapsed_ns=(\d+)\s*$")


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def det_const(mod_idx: int, fn_idx: int, salt: int) -> int:
    h = splitmix64((mod_idx << 32) ^ (fn_idx << 16) ^ salt ^ 0x243F6A8885A308D3)
    return (h | 1) & MASK64


def det_shift(mod_idx: int, fn_idx: int, salt: int, lo: int = 3, hi: int = 29) -> int:
    h = splitmix64((mod_idx << 24) ^ (fn_idx << 8) ^ salt ^ 0x13198A2E03707344)
    return lo + (h % (hi - lo + 1))


def u64_eon(v: int) -> str:
    return f"0x{v:016X}_u64"


def u64_rs(v: int) -> str:
    return f"0x{v:016X}u64"


def emit_common_types(lang: str, mod_idx: int) -> List[str]:
    is_eon = lang == "eon"
    lines: List[str] = []
    if is_eon:
        lines.append(f"module bench.cs.mod{mod_idx:02d};")
        lines.append("")
        if mod_idx >= 1:
            lines.append(f"import bench.cs.mod{mod_idx - 1:02d};")
        if mod_idx >= 2:
            lines.append(f"import bench.cs.mod{mod_idx - 2:02d};")
        if mod_idx >= 1:
            lines.append("")
        lines.extend([
            "pub enum OpMode : u8 {",
            "    Alpha = 0,",
            "    Beta = 1,",
            "    Gamma = 2,",
            "    Delta = 3,",
            "    Epsilon = 4,",
            "    Zeta = 5,",
            "}",
            "",
            "pub enum RouteKind : u8 {",
            "    Direct = 0,",
            "    Fold = 1,",
            "    Twist = 2,",
            "    Cascade = 3,",
            "}",
            "",
            "pub struct Record {",
            "    pub a: u64;",
            "    pub b: u64;",
            "    pub c: u32;",
            "    pub d: u32;",
            "    pub flags: u32;",
            "    pub tag: u32;",
            "}",
            "",
            "pub struct Metrics {",
            "    pub sum: u64;",
            "    pub xor_acc: u64;",
            "    pub count: u32;",
            "    pub mode: u32;",
            "}",
            "",
            "pub struct Window {",
            "    pub lanes: [8]u64;",
            "    pub cursor: u32;",
            "    pub stride: u32;",
            "}",
            "",
            "pub struct PipeState {",
            "    pub rec: Record;",
            "    pub met: Metrics;",
            "    pub score: u64;",
            "}",
            "",
        ])
    else:
        if mod_idx >= 1:
            lines.append(f"use crate::mod{mod_idx - 1:02d};")
        if mod_idx >= 2:
            lines.append(f"use crate::mod{mod_idx - 2:02d};")
        if mod_idx >= 1:
            lines.append("")
        lines.extend([
            "#[derive(Clone, Copy)]",
            "#[repr(u8)]",
            "pub enum OpMode {",
            "    Alpha = 0,",
            "    Beta = 1,",
            "    Gamma = 2,",
            "    Delta = 3,",
            "    Epsilon = 4,",
            "    Zeta = 5,",
            "}",
            "",
            "#[derive(Clone, Copy)]",
            "#[repr(u8)]",
            "pub enum RouteKind {",
            "    Direct = 0,",
            "    Fold = 1,",
            "    Twist = 2,",
            "    Cascade = 3,",
            "}",
            "",
            "#[derive(Clone, Copy)]",
            "pub struct Record {",
            "    pub a: u64,",
            "    pub b: u64,",
            "    pub c: u32,",
            "    pub d: u32,",
            "    pub flags: u32,",
            "    pub tag: u32,",
            "}",
            "",
            "#[derive(Clone, Copy)]",
            "pub struct Metrics {",
            "    pub sum: u64,",
            "    pub xor_acc: u64,",
            "    pub count: u32,",
            "    pub mode: u32,",
            "}",
            "",
            "#[derive(Clone, Copy)]",
            "pub struct Window {",
            "    pub lanes: [u64; 8],",
            "    pub cursor: u32,",
            "    pub stride: u32,",
            "}",
            "",
            "#[derive(Clone, Copy)]",
            "pub struct PipeState {",
            "    pub rec: Record,",
            "    pub met: Metrics,",
            "    pub score: u64,",
            "}",
            "",
        ])
    return lines


def emit_helpers(lang: str, mod_idx: int) -> List[str]:
    c1 = det_const(mod_idx, 0, 1)
    c2 = det_const(mod_idx, 0, 2)
    c3 = det_const(mod_idx, 0, 3)
    if lang == "eon":
        return [
            "fn classify_mode(v: u64) -> OpMode {",
            "    let rem = v & 7_u64;",
            "    if rem == 0_u64 { return .Alpha; }",
            "    if rem == 1_u64 { return .Beta; }",
            "    if rem == 2_u64 { return .Gamma; }",
            "    if rem == 3_u64 { return .Delta; }",
            "    if rem == 4_u64 { return .Epsilon; }",
            "    return .Zeta;",
            "}",
            "",
            "fn classify_route(v: u64) -> RouteKind {",
            "    let rem = (v >> 3_u64) & 3_u64;",
            "    if rem == 0_u64 { return .Direct; }",
            "    if rem == 1_u64 { return .Fold; }",
            "    if rem == 2_u64 { return .Twist; }",
            "    return .Cascade;",
            "}",
            "",
            "fn mix_local(x: u64) -> u64 {",
            f"    var z = x +% {u64_eon(c1)};",
            f"    z = (z ^ (z >> 30_u64)) *% {u64_eon(c2)};",
            f"    z = (z ^ (z >> 27_u64)) *% {u64_eon(c3)};",
            "    return z ^ (z >> 31_u64);",
            "}",
            "",
            "fn make_record(seed: u64) -> Record {",
            "    let h1 = mix_local(seed);",
            "    let h2 = mix_local(h1 ^ 0x517CC1B727220A95_u64);",
            "    return Record.{",
            "        a: h1,",
            "        b: h2,",
            "        c: (h1 & 0xFFFFFFFF_u64).trunc.[u32](),",
            "        d: ((h2 >> 17_u64) & 0xFFFFFFFF_u64).trunc.[u32](),",
            "        flags: ((h1 >> 32_u64) & 0xFFFF_u64).trunc.[u32](),",
            "        tag: ((h2 >> 40_u64) & 0xFFFF_u64).trunc.[u32](),",
            "    };",
            "}",
            "",
            "fn make_metrics(seed: u64) -> Metrics {",
            "    let h = mix_local(seed ^ 0xA0761D6478BD642F_u64);",
            "    return Metrics.{",
            "        sum: h,",
            "        xor_acc: h ^ (h >> 19_u64),",
            "        count: ((h & 255_u64) +% 1_u64).trunc.[u32](),",
            "        mode: ((h >> 8_u64) & 7_u64).trunc.[u32](),",
            "    };",
            "}",
            "",
        ]
    else:
        return [
            "fn classify_mode(v: u64) -> OpMode {",
            "    let rem = v & 7u64;",
            "    if rem == 0u64 { return OpMode::Alpha; }",
            "    if rem == 1u64 { return OpMode::Beta; }",
            "    if rem == 2u64 { return OpMode::Gamma; }",
            "    if rem == 3u64 { return OpMode::Delta; }",
            "    if rem == 4u64 { return OpMode::Epsilon; }",
            "    OpMode::Zeta",
            "}",
            "",
            "fn classify_route(v: u64) -> RouteKind {",
            "    let rem = (v >> 3u64) & 3u64;",
            "    if rem == 0u64 { return RouteKind::Direct; }",
            "    if rem == 1u64 { return RouteKind::Fold; }",
            "    if rem == 2u64 { return RouteKind::Twist; }",
            "    RouteKind::Cascade",
            "}",
            "",
            "fn mix_local(x: u64) -> u64 {",
            f"    let mut z = x.wrapping_add({u64_rs(c1)});",
            f"    z = (z ^ (z >> 30u64)).wrapping_mul({u64_rs(c2)});",
            f"    z = (z ^ (z >> 27u64)).wrapping_mul({u64_rs(c3)});",
            "    z ^ (z >> 31u64)",
            "}",
            "",
            "fn make_record(seed: u64) -> Record {",
            "    let h1 = mix_local(seed);",
            "    let h2 = mix_local(h1 ^ 0x517CC1B727220A95u64);",
            "    Record {",
            "        a: h1,",
            "        b: h2,",
            "        c: (h1 & 0xFFFFFFFFu64) as u32,",
            "        d: ((h2 >> 17u64) & 0xFFFFFFFFu64) as u32,",
            "        flags: ((h1 >> 32u64) & 0xFFFFu64) as u32,",
            "        tag: ((h2 >> 40u64) & 0xFFFFu64) as u32,",
            "    }",
            "}",
            "",
            "fn make_metrics(seed: u64) -> Metrics {",
            "    let h = mix_local(seed ^ 0xA0761D6478BD642Fu64);",
            "    Metrics {",
            "        sum: h,",
            "        xor_acc: h ^ (h >> 19u64),",
            "        count: ((h & 255u64).wrapping_add(1u64)) as u32,",
            "        mode: ((h >> 8u64) & 7u64) as u32,",
            "    }",
            "}",
            "",
        ]


def emit_bitmix_kernel(lang: str, mod_idx: int, fn_idx: int) -> Tuple[str, List[str]]:
    name = f"bitmix_kernel_{fn_idx:02d}"
    k1 = det_const(mod_idx, fn_idx, 11)
    k2 = det_const(mod_idx, fn_idx, 12)
    k3 = det_const(mod_idx, fn_idx, 13)
    k4 = det_const(mod_idx, fn_idx, 14)
    s1 = det_shift(mod_idx, fn_idx, 15, 5, 25)
    s2 = det_shift(mod_idx, fn_idx, 16, 7, 27)
    s3 = det_shift(mod_idx, fn_idx, 17, 9, 31)
    s4 = det_shift(mod_idx, fn_idx, 18, 4, 21)
    if lang == "eon":
        lines = [
            f"fn {name}(x: u64, y: u64) -> u64 {{",
            f"    var a = (x ^ {u64_eon(k1)}) *% {u64_eon(k2)};",
            f"    var b = (y +% {u64_eon(k3)}) ^ {u64_eon(k4)};",
            f"    let r0 = (a << {s1}_u64) | (a >> {64 - s1}_u64);",
            f"    let r1 = (b << {s2}_u64) | (b >> {64 - s2}_u64);",
            "    a = (a +% r1) ^ (r0 >> 7_u64);",
            "    b = (b -% r0) +% (r1 >> 11_u64);",
            "    if (a & 1_u64) != 0_u64 {",
            f"        a = (a ^ (b >> {s3}_u64)) *% {u64_eon(k2)};",
            "        b = b +% (a & 0xFFFF_u64);",
            "    } else {",
            f"        b = (b ^ (a >> {s4}_u64)) *% {u64_eon(k4)};",
            "        a = a -% (b & 0xFFFF_u64);",
            "    }",
            "    let rec = make_record(a ^ b);",
            "    let c_w = rec.c.widen.[u64]();",
            "    let d_w = rec.d.widen.[u64]();",
            "    let fl = rec.flags.widen.[u64]();",
            "    let tg = rec.tag.widen.[u64]();",
            "    var out = (rec.a ^ a) +% (rec.b ^ b);",
            f"    out = (out ^ ((c_w << {s4}_u64) +% d_w)) *% {u64_eon(k1)};",
            "    out = out +% ((fl << 16_u64) ^ tg);",
            "    if (out & 4_u64) != 0_u64 {",
            f"        out = out ^ (out >> {s2}_u64);",
            "    } else {",
            f"        out = out +% (out >> {s1}_u64);",
            "    }",
            "    return out ^ (out >> 29_u64);",
            "}",
            "",
        ]
    else:
        lines = [
            f"fn {name}(x: u64, y: u64) -> u64 {{",
            f"    let mut a = (x ^ {u64_rs(k1)}).wrapping_mul({u64_rs(k2)});",
            f"    let mut b = (y.wrapping_add({u64_rs(k3)})) ^ {u64_rs(k4)};",
            f"    let r0 = (a << {s1}u64) | (a >> {64 - s1}u64);",
            f"    let r1 = (b << {s2}u64) | (b >> {64 - s2}u64);",
            "    a = a.wrapping_add(r1) ^ (r0 >> 7u64);",
            "    b = b.wrapping_sub(r0).wrapping_add(r1 >> 11u64);",
            "    if (a & 1u64) != 0u64 {",
            f"        a = (a ^ (b >> {s3}u64)).wrapping_mul({u64_rs(k2)});",
            "        b = b.wrapping_add(a & 0xFFFFu64);",
            "    } else {",
            f"        b = (b ^ (a >> {s4}u64)).wrapping_mul({u64_rs(k4)});",
            "        a = a.wrapping_sub(b & 0xFFFFu64);",
            "    }",
            "    let rec = make_record(a ^ b);",
            "    let c_w = rec.c as u64;",
            "    let d_w = rec.d as u64;",
            "    let fl = rec.flags as u64;",
            "    let tg = rec.tag as u64;",
            "    let mut out = (rec.a ^ a).wrapping_add(rec.b ^ b);",
            f"    out = (out ^ ((c_w << {s4}u64).wrapping_add(d_w))).wrapping_mul({u64_rs(k1)});",
            "    out = out.wrapping_add((fl << 16u64) ^ tg);",
            "    if (out & 4u64) != 0u64 {",
            f"        out = out ^ (out >> {s2}u64);",
            "    } else {",
            f"        out = out.wrapping_add(out >> {s1}u64);",
            "    }",
            "    out ^ (out >> 29u64)",
            "}",
            "",
        ]
    return name, lines


def emit_fsm_kernel(lang: str, mod_idx: int, fn_idx: int) -> Tuple[str, List[str]]:
    name = f"fsm_dispatch_{fn_idx:02d}"
    k1 = det_const(mod_idx, fn_idx, 21)
    k2 = det_const(mod_idx, fn_idx, 22)
    k3 = det_const(mod_idx, fn_idx, 23)
    s1 = det_shift(mod_idx, fn_idx, 24, 3, 21)
    s2 = det_shift(mod_idx, fn_idx, 25, 5, 25)
    steps = 4 + (det_shift(mod_idx, fn_idx, 26, 0, 4))
    if lang == "eon":
        lines = [
            f"fn {name}(seed: u64) -> u64 {{",
            "    var rec = make_record(seed);",
            "    var met = make_metrics(seed ^ 0xD1B54A32D192ED03_u64);",
            "    var cur = seed;",
            f"    for step in 0_usize..{steps}_usize {{",
            "        let mode = classify_mode(cur +% step.widen.[u64]());",
            "        let route = classify_route(cur ^ rec.a);",
            "        match mode {",
            "            .Alpha => {",
            f"                rec.a = (rec.a +% cur) ^ {u64_eon(k1)};",
            f"                met.sum = met.sum +% (rec.a >> {s1}_u64);",
            "            },",
            "            .Beta => {",
            f"                rec.b = (rec.b *% {u64_eon(k2)}) -% cur;",
            "                met.xor_acc = met.xor_acc ^ rec.b;",
            "            },",
            "            .Gamma => {",
            f"                rec.a = rec.a ^ (rec.b >> {s2}_u64);",
            "                rec.c = rec.c +% 17_u32;",
            "                met.sum = met.sum +% rec.c.widen.[u64]();",
            "            },",
            "            .Delta => {",
            f"                rec.b = rec.b +% (rec.a << {s1}_u64);",
            "                rec.d = rec.d ^ 0x5A3C_u32;",
            "                met.xor_acc = met.xor_acc ^ rec.d.widen.[u64]();",
            "            },",
            "            .Epsilon => {",
            f"                let t = (rec.a ^ rec.b) *% {u64_eon(k3)};",
            "                rec.a = t;",
            "                met.sum = met.sum ^ t;",
            "            },",
            "            .Zeta => {",
            "                rec.flags = rec.flags +% rec.tag +% 3_u32;",
            "                met.sum = met.sum +% rec.flags.widen.[u64]();",
            "            },",
            "        }",
            "        match route {",
            "            .Direct => { cur = (cur +% rec.a) ^ met.sum; },",
            f"            .Fold => {{ cur = (cur ^ rec.b) *% {u64_eon(k1)}; }},",
            f"            .Twist => {{ cur = (cur >> {s1}_u64) +% met.xor_acc; }},",
            f"            .Cascade => {{ cur = (cur +% met.sum) ^ (rec.a >> {s2}_u64); }},",
            "        }",
            "        met.count = met.count +% 1_u32;",
            "    }",
            "    return (cur ^ met.sum) +% met.xor_acc +% met.count.widen.[u64]();",
            "}",
            "",
        ]
    else:
        lines = [
            f"fn {name}(seed: u64) -> u64 {{",
            "    let mut rec = make_record(seed);",
            "    let mut met = make_metrics(seed ^ 0xD1B54A32D192ED03u64);",
            "    let mut cur = seed;",
            f"    for step in 0usize..{steps}usize {{",
            "        let mode = classify_mode(cur.wrapping_add(step as u64));",
            "        let route = classify_route(cur ^ rec.a);",
            "        match mode {",
            "            OpMode::Alpha => {",
            f"                rec.a = rec.a.wrapping_add(cur) ^ {u64_rs(k1)};",
            f"                met.sum = met.sum.wrapping_add(rec.a >> {s1}u64);",
            "            }",
            "            OpMode::Beta => {",
            f"                rec.b = rec.b.wrapping_mul({u64_rs(k2)}).wrapping_sub(cur);",
            "                met.xor_acc = met.xor_acc ^ rec.b;",
            "            }",
            "            OpMode::Gamma => {",
            f"                rec.a = rec.a ^ (rec.b >> {s2}u64);",
            "                rec.c = rec.c.wrapping_add(17u32);",
            "                met.sum = met.sum.wrapping_add(rec.c as u64);",
            "            }",
            "            OpMode::Delta => {",
            f"                rec.b = rec.b.wrapping_add(rec.a << {s1}u64);",
            "                rec.d = rec.d ^ 0x5A3Cu32;",
            "                met.xor_acc = met.xor_acc ^ (rec.d as u64);",
            "            }",
            "            OpMode::Epsilon => {",
            f"                let t = (rec.a ^ rec.b).wrapping_mul({u64_rs(k3)});",
            "                rec.a = t;",
            "                met.sum = met.sum ^ t;",
            "            }",
            "            OpMode::Zeta => {",
            "                rec.flags = rec.flags.wrapping_add(rec.tag).wrapping_add(3u32);",
            "                met.sum = met.sum.wrapping_add(rec.flags as u64);",
            "            }",
            "        }",
            "        match route {",
            "            RouteKind::Direct => { cur = cur.wrapping_add(rec.a) ^ met.sum; }",
            f"            RouteKind::Fold => {{ cur = (cur ^ rec.b).wrapping_mul({u64_rs(k1)}); }}",
            f"            RouteKind::Twist => {{ cur = (cur >> {s1}u64).wrapping_add(met.xor_acc); }}",
            f"            RouteKind::Cascade => {{ cur = cur.wrapping_add(met.sum) ^ (rec.a >> {s2}u64); }}",
            "        }",
            "        met.count = met.count.wrapping_add(1u32);",
            "    }",
            "    (cur ^ met.sum).wrapping_add(met.xor_acc).wrapping_add(met.count as u64)",
            "}",
            "",
        ]
    return name, lines


def emit_window_kernel(lang: str, mod_idx: int, fn_idx: int) -> Tuple[str, List[str]]:
    name = f"window_filter_{fn_idx:02d}"
    k1 = det_const(mod_idx, fn_idx, 31)
    k2 = det_const(mod_idx, fn_idx, 32)
    s1 = det_shift(mod_idx, fn_idx, 33, 3, 19)
    s2 = det_shift(mod_idx, fn_idx, 34, 7, 27)
    rounds = 2 + (det_shift(mod_idx, fn_idx, 35, 0, 3))
    if lang == "eon":
        lines = [
            f"fn {name}(seed: u64) -> u64 {{",
            "    var win = Window.{",
            "        lanes: [0_u64; 8],",
            "        cursor: 0_u32,",
            "        stride: 3_u32,",
            "    };",
            "    var gen = seed;",
            "    for i in 0_usize..8_usize {",
            f"        gen = mix_local(gen +% (i.widen.[u64]() *% {u64_eon(k1)}));",
            "        win.lanes[i] = gen;",
            "    }",
            f"    for r in 0_usize..{rounds}_usize {{",
            "        let ru = r.widen.[u64]();",
            "        for i in 0_usize..8_usize {",
            "            let nxt = (i +% 1_usize) & 7_usize;",
            "            let prv = (i +% 7_usize) & 7_usize;",
            "            let cur_v = win.lanes[i];",
            "            let nxt_v = win.lanes[nxt];",
            "            let prv_v = win.lanes[prv];",
            "            var blended = (cur_v +% (nxt_v >> 3_u64)) ^ (prv_v << 2_u64);",
            "            if ((cur_v ^ ru) & 2_u64) != 0_u64 {",
            f"                blended = (blended *% {u64_eon(k2)}) ^ (blended >> {s1}_u64);",
            "            } else {",
            f"                blended = blended +% (nxt_v >> {s2}_u64);",
            "            }",
            "            win.lanes[i] = blended;",
            "        }",
            "        win.cursor = (win.cursor +% win.stride) & 7_u32;",
            "    }",
            "    var sum: u64 = 0_u64;",
            "    var xr: u64 = 0_u64;",
            "    for i in 0_usize..8_usize {",
            "        let v = win.lanes[i];",
            "        sum = sum +% (v ^ (i.widen.[u64]() << 5_u64));",
            "        xr = xr ^ (v +% sum);",
            "    }",
            "    let c_idx = (win.cursor & 7_u32).widen.[usize]();",
            "    let pivot = win.lanes[c_idx];",
            "    return (sum ^ xr) +% pivot;",
            "}",
            "",
        ]
    else:
        lines = [
            f"fn {name}(seed: u64) -> u64 {{",
            "    let mut win = Window {",
            "        lanes: [0u64; 8],",
            "        cursor: 0u32,",
            "        stride: 3u32,",
            "    };",
            "    let mut gen = seed;",
            "    for i in 0usize..8usize {",
            f"        gen = mix_local(gen.wrapping_add((i as u64).wrapping_mul({u64_rs(k1)})));",
            "        win.lanes[i] = gen;",
            "    }",
            f"    for r in 0usize..{rounds}usize {{",
            "        let ru = r as u64;",
            "        for i in 0usize..8usize {",
            "            let nxt = (i.wrapping_add(1usize)) & 7usize;",
            "            let prv = (i.wrapping_add(7usize)) & 7usize;",
            "            let cur_v = win.lanes[i];",
            "            let nxt_v = win.lanes[nxt];",
            "            let prv_v = win.lanes[prv];",
            "            let mut blended = cur_v.wrapping_add(nxt_v >> 3u64) ^ (prv_v << 2u64);",
            "            if ((cur_v ^ ru) & 2u64) != 0u64 {",
            f"                blended = blended.wrapping_mul({u64_rs(k2)}) ^ (blended >> {s1}u64);",
            "            } else {",
            f"                blended = blended.wrapping_add(nxt_v >> {s2}u64);",
            "            }",
            "            win.lanes[i] = blended;",
            "        }",
            "        win.cursor = win.cursor.wrapping_add(win.stride) & 7u32;",
            "    }",
            "    let mut sum: u64 = 0u64;",
            "    let mut xr: u64 = 0u64;",
            "    for i in 0usize..8usize {",
            "        let v = win.lanes[i];",
            "        sum = sum.wrapping_add(v ^ ((i as u64) << 5u64));",
            "        xr = xr ^ v.wrapping_add(sum);",
            "    }",
            "    let c_idx = (win.cursor & 7u32) as usize;",
            "    let pivot = win.lanes[c_idx];",
            "    (sum ^ xr).wrapping_add(pivot)",
            "}",
            "",
        ]
    return name, lines


def emit_loop_evolve_kernel(lang: str, mod_idx: int, fn_idx: int) -> Tuple[str, List[str]]:
    name = f"loop_evolve_{fn_idx:02d}"
    k1 = det_const(mod_idx, fn_idx, 41)
    k2 = det_const(mod_idx, fn_idx, 42)
    s1 = det_shift(mod_idx, fn_idx, 43, 4, 20)
    s2 = det_shift(mod_idx, fn_idx, 44, 6, 26)
    iters = 10 + (det_shift(mod_idx, fn_idx, 45, 0, 8))
    if lang == "eon":
        lines = [
            f"fn {name}(state: *mut PipeState, salt: u64) -> u64 {{",
            "    var i: u64 = 0_u64;",
            "    var acc: u64 = state.score ^ salt;",
            "    var prev: u64 = state.rec.a;",
            f"    while i < {iters}_u64 {{",
            "        let sel = (acc +% i) & 3_u64;",
            "        if sel == 0_u64 {",
            f"            acc = (acc *% {u64_eon(k1)}) +% (prev >> {s1}_u64);",
            "            state.rec.a = state.rec.a ^ acc;",
            "        } else if sel == 1_u64 {",
            f"            acc = (acc -% prev) ^ {u64_eon(k2)};",
            "            state.rec.b = state.rec.b +% acc;",
            "        } else if sel == 2_u64 {",
            f"            acc = (acc >> {s2}_u64) +% (state.met.sum ^ salt);",
            "            state.met.xor_acc = state.met.xor_acc ^ acc;",
            "        } else {",
            f"            acc = (acc ^ state.rec.b) +% (i *% {u64_eon(k1)});",
            "            state.met.sum = state.met.sum +% (acc & 0xFFFFFF_u64);",
            "        }",
            "        prev = (prev ^ acc) +% 0x9E37_u64;",
            "        if (acc & 0xFFF_u64) == 0x5A5_u64 {",
            "            break;",
            "        }",
            "        i = i +% 1_u64;",
            "    }",
            "    state.score = (state.score +% acc) ^ prev;",
            "    state.met.count = state.met.count +% i.trunc.[u32]();",
            "    return state.score ^ state.met.sum ^ state.met.xor_acc;",
            "}",
            "",
        ]
    else:
        lines = [
            f"fn {name}(state: &mut PipeState, salt: u64) -> u64 {{",
            "    let mut i: u64 = 0u64;",
            "    let mut acc: u64 = state.score ^ salt;",
            "    let mut prev: u64 = state.rec.a;",
            f"    while i < {iters}u64 {{",
            "        let sel = acc.wrapping_add(i) & 3u64;",
            "        if sel == 0u64 {",
            f"            acc = acc.wrapping_mul({u64_rs(k1)}).wrapping_add(prev >> {s1}u64);",
            "            state.rec.a = state.rec.a ^ acc;",
            "        } else if sel == 1u64 {",
            f"            acc = acc.wrapping_sub(prev) ^ {u64_rs(k2)};",
            "            state.rec.b = state.rec.b.wrapping_add(acc);",
            "        } else if sel == 2u64 {",
            f"            acc = (acc >> {s2}u64).wrapping_add(state.met.sum ^ salt);",
            "            state.met.xor_acc = state.met.xor_acc ^ acc;",
            "        } else {",
            f"            acc = (acc ^ state.rec.b).wrapping_add(i.wrapping_mul({u64_rs(k1)}));",
            "            state.met.sum = state.met.sum.wrapping_add(acc & 0xFFFFFFu64);",
            "        }",
            "        prev = (prev ^ acc).wrapping_add(0x9E37u64);",
            "        if (acc & 0xFFFu64) == 0x5A5u64 {",
            "            break;",
            "        }",
            "        i = i.wrapping_add(1u64);",
            "    }",
            "    state.score = state.score.wrapping_add(acc) ^ prev;",
            "    state.met.count = state.met.count.wrapping_add(i as u32);",
            "    state.score ^ state.met.sum ^ state.met.xor_acc",
            "}",
            "",
        ]
    return name, lines


def emit_pipe_combinator(lang: str, mod_idx: int, fn_idx: int) -> Tuple[str, List[str]]:
    name = f"pipe_combine_{fn_idx:02d}"
    k1 = det_const(mod_idx, fn_idx, 51)
    k2 = det_const(mod_idx, fn_idx, 52)
    s1 = det_shift(mod_idx, fn_idx, 53, 3, 22)
    s2 = det_shift(mod_idx, fn_idx, 54, 5, 28)
    if lang == "eon":
        lines = [
            f"fn {name}(r: Record, m: Metrics) -> PipeState {{",
            f"    let a1 = (r.a ^ m.sum) *% {u64_eon(k1)};",
            f"    let b1 = (r.b +% m.xor_acc) ^ {u64_eon(k2)};",
            "    let c1 = (r.c +% m.count) ^ (r.flags << 1_u32);",
            "    let d1 = (r.d -% m.mode) +% (r.tag ^ 0x1357_u32);",
            "    var score = (a1 ^ (b1 >> 13_u64)) +% (c1.widen.[u64]() << 7_u64);",
            "    score = score ^ (d1.widen.[u64]() *% 0x9E3779B9_u64);",
            "    if (score & 8_u64) != 0_u64 {",
            f"        score = (score +% a1) ^ (score >> {s1}_u64);",
            "    } else {",
            f"        score = (score -% b1) +% (score >> {s2}_u64);",
            "    }",
            "    let new_rec = Record.{",
            "        a: a1 ^ score,",
            "        b: b1 +% score,",
            "        c: c1,",
            "        d: d1,",
            "        flags: r.flags ^ 0xAA_u32,",
            "        tag: r.tag +% 5_u32,",
            "    };",
            "    let new_met = Metrics.{",
            "        sum: m.sum +% score,",
            "        xor_acc: m.xor_acc ^ a1 ^ b1,",
            "        count: m.count +% 2_u32,",
            "        mode: (m.mode +% 1_u32) & 7_u32,",
            "    };",
            "    return PipeState.{ rec: new_rec, met: new_met, score: score };",
            "}",
            "",
        ]
    else:
        lines = [
            f"fn {name}(r: Record, m: Metrics) -> PipeState {{",
            f"    let a1 = (r.a ^ m.sum).wrapping_mul({u64_rs(k1)});",
            f"    let b1 = r.b.wrapping_add(m.xor_acc) ^ {u64_rs(k2)};",
            "    let c1 = r.c.wrapping_add(m.count) ^ (r.flags << 1u32);",
            "    let d1 = r.d.wrapping_sub(m.mode).wrapping_add(r.tag ^ 0x1357u32);",
            "    let mut score = (a1 ^ (b1 >> 13u64)).wrapping_add((c1 as u64) << 7u64);",
            "    score = score ^ (d1 as u64).wrapping_mul(0x9E3779B9u64);",
            "    if (score & 8u64) != 0u64 {",
            f"        score = score.wrapping_add(a1) ^ (score >> {s1}u64);",
            "    } else {",
            f"        score = score.wrapping_sub(b1).wrapping_add(score >> {s2}u64);",
            "    }",
            "    let new_rec = Record {",
            "        a: a1 ^ score,",
            "        b: b1.wrapping_add(score),",
            "        c: c1,",
            "        d: d1,",
            "        flags: r.flags ^ 0xAAu32,",
            "        tag: r.tag.wrapping_add(5u32),",
            "    };",
            "    let new_met = Metrics {",
            "        sum: m.sum.wrapping_add(score),",
            "        xor_acc: m.xor_acc ^ a1 ^ b1,",
            "        count: m.count.wrapping_add(2u32),",
            "        mode: m.mode.wrapping_add(1u32) & 7u32,",
            "    };",
            "    PipeState { rec: new_rec, met: new_met, score }",
            "}",
            "",
        ]
    return name, lines


def emit_cross_exports_and_eval(
    lang: str,
    mod_idx: int,
    bitmix_names: List[str],
    fsm_names: List[str],
    win_names: List[str],
    loop_names: List[str],
    pipe_names: List[str],
    delta: int,
) -> List[str]:
    k_b = det_const(mod_idx, 99, 1)
    k_f = det_const(mod_idx, 99, 2)
    lines: List[str] = []

    if lang == "eon":
        lines.extend([
            "pub fn blend_state(x: u64, y: u64) -> u64 {",
            f"    let t = (x ^ {u64_eon(k_b)}) +% (y *% {u64_eon(k_f)});",
            "    return mix_local(t) ^ (x >> 13_u64);",
            "}",
            "",
            "pub fn fold_metric(x: u64) -> u64 {",
            "    let m = make_metrics(x);",
            f"    return (m.sum ^ m.xor_acc) +% (m.count.widen.[u64]() *% {u64_eon(k_b)});",
            "}",
            "",
        ])
        if mod_idx == TARGET_MOD_IDX:
            lines.extend([
                "pub fn inc_target_kernel(a: u64, b: u64) -> u64 {",
                f"    let tweak: u64 = {delta}_u64; // BENCH_INC_MARKER",
                f"    var v = (a ^ (tweak *% 0x9E3779B97F4A7C15_u64)) +% {u64_eon(k_b)};",
                "    v = (v ^ (b >> 11_u64)) *% 0xBF58476D1CE4E5B9_u64;",
                "    v = (v +% tweak) ^ (v >> 29_u64);",
                "    return v;",
                "}",
                "",
            ])
        lines.append(f"pub fn eval_mod_{mod_idx:02d}(seed: u64) -> u64 {{")
        lines.append("    var acc: u64 = mix_local(seed);")
        if mod_idx >= 1:
            lines.append(f"    acc = acc +% mod{mod_idx - 1:02d}.blend_state(acc, seed);")
        if mod_idx >= 2:
            lines.append(f"    acc = acc ^ mod{mod_idx - 2:02d}.fold_metric(acc);")
        if mod_idx == TARGET_MOD_IDX:
            lines.append("    acc = acc +% inc_target_kernel(acc, seed);")
        for bn in bitmix_names:
            lines.append(f"    acc = acc +% {bn}(acc, seed);")
        for fn in fsm_names:
            lines.append(f"    acc = acc ^ {fn}(acc);")
        for wn in win_names:
            lines.append(f"    acc = acc +% {wn}(acc);")
        lines.append("    var ps = PipeState.{ rec: make_record(acc), met: make_metrics(acc), score: acc };")
        for pn in pipe_names:
            lines.append(f"    ps = {pn}(ps.rec, ps.met);")
            lines.append("    acc = acc ^ ps.score;")
        for ln in loop_names:
            lines.append(f"    acc = acc +% {ln}(&mut ps, seed);")
        lines.append("    return mix_local(acc ^ ps.score);")
        lines.append("}")
        lines.append("")
    else:
        lines.extend([
            "pub fn blend_state(x: u64, y: u64) -> u64 {",
            f"    let t = (x ^ {u64_rs(k_b)}).wrapping_add(y.wrapping_mul({u64_rs(k_f)}));",
            "    mix_local(t) ^ (x >> 13u64)",
            "}",
            "",
            "pub fn fold_metric(x: u64) -> u64 {",
            "    let m = make_metrics(x);",
            f"    (m.sum ^ m.xor_acc).wrapping_add((m.count as u64).wrapping_mul({u64_rs(k_b)}))",
            "}",
            "",
        ])
        if mod_idx == TARGET_MOD_IDX:
            lines.extend([
                "pub fn inc_target_kernel(a: u64, b: u64) -> u64 {",
                f"    let tweak: u64 = {delta}u64; // BENCH_INC_MARKER",
                f"    let mut v = (a ^ tweak.wrapping_mul(0x9E3779B97F4A7C15u64)).wrapping_add({u64_rs(k_b)});",
                "    v = (v ^ (b >> 11u64)).wrapping_mul(0xBF58476D1CE4E5B9u64);",
                "    v = v.wrapping_add(tweak) ^ (v >> 29u64);",
                "    v",
                "}",
                "",
            ])
        lines.append(f"pub fn eval_mod_{mod_idx:02d}(seed: u64) -> u64 {{")
        lines.append("    let mut acc: u64 = mix_local(seed);")
        if mod_idx >= 1:
            lines.append(f"    acc = acc.wrapping_add(mod{mod_idx - 1:02d}::blend_state(acc, seed));")
        if mod_idx >= 2:
            lines.append(f"    acc = acc ^ mod{mod_idx - 2:02d}::fold_metric(acc);")
        if mod_idx == TARGET_MOD_IDX:
            lines.append("    acc = acc.wrapping_add(inc_target_kernel(acc, seed));")
        for bn in bitmix_names:
            lines.append(f"    acc = acc.wrapping_add({bn}(acc, seed));")
        for fn in fsm_names:
            lines.append(f"    acc = acc ^ {fn}(acc);")
        for wn in win_names:
            lines.append(f"    acc = acc.wrapping_add({wn}(acc));")
        lines.append("    let mut ps = PipeState { rec: make_record(acc), met: make_metrics(acc), score: acc };")
        for pn in pipe_names:
            lines.append(f"    ps = {pn}(ps.rec, ps.met);")
            lines.append("    acc = acc ^ ps.score;")
        for ln in loop_names:
            lines.append(f"    acc = acc.wrapping_add({ln}(&mut ps, seed));")
        lines.append("    mix_local(acc ^ ps.score)")
        lines.append("}")
        lines.append("")
    return lines


def generate_module(lang: str, mod_idx: int, delta: int) -> str:
    lines: List[str] = []
    lines.extend(emit_common_types(lang, mod_idx))
    lines.extend(emit_helpers(lang, mod_idx))

    bitmix_names: List[str] = []
    fsm_names: List[str] = []
    win_names: List[str] = []
    loop_names: List[str] = []
    pipe_names: List[str] = []

    # 9 bitmix + 8 fsm + 8 window + 7 loop_evolve + 7 pipe_combine = 39 kernels
    # + 5 helpers + 2..3 exports + eval_mod = ~47 functions, ~1,550 LOC/module
    # 20 modules * ~1,550 LOC = ~31,000+ LOC!
    for i in range(9):
        nm, body = emit_bitmix_kernel(lang, mod_idx, i)
        bitmix_names.append(nm)
        lines.extend(body)
    for i in range(8):
        nm, body = emit_fsm_kernel(lang, mod_idx, 10 + i)
        fsm_names.append(nm)
        lines.extend(body)
    for i in range(8):
        nm, body = emit_window_kernel(lang, mod_idx, 20 + i)
        win_names.append(nm)
        lines.extend(body)
    for i in range(7):
        nm, body = emit_loop_evolve_kernel(lang, mod_idx, 30 + i)
        loop_names.append(nm)
        lines.extend(body)
    for i in range(7):
        nm, body = emit_pipe_combinator(lang, mod_idx, 40 + i)
        pipe_names.append(nm)
        lines.extend(body)

    lines.extend(
        emit_cross_exports_and_eval(
            lang, mod_idx, bitmix_names, fsm_names, win_names, loop_names, pipe_names, delta
        )
    )
    return "\n".join(lines) + "\n"


def generate_main(lang: str) -> str:
    if lang == "eon":
        lines = [
            "module bench.cs.main;",
            "",
            "import std.rt;",
        ]
        for m in range(NUM_MODULES):
            lines.append(f"import bench.cs.mod{m:02d};")
        lines.extend([
            "",
            "pub fn main(env: *mut Env) -> i32 {",
            "    let t0 = rt.now_ns(&env.clock);",
            "    var checksum: u64 = 0x123456789ABCDEF0_u64;",
            "    for round in 0_usize..4_usize {",
            "        let seed = checksum ^ (round.widen.[u64]() *% 0x9E3779B97F4A7C15_u64);",
        ])
        for m in range(NUM_MODULES):
            lines.append(f"        checksum = checksum +% mod{m:02d}.eval_mod_{m:02d}(seed ^ {m}_u64);")
        lines.extend([
            "    }",
            "    let elapsed_ns = rt.elapsed_since(&env.clock, t0);",
            '    rt.print_str("RESULT compile_speed checksum=");',
            "    rt.print_u64(checksum);",
            '    rt.print_str(" elapsed_ns=");',
            "    rt.print_u64(elapsed_ns);",
            "    rt.print_nl();",
            "    return 0_i32;",
            "}",
            "",
        ])
        return "\n".join(lines)
    else:
        lines = [
            "#![forbid(unsafe_code)]",
            "",
            "use std::time::Instant;",
            "",
        ]
        for m in range(NUM_MODULES):
            lines.append(f"mod mod{m:02d};")
        lines.extend([
            "",
            "fn main() {",
            "    let t0 = Instant::now();",
            "    let mut checksum: u64 = 0x123456789ABCDEF0u64;",
            "    for round in 0usize..4usize {",
            "        let seed = checksum ^ (round as u64).wrapping_mul(0x9E3779B97F4A7C15u64);",
        ])
        for m in range(NUM_MODULES):
            lines.append(f"        checksum = checksum.wrapping_add(mod{m:02d}::eval_mod_{m:02d}(seed ^ {m}u64));")
        lines.extend([
            "    }",
            "    let elapsed_ns = t0.elapsed().as_nanos() as u64;",
            '    println!("RESULT compile_speed checksum={} elapsed_ns={}", checksum, elapsed_ns);',
            "}",
            "",
        ])
        return "\n".join(lines)


def generate_codebase(base_dir: Path, delta: int = 0) -> Dict[str, int]:
    eon_dir = base_dir / "gen_eon"
    rs_dir = base_dir / "gen_rs"
    rs_src = rs_dir / "src"

    eon_dir.mkdir(parents=True, exist_ok=True)
    rs_src.mkdir(parents=True, exist_ok=True)

    eon_loc = 0
    rs_loc = 0

    for m in range(NUM_MODULES):
        eon_src = generate_module("eon", m, delta)
        rs_code = generate_module("rs", m, delta)
        (eon_dir / f"mod{m:02d}.eon").write_text(eon_src, encoding="utf-8")
        (rs_src / f"mod{m:02d}.rs").write_text(rs_code, encoding="utf-8")
        eon_loc += eon_src.count("\n")
        rs_loc += rs_code.count("\n")

    eon_main = generate_main("eon")
    rs_main = generate_main("rs")
    (eon_dir / "main.eon").write_text(eon_main, encoding="utf-8")
    (rs_src / "main.rs").write_text(rs_main, encoding="utf-8")
    eon_loc += eon_main.count("\n")
    rs_loc += rs_main.count("\n")

    cargo_toml = """[package]
name = "bench_compile_speed"
version = "0.1.0"
edition = "2021"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
"""
    (rs_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")

    return {
        "modules": NUM_MODULES + 1,
        "eon_loc": eon_loc,
        "rs_loc": rs_loc,
    }


def mutate_one_function(base_dir: Path, delta: int) -> None:
    """Mutate ONLY 1 function body (`inc_target_kernel` in `mod10`) in both Eon and Rust."""
    eon_mod = base_dir / "gen_eon" / f"mod{TARGET_MOD_IDX:02d}.eon"
    rs_mod = base_dir / "gen_rs" / "src" / f"mod{TARGET_MOD_IDX:02d}.rs"

    eon_txt = eon_mod.read_text(encoding="utf-8")
    new_eon, n1 = re.subn(
        r"let tweak: u64 = \d+_u64; // BENCH_INC_MARKER",
        f"let tweak: u64 = {delta}_u64; // BENCH_INC_MARKER",
        eon_txt,
        count=1,
    )
    if n1 != 1:
        raise RuntimeError(f"failed to mutate Eon marker in {eon_mod}")
    eon_mod.write_text(new_eon, encoding="utf-8")

    rs_txt = rs_mod.read_text(encoding="utf-8")
    new_rs, n2 = re.subn(
        r"let tweak: u64 = \d+u64; // BENCH_INC_MARKER",
        f"let tweak: u64 = {delta}u64; // BENCH_INC_MARKER",
        rs_txt,
        count=1,
    )
    if n2 != 1:
        raise RuntimeError(f"failed to mutate Rust marker in {rs_mod}")
    rs_mod.write_text(new_rs, encoding="utf-8")


def run_cmd_timed(cmd: List[str], cwd: Path | None = None) -> Tuple[int, float, str, str]:
    t0 = time.perf_counter_ns()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt_ns = time.perf_counter_ns() - t0
    return proc.returncode, float(dt_ns), proc.stdout, proc.stderr


def run_binary_checksum(bin_path: Path) -> int:
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"binary {bin_path} exited {proc.returncode}:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            return int(m.group(2))
    raise RuntimeError(f"no RESULT line from {bin_path}: {proc.stdout!r}")


def benchmark_compile_speed(
    base_dir: Path,
    eonc: Path,
    eon_flags: List[str],
    runs: int = 5,
    warmup: int = 1,
) -> Dict[str, object]:
    stats_gen = generate_codebase(base_dir, delta=0)
    eon_dir = base_dir / "gen_eon"
    rs_main = base_dir / "gen_rs" / "src" / "main.rs"

    with tempfile.TemporaryDirectory(prefix="eon_cs_bench_") as td:
        tmp = Path(td)
        eon_cold_bin = tmp / "cs_eon_cold"
        rs_cold_bin = tmp / "cs_rs_cold"
        eon_inc_bin = tmp / "cs_eon_inc"
        rs_inc_bin = tmp / "cs_rs_inc"
        emi_cache = tmp / "emi_cache"
        rs_inc_cache = tmp / "rs_inc_cache"

        eon_cold_cmd = [str(eonc), "build", *eon_flags, str(eon_dir), "-o", str(eon_cold_bin)]
        rs_cold_cmd = [
            "rustc",
            "-C", "opt-level=3",
            "-C", "lto=fat",
            "-C", "codegen-units=1",
            "-C", "panic=abort",
            "-C", "target-cpu=native",
            "--edition", "2021",
            str(rs_main),
            "-o", str(rs_cold_bin),
        ]

        # 1. Cold build warmup + timed runs
        eon_cold_ns: List[float] = []
        rs_cold_ns: List[float] = []
        cold_checksum = None

        for i in range(warmup + runs):
            if eon_cold_bin.exists():
                eon_cold_bin.unlink()
            rc, dt_e, out_e, err_e = run_cmd_timed(eon_cold_cmd)
            if rc != 0:
                raise RuntimeError(f"eonc cold build failed (rc={rc}): {err_e}\n{out_e}")

            if rs_cold_bin.exists():
                rs_cold_bin.unlink()
            rc, dt_r, out_r, err_r = run_cmd_timed(rs_cold_cmd)
            if rc != 0:
                raise RuntimeError(f"rustc cold build failed (rc={rc}): {err_r}\n{out_r}")

            ck_e = run_binary_checksum(eon_cold_bin)
            ck_r = run_binary_checksum(rs_cold_bin)
            if ck_e != ck_r:
                raise RuntimeError(f"cold checksum mismatch: eon={ck_e} rust={ck_r}")
            cold_checksum = ck_e

            if i >= warmup:
                eon_cold_ns.append(dt_e)
                rs_cold_ns.append(dt_r)

        # 2. Prime incremental caches (delta=0)
        emi_cache.mkdir(parents=True, exist_ok=True)
        rs_inc_cache.mkdir(parents=True, exist_ok=True)

        eon_inc_cmd = [
            str(eonc), "build", *eon_flags,
            "--emi-cache", str(emi_cache),
            str(eon_dir), "-o", str(eon_inc_bin),
        ]
        rs_inc_cmd = [
            "rustc",
            "-C", "opt-level=3",
            "-C", f"incremental={rs_inc_cache}",
            "-C", "panic=abort",
            "-C", "target-cpu=native",
            "--edition", "2021",
            str(rs_main),
            "-o", str(rs_inc_bin),
        ]

        rc, _, out_e, err_e = run_cmd_timed(eon_inc_cmd)
        if rc != 0:
            raise RuntimeError(f"eonc incremental prime failed: {err_e}\n{out_e}")
        rc, _, out_r, err_r = run_cmd_timed(rs_inc_cmd)
        if rc != 0:
            raise RuntimeError(f"rustc incremental prime failed: {err_r}\n{out_r}")

        # 3. Incremental 1-function-edit warmup + timed runs
        eon_inc_ns: List[float] = []
        rs_inc_ns: List[float] = []
        last_inc_checksum = None

        try:
            for i in range(warmup + runs):
                delta = i + 1
                mutate_one_function(base_dir, delta)

                rc, dt_e, out_e, err_e = run_cmd_timed(eon_inc_cmd)
                if rc != 0:
                    raise RuntimeError(f"eonc incremental build failed: {err_e}\n{out_e}")
                rc, dt_r, out_r, err_r = run_cmd_timed(rs_inc_cmd)
                if rc != 0:
                    raise RuntimeError(f"rustc incremental build failed: {err_r}\n{out_r}")

                ck_e = run_binary_checksum(eon_inc_bin)
                ck_r = run_binary_checksum(rs_inc_bin)
                if ck_e != ck_r:
                    raise RuntimeError(f"inc checksum mismatch (delta={delta}): eon={ck_e} rust={ck_r}")
                if ck_e == cold_checksum:
                    raise RuntimeError(f"inc edit did not change checksum (delta={delta})")
                last_inc_checksum = ck_e

                if i >= warmup:
                    eon_inc_ns.append(dt_e)
                    rs_inc_ns.append(dt_r)
        finally:
            # Always restore delta=0
            mutate_one_function(base_dir, 0)

    eon_cold_med = statistics.median(eon_cold_ns)
    rs_cold_med = statistics.median(rs_cold_ns)
    eon_inc_med = statistics.median(eon_inc_ns)
    rs_inc_med = statistics.median(rs_inc_ns)

    return {
        "loc": stats_gen,
        "cold_checksum": cold_checksum,
        "inc_checksum": last_inc_checksum,
        "eon_cold_ns": eon_cold_ns,
        "rs_cold_ns": rs_cold_ns,
        "eon_inc_ns": eon_inc_ns,
        "rs_inc_ns": rs_inc_ns,
        "eon_cold_median_ns": eon_cold_med,
        "rs_cold_median_ns": rs_cold_med,
        "cold_ratio": eon_cold_med / rs_cold_med if rs_cold_med > 0 else math.nan,
        "eon_inc_median_ns": eon_inc_med,
        "rs_inc_median_ns": rs_inc_med,
        "inc_ratio": eon_inc_med / rs_inc_med if rs_inc_med > 0 else math.nan,
    }


def clean_generated(base_dir: Path) -> None:
    for sub in ("gen_eon", "gen_rs", ".emi_cache", ".rs_inc"):
        p = base_dir / sub
        if p.exists():
            shutil.rmtree(p)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_eonc = repo_root / "compiler" / "target" / "release" / "eonc"

    ap = argparse.ArgumentParser(description="Generate and benchmark 30k-LOC Eon vs Rust compile speed")
    ap.add_argument("--out-dir", type=Path, default=script_dir, help="Directory for gen_eon/ and gen_rs/")
    ap.add_argument("--delta", type=int, default=0, help="Constant tweak in mod10.inc_target_kernel")
    ap.add_argument("--mutate-one", action="store_true", help="Only mutate 1 function body in existing gen_eon/ and gen_rs/")
    ap.add_argument("--incremental", action="store_true", help="Run full cold + 1-fn incremental compile benchmark")
    ap.add_argument("--eonc", type=Path, default=default_eonc, help="Path to eonc binary")
    ap.add_argument("--eon-flags", type=str, default="", help="Extra flags passed to `eonc build`")
    ap.add_argument("--runs", type=int, default=5, help="Timed runs for --incremental mode")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs for --incremental mode")
    ap.add_argument("--clean", action="store_true", help="Delete generated gen_eon/ and gen_rs/ trees")
    args = ap.parse_args()

    if args.clean:
        clean_generated(args.out_dir)
        print(f"Cleaned generated trees in {args.out_dir}")
        return 0

    if args.mutate_one:
        mutate_one_function(args.out_dir, args.delta)
        print(f"Mutated mod{TARGET_MOD_IDX:02d}.inc_target_kernel with delta={args.delta}")
        return 0

    if args.incremental:
        res = benchmark_compile_speed(
            args.out_dir,
            args.eonc,
            shlex.split(args.eon_flags),
            runs=args.runs,
            warmup=args.warmup,
        )
        loc = res["loc"]
        print(
            f"Generated {loc['modules']} files: Eon={loc['eon_loc']} LOC, Rust={loc['rs_loc']} LOC"
        )
        print(
            f"Checksums verified: cold={res['cold_checksum']} inc(last)={res['inc_checksum']} (eon == rust)"
        )
        print(
            f"Cold build (median of {args.runs}): "
            f"Eon={res['eon_cold_median_ns']/1e6:.2f} ms  "
            f"Rust={res['rs_cold_median_ns']/1e6:.2f} ms  "
            f"ratio={res['cold_ratio']:.3f}x  "
            f"(Eon speedup={1.0/res['cold_ratio']:.2f}x)"
        )
        print(
            f"Incremental 1-fn edit (median of {args.runs}): "
            f"Eon={res['eon_inc_median_ns']/1e6:.2f} ms  "
            f"Rust={res['rs_inc_median_ns']/1e6:.2f} ms  "
            f"ratio={res['inc_ratio']:.3f}x  "
            f"(Eon speedup={1.0/res['inc_ratio']:.2f}x)"
        )
        return 0

    loc = generate_codebase(args.out_dir, delta=args.delta)
    print(
        f"Generated {loc['modules']} files in {args.out_dir}: "
        f"gen_eon={loc['eon_loc']} LOC, gen_rs={loc['rs_loc']} LOC"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
