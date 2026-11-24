#!/usr/bin/env python3
"""Reproducible Eon vs Safe Rust benchmark harness (Phase 9.1).

Features:
- Compiles Rust benchmarks with:
    rustc -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C target-cpu=native
- Compiles Eon benchmarks with:
    <eonc> build <eon-flags> ...
- Interleaved 1 warmup + N timed runs (default 15) to neutralize thermal/frequency drift.
- Verifies `eon_checksum == rust_checksum` on EVERY run (warmup + timed).
- Computes median, p10/p90, IQR%, CV%, noise indicator, Eon/Rust ratio,
  and geometric mean across runtime workloads (01-11) + cold & incremental compile ratios (12).
- Collects Linux hardware PMU counters via `ctypes` `perf_event_open` syscall
  (cycles, instructions, IPC, L1d misses, LLC/cache misses, branch misses) when permitted,
  with seamless fallback to `wait4` rusage + `elapsed_ns`.
- Auto-saves structured JSON + Markdown report under `bench/results/` (or `--save <path>`).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_EONC = REPO_ROOT / "selfhost" / "out" / "eonc2"
DEFAULT_RUST_FLAGS = "-C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C target-cpu=native"

RESULT_RE = re.compile(r"^RESULT\s+(\S+)\s+checksum=(\d+)\s+elapsed_ns=(\d+)\s*$")

KNOWN_BENCHMARKS: List[Tuple[str, str, str]] = [
    ("01_json_dom", "json_dom", "runtime"),
    ("02_eon_lexer_parser", "eon_lexer_parser", "runtime"),
    ("03_utf8_scan", "utf8_scan", "runtime"),
    ("04_ui_layout", "ui_layout", "runtime"),
    ("05_ecs_physics", "ecs_physics", "runtime"),
    ("06_hashmap_intern", "hashmap_intern", "runtime"),
    ("07_radix_pdqsort", "radix_pdqsort", "runtime"),
    ("08_matmul_f32", "matmul_f32", "runtime"),
    ("09_nbody_simd", "nbody_simd", "runtime"),
    ("10_par_reduce_graph", "par_reduce_graph", "runtime"),
    ("11_alloc_churn", "alloc_churn", "runtime"),
    ("12_compile_speed", "compile_speed", "compile"),
]


# ---------------------------------------------------------------------------
# Linux hardware PMU via ctypes `perf_event_open` (graceful fallback)
# ---------------------------------------------------------------------------

SYS_PERF_EVENT_OPEN_X86_64 = 298
PERF_TYPE_HARDWARE = 0
PERF_TYPE_HW_CACHE = 3

PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_MISSES = 3
PERF_COUNT_HW_BRANCH_MISSES = 5

PERF_COUNT_HW_CACHE_L1D = 0
PERF_COUNT_HW_CACHE_LL = 2
PERF_COUNT_HW_CACHE_OP_READ = 0
PERF_COUNT_HW_CACHE_RESULT_MISS = 1

PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FLAG_FD_CLOEXEC = 1 << 3

# attr.flags bits (Linux <linux/perf_event.h>)
PERF_ATTR_DISABLED = 1 << 0
PERF_ATTR_INHERIT = 1 << 1
PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
PERF_ATTR_EXCLUDE_HV = 1 << 6
PERF_ATTR_ENABLE_ON_EXEC = 1 << 12


class PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("__reserved_2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("__reserved_3", ctypes.c_uint32),
    ]


class PerfSampler:
    """Safely probes and samples Linux hardware PMU counters for child processes."""

    def __init__(self, enabled: bool = True) -> None:
        self.available = False
        self.reason = "disabled by flag"
        self.paranoid: Optional[int] = self._read_paranoid()
        self.libc: Any = None
        self.events: List[Tuple[str, int, int]] = []

        if not enabled:
            return
        if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
            self.reason = f"unsupported platform ({sys.platform}/{platform.machine()})"
            return

        try:
            self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError as exc:
            self.reason = f"libc load failed: {exc}"
            return

        # Candidate counters: each entry is (key, [(type, config), ...fallbacks])
        candidates = [
            ("cycles", [(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)]),
            ("instructions", [(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)]),
            (
                "l1d_misses",
                [
                    (
                        PERF_TYPE_HW_CACHE,
                        PERF_COUNT_HW_CACHE_L1D
                        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
                    )
                ],
            ),
            (
                "llc_misses",
                [
                    (
                        PERF_TYPE_HW_CACHE,
                        PERF_COUNT_HW_CACHE_LL
                        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                        | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
                    ),
                    (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES),
                ],
            ),
            ("branch_misses", [(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES)]),
        ]

        probe_flags = PERF_ATTR_INHERIT | PERF_ATTR_EXCLUDE_KERNEL | PERF_ATTR_EXCLUDE_HV
        last_err = 0
        for name, specs in candidates:
            for ev_type, ev_cfg in specs:
                fd, err = self._open_fd(0, ev_type, ev_cfg, probe_flags)
                if fd >= 0:
                    os.close(fd)
                    self.events.append((name, ev_type, ev_cfg))
                    break
                last_err = err

        if self.events:
            self.available = True
            self.reason = (
                f"ok ({len(self.events)} counters: {', '.join(e[0] for e in self.events)}; "
                f"perf_event_paranoid={self.paranoid})"
            )
        else:
            err_msg = os.strerror(last_err) if last_err else "unavailable"
            self.reason = f"fallback to rusage (perf_event_paranoid={self.paranoid}, errno={last_err}: {err_msg})"

    @staticmethod
    def _read_paranoid() -> Optional[int]:
        try:
            return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
        except OSError:
            return None

    def _open_fd(self, pid: int, ev_type: int, ev_cfg: int, flags: int) -> Tuple[int, int]:
        attr = PerfEventAttr()
        attr.type = ev_type
        attr.size = ctypes.sizeof(PerfEventAttr)
        attr.config = ev_cfg
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
        attr.flags = flags
        ctypes.set_errno(0)
        fd = int(
            self.libc.syscall(
                SYS_PERF_EVENT_OPEN_X86_64,
                ctypes.byref(attr),
                ctypes.c_int(pid),
                ctypes.c_int(-1),
                ctypes.c_int(-1),
                ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
            )
        )
        return fd, ctypes.get_errno()

    @staticmethod
    def _read_scaled(fd: int) -> Optional[int]:
        try:
            raw = os.read(fd, 24)
            if len(raw) != 24:
                return None
            val, enabled_t, running_t = struct.unpack("<QQQ", raw)
            if running_t == 0:
                return 0
            if running_t < enabled_t:
                return int(round(val * (enabled_t / running_t)))
            return int(val)
        except OSError:
            return None

    def run_binary(self, bin_path: Path) -> Tuple[int, str, str, Dict[str, Any], Optional[Dict[str, int]]]:
        """Execute `bin_path` capturing stdout, stderr, rusage, and optional PMU counters."""
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            if not self.available:
                t0 = time.perf_counter_ns()
                pid = os.fork()
                if pid == 0:
                    try:
                        os.dup2(out_f.fileno(), 1)
                        os.dup2(err_f.fileno(), 2)
                        os.execv(str(bin_path), [str(bin_path)])
                    finally:
                        os._exit(127)
                _, status, ru = os.wait4(pid, 0)
                wall_ns = time.perf_counter_ns() - t0
                rc = os.waitstatus_to_exitcode(status)
                out_f.seek(0)
                err_f.seek(0)
                stdout = out_f.read().decode("utf-8", errors="replace")
                stderr = err_f.read().decode("utf-8", errors="replace")
                rusage_dict = {
                    "wall_ns": wall_ns,
                    "user_ms": ru.ru_utime * 1000.0,
                    "sys_ms": ru.ru_stime * 1000.0,
                    "max_rss_kib": int(ru.ru_maxrss),
                    "minflt": int(ru.ru_minflt),
                    "majflt": int(ru.ru_majflt),
                    "nvcsw": int(ru.ru_nvcsw),
                    "nivcsw": int(ru.ru_nivcsw),
                }
                return rc, stdout, stderr, rusage_dict, None

            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                try:
                    os.close(w_fd)
                    os.read(r_fd, 1)
                    os.close(r_fd)
                    os.dup2(out_f.fileno(), 1)
                    os.dup2(err_f.fileno(), 2)
                    os.execv(str(bin_path), [str(bin_path)])
                finally:
                    os._exit(127)

            os.close(r_fd)
            child_flags = (
                PERF_ATTR_DISABLED
                | PERF_ATTR_INHERIT
                | PERF_ATTR_EXCLUDE_KERNEL
                | PERF_ATTR_EXCLUDE_HV
                | PERF_ATTR_ENABLE_ON_EXEC
            )
            fds: Dict[str, int] = {}
            for name, ev_type, ev_cfg in self.events:
                fd, _ = self._open_fd(pid, ev_type, ev_cfg, child_flags)
                if fd >= 0:
                    fds[name] = fd

            t0 = time.perf_counter_ns()
            os.write(w_fd, b"g")
            os.close(w_fd)
            _, status, ru = os.wait4(pid, 0)
            wall_ns = time.perf_counter_ns() - t0

            pmu_vals: Dict[str, int] = {}
            for name, fd in fds.items():
                v = self._read_scaled(fd)
                os.close(fd)
                if v is not None:
                    pmu_vals[name] = v

            rc = os.waitstatus_to_exitcode(status)
            out_f.seek(0)
            err_f.seek(0)
            stdout = out_f.read().decode("utf-8", errors="replace")
            stderr = err_f.read().decode("utf-8", errors="replace")
            rusage_dict = {
                "wall_ns": wall_ns,
                "user_ms": ru.ru_utime * 1000.0,
                "sys_ms": ru.ru_stime * 1000.0,
                "max_rss_kib": int(ru.ru_maxrss),
                "minflt": int(ru.ru_minflt),
                "majflt": int(ru.ru_majflt),
                "nvcsw": int(ru.ru_nvcsw),
                "nivcsw": int(ru.ru_nivcsw),
            }
            return rc, stdout, stderr, rusage_dict, (pmu_vals if pmu_vals else None)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize_ns(samples: List[float]) -> Dict[str, Any]:
    s = sorted(samples)
    med = percentile(s, 50.0)
    p10 = percentile(s, 10.0)
    p25 = percentile(s, 25.0)
    p75 = percentile(s, 75.0)
    p90 = percentile(s, 90.0)
    iqr = p75 - p25
    iqr_pct = (100.0 * iqr / med) if med > 0 else 0.0
    mean = statistics.fmean(s)
    stdev = statistics.stdev(s) if len(s) >= 2 else 0.0
    cv_pct = (100.0 * stdev / mean) if mean > 0 else 0.0
    noise_max = max(iqr_pct, cv_pct)
    if noise_max < 5.0:
        noise = "OK"
    elif noise_max < 10.0:
        noise = "WARN"
    else:
        noise = "NOISY"
    return {
        "n": len(s),
        "median_ns": med,
        "median_ms": med / 1e6,
        "min_ns": s[0],
        "max_ns": s[-1],
        "mean_ns": mean,
        "p10_ns": p10,
        "p10_ms": p10 / 1e6,
        "p25_ns": p25,
        "p75_ns": p75,
        "p90_ns": p90,
        "p90_ms": p90 / 1e6,
        "iqr_ns": iqr,
        "iqr_pct": iqr_pct,
        "cv_pct": cv_pct,
        "noise": noise,
        "samples_ns": samples,
    }


def median_dict(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d and d[k] is not None]
        if vals:
            out[k] = float(statistics.median(vals))
    if "instructions" in out and "cycles" in out and out["cycles"] > 0:
        out["ipc"] = out["instructions"] / out["cycles"]
    return out


def parse_result_line(stdout: str, expected_name: str) -> Tuple[str, int, int]:
    for line in stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            name, ck, ns = m.group(1), int(m.group(2)), int(m.group(3))
            if expected_name and name != expected_name:
                # Tolerate minor name alias while recording exact printed name
                pass
            return name, ck, ns
    raise RuntimeError(f"missing `RESULT <name> checksum=<u64> elapsed_ns=<u64>` line; got:\n{stdout}")


# ---------------------------------------------------------------------------
# Environment metadata & benchmark discovery
# ---------------------------------------------------------------------------

def get_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def file_sha256(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def get_rustc_version() -> str:
    try:
        r = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def discover_benchmarks(only_filter: Optional[str]) -> List[Tuple[str, str, str, Path]]:
    seen_dirs = set()
    all_benches: List[Tuple[str, str, str, Path]] = []

    for dname, sname, kind in KNOWN_BENCHMARKS:
        p = SCRIPT_DIR / dname
        if p.is_dir():
            if kind == "compile" and (p / "gen_codebase.py").is_file():
                all_benches.append((dname, sname, kind, p))
                seen_dirs.add(dname)
            elif kind == "runtime" and (p / "main.eon").is_file() and (p / "main.rs").is_file():
                all_benches.append((dname, sname, kind, p))
                seen_dirs.add(dname)

    # Also auto-discover any extra NN_* runtime benchmarks
    for p in sorted(SCRIPT_DIR.iterdir()):
        if not p.is_dir() or p.name in seen_dirs:
            continue
        if re.match(r"^\d+_", p.name) and (p / "main.eon").is_file() and (p / "main.rs").is_file():
            sname = p.name.split("_", 1)[1]
            all_benches.append((p.name, sname, "runtime", p))

    if not only_filter:
        return all_benches

    tokens = {t.strip().lower() for t in only_filter.split(",") if t.strip()}
    selected: List[Tuple[str, str, str, Path]] = []
    for item in all_benches:
        dname, sname, _, _ = item
        num_str = dname.split("_", 1)[0]
        num_int = str(int(num_str)) if num_str.isdigit() else ""
        if (
            dname.lower() in tokens
            or sname.lower() in tokens
            or num_str.lower() in tokens
            or (num_int and num_int in tokens)
        ):
            selected.append(item)
    return selected


# ---------------------------------------------------------------------------
# Runtime benchmark execution
# ---------------------------------------------------------------------------

def run_runtime_benchmark(
    dname: str,
    sname: str,
    bdir: Path,
    eonc: Path,
    eon_flags: List[str],
    rust_flags: List[str],
    runs: int,
    warmup: int,
    sampler: PerfSampler,
    build_dir: Path,
) -> Dict[str, Any]:
    rs_src = bdir / "main.rs"
    eon_files = sorted(bdir.glob("*.eon"))
    eon_input = bdir if len(eon_files) > 1 else (bdir / "main.eon")

    rs_bin = build_dir / f"bench_{sname}_rs"
    eon_bin = build_dir / f"bench_{sname}_eon"

    entry: Dict[str, Any] = {
        "dir": dname,
        "name": sname,
        "kind": "runtime",
        "status": "FAIL",
        "error": None,
    }

    # 1. Compile Rust
    rs_cmd = ["rustc", *rust_flags, str(rs_src), "-o", str(rs_bin)]
    t0 = time.perf_counter_ns()
    p_rs = subprocess.run(rs_cmd, capture_output=True, text=True)
    rs_build_ns = time.perf_counter_ns() - t0
    if p_rs.returncode != 0:
        entry["error"] = f"rustc failed (rc={p_rs.returncode}): {p_rs.stderr.strip()}"
        return entry

    # 2. Compile Eon
    eon_cmd = [str(eonc), "build", *eon_flags, str(eon_input), "-o", str(eon_bin)]
    t0 = time.perf_counter_ns()
    p_eon = subprocess.run(eon_cmd, capture_output=True, text=True)
    eon_build_ns = time.perf_counter_ns() - t0
    if p_eon.returncode != 0:
        entry["error"] = (
            f"eonc build failed (rc={p_eon.returncode}): "
            f"{(p_eon.stderr or p_eon.stdout).strip()}"
        )
        return entry

    entry["build"] = {
        "eon_build_ms": eon_build_ns / 1e6,
        "rust_build_ms": rs_build_ns / 1e6,
        "eon_bin_bytes": eon_bin.stat().st_size,
        "rust_bin_bytes": rs_bin.stat().st_size,
    }

    # 3. Interleaved warmup + timed runs
    eon_samples: List[float] = []
    rs_samples: List[float] = []
    eon_rusages: List[Dict[str, Any]] = []
    rs_rusages: List[Dict[str, Any]] = []
    eon_pmus: List[Dict[str, int]] = []
    rs_pmus: List[Dict[str, int]] = []
    expected_checksum: Optional[int] = None

    for i in range(warmup + runs):
        # Alternate execution order each iteration to eliminate systematic drift bias
        order = ("rust", "eon") if (i % 2 == 0) else ("eon", "rust")
        iter_res: Dict[str, Tuple[int, int, Dict[str, Any], Optional[Dict[str, int]]]] = {}

        for lang in order:
            bp = rs_bin if lang == "rust" else eon_bin
            rc, out, err, ru, pmu = sampler.run_binary(bp)
            if rc != 0:
                entry["error"] = f"{lang} binary exited {rc} on iter {i}: stderr={err.strip()!r} stdout={out.strip()!r}"
                return entry
            try:
                _, ck, elapsed_ns = parse_result_line(out, sname)
            except RuntimeError as exc:
                entry["error"] = f"{lang} parse error on iter {i}: {exc}"
                return entry
            iter_res[lang] = (ck, elapsed_ns, ru, pmu)

        ck_r, ns_r, ru_r, pmu_r = iter_res["rust"]
        ck_e, ns_e, ru_e, pmu_e = iter_res["eon"]

        if ck_e != ck_r:
            entry["error"] = f"CHECKSUM MISMATCH on iter {i}: eon={ck_e} != rust={ck_r}"
            return entry
        if expected_checksum is None:
            expected_checksum = ck_r
        elif ck_r != expected_checksum:
            entry["error"] = f"NON-DETERMINISTIC CHECKSUM on iter {i}: got {ck_r}, expected {expected_checksum}"
            return entry

        if i >= warmup:
            rs_samples.append(float(ns_r))
            eon_samples.append(float(ns_e))
            rs_rusages.append(ru_r)
            eon_rusages.append(ru_e)
            if pmu_r is not None:
                rs_pmus.append(pmu_r)
            if pmu_e is not None:
                eon_pmus.append(pmu_e)

    eon_stat = summarize_ns(eon_samples)
    rs_stat = summarize_ns(rs_samples)
    ratio = (eon_stat["median_ns"] / rs_stat["median_ns"]) if rs_stat["median_ns"] > 0 else math.nan

    entry["status"] = "PASS"
    entry["checksum"] = expected_checksum
    entry["eon"] = eon_stat
    entry["rust"] = rs_stat
    entry["ratio_eon_over_rust"] = ratio
    entry["rusage"] = {
        "eon": median_dict(eon_rusages),
        "rust": median_dict(rs_rusages),
    }
    entry["pmu"] = {
        "eon": median_dict(eon_pmus),
        "rust": median_dict(rs_pmus),
    } if (eon_pmus and rs_pmus) else None
    return entry


# ---------------------------------------------------------------------------
# Compile-speed workload (12_compile_speed)
# ---------------------------------------------------------------------------

def run_compile_benchmark(
    bdir: Path,
    eonc: Path,
    eon_flags: List[str],
    compile_runs: int,
    warmup: int,
) -> Dict[str, Any]:
    gen_path = bdir / "gen_codebase.py"
    spec = importlib.util.spec_from_file_location("bench_gen_codebase", gen_path)
    if spec is None or spec.loader is None:
        return {"dir": bdir.name, "name": "compile_speed", "kind": "compile", "status": "FAIL", "error": "cannot load gen_codebase.py"}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    entry: Dict[str, Any] = {
        "dir": bdir.name,
        "name": "compile_speed",
        "kind": "compile",
        "status": "FAIL",
        "error": None,
    }
    try:
        raw = mod.benchmark_compile_speed(
            bdir,
            eonc,
            eon_flags,
            runs=compile_runs,
            warmup=warmup,
        )
    except Exception as exc:
        entry["error"] = str(exc)
        return entry

    eon_cold = summarize_ns(raw["eon_cold_ns"])
    rs_cold = summarize_ns(raw["rs_cold_ns"])
    eon_inc = summarize_ns(raw["eon_inc_ns"])
    rs_inc = summarize_ns(raw["rs_inc_ns"])

    entry["status"] = "PASS"
    entry["loc"] = raw["loc"]
    entry["cold_checksum"] = raw["cold_checksum"]
    entry["inc_checksum"] = raw["inc_checksum"]
    entry["cold"] = {
        "eon": eon_cold,
        "rust": rs_cold,
        "ratio_eon_over_rust": eon_cold["median_ns"] / rs_cold["median_ns"] if rs_cold["median_ns"] > 0 else math.nan,
    }
    entry["incremental"] = {
        "eon": eon_inc,
        "rust": rs_inc,
        "ratio_eon_over_rust": eon_inc["median_ns"] / rs_inc["median_ns"] if rs_inc["median_ns"] > 0 else math.nan,
    }
    return entry


# ---------------------------------------------------------------------------
# Report rendering (Markdown + JSON)
# ---------------------------------------------------------------------------

def fmt_count(v: Optional[float]) -> str:
    if v is None or math.isnan(v):
        return "-"
    if v >= 1e9:
        return f"{v/1e9:.2f}G"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{int(round(v))}"


def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    runtime_res = [r for r in report["results"] if r["kind"] == "runtime"]
    compile_res = [r for r in report["results"] if r["kind"] == "compile"]

    lines: List[str] = []
    lines.append("# Eon vs Safe Rust Benchmark Report")
    lines.append("")
    lines.append(f"- **Timestamp (UTC):** `{meta['timestamp_utc']}`")
    lines.append(f"- **CPU:** {meta['cpu_model']} ({meta['cpu_count']} logical CPUs)")
    lines.append(f"- **OS / Kernel:** `{meta['os']}`")
    lines.append(f"- **Eon compiler:** `{meta['eonc']}` (flags: `{meta['eon_flags'] or '<none/--dev>'}`)")
    lines.append(f"- **Rust compiler:** `{meta['rustc_version']}` (flags: `{meta['rust_flags']}`)")
    lines.append(f"- **Runs:** {meta['warmup']} warmup + {meta['runs']} timed (interleaved)")
    lines.append(f"- **PMU counters:** {meta['pmu_status']}")
    lines.append("")

    if runtime_res:
        lines.append("## Runtime Workloads (01–11)")
        lines.append("")
        lines.append(
            "| Workload | Status | Eon med (ms) | Eon p10..p90 (ms) | Eon IQR/CV% | "
            "Rust med (ms) | Rust p10..p90 (ms) | Rust IQR/CV% | Eon/Rust | Noise | Checksum |"
        )
        lines.append(
            "| :--- | :---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: |"
        )
        for r in runtime_res:
            if r["status"] != "PASS":
                lines.append(
                    f"| `{r['name']}` | FAIL | - | - | - | - | - | - | - | - | `{r['error']}` |"
                )
                continue
            e = r["eon"]
            ru = r["rust"]
            noise = "NOISY" if "NOISY" in (e["noise"], ru["noise"]) else (
                "WARN" if "WARN" in (e["noise"], ru["noise"]) else "OK"
            )
            lines.append(
                f"| `{r['name']}` | PASS | {e['median_ms']:.2f} | "
                f"{e['p10_ms']:.2f}..{e['p90_ms']:.2f} | {e['iqr_pct']:.1f}/{e['cv_pct']:.1f}% | "
                f"{ru['median_ms']:.2f} | {ru['p10_ms']:.2f}..{ru['p90_ms']:.2f} | "
                f"{ru['iqr_pct']:.1f}/{ru['cv_pct']:.1f}% | **{r['ratio_eon_over_rust']:.3f}x** | "
                f"{noise} | `{r['checksum']}` |"
            )

        gmean = report["summary"].get("runtime_geomean_eon_over_rust")
        if gmean is not None:
            lines.append("")
            lines.append(
                f"**Geometric Mean (Eon / Rust across {report['summary']['runtime_passed']} passed runtime workloads):** "
                f"**`{gmean:.4f}x`** (< 1.0000x = Eon faster)"
            )
        lines.append("")

        # PMU / rusage details table
        has_pmu = any(r.get("pmu") for r in runtime_res if r["status"] == "PASS")
        if has_pmu:
            lines.append("### Hardware PMU & Memory Counters (Median)")
            lines.append("")
            lines.append(
                "| Workload | Lang | Cycles | Instrs | IPC | L1d Miss | LLC Miss | Br Miss | MaxRSS (KiB) |"
            )
            lines.append("| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in runtime_res:
                if r["status"] != "PASS" or not r.get("pmu"):
                    continue
                for lang in ("eon", "rust"):
                    p = r["pmu"][lang]
                    rss = r["rusage"][lang].get("max_rss_kib", 0)
                    ipc = f"{p['ipc']:.2f}" if "ipc" in p else "-"
                    lines.append(
                        f"| `{r['name']}` | {lang} | {fmt_count(p.get('cycles'))} | "
                        f"{fmt_count(p.get('instructions'))} | {ipc} | "
                        f"{fmt_count(p.get('l1d_misses'))} | {fmt_count(p.get('llc_misses'))} | "
                        f"{fmt_count(p.get('branch_misses'))} | {int(rss)} |"
                    )
            lines.append("")
        else:
            lines.append("### Process rusage Fallback (Median)")
            lines.append("")
            lines.append("| Workload | Lang | User (ms) | Sys (ms) | MaxRSS (KiB) | MinFlt | CtxSw |")
            lines.append("| :--- | :---: | ---: | ---: | ---: | ---: | ---: |")
            for r in runtime_res:
                if r["status"] != "PASS":
                    continue
                for lang in ("eon", "rust"):
                    ru = r["rusage"][lang]
                    csw = int(ru.get("nvcsw", 0) + ru.get("nivcsw", 0))
                    lines.append(
                        f"| `{r['name']}` | {lang} | {ru.get('user_ms', 0):.2f} | "
                        f"{ru.get('sys_ms', 0):.2f} | {int(ru.get('max_rss_kib', 0))} | "
                        f"{int(ru.get('minflt', 0))} | {csw} |"
                    )
            lines.append("")

    if compile_res:
        lines.append("## Compile Speed Workload (12 — 30,000+ LOC Multi-Module)")
        lines.append("")
        for r in compile_res:
            if r["status"] != "PASS":
                lines.append(f"- `compile_speed`: **FAIL** — `{r['error']}`")
                continue
            loc = r["loc"]
            lines.append(
                f"- **Codebase:** {loc['modules']} files — Eon `{loc['eon_loc']}` LOC, Rust `{loc['rs_loc']}` LOC"
            )
            lines.append(
                f"- **Verified checksums:** cold=`{r['cold_checksum']}`, inc=`{r['inc_checksum']}` (`eon == rust`)"
            )
            lines.append("")
            lines.append(
                "| Mode | Eon med (ms) | Eon p10..p90 (ms) | Eon IQR/CV% | "
                "Rust med (ms) | Rust p10..p90 (ms) | Rust IQR/CV% | Eon/Rust | Eon Speedup |"
            )
            lines.append("| :--- | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |")
            for label, key in (("Cold full build", "cold"), ("Incremental (1-fn edit)", "incremental")):
                sub = r[key]
                e = sub["eon"]
                ru = sub["rust"]
                rat = sub["ratio_eon_over_rust"]
                spd = (1.0 / rat) if rat and rat > 0 else math.nan
                lines.append(
                    f"| {label} | {e['median_ms']:.2f} | {e['p10_ms']:.2f}..{e['p90_ms']:.2f} | "
                    f"{e['iqr_pct']:.1f}/{e['cv_pct']:.1f}% | {ru['median_ms']:.2f} | "
                    f"{ru['p10_ms']:.2f}..{ru['p90_ms']:.2f} | {ru['iqr_pct']:.1f}/{ru['cv_pct']:.1f}% | "
                    f"**{rat:.3f}x** | **{spd:.2f}x** |"
                )
            lines.append("")

    return "\n".join(lines)


def resolve_save_paths(save_arg: Optional[Path], tag: str, ts_compact: str) -> Tuple[Path, Path]:
    results_dir = SCRIPT_DIR / "results"
    if save_arg is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        base = results_dir / f"bench_{tag}_{ts_compact}"
        return base.with_suffix(".json"), base.with_suffix(".md")

    p = save_arg
    if p.exists() and p.is_dir():
        base = p / f"bench_{tag}_{ts_compact}"
        return base.with_suffix(".json"), base.with_suffix(".md")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        return p, p.with_suffix(".md")
    if p.suffix == ".md":
        return p.with_suffix(".json"), p
    return p.with_suffix(".json"), p.with_suffix(".md")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible Eon vs Safe Rust Benchmark Runner")
    ap.add_argument(
        "--eonc",
        type=Path,
        default=DEFAULT_EONC,
        help=f"Path to eonc compiler binary (default: {DEFAULT_EONC})",
    )
    ap.add_argument(
        "--eon-flags",
        type=str,
        default="",
        help='Extra flags for `eonc build` (e.g. "--release" or "" for --dev)',
    )
    ap.add_argument(
        "--rust-flags",
        type=str,
        default=DEFAULT_RUST_FLAGS,
        help="Flags passed to rustc",
    )
    ap.add_argument("--runs", type=int, default=15, help="Timed runs per benchmark (default: 15)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per benchmark (default: 1)")
    ap.add_argument(
        "--compile-runs",
        type=int,
        default=None,
        help="Timed runs for 12_compile_speed (default: min(runs, 3))",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of benchmark names, directory names, or numbers (e.g. 10,11,12)",
    )
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save JSON + Markdown report to path (default: bench/results/bench_<tag>_<ts>.{json,md})",
    )
    ap.add_argument("--no-save", action="store_true", help="Do not save report files to disk")
    ap.add_argument("--no-pmu", action="store_true", help="Disable Linux perf_event_open PMU probing")
    args = ap.parse_args()

    eonc = args.eonc
    if not eonc.is_absolute():
        cand = (Path.cwd() / eonc).resolve()
        if cand.exists():
            eonc = cand
        else:
            eonc = (REPO_ROOT / eonc).resolve()
    if not eonc.is_file():
        print(f"ERROR: eonc binary not found at {eonc}", file=sys.stderr)
        return 2

    eon_flags = shlex.split(args.eon_flags)
    rust_flags = shlex.split(args.rust_flags)
    compile_runs = args.compile_runs if args.compile_runs is not None else min(args.runs, 3)

    selected = discover_benchmarks(args.only)
    if not selected:
        print(f"ERROR: no matching benchmarks found (filter={args.only!r})", file=sys.stderr)
        return 2

    sampler = PerfSampler(enabled=not args.no_pmu)
    now = datetime.datetime.now(datetime.timezone.utc)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_compact = now.strftime("%Y%m%d_%H%M%S")

    print(f"==> Eon vs Safe Rust Benchmark Harness ({len(selected)} workloads)", file=sys.stderr)
    print(f"    eonc: {eonc}  flags: {args.eon_flags or '<none/--dev>'}", file=sys.stderr)
    print(f"    runs: {args.warmup} warmup + {args.runs} timed (compile_speed: {compile_runs})", file=sys.stderr)
    print(f"    PMU : {sampler.reason}", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    any_failed = False

    with tempfile.TemporaryDirectory(prefix="eon_bench_bins_") as td:
        build_dir = Path(td)
        for dname, sname, kind, bdir in selected:
            print(f"==> Running [{dname}] ...", file=sys.stderr, flush=True)
            if kind == "compile":
                res = run_compile_benchmark(bdir, eonc, eon_flags, compile_runs, args.warmup)
            else:
                res = run_runtime_benchmark(
                    dname, sname, bdir, eonc, eon_flags, rust_flags,
                    args.runs, args.warmup, sampler, build_dir,
                )
            results.append(res)
            if res["status"] != "PASS":
                any_failed = True
                print(f"    FAIL: {res['error']}", file=sys.stderr, flush=True)
            elif kind == "runtime":
                print(
                    f"    PASS: eon={res['eon']['median_ms']:.2f}ms  "
                    f"rust={res['rust']['median_ms']:.2f}ms  "
                    f"ratio={res['ratio_eon_over_rust']:.3f}x  "
                    f"checksum={res['checksum']}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"    PASS: cold={res['cold']['ratio_eon_over_rust']:.3f}x  "
                    f"inc={res['incremental']['ratio_eon_over_rust']:.3f}x",
                    file=sys.stderr,
                    flush=True,
                )

    runtime_ratios = [
        r["ratio_eon_over_rust"]
        for r in results
        if r["kind"] == "runtime"
        and r["status"] == "PASS"
        and r["ratio_eon_over_rust"] > 0
        and not math.isnan(r["ratio_eon_over_rust"])
    ]
    gmean = (
        math.exp(sum(math.log(x) for x in runtime_ratios) / len(runtime_ratios))
        if runtime_ratios
        else None
    )

    compile_entry = next((r for r in results if r["kind"] == "compile" and r["status"] == "PASS"), None)
    summary: Dict[str, Any] = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] != "PASS"),
        "runtime_passed": len(runtime_ratios),
        "runtime_geomean_eon_over_rust": gmean,
        "compile_cold_ratio": compile_entry["cold"]["ratio_eon_over_rust"] if compile_entry else None,
        "compile_inc_ratio": compile_entry["incremental"]["ratio_eon_over_rust"] if compile_entry else None,
    }

    try:
        eonc_display = str(eonc.relative_to(REPO_ROOT))
    except ValueError:
        eonc_display = eonc.name

    report: Dict[str, Any] = {
        "meta": {
            "timestamp_utc": ts_iso,
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "cpu_model": get_cpu_model(),
            "cpu_count": os.cpu_count(),
            "eonc": eonc_display,
            "eonc_sha256": file_sha256(eonc),
            "eon_flags": args.eon_flags,
            "rustc_version": get_rustc_version(),
            "rust_flags": args.rust_flags,
            "runs": args.runs,
            "warmup": args.warmup,
            "compile_runs": compile_runs,
            "pmu_available": sampler.available,
            "pmu_status": sampler.reason,
        },
        "summary": summary,
        "results": results,
    }

    md_text = render_markdown(report)
    print(md_text)

    if not args.no_save:
        tag = "release" if "--release" in eon_flags else "dev"
        json_path, md_path = resolve_save_paths(args.save, tag, ts_compact)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(md_text + "\n", encoding="utf-8")
        print(f"Saved JSON report: {json_path}", file=sys.stderr)
        print(f"Saved Markdown   : {md_path}", file=sys.stderr)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
