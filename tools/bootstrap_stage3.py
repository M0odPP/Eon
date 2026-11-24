#!/usr/bin/env python3
"""Triple-bootstrap verifier for Eon: eonc0 -> eonc1 -> eonc2 -> eonc3.

Asserts bit-exact reproducibility: SHA-256(eonc2) == SHA-256(eonc3).
Supports both `--dev` (default) and `--release` (Phase 9+) modes.
"""

import argparse
import hashlib
import os
import resource
import subprocess
import sys
import time


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_stage(label: str, cmd: list[str]) -> None:
    code = (
        "import resource, subprocess, sys, time\n"
        "t0 = time.monotonic()\n"
        "p = subprocess.run(sys.argv[1:])\n"
        "dt = time.monotonic() - t0\n"
        "r = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
        "print(f'METRIC wall={dt:.3f}s user={r.ru_utime:.3f}s sys={r.ru_stime:.3f}s "
        "peak_rss_kib={r.ru_maxrss} ({r.ru_maxrss/1024:.1f} MiB)', flush=True)\n"
        "sys.exit(p.returncode)\n"
    )
    print(f"=== {label} ===", flush=True)
    p = subprocess.run(["python3", "-c", code, *cmd], check=False)
    if p.returncode != 0:
        print(f"ERROR: {label} failed with exit code {p.returncode}", file=sys.stderr)
        sys.exit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Eon triple bootstrap (Stage 2 == Stage 3).")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--eonc0", default=None, help="Path to Stage-0 seed compiler (Rust eonc0)")
    ap.add_argument("--out-dir", default=None, help="Output directory for eonc1/eonc2/eonc3")
    ap.add_argument("--release", action="store_true", help="Build Stage 2 and Stage 3 with --release")
    ap.add_argument("--skip-stage1", action="store_true", help="Reuse existing eonc1")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    eonc0 = args.eonc0 or os.path.join(repo, "compiler", "target", "release", "eonc")
    out_dir = args.out_dir or os.path.join(repo, "selfhost", "out")
    os.makedirs(out_dir, exist_ok=True)
    selfhost_src = os.path.join(repo, "selfhost", "src")
    ui_dir = os.path.join(repo, "tests", "ui")

    suffix = "_rel" if args.release else ""
    eonc1 = os.path.join(out_dir, "eonc1")
    eonc2 = os.path.join(out_dir, f"eonc2{suffix}")
    eonc3 = os.path.join(out_dir, f"eonc3{suffix}")

    if not args.skip_stage1:
        run_stage("Stage 1: eonc0 -> eonc1", [eonc0, "build", selfhost_src, "-o", eonc1])
        print(f"eonc1 size={os.path.getsize(eonc1)} sha256={sha256_file(eonc1)}", flush=True)

    s2_flags = ["--release"] if args.release else []
    run_stage(f"Stage 2: eonc1 {' '.join(s2_flags)} -> {os.path.basename(eonc2)}", [eonc1, "build", *s2_flags, selfhost_src, "-o", eonc2])
    h2 = sha256_file(eonc2)
    print(f"{os.path.basename(eonc2)} size={os.path.getsize(eonc2)} sha256={h2}", flush=True)

    run_stage(f"Stage 2 UI test: {os.path.basename(eonc2)} test tests/ui", [eonc2, "test", ui_dir])

    run_stage(f"Stage 3: {os.path.basename(eonc2)} {' '.join(s2_flags)} -> {os.path.basename(eonc3)}", [eonc2, "build", *s2_flags, selfhost_src, "-o", eonc3])
    h3 = sha256_file(eonc3)
    print(f"{os.path.basename(eonc3)} size={os.path.getsize(eonc3)} sha256={h3}", flush=True)

    print("\n=== TRIPLE BOOTSTRAP VERDICT ===", flush=True)
    print(f"SHA-256(Stage 2) = {h2}", flush=True)
    print(f"SHA-256(Stage 3) = {h3}", flush=True)
    if h2 != h3:
        print("MISMATCH: Stage 2 and Stage 3 binaries differ!", file=sys.stderr)
        return 1
    print("OK: Stage 2 == Stage 3 bit-for-bit identical!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
