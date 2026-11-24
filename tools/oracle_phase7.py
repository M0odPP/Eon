#!/usr/bin/env python3
"""Layer-by-layer differential oracle comparing eonc0 (Rust) vs eonc1 (Eon).

Runs both compilers across the full Eon verification corpus:
  - tests/ui/      (all 88 UI test cases, preserving pkg_* multi-file packages)
  - core/          (all .eon files)
  - std/           (all .eon files)
  - examples/      (all .eon files + example packages on semantic layers)
  - selfhost/src/  (all .eon files + selfhost/src package on semantic layers)

For every case in every selected layer, compares (exit_code, stdout, stderr)
byte-for-byte and prints unified diffs on any mismatch. Exits 0 iff 100% match.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LAYERS: tuple[str, ...] = ("7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8")

LAYER_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "7.2": ("lex --dump", ("lex", "--dump")),
    "7.3": ("parse --dump-ast-full", ("parse", "--dump-ast-full")),
    "7.4": ("resolve --dump", ("resolve", "--dump")),
    "7.5": ("check --dump-types", ("check", "--dump-types")),
    "7.6": ("check --dump-cfg", ("check", "--dump-cfg")),
    "7.7": ("check --dump-eir", ("check", "--dump-eir")),
    "7.8": ("check --dump-verify", ("check", "--dump-verify")),
}

# Exact non-deterministic wall-clock summary lines emitted by eonc0's parse/resolve/check.
# Stripping ONLY these lines preserves 100% byte-for-byte comparison on all diagnostics.
TIMING_LINE_RE = re.compile(
    rb"^(?:"
    rb"parsed \d+ file\(s\), \d+ total lines in [0-9.]+ ms \(\d+ lines/sec\)"
    rb"|resolved \d+ file\(s\), \d+ module\(s\), \d+ lines in [0-9.]+ ms"
    rb"|checked \d+ file\(s\), \d+ module\(s\), \d+ lines in [0-9.]+ ms"
    rb")\r?\n?$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CorpusCase:
    group: str
    rel_path: str
    use_no_stdlib: bool


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_eonc0(repo_root: Path) -> Path:
    rel = repo_root / "compiler" / "target" / "release" / "eonc"
    if rel.is_file():
        return rel
    return repo_root / "compiler" / "target" / "debug" / "eonc"


def default_eonc1(repo_root: Path) -> Path:
    return repo_root / "selfhost" / "out" / "eonc1"


def _ui_dir_mode(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("check_"):
            return "check"
        if part.startswith("parse_"):
            return "parse"
    return None


def _collect_ui_cases_rec(root: Path, repo_root: Path, out: list[CorpusCase]) -> None:
    if not root.exists():
        return
    if root.is_file():
        if root.name.endswith(".eon"):
            rel = root.relative_to(repo_root).as_posix()
            out.append(CorpusCase(group="ui", rel_path=rel, use_no_stdlib=True))
        return
    for entry in sorted(root.iterdir(), key=lambda p: p.as_posix()):
        if entry.is_dir():
            if entry.name.startswith("pkg_") and _ui_dir_mode(entry) == "check":
                if any(entry.rglob("*.eon")):
                    rel = entry.relative_to(repo_root).as_posix()
                    out.append(CorpusCase(group="ui", rel_path=rel, use_no_stdlib=True))
            else:
                _collect_ui_cases_rec(entry, repo_root, out)
        elif entry.is_file() and entry.name.endswith(".eon"):
            rel = entry.relative_to(repo_root).as_posix()
            out.append(CorpusCase(group="ui", rel_path=rel, use_no_stdlib=True))


def _collect_eon_files(root: Path, repo_root: Path, group: str, use_no_stdlib: bool) -> list[CorpusCase]:
    if not root.exists():
        return []
    files = sorted(
        (p for p in root.rglob("*.eon") if p.is_file()),
        key=lambda p: p.as_posix(),
    )
    return [
        CorpusCase(group=group, rel_path=p.relative_to(repo_root).as_posix(), use_no_stdlib=use_no_stdlib)
        for p in files
    ]


def collect_corpus(repo_root: Path, layer: str, corpus_filter: str) -> list[CorpusCase]:
    cases: list[CorpusCase] = []
    is_semantic_layer = layer in ("7.4", "7.5", "7.6", "7.7", "7.8")

    if corpus_filter in ("all", "ui"):
        _collect_ui_cases_rec(repo_root / "tests" / "ui", repo_root, cases)

    if corpus_filter in ("all", "core"):
        cases.extend(_collect_eon_files(repo_root / "core", repo_root, "core", use_no_stdlib=False))

    if corpus_filter in ("all", "std"):
        cases.extend(_collect_eon_files(repo_root / "std", repo_root, "std", use_no_stdlib=False))

    if corpus_filter in ("all", "examples"):
        ex_root = repo_root / "examples"
        cases.extend(_collect_eon_files(ex_root, repo_root, "examples", use_no_stdlib=False))
        if is_semantic_layer and ex_root.is_dir():
            for subdir in sorted(ex_root.iterdir(), key=lambda p: p.as_posix()):
                if subdir.is_dir() and len(list(subdir.rglob("*.eon"))) > 1:
                    rel = subdir.relative_to(repo_root).as_posix()
                    cases.append(CorpusCase(group="examples", rel_path=rel, use_no_stdlib=False))

    if corpus_filter in ("all", "selfhost"):
        sh_root = repo_root / "selfhost" / "src"
        sh_files = _collect_eon_files(sh_root, repo_root, "selfhost", use_no_stdlib=False)
        cases.extend(sh_files)
        if is_semantic_layer and len(sh_files) > 1:
            rel = sh_root.relative_to(repo_root).as_posix()
            cases.append(CorpusCase(group="selfhost", rel_path=rel, use_no_stdlib=False))

    return cases


def build_command(binary: Path, layer: str, case: CorpusCase) -> list[str]:
    _, base_args = LAYER_SPECS[layer]
    cmd = [str(binary), *base_args]
    if layer in ("7.4", "7.5", "7.6", "7.7", "7.8") and case.use_no_stdlib:
        cmd.append("--no-stdlib")
    cmd.append(case.rel_path)
    return cmd


def normalize_stderr(raw: bytes, layer: str, keep_timing: bool) -> bytes:
    if keep_timing or layer == "7.2":
        return raw
    # Strip only the non-deterministic wall-clock summary lines while preserving
    # every diagnostic byte verbatim.
    lines = raw.splitlines(keepends=True)
    kept = [line for line in lines if not TIMING_LINE_RE.match(line)]
    return b"".join(kept)


def run_compiler(cmd: Sequence[str], cwd: Path, timeout_sec: float) -> RunResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return RunResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode("utf-8", "replace")
        err = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode("utf-8", "replace")
        return RunResult(returncode=-999, stdout=out, stderr=err, timed_out=True)


def first_diff_offset(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def format_stream_diff(stream_name: str, expected: bytes, actual: bytes, max_lines: int = 120) -> str:
    exp_text = expected.decode("utf-8", errors="surrogateescape")
    act_text = actual.decode("utf-8", errors="surrogateescape")
    diff_lines = list(
        difflib.unified_diff(
            exp_text.splitlines(keepends=True),
            act_text.splitlines(keepends=True),
            fromfile=f"eonc0:{stream_name}",
            tofile=f"eonc1:{stream_name}",
            n=3,
        )
    )
    if not diff_lines:
        off = first_diff_offset(expected, actual)
        return (
            f"  [{stream_name}] raw byte mismatch at offset {off} "
            f"(eonc0 len={len(expected)}, eonc1 len={len(actual)})\n"
            f"    eonc0 snippet: {expected[max(0, off - 16):off + 32]!r}\n"
            f"    eonc1 snippet: {actual[max(0, off - 16):off + 32]!r}\n"
        )
    rendered = []
    for line in diff_lines[:max_lines]:
        safe = line.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="replace")
        if not safe.endswith("\n"):
            safe += "\n\\ No newline at end of file\n"
        rendered.append("  " + safe)
    if len(diff_lines) > max_lines:
        rendered.append(f"  ... ({len(diff_lines) - max_lines} more diff lines truncated)\n")
    return "".join(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(
        prog="oracle_phase7.py",
        description="Layer-by-layer byte-exact oracle comparing eonc0 vs eonc1 across Phase 7.2–7.8.",
    )
    parser.add_argument(
        "--layer",
        choices=[*LAYERS, "all"],
        default="all",
        help="Compiler layer to verify (7.2=lex, 7.3=parse, 7.4=resolve, 7.5=types, 7.6=cfg, 7.7=eir, 7.8=verify, all).",
    )
    parser.add_argument(
        "--eonc0",
        type=Path,
        default=default_eonc0(repo_root),
        help="Path to reference Rust compiler eonc0 (default: compiler/target/{release,debug}/eonc).",
    )
    parser.add_argument(
        "--eonc1",
        type=Path,
        default=default_eonc1(repo_root),
        help="Path to self-hosted Eon compiler eonc1 (default: selfhost/out/eonc1).",
    )
    parser.add_argument(
        "--corpus",
        choices=["all", "ui", "core", "std", "examples", "selfhost"],
        default="all",
        help="Subset of corpus to run (default: all).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root directory containing tests/ui, core, std, examples, selfhost.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first mismatch.",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=20,
        help="Maximum number of detailed diffs to print (0 = unlimited, default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-invocation timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--keep-timing",
        action="store_true",
        help="Compare raw stderr without stripping eonc0/eonc1 wall-clock ms timing summary lines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every passing test case.",
    )
    return parser.parse_args(argv)


def verify_binary(label: str, path: Path) -> bool:
    if not path.is_file():
        print(f"error: {label} binary not found at `{path}`", file=sys.stderr)
        return False
    if not os.access(path, os.X_OK):
        print(f"error: {label} binary is not executable: `{path}`", file=sys.stderr)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    eonc0 = args.eonc0.resolve()
    eonc1 = args.eonc1.resolve()

    if not verify_binary("eonc0", eonc0) or not verify_binary("eonc1", eonc1):
        return 2

    selected_layers = LAYERS if args.layer == "all" else (args.layer,)
    total_cases = 0
    total_passed = 0
    total_failed = 0
    printed_diffs = 0
    suppressed_diffs = 0
    t_start = time.monotonic()

    for layer in selected_layers:
        layer_desc, _ = LAYER_SPECS[layer]
        cases = collect_corpus(repo_root, layer, args.corpus)
        layer_passed = 0
        layer_failed = 0
        t_layer = time.monotonic()

        print(f"=== Layer {layer} ({layer_desc}): {len(cases)} case(s) ===")
        for case in cases:
            total_cases += 1
            cmd0 = build_command(eonc0, layer, case)
            cmd1 = build_command(eonc1, layer, case)

            res0 = run_compiler(cmd0, repo_root, args.timeout)
            res1 = run_compiler(cmd1, repo_root, args.timeout)

            err0 = normalize_stderr(res0.stderr, layer, args.keep_timing)
            err1 = normalize_stderr(res1.stderr, layer, args.keep_timing)

            same_rc = (res0.returncode == res1.returncode) and not res0.timed_out and not res1.timed_out
            same_out = res0.stdout == res1.stdout
            same_err = err0 == err1

            if same_rc and same_out and same_err:
                layer_passed += 1
                total_passed += 1
                if args.verbose:
                    print(f"  PASS [{case.group}] {case.rel_path} (exit={res0.returncode})")
                continue

            layer_failed += 1
            total_failed += 1

            should_print = args.max_diffs == 0 or printed_diffs < args.max_diffs
            if should_print:
                printed_diffs += 1
                print(f"FAIL [layer {layer} | {case.group}] {case.rel_path}")
                print(f"  cmd0: {' '.join(cmd0)}")
                print(f"  cmd1: {' '.join(cmd1)}")
                if res0.timed_out or res1.timed_out:
                    print(f"  timeout: eonc0={res0.timed_out}, eonc1={res1.timed_out} (limit={args.timeout}s)")
                if res0.returncode != res1.returncode:
                    print(f"  exit_code: eonc0={res0.returncode}, eonc1={res1.returncode}")
                if not same_out:
                    print(format_stream_diff("stdout", res0.stdout, res1.stdout), end="")
                if not same_err:
                    print(format_stream_diff("stderr", err0, err1), end="")
            else:
                suppressed_diffs += 1

            if args.fail_fast:
                elapsed = time.monotonic() - t_start
                print(
                    f"\nStopped early (--fail-fast) after {total_cases} case(s): "
                    f"{total_passed} passed, {total_failed} failed in {elapsed:.2f}s"
                )
                return 1

        layer_ms = (time.monotonic() - t_layer) * 1000.0
        status = "OK" if layer_failed == 0 else "MISMATCH"
        print(
            f"--- Layer {layer}: {layer_passed}/{len(cases)} matched "
            f"({layer_failed} failed) [{status}] in {layer_ms:.1f} ms ---"
        )

    elapsed = time.monotonic() - t_start
    if suppressed_diffs > 0:
        print(f"\n({suppressed_diffs} additional diff(s) suppressed by --max-diffs {args.max_diffs})")
    print(
        f"\nORACLE SUMMARY: {total_passed}/{total_cases} matched, "
        f"{total_failed} failed across {len(selected_layers)} layer(s) in {elapsed:.2f}s"
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
