# Contributing to Eon

Thank you for your interest in Eon!

## Core Engineering Rules

1. **Frozen Bootstrap Seed (`compiler/`):**
   - The Rust compiler in `compiler/` (`eonc0`, zero external crates, `#![forbid(unsafe_code)]`) is frozen as the **Stage 0 Bootstrap Seed** (see `compiler/BOOTSTRAP_SEED.md`).
   - All new compiler passes, optimizer improvements, language features, and target backends belong in `selfhost/src/` written in 100% safe Eon (zero `@hal`, `raw`, `asm`, or `extern "C"`).
   - Every change to `selfhost/src/` must remain compilable by `eonc0` so clean bootstrapping from source never breaks.
2. **Zero-C Toolchain:**
   - Do not introduce dependencies on `gcc`, `clang`, `ld`, `as`, or `libc` for compiling or running Eon user programs on Linux `x86_64`.
3. **100% Deterministic Builds:**
   - Multi-threaded compilation (`-j 1`, `-j 4`, `-j 12`, `auto`) and triple bootstrap (`Stage 2 == Stage 3`) must remain bit-for-bit identical in both `--dev` and `--release`.
4. **Honest Benchmarks & Verification:**
   - Never claim speedups without reproducible measurements (`bench/run_bench.py`) and identical output checksums.

## Local Development Workflow

```bash
# 1. Build Stage 0 Bootstrap Seed (Rust)
cargo build --release --manifest-path compiler/Cargo.toml

# 2. Run full Triple Bootstrap (eonc0 -> eonc1 -> eonc2 == eonc3)
python3 tools/bootstrap_stage3.py --release
python3 tools/bootstrap_stage3.py --skip-stage1

# 3. Run UI tests & selfhost soundness regressions
selfhost/out/eonc2_rel test tests/ui
python3 tests/selfhost/run_regressions.py --eonc selfhost/out/eonc2_rel

# 4. Run differential & soundness fuzzers
python3 tests/fuzz/soundness_fuzz.py -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel
python3 tests/fuzz/diff_backends.py --release --verify-opt -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel

# 5. Boot EonOS-Micro in QEMU (requires qemu-system-x86_64)
python3 kernel/run_qemu.py --eonc selfhost/out/eonc2_rel
```

## Pull Request Checklist

- [ ] `compiler/` was not modified (unless fixing a verified Stage 0 seed build break).
- [ ] `python3 tools/bootstrap_stage3.py --release` passes (`SHA-256(Stage 2) == SHA-256(Stage 3)`).
- [ ] `python3 tests/selfhost/run_regressions.py --eonc selfhost/out/eonc2_rel` passes.
- [ ] Differential and soundness fuzzers report `0` mismatches / `0` missed mutations.
- [ ] All documentation and comments are written in clear technical English.
