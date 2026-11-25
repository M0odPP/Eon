# Eon vs Safe Rust Benchmarks (Linux `x86_64`)

This document records reproducible benchmark results comparing the self-hosted Eon compiler (`selfhost/out/eonc2_rel`, `--release`) against Safe Rust (`#![forbid(unsafe_code)]`) across 12 workloads (`bench/01_json_dom` .. `bench/12_compile_speed`) and 3 zero-libc Linux utilities (`utils/`).

---

## Test Environment & Methodology

- **CPU:** AMD Ryzen 5 5600H with Radeon Graphics (12 logical CPUs)
- **OS / Kernel:** `Linux 7.3.5-arch1-1 (x86_64)`
- **Eon Compiler:** `selfhost/out/eonc2_rel` (`--release`, AVX2/FMA native target, zero C / zero libc)
- **Rust Compiler:** `rustc 1.100.0-nightly (5db7f4be8 2026-09-01)`
  - Runtime & utility flags: `-C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C target-cpu=native`
- **Protocol:** 1 warmup + 11 interleaved timed runs (`bench/run_bench.py`), 5 Linux hardware PMU counters (`cycles`, `instructions`, `l1d_misses`, `llc_misses`, `branch_misses`), strict bit-exact output checksum matching on every workload.

### Reproduce Locally

```bash
python3 tools/bootstrap_stage3.py --release
python3 bench/run_bench.py --eonc selfhost/out/eonc2_rel --eon-flags=--release --runs 11 --warmup 1 --compile-runs 3
python3 utils/bench_utils.py --eonc selfhost/out/eonc2_rel
```

---

## 1. Runtime Workloads (`01–11`)

| Workload | Eon med (ms) | Rust med (ms) | Eon / Rust | Eon Instrs | Rust Instrs | Eon IPC | Rust IPC | Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_json_dom` | 48.88 | 55.46 | **0.881x** | 695.65M | 788.70M | 3.05 | 3.40 | `12444077284796102572` |
| `02_eon_lexer_parser` | 59.76 | 60.77 | **0.983x** | 936.50M | 955.93M | 3.63 | 4.01 | `15414754795407181212` |
| `03_utf8_scan` | 57.38 | 75.04 | **0.765x** | 766.30M | 592.75M | 3.47 | 2.01 | `17091793384797853337` |
| `04_ui_layout` | 40.76 | 51.13 | **0.797x** | 477.08M | 362.58M | 3.15 | 1.88 | `9680502862931677198` |
| `05_ecs_physics` | 24.08 | 53.30 | **0.452x** | 272.21M | 169.04M | 2.54 | 0.75 | `12046516819319361567` |
| `06_hashmap_intern` | 68.86 | 73.72 | **0.934x** | 477.14M | 473.81M | 1.57 | 1.60 | `690718933089083097` |
| `07_radix_pdqsort` | 17.90 | 20.63 | **0.868x** | 230.91M | 183.50M | 2.61 | 1.88 | `11023039397254310171` |
| `08_matmul_f32` | 27.93 | 29.03 | **0.962x** | 378.81M | 169.50M | 3.10 | 1.51 | `639651823225821989` |
| `09_nbody_simd` | 36.22 | 41.01 | **0.883x** | 333.65M | 381.37M | 2.30 | 2.34 | `9042696147569759037` |
| `10_par_reduce_graph` | 30.30 | 49.67 | **0.610x** | 1.89G | 1.35G | 1.68 | 1.30 | `4477762117305765334` |
| `11_alloc_churn` | 40.63 | 43.95 | **0.924x** | 421.94M | 370.20M | 2.67 | 2.14 | `12785008313360714948` |

- **Geometric Mean (Eon / Rust across all 11 runtime workloads):** **`0.8056x`** (baseline Phase 9) improved to **`0.7820x`** after M1.1 auto-vectorization, GVN-PRE, and SCEV-lite (`< 1.0000x` = Eon faster; 11/11 workloads faster than Safe Rust; `08_matmul_f32` instruction count reduced `-48.9%` from `378.81M` to `193.5M`).
- **Standalone Multi-Threaded RSS (`10_par_reduce_graph`, fresh `execvp` child, 7 runs):** Eon **`27.96 MiB`** vs Safe Rust **`32.29 MiB`** (`-13.4%` memory). *(Note: `bench/run_bench.py`'s inline `MaxRSS` column includes parent fork high-water RSS overhead on multi-threaded runs.)*

---

## 2. Compilation Speed, Memory Scalability & `.emi` Incremental Builds

### Multi-Module Project (`12_compile_speed`: 21 files, 31,020 LOC Eon vs 31,136 LOC Rust)

| Mode | Eon med (ms) | Rust med (ms) | Speedup |
| :--- | ---: | ---: | ---: |
| Cold full build (`--release` vs Cargo fat-LTO) | **412.45** | 9,374.26 | **22.73×** |
| Incremental (1-fn body edit, `run_bench.py`) | **37.94** | 711.14 | **18.75×** |
| `.emi` unchanged rebuild (standalone) | **15.46** | — | — |
| `.emi` add new `pub fn` in root `mod00` (imported by all 20 modules) | **32.55–50.61** | — | vs 543 ms pre-item-diff |
| `.emi` edit/remove unused `pub fn` in root `mod00` | **67.7–104.4** | — | vs 546–569 ms pre-item-diff |

### Single-File (`01_json_dom`) vs Default `rustc` (Interleaved A/B, 7 Rounds)

| Mode | Eon | `rustc` | Comparison |
| :--- | ---: | ---: | :--- |
| Typecheck (`eonc check` vs `rustc --emit=metadata`) | **44.55 ms** (`56.3 MiB` RSS) | 44.83 ms (`122.3 MiB` RSS) | `1.01×` faster, **`2.17×` less RAM** |
| Release (`eonc build --release` vs `rustc -C opt-level=3`) | **100.15 ms** (`65.8 MiB` RSS) | 222.55 ms (`155.0 MiB` RSS) | **`2.22×` faster**, **`2.36×` less RAM** |

### Self-Hosted Compiler Build (`selfhost/src`, 47 files, ~211.9k LOC — Post-M1.2 Memory Cap)

Measured via `os.wait4` `ru_maxrss` across Stage 2 == Stage 3 triple bootstrap (`selfhost/src` expanded from 30 files / 171.5k LOC to 47 files / 211.9k LOC with AArch64, RISC-V 64, PE32+, Mach-O, APK, `@gpu`, `fmt`, and `lsp` while cutting Peak RSS in half):

- `eonc2_rel build --release selfhost/src` (`auto` threads): **`14.53–16.87 s`**, **`436.8–437.5 MiB` Peak RSS**, binary `15.50 MB` (`-Oz` + GC/ICF reduces binary size by `-15.1%`)
- `eonc2_rel build --release -j 1 selfhost/src`: **`27.13 s`**, **`393.6 MiB` Peak RSS** (bit-exact `SHA-256` match with `-j 4`, `-j 12`, and `auto`)
- `eonc2_dev build selfhost/src` (`--dev`): **`33.95 s`**, **`425.2 MiB` Peak RSS**, binary `22.86 MB`

---

## 3. Zero-libc Linux Utilities (`utils/`) & `EonUI` 100k-Widget Stress

All Eon utilities verified with `readelf -d` / `readelf -l` to have **0 `NEEDED` shared libraries and 0 `PT_INTERP`**.

| Utility | Eon Size | Rust Size | Eon Metric | Rust Metric | Comparison |
| :--- | ---: | ---: | :--- | :--- | :--- |
| `ewc` (32 MiB × 20) | **55.5 KiB** | 1.99 MiB | **8.23 GB/s** | 4.53 GB/s | **1.82× faster**, 36× smaller |
| `egrep` literal (32 MiB × 20) | **61.7 KiB** | 2.00 MiB | **11.70 GB/s** | 3.30 GB/s | **3.55× faster**, 32× smaller |
| `egrep` regex (32 MiB × 3) | **61.7 KiB** | 2.00 MiB | **5.77 GB/s** | 1.95 GB/s | **2.95× faster** |
| `ehttpd` (4-core `io_uring`, 96k reqs) | **76.4 KiB** | 2.18 MiB | `377.6k RPS`, `p99=233.0 us`, **`616 KiB` RSS** | `394.5k RPS`, `p99=249.4 us`, `3,028 KiB` RSS | Lower `p99`, **`4.9×` less RAM**, `0.957x` RPS |

### `EonUI` GPUI-Class Framework & `EonCalc` Frame Performance (`libs/eon_ui`, `apps/eon_calc`)

| Benchmark / Scenario | Measured Result | Budget / Gate | Notes |
| :--- | ---: | ---: | :--- |
| **100,000 live Flexbox/Grid widgets (`UiTree`)** | **`5.28–7.29 ms/frame`** (`137–189 FPS`) | `< 8.33 ms` (`120 FPS`) | 3-phase `request_layout -> prepaint -> paint`, **0 per-frame heap allocs** |
| **100,000-row virtualized `UniformList`** | **`0.029 ms/frame`** (`34,482 FPS`) | `< 8.33 ms` | O(1) visible-window layout & paint |
| **100,000-node `ArenaPool` layout grid** | **`0.368 ms/frame`** (`2,717 FPS`) | `< 8.33 ms` | Direct SoA `ArenaPool` pass |
| **`EonCalc` live Linux Wayland + Vulkan 1.1+** | `516.0 ms` / 30 frames (`58.1 FPS` VSync) | Live GPU + `rc=0` | `759 KiB` binary, 0 leaks under `--sanitize` |
| **`EonCalc` bare-metal QEMU `nostd` (3 CPUs)** | `3.09 s` (`x86_64`), `4.26 s` (`AArch64`), `4.89 s` (`RISC-V 64`) | Hardware FB boot + `EON_UI_NOSTD_OK` | Identical safe app source, 0 OS, 0 C |

---

## Caveats & Transparency

1. **In-Repo Benchmarks:** All 12 workloads were authored inside this repository and leverage Eon's design strengths (`pool` SoA layout, DBM-verifiable loops, arena allocation, explicit SIMD). Arbitrary third-party Rust programs using complex iterator chains or LLVM-specific vectorization idioms may still run faster under LLVM.
2. **Instruction Density:** M1.1 cut `08_matmul_f32` instruction count by `-48.9%` (`378.81M -> 193.5M` vs Rust's `169.50M`), narrowing the gap against LLVM substantially while keeping 11/11 runtime wins (`0.7820x` geomean). Further general-purpose vectorization breadth across arbitrary scalar loops remains an ongoing optimizer focus.
3. **Compile Benchmark Baseline:** Workload `12_compile_speed` compares Eon `--release` against Cargo with `lto=fat` (matching the runtime benchmark flags); single-file `01_json_dom` above also compares against default `rustc -C opt-level=3` without fat LTO.
