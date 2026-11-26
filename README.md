<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/eon-wordmark-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/eon-wordmark-light.svg">
    <img alt="Eon Programming Language" src="assets/brand/eon-wordmark-dark.svg" height="84">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/M0odPP/Eon/actions/workflows/ci.yml"><img src="https://github.com/M0odPP/Eon/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="assets/brand/README.md"><img src="https://img.shields.io/badge/Brand-Assets-111827.svg" alt="Brand Assets"></a>
  <a href="editors/vscode/README.md"><img src="https://img.shields.io/badge/Editor-VS%20Code%20%2B%20LSP-6366F1.svg" alt="VS Code + LSP"></a>
</p>

**Eon** is a self-hosted systems programming language designed around explicit cost, static memory and resource proofs, and a **zero-C, zero-LLVM** native compilation pipeline across **3 CPU architectures × 6 OS targets**, unified `@gpu` shaders (`SPIR-V 1.3` + `MSL 2.4`), and a pure-Eon GPUI-class cross-platform UI stack (`eon_win`, `eon_gpu`, `eon_text`, `eon_ui`).

The self-hosted compiler (`selfhost/src/`, 47 files, ~212k LOC of 100% safe Eon) compiles Eon programs directly into native binaries — **ELF64** (`x86_64` / `AArch64` / `RISC-V 64`, PIE, `.so`, and bare-metal `nostd` kernels), **PE32+** (Windows `.exe` / `.dll`), **Mach-O** (macOS / iOS with built-in SHA-256 ad-hoc `LC_CODE_SIGNATURE` & `.app`), and **Android** `ET_DYN` + `AXML` + `zipalign` + APK Signature Scheme v2 — with DWARF5 debug info, a built-in memory sanitizer (`--sanitize`), deterministic multi-threaded compilation (`-j 1 == -j 4 == -j 12 == auto`), item-level incremental `.emi` builds, built-in `eonc fmt` & `eonc lsp`, and a bit-for-bit reproducible **Stage 2 == Stage 3 triple bootstrap** under `450 MiB` Peak RSS — without invoking `gcc`, `clang`, `ld`, or `as`.

---

## Key Language, Compiler & Ecosystem Features

- **Zero-Mercy Static Safety (`ZM-01`..`ZM-18`)**:
  - **Relational DBM + Known-Bits Verifier:** Proves slice bounds, integer non-overflow, and non-zero division statically (including `min(a.len, b.len)` and `base + k` indexing); unchecked `buf[i]` is only accepted when proven safe, while fallible indexing (`buf.at(i)`) and checked arithmetic (`+?`, `*?`, `/?`) are explicit.
  - **Second-Class Borrows (`*T`, `*mut T`, `view fn`, `view struct`):** Local NLL borrow checking without viral lifetime annotations.
  - **Linear Resource Types + `errdefer`:** Values marked `linear` must be explicitly consumed or destroyed on every normal and error path.
  - **Structure-of-Arrays (`pool`) Built Into the Language:** `ArenaPool[T]` (branded 4-byte `Id[p]`, O(1) arena reset) and `SlotPool[T]` (ABA-safe generational `GenId[T]`), enabling 64-byte aligned column iteration and loop fusion.
  - **Effects, Capabilities & Static Stack Budgets:** Explicit `Env` capabilities (`AllocCap`, `FsCap`, `DirCap`, `ClockCap`, `IoCap`, `ArgsCap`), effect annotations (`@noalloc`, `@noblock`, `@interrupt`), and machine-frame verified `@stack(N)` bounds.
  - **Auditable Trust Boundary (`.hal.eon` + `--audit-tcb`):** Safe `.eon` files cannot contain `raw`, raw pointer dereferences, inline `asm`, or `extern "C"`. All unsafe/OS/hardware code is isolated in `.hal.eon` modules and audited with SHA-256 digests.
- **Multi-Target Optimizing Compiler (`EIR-Opt`, `MachIR`, `@gpu`, Zero-C Linkers)**:
  - **3 CPUs × 6 OSes (`--target=<arch>-<os>`):** Native `x86_64` (SSE2/AVX2/FMA), `AArch64` (ARMv8 + NEON), and `RISC-V 64` (`RV64IMAFD` / `LP64D`) targeting `linux`, `windows`, `macos`, `android`, `ios`, and bare-metal `nostd`, with `@target_arch()` / `@target_os()` / `@target_is_nostd()` dead-branch pruning.
  - **SSA Optimizer & Link-Time Density:** Mem2Reg, SROA, GVN-PRE, SCEV-lite, SLP + loop auto-vectorization, SoA fusion, `-Oz` code-size profile, ELF64 function dead-stripping (GC), and Safe Identical Code Folding (ICF).
  - **Unified `@gpu` Shader Compiler:** Direct `EIR -> SPIR-V 1.3` (Vertex/Fragment/Compute, `spirv-val` verified for Vulkan 1.1+) and `EIR -> MSL 2.4` (Metal) with `@embed_gpu_spirv` / `@embed_gpu_msl`.
  - **Built-in Developer Tooling:** Canonical formatter (`eonc fmt [--check]`), incremental JSON-RPC Language Server (`eonc lsp` surfacing proven DBM intervals & `@stack(N)`), multi-package `build.eon` + `eon.lock` + `.emi` library archives, and reference SSA interpreter (`eonc run --interp`).
- **Pure-Eon Cross-Platform Graphics & GPUI-Class UI Stack (`libs/` + `apps/eon_calc`)**:
  - **`libs/eon_win`:** Unified `PlatformWindow` over thin `.hal.eon` adapters (`Wayland` + `X11` fallback, `Win32`, `Cocoa`, `Android NativeActivity`, `UIKit`, 3-arch bare-metal `nostd` framebuffer, and headless test backend).
  - **`libs/eon_gpu`:** Focused GPUI UI-RHI (`Quad`, `Shadow`, `Underline`, `MonoSprite`, `PolySprite`, `PathVertex`, `ContentMask`) with shared `@gpu` shaders and exactly 3 backends: **Vulkan 1.1+**, **Metal**, and **pure-Eon SIMD software rasterizer (`eon_gpu_soft`)** — **zero OpenGL, zero C toolchain**.
  - **`libs/eon_text`:** Freestanding `core/` TrueType (`.ttf`) parser, analytical anti-aliased Bezier scanline rasterizer, shelf-packed R8 `GlyphAtlas`, UTF-8 shaper/wrapper/hit-tester, and embedded SIL OFL UI fonts.
  - **`libs/eon_ui`:** GPUI-class framework built on Eon's `SlotPool` (`Entity[T]` / `AppContext` reactive observers without `Rc<RefCell>`), Taffy-class Flexbox + 2D CSS Grid SoA & `ArenaPool` layout, 3-phase lifecycle (`request_layout -> prepaint -> paint`), fluent `div()`, `FocusTree`/`Keymap`/`Hitbox`, `Spring`/`Tween` animations, Dark/Light `Theme`, and widgets (`Text`, `TextInput`, `ScrollView`, 100k virtualized `UniformList`, `Button`, `Toggle`, `Overlay`) — sustaining **`> 135 FPS` (`5.3–7.3 ms/frame`) on `100,000` live Flexbox/Grid widgets** with 0 per-frame heap allocations.
  - **`apps/eon_calc`:** Full Standard / Scientific / Programmer (`HEX`/`DEC`/`OCT`/`BIN`) GUI calculator written in **100% safe Eon (0 `@hal`, 0 `raw`, 0 `@target_*`)**, compiling from identical source across all 10 targets (verified live on Linux Wayland+Vulkan, Windows PE32+ under Wine, and bare-metal QEMU `x86_64-nostd` / `aarch64-nostd` / `riscv64-nostd`).

---

## Language Taste

```eon
module examples.hello;

import core.mem;
import std.rt;

pub pool Body {
    pub x: f32;
    pub vx: f32;
}

pub fn integrate(bodies: *mut ArenaPool[Body], dt: f32) @noalloc {
    // DBM proves `i < bodies.len()` -> zero hidden bounds checks on SoA columns
    for i in 0_usize..bodies.len() {
        bodies.x[i] = bodies.x[i] + bodies.vx[i] * dt;
    }
}

pub fn main(env: *mut Env) -> i32 ! MemError {
    // Linear FrameArena MUST be destroyed on both normal and error paths
    var frame = try mem.FrameArena.init(&mut env.alloc, 4096_usize);
    errdefer frame.destroy(&mut env.alloc);

    var bodies = try mem.init_frame_pool.[Body](&mut frame, 4_usize);
    let _ = try bodies.push(.{ x: 1.0_f32, vx: 2.5_f32 });
    integrate(&mut bodies, 0.5_f32);

    rt.print_str("Integrated bodies: ");
    rt.print_u64(bodies.len().widen.[u64]());
    rt.print_nl();

    frame.destroy(&mut env.alloc);
    return 0_i32;
}
```

---

## Quick Start

Rust (`cargo`) is required **only** to build the frozen Stage 0 Bootstrap Seed (`compiler/`, zero external crates). Once bootstrapped, `selfhost/out/eonc2_rel` compiles and links native binaries for all 10 targets with zero C toolchain.

```bash
# 1. Build Stage 0 Bootstrap Seed (Rust, 0 external crates, frozen)
cargo build --release --manifest-path compiler/Cargo.toml

# 2. Run full Triple Bootstrap (eonc0 -> eonc1 -> eonc2_rel == eonc3_rel, < 450 MiB RSS)
python3 tools/bootstrap_stage3.py --release

# 3. Build & run EonCalc natively on Linux (Wayland/X11 + Vulkan 1.1+)
selfhost/out/eonc2_rel build --release --audit-tcb apps/eon_calc -o /tmp/eon_calc
/tmp/eon_calc

# 4. Cross-compile the EXACT SAME apps/eon_calc source to any target (Zero-C)
selfhost/out/eonc2_rel build --release --target=x86_64-windows apps/eon_calc -o /tmp/eon_calc.exe
selfhost/out/eonc2_rel build --release --target=aarch64-macos apps/eon_calc -o /tmp/eon_calc_macos
selfhost/out/eonc2_rel build --release --target=aarch64-android --shared apps/eon_calc -o /tmp/libeon_calc.so
selfhost/out/eonc2_rel build --release --target=x86_64-nostd apps/eon_calc -o /tmp/eon_calc_x64_nostd
selfhost/out/eonc2_rel build --release --target=aarch64-nostd apps/eon_calc -o /tmp/eon_calc_a64_nostd
selfhost/out/eonc2_rel build --release --target=riscv64-nostd apps/eon_calc -o /tmp/eon_calc_rv64_nostd

# 5. Built-in formatter, LSP server, sanitizer & EIR interpreter
selfhost/out/eonc2_rel fmt --check apps/eon_calc/src/engine.eon
selfhost/out/eonc2_rel lsp
selfhost/out/eonc2_rel build --sanitize apps/eon_calc -o /tmp/eon_calc_san
selfhost/out/eonc2_rel run --interp examples/layout_proto
```

### Verification & Bare-Metal Microkernel

```bash
# UI diagnostic suite (88 tests) + full selfhost regression suite (13 suites covering M1.1–M2.5)
selfhost/out/eonc2_rel test tests/ui
python3 tests/selfhost/run_regressions.py --eonc selfhost/out/eonc2_rel

# Differential & soundness mutation fuzzing
python3 tests/fuzz/soundness_fuzz.py -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel
python3 tests/fuzz/diff_backends.py --release --verify-opt -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel

# Boot EonOS-Micro (freestanding x86_64 microkernel: SMP=2, 1,000 timer IRQs, 4-level paging)
python3 kernel/run_qemu.py --eonc selfhost/out/eonc2_rel
```

---

## Benchmark Summary (Eon vs Safe Rust & EonUI 100k Stress)

Measured on Linux `x86_64` (AMD Ryzen 5 5600H, 12 logical CPUs) comparing `selfhost/out/eonc2_rel --release` against Safe Rust (`#![forbid(unsafe_code)]`, `-C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C target-cpu=native`), with bit-identical output checksums. Full tables and methodology are in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).

| Category | Metric | Eon (`--release`) | Safe Rust / Target | Result |
| :--- | :--- | ---: | ---: | :--- |
| **Runtime (11 workloads)** | Geometric mean (`01`–`11`) | **11 / 11 faster** | baseline | **`0.7820x`** (~21.8% faster overall) |
| `08_matmul_f32` | Dense AVX2/FMA auto-vec matmul | `193.5M` instrs (`-48.9%` vs pre-M1.1) | `169.5M` instrs | Faster wall time, major instr gap reduction |
| `05_ecs_physics` | 200k-body SoA `sys` loop | 24.08 ms | 53.30 ms | **`0.452x`** (2.21× faster) |
| `10_par_reduce_graph` | Multi-core `par for` + graph | 30.30 ms (`27.96 MiB` RSS*) | 49.67 ms (`32.29 MiB` RSS*) | **`0.610x`** (1.64× faster, `-13.4%` RAM) |
| **Compile (`selfhost/src`, 212k LOC)** | Stage 3 `--release` Peak RSS | **`437.5 MiB`** (`393.6 MiB` `-j 1`) | `< 450 MiB` budget | **`~2×` lower** than pre-M1.2 `865 MiB` |
| **Compile (31k LOC, 21 files)** | Cold `--release` vs Cargo fat-LTO | **412.45 ms** | 9,374.26 ms | **`22.73×` faster** |
| **Compile (31k LOC, 21 files)** | Incremental 1-fn edit (`.emi`) | **37.94 ms** | 711.14 ms | **`18.75×` faster** |
| **Zero-libc Utilities (`utils/`)** | `ewc` / `egrep` (32 MiB × 20) | **8.23 / 11.70 GB/s** (`55–62 KiB`) | 4.53 / 3.30 GB/s (`2.0 MiB`) | **1.82× / 3.55× faster**, 32–36× smaller |
| **Zero-libc HTTP (`ehttpd`)** | 4-core `io_uring` (96k reqs) | `377.6k RPS`, `p99=233 us`, **`616 KiB` RSS** | `394.5k RPS`, `p99=249 us`, `3,028 KiB` RSS | Lower `p99`, **`4.9×` less RAM**, `0.957x` RPS |
| **`EonUI` 100k Widget Stress** | 100,000 live Flexbox/Grid nodes | **`5.28–7.29 ms/frame`** (`137–189 FPS`) | `< 8.33 ms` (120 FPS) | **0 per-frame heap allocs**, 0 leaks under `--sanitize` |
| **`EonUI` 100k Virtualized List** | `UniformList` (100,000 rows) | **`0.029 ms/frame`** (`34,482 FPS`) | `< 8.33 ms` | O(1) visible-window layout & paint |

<sub>* Standalone fresh-process `MaxRSS` median (7 runs). Full caveats and measurement details are documented transparently in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).</sub>

---

## Current Status & Honest Limitations

Eon is **pre-1.0**. All 14 milestones across compiler optimization, multi-target codegen (`3 CPUs × 6 OSes`), `@gpu` (`SPIR-V` + `MSL`), `eon_win`, `eon_gpu`, `eon_text`, `eon_ui`, and `apps/eon_calc` are implemented and verified, with transparent boundaries on host testing:

- **Physical Apple & Mobile Device Testing:** `aarch64-macos` / `aarch64-ios` Mach-O executables (with SHA-256 `LC_CODE_SIGNATURE` and MSL 2.4 shaders) and `aarch64-android` `.so`/APK v2 binaries are cross-compiled and structurally/cryptographically verified on Linux; live GPU execution on physical Apple Silicon (Metal) and physical Android/iOS hardware awaits dedicated hardware CI runners.
- **No Package Registry Yet:** Multi-package local path dependencies, `eon.lock`, and `.emi` library archives are built into `build.eon`, while a centralized remote package registry is planned for `v1.0`.
- **Verification Maturity:** Guarded by bit-exact triple bootstrap, 13 regression suites, 88+ UI diagnostic tests, 140+ calculator unit/UI golden tests, and multi-architecture differential/soundness fuzzing, Eon has not yet undergone years of external production hardening.

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Repository map, compiler pipeline (`selfhost/src/`), multi-target backends, `@gpu`, `libs/` UI stack, and `.hal.eon` TCB model.
- **[ROADMAP.md](ROADMAP.md)** — Completed phases (`0–13`, including Milestones M1.1–M4.2) and Phase 14 (`v1.0`).
- **[bench/BENCHMARKS.md](bench/BENCHMARKS.md)** — Full Eon vs Safe Rust benchmark tables, PMU counters, `EonUI` 100k stress timings, and methodology caveats.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** & **[SECURITY.md](SECURITY.md)** — Development workflow, Stage 0 freeze policy, and soundness reporting.

### Language Specification (`v0.2`)

1. [spec/00-philosophy-and-guarantees.md](spec/00-philosophy-and-guarantees.md) — Zero-Mercy rules (`ZM-01`..`ZM-18`), safety boundaries, glossary.
2. [spec/01-lexical-and-syntax.md](spec/01-lexical-and-syntax.md) — Lexical rules, syntax, Pratt precedence table, modules, visibility.
3. [spec/grammar.ebnf](spec/grammar.ebnf) — Formal EBNF grammar for Eon `v0.2`.
4. [spec/02-types-and-memory-layout.md](spec/02-types-and-memory-layout.md) — Type system, SIMD, `bitfield`, SoA `pool`, UTF-8 strings, copy threshold (`<= 16 B`).
5. [spec/03-ownership-borrows-linearity.md](spec/03-ownership-borrows-linearity.md) — Second-class references, `view fn`, `view struct`, closures, `linear` + `errdefer`, `ArenaPool` & `SlotPool`.
6. [spec/04-verification-and-arithmetic.md](spec/04-verification-and-arithmetic.md) — Relational DBM verifier, `where` contracts, `buf[i]` vs `buf.at(i)`, arithmetic, `core.math`, error unions `! E`.
7. [spec/05-generics-interfaces-comptime.md](spec/05-generics-interfaces-comptime.md) — Generics `[T]`, Layout Sharing, `iface`/`impl` with `@stack(N)`, `comptime` VM, `build.eon`.
8. [spec/06-concurrency-and-memory-model.md](spec/06-concurrency-and-memory-model.md) — `sys`, `par for`, 1-thread-per-core scheduler, atomics, DMA barriers.
9. [spec/07-effects-capabilities-io.md](spec/07-effects-capabilities-io.md) — Effects (`@noalloc`, `@noblock`, `@interrupt`), `Env` capabilities, `IoRing`, static fibers.
10. [spec/08-hal-ffi-and-gpu.md](spec/08-hal-ffi-and-gpu.md) — `.hal.eon` isolation, `MmioReg`, `arch/`, C FFI (`extern "C"`), and `@gpu`.
11. [spec/09-abi-modules-and-hotreload.md](spec/09-abi-modules-and-hotreload.md) — Native `EonCall` ABI, `EonBin`, `.emi` module interface format, hot-reload.
12. [spec/10-compiler-ir-and-bootstrap.md](spec/10-compiler-ir-and-bootstrap.md) — `eonc` architecture, SSA `EIR`, native backends, Zero-C linkers, `--sanitize`, `--audit-tcb`, and triple bootstrap.
13. [spec/CHANGES-v0.2.md](spec/CHANGES-v0.2.md) — Summary of `v0.2` design decisions.

---

## License

Eon is released under the [MIT License](LICENSE).
