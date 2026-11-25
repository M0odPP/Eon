# Eon Roadmap

This roadmap tracks all completed engineering phases (`0–13`, including all 14 milestones of the multi-target compiler, `@gpu`, and cross-platform GPUI-class UI stack) and the remaining path to `v1.0`.

---

## Completed Milestones

| Phase | Milestone | Status | Key Verification Gate |
| :--- | :--- | :---: | :--- |
| **0–5** | Stage 0 Rust Bootstrap Seed (`compiler/`): lexer, parser, types, Zero-Mercy, borrows, `linear`, DBM verifier, `comptime`, initial x86_64/ELF64/SPIR-V | Done | Zero external crates, `#![forbid(unsafe_code)]`, 0 clippy warnings (FROZEN) |
| **5.5** | Reference `EIR-Interp`, `EIR-Lint`, `view struct`, `?T` niches, SIMD `vec[T, N]`, differential & soundness fuzzers | Done | 3-way parity (`Interp == PIE == --sanitize`), 0 missed soundness mutations |
| **6** | **Zero-C Linux `x86_64`:** `core/` vs `std/`, W^X `ELF64` (`PIE`/`TLS`/`.so`), Linux `vDSO`, TLS slab allocator + `--sanitize`, `clone`/`futex` scheduler, `io_uring`, sandboxed `build.eon`, `--audit-tcb` | Done | Isolated zero-C `PATH` (no `gcc`/`clang`/`ld`/`as`), 5,000-program diff fuzz |
| **7** | **Self-Hosted Compiler (`selfhost/`):** full compiler rewritten in 100% safe Eon (0 `@hal`/`raw`/`asm`/`extern "C"`) | Done | 923/923 layer-by-layer oracle matches against `eonc0` (`tools/oracle_phase7.py`) |
| **8** | **Bit-Exact Triple Bootstrap (`Stage 2 == Stage 3`):** `compiler/` frozen as Bootstrap Seed Only | Done | `SHA-256(eonc2) == SHA-256(eonc3)` bit-for-bit (`--dev` & `--release`) |
| **9** | **Optimizing Pipeline (`EIR-Opt` + `MachIR` + MT + `.emi`):** SSA optimizer, DAG ISel, Linear Scan RegAlloc + IPRA + FMA fusion, adaptive MT passes, item-level `.emi` diff | Done | 11/11 runtime wins vs Safe Rust, 16/16 MT determinism matrix, 20,000 diff-fuzz runs |
| **10** | **Bare-Metal Microkernel & Zero-libc Linux Stack:** `EonOS-Micro` (`SMP=2`, 1,000 timer IRQs, 4-level paging) + `ewc`, `egrep`, `ehttpd` (`io_uring`) | Done | Boots in `qemu-system-x86_64` (`rc=33`), 0 `NEEDED` / 0 `PT_INTERP`, `--sanitize` verified |
| **11 (M1.1–M1.4)** | **Optimizer Depth, `< 450 MiB` RAM, Expanded `core/`+`std/`, `fmt` & `lsp`:** SLP + loop auto-vectorizer, GVN-PRE, SCEV-lite, `-Oz`, ELF64 function GC + Safe ICF, AST/CFG arena release & EIR compaction, multi-package `eon.lock`/`.emi` libs + `impl` stub diff, `core/{math_vec,color,fmt}`, `std/{dl,fs,time,net,sync}`, `eonc fmt` & `eonc lsp` | Done | `08_matmul_f32` instructions `-48.9%` (`378.8M -> 193.5M`), 11/11 runtime wins (`0.7820x` geomean), `selfhost/src` Peak RSS `< 450 MiB`, `fmt`/`lsp` automated tests pass |
| **12 (M2.1–M2.5)** | **Multi-Target `3 CPUs x 6 OSes` & Unified `@gpu`:** `TargetSpec` + `--target=<arch>-<os>` + `@target_*` comptime pruning, 32-reg `MachIR`/`RegAlloc`, native `AArch64` (ARMv8+NEON) & `RISC-V 64` (RV64GC/LP64D), Zero-C `PE32+` (Windows), `Mach-O` + SHA-256 `LC_CODE_SIGNATURE` (macOS/iOS), Android `ET_DYN` + `AXML` + APK v2 ECDSA, 3-arch `nostd` framebuffers, unified `@gpu` (`SPIR-V 1.3` + `MSL 2.4` + `@embed_gpu_*`) | Done | Multi-arch diff fuzz (`Interp == x86_64 == qemu-aarch64 == qemu-riscv64`), live `wine` Win64, 3-arch QEMU `nostd` boots, `spirv-val` PASS on all shaders |
| **13A (M3.1–M3.3)** | **Cross-Platform Windowing, Focused UI-RHI & TrueType Engine (Zero OpenGL):** `libs/eon_win` (Wayland/X11, Win32, Cocoa, Android, UIKit, `nostd` FB), `libs/eon_gpu` (Vulkan 1.1+, Metal, pure-Eon SIMD `eon_gpu_soft`), `libs/eon_text` (TrueType `.ttf` parser, Bezier scanline rasterizer, R8 `GlyphAtlas`, UTF-8 shaper/wrapper/hit-tester, embedded SIL OFL fonts) | Done | Live Linux Wayland + Vulkan (`~143 FPS`), live `wine` Win64 Vulkan, 3-arch QEMU `nostd` text+UI boots, 10-target cross-build, 0 leaks under `--sanitize` |
| **13B (M4.1–M4.2)** | **GPUI-Class UI Framework (`libs/eon_ui`) & Cross-Platform Calculator (`apps/eon_calc`):** `SlotPool` `Entity[T]`/`AppContext`, Taffy-class Flexbox + 2D CSS Grid SoA + `ArenaPool`, 3-phase lifecycle, fluent `div()`, `FocusTree`/`Keymap`/`Hitbox`, `Spring`/`Tween`, Dark/Light `Theme`, widgets (`Text`, `TextInput`, `ScrollView`, 100k virtualized `UniformList`, `Button`, `Toggle`, `Overlay`), and `apps/eon_calc` (Standard/Scientific/Programmer, 0 `@hal`/`raw`/`@target_*`) | Done | 100k live Flexbox/Grid widgets in `5.3–7.3 ms/frame` (`> 135 FPS`), 100k `UniformList` in `0.029 ms/frame`, 140/140 unit+golden UI tests (`--release` & `--sanitize`, 0 leaks), live Wayland+Vulkan, live `wine` Win64, 3-arch QEMU `nostd` boots, 10/10 target builds |

---

## Upcoming Roadmap

### Phase 14 — `Eon v1.0` Freeze & Ecosystem Hardening
- **Conformance Suite:** Expand language, verifier, multi-target backend, and `@gpu` conformance suite to `>= 5,000` tests.
- **Hardware CI Runners:** Continuous native CI execution on physical Apple Silicon (macOS/iOS Metal), Windows 11 ARM64/x64, Android ARM64 devices, and RISC-V 64 boards (supplementing current Linux Vulkan, Wine Win64, `spirv-val`, and QEMU multi-arch verification).
- **Specification Stabilization:** Freeze Eon `v1.0` grammar, `EonCall` ABI, `.emi` archive format, and `EonUI` public APIs.
