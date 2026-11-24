# Eon Roadmap

This roadmap tracks completed engineering milestones and the upcoming path toward multi-architecture support, `@gpu`, `EonUI`, and `v1.0`.

---

## Completed Milestones

| Phase | Milestone | Status | Key Verification Gate |
| :--- | :--- | :---: | :--- |
| **0–5** | Stage 0 Rust Bootstrap Seed (`compiler/`): lexer, parser, types, Zero-Mercy, borrows, `linear`, DBM verifier, `comptime`, initial x86_64/ELF64/SPIR-V | Done | Zero external crates, `#![forbid(unsafe_code)]`, 0 clippy warnings |
| **5.5** | Reference `EIR-Interp`, `EIR-Lint`, `view struct`, `?T` niches, SIMD `vec[T, N]`, differential & soundness fuzzers | Done | 3-way parity (`Interp == PIE == --sanitize`), 0 missed soundness mutations |
| **6** | **Zero-C Linux `x86_64`:** `core/` vs `std/`, W^X `ELF64` (`PIE`/`TLS`/`.so`/dynamic linker), Linux `vDSO`, TLS slab allocator + `--sanitize`, `clone`/`futex` scheduler, `io_uring`, sandboxed `build.eon`, `--audit-tcb`, removal of C backend | Done | Isolated zero-C `PATH` (no `gcc`/`clang`/`ld`/`as`), 5,000-program diff fuzz |
| **7** | **Self-Hosted Compiler (`selfhost/`):** full compiler rewritten in 100% safe Eon (0 `@hal`/`raw`/`asm`/`extern "C"`) | Done | 923/923 layer-by-layer oracle matches against `eonc0` (`tools/oracle_phase7.py`) |
| **8** | **Bit-Exact Triple Bootstrap (`Stage 2 == Stage 3`):** `compiler/` frozen as Bootstrap Seed Only | Done | `SHA-256(eonc2) == SHA-256(eonc3)` bit-for-bit (`--dev` & `--release`) |
| **9** | **Optimizing Pipeline (`EIR-Opt` + `MachIR` + Multi-Threading + `.emi`):** SSA optimizer, DAG ISel, Linear Scan RegAlloc + IPRA + reg-mem spill folding + FMA fusion, adaptive MT passes, item-level `.emi` interface diff | Done | 11/11 runtime wins vs Safe Rust (`0.8056x` geomean), 16/16 MT determinism matrix, 20,000 diff-fuzz runs |
| **10** | **Bare-Metal Microkernel & Zero-libc Linux Stack:** `EonOS-Micro` (`SMP=2`, 1,000 timer IRQs via `@interrupt @noalloc @stack(512)`, 4-level paging, linear PMM) + `ewc`, `egrep`, `ehttpd` (`io_uring`) | Done | Boots in `qemu-system-x86_64` (`rc=33`), 0 `NEEDED` / 0 `PT_INTERP` via `readelf -d`, `--sanitize` verified |

---

## Upcoming Roadmap

### Phase 11 — Optimizer Depth, Memory Scalability & Developer Toolchain
- **Universal Auto-Vectorization & Code Density:** General-purpose SLP and loop auto-vectorizer in `eir_opt.eon`, SCEV-lite strength reduction, `-Oz` code-size profile, function dead-stripping, and Identical Code Folding (ICF).
- **Memory-Bounded Compilation:** Release AST/CFG arenas immediately after EIR lowering and stream codegen batches to bring `selfhost/src` Peak RSS below `450 MiB`, plus multi-package `.emi` library archives and lockfiles (`eon.lock`).
- **Expanded `core/` & `std/`:** Freestanding vector/color/formatting libraries in `core/`; dynamic loader (`std.dl`), filesystem, sockets (`std.net`), clock/time, and synchronization primitives (`std.sync`).
- **Built-in Tooling:** `eonc fmt` canonical formatter and `eonc lsp` incremental Language Server (surfacing proven DBM intervals, exact `@stack(N)` budgets, and Zero-Mercy quick-fixes).

### Phase 12 — Multi-Architecture & Multi-OS Native Backends
- **Target Abstraction:** `--target=<arch>-<os>` with `@target_arch()` / `@target_os()` comptime pruning and target-neutral register files in `MachIR`.
- **CPU Backends:** Native **`AArch64`** (ARM64 + NEON + `AAPCS64`) and **`RISC-V 64`** (`RV64GC` + `LP64D`) alongside `x86_64` (verified via differential fuzzing under `qemu-aarch64` / `qemu-riscv64`).
- **OS Linkers & Runtimes (Zero-C):**
  - **Windows (`x86_64`/`arm64`):** Built-in `PE/COFF` (`.exe`/`.dll`, `.idata`, `.reloc`, `.pdata`/`.xdata`) + Win64 ABI.
  - **macOS & iOS (`arm64`/`x86_64`):** Built-in `Mach-O` + SHA-256 ad-hoc `LC_CODE_SIGNATURE` + `.app` bundles.
  - **Android (`arm64-v8a`/`x86_64`):** `ET_DYN` `libmain.so` + binary `AXML` + 4 KiB `zipalign` + APK Signature Scheme v2.
  - **`nostd` Freestanding:** Unified framebuffer & static allocator across `x86_64`, `AArch64`, and `RISC-V 64`.

### Phase 13 — Unified `@gpu` (`SPIR-V` + `MSL`) & Cross-Platform `EonUI`
- **Unified `@gpu` Compiler in `selfhost/`:** Direct `EIR -> SPIR-V 1.3` (Vulkan on Linux/Windows/Android) and `EIR -> MSL` (Metal on macOS/iOS) with `@layout(std430)` and SIMT uniform-barrier verification.
- **Cross-Platform Graphics & Windowing (`eon_win`, `eon_gpu`, `eon_text`):** Thin `.hal.eon` platform window adapters (`Wayland/X11`, `Win32`, `Cocoa`, `Android NativeActivity`, `UIKit`, `nostd` framebuffer), focused UI-RHI (Vulkan + Metal + pure-Eon SIMD software rasterizer for `nostd`/CI), and pure-Eon TrueType rasterizer + glyph atlas.
- **GPUI-Class UI Framework (`EonUI`):** Retained `SlotPool` entity tree, reactive observers, Flexbox + 2D CSS Grid layout on `ArenaPool`, 3-phase element lifecycle (`request_layout -> prepaint -> paint`), focus/keymap/action dispatch, virtualized lists, and spring animations.

### Phase 14 — `Eon v1.0` Freeze
- Conformance suite (`>= 5,000` tests), continuous multi-architecture fuzzing, and specification stabilization.
