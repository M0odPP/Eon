# Eon Programming Language

[![CI](https://github.com/M0odPP/Eon/actions/workflows/ci.yml/badge.svg)](https://github.com/M0odPP/Eon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Eon** is a self-hosted systems programming language designed around explicit cost, static memory and resource proofs, and a **zero-C, zero-LLVM, zero-libc** native compilation pipeline.

On Linux `x86_64`, the self-hosted compiler (`selfhost/src/`, ~171k LOC of 100% safe Eon) compiles Eon programs directly to **W^X ELF64** executables (`ET_DYN` PIE, `.so` shared libraries, or `--freestanding` Multiboot2/PVH bare-metal kernels) with DWARF5 debug info, a built-in memory sanitizer (`--sanitize`), deterministic multi-threaded compilation (`-j 1 == -j 4 == -j 12 == auto`), item-level incremental `.emi` builds, and a bit-for-bit reproducible **Stage 2 == Stage 3 triple bootstrap** — without invoking `gcc`, `clang`, `ld`, `as`, or linking `libc`.

---

## Key Language & Compiler Features

- **Zero-Mercy Static Safety (`ZM-01`..`ZM-18`)**:
  - **Relational DBM + Known-Bits Verifier:** Proves slice bounds, integer non-overflow, and non-zero division statically; unchecked `buf[i]` is only accepted when proven safe, while fallible indexing (`buf.at(i)`) and checked arithmetic (`+?`, `*?`) are explicit.
  - **Second-Class Borrows (`*T`, `*mut T`, `view fn`, `view struct`):** Local NLL borrow checking without viral lifetime annotations.
  - **Linear Resource Types + `errdefer`:** Values marked `linear` must be explicitly consumed or destroyed on every normal and error path.
  - **Structure-of-Arrays (`pool`) Built Into the Language:** `ArenaPool[T]` (branded 4-byte `Id[p]`, O(1) arena reset) and `SlotPool[T]` (ABA-safe generational `GenId[T]`), enabling 64-byte aligned column iteration and loop fusion.
  - **Effects, Capabilities & Static Stack Budgets:** Explicit `Env` capabilities (`AllocCap`, `FsCap`, `DirCap` backed by `openat2(RESOLVE_BENEATH)`, `ClockCap` via Linux `vDSO`, `IoCap` via `io_uring`, `ArgsCap`), effect annotations (`@noalloc`, `@noblock`, `@interrupt`), and machine-frame verified `@stack(N)` bounds.
  - **Auditable Trust Boundary (`.hal.eon` + `--audit-tcb`):** Safe `.eon` files cannot contain `raw`, raw pointer dereferences, inline `asm`, or `extern "C"`. All unsafe/OS/hardware code is isolated in `.hal.eon` modules and audited with SHA-256 digests.
- **Self-Hosted Optimizing Compiler (`EIR-Opt` + `MachIR`)**:
  - Multi-threaded frontend, verifier, optimizer, and native x86_64 backend with bit-exact determinism across thread counts.
  - Reference SSA interpreter (`eonc run --interp`) serving as a semantic oracle for differential fuzzing (`Interp == --dev == --release == --sanitize`).
  - Built-in ELF64 linker, DWARF5 emitter (`gdb`-verified), TLS slab allocator, `clone`/`futex` work-stealing scheduler (`sys` DAGs & `par for`), and item-level `.emi` incremental cache.

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

## Quick Start (Linux `x86_64`)

Rust (`cargo`) is required **only** to build the frozen Stage 0 Bootstrap Seed (`compiler/`, zero external crates). Once bootstrapped, `selfhost/out/eonc2_rel` compiles and links Eon binaries with zero C toolchain or `libc`.

```bash
# 1. Build Stage 0 Bootstrap Seed (Rust, 0 external crates)
cargo build --release --manifest-path compiler/Cargo.toml

# 2. Run full Triple Bootstrap (eonc0 -> eonc1 -> eonc2_rel == eonc3_rel)
python3 tools/bootstrap_stage3.py --release

# 3. Compile & run a Zero-C native PIE executable (or run via the reference EIR interpreter)
selfhost/out/eonc2_rel build --release examples/layout_proto -o /tmp/layout_proto
/tmp/layout_proto
selfhost/out/eonc2_rel run --interp examples/layout_proto

# 4. Run with the built-in memory sanitizer & audit the Trusted Computing Base
selfhost/out/eonc2_rel build --sanitize examples/layout_proto -o /tmp/layout_proto_san
/tmp/layout_proto_san
selfhost/out/eonc2_rel build --audit-tcb selfhost/src
```

### Verification & Bare-Metal Microkernel

```bash
# UI diagnostic tests (88 tests) + selfhost regressions (.emi, @stack(N), DWARF5 + gdb)
selfhost/out/eonc2_rel test tests/ui
python3 tests/selfhost/run_regressions.py --eonc selfhost/out/eonc2_rel

# Differential & soundness mutation fuzzing
python3 tests/fuzz/soundness_fuzz.py -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel
python3 tests/fuzz/diff_backends.py --release --verify-opt -n 1000 --jobs 6 --eonc selfhost/out/eonc2_rel

# Boot EonOS-Micro (48.7 KiB W^X freestanding x86_64 kernel: SMP=2, 1,000 timer IRQs, 4-level paging)
python3 kernel/run_qemu.py --eonc selfhost/out/eonc2_rel
```

---

## Benchmark Summary (Eon vs Safe Rust)

Measured on Linux `7.3.5` `x86_64` (AMD Ryzen 5 5600H, 12 logical CPUs) comparing `selfhost/out/eonc2_rel --release` against Safe Rust (`#![forbid(unsafe_code)]`, `rustc 1.100.0-nightly`, `-C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C target-cpu=native`), 1 warmup + 11 interleaved runs with bit-identical output checksums. Full PMU tables and methodology are in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).

| Category | Metric | Eon (`--release`) | Safe Rust | Result |
| :--- | :--- | ---: | ---: | :--- |
| **Runtime (11 workloads)** | Geometric mean (`01`–`11`) | **11 / 11 faster** | baseline | **`0.8056x`** (~19.4% faster overall) |
| `05_ecs_physics` | 200k-body SoA `sys` loop | 24.08 ms | 53.30 ms | **`0.452x`** (2.21× faster) |
| `10_par_reduce_graph` | Multi-core `par for` + graph | 30.30 ms (`27.96 MiB` RSS*) | 49.67 ms (`32.29 MiB` RSS*) | **`0.610x`** (1.64× faster, `-13.4%` RAM) |
| `08_matmul_f32` | Dense AVX2/FMA `f32` matmul | 27.93 ms | 29.03 ms | **`0.962x`** (Eon faster) |
| **Compile (31k LOC, 21 files)** | Cold `--release` vs Cargo fat-LTO | **412.45 ms** | 9,374.26 ms | **`22.73×` faster** |
| **Compile (31k LOC, 21 files)** | Incremental 1-fn edit (`.emi`) | **37.94 ms** | 711.14 ms | **`18.75×` faster** |
| **Compile (single file `01`)** | `check` vs `rustc --emit=metadata` | **44.55 ms** (`56.3 MiB`) | 44.83 ms (`122.3 MiB`) | `1.01×` faster, **`2.17×` less RAM** |
| **Compile (single file `01`)** | `--release` vs `rustc -O3` | **100.15 ms** (`65.8 MiB`) | 222.55 ms (`155.0 MiB`) | **`2.22×` faster**, **`2.36×` less RAM** |
| **Zero-libc Utilities (`utils/`)** | `ewc` / `egrep` (32 MiB × 20) | **8.23 / 11.70 GB/s** (`55–62 KiB`) | 4.53 / 3.30 GB/s (`2.0 MiB`) | **1.82× / 3.55× faster**, 32–36× smaller |
| **Zero-libc HTTP (`ehttpd`)** | 4-core `io_uring` (96k reqs) | `377.6k RPS`, `p99=233 us`, **`616 KiB` RSS** | `394.5k RPS`, `p99=249 us`, `3,028 KiB` RSS | Lower `p99`, **`4.9×` less RAM**, `0.957x` RPS |

<sub>* Standalone fresh-process `MaxRSS` median (7 runs). All workloads and caveats — including higher instruction counts on some loops vs LLVM — are documented transparently in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).</sub>

---

## Current Status & Honest Limitations

Eon is **pre-1.0** and already self-hosted on Linux `x86_64`, but it is **not yet a drop-in replacement for Rust**. Known current limitations:

- **Single Target Platform:** Only Linux `x86_64` (PIE / `.so`) and bare-metal `x86_64` (`--freestanding`) are currently supported. `AArch64`, `RISC-V 64`, `Windows`, `macOS`, `Android`, and `iOS` are next on the roadmap.
- **Optimizer Breadth & Instruction Density:** On several tight loops (`08_matmul_f32`, `05_ecs_physics`, `03_utf8_scan`), Eon still emits more machine instructions than LLVM, winning wall time through IPC, layout, and eliminated bounds checks rather than general-purpose auto-vectorization across arbitrary scalar loops.
- **Compiler Memory on Very Large Codebases:** Building the ~171.5k LOC self-hosted compiler currently peaks at `~860 MiB` RSS because whole-program arenas remain resident during compilation.
- **Minimal Standard Library & Tooling:** `core/` + `std/` currently contain 10 core modules (`~3.8k LOC`). Networking (`std.net`), rich filesystem utilities, `eonc fmt`, `eonc lsp`, and a package registry are not yet implemented.
- **Verification Maturity:** While guarded by triple bootstrap, regression tests, and thousands of differential/soundness fuzz runs, the verifier and borrow checker have not yet undergone years of external production hardening.

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Repository map, compiler pipeline (`selfhost/src/`), `core/` vs `std/`, `.hal.eon` TCB model, and triple bootstrap.
- **[ROADMAP.md](ROADMAP.md)** — Completed phases (`0–10`) and upcoming phases (`11–14`: optimizer depth, multi-target backends, `@gpu`, `EonUI`, `v1.0`).
- **[bench/BENCHMARKS.md](bench/BENCHMARKS.md)** — Full Eon vs Safe Rust benchmark tables, PMU counters, reproduction commands, and methodology caveats.
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
12. [spec/10-compiler-ir-and-bootstrap.md](spec/10-compiler-ir-and-bootstrap.md) — `eonc` architecture, SSA `EIR`, native `x86_64`, `ELF64`, `--sanitize`, `--audit-tcb`, and triple bootstrap.
13. [spec/CHANGES-v0.2.md](spec/CHANGES-v0.2.md) — Summary of `v0.2` design decisions.

---

## License

Eon is released under the [MIT License](LICENSE).
