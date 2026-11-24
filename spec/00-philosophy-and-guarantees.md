# 00. Eon Language Philosophy and Formal Guarantee Boundaries

**Status:** Specification `v0.2` (changes from `v0.1` are documented in [CHANGES-v0.2.md](CHANGES-v0.2.md); baseline architecture for implementing `Stage 0` and reaching `Stage 1` Self-Hosting)  
**Language:** `Eon` (file extensions: `.eon` — safe modules, `.hal.eon` — hardware abstraction modules, `.emi` — binary module interfaces)

---

## 1. Purpose of the Language

**Eon** is a strongly typed, statically verifiable systems programming language designed to build the **entire vertical software stack** in a single language:

```
Bootloader → OS Kernel → Drivers (CPU/DMA/MMIO) → Compositor & UI Engine (CPU+GPU) → Applications → Eon Compiler Itself
```

At the same time, Eon is **cross-platform**: Eon programs and the UI engine compile and run with **identical semantics and deterministic results** (given identical inputs; floating-point reduction order is fixed, see [06](06-concurrency-and-memory-model.md) §3.1) on **EonOS**, **Linux**, **Windows**, and **macOS**, across **x86_64**, **ARM64 (AArch64)**, and **RISC-V (RV64GC + RVV)** architectures, while compute kernels and shaders target **GPUs (SPIR-V / MSL / HLSL)**.

---

## 2. The "Zero-Mercy" Philosophy (Zero-Tolerance Compiler)

Unlike C/C++ (which permit silent UB) and even Rust (where many hazardous or sloppy patterns are mere `warning`s or easily bypassed via `.unwrap()` and `unsafe {}`), the Eon compiler operates on a principle of **zero tolerance**:

> **Any ambiguity, hidden cost, unhandled execution path, or unproven invariant is a hard compile-time error, never a warning.**

### 2.1. Registry of Hard Compiler Errors (Zero-Mercy Rules)

The following violations **always** halt compilation (including in dev builds; the language has no flag to "allow UB" or "disable checks"):

| # | Violation | Eon Compiler Reaction |
| :--- | :--- | :--- |
| **ZM-01** | Unused local variable, parameter, import, or private function | Error (`E0101`). Explicit discard only via `_ = expr;` or `_name` prefix. |
| **ZM-02** | Discarded return value (all non-`void` functions are `must_use` by default) | Error (`E0102`). |
| **ZM-03** | Variable name shadowing in a nested scope | Error (`E0103`). |
| **ZM-04** | Non-exhaustive `match` over an `enum`, `union`, `error`, or bounded integer | Error (`E0104`). |
| **ZM-05** | Implicit numeric conversion (even `u8 -> u32`) | Error (`E0105`). Requires `.widen.[T]()`, `.trunc.[T]()`, `.clamp.[T]()`, or `.try_into.[T]()`. |
| **ZM-06** | Implicit copying of a value larger than `16 bytes` is impossible: such types have `move` semantics, and reading after a move is an error | Error (`E0301`; error code `E0106` is reserved and unused). Requires borrowing `&T`, `move`, or the explicit `copy expr` operator. |
| **ZM-07** | Indexing `buf[i]` without a static proof that `0 <= i < buf.len` | Error (`E0107`). Requires a proof or explicit `buf.at(i) ! BoundsError` / `buf.get(i)`. |
| **ZM-08** | Arithmetic `+`, `-`, `*`, `/`, `%`, `<<`, `>>` without a proof of no overflow / no division by zero / no out-of-width shift | Error (`E0108`). Requires a range proof or explicit `+%`, `+|`, `+?`, `/?`. |
| **ZM-09** | Unconsumed value of a `linear` type on *any* exit path (`return`, `break`, `continue`, `try`) | Error (`E0109`). |
| **ZM-10** | Global mutable variable (module-level `var`) | Error (`E0110`). Forbidden even in `.hal.eon`. |
| **ZM-11** | Use of `raw` pointers, `asm`, `volatile`, or `extern "C"` outside a `.hal.eon` module | Error (`E0111`). These keywords are reserved by the lexer; in the `.eon` profile, the parser rejects these constructs immediately (marked `HAL-ONLY` in `grammar.ebnf`). |
| **ZM-12** | Calling a function that violates an effect contract (`@noalloc`, `@noblock`, `@interrupt`, `@stack(N)`, `@gpu`, `@pure`) | Error (`E0112`). |
| **ZM-13** | Calling a recursive function without an explicit `@recursive` annotation | Error (`E0113`). Prevents accidental stack overflow. |
| **ZM-14** | Dead / unreachable code, or an `if` branch statically proven always true or always false outside `comptime if` | Error (`E0114`). Proofs are carried out **exclusively** by the analyzer at the pinned level `@analysis(N)` ([04](04-verification-and-arithmetic.md) §2.4), so upgrading the compiler within the same level never creates new errors. |
| **ZM-15** | Using `unreachable;` without a mathematical proof that the program point is unreachable | Error (`E0115`). |
| **ZM-16** | Uninitialized variable or struct field | Error (`E0116`). Safe Eon has no concept of uninitialized ("garbage") memory. |
| **ZM-17** | Missing curly braces `{}` on `if`, `else`, `while`, `for`, `loop` | Error (`E0117`). |
| **ZM-18** | Overlapping borrows (aliasing `*mut` with any other `*` or `*mut`) | Error (`E0118`). |

---

## 3. The Seven Architectural Pillars of Eon

1. **Local Borrows and Linear Resources**
   - References (`*T`, `*mut T`, `*[T]`, `*mut [T]`, `*str`) are **second-class references**. They cannot be stored in structs or pools. Lifetimes are verified via local control flow and declared `view fn` sources, without `'a` lifetime annotations.
   - Graphs represent edges using **branded indices** `Id[p]` into `ArenaPool[T]` and **generational handles** `GenId[T]` into `SlotPool[T]`. Generation checks prevent a deleted handle from resolving again in the pool instance that issued it ([03 §3.3](03-ownership-borrows-linearity.md)).
   - **Linear types** (`linear struct`) require explicit release or ownership transfer of critical resources.
2. **Explicit Operational Cost**
   - **Struct of Arrays (`pool`)** layout allows reading only the required columns and enables vectorizing dense passes.
   - Zero hidden bounds checks, zero `panic` / stack-unwinding (`unwind`) tables, zero hidden `memcpy`s, and zero hidden allocations.
   - A native register-based ABI (errors and slices are passed in CPU registers, eliminating the C ABI tax inside Eon and EonOS).
3. **Static Bounds and Arithmetic Proofs**
   - A verifier based on intervals and Difference-Bound Matrices checks safety conditions within its domain. For dynamic cases, the language provides explicit checked operations.
   - The language physically **has no** `panic()`, `.unwrap()`, or `throw/catch` exceptions.
4. **Automatic Lock-Free Data Parallelism**
   - Global mutable state does not exist.
   - Systems (`sys`) declare `read`/`write` access on specific pool columns, allowing the "1 thread = 1 physical core" scheduler to run non-conflicting subsystems in parallel without a single `Mutex`.
   - `par for` guarantees per-element isolation at the type level (data races are statically impossible) and degrades into a linear SIMD loop on the current core for small datasets.
5. **Safe Non-Blocking I/O Without `async/await`**
   - Instead of colored `async` functions and `Pin`, Eon uses **I/O rings (the `io_uring` model)** and **lightweight fibers with fixed, statically computed stack sizes**.
6. **Unified CPU + GPU + HAL Code**
   - OS kernels, drivers, UI engines, applications, and GPU shaders (`@gpu`) are written in one language with shared data types and compile-time memory layout verification.
7. **Compilation Budgets and Reproducible Self-Hosting**
   - The grammar is strictly **LL(1) + Pratt** (no backtracking and no turbofish).
   - A custom SSA IR (`EIR`), per-function incremental caching, polymorphism with layout-based code sharing (`Layout Sharing`), and zero LLVM dependency in the target self-hosting architecture (`Stage 1+`).

---

## 4. Guarantees and Trusted Computing Base

The following describes the requirements for a complete Eon implementation. Satisfying them depends on the correctness of the compiler, the runtime, and trusted HAL code.

### 4.1. Requirements for Safe Modules
* **Memory Safety:** No Use-After-Free, Double-Free, Buffer Overflow/Underflow, Null Pointer Dereference, or reading uninitialized memory.
* **Data-Race Freedom:** Two threads never have simultaneous unsynchronized access to the same memory location if at least one of them is writing.
* **Zero Undefined Behavior (Zero UB):** No operation in `.eon` (including integer arithmetic, bit shifts, and bitcasts) has Undefined Behavior.
* **Zero Linear Resource Leaks:** Every `linear` object is guaranteed to be consumed exactly once along all execution paths (both normal and error paths).
* **Static Stack Boundedness:** Every function not annotated with `@recursive` has an exact upper bound on stack usage in bytes, known at compile time.
* **Capability Soundness:** A module cannot access disk, network, clocks, GPU, or an allocator unless the corresponding right (`Capability`) is explicitly passed to it via arguments.

### 4.2. What No Compiler Can Guarantee (Limits of Physics and Computability Theory)
* **Program Performance:** Performance depends on algorithms, data layout, workload, and target hardware. Build and runtime costs are evaluated on reproducible benchmarks.
* **Correctness of the Trusted Core (`.hal.eon`):** Code inside `raw { ... }` (accessing physical CPU registers, MMIO, page tables, C FFI) is part of the **Trusted Computing Base (TCB)** and is responsible for upholding invariants at its boundary.
* **Hardware Faults:** RAM bit-flips, disk failures, power loss, and hardware `#MC` (Machine Check) exceptions are handled by OS fault-tolerance mechanisms, not by the type system.
* **Logical Deadlocks on High-Level Channels/Queues:** `sys` and `par for` are deadlock-free by construction, but two fibers manually waiting on messages from each other over user-defined channels can still deadlock logically (the scheduler provides timeouts and a debug detector for this).
* **Termination of Arbitrary Loops (`while`/`loop`):** Eon is Turing-complete; the halting problem is undecidable (however, `@interrupt` supports an optional `@bounded_loops` check).

---

## 5. Specification Glossary

| Term | Meaning in the Eon Specification |
| :--- | :--- |
| **Safe Module (`.eon`)** | A standard Eon source file. Cannot contain `raw`, `asm`, raw pointers, or `extern "C"`. |
| **HAL Module (`.hal.eon`)** | A hardware abstraction module with the header `module ... @hal;`. The only place where `raw { ... }` is allowed. |
| **TCB (Trusted Computing Base)** | The minimal set of `.hal.eon` modules plus the compiler itself upon whose correctness the safety of the entire system depends. |
| **Second-Class Reference** | A borrow (`*T`, `*mut T`, `*[T]`, `*mut [T]`, `*str`) that exists only in parameters or local bindings and cannot be stored in structs. |
| **Linear Type (`linear`)** | A type whose instances must be transferred by ownership or consumed **exactly once** on every execution path. |
| **Pool (`pool`)** | A typed data structure physically storing columns as dense arrays (**SoA**). |
| **Brand (`Id[p]`)** | An implicit unique zero-sized marker type binding a 32-bit index to a specific `ArenaPool` instance. A local second-class value: storable only in columns of its own pool via `Link[Self]`. |
| **Link (`Link[Self]`)** | A field type inside a `pool` schema referring to another row of the same pool. Resolves to `Id[p]` (4 bytes) in `ArenaPool` and `GenId[T]` (8 bytes) in `SlotPool`. |
| **Borrowing Value** | A value (`ArenaPool[T]`, `ArenaStr`, or a non-reference return value of a `view fn`) that carries a source lock like a second-class reference: cannot be stored in a `struct`/`pool` and cannot be returned from a regular function (`E0311`). |
| **GenId (`GenId[T]`)** | A 64-bit handle `(index: u32, gen: u32)`. Protection against resolving a handle after deletion applies within the `SlotPool[T]` instance that issued it. |
| **DBM (Difference-Bound Matrix)** | A lightweight relational data structure used by the compiler analyzer to prove inequalities of the form `x - y <= c`. |
| **Capability** | A linear or borrowable permission object (`AllocCap`, `FsCap`, `NetCap`) granting access to external resources. |
| **EIR** | Eon Intermediate Representation — the compiler's typed SSA representation preserving pool, effect, and interval information. |
| **`.emi`** | Eon Module Interface — a compact binary file containing exported signatures, `where` contracts, and type layouts of a module. |

---

## 6. Specification Map

1. [01-lexical-and-syntax.md](01-lexical-and-syntax.md) and [grammar.ebnf](grammar.ebnf) — lexical structure, formal grammar, modules, and visibility.
2. [02-types-and-memory-layout.md](02-types-and-memory-layout.md) — type system, memory layout, `bitfield`, `pool` (SoA), and UTF-8 strings.
3. [03-ownership-borrows-linearity.md](03-ownership-borrows-linearity.md) — second-class borrows, `linear`, `errdefer`, `ArenaPool`, and `SlotPool`.
4. [04-verification-and-arithmetic.md](04-verification-and-arithmetic.md) — relational bounds analyzer, `where` contracts, arithmetic, and `! E` errors.
5. [05-generics-interfaces-comptime.md](05-generics-interfaces-comptime.md) — generics `[T]`, `comptime` VM, Layout Sharing, interfaces, and `dyn` with `@stack`.
6. [06-concurrency-and-memory-model.md](06-concurrency-and-memory-model.md) — `sys`, `par for`, the "1 thread = 1 core" scheduler, atomics, and DMA barriers.
7. [07-effects-capabilities-io.md](07-effects-capabilities-io.md) — effects, Capabilities, I/O rings, and static fibers.
8. [08-hal-ffi-and-gpu.md](08-hal-ffi-and-gpu.md) — `@hal`, MMIO, `arch/`, C FFI, and the `@gpu` model (SPIR-V / MSL / HLSL).
9. [09-abi-modules-and-hotreload.md](09-abi-modules-and-hotreload.md) — native ABI, EonOS zero-copy ABI, `.emi`, and Hot-Reload.
10. [10-compiler-ir-and-bootstrap.md](10-compiler-ir-and-bootstrap.md) — `EIR`, backends (Dev/Release), DWARF/PDB, LSP, and the `Stage 0 -> Stage 2` bootstrap protocol.
