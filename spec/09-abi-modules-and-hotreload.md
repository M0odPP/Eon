# 09. Native ABI, Binary Modules (`.emi`), Binary Size, and Hot-Reload

This chapter explains how Eon achieves:
1. **10–40 KB binaries** on EonOS (and 150–350 KB with a full UI on Windows/Linux/macOS).
2. **Call performance faster than the C ABI** inside programs and across the application–EonOS boundary.
3. **Instant incremental rebuilds (1–5 ms)** and safe **Hot-Reload** (including data schema changes).

---

## 1. Native Eon ABI (`EonABI v1`)

Inside Eon programs and across the EonOS application boundary, **the archaic 1970s C ABI is not used** (the C ABI applies only to explicit `@extern("C")` functions inside `.hal.eon`).

### 1.1. Calling and Register Convention (`EonCall`)
Optimized for Eon's core types (slices, `! E` errors, SIMD vectors, `GenId`, `dyn Iface`):

| Entity | `x86_64` (`EonCall`) | `ARM64` (`EonCall`) | `RISC-V 64` (`EonCall`) |
| :--- | :--- | :--- | :--- |
| **Arguments (integer/pointer)** | `rdi, rsi, rdx, rcx, r8, r9, r10, r11` (8 registers!) | `x0..x7` (8 registers) | `a0..a7` (8 registers) |
| **Arguments (float/SIMD)** | `xmm0..xmm7` / `ymm0..ymm7` | `v0..v7` | `fa0..fa7` / `v0..v7` |
| **Return value `T` (up to 32 B)** | `rax, rdx, rcx, r8` (4 registers!) + `xmm0..xmm3` | `x0..x3` + `v0..v3` | `a0..a3` + `fa0..fa3` |
| **Error code return `! E`** | **`r15`** — source of truth: `0` = success, `u16` = error code. In addition, the callee sets flag `CF` (`1` = error) as a fast mirror | **`x8`** — source of truth (`0` = success, `u16` = code); mirrored in flag `C` | **`t0`**: `0` = success, `u16` = error code (RISC-V has no condition flags: checked with `bnez t0`) |
| **Passing `*[T]` / `*str` / `*dyn`** | Unpacked into 2 registers `(ptr, len)` (no stack spill!) | 2 registers | 2 registers |
| **Interprocedural Register Allocation (IPRA)** | For module-private functions, the compiler **freely departs from** `EonCall`, passing values in any free registers and skipping saves of untouched callee-saved registers! | Same | Same |

### 1.2. Callee-Saved Registers and Passing Large Structs (`> 32 B`)
* **Callee-saved (preserved by the called function):**
  - `x86_64`: `rbx, rbp, r12, r13, r14` (5 registers; `r15` is reserved for the `EonCall` error code; `rsp` is 16-byte aligned before `call`).
  - `ARM64`: `x19..x28, x29 (fp), x30 (lr), v8..v15 (low 64 bits)`.
  - `RISC-V 64`: `s0..s11, ra, fs0..fs11`.
* **Passing arguments `> 16 B` by value (`move` / `copy`):** passed via a hidden pointer to a stack slot in the caller (no redundant copying inside the prologue).
* **Returning values `> 32 B`:** the caller allocates a slot on its own stack and passes a hidden first argument `sret_ptr` (RVO — Return Value Optimization is guaranteed by the specification, 0 copies!).

**Why returning errors in a dedicated register + `CF` outperforms C++ and Rust:**
* The normal happy path after `call` is checked with a single `jc .on_err` instruction — the value register `rax` is never polluted with an error discriminant! The `CF` flag is valid **only immediately after `call`** (prior to the caller's first arithmetic instruction); the backend must lower `BranchOnErr` either to `jc`/`jb` right after `call`, or to `test r15, r15`. The callee's cost on the success exit is simply `xor r15, r15` + `clc`.
* The size of exception unwinding tables (`.eh_frame`) in the binary is **0 bytes**.

### 1.3. Error Code Numbering (`u16`)
* An `error`-set variant code is a `u16`, **unique across the entire program**; `0` denotes success and is never assigned as an error code.
* Ranges: `0x0001–0x0FFF` — system errors (`core.*`, `EonSysImage`): codes are fixed by the `EonABI v1` table and included in `sys_abi_hash`; `0x1000–0xFFFF` — application and library errors.
* Object files and `.emi` files store errors **symbolically** — `(module, set, variant)`. The built-in linker assigns codes `>= 0x1000` densely in deterministic order (sorted by fully qualified name) and patches `ErrCodeReloc` relocations. Consequently, the same variant has the exact same code in any two modules, and `try` from a subset of errors into a superset **requires no code remapping** (`E0401` checks only set membership).
* Overflowing the 16-bit space is **`E0902`**. During hot-reload, new variants receive the next free codes; changing the code of an already-issued variant is a schema incompatibility (`E09xx`).
* `match` over an error set compiles to comparisons or a jump table over the assigned codes (contiguous density is not required).

---

## 2. Unified Safe EonOS ABI and Micro-Sized Binaries

### 2.1. Why a UI Application on EonOS Weighs 10–40 KB
On classic operating systems (Linux/Windows), every application is forced to statically bundle its own copy of the standard library, font engine, UI framework, and allocator, because OS system calls are limited to a primitive C ABI while dynamic libraries (`.so`/`.dll`) suffer from version conflicts and slow `PLT/GOT` indirection.

In **EonOS**, the entire system is built on `EonABI v1`:
1. **Unified Shared OS Code Pages (`EonSysImage`):** the `core` standard library, UI engine (layout + renderer + fonts), scheduler, and I/O rings are mapped into every process's address space as immutable (Read+Exec) pages with a deterministic table of versioned symbols (`EonABI Hash`).
2. **`EonBin` Executable Format (exact 64-byte header, 0 bytes of hidden padding):**
   ```
   @layout(c) @align(8)
   pub struct EonBinHeader {
       magic:        [4]u8;   // 0..4:   "EONB" (0x45, 0x4F, 0x4E, 0x42)
       abi_version:  u16;     // 4..6:   1
       arch:         u8;      // 6..7:   1=x86_64, 2=arm64, 3=riscv64
       flags:        u8;      // 7..8:   bit0=HasHotReloadMeta, bit1=Freestanding, bit2=HasGpuSpirv
       entry_offset: u32;     // 8..12:  entry point offset in .text
       text_size:    u32;     // 12..16: size of machine code
       rodata_size:  u32;     // 16..20: size of constants
       meta_offset:  u32;     // 20..24: 0 in release; offset of .type_meta section in dev
       cap_mask:     u64;     // 24..32: bitmask of requested Capabilities (FS/NET/GPU/CLOCK)
       sys_abi_hash: [32]u8;  // 32..64: SHA-256 of the EonSysImage system symbol table
   }
   ```
   - `.text` section (only the application's own pure logic with `Layout Sharing` -> fits entirely in `L1i`!).
   - `.rodata` section (constants, UTF-8 strings, `comptime` tables).
   - `.type_meta` section (optional, for Hot-Reload and the profiler).
   - **No `.eh_frame`, no `.gcc_except_table`, no bloated libc symbol tables.**
3. **Zero-Copy IPC and Trusted Sandboxes:** because EonOS programs are compiler-verified (0 `raw` code in user applications, cryptographically signed builds), submitting a UI draw command tree to the compositor happens through shared memory pools (`ArenaPool` in shared memory) with zero copying and no heavy serialization.

### 2.2. Cross-Platform Builds for Windows / Linux / macOS
On foreign operating systems (which lack `EonSysImage`), the compiler builds a **self-contained static executable** (`ELF` / `PE` / `Mach-O`):
* Includes only genuinely reachable functions (aggressive call-graph tree-shaking), a micro scheduler runtime (~25 KB), the UI engine (~90–150 KB), and a thin platform adapter (~15 KB).
* **Total size of a full graphical application with GPU-UI on Windows/Linux/macOS:** **150–350 KB**, with zero external dependencies beyond base OS DLLs/shared libraries (`kernel32`/`user32`/`vulkan-1` or `libc`/`libvulkan`).

---

## 3. Binary Module Interface Format (`.emi`) and Incremental Builds

So that compiling 200,000+ lines takes fractions of a second, and editing a single function rebuilds in **1–5 ms**:

### 3.1. No Header Files (`.h`)
Every compiled module produces a compact binary **`.emi` (Eon Module Interface)** file with a fixed 64-byte header (`magic = "EOMI"`, `compiler_ver`, `interface_hash: u128`, table offsets), containing:
1. **Interface Hash (`InterfaceHash: u128`):** computed ONLY from `pub` declarations (function signatures, `where` / `ensures` contracts, `@stack`/`@noalloc` effects, `pub struct`/`pub pool` layouts, public `comptime` constants) **and from the serialized bodies** of `pub` generics, `@inline` functions, and `comptime` functions (item 3 below). Such a function's body is part of the interface: omitting it from the hash would prevent dependent modules from rebuilding when its body changes (making Early Cutoff unsound).
2. **Table of public types and layouts** (offsets, alignment, niches, `pool` columns).
3. **Serialized AST/EIR exclusively for `pub` generics, `@inline`, and `comptime fn`.**
4. **Table of per-function hashes (`FnBodyHash[fn_id]`)** for caching machine code.

### 3.2. Per-Function Incrementality (Early Cutoff)
When a programmer edits code inside the body of function `foo`:
1. The parser re-parses only the modified file (~0.2 ms).
2. The compiler computes the module's new `InterfaceHash`. If `foo`'s signature did not change, **`InterfaceHash` matches** -> not a single dependent module in the project is **even re-read** (Early Cutoff)!
3. Inside the module itself, **exactly one function `foo`** is re-verified and recompiled to machine code (all other functions in the module are reused directly from the `.eobj` cache).
4. Patching the binary (or Hot-Reload segment) takes **1–5 milliseconds** end-to-end!

---

## 4. Live Code Replacement (Hot-Reload) and Data Migration

Because Eon has no global `var` declarations and all program and UI state lives in explicit `Arena` / `ArenaPool` / `SlotPool` instances, Hot-Reload is built into the language architecture at the compiler level (in dev mode `--hot-reload`):

### 4.1. Scenario 1: Only Function Code Changed (90% of edits)
* In dev mode, all cross-module calls route through a compact redirection table (`JumpTable`, 1 indirect jump, completely stripped in release builds).
* Within 2 ms, the compiler recompiles the changed function `foo` into a fresh executable memory page and atomically (`store(.Release)`) updates its slot in `JumpTable`.
* The very next UI frame (or loop iteration) immediately executes the new code over **the exact same live in-memory state** without restarting the window!

### 4.2. Scenario 2: A `struct` or `pool` Schema Changed (field added/removed/renamed)
An honest treatment of a problem many languages sweep under the rug: if you add a new field `opacity: f32` to `pool Widget`, the old in-memory layout of the running process no longer matches the new code!

Eon solves this **strictly and safely (with zero UB and zero memory corruption)** using **schema metadata (`.type_meta`)**:
1. In dev mode, every `pool` and the root application state (`AppState`) carry a schema hash `SchemaHash`.
2. When a new module is loaded, the Hot-Reload runtime compares the old and new `TypeInfo` schemas:
   - **Field added with a default value** (e.g., `opacity: f32 = 1.0_f32;`) -> for a `pool` (SoA!), it suffices to allocate **exactly one new column** and fill it with the default value! Existing columns (`pos`, `size`, `color`) are not even copied!
   - **Field removed** -> its column is freed (provided it is not `linear`).
   - **Field type changed, or field added without a default value** -> automatic migration is impossible; Hot-Reload **does not corrupt memory**, and instead invokes the registered `on_schema_mismatch` hook (which either cleanly reconstructs the state/arena or runs a custom migration).
3. If active fibers currently have stack frames executing inside a recompiled function, the `JumpTable` update applies to *new* calls, while the old code page is reclaimed (via epoch-based reclamation) only after all fibers have exited the old frame.

### 4.3. Hot-Reload and Statically Computed Stacks
A fiber's stack is sized from the `@stack(N)` bound of the **old** version of the code, and function prologues contain no stack-overflow checks. Therefore, a function patch is applied to live fibers **only if** the newly computed stack ceiling of every patched function and all its callers along live frames is `<=` the old ceiling. Otherwise, the patch is **deferred** (pending): it takes effect for new calls from newly spawned fibers, while live fibers continue running the old version until they exit the affected functions (with epoch-based reclamation of the old code page, as in item 3 above). In dev mode, a guard page is additionally allocated for each fiber stack.
