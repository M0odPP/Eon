# 05. Generics, Interfaces, Stack Budgets, and `comptime`

This chapter solves three C++/Rust pain points simultaneously:
1. **Binary bloat (Code Bloat)** from indiscriminate template monomorphization -> solved via **Layout Sharing**.
2. **Slow compilation from `proc-macro` and SFINAE** -> solved via **definition-site-checked typed generics** + a **deterministic `comptime` VM**.
3. **Loss of static stack analysis under dynamic polymorphism (`vtable`)** -> solved via **`@stack(N)` and effect contracts directly inside `iface`**.

---

## 1. Generics (`[T]`)

### 1.1. Syntax and Definition-Site Checking
```
pub fn min_val[T: Ord](a: T, b: T) -> T {
    if a.cmp(&b) == .Less {
        return a;
    }
    return b;
}
```
* **Definition-site checking (like Rust/Swift, NOT like C++/Zig):** the body of `min_val` is verified by the compiler **once** when its definition is analyzed, based on the constraint `T: Ord` (plus linearity and interval verification!). You cannot accidentally write an uncompilable generic that "only fails when someone instantiates it with `u32`" (`E0501`).
* **Constant parameters (`comptime`):**
  ```
  pub struct RingBuf[T, comptime N: usize]
      where N > 0 && (N & (N - 1)) == 0
  {
      data: [N]T;
      head: usize;
      tail: usize;
  }
  ```
* **Explicit instantiation in expressions:** in 99% of cases, type arguments are inferred from parameter types (`min_val(a, b)`). When inference is impossible (e.g., parsing a type from a string), the **`.[...]`** syntax is used (100% LL(1)):
  ```
  let port = try parse_uint.[u16](text);
  let pool = try ArenaPool.[Widget].init(&mut arena, 1024_usize); // type position uses ArenaPool[Widget]; expression position uses ArenaPool.[Widget]
  ```
  Rule: in **type** position, arguments are written `Name[A, B]`; in **expression** position, always `Name.[A, B]` (otherwise `[` denotes indexing).

### 1.2. Combating Binary Bloat: Layout Sharing vs. Monomorphization
In C++ and Rust, `Vec<A>`, `Vec<B>`, `Vec<C>` generate hundreds of copies of identical machine code, bloating binaries and evicting the 32 KB `L1i` instruction cache.

Eon uses a **dual-mode code generation strategy for generics**:

1. **Default mode — Layout-Based Polymorphism (`Layout Sharing`):**
   For generic functions that merely move or manage data (containers `Buf[T]`, `SlotPool[T]`, `HashMap[K, V]`, channels, comparator-based sorting), the compiler groups instantiations by their physical layout key `(size_bytes, align_bytes, is_linear_layout)`!
   - `Buf[u64]`, `Buf[PhysAddr]`, `Buf[GenId[Widget]]`, `Buf[f64]` all share layout `(8, 8)` -> the compiler emits **exactly one** shared machine function `Buf::push$8_8`!
   - For variable-size container operations (e.g., `Buf.reserve`, `SlotPool.remove`), a hidden descriptor `(elem_size, elem_align)` is passed — zero code duplication at all!
2. **Hot specialization mode (`@specialize` / automatic monomorphization):**
   - If a generic carries arithmetic/SIMD bounds (`T: Float`, `T: Int`) or the function is marked `@hot` / `@inline` / `@gpu`, the compiler performs **full monomorphization**, unfolding `comptime N` constants, vectorization, and register passing at 100% speed.

**How methods of `T` are called in shared code.** A shared function like `Buf::sort$8_8` does not know the concrete `T`. Constraint methods (`T: Ord`, `K: Hash + Eq`) are passed via a **hidden dictionary** (method pointers + `elem_size` / `elem_align`) — just like `*dyn Iface`, only without a fat object. Dictionary calls are indirect, and their stack usage and effects are taken from the `iface` contracts (just as with `dyn`, §2.3), so `@stack(N)` and `@noalloc` remain verifiable. Consequences:
* Layout Sharing is chosen only where the method call does not sit on a hot path; for `@hot` / `@inline` and numeric constraints (`T: Int`, `T: Float`), the compiler monomorphizes.
* A shared function does not observe value intervals of `T`, so DBM facts about elements do not cross the shared-code boundary (monomorphization preserves them).

**Result:** maximum speed in hot math code and a tiny machine-code footprint (fitting inside `L1i`) everywhere else.

---

## 2. Interfaces (`iface`), `impl`, and Dynamic Polymorphism (`dyn`)

Without dynamic polymorphism, it is impossible to write an OS kernel VFS (pluggable filesystems and disk drivers) or custom UI widgets/renderers.

### 2.1. Declaring an Interface with Effect and Stack Contracts
```
pub iface BlockDevice {
    fn sector_size(self: *Self) -> usize @pure @noalloc @stack(64);
    fn read_sector(self: *mut Self, lba: u64, dst: *mut [u8]) -> !IoError
        @noalloc @stack(1024)
        where dst.len >= 512;
}
```
### 2.2. Implementation (`impl`)
```
impl BlockDevice for NvmeDrive {
    fn sector_size(self: *Self) -> usize @pure @noalloc @stack(64) {
        return self.lba_bytes;
    }

    fn read_sector(self: *mut Self, lba: u64, dst: *mut [u8]) -> !IoError
        @noalloc @stack(1024)
        where dst.len >= 512
    {
        return self.submit_read(lba, dst);
    }
}
```
* **Covariant implementation checks:** an implementation may consume **less than or equal to** the stack budget declared in `iface` (e.g., `384 <= 1024` — OK; `1536 > 1024` — **compile error `E0510`**!), may be *purer* in its effects, and may not require a stricter `where` clause than in `iface` (`E0511`).
* **Orphan Rule:** `impl Iface for Type` is allowed only in the module where `Iface` is declared or the module where `Type` is declared (preventing conflicting implementations during builds).

### 2.3. Static and Dynamic Dispatch
* **Static dispatch (0 overhead):**
  ```
  fn read_mbr[D: BlockDevice](dev: *mut D, buf: *mut [512]u8) -> !IoError {
      return dev.read_sector(0_u64, &mut buf[0..512]);
  }
  ```
* **Dynamic dispatch (`*dyn Iface` / `*mut dyn Iface`):**
  - Is a **second-class reference** (a fat pointer `{ data_ptr: usize, vtable_ptr: usize }` — exactly 16 bytes, passed in 2 registers).
  - Requires no heap (`Box<dyn Trait>` is not needed in Eon — the concrete driver lives in an `Arena` or `SlotPool`, and `*mut dyn BlockDevice` is passed into functions).
  - **Preservation of static stack verification:** when calling a method through `*mut dyn BlockDevice`, the stack analyzer takes the guaranteed `@stack(1024)` ceiling from the interface signature! Thus, even a dynamically dispatched driver call in an OS kernel has a **mathematically proven upper bound on stack usage**.
  - Object safety (`dyn`-compatibility): a method with generic parameters or returning `Self` by value is marked incompatible with `dyn` (error `E0512` when attempting to construct `dyn Iface`).

---

## 3. Compile-Time Execution (`comptime`) — Without Macros

Eon has **no preprocessor (`#define`) and no procedural macros**. Instead, all build-time computations (CRC32 tables, sine tables, format validation, serialization metadata, `build.eon` project builds) are written in **regular Eon** and executed by the compiler's built-in **deterministic register-based `comptime` VM**.

### 3.1. Capabilities and Limits of `comptime`
```
const CRC32_TABLE: [256]u32 = comptime {
    var table: [256]u32 = [0_u32; 256];
    for i in 0..256 {
        var c: u32 = i.trunc.[u32]();
        for _ in 0..8 {
            if (c & 1_u32) != 0_u32 {
                c = 0xEDB8_8320_u32 ^ (c >> 1_u32);
            } else {
                c = c >> 1_u32;
            }
        }
        table[i] = c;
    }
    table
};
```
* **Type Reflection (Read-Only, Type-Safe):** the built-in `@type_info(T)` function in `comptime` returns an immutable `TypeInfo` struct (field names, types, offsets, `@bits` attributes, `pool` columns). This makes it possible to write a zero-cost binary packet serializer, JSON parser, GPU buffer binding generator, or Hot-Reload migration table in 20 lines of pure Eon!
* **Conditional Compilation (`comptime if`):**
  ```
  comptime if @target_arch() == .X86_64 {
      // Branch is type-checked and compiled ONLY for x86_64
  } else comptime if @target_arch() == .Arm64 {
      // ...
  } else {
      @compile_error("Unsupported architecture");
  }
  ```
  *(Unlike a normal `if`, a dead `comptime if` branch does not trigger `ZM-14`.)*
* **Strict Safety and Speed Guarantees of the `comptime` VM:**
  1. **Complete purity and hermeticity:** `comptime` code has **no access** to the filesystem, network, clock, environment variables, or random numbers (except in `build.eon` with an explicit `BuildCap`). Builds are 100% reproducible!
  2. **Step limit (Fuel Limit):** by default, a `comptime` block is capped by a budget (e.g., `10_000_000` VM instructions; explicitly extensible via `@comptime_budget(...)`). An infinite loop in `comptime` never hangs the compiler — it immediately reports error `E0520`.
  3. **All Zero-Mercy checks apply inside `comptime` as well:** overflow or out-of-bounds array indexing inside `comptime` is an immediate compile error.

---

## 4. Build System (`build.eon`)

An Eon project has no `Makefile`, `CMakeLists.txt`, `Cargo.toml` scripts, or `build.rs`. A package's build description is a standard **`build.eon`** file at the repository root, compiled and executed by the `eon build` compiler itself inside an isolated sandbox with a `BuildCap` object (granting access solely to the project tree and the `out/` output directory).

---

## 5. Specification Tests

### Must Compile
```
module example.generics_ok;

pub iface Hasher {
    fn write(self: *mut Self, data: *[u8]) @noalloc @stack(128);
    fn finish(self: *Self) -> u64 @pure @noalloc @stack(64);
}

pub fn hash_dyn(h: *mut dyn Hasher, msg: *str) -> u64 @noalloc @stack(256) {
    h.write(msg.as_bytes());
    return h.finish();
}
```

### Must Fail Compilation
```
module example.generics_fail;

pub iface SmallStack {
    fn run(self: *Self) @noalloc @stack(64);
}

pub struct Huge { x: u32; }

impl SmallStack for Huge {
    fn run(self: *Self) @noalloc @stack(64) {
        var big_local: [4096]u8 = [0_u8; 4096]; // ERROR E0510: stack usage > 64 bytes!
        _ = big_local[0];
    }
}
```
