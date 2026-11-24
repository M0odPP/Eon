# 02. Type System and Physical Memory Layout

---

## 1. Foundational Principle: Deterministic Memory

In Eon, every type has a **strictly defined size (`@size_of(T)`) and alignment (`@align_of(T)`)**, computed at compile time for the target architecture.
* There is no "platform-dependent `int`": `u32` is always 32 bits on every architecture; `usize`/`isize` always match the target platform's pointer width (64 bits on `x86_64`/`ARM64`/`RV64`, 32 bits on `RV32`).
* Memory endianness is determined by the target architecture (little-endian by default on x86_64/ARM64/RV64). For network and on-disk protocols, explicit methods `to_be_bytes()` / `from_be_bytes()` / `to_le_bytes()` are provided.
* **All** implicit numeric conversions are forbidden (`ZM-05`).

---

## 2. Primitive Types

### 2.1. Integers
* **Unsigned machine integers:** `u8`, `u16`, `u32`, `u64`, `u128`, `usize`.
* **Signed machine integers (strictly two's complement):** `i8`, `i16`, `i32`, `i64`, `i128`, `isize`.
* **Bit-width integers (`u1`..`u63`):** permitted inside `bitfield` declarations and for exact ranges. Outside a `bitfield`, they are stored in the smallest machine register wide enough to hold them, with the guaranteed invariant `0 <= x < 2^N`.

### 2.2. Floating-Point Numbers (IEEE-754)
* `f16` (2 bytes, storage/GPU/SIMD), `f32` (4 bytes), `f64` (8 bytes).
* Comparing `NaN`: per IEEE-754, `NaN != NaN`. For sorting and hashing, `f32`/`f64` **do not implement** standard `Ord`/`Eq` — the programmer must explicitly choose `TotalF32` or the `total_cmp` method (Zero-Mercy protection against broken sorts).

* **Computational determinism:** the compiler may not fuse `a * b + c` into `fma`, nor reorder or alter the precision of IEEE operations (no `-ffast-math`): fused multiply-add is only performed via an explicit `fma(a, b, c)` call. This preserves identical results across x86_64, ARM64, and RISC-V.

### 2.3. Boolean, Character, and Special Types
* `bool`: size `1` byte; the only valid bit representations are `0` (`false`) and `1` (`true`).
* `char`: size `4` bytes (`u32`), guaranteed to be a valid Unicode scalar value (`0x0000..=0xD7FF` or `0xE000..=0x10FFFF`).
* `void`: zero-sized type (`0` bytes, alignment `1`). Has a single value, `void`.
* `never`: uninhabited type (`0` bytes) returned by diverging expressions (`return`, `break`, `continue`, `loop {}` without `break`, `unreachable;`). Coerces to any type.
* `type`: first-class type existing **only at `comptime`** (no runtime representation; attempting to store `type` in a runtime variable is error `E0210`).

### 2.4. Built-In SIMD Vectors (`vec[T, N]`)
SIMD types are built into the core language and lower directly to hardware registers (`SSE/AVX2/AVX-512`, `ARM NEON`, `RISC-V RVV`, `GPU vecN`):
* `vec[T, N]`, where `T` is a scalar numeric type or `bool`, and `N` is a power of two from `2` to `64`.
* Standard aliases: `f32x2`, `f32x4`, `f32x8`, `f64x2`, `f64x4`, `i32x4`, `i32x8`, `u8x4`, `u8x16`, `u8x32`, `u32x4`.
* Alignment: `min(sizeof(T) * N, max_simd_align_of_target)` (unless overridden by `@layout(std430)`).
* Vectors support element-wise operations (`+%`, `+|`, `+`, `-`, `*`, `/`, `min`, `max`, `fma`, `dot`, `shuffle`, swizzles `.xy`, `.zw`, `.rgba`).
* If width `N` exceeds the target CPU's hardware register width (for example, `f32x8` on ARM64 NEON with 128-bit registers), the compiler deterministically splits the operation into 2 hardware instructions — **a single codebase runs across all architectures without `#ifdef`**.

### 2.5. Strong Kernel Address Types (in `core.addr`)
To prevent catastrophic OS bugs, addresses from distinct address spaces are **incompatible strong types** (size = `usize`):
* `PhysAddr` — physical RAM / bus address.
* `VirtAddr` — virtual address in a process or kernel address space.
* `DmaAddr` — address in an IOMMU / device DMA controller space.

Adding two addresses together or passing a `VirtAddr` where a `PhysAddr` is expected is impossible (`E0211`). Only explicit offset arithmetic `PhysAddr + usize -> PhysAddr` (with overflow checking or `+%`) is allowed.

---

## 3. Explicit Type Conversions (Zero Implicit Coercion)

Every type conversion is performed explicitly via built-in methods:

| Method | Semantics | Machine Code Cost |
| :--- | :--- | :--- |
| `x.widen.[To]()` | Lossless widening (e.g., `u8 -> u32`, `i16 -> i64`). Compile-time error if sign or bit loss is possible. | 0–1 instructions (`movzx`/`movsx`) |
| `x.trunc.[To]()` | Explicit truncation of high bits (e.g., `u32 -> u8`). | 0–1 instructions |
| `x.clamp.[To]()` | Saturating conversion clamped to `[To.MIN, To.MAX]`. | 1–2 instructions (`min`/`max`) |
| `x.try_into.[To]()` | Range check, returning `To ! ArithError.Overflow`. | 1 comparison + branch |
| `x.bitcast.[To]()` | Bitwise reinterpretation when `@size_of` is strictly equal (e.g., `f32 -> u32`). Forbidden for types with niches (`bool`, `char`, `?T`, references). | 0 instructions (register move/reinterpret) |
| `x.to_float.[f32]()` | Integer -> float (IEEE-754 round-to-nearest-even). | 1 instruction (`cvtsi2ss`) |
| `x.trunc_to_int.[i32]()` | Float -> integer with saturation (`NaN -> 0`, `Inf -> MAX/MIN`, zero UB). | 1–2 instructions |

---

## 4. Composite Types and Their Memory Layout

### 4.1. Structs (`struct`)
```
pub struct Rect {
    pub x: f32;
    pub y: f32;
    pub w: f32;
    pub h: f32;
}
```
* **Default layout (`@layout(eon)`):** the compiler **freely reorders fields** by descending alignment to minimize padding holes and shrink struct size in CPU caches.
* **`@layout(c)` (or `extern "C" struct` in `.hal.eon`):** strict declaration order and C ABI alignment rules of the target platform.
* **`@layout(std430)`:** strict layout for GPU buffers (SSBO / Uniforms), verified at compile time for both backends (CPU and GPU).
* **`@packed`:** alignment `1`, zero padding between fields. Taking a `*T` or `*mut T` reference to an unaligned field of a `@packed` struct is **forbidden** (`E0215`) — access is by value (copy) only.
* **Mandatory complete initialization (`ZM-16`):** when constructing `Rect.{ x: 0.0, y: 0.0, w: 10.0, h: 10.0 }`, every field without a default value must be specified.

### 4.2. Bitfields (`bitfield`)
Designed specifically for hardware registers, MMU page tables, network headers, and compact UI flags:
```
pub bitfield PageEntry : u64 {
    present:    u1  @bits(0..1);
    writable:   u1  @bits(1..2);
    user:       u1  @bits(2..3);
    pwt:        u1  @bits(3..4);
    pcd:        u1  @bits(4..5);
    accessed:   u1  @bits(5..6);
    dirty:      u1  @bits(6..7);
    huge_page:  u1  @bits(7..8);
    phys_frame: u40 @bits(12..52);
    no_exec:    u1  @bits(63..64);
}
```
* **Compiler guarantees:**
  1. Zero bit-ordering ambiguity (unlike C bitfields): bit `0` is always the least significant bit (`LSB`) of the backing integer `u64`.
  2. Overlapping `@bits(lo..hi)` ranges or exceeding the backing integer width is compile-time error `E0220`.
  3. Unassigned bits (e.g., `8..12` and `52..63`) are guaranteed to be zeroed on construction.
  4. Field reads and writes compile to exact bit shifts/masks (`bextr`/`bfi`) on the backing register.

### 4.3. Enumerations (`enum`) and Tagged Unions (`union`)
* **`enum`:** always has an explicit or minimal sufficient integer backing type (`enum Color : u8 { Red = 0, Green = 1, Blue = 2, }`). An invalid `enum` value cannot be constructed in `.eon`.
* **`union` (Tagged Union / Sum Type):** always carries a hidden compact discriminant tag:
  ```
  pub union UiEvent {
      MouseMove: f32x2,
      KeyDown:   u32,
      Resize:    struct { w: u32; h: u32; },
      Quit,
  }
  ```
  - Size of `union` = aligned size of the largest variant + minimal tag (reusing variant niches when available).
  - Untagged ("bare" C-style) unions do not exist in `.eon`. In `.hal.eon`, `extern "C" union` is provided for C FFI with access restricted to `raw { ... }`.

### 4.4. Optional Types (`?T`) and Niche Optimization
* `?T` represents either a valid value of `T` or `null`.
* The compiler **must** apply niche optimization (zero-overhead optional):
  - `?*T`, `?*mut T`, `?rawptr[T]` -> `null` is encoded as address `0` (size = `usize`).
  - `?bool` -> `null` is encoded as byte `2` (size = `1` byte).
  - `?char` -> `null` is encoded as `0x11_0000` (size = `4` bytes).
  - `?GenId[T]` -> generation `0` is reserved as invalid (size = exactly `8` bytes!).
  - `?Id[p]` (and `?Link[Self]` in `ArenaPool`) -> encoded as `0xFFFF_FFFF` (size = `4` bytes); consequently `ArenaPool` capacity cannot exceed `2^32 - 2` elements.
  - `?Enum` with `< 256` variants -> next unused tag value.

### 4.5. Error Unions (`T ! ErrorSet`)
* `error IoError { NotFound, AccessDenied, BrokenPipe, OutOfSpace }` — each variant receives a compact `u16` numeric code, **unique program-wide**. Codes are assigned by the linker from symbolic names `(module, set, variant)`; the numbering scheme and reserved ranges for EonOS system errors are specified in [09-abi-modules-and-hotreload.md](09-abi-modules-and-hotreload.md) §1.3.
* Error set union: `IoError | MemError` (a flat set of unique codes; order is irrelevant).
* In memory and the ABI, returning `T ! E` passes values in **CPU registers**: a dedicated error-tag register (`0` = success, `!= 0` = error code) plus registers for the payload `T`. See [04-verification-and-arithmetic.md](04-verification-and-arithmetic.md) and [09-abi-modules-and-hotreload.md](09-abi-modules-and-hotreload.md) for details.

### 4.6. Arrays, Slices, and UTF-8 Strings
* **Fixed-size array `[N]T`:** a contiguous block of `N` elements of size `N * @size_of(T)`. Value type.
* **Slices `*[T]` and `*mut [T]`:** second-class references (non-owning), physically represented as the pair `{ ptr: usize, len: usize }` (`16` bytes on a 64-bit CPU, passed in 2 registers).
* **Strings (`*str`, `StrBuf`, `ArenaStr`):**
  - `str` — unsized byte sequence type **mathematically guaranteed to be valid UTF-8**.
  - `*str` — immutable string borrow `{ ptr, len }`. String literals have type `*str`.
  - Constructing a `*str` from a raw byte slice `*[u8]` is possible **only** through the validating constructor `str.from_utf8(bytes) -> *str ! Utf8Error` (SIMD validation on AVX2/NEON).
  - Byte indexing `s[i]` returns `u8` (only with a proof that `i < s.len`); substring slicing `s[lo..hi]` is forbidden without a proof of UTF-8 codepoint boundaries (use `s.slice_utf8(lo, hi) -> *str ! Utf8BoundaryError`, or iterate explicitly with `s.bytes()` / `s.chars()`).
  - Dynamic string construction in UI and OS code incurs zero hidden allocations: use `StrBuf` (a linear buffer taking `AllocCap`) or an `Arena` (via `arena.fmt(...)`, freed all at once in 0 cycles).

---

## 5. SoA Pool Types (`pool`)

A defining feature of Eon for outperforming C/C++/Rust is **first-class separation between logical object schema and physical memory layout (Struct of Arrays)**:

```
pub pool Widget {
    pos:    f32x2;
    size:   f32x2;
    color:  u8x4;
    flags:  u32;
    parent: ?Link[Self];   // link to another row of THIS pool, see §5.4
}
```

### 5.1. Physical Memory Layout
Instead of a single array of structs `[ {pos, size, color, flags, parent}, ... ]` (AoS), an instance of `ArenaPool[Widget]` or `SlotPool[Widget]` stores **N independent aligned column arrays** (aligned to a 64-byte cache-line boundary by default):

```
pos:    [ f32x2 | f32x2 | f32x2 | f32x2 | ... ]  <-- 64-byte aligned (ideal for AVX2/NEON)
size:   [ f32x2 | f32x2 | f32x2 | f32x2 | ... ]
color:  [ u8x4  | u8x4  | u8x4  | u8x4  | ... ]
flags:  [ u32   | u32   | u32   | u32   | ... ]
parent: [ ?Link | ?Link | ?Link | ?Link | ... ]
```

**Single-Column Passes**
1. During a color animation pass (`fade_animations`), the CPU loads **only the `color` column** into L1 cache (4 bytes per widget -> 16 widgets per 64-byte cache line!), without polluting the cache with `pos`, `size`, and `parent`.
2. A dense pass over a column is well suited for SIMD vectorization. The exact instruction count depends on the operation, backend, and target architecture.
3. Two `sys` functions can safely and concurrently write to `Widget.pos` and `Widget.color` on different CPU cores with zero false sharing (columns live on separate cache lines) and zero mutexes.

### 5.2. Hybrid Column Grouping (AoSoA / Hot-Cold)
If two fields are **always** accessed together (for example, `pos` and `size` during layout), they can be explicitly grouped via an embedded struct:
```
pub struct Box2D { pos: f32x2; size: f32x2; }
pub pool Widget {
    bounds: Box2D; // stored densely as a single [Box2D] column
    color:  u8x4;  // stored in a separate [u8x4] column
}
```
The programmer thus gets full control over memory layout (from pure AoS to pure SoA and hybrid AoSoA) without writing a single line of manual pointer arithmetic.

### 5.3. Row View: What `*T` Means for a `pool` Schema
For a `pool` schema `T`, no contiguous physical row `T` exists in memory (data is stored column-wise). Therefore, `*T` and `*mut T` where `T` is a `pool` schema represent a **Row View**: a pair `(pool_ptr, index)` (2 registers). Accessing `row.pos` compiles to `pool.pos_base[index]`. A Row View is a regular second-class reference (cannot be stored in a `struct`/`pool`), and the Law of Exclusivity distinguishes access paths per column: `&mut row.pos` and `&row.color` do not conflict. This is the exact view returned by `SlotPool.get/get_mut`, `each()` / `each_mut()`, and the `row` binding in `par for`. `SlotPool.remove(id) -> T` returns an **assembled** ordinary value (`T` by value; subject to the `> 16 bytes` rule in §6).

### 5.4. Inter-Row Links: `Link[Self]`
A field of type `Link[Self]` (or `?Link[Self]`) is allowed **only** inside a `pool` schema and refers to a row in **the same** pool. Its concrete representation depends on the pool container:
* in `ArenaPool` — `Id[p]` (4 bytes, where `p` is that pool instance; niche `0xFFFF_FFFF`);
* in `SlotPool` — `GenId[T]` (8 bytes, niche `gen == 0`).

A `Link` column belongs to pool `p`. `ArenaPool.reset()` resets all columns along with all links stored in them. Storing `Id[p]` in a `struct` or in columns of another pool is forbidden (`E0332`).

For an external link to a row in `SlotPool[Other]`, store `GenId[Other]` together with the context identifying the target pool instance ([03 §3.3](03-ownership-borrows-linearity.md)). For an `ArenaPool`, you may store a `u32` and re-acquire a branded index via `p.id_at(idx) -> Id[p] ! BoundsError`. That check confirms the index is within the current bounds of `p`; the application remains responsible for ensuring the row still corresponds to the intended logical object after a pool reset.

---

## 6. Copy Threshold Rule (`TrivialCopy <= 16 bytes`)

To enforce rule `ZM-06` (eliminating accidental `memcpy`):

1. **A type is `TrivialCopy`** if and only if:
   - It is not marked `linear` and contains no `linear` fields.
   - Its size satisfies `@size_of(T) <= 16` bytes (fits in two 64-bit registers or one 128-bit XMM/NEON CPU register).
2. **All types larger than `16` bytes** (and not `linear`) have **move-by-default** semantics:
   - Passing `big_struct` to a function or writing `let b = a;` **moves** the value (the source variable `a` becomes uninitialized; reading `a` triggers error `E0301`).
   - To read `big_struct` without moving it, pass a borrow `&big_struct` of type `*BigStruct` (1 CPU register).
   - To deliberately perform a bitwise copy (`memcpy`), the programmer **must** write the explicit operator `let b = copy a;`.

---

## 7. Specification Tests

### Must Compile
```
module example.types_ok;

pub struct Small { a: u64; b: u64; }        // 16 bytes -> TrivialCopy
pub struct Big   { data: [64]u8; }          // 64 bytes -> Move-by-default

pub fn use_types(s: Small, big: Big) -> (Small, Big) {
    let s2 = s;           // OK: 16 bytes, copied in registers
    let _s3 = s;          // OK: s is still usable
    let b2 = copy big;    // OK: explicit copy
    let _b3 = move big;   // OK: move
    return (s2, b2);
}
```

### Must Fail Compilation
```
module example.types_fail;

pub struct Big { data: [64]u8; }

pub fn bad_types(x: u8, big: Big) -> u32 {
    let b2 = big;
    let _b3 = big;        // ERROR E0301 (ZM-06): `big` already moved; use `&big` or `copy big`
    let y: u32 = x;       // ERROR E0105 (ZM-05): implicit u8 -> u32 coercion is forbidden
    return y;
}
```
