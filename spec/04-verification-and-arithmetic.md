# 04. Relational Bounds Verification, Arithmetic, and Error Handling

This chapter describes the core of Eon's "mathematical safety without runtime overhead": how the compiler proves `buf[i]` bounds and absence of integer overflow in microseconds **without Z3**, and how error handling works **without panics**.

---

## 1. Why Plain `min..max` Intervals Are Insufficient

Standard interval analysis tracks only constant bounds (`0 <= i <= 100`). It is **powerless** on dynamic slices:
```
fn sum(xs: *[u32]) -> u64 ! ArithError {
    var acc: u64 = 0_u64;
    var i: usize = 0_usize;
    while i < xs.len {
        acc = try acc +? xs[i].widen.[u64](); // `xs[i]`: how to prove i < xs.len when xs.len is unknown?
        i = i + 1_usize;                      // and how to prove that i + 1 cannot overflow usize?!
    }                                         // (the analyzer does NOT prove the accumulator `acc` — see §2.5, hence `+?`)
    return acc;
}
```
Here `xs.len` is dynamic. Plain intervals see only `i ∈ [0, MAX]` and `xs.len ∈ [0, MAX]`.

---

## 2. The Eon Relational Analyzer (Intervals + DBM)

The Eon compiler implements a deterministic **abstract interpreter over the product of two domains**:
1. **Interval Domain (`Interval`):** stores `[lo(v), hi(v)]` for each tracked integer variable `v` in exact mathematics (`i129`, avoiding overflow inside the analyzer itself).
2. **Difference-Bound Matrix Domain (DBM):** stores a directed weighted graph of inequalities of the form:
   $$x - y \le c \quad \text{and} \quad x \le c, \quad -x \le c$$
   where the graph vertices are:
   - local integer variables in SSA form;
   - symbolic lengths of slices and arrays (`xs.len`, `pool.len`);
   - a special zero vertex `0` (encoding constant intervals).

### 2.1. How DBM Instantly Proves the `sum(xs)` Loop
1. Upon entering the function, the analyzer knows `0 <= xs.len <= usize.MAX` (more precisely, on 64-bit platforms the length of any valid byte/element slice is guaranteed to be `<= isize.MAX`, providing massive headroom against `i + 1` overflow!).
2. Before the loop: `i = 0` -> edges `0 <= i <= 0` are added to the DBM.
3. On the branch `while i < xs.len`: inside the loop body, the edge **`i - xs.len <= -1`** (equivalent to `i < xs.len`) is added.
4. Checking `xs[i]`: requires `0 <= i` (proven, `i: usize`) and `i - xs.len <= -1` (present in the DBM!). **Proven in `O(1)`!**
5. Checking `i + 1`: from `i <= xs.len - 1` and `xs.len <= isize.MAX`, it follows that `i + 1 <= isize.MAX <= usize.MAX`. **Overflow is impossible; `+` is proven in `O(1)`!**

### 2.2. Compilation Speed Guarantee (Predictable Time Budget)
Unlike SMT solvers (Z3), which can stall exponentially on a single function:
* Only **integer variables participating in indexing, comparisons, shifts, or `where` contracts** are entered into the DBM (typically `k <= 32` variables per function).
* Complexity cap: if more than `64` active variables exist at a single program point, the least relevant variables are projected into the pure interval domain (`O(1)`), keeping per-function verification strictly bounded by a polynomial with a tiny constant — **fractions of a millisecond per function**.
* **Loop Convergence Algorithm (Widening + Narrowing):** at a `while`/`loop` header, a fixed number of iterations (up to 3) is executed, followed by a `widening` operator (to accelerate convergence) + `narrowing` using the exit condition or an explicit `where` loop invariant:
  ```
  while i < n where i <= n {
      ...
  }
  ```
* **Stability and Determinism:** if code compiles today, it is guaranteed to compile tomorrow on any machine **at the same analysis level `@analysis(N)`** (§2.4); heuristics and timeouts (as in Z3) are **strictly forbidden**.

### 2.3. Fact Inference Rules (Path-Sensitive Facts)
The analyzer automatically extracts facts from:
* Conditions `if a < b`, `if x != 0`, `if x <= 255` (and their negations in the `else` branch!).
* Early exits: after `if idx >= buf.len { return .OutOfBounds; }`, the analyzer knows `idx < buf.len` for the remainder of the block!
* Bitmasks: `x & 0xFF` yields interval `[0, 255]`; `x >> 24` on `u32` yields `[0, 255]`; `x % k` (with `k > 0`) yields `0 <= res < k`.
* `for i in 0..n` constructs -> inside the body, `0 <= i < n` holds automatically!
* `for item in slice` / `for id in pool.indices()` constructs -> 0 bounds checks by construction.
* `match` arms: inside the arm `0..=15 => ...`, the variable has interval `[0, 15]`.
* The `unreachable;` statement compiles **ONLY** if the DBM/interval domain at that point is contradictory (empty set of states — for example, after checking all possible values) (`ZM-15`).

### 2.4. Normative Analyzer Specification: `@analysis(N)` Levels
Decisions for `ZM-07` / `ZM-08` / `ZM-14` / `ZM-15` depend on analyzer precision, so the analyzer is **part of the language**, not an implementation detail. Each version of the algorithm (set of domains, widening/narrowing iteration count, inference rules in §2.3, projection thresholds) has an **analysis level** number. A package declares its level in `build.eon` (`analysis = 1`); the default is the earliest level defined in the specification. The compiler must:
* support all levels `<=` its own version and verify code **strictly according to the algorithm of the declared level**;
* treat any precision improvement (including new proofs that could make an `if` branch "always true" and trigger `ZM-14`) as a **new level**, enabled only explicitly.

Code that builds at level `N` builds on any compiler supporting `N`, with the exact same errors and the exact same eliminated checks. Level `1` is precisely the "Intervals × DBM" domain specified in this chapter.

### 2.5. Analyzer Axioms and Limits (Level 1)
**Axioms** (part of the specification rather than derived):
* **A1.** For every slice, array, and pool, `len <= isize.MAX`. Therefore `i < xs.len` implies `i + 1 <= isize.MAX <= usize.MAX`: `i = i + 1` under the loop condition `i < xs.len` is proven.
* **A2.** All `Id[p]` indices are strictly less than `p.len`.

**What the Level 1 analyzer does not prove** (and how to write it):
| Case | Why | What to Write |
| :--- | :--- | :--- |
| Accumulator `acc + x` in a loop | requires an invariant of the form `acc <= c·i`, which is not a difference constraint `x - y <= c` | `+?` (recommended: `add` + `jo`), `+|`, or `+%` where wraparound is the intended semantics (hashes, PRNGs) |
| Nonlinear index `y*w + x < w*h`, `i*stride` | products of variables lie outside DBM | `buf.at(i)` (a single comparison) or a function precondition `where idx < buf.len` |
| Backward inference from masks: `(x & M) == 0` ⇒ `x <= ~M` | bitmasks propagate only "forward" (§2.3) | an explicit comparison `if x > 255 { ... }` |

---

## 3. Interprocedural Contracts (`where`)

So that proofs are not lost across function calls, Eon supports **`where` preconditions** directly in the function signature (and in the exported `.emi` interface):

```
pub fn copy_prefix(dst: *mut [u8], src: *[u8], n: usize)
    where n <= dst.len && n <= src.len
{
    for i in 0..n {
        // DBM knows: i < n, n <= dst.len, n <= src.len -> i < dst.len && i < src.len!
        // 0 runtime checks!
        dst[i] = src[i];
    }
}
```

### 3.1. Permitted `where` Predicate Language
To keep contract checking instantaneous and decidable in polynomial time, `where` permits:
* Comparisons `==`, `!=`, `<`, `<=`, `>`, `>=` over parameters, their `.len`, fields, and integer constants.
* Linear offset combinations `a + c <= b`, `a + b <= c`.
* Bitwise properties: `x != 0`, `(x & (x - 1)) == 0` (power of two), `(x % align) == 0`.
* Logical `&&`.

### 3.2. Postconditions (`ensures`) for Return Values
A function guarantees a property of its result via an `ensures` clause placed after `where` and before the body; inside `ensures`, the `return` keyword refers to the returned value (outside `ensures` — `E0126`). The predicate language is the same as in §3.1; for `T ! E`, the postcondition applies to the success value. The compiler checks `ensures` at every `return` in the function, and the caller receives it as a fact:
```
pub fn clamp_u32(x: u32, hi: u32) -> u32
    ensures return <= hi
{
    if x > hi { return hi; }
    return x;
}
```
The caller automatically receives the fact `res <= hi` in its DBM!

---

## 4. Indexing and Slices: Strict Mode (`buf[i]` vs `buf.at(i)`)

1. **`buf[i]` (zero-cost):**
   - Compiles **only** if the analyzer has proven `0 <= i < buf.len`.
   - Emits exactly **1 load instruction** with no `cmp`/`jae` and no panic-handling code.
   - If unproven — **compile error `E0107` (`ZM-07`)** showing the exact facts known to the analyzer:
     ```
     error[E0107]: cannot prove index bounds `i < buf.len`
       --> src/parser.eon:42:16
        |
     42 |     let b = buf[i];
        |                 ^ known: i in [0..=usize.MAX], i - buf.len <= 0 (could be equal!)
        |
        = help: add a check `if i < buf.len { ... }`, strengthen `where`, or use `try buf.at(i)`
     ```
2. **`buf[lo..hi]` (zero-cost):**
   - Compiles only when `0 <= lo <= hi <= buf.len` is proven.
3. **Explicit checked methods (for unproven / external data):**
   - `buf.at(i) -> *T ! BoundsError` (or `*mut T ! BoundsError` via `buf.at_mut(i)`): inserts exactly one explicit comparison.
   - `buf.get(i) -> ?*T`: returns `null` when out of bounds.
   - `buf.sub(lo, hi) -> *[T] ! BoundsError`: safe dynamic subslice.

---

## 5. Safe Arithmetic Without UB or Hidden Overhead

Eon has no Undefined Behavior on overflow (unlike C) and no divergence between Debug and Release builds (unlike Rust).

| Category | Operators | Compilation Rule | Machine Code |
| :--- | :--- | :--- | :--- |
| **Proven (default)** | `+`, `-`, `*`, `/`, `%`, `<<`, `>>` | Compiles **only** with a static proof: no overflow, divisor `!= 0`, for `iN / -1` dividend `!= MIN`, shift `< bitwidth`. Otherwise `E0108`. | 1 clean CPU instruction (`add`, `sub`, `imul`, `shl`), **0 checks**! |
| **Wrapping** | `+%`, `-%`, `*%`, `<<%` | Always compiles (two's-complement wraparound modulo `2^N`; `<<%` masks the shift amount by `bitwidth - 1`). | 1 CPU instruction |
| **Saturating** | `+\|`, `-\|`, `*\|` | Always compiles; clamps the result to `[MIN, MAX]`. Ideal for colors, UI coordinates, and audio. | 1–2 instructions (native on SIMD/ARM) |
| **Checked** | `+?`, `-?`, `*?`, `/?`, `%?` | Returns `T ! ArithError` (`Overflow`, `DivByZero`). Must be handled via `try`/`catch`. | `add` + `jo` (CPU overflow flag!) |

> **Shift semantics (identical across all backends: native `x86_64`, `EIR-Interp`, `comptime`, and `@gpu`):**
> * `x >> k` on unsigned `uN` is a **logical** shift (vacated high bits filled with `0`, `shr`).
> * `x >> k` on signed `iN` is an **arithmetic** shift (high bits replicate the sign bit, `sar` / `OpShiftRightArithmetic`), i.e., rounding toward `-∞`: `-7_i32 >> 1 == -4`, `-1_i64 >> 63 == -1`.
> * `x << k` is bitwise identical for `uN` and `iN`; proven `<<` additionally requires that the result does not overflow the type.
> * For a logical shift of a signed value, use an explicit bitcast: `x.bitcast.[uN]() >> k`.

> **Notes on widening, floating-point numbers, and `libm`-free `core.math`:**
> * When adding two `u16` values, the compiler knows that `a.widen.[u32]() + b.widen.[u32]()` lies in `[0, 131070]`, which is guaranteed not to overflow `u32` — so plain `+` compiles with zero checks and without `+%`.
> * Floating-point numbers (`f16`, `f32`, `f64`) have no UB on overflow and strictly follow IEEE-754 (`Inf` / `NaN`), so plain `+`, `-`, `*`, `/` are permitted without overflow proofs.
> * The `core.math` module provides deterministic math without external `libm` in both the native `x86_64` backend and the reference `EIR-Interp`: hardware intrinsics `@sqrt` (`sqrtss`/`sqrtsd`) and `@fma` (`vfmadd213ss`/`vfmadd213sd`) are combined with pure-Eon implementations (`exp`, `log`, `sin`, `cos`, `tan`, `atan`, `atan2`, `pow`) accurate to `<= 1 ULP`, while time measurements via `ClockCap.now_ns()` bypass `libc` via the Linux `vDSO`.

---

## 6. Error Handling: `T ! ErrorSet` (Zero Panics, Zero Exceptions)

Eon has **no** `panic!`, `unwrap()`, `expect()`, `throw`, `setjmp/longjmp`, or stack unwinding tables (`.eh_frame` / `DWARF unwind`).

### 6.1. Declaring Error Sets
```
pub error ParseError {
    UnexpectedEof,
    InvalidUtf8,
    NumberOverflow,
}
```
Errors in Eon are lightweight typed codes (`u16` in memory, passed in a dedicated CPU register) carrying **no dynamic string allocations**. (If an application needs rich diagnostics — such as line/column positions in a compiler or UI — diagnostic context is written into a passed `*mut DiagArena`, keeping the error-return hot path weightless.)

### 6.2. Handling and Propagation
* **`try expr`:** (the operand is the entire expression to the right; see [01](01-lexical-and-syntax.md) §3) if `expr` returns an error `e`, executes active `errdefer` statements (in LIFO order) and immediately returns `e` from the current function. The caller's error type must be a superset of `E` (otherwise error `E0401`).
* **`expr catch |e| { ... }`:** intercepts the error and either handles it, produces a value of the same type `T`, or executes `return` / `break` / `continue`.
* **`expr catch fallback_value`:** shorthand to substitute a fallback value on any error.
* **Exhaustive `match` on errors:**
  ```
  let val = parse_int(input) catch |err| match err {
      .UnexpectedEof  => 0_u32,
      .InvalidUtf8    => return .BadEncoding,
      .NumberOverflow => u32.MAX,
  };
  ```
* **Mandatory handling (`ZM-02`):** ignoring `T ! E` is forbidden even via `_ = fallible();`! An error can only be discarded explicitly via `_ = fallible() catch {};` (and in strict `@kernel` profiles, even an empty `catch {}` triggers a lint requiring a comment).

---

## 7. Specification Tests

### Must Compile
```
module example.verify_ok;

pub fn dot_product(a: *[f32], b: *[f32]) -> f32
    where a.len == b.len
{
    var sum: f32 = 0.0_f32;
    for i in 0..a.len {
        // DBM: 0 <= i < a.len && a.len == b.len => i < b.len!
        // Both a[i] and b[i] compile with 0 bounds checks!
        sum = sum + (a[i] * b[i]);
    }
    return sum;
}

pub fn safe_div(a: u32, b: u32) -> u32 ! ArithError {
    if b == 0_u32 {
        return .DivByZero;
    }
    // DBM knows: b >= 1 -> division `/` is safe!
    return a / b;
}

pub fn sum_bytes(xs: *[u8]) -> u64 ! ArithError {
    var acc: u64 = 0_u64;
    for i in 0..xs.len {
        acc = try acc +? xs[i].widen.[u64]();   // accumulator: checked `+?`, index proven by loop
    }
    return acc;
}
```

### Must Fail Compilation
```
module example.verify_fail;

pub fn bad_access(xs: *[u8], idx: usize) -> u8 {
    return xs[idx];            // ERROR E0107 (ZM-07): cannot prove idx < xs.len
}

pub fn bad_add(a: u32, b: u32) -> u32 {
    return a + b;              // ERROR E0108 (ZM-08): u32 overflow possible; use +%, +|, or +?
}

pub fn bad_unreachable(x: u32) -> u32 {
    if x < 10_u32 {
        return x;
    }
    unreachable;               // ERROR E0115 (ZM-15): point is reachable (e.g., x == 10)
}
```
