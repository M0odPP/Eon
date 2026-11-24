# 03. Ownership, Lifetimeless Borrows (No `'a`), Linear Types, and Pools

This chapter specifies Eon's memory and resource model. Borrows are verified without lifetime annotations, while resource-release obligations are expressed through linear types and explicit cleanup operations.

---

## 1. Second-Class References

Instead of a reference graph annotated with `'a` lifetimes, Eon uses a **second-class reference model** (inspired by Hylo/Val and Austral, adapted for `pool`s and OS development):

### 1.1. What a Reference Is in Eon
> **Syntax (v0.2):** a borrow *type* is written `*T` / `*mut T`, a borrow *expression* is `&x` / `&mut x`, and *dereferencing* is postfix `p.*`. The token `*` in expressions is strictly multiplication.

* `*T` — shared read-only borrow.
* `*mut T` — exclusive mutable borrow.
* `*[T]`, `*mut [T]` — borrow of a contiguous slice `{ ptr, len }`.
* `*str` — borrow of a valid UTF-8 string `{ ptr, len }`.

### 1.2. Three Strict Restrictions on Second-Class References
1. **Forbidden in data types:** A reference (`*T`, `*mut T`, `*[T]`, `*str`) **cannot** be a field of a `struct`, `union`, `pool`, an element of an array `[N]T`, or stored in a container `Buf[T]`. Attempting to declare `struct Node { next: *Node; }` is an immediate compile-time error `E0310`.
   * Object graphs are built using `Id[pool]` and `GenId[T]`.
2. **Lifetime bounded by the lexical scope of a call or binding:** References exist only as function parameters and local `let` bindings tied to a concrete owner higher up the call stack.
3. **The Law of Exclusivity is checked locally at the call site in `O(1)`:**
   When passing arguments to a function (or creating a local borrow), the compiler checks **access paths**:
   - Two `*T` borrows of the same path are **allowed**.
   - A `*mut T` conflicts with any other `*T` or `*mut T` to the same root or any prefix/suffix of it (`E0118`).
   - **Disjoint field and column splitting is allowed:** `swap(&mut s.left, &mut s.right)` and `update(&mut pool.pos, &pool.color)` are completely legal — the compiler knows that `s.left` and `s.right` do not overlap!

### 1.3. Returning a Borrow from a Method: `view fn` (Projected Borrows)
How do you write `pool.get(id)` or `map.lookup(key)` returning a borrow to an element without `'a` lifetimes?
For this, Eon provides the **`view fn`** modifier:
```
impl[T] SlotPool[T] {
    pub view fn get(self: *Self, id: GenId[T]) -> ?*T { ... }
    pub view fn get_mut(self: *mut Self, id: GenId[T]) -> ?*mut T { ... }
}
```
**Contract of `view fn`**
* A `view fn` has exactly **one** borrow source (`self`, or its sole reference parameter; if multiple reference parameters are present, the returned borrow is always tied to `self`, or — if there is no `self` — to the first parameter explicitly marked `from`).
* At the call site, the returned reference `let w = pool.get_mut(id);` **locks exactly `pool`** until the last use of `w` (lexically or via single-function NLL — Non-Lexical Lifetimes).
* On every exit path, any returned borrow must originate from the declared source. This rule also applies to references nested inside `?T` or the success payload of `T ! E`.
* A `view fn` may return not only `*T`, but also a **borrowing value** (see §1.5) — this is how constructors like `ArenaPool.init` and `Arena.alloc` work.

### 1.4. Non-Escaping Closures (Second-Class Closures)
Closures of the form `|arg| { ... }` are also second-class values:
* They cannot be stored in a `struct`/`pool` or returned from a function (zero heap allocations, no `Fn`/`FnMut`/`FnOnce`/`Box<dyn Fn>`).
* They are passed as a register pair `{ fn_ptr, env_ptr }` or completely inlined (in `drain`, `with_scope`, `sort_by`).
* Variables captured from enclosing scopes obey the exact same `*T` / `*mut T` exclusivity and `linear` ownership rules as ordinary function arguments.

### 1.5. Borrowing Values
Some values are syntactically not references, yet carry a lock on a source: `ArenaPool[T]`, `ArenaStr`, or a slice returned by `arena.alloc`. They obey the same rules as second-class references:
1. Such a value can only be obtained from a `view fn` (including `init` constructors). The source (`self` or the `from` parameter) remains locked **until the value's last use** (single-function NLL).
2. It cannot be stored in a `struct` / `union` / `pool` / `[N]T` / `Buf[T]`, returned from a regular function, or captured by a closure that outlives the stack frame (`E0311`).
3. It can be passed down the stack via `*T` / `*mut T` references (`p: *mut ArenaPool[Widget]`).
4. It can be moved within its own stack frame (`ui = ui.reset();`).

Consequence: a function cannot construct an `ArenaPool` and return it — the caller creates the pool and passes it downward. Pools that must live in a `struct` or outlive a stack frame are `SlotPool`s (which own their memory directly, see §3.3).

### 1.6. Second-Class View Structs (`view struct`)
For parsers, cursors, AST walkers, and code generators that benefit from grouping several borrows and indices without `'a` annotations, Eon provides **`view struct`**:
```
pub view struct Parser {
    src: *[u8];
    toks: *[Tok];
    pos: usize;
    out: *mut Buf[u8];
}
```
**Rules for `view struct`:**
1. **Borrowed fields allowed:** a `view struct` may contain `*T`, `*mut T`, `*[T]`, `*mut [T]`, `*str`, borrowing values (§1.5), and other `view struct`s. Fields of `comptime type` (`E0210`) and branded `Id[p]` (`E0332`) remain forbidden.
2. **Strictly second-class (`E0310`):** a `view struct` is itself a second-class value — it cannot be stored in a regular `struct`, `union`, `pool`, `[N]T`, or `Buf[T]`, and cannot be returned from a regular (non-`view`) function.
3. **Law of Exclusivity at construction (`E0118`):** when constructing `Parser.{ a: &mut x, b: &mut x }`, field initializers are checked for disjoint access paths in the exact same way as function call arguments; active borrows for all fields are held by the local `view struct` variable until its last use (NLL).
4. **Passing down the stack and `view fn`:** a `view struct` can be passed to functions and methods (`self: *Parser`, `self: *mut Parser`, or by value) and returned from a `view fn` provided that the provenance of all its borrowed fields reduces to the declared source of that `view fn`.

---

## 2. Linear Types (`linear`) and `defer` / `errdefer`

In Rust, a `File` or `MutexGuard` is *affine*: it can be implicitly dropped at the end of a block or "forgotten" via `core::mem::forget`, causing a silent physical page leak or permanent deadlock in an OS kernel.

In Eon, critical resources are declared as **`linear struct`** (or **`linear union`**):
```
pub linear struct PhysPage {
    base: PhysAddr;
}
```

### 2.1. Axioms of Linearity
1. **No copying:** a `linear` value can never be copied, neither implicitly nor via `copy` (`E0320`).
2. **No implicit destruction:** exiting a scope (via `}`, `return`, `break`, `continue`, `try`) while a `linear` variable remains "live" (unconsumed) is a **hard compile-time error `E0109` (`ZM-09`)**.
3. **No double consumption:** a `linear` variable can be passed by value (moved) **exactly once** (`E0321`).
4. **Linearity propagation:** any `struct` or `union` containing at least one `linear` field **must** itself be declared `linear` (`E0322`).
5. **The sole way to destroy a `linear` value** is to destructure it into its fields (inside a module where its fields are visible, e.g., in `pub fn free_page(p: PhysPage, pmm: *mut Pmm)`) and handle all linear fields:
   ```
   pub fn free_page(page: PhysPage, pmm: *mut Pmm) {
       let PhysPage.{ base } = page; // `page` is consumed; `base: PhysAddr` is not linear
       pmm.release_frame(base);
   }
   ```
6. **Borrowing a `linear` value is allowed:** passing `*PhysPage` or `*mut PhysPage` to functions does not consume ownership.

### 2.2. Branching and Loops with `linear`
* In `if / else` and `match`: the state of every linear variable (live vs. consumed) **must match at the exit of all reachable branches** (`E0323`).
* Inside loops (`while`, `for`, `loop`):
  - An outer `linear` variable cannot be consumed inside a loop body (except on a branch that immediately executes `break` or `return`), otherwise the next iteration would double-consume it (`E0324`).
  - Every `linear` variable created inside a loop iteration must be consumed before the end of the iteration (and before every `continue`/`break`/`return`/`try`).

### 2.3. Resolving `linear` + `try`: `defer` and `errdefer`
To keep working with multiple linear resources and early error returns (`try`) **simple and ergonomic** (without deeply nested `catch` blocks), the linearity analyzer understands `defer` and `errdefer` as **conditional consumers**:

```
pub fn map_two_pages(pmm: *mut Pmm, vmm: *mut Vmm) -> (PhysPage, PhysPage) ! MemError {
    let p1 = try pmm.alloc_page();
    errdefer pmm.free_page(p1); // Consumes p1 ONLY on error paths below!

    let p2 = try pmm.alloc_page();
    errdefer pmm.free_page(p2); // On error: first free_page(p2), then free_page(p1) (LIFO!)

    try vmm.map(&p1, &p2);      // If this fails: p2 and p1 are freed automatically!

    // On the normal path, errdefer does NOT run -> p1 and p2 remain live and are returned to the caller!
    return (p1, p2);
}
```

**Exact Compiler Checking Semantics:**
1. The compiler tracks **two independent ownership states** for each local `linear` variable `x`:
   - `NormalState(x) ∈ { Live, Consumed }`
   - `ErrorState(x)  ∈ { Live, Consumed }`
2. Upon binding `let p1 = ...;`: `Normal = Live`, `Error = Live`.
3. Upon registering `errdefer consume(p1);`:
   - Checks that `ErrorState(p1) == Live`.
   - Transitions `ErrorState(p1)` to `Consumed` (while `NormalState(p1)` stays `Live`!).
   - Performing `move p1` on the normal path is forbidden if another `try` appears below (otherwise an error would cause `errdefer` to free an already-moved `p1` -> error `E0325`!). To safely transfer ownership of `p1` before subsequent `try` operations, use `disarm` or return at the end of the function.
4. Upon registering `defer consume(p1);`: both states (`Normal` and `Error`) transition to `Consumed`, while `p1` remains borrowable via `*p1`/`*mut p1` until the end of the block (with `move p1` disallowed).
5. At every `try` or `return Err`: the compiler checks that `ErrorState == Consumed` for all `linear` variables.
6. At every normal `return val`: the compiler checks that `NormalState == Consumed` for all `linear` variables (other than the returned value).

### 2.4. Linear Containers (`LinearBuf[T]`)
If `T` is a `linear` type (such as `PhysPage` or `FileHandle`), it cannot be stored in an ordinary `Buf[T]` (`E0326`), because an ordinary buffer can be cleared by wiping memory in bulk.
To store collections of linear resources, Eon provides **`LinearBuf[T]`** (itself a `linear struct`):
* A `LinearBuf[T]` can be destroyed **only** through an exhaustive `drain` loop that is guaranteed to yield every live element by value:
  ```
  pages.drain(alloc, |page| {
      pmm.free_page(page); // Must consume every page!
  });
  ```

---

## 3. Hybrid Memory Model: `Arena`, `ArenaPool`, and `SlotPool`

Eon eliminates the graph problem in OS and UI code (where Rust forces a choice between `unsafe` and slow `Rc<RefCell<T>>`) by providing **three complementary memory primitives**:

### 3.1. `Arena` (1-Cycle Linear Bump Allocator)
* `linear struct Arena`: owns a contiguous memory region (or chain of pages).
* Allocation: `arena.alloc(size)` — a simple aligned pointer bump (`1–2` CPU cycles, zero fragmentation).
* Deallocation: **all at once only** (`arena.reset()` for UI frame reuse, or `arena.destroy(alloc)` to return memory to the OS).
* Safety on `reset()`: `arena.reset()` takes `*mut Arena`. Everything allocated from the arena (slices from `arena.alloc`, `ArenaPool.init(&mut arena, …)`, `ArenaStr`) is a **borrowing value** (§1.5) that locks the arena until its last use. Therefore, at the moment `reset()` or `destroy()` is called, no live values derived from that arena can exist: any violation is error `E0118` (conflict between `&mut arena` and a live borrow), never a dangling pointer.

### 3.2. `ArenaPool[T]` — Branded Dense SoA Pool (32-Bit `Id`, 0 Checks!)
Ideal for a **UI frame arena**, **compiler AST**, **parsers**, and **temporary render buffers** — anywhere objects are only appended and the entire pool is later reset at once.

```
var ui = try ArenaPool.[Widget].init(&mut frame_arena, 1024);
// `ArenaPool.init` is a `view fn`: `ui` is a borrowing value locking `frame_arena`
// until the last use of `ui` (calling `frame_arena.reset()` while `ui` is live -> E0118).
// `ui` receives a unique zero-sized lexical brand!
// Returned index type: `Id[ui]` (32-bit u32, incompatible with `Id[ui2]`!).
let panel: Id[ui] = try ui.push(.{
    pos: .{ 0.0, 0.0 },
    size: .{ 800.0, 48.0 },
    color: .{ 24, 24, 28, 255 },
    flags: 0_u32,
    parent: null,
});

// 0 bounds checks and 0 generation checks in machine code!
// The compiler mathematically knows: `panel` came from `ui`, and `ui` cannot delete individually!
ui.pos[panel] = .{ 10.0, 20.0 };
```

**Invariants of `ArenaPool[T]` (why 0 checks are 100% sound):**
1. `ArenaPool[T]` **physically has no** `remove(id)` method. Pool length is monotonic: `len_after >= len_before`.
2. An index `Id[ui]` carries a brand tied to the specific variable `ui` (or pool parameter). Passing `Id[ui_a]` into `ui_b` is type error `E0330`.
3. Calling `ui.reset()` **invalidates** the old brand (consumed via linear rebinding `ui = ui.reset();`), rendering every previous `Id[ui]` invalid at compile time (`E0331`)!
4. Passing to functions uses an explicit branded parameter (only in function signatures and local bindings, **never in struct fields**):
   ```
   fn move_widget(p: *mut ArenaPool[Widget], id: Id[p], dx: f32) {
       p.pos[id].x = p.pos[id].x + dx; // 0 bounds checks!
   }
   ```
5. **Where `Id[p]` may live:** only in local bindings, function parameters, and `Link[Self]` columns of **its own** pool `p` ([02 §5.4](02-types-and-memory-layout.md)). Storing `Id[p]` in a `struct` or in another pool is error `E0332`. This prevents brands from "infecting" structs: the language has no brand-parameterized structs. When an index must persist across frames, store a `u32` and re-obtain an `Id` via the explicit checked lookup `p.id_at(idx) -> Id[p] ! BoundsError` (a single comparison).
6. **Reset by owner only:** `reset(self: ArenaPool[T]) -> ArenaPool[T]` takes the pool **by value**, so only the stack frame owning `ui` can reset it; a function receiving `p: *mut ArenaPool[T]` along with `Id[p]` cannot reset the pool.
7. **Capacity limit:** `cap <= 2^32 - 2` (`0xFFFF_FFFF` is reserved as the `?Id[p]` niche).

### 3.3. `SlotPool[T]` — Long-Lived Generational SoA Pool
Ideal for **OS kernel process/file/socket tables**, **persistent UI widget trees**, and **game entities** — anywhere objects are dynamically created and deleted in arbitrary order and handles are freely stored across other structs and pools.

* **Ownership:** `SlotPool[T]` is a `linear struct` that **owns its own memory** (unlike `ArenaPool`). It can be stored in a `struct` and outlives a stack frame. Created and freed via `AllocCap`: `SlotPool.[T].init(alloc: *mut AllocCap, cap) -> SlotPool[T] ! OutOfMemory`, `pool.destroy(alloc)` (consumes the pool; any remaining `linear` elements must be handled via `drain`).
* **Physical structure of `SlotPool[T]`:**
  - Stores data in dense SoA columns + a parallel `gen: []u32` column (slot generation: **odd = live, even = free**, `0` = retired slot, see below) + an intrusive free-list of free indices.
  - Handle `GenId[T]` = `{ index: u32, gen: u32 }` (`8` bytes -> `TrivialCopy`!). A live slot always has an odd `gen`, so a valid `GenId` never has `gen == 0` — providing the niche for `?GenId[T]` (exactly `8` bytes).
  - **Generation progression:** `insert` increments `gen` by 1 (even → odd); `remove` increments `gen` by 1 (odd → even). On removal when `gen == 0xFFFF_FFFF`, the generation wraps to `0` and the slot is permanently retired. Thus a `(index, gen)` pair is never re-issued during an instance's lifetime.
* **Handle provenance:** The type `GenId[T]` identifies the element schema. The application must use a handle with the pool instance that issued it. A different `SlotPool[T]` instance — including one created to replace a destroyed instance — could accept the same `(index, gen)` pair and resolve to a different live element. If handles from multiple instances are stored together, the application must store an instance identifier alongside `GenId[T]` and validate it before lookup.
* **Operations:**
  - `pool.insert(alloc, val) -> GenId[T] ! OutOfMemory` (amortized `O(1)`; `alloc: *mut AllocCap` is needed only when growing) and `pool.try_insert(val) -> GenId[T] ! OutOfMemory` (fixed capacity, no allocator).
  - `pool.remove(id: GenId[T]) -> T ! StaleId` validates the handle, returns the element, and invalidates the handle within this pool instance.
  - `pool.get(id: GenId[T]) -> ?*T`, `pool.get_mut(id: GenId[T]) -> ?*mut T`, and `pool.at(id: GenId[T]) -> *T ! StaleId` check `id.index < cap && pool.gen[id.index] == id.gen` and return a borrow to the live element. For a `pool` schema `T`, this is a Row View ([02 §5.3](02-types-and-memory-layout.md)).
  - **Fast linear iteration (`for` / `par for`):** `SlotPool` also maintains a compact dense index of live elements (a sparse set), so iterating with `par for pool.each_mut()` scans **contiguously over dense SoA columns with no gaps and no generation checks**!

---

## 4. Summary Table: When to Use What

| Use Case | Eon Tool | Reference Size | Runtime Checks | UAF / ABA Protection |
| :--- | :--- | :---: | :---: | :--- |
| Function argument, local view | `*T`, `*mut T`, `*[T]` | 8 / 16 bytes | **0** | Static (Second-Class) |
| Kernel/OS resource (page, file, lock) | `linear struct` + `errdefer` | — | **0** | Static (exact single release) |
| UI frame, compiler AST, parser (frame-scoped) | `ArenaPool[T]` + `Id[p]` | 4 bytes | **0** | Static (Brand + epoch reset + arena lock) |
| OS kernel (processes), persistent UI | `SlotPool[T]` + `GenId[T]` | 8 bytes | Index & generation (`get`) / **0** (`each`) | Per-instance generation check |

> A pool that needs to be stored in a `struct` or passed across stack frames is always an owning `SlotPool`. `ArenaPool` is a borrowing value (§1.5) and lives only within its creator's stack frame.

---

## 5. Specification Tests

### Must Compile
```
module example.ownership_ok;

pub fn swap_fields(p: *mut struct { a: u32; b: u32; }) {
    let tmp = p.a;
    p.a = p.b;
    p.b = tmp;
}

pub fn pass_disjoint(a: *mut u32, b: *mut u32) {
    a.* = a.* +% 1_u32;
    b.* = b.* +% 2_u32;
}

pub fn caller() {
    var pair = .{ a: 1_u32, b: 2_u32 };
    pass_disjoint(&mut pair.a, &mut pair.b); // OK: disjoint paths!
}
```

### Must Fail Compilation
```
module example.ownership_fail;

pub linear struct LockToken { id: u32; }

pub struct BadHolder {
    ref_x: *u32;                  // ERROR E0310: second-class reference cannot be a struct field
}

pub fn leak_lock(tok: LockToken, cond: bool) {
    if cond {
        let LockToken.{ id: _ } = tok;
        return;
    }
    return;                       // ERROR E0109 (ZM-09): `tok` is unconsumed on the `!cond` branch
}

pub fn alias_bug() {
    var x: u32 = 10_u32;
    pass_disjoint(&mut x, &mut x); // ERROR E0118 (ZM-18): aliasing two *mut borrows of `x`
}

pub pool Node { next: ?Link[Self]; }

pub struct HoldsPool {
    nodes: ArenaPool[Node];       // ERROR E0311: borrowing value cannot be stored in a struct
}

pub fn reset_while_borrowed(arena: *mut Arena) ! MemError {
    var nodes = try ArenaPool.[Node].init(arena, 64_usize);  // `nodes` locks `arena`
    arena.reset();                // ERROR E0118: `arena` is borrowed by live `nodes`
    _ = nodes.len();              // last use of `nodes`
}
```
