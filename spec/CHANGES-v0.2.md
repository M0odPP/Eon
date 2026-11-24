# Eon: Changes from `v0.1` → `v0.2`

Version `v0.2` is the result of cross-checking all specification chapters and `grammar.ebnf` against one another. This document lists what was fixed, which design decisions were made (worth review by the language author), and what was deliberately left open.

---

## 1. Breaking Syntax Changes (Invalidate `v0.1` Examples)

| Before (`v0.1`) | After (`v0.2`) | Rationale |
| :--- | :--- | :--- |
| `*x`, `*mut x` in expressions: both "take a borrow" and "dereference" | Creation: `&x`, `&mut x`. Dereference: postfix `p.*`. Types remain `*T`, `*mut T`, `*[T]`, `*str` | The same token meant different operations depending on operand type; `*x` for `x: *u32` was ambiguous |
| `ArenaPool[Widget].init(...)` in an expression | `ArenaPool.[Widget].init(...)` | Inside an expression, `[` is indexing; explicit generic arguments use `.[T]` only |
| `try a +? b` parsed as `(try a) +? b` | `try` is a prefix over the **entire** expression: `try a +? b` == `try (a +? b)`; valid only at the start of an expression | Otherwise `try` would be unusable with checked arithmetic |
| `where return <= hi` | `ensures return <= hi` | `return` is not an expression; postconditions receive their own dedicated clause (`E0126` outside `ensures`) |
| Block return: draft using `yield`, `break :label` | Block expression = trailing expression without `;`; `fn` bodies return only via `return` (`E0124`); `break 'l value;` is for `loop` only | Section §4.2 of Chapter 01 was an early working draft that contradicted the grammar |
| `par reduce` (syntax absent from the grammar) | Method `it.par_reduce(op, init)` with a fixed set of `op` | See §3 below |
| Keywords: "51", actually 54 | 54 hard keywords (`dyn`, `ensures`, `from` added; `with` removed) + 2 contextual keywords (`read`, `write`) | Examples in the spec itself use `.read()`, `.write()`, and `fn write(...)` |

New grammar elements: variant literal `.Variant`; struct patterns `Type.{ a, b: p }` and `.{ … }`; pattern on the right-hand side of `is`; attributes after a field type (`@offset`); `read`/`write` on `@gpu` parameters; attributes on `let`/`var` (`@gpu_shared`); `extern "C" union`; parameter modifiers `mut` and `from`; empty closure `|| { … }`; `where` on `struct`/`pool`/`union`/`impl`/`type`; `comptime if` and `builtin_expr` (`@target_arch()`, `@size_of(T)`); `return`/`break`/`continue` in `match`/`orelse`/`catch`; `asm` operands of the form `out(reg) val`; shorthand error return `fn f() ! Err`; `_ = expr;` (`discard_stmt`); `loop` as an expression; `@extern("C")`.

## 2. Safety and Semantics Fixes

1. **`ArenaPool` and arenas (03 §1.5, §3.1, §3.2).** Previously, a pool created from an arena could outlive `arena.reset()` (since a pool cannot store a reference). **Borrowed values** are now introduced: `ArenaPool.init` is a `view fn` whose result pins the arena until last use; calling `reset()` while the pool is live is `E0118`. A borrowed value cannot be stored in a `struct` or returned from a function (`E0311`).
2. **`Id[p]` brands do not "infect" structs (03 §3.2, 02 §5.4).** `Id[p]` lives only in locals, parameters, and `Link[Self]` columns of its own pool (`E0332` everywhere else). Across frames, store a plain `u32` and reconstruct `Id` via `p.id_at(idx) ! BoundsError`.
3. **`Link[Self]` (02 §5.4).** A `pool`-schema field type for referencing another row of the same pool; expands to `Id[p]` (`ArenaPool`) or `GenId[T]` (`SlotPool`). This eliminates the contradiction between `parent: ?GenId[Widget]` and using the schema inside an `ArenaPool`.
4. **Row View (02 §5.3).** For a `pool` schema, `*T` is a `(pool_ptr, index)` pair: no physical row exists in SoA layout.
5. **`SlotPool` bug (03 §3.3).** `remove` previously performed `gen +% 2`, leaving the slot marked "live". It now does `+1` on `insert` and `+1` on `remove`; a slot reaching `gen == 0xFFFF_FFFF` is permanently retired, making ABA protection strict. `SlotPool` is now an owning (`linear`, backed by `AllocCap`) structure.
6. **Violations of our own `E0310`.** `SpinGuard` stores `rawptr_mut` (in an `@hal` module); `Env` is specified as a struct of capability values rather than references.
7. **Fibers (07 §3.1, §3.3).** Application code holds `IoCap` (a copyable handle) rather than `*mut IoRing`. `spawn_fiber` accepts only owned arguments (`E0731`). Borrowing parent data is allowed only via `with_scope`, whose soundness is an explicit TCB axiom.
8. **Closure effects (07 §1.3).** `@effects_of(f)` and `@stack_of(f)` for higher-order functions.
9. **ZM-06.** Implicit copying of values `>16 B` does not exist (move-by-default); the rule is redefined as diagnostic `E0301`, with `E0106` reserved.
10. **Analysis levels `@analysis(N)` (04 §2.4).** The analyzer is now part of the language specification: precision improvements ship as a new opt-in level, so compiler upgrades never trigger spurious new ZM-14 errors or break builds.
11. **Analyzer axioms and limits (04 §2.5).** `len <= isize.MAX` is an axiom. Added a table of what level 1 does not prove (accumulators, nonlinear indices, backward inference from bitmasks) and how to write code around them. The `sum` example was rewritten with `+?`.
12. **Determinism.** `par_reduce` on `f32/f64` permits only `.Min`, `.Max`, and `.AddTree` (a fixed reduction tree independent of core count; `E0603` for plain `+`). Automatic contraction into `fma` is forbidden. The wording "bit-for-bit across all OSes" was replaced with "identical semantics and deterministic results".
13. **ABI (09).** Error code in `r15` (source of truth) + mirror in `CF`; `r11` is no longer overloaded. Introduced program-wide error code numbering assigned by the linker, with `0x0001–0x0FFF` reserved for system errors (09 §1.3).
14. **`.emi` (09 §3.1).** `InterfaceHash` now includes the bodies of `pub` generics, `@inline`, and `comptime fn` — otherwise Early Cutoff would yield unsound incremental builds.
15. **Hot-Reload (09 §4.3).** A patch is not applied to live fibers if the computed stack bound increased; it is applied lazily instead.
16. **Layout Sharing (05 §1.2).** Specified passing `T`'s methods via a hidden dictionary and documented the boundaries of applicability.
17. **Miscellaneous.** `i64.MIN` literal; lexer rule "character literal vs label"; unified `BoundsError` and `ArithError` names replacing ad-hoc sets in tests; separated `extern "C"` (import) and `@extern("C")` (export); fixed the error code registry; `clamp_byte` test no longer asserts an unprovable fact; honest lookahead budget in Chapter 01 §1.

## 3. New Error Codes

`E0124` (trailing expression in a `fn` body) · `E0126` (`return` outside `ensures`) · `E0311` (borrowed value stored or returned) · `E0332` (`Id[p]` in a `struct` or foreign pool) · `E0603` (`+` in `par_reduce` for floats) · `E0731` (non-owned argument to `spawn_fiber`) · `E0823` (`read`/`write` in a regular `fn`) · `E0902` (error code space overflow).

---

## 4. Decisions Made in This Revision (Please Review)

These items reflect choices made where the original spec was ambiguous. They should be reviewed before starting Stage 0, as they shape the feel of the language:

- `&x` / `&mut x` / `p.*` syntax in place of overloaded `*`;
- the "borrowed values" model instead of a storable `ArenaPool` (trade-off: a pool cannot be returned from a function or stored in a `struct`; use `SlotPool` for that);
- `Id[p]` cannot be stored outside columns of its own pool (trade-off: explicit revalidation via `id_at`);
- `IoCap` as a copyable ring handle plus the `with_scope` axiom;
- `try` as a prefix over the whole expression;
- `ensures` as a dedicated keyword;
- unified linker-assigned error code numbering.

## 5. Deliberately Left Open

These questions are better answered empirically via the real-world validation projects (Design Freeze Gate) rather than decided prematurely in the spec:

1. **Cost of Row View** `(pool_ptr, index)` for wide pools: needs measurement on UI layout workloads.
2. **Ergonomics without storable `Id`:** how inconvenient it is to write an AST and widget tree without `Id[p]` inside structs. To be tested on the compiler itself and the UI project.
3. **Known-bits domain** for backward inference from bitmasks: candidate for `@analysis(2)`.
4. **Relaxing hygiene rules** (ZM-01, 03, 14, 17) in dev mode: a product decision rather than a safety one; for now the spec keeps them as errors always.
5. **Performance.** Comparative results for C, Rust, and Eon are established by benchmarks (see [bench/BENCHMARKS.md](../bench/BENCHMARKS.md) and [ROADMAP.md](../ROADMAP.md)).
6. **Formalizing borrowed values** (03 §1.5) and the `with_scope` axiom — prime candidates for validation in Stage 0 via UI tests and a sketch of a formal model.
