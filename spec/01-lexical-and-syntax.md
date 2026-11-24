# 01. Lexical Structure, Syntax, Modules, and Name Resolution Rules

---

## 1. Syntactic Design Goals

Eon's syntax is designed for two simultaneous goals:
1. **Maximum lexing and parsing throughput (target threshold: > 500,000 lines/sec on 1 CPU core):**
   - The declaration grammar is strictly **LL(1)** (zero backtracking). Honest lookahead budget: exactly three places need slightly more — (1) distinguishing the pattern `IDENT` vs. `IDENT.…` (one extra token), (2) the lexer distinguishing a character literal `'x'` from a loop label `'x:` by inspecting the character after the first scalar, and (3) the token `||` is always logical OR, while a zero-argument closure is written `|| { … }` and parsed by the `closure_expr` rule. There is no backtracking anywhere.
   - Expressions are parsed via **Pratt Parsing** with a fixed precedence table.
   - Explicit semicolons `;` and mandatory curly braces `{}` ensure instant error recovery and eliminate automatic semicolon insertion (ASI) bugs.
   - Generics use `[T]` in type positions and `.[T]` for explicit instantiation in expressions — completely eliminating the `< >` (turbofish) ambiguity of C++ and Rust.
2. **Human clarity and Zero-Mercy verification:**
   - Every declaration begins with a unique keyword (`fn`, `sys`, `let`, `var`, `struct`, `pool`, ...).
   - There is no preprocessor, no textual macros, and no procedural macros.

---

## 2. Encoding and Lexical Structure

### 2.1. Source File Encoding
* A source file **must** be valid **UTF-8** without a BOM (`EF BB BF` is lexer error `E0001`).
* ASCII control characters (`0x00..0x08`, `0x0B`, `0x0C`, `0x0E..0x1F`, `0x7F`) are forbidden outside string and character literals (`E0002`).
* Unicode bidirectional control characters (Trojan Source: `U+202A..U+202E`, `U+2066..U+2069`) are **forbidden everywhere**, including inside comments and strings (`E0003`); inside strings they may only be written using an escape sequence such as `\u{202E}`.
* Line endings: `\n` (LF) or `\r\n` (CRLF, normalized by the lexer to `\n`).

### 2.2. Whitespace and Comments
* Whitespace characters: `' '` (`0x20`), `'\t'` (`0x09`), `'\n'` (`0x0A`), `'\r'` (`0x0D`).
* Line comments: `// ...` until end of line.
* Documentation comments: `/// ...` (attached to the following declaration; parsed by the compiler for documentation and LSP) and `//! ...` (module-level documentation).
* Block comments: `/* ... */` — **support nesting** (`/* a /* b */ c */`), making it convenient to comment out blocks of code.

### 2.3. Identifiers
* Regular identifier: `[A-Za-z_][A-Za-z0-9_]*` (strictly ASCII for lexer speed and to prevent homoglyph attacks such as substituting Cyrillic `U+0430` for Latin `'a'`).
* The identifier `_` is a special discard token. It cannot be read.
* Names with a leading `_` (such as `_unused`) allow an unused parameter or variable without triggering `ZM-01`, but attempting to *read* a variable with a leading `_` is a **compile-time error** (`E0119`), preventing leftover `_foo` names once a binding starts being used.
* Style conventions (enforced by the standard formatter `eon fmt`):
  - Types, structs, pools, interfaces, errors: `PascalCase`
  - Functions, variables, fields, modules: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`

### 2.4. Reserved Keywords (54 Keywords + 2 Contextual)

```
// Declarations
module   import   pub      fn       sys      let      var      const
type     struct   linear   pool     enum     union    bitfield iface
impl     error    extern   comptime view

// Control flow
if       else     match    while    for      par      loop     break
continue return   defer    errdefer try      catch    orelse   unreachable

// Modifiers and contracts
mut      move     copy     where    ensures  in       as       is
from

// Literals and special names
true     false    null     void     never    self     Self     dyn
```

**Contextual keywords (`read`, `write`):** after `:` in `sys` and `@gpu` parameters they specify access mode; everywhere else they remain ordinary identifiers (e.g., `.read()`, `.write()` methods).

**Keywords permitted ONLY in `.hal.eon` (in `.eon` they trigger an immediate lexer/parser error `E0111`):**
```
raw      asm      volatile
```

### 2.5. Literals

#### Integer Literals
* Decimal: `0`, `42`, `1_000_000` (`_` separators are allowed between digits).
* Hexadecimal: `0xFF`, `0xDEAD_BEEF`.
* Binary: `0b1010_0101`.
* Octal: `0o755`.
* Without a suffix, an integer literal has the comptime type `comptime_int` (arbitrary precision at compile time). It implicitly adapts to a concrete integer type (`u8`, `i32`, `usize`, ...) **only if its value is guaranteed to fit in that type**. Otherwise, compilation fails with `E0201`.
* An explicit type suffix is allowed with an underscore or directly attached: `42_u32`, `0xFF_u8`, `-1_i64`.
* A leading `-` immediately before a literal participates in range checking as part of the literal (*constant folding* before type checking): `-9223372036854775808_i64` (`i64.MIN`) is valid even though `9223372036854775808` does not fit in `i64`.

#### Floating-Point Literals
* Decimal: `3.14`, `0.5`, `1.0e-9`, `2.5E+3`. (Forms like `.5` or `5.` are forbidden — digits are required on both sides of the decimal point to avoid conflicts with field access and `..`).
* Hexadecimal (exact IEEE-754): `0x1.fp+3`.
* Comptime type `comptime_float` — adapts to `f16`/`f32`/`f64` as long as it does not overflow to `Inf` (otherwise error `E0202`).

#### String and Character Literals
* Regular string (valid UTF-8, type `*str` with static lifetime): `"hello\nworld"`.
* Escape sequences: `\n`, `\r`, `\t`, `\\`, `\"`, `\'`, `\0`, `\xHH` (ASCII `00..7F` only inside strings, so UTF-8 validity cannot be broken!), `\u{H..HHHHHH}` (valid Unicode scalar value).
* Raw multiline string: begins with `\\` at the start of each line (as in Zig — no confusion with quote escaping or indentation):
  ```
  const SHADER_DOC: *str =
      \\First line without escaping "quotes"
      \\Second line
  ;
  ```
* Byte string literal (type `*[N]u8`): `b"\xDE\xAD\xBE\xEF"`.
* Character literal (type `char`, 32-bit Unicode scalar value): `'A'`, `'λ'`, `'\n'`, `'\u{1F680}'`. Lexer rule: `'` + exactly one scalar (or escape sequence) + closing `'` is a character literal; otherwise `'ident` is a loop label (`'outer: for …`, `break 'outer;`).
* Byte character literal (type `u8`, ASCII only): `b'a'`.

---

## 3. Operators and Precedence Table (Pratt Parser)

From highest precedence (`13`) to lowest (`1`). Associativity is specified explicitly.

| Precedence | Operators | Purpose | Associativity |
| :---: | :--- | :--- | :--- |
| **13** | `expr.field`, `expr.method(...)`, `expr.[T](...)`, `expr(...)`, `expr[i]`, `expr[lo..hi]`, `expr.*` | Postfix field access, call, explicit generic, indexing, slicing, dereference | Left |
| **12** | `-x`, `!x`, `~x`, `&x`, `&mut x`, `move x`, `copy x` | Unary prefix operators (`&`/`&mut` create a borrow) | Right |
| **11** | `*`, `/`, `%`, `*%`, `*\|`, `*?`, `/?`, `%?` | Multiplication, division, remainder (verified / explicit variants) | Left |
| **10** | `+`, `-`, `+%`, `-%`, `+\|`, `-\|`, `+?`, `-?` | Addition, subtraction (verified / explicit variants) | Left |
| **9** | `<<`, `>>`, `<<%` | Bitwise shifts | Left |
| **8** | `&` | Bitwise AND | Left |
| **7** | `^` | Bitwise XOR | Left |
| **6** | `\|` | Bitwise OR | Left |
| **5** | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is` | Comparison and variant test (chaining like `a < b < c` is forbidden -> error!); the right-hand side of `is` is a **pattern**, not an expression: `x is .Quit` | None |
| **4** | `&&` | Logical AND (short-circuiting) | Left |
| **3** | `\|\|` | Logical OR (short-circuiting) | Left |
| **2** | `orelse`, `catch`; prefix `try` | Unwrapping `?T` and handling `T ! E`. `try` may only appear at the start of an expression, and its operand is the **entire** binary expression to its right: `try acc +? x` == `try (acc +? x)`; nested `a + try f()` is forbidden — write `let t = try f(); a + t;` | Left |
| **1** | `=`, `+=`, `-=`, `*=`, `+% =`, `+\|=`, `&=`, `\|=`, `^=`, `<<=`, `>>=` | Assignment (a **statement**, not an expression — `a = b = 1` is forbidden!) | None |

> **Borrows and dereferencing (v0.2).** The token `*` in expressions is **exclusively** binary multiplication. Borrowing uses prefix `&x` / `&mut x`; dereferencing uses postfix `p.*`. Binary `&` (bitwise AND) and prefix `&` (borrow) are distinguished by position (infix vs. prefix), while reference types continue to be written `*T`, `*mut T`, `*[T]`, `*str`. Example: `*a = *a +% 1` from v0.1 is now `a.* = a.* +% 1_u32;`, and `f(*mut x)` is `f(&mut x)`. Field access and method calls auto-dereference: `p.field` and `p.method()` work directly on `p: *T`.

> **Precedence Clarity Rule (Zero-Mercy):** Mixing bitwise operators (`&`, `|`, `^`, `<<`, `>>`) with arithmetic (`+`, `-`, `*`) or comparison operators **without parentheses** is compile-time error `E0120`! For example, `a & 1 == 0` **will not compile**; you must write `(a & 1) == 0`. The same applies to mixing `&&` and `||` without parentheses (`E0121`).

---

## 4. Statements, Expressions, and Control Flow

### 4.1. Variable Bindings
* Function parameters are immutable. The `mut` modifier before a parameter name (`fn f(mut n: u32)`) makes the parameter a **mutable local copy** (passed by value; does not affect the caller). The `from` modifier (`view fn f(from a: *T, b: *U)`) designates the single borrow source of a `view fn`.
* `let x: u32 = 10;` — immutable binding (the type may be omitted if unambiguously inferred: `let x = 10_u32;`, whereas an untyped literal `let x = 10;` without context is error `E0203` — either a type annotation or a suffix is required!).
* `var y: u32 = 0;` — mutable binding. If a variable is declared with `var` but is never mutated, compilation fails with error `E0122` ("replace `var` with `let`").
* `const PAGE_SIZE: usize = 4096;` — compile-time constant.

### 4.2. Branching and Loops
* **Mandatory braces `{}`:** `if cond { ... } else { ... }`. Parentheses around the condition are neither required nor used.
* **Blocks and values.** A block `{ stmt; ...; final_expr }` ending with an expression **without** `;` is a *block expression*; its value is `final_expr`. Block expressions are used for `if`-expression branches, `match` arm bodies of the form `=> { ... }`, closure bodies, and `comptime { ... }`.
* **Function returns use explicit `return expr;` only.** A trailing expression without `;` in a `fn` body is error `E0124` (eliminating confusion over adding/omitting `;`). There is no `yield` keyword in the language.
* **`if` as an expression:** allowed only with an `else` branch where both branches have the same type: `let m = if a < b { a } else { b };`
* **Statement vs. expression:** a construct beginning with `if` / `match` / `while` / `for` / `par` / `loop` / `raw` at the start of a statement is always a statement. To use its value, it must appear in an expression position (the right-hand side of `let`/`=`, a call argument, `return`) or inside parentheses.
* **`break` with a value:** `break 'label value;` is allowed only for `loop` (where `value` becomes the result of the loop). For `while` and `for`, only `break;` / `break 'label;` is permitted.
* **Unwrapping optionals (`if let`):**
  ```
  if let parent_id = widget.parent {
      // parent_id has type GenId[Widget], not ?GenId[Widget]
  } else {
      // ...
  }
  ```
* **`match`:** must cover all variants (`ZM-04`). Supports literals, ranges `0..=255`, `enum`/`union`/`error` variants, tuples, and struct destructuring. A wildcard arm `_ => ...` is forbidden if all variants are already covered (dead branch -> `ZM-14`).
* **Loops:**
  - `while cond { ... }` (optionally with an explicit loop invariant via `where`: `while i < n where i <= n { ... }`).
  - `for item in slice { ... }` / `for i in 0..n { ... }` / `for id in pool.indices() { ... }`.
  - `loop { ... }` — infinite loop, exited via `break`.
  - Loop labels: `'outer: for i in 0..h { for j in 0..w { break 'outer; } }`.
* **`defer` and `errdefer`:**
  - `defer { ... }` or `defer close(fd);` — executes on **every** exit from the current lexical scope (`return`, `break`, `continue`, `try`).
  - `errdefer |err| { ... }` or `errdefer free(page);` — executes **only** when exiting the function with an error (`try` or `return Err`).
  - Execution order: strictly LIFO (reverse order of declaration).
  - Inside `defer` and `errdefer`, `return`, `break`, `continue`, and `try` are forbidden (`E0123`).

---

## 5. Modules, Imports, and Visibility

### 5.1. One File = One Module (or Directory)
* Every `.eon` file **must** begin with a `module` header:
  ```
  module ui.layout;
  ```
  For a HAL module (`.hal.eon`):
  ```
  module kernel.arch.x86_64.apic @hal;
  ```
* The module name must strictly match the file path relative to the package root (`src/ui/layout.eon` -> `module ui.layout;`). A mismatch is compile-time error `E0130`.

### 5.2. Imports
* Wildcard imports are forbidden (`import foo.*;` does not exist in the grammar!).
* Import forms:
  ```
  import core.mem;                       // accessed as mem.Arena
  import ui.core as uic;                 // accessed as uic.Widget
  import kernel.pmm.{PhysPage, PhysAddr}; // explicit symbol list
  ```
* **No cyclic dependencies between modules:** the import graph must be acyclic (a DAG). A cycle `A -> B -> A` is compile-time error `E0131`. This guarantees parallel module parsing and fast incremental `.emi` builds.

### 5.3. Visibility
Eon has exactly **two** visibility levels — with no complex `pub(in super::foo)` hierarchies:
* **No modifier (default):** private, visible only within the current module.
* **`pub`:** public, exported in the module's binary `.emi` interface.
* For structs that uphold invariants, you may declare a public type with private fields (`pub struct Ring { head: u32; tail: u32; }`) — struct literals `.{ ... }` can then only be constructed inside the defining module, while external callers must go through `pub fn` constructors.

---

## 6. Specification Tests (Must Compile / Must Fail)

### Must Compile
```
module example.syntax;

pub fn clamp_byte(x: u32) -> u8 {
    if x > 255_u32 {
        return 255_u8;
    }
    // Here x <= 255 (from the condition above). `trunc` is an explicit operation
    // and requires no proof; at this point it simply loses no bits.
    return x.trunc.[u8]();
}
```

### Must Fail Compilation
```
module example.syntax_fail;

pub fn bad_examples(a: u32, b: u32) -> u32 {
    let unused = 1_u32;          // ERROR ZM-01 (E0101): unused variable
    let c = a & 1_u32 == 0_u32;  // ERROR E0120: missing parentheses around bitwise & and ==
    if a > b return a;           // ERROR ZM-17 (E0117): missing {}
    let a = 5_u32;               // ERROR ZM-03 (E0103): shadowing name `a`
    return a +% b;
}
```
