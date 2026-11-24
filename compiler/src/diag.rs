use std::fmt;
use std::fmt::Write;

use crate::span::{FileId, SourceMap, Span};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorCode(pub u16);

impl ErrorCode {
    pub const E0001: Self = Self(1); // Invalid UTF-8 / BOM
    pub const E0002: Self = Self(2); // Forbidden ASCII control char
    pub const E0003: Self = Self(3); // Forbidden Unicode bidi control (Trojan Source)
    pub const E0004: Self = Self(4); // General lexical error
    pub const E0005: Self = Self(5); // Syntax / parse error

    pub const E0101: Self = Self(101); // ZM-01: Unused local, parameter, import, or private fn
    pub const E0102: Self = Self(102); // ZM-02: Unused non-void return value / discarded error union
    pub const E0103: Self = Self(103); // ZM-03: Variable name shadowing
    pub const E0104: Self = Self(104); // ZM-04: Non-exhaustive match
    pub const E0105: Self = Self(105); // ZM-05: Implicit numeric coercion
    pub const E0107: Self = Self(107); // ZM-07: Unproven bounds check
    pub const E0108: Self = Self(108); // ZM-08: Unproven integer overflow / div-by-zero
    pub const E0109: Self = Self(109); // ZM-09: Linear value dropped without consumption
    pub const E0110: Self = Self(110); // ZM-10: Module-level mutable `var`
    pub const E0111: Self = Self(111); // ZM-11: HAL-only construct in .eon (or missing @hal in .hal.eon)
    pub const E0112: Self = Self(112); // ZM-12: Effect contract violation (@pure/@noalloc/@noblock/@interrupt/@gpu)
    pub const E0113: Self = Self(113); // ZM-13: Disallowed recursion (missing @recursive or incompatible with @stack/@interrupt/@gpu)
    pub const E0114: Self = Self(114); // ZM-14: Unreachable / redundant match arm or dead branch
    pub const E0115: Self = Self(115); // ZM-15: `unreachable;` statement is statically reachable
    pub const E0116: Self = Self(116); // ZM-16: Uninitialized struct field
    pub const E0117: Self = Self(117); // ZM-17: Missing braces around control-flow body
    pub const E0118: Self = Self(118); // ZM-18: Exclusivity violation (mutable borrow aliasing / conflict)
    pub const E0119: Self = Self(119); // Reading a `_`-prefixed variable or parameter
    pub const E0120: Self = Self(120); // Mixing bitwise and arithmetic/comparison without parens
    pub const E0121: Self = Self(121); // Mixing && and || without parens
    pub const E0122: Self = Self(122); // `var` binding is never mutated
    pub const E0124: Self = Self(124); // Trailing expression without ';' in fn/sys body
    pub const E0125: Self = Self(125); // Chained comparisons or chained/expression assignment
    pub const E0126: Self = Self(126); // `return` in `where` clause (only allowed in `ensures`)
    pub const E0131: Self = Self(131); // Cyclic module import dependency (DAG violation)
    pub const E0132: Self = Self(132); // Unresolved name, module, or duplicate definition
    pub const E0133: Self = Self(133); // Private symbol or field access violation

    pub const E0201: Self = Self(201); // Integer literal out of range for target type
    pub const E0202: Self = Self(202); // Float literal overflows target type
    pub const E0203: Self = Self(203); // Untyped integer/float literal without context
    pub const E0210: Self = Self(210); // `type` value used at runtime
    pub const E0211: Self = Self(211); // Incompatible address types (PhysAddr/VirtAddr/DmaAddr/usize)
    pub const E0215: Self = Self(215); // Borrowing a misaligned field of a `@packed` struct
    pub const E0220: Self = Self(220); // Invalid `bitfield` bit range or overlap
    pub const E0230: Self = Self(230); // General type error (mismatch, arity, invalid cast, missing return)

    pub const E0301: Self = Self(301); // ZM-06: Use of moved value (> 16B)
    pub const E0310: Self = Self(310); // Second-class reference or closure stored in aggregate / escaping
    pub const E0311: Self = Self(311); // Borrowing value stored in aggregate / escaping frame
    pub const E0320: Self = Self(320); // Attempt to `copy` a `linear` value
    pub const E0321: Self = Self(321); // Double consumption / use after move of `linear` value
    pub const E0322: Self = Self(322); // Struct/union containing `linear` field must be `linear`
    pub const E0323: Self = Self(323); // Linear state mismatch across branches
    pub const E0324: Self = Self(324); // Consuming outer `linear` variable inside loop
    pub const E0325: Self = Self(325); // `errdefer` conflict with normal-path consumption before `try`
    pub const E0326: Self = Self(326); // Linear type requires `LinearBuf` instead of `Buf`
    pub const E0330: Self = Self(330); // Brand mismatch: `Id[p1]` used with incompatible pool
    pub const E0331: Self = Self(331); // Use of invalidated `Id[p]` after `reset()`
    pub const E0332: Self = Self(332); // Branded `Id[p]` stored in struct/union or foreign pool

    pub const E0501: Self = Self(501); // Generic / interface constraint / orphan rule error
    pub const E0510: Self = Self(510); // Implementation exceeds interface @stack(N) budget
    pub const E0511: Self = Self(511); // Implementation requires stronger `where` than interface
    pub const E0512: Self = Self(512); // Interface is not dyn-compatible (object-safe)
    pub const E0520: Self = Self(520); // Comptime VM execution error (fuel budget exceeded, overflow, OOB, @compile_error)
    pub const E0601: Self = Self(601); // Conflicting `sys` column access in Schedule without ordering
    pub const E0602: Self = Self(602); // Mutating outer `var` or capturing `*mut` inside `par for`
    pub const E0603: Self = Self(603); // Non-deterministic float reduction in `par_reduce`
    pub const E0610: Self = Self(610); // Type is not hardware lock-free for `Atomic[T]`
    pub const E0701: Self = Self(701); // Function exceeds static `@stack(N)` budget
    pub const E0805: Self = Self(805); // MmioReg access violation (read of WriteOnly or write to ReadOnly)
    pub const E0810: Self = Self(810); // Invalid Eon-specific type at `@extern("C")` FFI boundary
    pub const E0820: Self = Self(820); // Struct passed to `@gpu` function must have `@layout(std430)`
    pub const E0821: Self = Self(821); // Disallowed CPU-only operation or type inside `@gpu` function
    pub const E0822: Self = Self(822); // Non-uniform control flow reaching `@gpu_workgroup_barrier()`
    pub const E0823: Self = Self(823); // `read`/`write` parameter qualifier outside `sys`/`@gpu`
    pub const E0901: Self = Self(901); // ABI / `.emi` interface / native linker error
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:04}", self.0)
    }
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub code: ErrorCode,
    pub message: String,
    pub span: Span,
    pub labels: Vec<(Span, String)>,
    pub helps: Vec<String>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(code: ErrorCode, span: Span, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            labels: Vec::new(),
            helps: Vec::new(),
            notes: Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct DiagBag {
    diags: Vec<Diagnostic>,
}

impl DiagBag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        self.diags.push(diag);
    }

    pub fn error(&mut self, code: ErrorCode, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::error(code, span, msg));
    }

    pub fn has_errors(&self) -> bool {
        !self.diags.is_empty()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diags
    }

    pub fn truncate(&mut self, len: usize) {
        self.diags.truncate(len);
    }

    pub fn render_all(&self, sm: &SourceMap) -> String {
        let mut out = String::new();
        for (i, d) in self.diags.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            render_one(d, sm, &mut out);
        }
        out
    }
}

fn render_one(d: &Diagnostic, sm: &SourceMap, out: &mut String) {
    let _ = writeln!(out, "error[{}]: {}", d.code, d.message);
    if d.span.file != FileId::DUMMY {
        let file = sm.file(d.span.file);
        let (line, col) = file.line_col(d.span.start);
        let line_str = line.to_string();
        let pad = line_str.len();
        let _ = writeln!(out, "{:>pad$}--> {}:{}:{}", "", file.name, line, col, pad = pad);
        let src_line = file.line_text(line);
        let _ = writeln!(out, "{:>pad$} |", "", pad = pad);
        let _ = writeln!(out, "{} | {}", line_str, src_line);

        let (_, end_col) = if d.span.end > d.span.start {
            let (end_line, ec) = file.line_col(d.span.end);
            if end_line == line {
                (end_line, ec)
            } else {
                (line, (src_line.chars().count() as u32) + 1)
            }
        } else {
            (line, col + 1)
        };
        let caret_start = col.saturating_sub(1) as usize;
        let caret_len = end_col.saturating_sub(col).max(1) as usize;

        let mut spaces = String::new();
        for ch in src_line.chars().take(caret_start) {
            spaces.push(if ch == '\t' { '\t' } else { ' ' });
        }
        let carets = "^".repeat(caret_len);
        let _ = writeln!(out, "{:>pad$} | {}{}", "", spaces, carets, pad = pad);

        for (lspan, lmsg) in &d.labels {
            if lspan.file != FileId::DUMMY {
                let lf = sm.file(lspan.file);
                let (ll, lc) = lf.line_col(lspan.start);
                let _ = writeln!(
                    out,
                    "{:>pad$} = note: {}:{} — {}",
                    "",
                    ll,
                    lc,
                    lmsg,
                    pad = pad
                );
            }
        }
        for help in &d.helps {
            let _ = writeln!(out, "{:>pad$} = help: {}", "", help, pad = pad);
        }
        for note in &d.notes {
            let _ = writeln!(out, "{:>pad$} = note: {}", "", note, pad = pad);
        }
    } else {
        for help in &d.helps {
            let _ = writeln!(out, " = help: {}", help);
        }
        for note in &d.notes {
            let _ = writeln!(out, " = note: {}", note);
        }
    }
}
