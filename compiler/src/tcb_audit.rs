//! Trusted Computing Base (TCB) auditor and FIPS 180-4 SHA-256 implementation.
//!
//! Scans `@hal` / `.hal.eon` compilation units to report exact SHA-256 digests,
//! distinct `raw { ... }` source line counts, inline `asm(...)` sites, and
//! `extern "C"` imports, and enforces the `allow_hal` whitelist from `build.eon`.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::ast::{
    AsmOperandKind, Attr, Block, BorrowTarget, CatchBody, CompilationUnit, Decl, DeclKind,
    DeferBody, ElseBranch, Expr, ExprKind, Field, GenericArg, GenericArgKind, GenericParam,
    IfCond, IfExpr, JumpAction, MatchArm, MatchArmBody, Param, Pat, PatKind, Stmt, StmtKind,
    SysParam, TypeExpr, TypeKind,
};
use crate::span::{FileId, FileProfile, Interner, SourceMap, Span};

const SHA256_H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
    0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
    0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
    0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
    0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    let (word_chunks, _) = block.as_chunks::<4>();
    for (dst, chunk) in w[..16].iter_mut().zip(word_chunks.iter()) {
        *dst = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (&k_i, &w_i) in SHA256_K.iter().zip(w.iter()) {
        let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(k_i)
            .wrapping_add(w_i);
        let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = big_s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Computes the FIPS 180-4 SHA-256 digest of `data`.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let (blocks, rem) = data.as_chunks::<64>();
    for block in blocks {
        compress_block(&mut state, block);
    }

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut tail = [0u8; 64];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;

    if rem.len() >= 56 {
        compress_block(&mut state, &tail);
        tail = [0u8; 64];
    }
    tail[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress_block(&mut state, &tail);

    let mut out = [0u8; 32];
    let (out_words, _) = out.as_chunks_mut::<4>();
    for (dst, word) in out_words.iter_mut().zip(state.iter()) {
        *dst = word.to_be_bytes();
    }
    out
}

/// Computes the lowercase hexadecimal FIPS 180-4 SHA-256 digest of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256_bytes(data);
    let mut hex = String::with_capacity(64);
    for b in digest {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// Audit summary for a single `.hal.eon` / `@hal` source file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcbFileReport {
    pub path: String,
    pub module_name: String,
    pub sha256: String,
    pub raw_lines: usize,
    pub asm_sites: usize,
    pub extern_c_imports: Vec<String>,
}

/// Aggregate TCB audit report across all `@hal` compilation units in a build.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TcbAuditReport {
    pub files: Vec<TcbFileReport>,
    pub total_raw_lines: usize,
    pub total_asm_sites: usize,
    pub total_extern_c: usize,
}

impl TcbAuditReport {
    /// Formats a structured human-readable report for `--audit-tcb`.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== EON TCB AUDIT REPORT ===");
        let _ = writeln!(out, "hal_files: {}", self.files.len());
        for f in &self.files {
            let _ = writeln!(out, "  - module: {}", f.module_name);
            let _ = writeln!(out, "    path: {}", f.path);
            let _ = writeln!(out, "    sha256: {}", f.sha256);
            let _ = writeln!(out, "    raw_lines: {}", f.raw_lines);
            let _ = writeln!(out, "    asm_sites: {}", f.asm_sites);
            if f.extern_c_imports.is_empty() {
                let _ = writeln!(out, "    extern_c: none");
            } else {
                let _ = writeln!(out, "    extern_c: [{}]", f.extern_c_imports.join(", "));
            }
        }
        let _ = writeln!(out, "totals:");
        let _ = writeln!(out, "  files: {}", self.files.len());
        let _ = writeln!(out, "  raw_lines: {}", self.total_raw_lines);
        let _ = writeln!(out, "  asm_sites: {}", self.total_asm_sites);
        let _ = writeln!(out, "  extern_c: {}", self.total_extern_c);
        out
    }

    /// Verifies that every audited `@hal` module belongs to either `std_prefixes`
    /// or the package's explicit `allowed` (`allow_hal`) whitelist.
    pub fn check_hal_whitelist(
        &self,
        allowed: &BTreeSet<String>,
        std_prefixes: &[&str],
    ) -> Result<(), String> {
        let mut violations = Vec::new();
        for f in &self.files {
            if is_hal_module_whitelisted(&f.module_name, &f.path, allowed, std_prefixes) {
                continue;
            }
            let display_mod = if f.module_name.is_empty() {
                "<unnamed>"
            } else {
                &f.module_name
            };
            violations.push(format!(
                "disallowed @hal module `{}` ({}); add `{}` to `allow_hal` in build.eon to trust it",
                display_mod, f.path, display_mod
            ));
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }
}

fn matches_module_prefix(module_name: &str, prefix: &str) -> bool {
    if module_name.is_empty() || prefix.is_empty() {
        return false;
    }
    let base = prefix.trim_end_matches('.');
    if base.is_empty() {
        return false;
    }
    module_name == base
        || (module_name.starts_with(base) && module_name.as_bytes().get(base.len()) == Some(&b'.'))
}

fn is_hal_module_whitelisted(
    module_name: &str,
    path: &str,
    allowed: &BTreeSet<String>,
    std_prefixes: &[&str],
) -> bool {
    if std_prefixes
        .iter()
        .any(|p| matches_module_prefix(module_name, p))
    {
        return true;
    }
    if (!module_name.is_empty() && allowed.contains(module_name)) || allowed.contains(path) {
        return true;
    }
    for entry in allowed {
        if let Some(prefix) = entry.strip_suffix(".*")
            && matches_module_prefix(module_name, prefix)
        {
            return true;
        }
    }
    false
}

/// Result of scanning a single `CompilationUnit` AST for TCB / HAL constructs.
#[derive(Clone, Debug, Default)]
pub(crate) struct TcbConstructScan {
    pub raw_spans: Vec<Span>,
    pub asm_sites: usize,
    pub extern_c_imports: Vec<String>,
    pub has_hal_attr: bool,
}

/// Recursively walks `unit` to collect `raw` block spans, `asm` sites,
/// `extern` declarations, and any `@hal` attributes.
pub(crate) fn scan_unit_constructs(
    unit: &CompilationUnit,
    interner: &Interner,
) -> TcbConstructScan {
    let mut scanner = AstTcbScanner {
        interner,
        scan: TcbConstructScan::default(),
    };
    for decl in &unit.decls {
        scanner.visit_decl(decl);
    }
    scanner.scan.extern_c_imports.sort();
    scanner.scan
}

struct AstTcbScanner<'a> {
    interner: &'a Interner,
    scan: TcbConstructScan,
}

impl AstTcbScanner<'_> {
    fn visit_attr(&mut self, attr: &Attr) {
        if self.interner.get(attr.name.sym) == "hal" {
            self.scan.has_hal_attr = true;
        }
        for arg in &attr.args {
            self.visit_expr(&arg.value);
        }
    }

    fn visit_decl(&mut self, decl: &Decl) {
        for attr in &decl.attrs {
            self.visit_attr(attr);
        }
        match &decl.kind {
            DeclKind::Fn(f) => {
                for attr in &f.effects {
                    self.visit_attr(attr);
                }
                for gp in &f.generics {
                    self.visit_generic_param(gp);
                }
                for p in &f.params {
                    self.visit_param(p);
                }
                if let Some(rt) = &f.ret_type {
                    self.visit_type_expr(rt);
                }
                for e in &f.where_clauses {
                    self.visit_expr(e);
                }
                for e in &f.ensures_clauses {
                    self.visit_expr(e);
                }
                self.visit_block(&f.body);
            }
            DeclKind::Sys(s) => {
                for attr in &s.effects {
                    self.visit_attr(attr);
                }
                for gp in &s.generics {
                    self.visit_generic_param(gp);
                }
                for p in &s.params {
                    self.visit_sys_param(p);
                }
                if let Some(rt) = &s.ret_type {
                    self.visit_type_expr(rt);
                }
                for e in &s.where_clauses {
                    self.visit_expr(e);
                }
                for e in &s.ensures_clauses {
                    self.visit_expr(e);
                }
                self.visit_block(&s.body);
            }
            DeclKind::Struct(s) => {
                for gp in &s.generics {
                    self.visit_generic_param(gp);
                }
                for e in &s.where_clauses {
                    self.visit_expr(e);
                }
                for f in &s.fields {
                    self.visit_field(f);
                }
            }
            DeclKind::Pool(p) => {
                for gp in &p.generics {
                    self.visit_generic_param(gp);
                }
                for e in &p.where_clauses {
                    self.visit_expr(e);
                }
                for f in &p.fields {
                    self.visit_field(f);
                }
            }
            DeclKind::Enum(e) => {
                for v in &e.variants {
                    if let Some(val) = &v.value {
                        self.visit_expr(val);
                    }
                }
            }
            DeclKind::Union(u) => {
                for gp in &u.generics {
                    self.visit_generic_param(gp);
                }
                for e in &u.where_clauses {
                    self.visit_expr(e);
                }
                for v in &u.variants {
                    if let Some(ty) = &v.ty {
                        self.visit_type_expr(ty);
                    }
                }
            }
            DeclKind::Bitfield(b) => {
                for m in &b.members {
                    if let Some((lo, hi)) = &m.bits {
                        self.visit_expr(lo);
                        self.visit_expr(hi);
                    }
                }
            }
            DeclKind::Iface(i) => {
                for gp in &i.generics {
                    self.visit_generic_param(gp);
                }
                for m in &i.methods {
                    for attr in &m.attrs {
                        self.visit_attr(attr);
                    }
                    for attr in &m.effects {
                        self.visit_attr(attr);
                    }
                    for gp in &m.generics {
                        self.visit_generic_param(gp);
                    }
                    for p in &m.params {
                        self.visit_param(p);
                    }
                    if let Some(rt) = &m.ret_type {
                        self.visit_type_expr(rt);
                    }
                    for e in &m.where_clauses {
                        self.visit_expr(e);
                    }
                    for e in &m.ensures_clauses {
                        self.visit_expr(e);
                    }
                }
            }
            DeclKind::Impl(i) => {
                for gp in &i.generics {
                    self.visit_generic_param(gp);
                }
                if let Some(it) = &i.iface_type {
                    self.visit_type_expr(it);
                }
                self.visit_type_expr(&i.target_type);
                for e in &i.where_clauses {
                    self.visit_expr(e);
                }
                for sub in &i.decls {
                    self.visit_decl(sub);
                }
            }
            DeclKind::Error(_) => {}
            DeclKind::TypeAlias(t) => {
                for gp in &t.generics {
                    self.visit_generic_param(gp);
                }
                for e in &t.where_clauses {
                    self.visit_expr(e);
                }
                self.visit_type_expr(&t.ty);
            }
            DeclKind::Const(c) => {
                self.visit_type_expr(&c.ty);
                self.visit_expr(&c.value);
            }
            DeclKind::ExternFn(e) => {
                self.scan
                    .extern_c_imports
                    .push(self.interner.get(e.name.sym).to_owned());
                for p in &e.params {
                    self.visit_param(p);
                }
                if let Some(rt) = &e.ret_type {
                    self.visit_type_expr(rt);
                }
            }
            DeclKind::ExternStruct(e) | DeclKind::ExternUnion(e) => {
                self.scan
                    .extern_c_imports
                    .push(self.interner.get(e.name.sym).to_owned());
                for f in &e.fields {
                    self.visit_field(f);
                }
            }
        }
    }

    fn visit_generic_param(&mut self, gp: &GenericParam) {
        if let Some(ct) = &gp.comptime_ty {
            self.visit_type_expr(ct);
        }
        for b in &gp.bounds {
            self.visit_type_expr(b);
        }
    }

    fn visit_generic_arg(&mut self, ga: &GenericArg) {
        match &ga.kind {
            GenericArgKind::Type(t) => self.visit_type_expr(t),
            GenericArgKind::Expr(e) => self.visit_expr(e),
            GenericArgKind::TypeOrConst(_) => {}
        }
    }

    fn visit_param(&mut self, p: &Param) {
        for attr in &p.attrs {
            self.visit_attr(attr);
        }
        self.visit_type_expr(&p.ty);
    }

    fn visit_sys_param(&mut self, p: &SysParam) {
        for attr in &p.attrs {
            self.visit_attr(attr);
        }
        self.visit_type_expr(&p.ty);
    }

    fn visit_field(&mut self, f: &Field) {
        for attr in &f.attrs {
            self.visit_attr(attr);
        }
        self.visit_type_expr(&f.ty);
        if let Some(def) = &f.default {
            self.visit_expr(def);
        }
    }

    fn visit_type_expr(&mut self, te: &TypeExpr) {
        match &te.kind {
            TypeKind::ErrorUnion { ok, errors } => {
                if let Some(ok_ty) = ok {
                    self.visit_type_expr(ok_ty);
                }
                for err in errors {
                    self.visit_type_expr(err);
                }
            }
            TypeKind::Optional(inner) => self.visit_type_expr(inner),
            TypeKind::Borrow { target, .. } => match target {
                BorrowTarget::Single(t) | BorrowTarget::Slice(t) => self.visit_type_expr(t),
                BorrowTarget::Str => {}
            },
            TypeKind::Array { len, elem } => {
                self.visit_expr(len);
                self.visit_type_expr(elem);
            }
            TypeKind::Dyn(tp) | TypeKind::Path(tp) => {
                for ga in &tp.generics {
                    self.visit_generic_arg(ga);
                }
            }
            TypeKind::FnPtr {
                params,
                ret,
                effects,
            } => {
                for p in params {
                    self.visit_type_expr(p);
                }
                if let Some(r) = ret {
                    self.visit_type_expr(r);
                }
                for a in effects {
                    self.visit_attr(a);
                }
            }
            TypeKind::InlineStruct(fields) => {
                for f in fields {
                    self.visit_field(f);
                }
            }
            TypeKind::Tuple(elems) => {
                for t in elems {
                    self.visit_type_expr(t);
                }
            }
            TypeKind::Void
            | TypeKind::Never
            | TypeKind::TypeType
            | TypeKind::SelfType
            | TypeKind::Inferred
            | TypeKind::Error => {}
        }
    }

    fn visit_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
        if let Some(tail) = &block.tail_expr {
            self.visit_expr(tail);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let {
                attrs,
                pat,
                ty,
                init,
            } => {
                for a in attrs {
                    self.visit_attr(a);
                }
                self.visit_pat(pat);
                if let Some(t) = ty {
                    self.visit_type_expr(t);
                }
                self.visit_expr(init);
            }
            StmtKind::Var {
                attrs, ty, init, ..
            } => {
                for a in attrs {
                    self.visit_attr(a);
                }
                if let Some(t) = ty {
                    self.visit_type_expr(t);
                }
                self.visit_expr(init);
            }
            StmtKind::Const(c) => {
                self.visit_type_expr(&c.ty);
                self.visit_expr(&c.value);
            }
            StmtKind::Defer(body) | StmtKind::Errdefer { body, .. } => {
                self.visit_defer_body(body);
            }
            StmtKind::Jump(j) => self.visit_jump(j),
            StmtKind::If { data, .. } => self.visit_if(data),
            StmtKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_match_arm(arm);
                }
            }
            StmtKind::While {
                cond,
                where_clauses,
                body,
                ..
            } => {
                self.visit_expr(cond);
                for e in where_clauses {
                    self.visit_expr(e);
                }
                self.visit_block(body);
            }
            StmtKind::For {
                pat,
                iter_start,
                iter_end,
                body,
                ..
            }
            | StmtKind::ParFor {
                pat,
                iter_start,
                iter_end,
                body,
            } => {
                self.visit_pat(pat);
                self.visit_expr(iter_start);
                if let Some((_, end)) = iter_end {
                    self.visit_expr(end);
                }
                self.visit_block(body);
            }
            StmtKind::Loop { body, .. } | StmtKind::ComptimeBlock(body) => {
                self.visit_block(body);
            }
            StmtKind::RawBlock(body) => {
                self.scan.raw_spans.push(body.span);
                self.visit_block(body);
            }
            StmtKind::Assign { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            StmtKind::Discard(e) | StmtKind::Expr(e) => self.visit_expr(e),
        }
    }

    fn visit_defer_body(&mut self, body: &DeferBody) {
        match body {
            DeferBody::Block(b) => self.visit_block(b),
            DeferBody::Expr(e) => self.visit_expr(e),
        }
    }

    fn visit_jump(&mut self, jump: &JumpAction) {
        match jump {
            JumpAction::Return(Some(e)) | JumpAction::Break { value: Some(e), .. } => {
                self.visit_expr(e);
            }
            JumpAction::Return(None)
            | JumpAction::Break { value: None, .. }
            | JumpAction::Continue { .. }
            | JumpAction::Unreachable => {}
        }
    }

    fn visit_if(&mut self, if_expr: &IfExpr) {
        match &if_expr.cond {
            IfCond::Expr(e) => self.visit_expr(e),
            IfCond::Let { pat, expr } => {
                self.visit_pat(pat);
                self.visit_expr(expr);
            }
        }
        self.visit_block(&if_expr.then_block);
        if let Some(eb) = &if_expr.else_branch {
            match eb {
                ElseBranch::Block(b) => self.visit_block(b),
                ElseBranch::If { data, .. } => self.visit_if(data),
            }
        }
    }

    fn visit_match_arm(&mut self, arm: &MatchArm) {
        self.visit_pat(&arm.pat);
        if let Some(g) = &arm.guard {
            self.visit_expr(g);
        }
        match &arm.body {
            MatchArmBody::Block(b) => self.visit_block(b),
            MatchArmBody::Expr(e) => self.visit_expr(e),
            MatchArmBody::Jump(j) => self.visit_jump(j),
        }
    }

    fn visit_pat(&mut self, pat: &Pat) {
        match &pat.kind {
            PatKind::Variant { args, .. } | PatKind::Tuple(args) => {
                for p in args {
                    self.visit_pat(p);
                }
            }
            PatKind::Struct { path, fields } => {
                if let Some(tp) = path {
                    for ga in &tp.generics {
                        self.visit_generic_arg(ga);
                    }
                }
                for f in fields {
                    if let Some(p) = &f.pat {
                        self.visit_pat(p);
                    }
                }
            }
            PatKind::Wildcard
            | PatKind::Ident(_)
            | PatKind::IntLit { .. }
            | PatKind::CharLit { .. }
            | PatKind::BoolLit(_)
            | PatKind::Null
            | PatKind::Error => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::AnonStructLit(fields) => {
                for f in fields {
                    self.visit_expr(&f.value);
                }
            }
            ExprKind::NamedStructLit { target, fields } => {
                self.visit_expr(target);
                for f in fields {
                    self.visit_expr(&f.value);
                }
            }
            ExprKind::ArrayLit(es) | ExprKind::Tuple(es) => {
                for e in es {
                    self.visit_expr(e);
                }
            }
            ExprKind::ArrayRepeat { val, count } => {
                self.visit_expr(val);
                self.visit_expr(count);
            }
            ExprKind::Paren(e)
            | ExprKind::ComptimeExpr(e)
            | ExprKind::Deref(e)
            | ExprKind::Unary { expr: e, .. } => self.visit_expr(e),
            ExprKind::Closure { params, body } => {
                for p in params {
                    self.visit_param(p);
                }
                self.visit_block(body);
            }
            ExprKind::If { data, .. } => self.visit_if(data),
            ExprKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_match_arm(arm);
                }
            }
            ExprKind::Loop { body, .. } | ExprKind::ComptimeBlock(body) => {
                self.visit_block(body);
            }
            ExprKind::Builtin { args, .. } => {
                for a in args {
                    self.visit_generic_arg(a);
                }
            }
            ExprKind::Asm { operands, .. } => {
                self.scan.asm_sites += 1;
                for op in operands {
                    match &op.kind {
                        AsmOperandKind::Named {
                            constraint, arg, ..
                        } => {
                            self.visit_expr(constraint);
                            if let Some(a) = arg {
                                self.visit_expr(a);
                            }
                        }
                        AsmOperandKind::Expr(e) => self.visit_expr(e),
                    }
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::Is { expr: inner, pat } => {
                self.visit_expr(inner);
                self.visit_pat(pat);
            }
            ExprKind::Orelse { lhs, rhs } | ExprKind::Catch { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                match rhs {
                    CatchBody::Block(b) => self.visit_block(b),
                    CatchBody::Expr(e) => self.visit_expr(e),
                    CatchBody::Jump(j) => self.visit_jump(j),
                }
            }
            ExprKind::Field { base, .. } => self.visit_expr(base),
            ExprKind::GenericApply { base, args } => {
                self.visit_expr(base);
                for a in args {
                    self.visit_generic_arg(a);
                }
            }
            ExprKind::Call { callee, args } => {
                self.visit_expr(callee);
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::Index { base, index } => {
                self.visit_expr(base);
                self.visit_expr(index);
            }
            ExprKind::Slice { base, lo, hi } => {
                self.visit_expr(base);
                if let Some(l) = lo {
                    self.visit_expr(l);
                }
                if let Some(h) = hi {
                    self.visit_expr(h);
                }
            }
            ExprKind::IntLit { .. }
            | ExprKind::FloatLit { .. }
            | ExprKind::StringLit(_)
            | ExprKind::RawStringLit(_)
            | ExprKind::ByteStringLit(_)
            | ExprKind::CharLit(_)
            | ExprKind::ByteCharLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::Null
            | ExprKind::SelfVal
            | ExprKind::ReturnVal
            | ExprKind::Ident(_)
            | ExprKind::EnumLit(_)
            | ExprKind::Error => {}
        }
    }
}

/// Audits all `@hal` / `.hal.eon` compilation units and produces a [`TcbAuditReport`].
pub fn audit_tcb(units: &[CompilationUnit], sm: &SourceMap, interner: &Interner) -> TcbAuditReport {
    let mut report = TcbAuditReport::default();

    for unit in units {
        let sf = sm.file(unit.file);
        if sf.profile != FileProfile::Hal && !unit.is_hal {
            continue;
        }

        let module_name = unit
            .module_path
            .iter()
            .map(|seg| interner.get(seg.sym))
            .collect::<Vec<_>>()
            .join(".");
        let sha256 = sha256_hex(sf.text().as_bytes());
        let scan = scan_unit_constructs(unit, interner);

        let mut raw_line_set = BTreeSet::new();
        for span in &scan.raw_spans {
            let target_sf = if span.file != FileId::DUMMY && (span.file.0 as usize) < sm.len() {
                sm.file(span.file)
            } else {
                sf
            };
            let s = span.start.min(span.end);
            let e = span.start.max(span.end);
            let l0 = target_sf.line_col(s).0;
            let l1 = target_sf.line_col(e).0;
            for line in l0.min(l1)..=l0.max(l1) {
                raw_line_set.insert(line);
            }
        }

        let raw_lines = raw_line_set.len();
        let asm_sites = scan.asm_sites;
        let extern_c_imports = scan.extern_c_imports;

        report.total_raw_lines += raw_lines;
        report.total_asm_sites += asm_sites;
        report.total_extern_c += extern_c_imports.len();
        report.files.push(TcbFileReport {
            path: sf.name.clone(),
            module_name,
            sha256,
            raw_lines,
            asm_sites,
            extern_c_imports,
        });
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::DiagBag;
    use crate::lexer::lex_file;
    use crate::parser::parse_file;

    fn parse_unit(
        path: &str,
        src: &str,
        sm: &mut SourceMap,
        interner: &mut Interner,
    ) -> CompilationUnit {
        let fid = sm.add_file_str(path, src);
        let file = sm.file(fid);
        let mut diags = DiagBag::new();
        let lexed = lex_file(file, &mut diags);
        let unit = parse_file(file, &lexed, interner, &mut diags);
        assert!(
            !diags.has_errors(),
            "unexpected parse errors in `{path}`:\n{}",
            diags.render_all(sm)
        );
        unit
    }

    #[test]
    fn sha256_fips_180_4_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(sha256_hex(&[b'a'; 55]).len(), 64);
        assert_eq!(sha256_hex(&[b'a'; 64]).len(), 64);
    }

    #[test]
    fn audit_tcb_counts_raw_asm_and_externs() {
        let safe_src = "module app.main;\npub fn hello() {}\n";
        let hal_src = "\
module drv.uart @hal;

extern \"C\" struct CInfo {
    size: u32;
}

extern \"C\" union CVal {
    bits: u64;
}

extern \"C\" fn c_write(b: u8);

pub fn write_byte(b: u8) {
    raw {
        if b > 0_u8 {
            raw {
                asm(\"nop\");
            }
        }
        while false {
            asm(\"pause\");
        }
    }
    let _f = || {
        raw {
            asm(\"hlt\");
        }
    };
}
";
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let u_safe = parse_unit("app/main.eon", safe_src, &mut sm, &mut interner);
        let u_hal = parse_unit("drv/uart.hal.eon", hal_src, &mut sm, &mut interner);

        let report = audit_tcb(&[u_safe, u_hal], &sm, &interner);
        assert_eq!(report.files.len(), 1);
        let f = &report.files[0];
        assert_eq!(f.module_name, "drv.uart");
        assert_eq!(f.path, "drv/uart.hal.eon");
        assert_eq!(f.sha256, sha256_hex(hal_src.as_bytes()));
        // Outer `raw { ... }` spans lines 14..=23 (10 lines); inner `raw` is nested inside it;
        // closure `raw { ... }` spans lines 25..=27 (3 lines). Total distinct lines = 13.
        assert_eq!(f.raw_lines, 13);
        assert_eq!(f.asm_sites, 3);
        assert_eq!(f.extern_c_imports, vec!["CInfo", "CVal", "c_write"]);
        assert_eq!(report.total_raw_lines, 13);
        assert_eq!(report.total_asm_sites, 3);
        assert_eq!(report.total_extern_c, 3);

        let rendered = report.format_report();
        assert!(rendered.contains("EON TCB AUDIT REPORT"));
        assert!(rendered.contains("drv.uart"));
        assert!(rendered.contains(&f.sha256));
    }

    #[test]
    fn check_hal_whitelist_enforces_third_party_trust() {
        let report = TcbAuditReport {
            files: vec![
                TcbFileReport {
                    path: "core/hal.hal.eon".into(),
                    module_name: "core.hal".into(),
                    sha256: "0".repeat(64),
                    raw_lines: 0,
                    asm_sites: 0,
                    extern_c_imports: Vec::new(),
                },
                TcbFileReport {
                    path: "vendor/evil.hal.eon".into(),
                    module_name: "vendor.evil".into(),
                    sha256: "1".repeat(64),
                    raw_lines: 4,
                    asm_sites: 1,
                    extern_c_imports: Vec::new(),
                },
            ],
            total_raw_lines: 4,
            total_asm_sites: 1,
            total_extern_c: 0,
        };

        let empty = BTreeSet::new();
        let err = report
            .check_hal_whitelist(&empty, &["core.", "std."])
            .expect_err("untrusted third-party @hal module must be rejected");
        assert!(err.contains("vendor.evil"));

        // Prefix without dot boundary must not accidentally trust `vendor.evil` when allowing `vendor_ok`.
        let mut wrong_allow = BTreeSet::new();
        wrong_allow.insert("vendor".to_owned());
        assert!(report.check_hal_whitelist(&wrong_allow, &["core", "std"]).is_err());

        let mut allowed = BTreeSet::new();
        allowed.insert("vendor.evil".to_owned());
        assert!(report.check_hal_whitelist(&allowed, &["core.", "std."]).is_ok());
    }
}
