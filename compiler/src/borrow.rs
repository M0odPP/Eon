use std::collections::BTreeMap;

use crate::ast::{
    self, Block, CatchBody, CompilationUnit, DeclKind, DeferBody, ElseBranch, Expr, ExprKind,
    IfCond, IfExpr, JumpAction, MatchArmBody, ParamModifier, Pat, PatKind, Stmt, StmtKind,
    UnaryOp,
};
use crate::check::TypeTable;
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{BuiltinKind, ResolvedProgram, TopDefKind, TopScopeTarget};
use crate::span::{Interner, Span, Symbol};
use crate::types::{FnSig, GenericVal, MethodTarget, ParamSig, Ty, TyId, TypeEnv};

pub fn check_borrows_and_linearity(
    units: &[CompilationUnit],
    prog: &ResolvedProgram,
    env: &TypeEnv,
    type_table: &TypeTable,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    // 1. Check top-level `Fn` and `Sys` bodies.
    for i in 0..prog.defs.len() {
        let def_mod = prog.defs[i].module;
        let def_kind = prog.defs[i].kind.clone();
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else {
            continue;
        };
        match def_kind {
            TopDefKind::Fn(decl_idx) => {
                let Some(sig) = env.fns.get(&crate::resolve::DefId(i as u32)).cloned() else {
                    continue;
                };
                if let DeclKind::Fn(f) = &units[ui].decls[decl_idx].kind {
                    check_single_fn(def_mod, &sig, &f.body, prog, env, type_table, interner, diags);
                }
            }
            TopDefKind::Sys(decl_idx) => {
                let Some(sig) = env.fns.get(&crate::resolve::DefId(i as u32)).cloned() else {
                    continue;
                };
                if let DeclKind::Sys(s) = &units[ui].decls[decl_idx].kind {
                    check_single_fn(def_mod, &sig, &s.body, prog, env, type_table, interner, diags);
                }
            }
            _ => {}
        }
    }

    // 2. Check `impl` method bodies.
    for im in &env.impls {
        let Some(ui) = prog.modules[im.module.0 as usize].unit_idx else {
            continue;
        };
        let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
            continue;
        };
        for &(_, sub_idx, ref sig) in &im.methods {
            if let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind {
                check_single_fn(im.module, sig, &f.body, prog, env, type_table, interner, diags);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn check_single_fn(
    mod_id: crate::resolve::ModuleId,
    sig: &FnSig,
    body: &Block,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    type_table: &TypeTable,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    let mut pre = PosPrepass::new();
    pre.visit_block(body);

    let mut checker = BorrowChecker::new(
        mod_id,
        sig,
        pre.last_use,
        pre.span_pos,
        prog,
        env,
        type_table,
        interner,
        diags,
    );
    checker.validate_view_signature();
    checker.bind_params();
    checker.check_block(body, false);
    checker.exit_scope(body.span);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PathIndexKey {
    Const(i128),
    Ident(Symbol),
    Dynamic,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PathSeg {
    Field(Symbol),
    Index(PathIndexKey),
    Deref,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AccessPath {
    pub(crate) root: Symbol,
    pub(crate) segs: Vec<PathSeg>,
}

impl AccessPath {
    pub(crate) fn format(&self, interner: &Interner) -> String {
        let mut s = interner.get(self.root).to_owned();
        for seg in &self.segs {
            match seg {
                PathSeg::Field(f) => {
                    s.push('.');
                    s.push_str(interner.get(*f));
                }
                PathSeg::Index(PathIndexKey::Const(n)) => {
                    s.push('[');
                    s.push_str(&n.to_string());
                    s.push(']');
                }
                PathSeg::Index(PathIndexKey::Ident(id)) => {
                    s.push('[');
                    s.push_str(interner.get(*id));
                    s.push(']');
                }
                PathSeg::Index(PathIndexKey::Dynamic) => s.push_str("[_]"),
                PathSeg::Deref => s.push_str(".*"),
            }
        }
        s
    }
}

pub(crate) fn paths_conflict(a: &AccessPath, b: &AccessPath) -> bool {
    if a.root != b.root {
        return false;
    }
    for (sa, sb) in a.segs.iter().zip(b.segs.iter()) {
        match (sa, sb) {
            (PathSeg::Field(f1), PathSeg::Field(f2)) => {
                if f1 != f2 {
                    return false;
                }
            }
            (PathSeg::Index(PathIndexKey::Const(i1)), PathSeg::Index(PathIndexKey::Const(i2))) => {
                if i1 != i2 {
                    return false;
                }
            }
            (PathSeg::Index(_), PathSeg::Index(_)) | (PathSeg::Deref, PathSeg::Deref) => {}
            _ => return false,
        }
    }
    true
}

#[derive(Clone, Debug)]
struct ActiveLoan {
    holder: Option<Symbol>,
    path: AccessPath,
    is_mut: bool,
    expires_after_pos: u32,
}

#[derive(Clone, Debug)]
struct LinearState {
    sym: Symbol,
    name: String,
    span: Span,
    is_linear: bool,
    is_borrowing_val: bool,
    ty: Option<TyId>,
    scope_depth: usize,
    normal_consumed: Option<Span>,
    error_consumed: Option<Span>,
    has_errdefer: bool,
    has_defer: bool,
    errdefer_moved: Option<Span>,
    pool_epoch: u32,
    id_brand: Option<(Symbol, u32)>,
    borrow_origin: Option<ExprOrigin>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ExprOrigin {
    ParamBorrow(Symbol),
    LocalStack(Symbol),
    NoBorrow,
    Other,
}

impl ExprOrigin {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::NoBorrow, origin) | (origin, Self::NoBorrow) => origin,
            (a, b) if a == b => a,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Debug)]
struct ExprFacts {
    path: Option<AccessPath>,
    is_borrowing_val: bool,
    origin: ExprOrigin,
    brand: Option<(Symbol, u32)>,
    ty: Option<TyId>,
    extra_loans: Vec<(AccessPath, bool)>,
}

impl ExprFacts {
    fn plain(ty: Option<TyId>) -> Self {
        Self {
            path: None,
            is_borrowing_val: false,
            origin: ExprOrigin::Other,
            brand: None,
            ty,
            extra_loans: Vec::new(),
        }
    }

    fn from_view_source(source: &Self, ty: Option<TyId>, env: &TypeEnv) -> Self {
        Self {
            path: source.path.clone(),
            is_borrowing_val: ty.is_some_and(|t| env.contains_borrowing_value(t)),
            origin: source.origin,
            brand: None,
            ty,
            extra_loans: source.extra_loans.clone(),
        }
    }
}

pub(crate) enum ViewSourceError {
    Missing,
    MultipleMarkers,
    NotBorrowed(Span),
}

pub(crate) fn is_borrow_source(env: &TypeEnv, param: &ParamSig) -> bool {
    env.is_borrow_or_view(param.ty)
}

pub(crate) fn view_source_index(env: &TypeEnv, sig: &FnSig) -> Result<usize, ViewSourceError> {
    let mut explicit = sig.params.iter().enumerate().filter(|(_, p)| p.modifier == Some(ParamModifier::From));
    let marked = explicit.next();
    if explicit.next().is_some() {
        return Err(ViewSourceError::MultipleMarkers);
    }
    if let Some((_, param)) = marked
        && !is_borrow_source(env, param)
    {
        return Err(ViewSourceError::NotBorrowed(param.span));
    }
    if sig.params.first().is_some_and(|p| p.is_self && is_borrow_source(env, p)) {
        return Ok(0);
    }
    if let Some((index, _)) = marked {
        return Ok(index);
    }
    let mut candidates = sig.params.iter().enumerate().filter(|(_, p)| is_borrow_source(env, p));
    match (candidates.next(), candidates.next()) {
        (Some((index, _)), None) => Ok(index),
        _ => Err(ViewSourceError::Missing),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ExecMode {
    Normal,
    Defer,
    Errdefer,
}

struct LoopCtx {
    label: Option<Symbol>,
    entry_depth: usize,
    outer_consumed_on_non_break: Vec<(Symbol, Span)>,
    break_outer_states: Vec<Vec<LinearState>>,
    break_origin: ExprOrigin,
}

/// Pre-pass assigning deterministic AST order positions and recording each binding's last-use pos for NLL.
struct PosPrepass {
    cur: u32,
    span_pos: BTreeMap<(u32, u32), u32>,
    last_use: BTreeMap<Symbol, u32>,
    loop_depth: u32,
    used_in_loop: Vec<Vec<Symbol>>,
}

impl PosPrepass {
    fn new() -> Self {
        Self {
            cur: 0,
            span_pos: BTreeMap::new(),
            last_use: BTreeMap::new(),
            loop_depth: 0,
            used_in_loop: Vec::new(),
        }
    }

    fn bump(&mut self, sp: Span) -> u32 {
        self.cur += 1;
        self.span_pos.insert((sp.start, sp.end), self.cur);
        self.cur
    }

    fn note_use(&mut self, sym: Symbol, pos: u32) {
        self.last_use
            .entry(sym)
            .and_modify(|p| *p = (*p).max(pos))
            .or_insert(pos);
        if self.loop_depth > 0
            && let Some(top) = self.used_in_loop.last_mut()
        {
            top.push(sym);
        }
    }

    fn enter_loop(&mut self) {
        self.loop_depth += 1;
        self.used_in_loop.push(Vec::new());
    }

    fn exit_loop(&mut self) {
        let end_pos = self.cur;
        self.loop_depth -= 1;
        if let Some(syms) = self.used_in_loop.pop() {
            for s in syms {
                self.note_use(s, end_pos);
            }
        }
    }

    fn visit_block(&mut self, b: &Block) {
        for s in &b.stmts {
            self.visit_stmt(s);
        }
        if let Some(t) = &b.tail_expr {
            self.visit_expr(t);
        }
        self.bump(b.span);
    }

    fn visit_stmt(&mut self, s: &Stmt) {
        let _ = self.bump(s.span);
        match &s.kind {
            StmtKind::Let { init, .. } | StmtKind::Var { init, .. } => self.visit_expr(init),
            StmtKind::Const(c) => self.visit_expr(&c.value),
            StmtKind::Defer(d) | StmtKind::Errdefer { body: d, .. } => match d {
                DeferBody::Block(b) => self.visit_block(b),
                DeferBody::Expr(e) => self.visit_expr(e),
            },
            StmtKind::Jump(j) => match j {
                JumpAction::Return(Some(v)) | JumpAction::Break { value: Some(v), .. } => {
                    self.visit_expr(v)
                }
                _ => {}
            },
            StmtKind::If { data, .. } => self.visit_if(data),
            StmtKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.visit_expr(g);
                    }
                    match &a.body {
                        MatchArmBody::Block(b) => self.visit_block(b),
                        MatchArmBody::Expr(e) => self.visit_expr(e),
                        MatchArmBody::Jump(j) => {
                            if let JumpAction::Return(Some(v))
                            | JumpAction::Break { value: Some(v), .. } = j
                            {
                                self.visit_expr(v);
                            }
                        }
                    }
                }
            }
            StmtKind::While { cond, body, .. } => {
                self.enter_loop();
                self.visit_expr(cond);
                self.visit_block(body);
                self.exit_loop();
            }
            StmtKind::For { iter_start, iter_end, body, .. }
            | StmtKind::ParFor { iter_start, iter_end, body, .. } => {
                self.visit_expr(iter_start);
                if let Some((_, e)) = iter_end {
                    self.visit_expr(e);
                }
                self.enter_loop();
                self.visit_block(body);
                self.exit_loop();
            }
            StmtKind::Loop { body, .. } => {
                self.enter_loop();
                self.visit_block(body);
                self.exit_loop();
            }
            StmtKind::ComptimeBlock(b) | StmtKind::RawBlock(b) => self.visit_block(b),
            StmtKind::Assign { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            StmtKind::Discard(e) | StmtKind::Expr(e) => self.visit_expr(e),
        }
    }

    fn visit_if(&mut self, data: &IfExpr) {
        match &data.cond {
            IfCond::Expr(e) | IfCond::Let { expr: e, .. } => self.visit_expr(e),
        }
        self.visit_block(&data.then_block);
        if let Some(eb) = &data.else_branch {
            match eb {
                ElseBranch::Block(b) => self.visit_block(b),
                ElseBranch::If { data: nested, .. } => self.visit_if(nested),
            }
        }
    }

    fn visit_expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Ident(sym) => {
                let p = self.bump(e.span);
                self.note_use(*sym, p);
            }
            ExprKind::AnonStructLit(fs) => {
                for f in fs {
                    self.visit_expr(&f.value);
                }
                self.bump(e.span);
            }
            ExprKind::NamedStructLit { fields, .. } => {
                for f in fields {
                    self.visit_expr(&f.value);
                }
                self.bump(e.span);
            }
            ExprKind::ArrayLit(es) | ExprKind::Tuple(es) => {
                for x in es {
                    self.visit_expr(x);
                }
                self.bump(e.span);
            }
            ExprKind::ArrayRepeat { val, count } => {
                self.visit_expr(val);
                self.visit_expr(count);
                self.bump(e.span);
            }
            ExprKind::Paren(x)
            | ExprKind::ComptimeExpr(x)
            | ExprKind::Unary { expr: x, .. }
            | ExprKind::Deref(x)
            | ExprKind::GenericApply { base: x, .. }
            | ExprKind::Is { expr: x, .. } => {
                self.visit_expr(x);
                self.bump(e.span);
            }
            ExprKind::Field { base, .. } => {
                self.visit_expr(base);
                self.bump(e.span);
            }
            ExprKind::Binary { lhs, rhs, .. } | ExprKind::Index { base: lhs, index: rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
                self.bump(e.span);
            }
            ExprKind::Slice { base, lo, hi } => {
                self.visit_expr(base);
                if let Some(l) = lo {
                    self.visit_expr(l);
                }
                if let Some(h) = hi {
                    self.visit_expr(h);
                }
                self.bump(e.span);
            }
            ExprKind::Call { callee, args } => {
                self.visit_expr(callee);
                for a in args {
                    self.visit_expr(a);
                }
                self.bump(e.span);
            }
            ExprKind::Orelse { lhs, rhs } | ExprKind::Catch { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                match rhs {
                    CatchBody::Block(b) => self.visit_block(b),
                    CatchBody::Expr(ex) => self.visit_expr(ex),
                    CatchBody::Jump(JumpAction::Return(Some(v)))
                    | CatchBody::Jump(JumpAction::Break { value: Some(v), .. }) => {
                        self.visit_expr(v)
                    }
                    _ => {}
                }
                self.bump(e.span);
            }
            ExprKind::If { data, .. } => {
                self.visit_if(data);
                self.bump(e.span);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.visit_expr(g);
                    }
                    match &a.body {
                        MatchArmBody::Block(b) => self.visit_block(b),
                        MatchArmBody::Expr(ex) => self.visit_expr(ex),
                        MatchArmBody::Jump(JumpAction::Return(Some(v)))
                        | MatchArmBody::Jump(JumpAction::Break { value: Some(v), .. }) => {
                            self.visit_expr(v);
                        }
                        _ => {}
                    }
                }
                self.bump(e.span);
            }
            ExprKind::Loop { body, .. } => {
                self.enter_loop();
                self.visit_block(body);
                self.exit_loop();
                self.bump(e.span);
            }
            ExprKind::ComptimeBlock(b) | ExprKind::Closure { body: b, .. } => {
                self.visit_block(b);
                self.bump(e.span);
            }
            _ => {
                self.bump(e.span);
            }
        }
    }
}

struct BorrowChecker<'a> {
    mod_id: crate::resolve::ModuleId,
    sig: &'a FnSig,
    last_use: BTreeMap<Symbol, u32>,
    span_pos: BTreeMap<(u32, u32), u32>,
    prog: &'a ResolvedProgram,
    env: &'a TypeEnv,
    type_table: &'a TypeTable,
    interner: &'a Interner,
    diags: &'a mut DiagBag,

    view_source_param: Option<Symbol>,
    vars: Vec<LinearState>,
    scope_depth: usize,
    closure_depth: usize,
    loans: Vec<ActiveLoan>,
    mode: ExecMode,
    loop_stack: Vec<LoopCtx>,
}

impl<'a> BorrowChecker<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mod_id: crate::resolve::ModuleId,
        sig: &'a FnSig,
        last_use: BTreeMap<Symbol, u32>,
        span_pos: BTreeMap<(u32, u32), u32>,
        prog: &'a ResolvedProgram,
        env: &'a TypeEnv,
        type_table: &'a TypeTable,
        interner: &'a Interner,
        diags: &'a mut DiagBag,
    ) -> Self {
        Self {
            mod_id,
            sig,
            last_use,
            span_pos,
            prog,
            env,
            type_table,
            interner,
            diags,
            view_source_param: None,
            vars: Vec::new(),
            scope_depth: 0,
            closure_depth: 0,
            loans: Vec::new(),
            mode: ExecMode::Normal,
            loop_stack: Vec::new(),
        }
    }

    fn ty_of(&self, sp: Span) -> Option<TyId> {
        self.type_table.get(sp).filter(|&t| t != TyId::ERROR)
    }

    fn pos_of(&self, sp: Span) -> u32 {
        self.span_pos.get(&(sp.start, sp.end)).copied().unwrap_or(0)
    }

    fn purge_expired(&mut self, cur_pos: u32) {
        self.loans.retain(|l| cur_pos <= l.expires_after_pos);
    }

    fn validate_view_signature(&mut self) {
        if !self.sig.is_view {
            return;
        }
        let first_span = self.sig.params.first().map(|p| p.span).unwrap_or(Span::DUMMY);
        match view_source_index(self.env, self.sig) {
            Ok(index) => self.view_source_param = Some(self.sig.params[index].name),
            Err(ViewSourceError::MultipleMarkers) => self.diags.error(
                ErrorCode::E0310,
                first_span,
                "`view fn` cannot have multiple `from` source parameters",
            ),
            Err(ViewSourceError::NotBorrowed(span)) => self.diags.error(
                ErrorCode::E0310,
                span,
                "`from` must mark a reference or borrowing value",
            ),
            Err(ViewSourceError::Missing) => self.diags.error(
                ErrorCode::E0310,
                first_span,
                "`view fn` requires `self`, a single borrow parameter, or explicit `from` marker to anchor returned borrow",
            ),
        }
    }

    fn bind_params(&mut self) {
        for p in &self.sig.params {
            let brand = match self.env.get(p.ty) {
                Ty::Builtin(BuiltinKind::IdBrand, args) => {
                    if let Some(GenericVal::Variant(b)) = args.first().copied() {
                        Some((b, 0))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            self.push_var(p.name, p.span, Some(p.ty), brand);
            if is_borrow_source(self.env, p) {
                self.set_borrow_origin(p.name, ExprOrigin::ParamBorrow(p.name));
            }
        }
    }

    fn push_var(
        &mut self,
        sym: Symbol,
        span: Span,
        ty: Option<TyId>,
        brand: Option<(Symbol, u32)>,
    ) {
        let name = self.interner.get(sym).to_owned();
        if name == "_" {
            return;
        }
        let actual_linear = ty.is_some_and(|t| {
            !matches!(self.env.get(t), Ty::Borrow { .. }) && self.env.is_linear(t)
        });
        let is_borrowing_val = ty.is_some_and(|t| self.env.contains_borrowing_value(t));

        if let Some(existing) = self.vars.iter_mut().find(|v| v.sym == sym) {
            if existing.is_linear && existing.normal_consumed.is_none() {
                self.diags.error(
                    ErrorCode::E0109,
                    span,
                    format!("ZM-09: overwriting unconsumed `linear` variable `{}`", name),
                );
            }
            existing.normal_consumed = None;
            existing.error_consumed = None;
            existing.has_defer = false;
            existing.errdefer_moved = None;
            existing.ty = ty;
            existing.is_linear = actual_linear;
            existing.is_borrowing_val = is_borrowing_val;
            existing.borrow_origin = None;
            existing.pool_epoch += 1;
            if let Some(b) = brand {
                existing.id_brand = Some(b);
            }
            return;
        }

        self.vars.push(LinearState {
            sym,
            name,
            span,
            is_linear: actual_linear,
            is_borrowing_val,
            ty,
            scope_depth: self.scope_depth,
            normal_consumed: None,
            error_consumed: None,
            has_errdefer: false,
            has_defer: false,
            errdefer_moved: None,
            pool_epoch: 0,
            id_brand: brand,
            borrow_origin: None,
        });
    }

    fn find_var_idx(&self, sym: Symbol) -> Option<usize> {
        self.vars.iter().rposition(|v| v.sym == sym)
    }

    fn set_borrow_origin(&mut self, sym: Symbol, origin: ExprOrigin) {
        if let Some(index) = self.find_var_idx(sym) {
            self.vars[index].borrow_origin = Some(origin);
        }
    }

    fn has_borrow(&self, ty: TyId) -> bool {
        self.env.contains_second_class_ref(ty) || self.env.contains_borrowing_value(ty)
    }

    fn value_origin(&self, facts: &ExprFacts) -> ExprOrigin {
        if facts.ty.is_some_and(|ty| !self.has_borrow(ty)) {
            ExprOrigin::NoBorrow
        } else {
            facts.origin
        }
    }

    fn exit_scope(&mut self, _span: Span) {
        let depth = self.scope_depth;
        while let Some(v) = self.vars.last() {
            if v.scope_depth < depth {
                break;
            }
            let v = self.vars.pop().unwrap();
            if v.is_linear && v.normal_consumed.is_none() && self.mode != ExecMode::Errdefer {
                self.diags.error(
                    ErrorCode::E0109,
                    v.span,
                    format!(
                        "ZM-09: `linear` value `{}` dropped at scope exit without being consumed",
                        v.name
                    ),
                );
            } else if v.is_linear && self.mode == ExecMode::Errdefer && v.normal_consumed.is_none() {
                self.diags.error(
                    ErrorCode::E0109,
                    v.span,
                    format!("ZM-09: `linear` value `{}` created inside `errdefer` is unconsumed", v.name),
                );
            }
            self.loans.retain(|l| l.holder != Some(v.sym));
        }
    }

    fn consume_var(&mut self, sym: Symbol, span: Span, err_desc: &str) {
        let Some(idx) = self.find_var_idx(sym) else {
            return;
        };
        let is_linear = self.vars[idx].is_linear;
        let name = self.vars[idx].name.clone();
        let depth = self.vars[idx].scope_depth;
        let has_defer = self.vars[idx].has_defer;
        let has_errdefer = self.vars[idx].has_errdefer;

        if !is_linear {
            return;
        }

        match self.mode {
            ExecMode::Normal => {
                if self.vars[idx].normal_consumed.is_some() {
                    if has_defer {
                        self.diags.error(
                            ErrorCode::E0321,
                            span,
                            format!(
                                "cannot consume `linear` value `{}` twice (already scheduled for `defer`)",
                                name
                            ),
                        );
                    } else {
                        self.diags.error(
                            ErrorCode::E0321,
                            span,
                            format!("`linear` value `{}` was already consumed", name),
                        );
                    }
                    return;
                }
                self.vars[idx].normal_consumed = Some(span);
                if has_errdefer {
                    self.vars[idx].errdefer_moved = Some(span);
                }
                if let Some(top_loop) = self.loop_stack.last_mut()
                    && depth <= top_loop.entry_depth
                {
                    top_loop.outer_consumed_on_non_break.push((sym, span));
                }
            }
            ExecMode::Errdefer => {
                if self.vars[idx].error_consumed.is_some() {
                    self.diags.error(
                        ErrorCode::E0325,
                        span,
                        format!(
                            "`errdefer` attempts to {} `{}` which is already consumed on the error path",
                            err_desc, name
                        ),
                    );
                    return;
                }
                self.vars[idx].error_consumed = Some(span);
                self.vars[idx].has_errdefer = true;
            }
            ExecMode::Defer => {
                if self.vars[idx].normal_consumed.is_some() || self.vars[idx].error_consumed.is_some() {
                    self.diags.error(
                        ErrorCode::E0321,
                        span,
                        format!("`defer` consumes `linear` value `{}` which is already consumed", name),
                    );
                    return;
                }
                self.vars[idx].normal_consumed = Some(span);
                self.vars[idx].error_consumed = Some(span);
                self.vars[idx].has_defer = true;
            }
        }
    }

    fn check_error_edge(&mut self, span: Span) {
        if self.mode != ExecMode::Normal {
            return;
        }
        for v in &self.vars {
            if !v.is_linear {
                continue;
            }
            if v.errdefer_moved.is_some() {
                self.diags.error(
                    ErrorCode::E0325,
                    span,
                    format!(
                        "`{}` was moved on normal path after `errdefer` registration before potential error",
                        v.name
                    ),
                );
            } else if v.error_consumed.is_none() {
                self.diags.error(
                    ErrorCode::E0109,
                    span,
                    format!(
                        "ZM-09: `linear` value `{}` is unconsumed on error path (`try` / error `return`); add `errdefer`",
                        v.name
                    ),
                );
            }
        }
    }

    fn check_return_normal_all_consumed(&mut self, span: Span) {
        for v in &self.vars {
            if v.is_linear && v.normal_consumed.is_none() {
                self.diags.error(
                    ErrorCode::E0109,
                    span,
                    format!(
                        "ZM-09: `linear` value `{}` leaks at `return` without being consumed",
                        v.name
                    ),
                );
            }
        }
    }

    fn check_block(&mut self, b: &Block, pop_on_end: bool) -> bool {
        self.check_block_value(b, pop_on_end).0
    }

    fn check_block_value(&mut self, b: &Block, pop_on_end: bool) -> (bool, ExprOrigin) {
        self.scope_depth += 1;
        let mut diverged = false;
        for stmt in &b.stmts {
            if self.check_stmt(stmt) {
                diverged = true;
            }
        }
        let origin = if let Some(tail) = &b.tail_expr {
            let f = self.check_expr(tail, false);
            if diverged { ExprOrigin::NoBorrow } else { self.value_origin(&f) }
        } else {
            ExprOrigin::NoBorrow
        };
        if pop_on_end || !diverged {
            if diverged && !pop_on_end {
                let depth = self.scope_depth;
                while let Some(v) = self.vars.last() {
                    if v.scope_depth < depth {
                        break;
                    }
                    let v = self.vars.pop().unwrap();
                    self.loans.retain(|l| l.holder != Some(v.sym));
                }
            } else {
                self.exit_scope(b.span);
            }
        } else {
            let depth = self.scope_depth;
            while let Some(v) = self.vars.last() {
                if v.scope_depth < depth {
                    break;
                }
                let v = self.vars.pop().unwrap();
                self.loans.retain(|l| l.holder != Some(v.sym));
            }
        }
        self.scope_depth -= 1;
        (diverged, origin)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        let cur_pos = self.pos_of(stmt.span);
        self.purge_expired(cur_pos);

        match &stmt.kind {
            StmtKind::Let { pat, init, .. } => {
                if matches!(pat.kind, PatKind::Wildcard)
                    && self.ty_of(init.span).is_some_and(|t| self.env.is_linear(t))
                {
                    let _ = self.check_expr(init, false);
                    self.diags.error(
                        ErrorCode::E0109,
                        stmt.span,
                        "ZM-09: cannot drop `linear` value with `let _ = ...`; destructure fields or call a consuming function",
                    );
                    if let ExprKind::Ident(sym) = &init.kind {
                        self.consume_var(*sym, stmt.span, "discard");
                    }
                    return false;
                }
                let init_facts = self.check_expr(init, true);
                let pat_ty = init_facts.ty.or_else(|| self.ty_of(init.span));
                self.bind_pattern_vars(pat, pat_ty, init_facts.brand, init_facts.origin);
                self.register_holder_loan(pat, cur_pos, &init_facts, pat_ty);
                false
            }
            StmtKind::Var { name, init, .. } => {
                let init_facts = self.check_expr(init, true);
                let var_ty = init_facts.ty.or_else(|| self.ty_of(init.span));
                self.push_var(name.sym, name.span, var_ty, init_facts.brand);
                if var_ty.is_some_and(|ty| self.has_borrow(ty)) {
                    self.set_borrow_origin(name.sym, ExprOrigin::Other);
                }
                let exp = self.last_use.get(&name.sym).copied().unwrap_or(cur_pos);
                if exp > cur_pos {
                    if (init_facts.is_borrowing_val
                        || var_ty.is_some_and(|t| self.env.contains_borrowing_value(t)))
                        && let Some(src_path) = init_facts.path
                    {
                        self.loans.push(ActiveLoan {
                            holder: Some(name.sym),
                            path: src_path,
                            is_mut: true,
                            expires_after_pos: exp,
                        });
                    }
                    for (lp, lm) in init_facts.extra_loans {
                        self.loans.push(ActiveLoan {
                            holder: Some(name.sym),
                            path: lp,
                            is_mut: lm,
                            expires_after_pos: exp,
                        });
                    }
                }
                false
            }
            StmtKind::Const(c) => {
                let _ = self.check_expr(&c.value, false);
                false
            }
            StmtKind::Defer(body) => {
                let saved_mode = self.mode;
                self.mode = ExecMode::Defer;
                self.check_defer_body(body);
                self.mode = saved_mode;
                false
            }
            StmtKind::Errdefer { body, .. } => {
                let saved_mode = self.mode;
                self.mode = ExecMode::Errdefer;
                self.check_defer_body(body);
                self.mode = saved_mode;
                false
            }
            StmtKind::Jump(j) => {
                self.check_jump(j, stmt.span);
                true
            }
            StmtKind::If { data, .. } => self.check_if(data).0,
            StmtKind::Match { scrutinee, arms } => self.check_match(scrutinee, arms, stmt.span).0,
            StmtKind::While { label, cond, body, .. } => {
                self.check_expr(cond, false);
                self.check_loop_body(body, label.map(|l| l.sym), false).0
            }
            StmtKind::For { pat, iter_start, iter_end, body, .. }
            | StmtKind::ParFor { pat, iter_start, iter_end, body, .. } => {
                self.check_expr(iter_start, false);
                if let Some((_, e)) = iter_end {
                    self.check_expr(e, false);
                }
                self.scope_depth += 1;
                self.bind_pattern_vars(pat, None, None, ExprOrigin::Other);
                let label = match &stmt.kind {
                    StmtKind::For { label, .. } => label.map(|l| l.sym),
                    _ => None,
                };
                let div = self.check_loop_body(body, label, false).0;
                self.exit_scope(body.span);
                self.scope_depth -= 1;
                div
            }
            StmtKind::Loop { label, body } => self.check_loop_body(body, label.map(|l| l.sym), true).0,
            StmtKind::ComptimeBlock(b) | StmtKind::RawBlock(b) => self.check_block(b, true),
            StmtKind::Assign { lhs, rhs, .. } => {
                if self.is_pool_reset_call(rhs, lhs)
                    && let ExprKind::Ident(lhs_sym) = &lhs.kind
                    && let Some(idx) = self.find_var_idx(*lhs_sym)
                {
                    let lpath = AccessPath {
                        root: *lhs_sym,
                        segs: Vec::new(),
                    };
                    self.check_access_against_loans(&lpath, true, lhs.span);
                    self.vars[idx].pool_epoch += 1;
                    return false;
                }

                let rhs_facts = self.check_expr(rhs, true);
                if let ExprKind::Ident(lhs_sym) = &lhs.kind {
                    self.check_access_against_loans(
                        &AccessPath {
                            root: *lhs_sym,
                            segs: Vec::new(),
                        },
                        true,
                        lhs.span,
                    );
                    let lhs_ty = self.find_var_idx(*lhs_sym).and_then(|i| self.vars[i].ty);
                    self.push_var(*lhs_sym, lhs.span, lhs_ty.or(rhs_facts.ty), rhs_facts.brand);
                    if lhs_ty.is_some_and(|ty| self.has_borrow(ty)) {
                        self.set_borrow_origin(*lhs_sym, ExprOrigin::Other);
                    }
                } else {
                    let lf = self.build_place_path(lhs);
                    if let Some(p) = lf.path {
                        self.check_access_against_loans(&p, true, lhs.span);
                    }
                }
                false
            }
            StmtKind::Discard(e) => {
                if self.ty_of(e.span).is_some_and(|t| self.env.is_linear(t)) {
                    let _ = self.check_expr(e, false);
                    self.diags.error(
                        ErrorCode::E0109,
                        stmt.span,
                        "ZM-09: cannot discard `linear` value with `_ = ...`; destructure or consume it",
                    );
                } else {
                    let _ = self.check_expr(e, false);
                }
                false
            }
            StmtKind::Expr(e) => {
                let _ = self.check_expr(e, false);
                false
            }
        }
    }

    fn register_holder_loan(
        &mut self,
        pat: &Pat,
        cur_pos: u32,
        init_facts: &ExprFacts,
        pat_ty: Option<TyId>,
    ) {
        let PatKind::Ident(sym) = &pat.kind else {
            return;
        };
        let exp = self.last_use.get(sym).copied().unwrap_or(cur_pos);
        if exp <= cur_pos {
            return;
        }
        for (lp, lm) in &init_facts.extra_loans {
            self.loans.push(ActiveLoan {
                holder: Some(*sym),
                path: lp.clone(),
                is_mut: *lm,
                expires_after_pos: exp,
            });
        }
        let Some(src_path) = init_facts.path.clone() else {
            return;
        };
        let holds_borrowing_val = init_facts.is_borrowing_val
            || pat_ty.is_some_and(|t| self.env.contains_borrowing_value(t));
        let holds_borrow_ref = pat_ty.is_some_and(|t| {
            matches!(self.env.get(t), Ty::Borrow { .. })
                || matches!(self.env.get(t), Ty::Optional(i) if matches!(self.env.get(*i), Ty::Borrow { .. }))
        });
        if holds_borrowing_val || holds_borrow_ref {
            let is_mut = holds_borrowing_val
                || pat_ty.is_some_and(|t| {
                    matches!(self.env.get(t), Ty::Borrow { is_mut: true, .. })
                        || matches!(self.env.get(t), Ty::Optional(i) if matches!(self.env.get(*i), Ty::Borrow { is_mut: true, .. }))
                });
            self.loans.push(ActiveLoan {
                holder: Some(*sym),
                path: src_path,
                is_mut,
                expires_after_pos: exp,
            });
        }
    }

    fn is_pool_reset_call(&self, rhs: &Expr, lhs: &Expr) -> bool {
        if let ExprKind::Ident(lsym) = &lhs.kind
            && let ExprKind::Call { callee, args } = &rhs.kind
            && args.is_empty()
            && let ExprKind::Field { base, field } = &callee.kind
            && self.interner.get(field.sym) == "reset"
            && let ExprKind::Ident(rsym) = &base.kind
        {
            return lsym == rsym;
        }
        false
    }

    fn check_defer_body(&mut self, body: &DeferBody) {
        self.scope_depth += 1;
        match body {
            DeferBody::Block(b) => {
                for s in &b.stmts {
                    self.check_stmt(s);
                }
                if let Some(t) = &b.tail_expr {
                    self.check_expr(t, false);
                }
            }
            DeferBody::Expr(e) => {
                self.check_expr(e, false);
            }
        }
        self.exit_scope(Span::DUMMY);
        self.scope_depth -= 1;
    }

    fn check_jump(&mut self, j: &JumpAction, span: Span) {
        match j {
            JumpAction::Return(val_opt) => {
                if let Some(val) = val_opt {
                    let f = self.check_expr(val, true);
                    if let ExprKind::EnumLit(ev) = &val.kind
                        && let Ty::ErrorUnion { errors, .. } = self.env.get(self.sig.ret)
                        && self.env.error_sources_have_variant(errors, ev.sym, self.interner)
                    {
                        self.check_error_edge(span);
                        return;
                    }
                    self.check_view_return_origin(&f, val.span);
                }
                self.check_return_normal_all_consumed(span);
            }
            JumpAction::Break { label, value } => {
                let origin = if let Some(v) = value {
                    let facts = self.check_expr(v, true);
                    self.value_origin(&facts)
                } else {
                    ExprOrigin::NoBorrow
                };
                if let Some(index) = self.loop_target(label.map(|l| l.sym)) {
                    let target = &mut self.loop_stack[index];
                    let entry = target.entry_depth;
                    for v in &self.vars {
                        if v.scope_depth > entry && v.is_linear && v.normal_consumed.is_none() {
                            self.diags.error(
                                ErrorCode::E0109,
                                v.span,
                                format!(
                                    "ZM-09: `linear` value `{}` created inside loop leaks on `break`",
                                    v.name
                                ),
                            );
                        }
                    }
                    target.break_outer_states.push(
                        self.vars.iter().filter(|v| v.scope_depth <= entry).cloned().collect(),
                    );
                    target.break_origin = target.break_origin.merge(origin);
                }
            }
            JumpAction::Continue { .. } => {
                if let Some(top_loop) = self.loop_stack.last() {
                    let entry = top_loop.entry_depth;
                    for &(outer_sym, cspan) in &top_loop.outer_consumed_on_non_break {
                        let name = self.interner.get(outer_sym);
                        self.diags.error(
                            ErrorCode::E0324,
                            cspan,
                            format!(
                                "cannot consume outer `linear` variable `{}` before `continue` (next loop iteration would double-consume it)",
                                name
                            ),
                        );
                    }
                    for v in &self.vars {
                        if v.scope_depth > entry && v.is_linear && v.normal_consumed.is_none() {
                            self.diags.error(
                                ErrorCode::E0109,
                                v.span,
                                format!(
                                    "ZM-09: `linear` value `{}` created inside loop leaks on `continue`",
                                    v.name
                                ),
                            );
                        }
                    }
                }
            }
            JumpAction::Unreachable => {}
        }
    }

    fn check_view_return_origin(&mut self, facts: &ExprFacts, span: Span) {
        let Some(source) = self.view_source_param else {
            return;
        };
        if self.closure_depth > 0 || !self.has_borrow(self.sig.ret) {
            return;
        }
        let source_name = self.interner.get(source);
        let message = match facts.origin {
            ExprOrigin::NoBorrow => return,
            ExprOrigin::ParamBorrow(sym) if sym == source => return,
            ExprOrigin::ParamBorrow(sym) => format!(
                "`view fn` returns a borrow from `{}`; expected a borrow derived from `{source_name}`",
                self.interner.get(sym)
            ),
            ExprOrigin::LocalStack(sym) => format!(
                "`view fn` cannot return borrow derived from local variable `{}`",
                self.interner.get(sym)
            ),
            ExprOrigin::Other => format!("cannot prove `view fn` result is derived from `{source_name}`"),
        };
        self.diags.error(ErrorCode::E0310, span, message);
    }

    fn loop_target(&self, label: Option<Symbol>) -> Option<usize> {
        match label {
            Some(label) => self.loop_stack.iter().rposition(|ctx| ctx.label == Some(label)),
            None => self.loop_stack.len().checked_sub(1),
        }
    }

    fn check_loop_body(&mut self, body: &Block, label: Option<Symbol>, is_forever: bool) -> (bool, ExprOrigin) {
        self.loop_stack.push(LoopCtx {
            label,
            entry_depth: self.scope_depth,
            outer_consumed_on_non_break: Vec::new(),
            break_outer_states: Vec::new(),
            break_origin: ExprOrigin::NoBorrow,
        });
        let diverged = self.check_block(body, true);
        let ctx = self.loop_stack.pop().unwrap();
        if !diverged {
            for &(outer_sym, cspan) in &ctx.outer_consumed_on_non_break {
                let name = self.interner.get(outer_sym);
                self.diags.error(
                    ErrorCode::E0324,
                    cspan,
                    format!(
                        "cannot consume outer `linear` variable `{}` in loop body without exiting via `break` or `return`",
                        name
                    ),
                );
            }
        }
        // If `loop { ... }` exits via `break`, apply outer linear state from `break` path(s).
        if is_forever && let Some(first_break) = ctx.break_outer_states.first() {
            for (v, bv) in self.vars.iter_mut().zip(first_break.iter()) {
                if v.sym == bv.sym {
                    *v = bv.clone();
                }
            }
        }
        (is_forever && ctx.break_outer_states.is_empty(), ctx.break_origin)
    }

    fn check_if(&mut self, data: &IfExpr) -> (bool, ExprOrigin) {
        self.scope_depth += 1;
        match &data.cond {
            IfCond::Expr(cond) => {
                self.check_expr(cond, false);
            }
            IfCond::Let { pat, expr } => {
                let f = self.check_expr(expr, true);
                self.bind_pattern_vars(pat, f.ty, f.brand, f.origin);
            }
        }

        let base_vars = self.vars.clone();
        let outer_count = base_vars.len();

        let (then_div, then_origin) = self.check_block_value(&data.then_block, true);
        let then_outer: Vec<LinearState> = self.vars[..outer_count].to_vec();

        self.vars = base_vars.clone();
        let (else_div, else_origin, else_outer) = if let Some(eb) = &data.else_branch {
            let (diverged, origin) = match eb {
                ElseBranch::Block(b) => self.check_block_value(b, true),
                ElseBranch::If { data: nested, .. } => self.check_if(nested),
            };
            (diverged, origin, self.vars[..outer_count].to_vec())
        } else {
            (false, ExprOrigin::NoBorrow, base_vars.clone())
        };

        let mut live_branches: Vec<&[LinearState]> = Vec::new();
        if !then_div {
            live_branches.push(&then_outer);
        }
        if !else_div {
            live_branches.push(&else_outer);
        }
        if live_branches.len() == 2 {
            for i in 0..outer_count {
                if !base_vars[i].is_linear {
                    continue;
                }
                let a = &live_branches[0][i];
                let b = &live_branches[1][i];
                if a.normal_consumed.is_some() != b.normal_consumed.is_some()
                    || a.error_consumed.is_some() != b.error_consumed.is_some()
                {
                    self.diags.error(
                        ErrorCode::E0323,
                        data.span,
                        format!(
                            "`linear` variable `{}` has inconsistent consumption state across `if`/`else` branches",
                            base_vars[i].name
                        ),
                    );
                }
            }
        }
        if let Some(first_live) = live_branches.first() {
            self.vars = first_live.to_vec();
        } else {
            self.vars = base_vars;
        }
        if let Some(top_loop) = self.loop_stack.last_mut()
            && !live_branches.is_empty()
        {
            top_loop.outer_consumed_on_non_break.retain(|(s, _)| {
                live_branches
                    .iter()
                    .all(|st| st.iter().any(|v| v.sym == *s && v.normal_consumed.is_some()))
            });
        }

        self.exit_scope(data.span);
        self.scope_depth -= 1;
        (then_div && else_div, then_origin.merge(else_origin))
    }

    fn check_match(&mut self, scrutinee: &Expr, arms: &[ast::MatchArm], span: Span) -> (bool, ExprOrigin) {
        let sf = self.check_expr(scrutinee, true);
        let base_vars = self.vars.clone();
        let outer_count = base_vars.len();
        let mut live_outs: Vec<Vec<LinearState>> = Vec::new();
        let mut all_div = !arms.is_empty();
        let mut origin = ExprOrigin::NoBorrow;

        for arm in arms {
            self.vars = base_vars.clone();
            self.scope_depth += 1;
            self.bind_pattern_vars(&arm.pat, sf.ty, sf.brand, sf.origin);
            if let Some(g) = &arm.guard {
                self.check_expr(g, false);
            }
            let (div, arm_origin) = match &arm.body {
                MatchArmBody::Block(b) => self.check_block_value(b, true),
                MatchArmBody::Expr(e) => {
                    let f = self.check_expr(e, true);
                    (false, self.value_origin(&f))
                }
                MatchArmBody::Jump(j) => {
                    self.check_jump(j, arm.span);
                    (true, ExprOrigin::NoBorrow)
                }
            };
            self.exit_scope(arm.span);
            self.scope_depth -= 1;
            all_div &= div;
            origin = origin.merge(arm_origin);
            if !div {
                live_outs.push(self.vars[..outer_count].to_vec());
            }
        }

        if live_outs.len() > 1 {
            let first = &live_outs[0];
            for other in &live_outs[1..] {
                for i in 0..outer_count {
                    if base_vars[i].is_linear
                        && (first[i].normal_consumed.is_some() != other[i].normal_consumed.is_some()
                            || first[i].error_consumed.is_some() != other[i].error_consumed.is_some())
                    {
                        self.diags.error(
                            ErrorCode::E0323,
                            span,
                            format!(
                                "`linear` variable `{}` has inconsistent consumption state across `match` arms",
                                base_vars[i].name
                            ),
                        );
                    }
                }
            }
        }
        if let Some(first) = live_outs.first() {
            self.vars = first.clone();
        } else {
            self.vars = base_vars;
        }
        (all_div, origin)
    }

    /// Build a place `AccessPath` without emitting intermediate prefix loan conflicts on nested projections.
    fn build_place_path(&mut self, expr: &Expr) -> ExprFacts {
        let ty = self.ty_of(expr.span);
        match &expr.kind {
            ExprKind::Ident(sym) => {
                let mut f = ExprFacts::plain(ty);
                if let Some(idx) = self.find_var_idx(*sym) {
                    let v = &self.vars[idx];
                    if let Some((pool_sym, birth_epoch)) = v.id_brand
                        && let Some(pidx) = self.find_var_idx(pool_sym)
                        && birth_epoch != self.vars[pidx].pool_epoch
                    {
                        self.diags.error(
                            ErrorCode::E0331,
                            expr.span,
                            format!(
                                "use of invalidated `Id[{}]` after `{}.reset()`",
                                self.interner.get(pool_sym),
                                self.interner.get(pool_sym)
                            ),
                        );
                    }
                    f.is_borrowing_val = v.is_borrowing_val;
                    f.origin = v.borrow_origin.unwrap_or(ExprOrigin::LocalStack(*sym));
                    f.brand = v.id_brand;
                    f.ty = f.ty.or(v.ty);
                }
                f.path = Some(AccessPath {
                    root: *sym,
                    segs: Vec::new(),
                });
                f
            }
            ExprKind::SelfVal => {
                let sym = self
                    .sig
                    .params
                    .first()
                    .filter(|p| p.is_self)
                    .map(|p| p.name)
                    .unwrap_or(Symbol(0));
                let mut f = ExprFacts::plain(ty);
                f.path = Some(AccessPath {
                    root: sym,
                    segs: Vec::new(),
                });
                f.origin = self.find_var_idx(sym).and_then(|i| self.vars[i].borrow_origin)
                    .unwrap_or(ExprOrigin::LocalStack(sym));
                f
            }
            ExprKind::Paren(inner) => self.build_place_path(inner),
            ExprKind::Field { base, field } => {
                let mut f = self.build_place_path(base);
                if let Some(ref mut p) = f.path {
                    p.segs.push(PathSeg::Field(field.sym));
                }
                f.ty = ty;
                f
            }
            ExprKind::Deref(inner) => {
                let mut f = self.build_place_path(inner);
                if let Some(ref mut p) = f.path {
                    p.segs.push(PathSeg::Deref);
                }
                f.ty = ty;
                f
            }
            ExprKind::Index { base, index } => {
                let mut bf = self.build_place_path(base);
                let _idx_f = self.check_expr(index, false);
                let key = match &index.kind {
                    ExprKind::IntLit { raw, .. } => {
                        if let Some(n) = crate::types::parse_int_literal(self.interner.get(*raw)) {
                            PathIndexKey::Const(n as i128)
                        } else {
                            PathIndexKey::Dynamic
                        }
                    }
                    ExprKind::Ident(s) => PathIndexKey::Ident(*s),
                    _ => PathIndexKey::Dynamic,
                };
                if let Some(ref mut p) = bf.path {
                    p.segs.push(PathSeg::Index(key));
                }
                bf.ty = ty;
                bf
            }
            _ => self.check_expr(expr, false),
        }
    }

    fn address_origin(&self, place: &Expr, origin: ExprOrigin) -> ExprOrigin {
        match &place.kind {
            ExprKind::Ident(sym) => ExprOrigin::LocalStack(*sym),
            ExprKind::SelfVal => self.sig.params.first().map_or(ExprOrigin::Other, |p| ExprOrigin::LocalStack(p.name)),
            ExprKind::Paren(inner) => self.address_origin(inner, origin),
            ExprKind::Deref(_) => origin,
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } | ExprKind::Slice { base, .. } => {
                if self.ty_of(base.span).is_some_and(|ty| {
                    matches!(self.env.get(ty), Ty::Borrow { .. } | Ty::Builtin(BuiltinKind::ArenaPool, _))
                }) {
                    origin
                } else {
                    self.address_origin(base, origin)
                }
            }
            _ => ExprOrigin::Other,
        }
    }

    fn check_expr(&mut self, expr: &Expr, as_move: bool) -> ExprFacts {
        let cur_pos = self.pos_of(expr.span);
        self.purge_expired(cur_pos);
        let ty = self.ty_of(expr.span);

        match &expr.kind {
            ExprKind::Ident(_)
            | ExprKind::SelfVal
            | ExprKind::Field { .. }
            | ExprKind::Deref(_)
            | ExprKind::Index { .. } => {
                let f = self.build_place_path(expr);
                if let Some(p) = &f.path {
                    self.check_access_against_loans(p, as_move, expr.span);
                }
                if as_move && let ExprKind::Ident(sym) = &expr.kind {
                    self.consume_var(*sym, expr.span, "move");
                }
                f
            }
            ExprKind::Paren(inner) => self.check_expr(inner, as_move),
            ExprKind::Unary { op, expr: inner } => match op {
                UnaryOp::Borrow | UnaryOp::BorrowMut => {
                    let is_mut = *op == UnaryOp::BorrowMut;
                    let inner_f = self.build_place_path(inner);
                    if let Some(p) = &inner_f.path {
                        self.check_access_against_loans(p, is_mut, expr.span);
                    }
                    ExprFacts {
                        path: inner_f.path,
                        is_borrowing_val: false,
                        origin: self.address_origin(inner, inner_f.origin),
                        brand: None,
                        ty,
                        extra_loans: Vec::new(),
                    }
                }
                UnaryOp::Move => {
                    let f = self.check_expr(inner, true);
                    if let Some(p) = &f.path
                        && p.segs.is_empty()
                    {
                        self.check_access_against_loans(p, true, expr.span);
                    }
                    f
                }
                UnaryOp::Try => {
                    let mut f = self.check_expr(inner, true);
                    self.check_error_edge(expr.span);
                    f.ty = ty;
                    f
                }
                UnaryOp::Copy => self.check_expr(inner, false),
                _ => self.check_expr(inner, false),
            },
            ExprKind::Slice { base, lo, hi } => {
                let mut bf = self.check_expr(base, false);
                if let Some(l) = lo {
                    self.check_expr(l, false);
                }
                if let Some(h) = hi {
                    self.check_expr(h, false);
                }
                bf.origin = self.address_origin(expr, bf.origin);
                bf.ty = ty;
                bf
            }
            ExprKind::Call { callee, args } => self.check_call_expr(callee, args, ty, cur_pos, expr.span),
            ExprKind::Orelse { lhs, rhs } => {
                let mut lf = self.check_expr(lhs, as_move);
                lf.origin = lf.origin.merge(self.check_catch_body(rhs));
                lf.ty = ty;
                lf
            }
            ExprKind::Catch { lhs, err_bind, rhs } => {
                let mut lf = self.check_expr(lhs, as_move);
                self.scope_depth += 1;
                if let Some(eb) = err_bind {
                    self.push_var(eb.sym, eb.span, None, None);
                }
                lf.origin = lf.origin.merge(self.check_catch_body(rhs));
                self.exit_scope(expr.span);
                self.scope_depth -= 1;
                lf.ty = ty;
                lf
            }
            ExprKind::If { data, .. } => ExprFacts {
                origin: self.check_if(data).1,
                ..ExprFacts::plain(ty)
            },
            ExprKind::Match { scrutinee, arms } => ExprFacts {
                origin: self.check_match(scrutinee, arms, expr.span).1,
                ..ExprFacts::plain(ty)
            },
            ExprKind::Closure { params, body } => self.check_closure_expr(params, body, ty),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.check_expr(lhs, false);
                self.check_expr(rhs, false);
                ExprFacts::plain(ty)
            }
            ExprKind::Tuple(es) => {
                let mut facts = ExprFacts { origin: ExprOrigin::NoBorrow, ..ExprFacts::plain(ty) };
                for x in es {
                    let field = self.check_expr(x, true);
                    facts.origin = facts.origin.merge(self.value_origin(&field));
                }
                facts
            }
            ExprKind::ArrayLit(es) => {
                for x in es {
                    self.check_expr(x, true);
                }
                ExprFacts::plain(ty)
            }
            ExprKind::AnonStructLit(fs) => self.check_struct_lit_fields(fs, ty),
            ExprKind::NamedStructLit { fields, .. } => self.check_struct_lit_fields(fields, ty),
            ExprKind::ArrayRepeat { val, count } => {
                self.check_expr(val, true);
                self.check_expr(count, false);
                ExprFacts::plain(ty)
            }
            ExprKind::GenericApply { base, .. }
            | ExprKind::ComptimeExpr(base)
            | ExprKind::Is { expr: base, .. } => self.check_expr(base, as_move),
            ExprKind::ComptimeBlock(b) => ExprFacts {
                origin: self.check_block_value(b, true).1,
                ..ExprFacts::plain(ty)
            },
            ExprKind::Loop { label, body } => ExprFacts {
                origin: self.check_loop_body(body, label.map(|l| l.sym), true).1,
                ..ExprFacts::plain(ty)
            },
            ExprKind::Null
            | ExprKind::EnumLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::RawStringLit(_)
            | ExprKind::ByteStringLit(_) => ExprFacts {
                origin: ExprOrigin::NoBorrow,
                ..ExprFacts::plain(ty)
            },
            _ => ExprFacts::plain(ty),
        }
    }

    fn check_struct_lit_fields(&mut self, fields: &[ast::FieldInit], ty: Option<TyId>) -> ExprFacts {
        let mut facts = ExprFacts {
            origin: ExprOrigin::NoBorrow,
            ..ExprFacts::plain(ty)
        };
        let mut field_borrows: Vec<(AccessPath, bool, Span)> = Vec::new();
        for f in fields {
            let ff = self.check_expr(&f.value, true);
            facts.origin = facts.origin.merge(self.value_origin(&ff));
            if let Some(bpath) = self.borrow_arg_path(&f.value, &ff) {
                let is_mut = matches!(
                    f.value.kind,
                    ExprKind::Unary { op: UnaryOp::BorrowMut, .. }
                ) || ff.ty.is_some_and(|t| {
                    matches!(self.env.get(t), Ty::Borrow { is_mut: true, .. })
                        || self.env.contains_borrowing_value(t)
                });
                field_borrows.push((bpath.clone(), is_mut, f.value.span));
                facts.extra_loans.push((bpath, is_mut));
            } else if ff.is_borrowing_val
                && let Some(p) = &ff.path
            {
                field_borrows.push((p.clone(), true, f.value.span));
                facts.extra_loans.push((p.clone(), true));
            }
            for (lp, lm) in ff.extra_loans {
                field_borrows.push((lp.clone(), lm, f.value.span));
                facts.extra_loans.push((lp, lm));
            }
        }
        self.check_call_disjoint(&field_borrows);
        facts
    }

    fn check_catch_body(&mut self, body: &CatchBody) -> ExprOrigin {
        match body {
            CatchBody::Block(b) => self.check_block_value(b, true).1,
            CatchBody::Expr(e) => {
                let f = self.check_expr(e, true);
                self.value_origin(&f)
            }
            CatchBody::Jump(j) => {
                self.check_jump(j, Span::DUMMY);
                ExprOrigin::NoBorrow
            }
        }
    }

    fn check_closure_expr(&mut self, params: &[ast::Param], body: &Block, ty: Option<TyId>) -> ExprFacts {
        self.scope_depth += 1;
        self.closure_depth += 1;
        for p in params {
            self.push_var(p.name.sym, p.span, self.ty_of(p.span), None);
        }
        self.check_block(body, true);
        self.exit_scope(body.span);
        self.scope_depth -= 1;
        self.closure_depth -= 1;
        ExprFacts::plain(ty)
    }

    fn check_call_expr(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        ret_ty: Option<TyId>,
        _cur_pos: u32,
        span: Span,
    ) -> ExprFacts {
        let mut call_borrows: Vec<(AccessPath, bool, Span)> = Vec::new();
        let mut arg_facts: Vec<ExprFacts> = Vec::with_capacity(args.len());

        if let ExprKind::Field { base: recv, field } = &callee.kind {
            let mname = self.interner.get(field.sym).to_owned();
            let is_arena_pool_init = matches!(&recv.kind, ExprKind::GenericApply { base, .. }
                if matches!(&base.kind, ExprKind::Ident(s) if self.interner.get(*s) == "ArenaPool"))
                && mname == "init";
            if is_arena_pool_init {
                for a in args {
                    arg_facts.push(self.check_expr(a, false));
                }
                if let Some(a0) = arg_facts.first()
                    && let Some(arena_path) = &a0.path
                {
                    self.check_access_against_loans(arena_path, true, span);
                    return ExprFacts {
                        path: Some(arena_path.clone()),
                        is_borrowing_val: true,
                        origin: a0.origin,
                        brand: None,
                        ty: ret_ty,
                        extra_loans: Vec::new(),
                    };
                }
                return ExprFacts::plain(ret_ty);
            }

            let recv_ty = self.ty_of(recv.span);
            if recv_ty.is_some() || self.expr_has_local_root(recv) {
                let method_sig = recv_ty.and_then(|rt| self.resolve_method_sig(rt, field.sym));
                let mut_method = matches!(
                    mname.as_str(),
                    "reset" | "destroy" | "alloc" | "push" | "push_slice" | "insert" | "try_insert" | "remove" | "get_mut" | "at_mut" | "sub_mut" | "drain"
                ) || method_sig.as_ref().is_some_and(|s| {
                    s.params.first().is_some_and(|p| matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. }))
                });
                let consumes_recv = matches!(mname.as_str(), "destroy" | "drain")
                    || method_sig.as_ref().is_some_and(|s| {
                        s.params.first().is_some_and(|p| p.is_self && !matches!(self.env.get(p.ty), Ty::Borrow { .. }))
                    });

                let rf = if consumes_recv {
                    self.check_expr(recv, true)
                } else {
                    self.build_place_path(recv)
                };
                if let Some(rpath) = &rf.path {
                    call_borrows.push((rpath.clone(), mut_method, recv.span));
                    if !consumes_recv {
                        self.check_access_against_loans(rpath, mut_method, span);
                    }
                }

                if mname == "drain" {
                    let elem_ty = rf.ty.and_then(|t| self.linear_elem_of_collection(t));
                    for a in args {
                        if let ExprKind::Closure { params, body } = &a.kind {
                            self.scope_depth += 1;
                            self.closure_depth += 1;
                            for (pi, p) in params.iter().enumerate() {
                                let pt = if pi == 0 { elem_ty } else { None };
                                self.push_var(p.name.sym, p.span, pt, None);
                            }
                            self.check_block(body, true);
                            self.exit_scope(body.span);
                            self.scope_depth -= 1;
                            self.closure_depth -= 1;
                            arg_facts.push(ExprFacts::plain(None));
                        } else {
                            arg_facts.push(self.check_expr(a, false));
                        }
                    }
                    self.check_call_disjoint(&call_borrows);
                    return ExprFacts::plain(ret_ty);
                }

                for (i, a) in args.iter().enumerate() {
                    let p_offset = if method_sig.as_ref().is_some_and(|s| s.params.first().is_some_and(|p| p.is_self)) {
                        i + 1
                    } else {
                        i
                    };
                    let sig_param = method_sig.as_ref().and_then(|s| s.params.get(p_offset));
                    let arg_is_linear = self.ty_of(a.span).is_some_and(|t| self.env.is_linear(t));
                    let is_arg_move = !matches!(a.kind, ExprKind::Unary { op: UnaryOp::Borrow | UnaryOp::BorrowMut, .. })
                        && (sig_param.is_some_and(|p| !matches!(self.env.get(p.ty), Ty::Borrow { .. }))
                            || (sig_param.is_none() && arg_is_linear));
                    let af = self.check_expr(a, is_arg_move);
                    if let Some(bpath) = self.borrow_arg_path(a, &af) {
                        let bm = matches!(a.kind, ExprKind::Unary { op: UnaryOp::BorrowMut, .. });
                        call_borrows.push((bpath, bm, a.span));
                    }
                    arg_facts.push(af);
                }
                self.check_call_disjoint(&call_borrows);

                if mname == "as_bytes" && args.is_empty()
                    && recv_ty.is_some_and(|t| matches!(self.env.get(t), Ty::Str | Ty::Borrow { inner: crate::types::BorrowInner::Str, .. }))
                {
                    return ExprFacts::from_view_source(&rf, ret_ty, self.env);
                }

                if let Some(sig) = &method_sig
                    && sig.is_view
                    && let Ok(index) = view_source_index(self.env, sig)
                {
                    let has_receiver = sig.params.first().is_some_and(|p| p.is_self);
                    if has_receiver && index == 0 {
                        let mut result = ExprFacts::from_view_source(&rf, ret_ty, self.env);
                        if !rf.ty.is_some_and(|t| matches!(self.env.get(t), Ty::Borrow { .. }))
                            && matches!(self.env.get(sig.params[0].ty), Ty::Borrow { .. })
                        {
                            result.origin = self.address_origin(recv, rf.origin);
                        }
                        return result;
                    }
                    let source = arg_facts.get(index - usize::from(has_receiver));
                    if let Some(source) = source {
                        return ExprFacts::from_view_source(source, ret_ty, self.env);
                    }
                }

                if let Some(pool_sym) = rf.path.as_ref().map(|p| p.root) {
                    if mname == "push" || mname == "id_at" {
                        let ep = self.find_var_idx(pool_sym).map(|i| self.vars[i].pool_epoch).unwrap_or(0);
                        return ExprFacts {
                            path: None,
                            is_borrowing_val: false,
                            origin: rf.origin,
                            brand: Some((pool_sym, ep)),
                            ty: ret_ty,
                            extra_loans: Vec::new(),
                        };
                    }
                    if mname == "alloc"
                        && let Some(rp) = rf.path
                    {
                        return ExprFacts {
                            path: Some(rp),
                            is_borrowing_val: true,
                            origin: rf.origin,
                            brand: None,
                            ty: ret_ty,
                            extra_loans: Vec::new(),
                        };
                    }
                    if method_sig.is_none()
                        && matches!(mname.as_str(), "get" | "get_mut" | "at" | "at_mut" | "sub" | "sub_mut" | "as_slice" | "as_mut_slice")
                        && let Some(rp) = rf.path
                    {
                        return ExprFacts {
                            path: Some(rp),
                            is_borrowing_val: ret_ty.is_some_and(|t| self.env.contains_borrowing_value(t)),
                            origin: rf.origin,
                            brand: None,
                            ty: ret_ty,
                            extra_loans: Vec::new(),
                        };
                    }
                }
                if method_sig.is_none()
                    && mname == "get"
                    && recv_ty.is_some_and(|t| {
                        let inner = match self.env.get(t) {
                            Ty::Borrow { inner: crate::types::BorrowInner::Single(i), .. } => *i,
                            _ => t,
                        };
                        matches!(self.env.get(inner), Ty::Builtin(BuiltinKind::ArgsCap, _))
                    })
                {
                    return ExprFacts::from_view_source(&rf, ret_ty, self.env);
                }
                return ExprFacts::plain(ret_ty);
            }
        }

        let sig_opt = self.resolve_called_sig(callee);
        for (i, a) in args.iter().enumerate() {
            let param_sig = sig_opt.as_ref().and_then(|s| s.params.get(i));
            let arg_is_linear = self.ty_of(a.span).is_some_and(|t| self.env.is_linear(t));
            let is_by_val = param_sig.is_some_and(|p| {
                !matches!(self.env.get(p.ty), Ty::Borrow { .. })
                    && !matches!(p.modifier, Some(ParamModifier::Mut))
            }) || (param_sig.is_none() && arg_is_linear);
            let as_move = is_by_val
                && !matches!(a.kind, ExprKind::Unary { op: UnaryOp::Borrow | UnaryOp::BorrowMut, .. });
            let af = self.check_expr(a, as_move);
            if let Some(bpath) = self.borrow_arg_path(a, &af) {
                let bm = matches!(a.kind, ExprKind::Unary { op: UnaryOp::BorrowMut, .. });
                call_borrows.push((bpath, bm, a.span));
            }
            arg_facts.push(af);
        }

        self.check_call_disjoint(&call_borrows);

        if let Some(sig) = &sig_opt
            && sig.is_view
            && let Ok(index) = view_source_index(self.env, sig)
            && let Some(source) = arg_facts.get(index)
        {
            return ExprFacts::from_view_source(source, ret_ty.or(Some(sig.ret)), self.env);
        }

        let _ = span;
        ExprFacts::plain(ret_ty.or_else(|| sig_opt.map(|s| s.ret)))
    }

    fn resolve_method_sig(&self, recv_ty: TyId, name: Symbol) -> Option<FnSig> {
        let deref = match self.env.get(recv_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => recv_ty,
        };
        let target = match self.env.get(deref) {
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def) => MethodTarget::Def(*def),
            Ty::Builtin(bk, _) => MethodTarget::Builtin(*bk),
            _ => return None,
        };
        self.resolve_method_target(target, name)
    }

    fn resolve_method_target(&self, target: MethodTarget, name: Symbol) -> Option<FnSig> {
        for im in &self.env.impls {
            if im.target == target
                && let Some((_, _, sig)) = im.methods.iter().find(|(_, _, m)| m.name == name)
            {
                return Some(sig.clone());
            }
        }
        None
    }

    fn linear_elem_of_collection(&self, col_ty: TyId) -> Option<TyId> {
        let deref = match self.env.get(col_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => col_ty,
        };
        if let Ty::Builtin(BuiltinKind::SlotPool | BuiltinKind::LinearBuf | BuiltinKind::Buf, args) =
            self.env.get(deref)
            && let Some(GenericVal::Ty(el)) = args.first()
        {
            return Some(*el);
        }
        None
    }

    fn borrow_arg_path(&self, arg: &Expr, af: &ExprFacts) -> Option<AccessPath> {
        match &arg.kind {
            ExprKind::Unary { op: UnaryOp::Borrow | UnaryOp::BorrowMut, .. } => af.path.clone(),
            _ => {
                if let Some(t) = af.ty
                    && matches!(self.env.get(t), Ty::Borrow { .. })
                {
                    return af.path.clone();
                }
                None
            }
        }
    }

    fn check_call_disjoint(&mut self, borrows: &[(AccessPath, bool, Span)]) {
        for i in 0..borrows.len() {
            for j in (i + 1)..borrows.len() {
                let (p1, m1, _sp1) = &borrows[i];
                let (p2, m2, sp2) = &borrows[j];
                if (*m1 || *m2) && paths_conflict(p1, p2) {
                    self.diags.error(
                        ErrorCode::E0118,
                        *sp2,
                        format!(
                            "ZM-18: exclusivity violation — conflicting borrow of `{}` while `{}` is mutably borrowed",
                            p2.format(self.interner),
                            p1.format(self.interner)
                        ),
                    );
                }
            }
        }
    }

    fn check_access_against_loans(&mut self, path: &AccessPath, is_mut_access: bool, span: Span) {
        for loan in &self.loans {
            if !paths_conflict(&loan.path, path) {
                continue;
            }
            if loan.is_mut || is_mut_access {
                let holder_name = loan
                    .holder
                    .map(|h| self.interner.get(h))
                    .unwrap_or("<borrow>");
                self.diags.error(
                    ErrorCode::E0118,
                    span,
                    format!(
                        "ZM-18: cannot access `{}` while borrowed by `{}`",
                        path.format(self.interner),
                        holder_name
                    ),
                );
                return;
            }
        }
    }

    fn expr_has_local_root(&self, mut expr: &Expr) -> bool {
        loop {
            match &expr.kind {
                ExprKind::Ident(s) => return self.find_var_idx(*s).is_some(),
                ExprKind::SelfVal => return true,
                ExprKind::Field { base, .. }
                | ExprKind::Deref(base)
                | ExprKind::Index { base, .. }
                | ExprKind::Paren(base) => expr = base,
                _ => return false,
            }
        }
    }

    fn resolve_called_sig(&self, callee: &Expr) -> Option<FnSig> {
        let mut cur = callee;
        if let ExprKind::GenericApply { base, .. } = &cur.kind {
            cur = base;
        }
        match &cur.kind {
            ExprKind::Field { base, field } => match self.lookup_call_path(base)? {
                TopScopeTarget::Module(module) => match self.prog.modules[module.0 as usize].exports.get(&field.sym)?.target {
                    TopScopeTarget::Def(def) => self.env.fns.get(&def).cloned(),
                    _ => None,
                },
                TopScopeTarget::Def(def) => self.resolve_method_target(MethodTarget::Def(def), field.sym),
                _ => None,
            },
            _ => match self.lookup_call_path(cur)? {
                TopScopeTarget::Def(def) => self.env.fns.get(&def).cloned(),
                _ => None,
            },
        }
    }

    fn lookup_call_path(&self, expr: &Expr) -> Option<TopScopeTarget> {
        match &expr.kind {
            ExprKind::Ident(sym) if self.find_var_idx(*sym).is_none() => self.prog.lookup_in_module(self.mod_id, *sym),
            ExprKind::Field { base, field } => {
                let TopScopeTarget::Module(module) = self.lookup_call_path(base)? else {
                    return None;
                };
                Some(self.prog.modules[module.0 as usize].exports.get(&field.sym)?.target)
            }
            _ => None,
        }
    }

    fn bind_pattern_vars(&mut self, pat: &Pat, ty: Option<TyId>, brand: Option<(Symbol, u32)>, origin: ExprOrigin) {
        match &pat.kind {
            PatKind::Wildcard
            | PatKind::Null
            | PatKind::BoolLit(_)
            | PatKind::CharLit { .. }
            | PatKind::IntLit { .. }
            | PatKind::Error => {}
            PatKind::Ident(sym) => {
                self.push_var(*sym, pat.span, ty, brand);
                // Only immutable bindings retain provenance across control flow in this pass.
                self.set_borrow_origin(*sym, origin);
            }
            PatKind::Tuple(ps) => {
                let elems = ty.and_then(|t| match self.env.get(t) {
                    Ty::Tuple(es) => Some(es.clone()),
                    _ => None,
                });
                for (i, p) in ps.iter().enumerate() {
                    let et = elems.as_ref().and_then(|es| es.get(i).copied());
                    self.bind_pattern_vars(p, et, brand, origin);
                }
            }
            PatKind::Variant { args, .. } => {
                for a in args {
                    self.bind_pattern_vars(a, None, brand, origin);
                }
            }
            PatKind::Struct { path: spath, fields } => {
                let mut struct_def = None;
                if let Some(tp) = spath
                    && tp.segments.len() == 1
                    && let Some(TopScopeTarget::Def(d)) =
                        self.prog.lookup_in_module(self.mod_id, tp.segments[0].sym)
                {
                    struct_def = Some(d);
                } else if let Some(t) = ty
                    && let Ty::Struct { def, .. } = self.env.get(t)
                {
                    struct_def = Some(*def);
                }
                let s_meta = struct_def.and_then(|d| self.env.structs.get(&d));
                for fp in fields {
                    let fty = s_meta
                        .and_then(|m| m.fields.iter().find(|f| f.name == fp.name.sym))
                        .map(|f| f.ty);
                    if let Some(sub) = &fp.pat {
                        self.bind_pattern_vars(sub, fty, None, origin);
                    } else {
                        self.push_var(fp.name.sym, fp.name.span, fty, None);
                        self.set_borrow_origin(fp.name.sym, origin);
                    }
                }
            }
        }
    }
}
