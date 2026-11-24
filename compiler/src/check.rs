use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{
    self, AccessMode, AssignOp, BinaryOp, Block, CatchBody, CompilationUnit, DeclKind, DeferBody,
    ElseBranch, Expr, ExprKind, GenericArg, GenericArgKind, IfCond, IfExpr, JumpAction, MatchArm,
    MatchArmBody, ParamModifier, Pat, PatKind, RangeKind, Stmt, StmtKind, UnaryOp,
};
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{
    BuiltinKind, DefId, ModuleId, PrimKind, ResolvedProgram, TopDefKind, TopScopeTarget,
};
use crate::span::{Interner, Span, Symbol};
use crate::types::{
    eval_const_int, int_bounds, parse_int_literal, prim_to_ty_id_in_env, resolve_path_segments,
    resolve_type_expr, BuiltinErrorKind, ErrorSource, FnSig, GenericCtx, GenericVal, ImplRef,
    MethodTarget, PoolHandleKind, Ty, TyId, TypeEnv,
};

#[derive(Clone, Debug, Default)]
pub struct TypeTable {
    map: BTreeMap<(u32, u32, u32), TyId>,
    const_ints: BTreeMap<(u32, u32, u32), i128>,
    comptime_branches: BTreeMap<(u32, u32, u32), bool>,
    comptime_vals: BTreeMap<(u32, u32, u32), crate::comptime::CtVal>,
    call_generic_insts: BTreeMap<(u32, u32, u32), (crate::cfg::CfgFnKey, Vec<GenericVal>)>,
    dyn_coercions: BTreeMap<(u32, u32, u32), (DefId, TyId)>,
    dyn_calls: BTreeMap<(u32, u32, u32), (DefId, usize)>,
}

impl TypeTable {
    pub fn insert(&mut self, span: Span, ty: TyId) {
        self.map.insert((span.file.0, span.start, span.end), ty);
    }

    pub fn get(&self, span: Span) -> Option<TyId> {
        self.map.get(&(span.file.0, span.start, span.end)).copied()
    }

    pub fn insert_const_int(&mut self, span: Span, val: i128) {
        self.const_ints.insert((span.file.0, span.start, span.end), val);
    }

    pub fn get_const_int(&self, span: Span) -> Option<i128> {
        self.const_ints.get(&(span.file.0, span.start, span.end)).copied()
    }

    pub fn insert_comptime_branch(&mut self, span: Span, taken: bool) {
        self.comptime_branches.insert((span.file.0, span.start, span.end), taken);
    }

    pub fn get_comptime_branch(&self, span: Span) -> Option<bool> {
        self.comptime_branches.get(&(span.file.0, span.start, span.end)).copied()
    }

    pub fn insert_comptime_val(&mut self, span: Span, val: crate::comptime::CtVal) {
        self.comptime_vals.insert((span.file.0, span.start, span.end), val);
    }

    pub fn get_comptime_val(&self, span: Span) -> Option<&crate::comptime::CtVal> {
        self.comptime_vals.get(&(span.file.0, span.start, span.end))
    }

    pub fn insert_call_generic_inst(&mut self, span: Span, key: crate::cfg::CfgFnKey, args: Vec<GenericVal>) {
        self.call_generic_insts.insert((span.file.0, span.start, span.end), (key, args));
    }

    pub fn get_call_generic_inst(&self, span: Span) -> Option<&(crate::cfg::CfgFnKey, Vec<GenericVal>)> {
        self.call_generic_insts.get(&(span.file.0, span.start, span.end))
    }

    pub fn all_generic_insts(&self) -> Vec<(crate::cfg::CfgFnKey, Vec<GenericVal>)> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for pair in self.call_generic_insts.values() {
            if seen.insert(pair.clone()) {
                out.push(pair.clone());
            }
        }
        out
    }

    pub fn insert_dyn_coercion(&mut self, span: Span, iface: DefId, concrete: TyId) {
        self.dyn_coercions.insert((span.file.0, span.start, span.end), (iface, concrete));
    }

    pub fn get_dyn_coercion(&self, span: Span) -> Option<(DefId, TyId)> {
        self.dyn_coercions.get(&(span.file.0, span.start, span.end)).copied()
    }

    pub fn all_dyn_impl_pairs(&self) -> Vec<(DefId, TyId)> {
        let mut set = BTreeSet::new();
        for &pair in self.dyn_coercions.values() {
            set.insert(pair);
        }
        set.into_iter().collect()
    }

    pub fn insert_dyn_call(&mut self, span: Span, iface: DefId, method_idx: usize) {
        self.dyn_calls.insert((span.file.0, span.start, span.end), (iface, method_idx));
    }

    pub fn get_dyn_call(&self, span: Span) -> Option<(DefId, usize)> {
        self.dyn_calls.get(&(span.file.0, span.start, span.end)).copied()
    }
}

pub fn check_program(
    units: &[CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> TypeTable {
    let mut table = TypeTable::default();
    // Check interface conformance on `impl Iface for Type` blocks (`E0501`).
    check_impl_conformance(prog, env, interner, diags);

    // Check top-level definitions (`Fn`, `Sys`, `Const`).
    for i in 0..prog.defs.len() {
        let def_mod = prog.defs[i].module;
        let def_kind = prog.defs[i].kind.clone();
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else {
            continue;
        };
        match def_kind {
            TopDefKind::Fn(decl_idx) => {
                let Some(sig) = env.fns.get(&DefId(i as u32)).cloned() else {
                    continue;
                };
                if let DeclKind::Fn(f) = &units[ui].decls[decl_idx].kind {
                    let mut checker = FnChecker::new(
                        def_mod,
                        &sig,
                        false,
                        units,
                        prog,
                        env,
                        &mut table,
                        interner,
                        diags,
                    );
                    checker.check_fn_body(f);
                }
            }
            TopDefKind::Sys(decl_idx) => {
                let Some(sig) = env.fns.get(&DefId(i as u32)).cloned() else {
                    continue;
                };
                if let DeclKind::Sys(s) = &units[ui].decls[decl_idx].kind {
                    let mut checker = FnChecker::new(
                        def_mod,
                        &sig,
                        false,
                        units,
                        prog,
                        env,
                        &mut table,
                        interner,
                        diags,
                    );
                    checker.check_sys_body(s);
                }
            }
            TopDefKind::Const(decl_idx) => {
                let Some(meta) = env.consts.get(&DefId(i as u32)).cloned() else {
                    continue;
                };
                if let DeclKind::Const(c) = &units[ui].decls[decl_idx].kind {
                    let dummy_sig = FnSig {
                        def: None,
                        module: def_mod,
                        name: c.name.sym,
                        is_pub: units[ui].decls[decl_idx].is_pub,
                        is_view: false,
                        is_comptime: true,
                        is_extern: false,
                        generics: Vec::new(),
                        params: Vec::new(),
                        ret: meta.ty,
                        has_gpu_attr: false,
                        effects: crate::types::EffectSet::default(),
                    };
                    let mut checker = FnChecker::new(
                        def_mod,
                        &dummy_sig,
                        false,
                        units,
                        prog,
                        env,
                        &mut table,
                        interner,
                        diags,
                    );
                    checker.in_comptime = true;
                    let got = checker.check_expr(&c.value, Some(meta.ty));
                    checker.expect_assignable(got, meta.ty, c.value.span);
                }
            }
            _ => {}
        }
    }

    // Check `impl` method bodies.
    let impls: Vec<ImplRef> = env.impls.clone();
    for im in &impls {
        let Some(ui) = prog.modules[im.module.0 as usize].unit_idx else {
            continue;
        };
        let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
            continue;
        };
        let in_iface_impl = im.iface.is_some();
        let target_ty = target_key_to_ty(im.target, env);
        let impl_generics: Vec<(Symbol, u32, bool)> = impl_decl
            .generics
            .iter()
            .enumerate()
            .map(|(idx, gp)| (gp.name.sym, idx as u32, gp.is_comptime))
            .collect();

        for &(_, sub_idx, ref sig) in &im.methods {
            if let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind {
                let mut checker = FnChecker::new(
                    im.module,
                    sig,
                    in_iface_impl,
                    units,
                    prog,
                    env,
                    &mut table,
                    interner,
                    diags,
                );
                checker.generic_ctx.self_ty = target_ty;
                let base_len = checker.generic_ctx.params.len() as u32;
                for (name, idx, is_ct) in &impl_generics {
                    if !checker.generic_ctx.params.iter().any(|(n, _, _)| n == name) {
                        checker.generic_ctx.params.push((*name, base_len + *idx, *is_ct));
                    }
                }
                checker.check_fn_body(f);
            }
        }
    }

    // ZM-01: report unused imports and unused private functions.
    prog.check_unused_module_items(diags);
    table
}

fn target_key_to_ty(target: MethodTarget, env: &TypeEnv) -> Option<TyId> {
    match target {
        MethodTarget::Def(d) => {
            if env.structs.contains_key(&d) {
                Some(env.structs[&d].def).map(|_| TyId::ERROR) // resolved per method self param already
            } else {
                None
            }
        }
        MethodTarget::Builtin(_) => None,
    }
}

fn check_impl_conformance(
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    for im in &env.impls {
        let Some(iface_id) = im.iface else {
            continue;
        };
        let Some(iface) = env.ifaces.get(&iface_id) else {
            continue;
        };
        let impl_span = prog.impls.iter().find(|x| x.decl_idx == im.impl_decl_idx && x.module == im.module).map(|x| x.span).unwrap_or(Span::DUMMY);
        for req in &iface.methods {
            let found = im.methods.iter().find(|(_, _, m)| m.name == req.name);
            let Some((_, _, impl_m)) = found else {
                diags.error(
                    ErrorCode::E0501,
                    impl_span,
                    format!(
                        "missing interface method `{}` in `impl`",
                        interner.get(req.name)
                    ),
                );
                continue;
            };
            if impl_m.params.len() != req.params.len() {
                diags.error(
                    ErrorCode::E0501,
                    impl_span,
                    format!(
                        "method `{}` parameter count mismatch with interface (expected {}, found {})",
                        interner.get(req.name),
                        req.params.len(),
                        impl_m.params.len()
                    ),
                );
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct LocalId(usize);

#[derive(Clone, Debug)]
struct LocalInfo {
    name: Symbol,
    name_str: String,
    span: Span,
    ty: TyId,
    is_mut: bool,
    is_param: bool,
    is_self: bool,
    is_linear: bool,
    is_copy_val: bool,
    used: bool,
    mutated: bool,
    moved: Option<Span>,
}

struct FnChecker<'a> {
    mod_id: ModuleId,
    sig: FnSig,
    in_iface_impl: bool,
    units: &'a [CompilationUnit],
    prog: &'a mut ResolvedProgram,
    env: &'a mut TypeEnv,
    table: &'a mut TypeTable,
    interner: &'a mut Interner,
    diags: &'a mut DiagBag,

    generic_ctx: GenericCtx,
    locals: Vec<LocalInfo>,
    scopes: Vec<Vec<LocalId>>,
    loop_stack: Vec<(Option<Symbol>, Vec<TyId>)>,
    in_comptime: bool,
    pool_handle_kind: Option<PoolHandleKind>,
}

impl<'a> FnChecker<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mod_id: ModuleId,
        sig: &FnSig,
        in_iface_impl: bool,
        units: &'a [CompilationUnit],
        prog: &'a mut ResolvedProgram,
        env: &'a mut TypeEnv,
        table: &'a mut TypeTable,
        interner: &'a mut Interner,
        diags: &'a mut DiagBag,
    ) -> Self {
        let mut generic_ctx = GenericCtx::empty();
        for (i, gp) in sig.generics.iter().enumerate() {
            generic_ctx.params.push((gp.name, i as u32, gp.is_comptime));
        }
        Self {
            mod_id,
            sig: sig.clone(),
            in_iface_impl,
            units,
            prog,
            env,
            table,
            interner,
            diags,
            generic_ctx,
            locals: Vec::new(),
            scopes: vec![Vec::new()],
            loop_stack: Vec::new(),
            in_comptime: sig.is_comptime,
            pool_handle_kind: None,
        }
    }

    fn bind_fn_params(&mut self) {
        let params = self.sig.params.clone();
        for p in params {
            let is_mut = matches!(p.modifier, Some(ParamModifier::Mut))
                || matches!(p.access, Some(AccessMode::Write));
            self.bind_local(p.name, p.span, p.ty, is_mut, true, p.is_comptime, p.is_self);
        }
    }

    fn check_fn_body(&mut self, f: &ast::FnDecl) {
        self.bind_fn_params();
        let diverged = self.check_block_stmts(&f.body);
        if let Some(tail) = &f.body.tail_expr {
            // Parser already emits E0124 for tail expr in fn/sys bodies, check for completeness.
            self.check_expr(tail, Some(self.sig.ret));
        } else if !diverged && self.needs_explicit_return(self.sig.ret) {
            self.diags.error(
                ErrorCode::E0230,
                f.body.span,
                format!(
                    "function `{}` must return a value of type `{}` on all control-flow paths",
                    self.interner.get(f.name.sym),
                    self.fmt_ty(self.sig.ret)
                ),
            );
        }
        self.pop_scope();
    }

    fn check_sys_body(&mut self, s: &ast::SysDecl) {
        self.bind_fn_params();
        let diverged = self.check_block_stmts(&s.body);
        if let Some(tail) = &s.body.tail_expr {
            self.check_expr(tail, Some(self.sig.ret));
        } else if !diverged && self.needs_explicit_return(self.sig.ret) {
            self.diags.error(
                ErrorCode::E0230,
                s.body.span,
                format!(
                    "system `{}` must return a value of type `{}`",
                    self.interner.get(s.name.sym),
                    self.fmt_ty(self.sig.ret)
                ),
            );
        }
        self.pop_scope();
    }

    fn needs_explicit_return(&self, ret: TyId) -> bool {
        if matches!(ret, TyId::VOID | TyId::NEVER | TyId::ERROR) {
            return false;
        }
        if let Ty::ErrorUnion { ok, .. } = self.env.get(ret)
            && *ok == TyId::VOID
        {
            return false;
        }
        true
    }

    fn fmt_ty(&self, ty: TyId) -> String {
        self.env.format_ty_named(ty, self.prog, self.interner)
    }

    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        let Some(ids) = self.scopes.pop() else {
            return;
        };
        for id in ids {
            let info = &self.locals[id.0];
            if info.name_str == "_" {
                continue;
            }
            if info.name_str.starts_with('_') {
                continue;
            }
            if info.is_self && self.in_iface_impl {
                continue;
            }
            if !info.used {
                let what = if info.is_param {
                    "parameter"
                } else {
                    "local variable"
                };
                self.diags.error(
                    ErrorCode::E0101,
                    info.span,
                    format!("ZM-01: unused {} `{}`", what, info.name_str),
                );
            } else if info.is_mut && !info.is_param && !info.mutated {
                self.diags.error(
                    ErrorCode::E0122,
                    info.span,
                    format!(
                        "`var` binding `{}` is never mutated; consider using `let`",
                        info.name_str
                    ),
                );
            }
        }
    }

    fn lookup_local(&self, sym: Symbol) -> Option<LocalId> {
        for scope in self.scopes.iter().rev() {
            for &id in scope.iter().rev() {
                if self.locals[id.0].name == sym {
                    return Some(id);
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_local(
        &mut self,
        sym: Symbol,
        span: Span,
        ty: TyId,
        is_mut: bool,
        is_param: bool,
        is_comptime: bool,
        is_self: bool,
    ) -> LocalId {
        let name_str = self.interner.get(sym).to_owned();
        if name_str != "_" {
            if let Some(prev_id) = self.lookup_local(sym) {
                let prev_name = &self.locals[prev_id.0].name_str;
                self.diags.error(
                    ErrorCode::E0103,
                    span,
                    format!("ZM-03: shadowing variable `{}` is forbidden", prev_name),
                );
            }
            if !self.in_comptime && self.env.contains_type_type(ty) && !is_comptime {
                self.diags.error(
                    ErrorCode::E0210,
                    span,
                    "cannot store `type` value in a runtime variable",
                );
            }
        }

        let is_linear = ty != TyId::ERROR && self.env.is_linear(ty);
        let is_copy_val = ty == TyId::ERROR || (!is_linear && self.env.is_trivial_copy(ty));
        let id = LocalId(self.locals.len());
        self.locals.push(LocalInfo {
            name: sym,
            name_str,
            span,
            ty,
            is_mut,
            is_param,
            is_self,
            is_linear,
            is_copy_val,
            used: false,
            mutated: false,
            moved: None,
        });
        if let Some(top) = self.scopes.last_mut() {
            top.push(id);
        }
        id
    }

    fn snapshot_moves(&self) -> Vec<Option<Span>> {
        self.locals.iter().map(|l| l.moved).collect()
    }

    fn restore_moves(&mut self, snap: &[Option<Span>]) {
        for (l, &m) in self.locals.iter_mut().zip(snap.iter()) {
            l.moved = m;
        }
    }

    fn merge_branch_moves(&mut self, base: &[Option<Span>], branch_results: &[(bool, Vec<Option<Span>>)]) {
        self.restore_moves(base);
        let non_div: Vec<&Vec<Option<Span>>> = branch_results
            .iter()
            .filter(|(div, _)| !div)
            .map(|(_, s)| s)
            .collect();
        if non_div.is_empty() {
            return;
        }
        for (idx, l) in self.locals.iter_mut().enumerate() {
            if let Some(m) = non_div.iter().find_map(|s| s.get(idx).and_then(|x| *x)) {
                l.moved = Some(m);
            }
        }
    }

    fn read_local(&mut self, id: LocalId, span: Span, as_move: bool) -> TyId {
        let (is_discard_named, is_linear, moved, is_copy, ty, name_str) = {
            let l = &mut self.locals[id.0];
            l.used = true;
            (
                l.name_str.starts_with('_') && l.name_str != "_",
                l.is_linear,
                l.moved,
                l.is_copy_val,
                l.ty,
                l.name_str.clone(),
            )
        };
        if is_discard_named {
            self.diags.error(
                ErrorCode::E0119,
                span,
                format!("cannot read `_`-prefixed binding `{}`", name_str),
            );
        }
        if !is_linear {
            if let Some(moved_span) = moved {
                let _ = moved_span;
                self.diags.error(
                    ErrorCode::E0301,
                    span,
                    format!("ZM-06: use of moved value `{}`", name_str),
                );
            } else if as_move && !is_copy {
                self.locals[id.0].moved = Some(span);
            }
        }
        ty
    }

    fn check_block_stmts(&mut self, block: &Block) -> bool {
        let mut diverged = false;
        for stmt in &block.stmts {
            if diverged {
                // Dead code after unconditional jump; still typecheck or stop? Typecheck for remaining diagnostics.
            }
            if self.check_stmt(stmt) {
                diverged = true;
            }
        }
        diverged
    }

    fn check_block_expr(&mut self, block: &Block, expected: Option<TyId>) -> (TyId, bool) {
        self.push_scope();
        let diverged = self.check_block_stmts(block);
        let ty = if let Some(tail) = &block.tail_expr {
            self.check_expr(tail, expected)
        } else if diverged {
            TyId::NEVER
        } else {
            TyId::VOID
        };
        self.pop_scope();
        (ty, diverged || ty == TyId::NEVER)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Let { attrs: _, pat, ty, init } => {
                let annot_ty = ty.as_ref().map(|t| {
                    let rt = self.resolve_ty(t);
                    self.table.insert(t.span, rt);
                    rt
                });
                let init_ty = self.check_expr_rvalue(init, annot_ty);
                let final_ty = if let Some(at) = annot_ty {
                    self.expect_assignable(init_ty, at, init.span);
                    at
                } else {
                    self.check_concrete_binding_ty(init_ty, init.span)
                };
                self.bind_pattern(pat, final_ty, false);
                false
            }
            StmtKind::Var { attrs: _, name, ty, init } => {
                let annot_ty = ty.as_ref().map(|t| {
                    let rt = self.resolve_ty(t);
                    self.table.insert(t.span, rt);
                    rt
                });
                let init_ty = self.check_expr_rvalue(init, annot_ty);
                let final_ty = if let Some(at) = annot_ty {
                    self.expect_assignable(init_ty, at, init.span);
                    at
                } else {
                    self.check_concrete_binding_ty(init_ty, init.span)
                };
                self.bind_local(name.sym, name.span, final_ty, true, false, false, false);
                false
            }
            StmtKind::Const(c) => {
                let c_ty = self.resolve_ty(&c.ty);
                let saved = self.in_comptime;
                self.in_comptime = true;
                let init_ty = self.check_expr(&c.value, Some(c_ty));
                self.expect_assignable(init_ty, c_ty, c.value.span);
                self.in_comptime = saved;
                self.bind_local(c.name.sym, c.name.span, c_ty, false, false, true, false);
                false
            }
            StmtKind::Defer(body) => {
                self.check_defer_body(body);
                false
            }
            StmtKind::Errdefer { err_bind, body } => {
                self.push_scope();
                if let Some(eb) = err_bind {
                    let ety = self.enclosing_error_set_ty().unwrap_or(TyId::ERROR);
                    self.bind_local(eb.sym, eb.span, ety, false, false, false, false);
                }
                self.check_defer_body(body);
                self.pop_scope();
                false
            }
            StmtKind::Jump(j) => {
                self.check_jump(j, stmt.span);
                true
            }
            StmtKind::If { is_comptime, data } => {
                let (_ty, div) = self.check_if(*is_comptime, data, None);
                div
            }
            StmtKind::Match { scrutinee, arms } => {
                let (_ty, div) = self.check_match(scrutinee, arms, None, stmt.span);
                div
            }
            StmtKind::While { label, cond, where_clauses: _, body } => {
                let cond_ty = self.check_expr(cond, Some(TyId::BOOL));
                if cond_ty != TyId::ERROR && cond_ty != TyId::BOOL {
                    self.diags.error(
                        ErrorCode::E0230,
                        cond.span,
                        format!("`while` condition must be `bool`, found `{}`", self.fmt_ty(cond_ty)),
                    );
                }
                let lbl = label.map(|l| l.sym);
                self.loop_stack.push((lbl, Vec::new()));
                self.push_scope();
                self.check_block_stmts(body);
                if let Some(tail) = &body.tail_expr {
                    let t = self.check_expr(tail, None);
                    if !matches!(t, TyId::VOID | TyId::NEVER | TyId::ERROR) {
                        self.diags.error(
                            ErrorCode::E0102,
                            tail.span,
                            format!("ZM-02: unused expression value of type `{}`", self.fmt_ty(t)),
                        );
                    }
                }
                self.pop_scope();
                self.loop_stack.pop();
                false
            }
            StmtKind::For { label, pat, iter_start, iter_end, body } => {
                let elem_ty = self.check_for_range(iter_start, iter_end.as_ref());
                let lbl = label.map(|l| l.sym);
                self.loop_stack.push((lbl, Vec::new()));
                self.push_scope();
                self.bind_pattern(pat, elem_ty, false);
                self.check_block_stmts(body);
                self.pop_scope();
                self.loop_stack.pop();
                false
            }
            StmtKind::ParFor { pat, iter_start, iter_end, body } => {
                let elem_ty = self.check_for_range(iter_start, iter_end.as_ref());
                self.table.insert(pat.span, elem_ty);
                self.loop_stack.push((None, Vec::new()));
                self.push_scope();
                self.bind_pattern(pat, elem_ty, false);
                self.check_block_stmts(body);
                self.pop_scope();
                self.loop_stack.pop();
                false
            }
            StmtKind::Loop { label, body } => {
                let lbl = label.map(|l| l.sym);
                self.loop_stack.push((lbl, Vec::new()));
                let base_moves = self.snapshot_moves();
                self.push_scope();
                self.check_block_stmts(body);
                self.pop_scope();
                self.restore_moves(&base_moves);
                let (_l, breaks) = self.loop_stack.pop().unwrap_or((None, Vec::new()));
                // `loop` diverges unless it has at least one `break`.
                breaks.is_empty()
            }
            StmtKind::ComptimeBlock(b) => {
                let saved = self.in_comptime;
                self.in_comptime = true;
                self.push_scope();
                let div = self.check_block_stmts(b);
                self.pop_scope();
                self.in_comptime = saved;
                div
            }
            StmtKind::RawBlock(b) => {
                self.push_scope();
                let div = self.check_block_stmts(b);
                self.pop_scope();
                div
            }
            StmtKind::Assign { lhs, op, rhs } => {
                let lhs_ty = self.check_lvalue_write(lhs, *op != AssignOp::Assign);
                let rhs_ty = self.check_expr_rvalue(rhs, Some(lhs_ty));
                match op {
                    AssignOp::Assign => {
                        self.expect_assignable(rhs_ty, lhs_ty, rhs.span);
                    }
                    AssignOp::ShlAssign | AssignOp::ShrAssign => {
                        if lhs_ty != TyId::ERROR && !self.env.is_integer(lhs_ty) {
                            self.diags.error(
                                ErrorCode::E0230,
                                lhs.span,
                                "shift assignment requires integer left operand",
                            );
                        }
                        if rhs_ty != TyId::ERROR && !self.env.is_integer(rhs_ty) {
                            self.diags.error(
                                ErrorCode::E0230,
                                rhs.span,
                                "shift amount must be an integer",
                            );
                        }
                    }
                    AssignOp::BitAndAssign | AssignOp::BitOrAssign | AssignOp::BitXorAssign => {
                        if lhs_ty != TyId::ERROR && !self.env.is_integer(lhs_ty) {
                            self.diags.error(
                                ErrorCode::E0230,
                                lhs.span,
                                "bitwise assignment requires integer operand",
                            );
                        }
                        self.expect_assignable(rhs_ty, lhs_ty, rhs.span);
                    }
                    _ => {
                        if lhs_ty != TyId::ERROR && !self.env.is_numeric(lhs_ty) {
                            self.diags.error(
                                ErrorCode::E0230,
                                lhs.span,
                                format!("arithmetic assignment requires numeric type, found `{}`", self.fmt_ty(lhs_ty)),
                            );
                        }
                        self.expect_assignable(rhs_ty, lhs_ty, rhs.span);
                    }
                }
                false
            }
            StmtKind::Discard(expr) => {
                let t = self.check_expr(expr, None);
                if t != TyId::ERROR && matches!(self.env.get(t), Ty::ErrorUnion { .. }) {
                    self.diags.error(
                        ErrorCode::E0102,
                        stmt.span,
                        "ZM-02: cannot discard an error union with `_ = ...`; handle with `try` or `catch`",
                    );
                }
                false
            }
            StmtKind::Expr(expr) => {
                let t = self.check_expr(expr, None);
                if !matches!(t, TyId::VOID | TyId::NEVER | TyId::ERROR) {
                    self.diags.error(
                        ErrorCode::E0102,
                        stmt.span,
                        format!(
                            "ZM-02: unused return value of type `{}`; bind it or explicitly discard with `_ = ...`",
                            self.fmt_ty(t)
                        ),
                    );
                }
                t == TyId::NEVER
            }
        }
    }

    fn check_concrete_binding_ty(&mut self, ty: TyId, span: Span) -> TyId {
        if self.in_comptime {
            return ty;
        }
        match self.env.get(ty) {
            Ty::ComptimeInt => {
                self.diags.error(
                    ErrorCode::E0203,
                    span,
                    "cannot infer concrete integer type for unsuffixed literal; add a type annotation or suffix like `_i32`",
                );
                TyId::ERROR
            }
            Ty::ComptimeFloat => {
                self.diags.error(
                    ErrorCode::E0203,
                    span,
                    "cannot infer concrete float type for unsuffixed literal; add a type annotation or suffix like `_f64`",
                );
                TyId::ERROR
            }
            _ => ty,
        }
    }

    fn check_defer_body(&mut self, body: &DeferBody) {
        let snap = self.snapshot_moves();
        match body {
            DeferBody::Block(b) => {
                self.push_scope();
                self.check_block_stmts(b);
                self.pop_scope();
            }
            DeferBody::Expr(e) => {
                let _ = self.check_expr(e, None);
            }
        }
        self.restore_moves(&snap);
    }

    fn check_jump(&mut self, j: &JumpAction, span: Span) {
        match j {
            JumpAction::Return(val_opt) => {
                let ret_ty = self.sig.ret;
                match val_opt {
                    None => {
                        let void_ok = matches!(ret_ty, TyId::VOID | TyId::ERROR)
                            || matches!(self.env.get(ret_ty), Ty::ErrorUnion { ok, .. } if *ok == TyId::VOID);
                        if !void_ok {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("expected return value of type `{}`", self.fmt_ty(ret_ty)),
                            );
                        }
                    }
                    Some(val) => {
                        if let Ty::ErrorUnion { ok, errors } = self.env.get(ret_ty).clone() {
                            if let ExprKind::EnumLit(ev) = &val.kind
                                && self.error_union_has_variant(&errors, ev.sym)
                            {
                                self.table.insert(val.span, ret_ty);
                                return;
                            }
                            let got = self.check_expr_rvalue(val, Some(ret_ty));
                            if got != ret_ty {
                                self.expect_assignable(got, ok, val.span);
                            }
                        } else {
                            let got = self.check_expr_rvalue(val, Some(ret_ty));
                            self.expect_assignable(got, ret_ty, val.span);
                        }
                    }
                }
            }
            JumpAction::Break { label, value } => {
                let target_idx = if let Some(lbl) = label {
                    self.loop_stack.iter().rposition(|(l, _)| *l == Some(lbl.sym))
                } else if !self.loop_stack.is_empty() {
                    Some(self.loop_stack.len() - 1)
                } else {
                    None
                };
                if target_idx.is_none() {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        "`break` outside of a loop or unknown loop label",
                    );
                }
                let v_ty = match value {
                    Some(v) => self.check_expr_rvalue(v, None),
                    None => TyId::VOID,
                };
                if let Some(idx) = target_idx {
                    self.loop_stack[idx].1.push(v_ty);
                }
            }
            JumpAction::Continue { label } => {
                let ok = if let Some(lbl) = label {
                    self.loop_stack.iter().any(|(l, _)| *l == Some(lbl.sym))
                } else {
                    !self.loop_stack.is_empty()
                };
                if !ok {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        "`continue` outside of a loop or unknown loop label",
                    );
                }
            }
            JumpAction::Unreachable => {}
        }
    }

    fn check_for_range(&mut self, start: &Expr, end: Option<&(RangeKind, Expr)>) -> TyId {
        if let Some((_, end_expr)) = end {
            let st = self.check_expr(start, None);
            let en = self.check_expr(end_expr, if st != TyId::COMPTIME_INT { Some(st) } else { None });
            if st == TyId::COMPTIME_INT && en == TyId::COMPTIME_INT {
                return TyId::USIZE;
            }
            if st == TyId::COMPTIME_INT && self.env.is_concrete_integer(en) {
                let _ = self.check_expr(start, Some(en));
                return en;
            }
            if st != TyId::ERROR && !self.env.is_integer(st) {
                self.diags.error(
                    ErrorCode::E0230,
                    start.span,
                    format!("range bound must be integer, found `{}`", self.fmt_ty(st)),
                );
                return TyId::ERROR;
            }
            self.expect_assignable(en, st, end_expr.span);
            st
        } else {
            let is_each_mut = matches!(
                &start.kind,
                ExprKind::Call { callee, args }
                    if args.is_empty()
                        && matches!(&callee.kind, ExprKind::Field { field, .. } if self.interner.get(field.sym) == "each_mut")
            );
            let is_each = matches!(
                &start.kind,
                ExprKind::Call { callee, args }
                    if args.is_empty()
                        && matches!(&callee.kind, ExprKind::Field { field, .. } if self.interner.get(field.sym) == "each")
            );
            let col_ty = self.check_expr(start, None);
            match self.env.get(col_ty).clone() {
                Ty::Borrow { inner: crate::types::BorrowInner::Slice(el), .. }
                | Ty::Array { elem: el, .. } => {
                    if is_each_mut {
                        self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Single(el),
                        })
                    } else if is_each {
                        self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Single(el),
                        })
                    } else {
                        el
                    }
                }
                Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str => TyId::U8,
                Ty::Builtin(BuiltinKind::SlotPool, gargs) => {
                    let elem = match gargs.first() {
                        Some(GenericVal::Ty(t)) => *t,
                        _ => TyId::ERROR,
                    };
                    self.env.intern(Ty::Builtin(BuiltinKind::GenId, vec![GenericVal::Ty(elem)]))
                }
                Ty::Borrow { inner: crate::types::BorrowInner::Single(inner), .. }
                    if matches!(self.env.get(inner), Ty::Builtin(BuiltinKind::SlotPool, _)) =>
                {
                    let elem = match self.env.get(inner) {
                        Ty::Builtin(BuiltinKind::SlotPool, gargs) => match gargs.first() {
                            Some(GenericVal::Ty(t)) => *t,
                            _ => TyId::ERROR,
                        },
                        _ => TyId::ERROR,
                    };
                    self.env.intern(Ty::Builtin(BuiltinKind::GenId, vec![GenericVal::Ty(elem)]))
                }
                Ty::Error => TyId::ERROR,
                _ => {
                    self.diags.error(
                        ErrorCode::E0230,
                        start.span,
                        format!("cannot iterate over `{}`", self.fmt_ty(col_ty)),
                    );
                    TyId::ERROR
                }
            }
        }
    }

    fn enclosing_error_set_ty(&mut self) -> Option<TyId> {
        match self.env.get(self.sig.ret).clone() {
            Ty::ErrorUnion { errors, .. } => errors.first().map(|s| self.env.intern(Ty::ErrorSet(*s))),
            _ => None,
        }
    }

    fn error_union_has_variant(&self, errors: &[ErrorSource], variant: Symbol) -> bool {
        let vname = self.interner.get(variant);
        for src in errors {
            match src {
                ErrorSource::User(d) => {
                    if let Some(meta) = self.env.errors.get(d)
                        && meta.variants.contains(&variant)
                    {
                        return true;
                    }
                }
                ErrorSource::Builtin(b) => {
                    let ok = match b {
                        BuiltinErrorKind::ArithError => matches!(vname, "Overflow" | "DivByZero" | "Underflow"),
                        BuiltinErrorKind::BoundsError => matches!(vname, "OutOfBounds"),
                        BuiltinErrorKind::Utf8Error => matches!(vname, "InvalidUtf8" | "UnexpectedEof"),
                        BuiltinErrorKind::OutOfMemory => matches!(vname, "OutOfMemory"),
                        BuiltinErrorKind::MemError => {
                            matches!(vname, "OutOfMemory" | "InvalidSize" | "BadAlignment" | "AllocFailed")
                        }
                        BuiltinErrorKind::StaleId => vname == "StaleId",
                        BuiltinErrorKind::IoError => matches!(
                            vname,
                            "NotFound" | "PermissionDenied" | "PathEscape" | "AlreadyExists" | "InvalidInput" | "Other"
                        ),
                    };
                    if ok {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn check_lvalue_write(&mut self, expr: &Expr, needs_prior_val: bool) -> TyId {
        let res = match &expr.kind {
            ExprKind::SelfVal => {
                let sym = self.interner.intern("self");
                if let Some(id) = self.lookup_local(sym) {
                    let (is_mut, is_self, ty, name) = {
                        let l = &mut self.locals[id.0];
                        l.used = true;
                        l.mutated = true;
                        (l.is_mut, l.is_self, l.ty, l.name_str.clone())
                    };
                    let mut_through_borrow = needs_prior_val
                        && matches!(
                            self.env.get(ty),
                            Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. }
                        );
                    let immut_borrow = needs_prior_val
                        && matches!(
                            self.env.get(ty),
                            Ty::Borrow { is_mut: false, .. } | Ty::RawPtr { is_mut: false, .. }
                        );
                    if immut_borrow {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot mutate through immutable borrow `{}`", name),
                        );
                    } else if !is_mut && !is_self && !mut_through_borrow {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot assign to immutable `{}`", name),
                        );
                    }
                    ty
                } else {
                    self.diags.error(ErrorCode::E0230, expr.span, "`self` is not in scope");
                    TyId::ERROR
                }
            }
            ExprKind::Ident(sym) => {
                if let Some(id) = self.lookup_local(*sym) {
                    let (is_mut, is_self, moved, ty, name) = {
                        let l = &mut self.locals[id.0];
                        l.used = true;
                        l.mutated = true;
                        (l.is_mut, l.is_self, l.moved, l.ty, l.name_str.clone())
                    };
                    let mut_through_borrow = needs_prior_val
                        && matches!(
                            self.env.get(ty),
                            Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. }
                        );
                    let immut_borrow = needs_prior_val
                        && matches!(
                            self.env.get(ty),
                            Ty::Borrow { is_mut: false, .. } | Ty::RawPtr { is_mut: false, .. }
                        );
                    if immut_borrow {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot mutate through immutable borrow `{}`", name),
                        );
                    } else if !is_mut && !is_self && !mut_through_borrow {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot assign to immutable `let` binding `{}`", name),
                        );
                    }
                    if needs_prior_val && moved.is_some() {
                        self.diags.error(
                            ErrorCode::E0301,
                            expr.span,
                            format!("ZM-06: use of moved value `{}`", name),
                        );
                    } else if !needs_prior_val {
                        // Re-assigning clears moved state.
                        self.locals[id.0].moved = None;
                    }
                    ty
                } else {
                    self.diags.error(
                        ErrorCode::E0230,
                        expr.span,
                        "cannot assign to non-local or undeclared identifier",
                    );
                    TyId::ERROR
                }
            }
            ExprKind::Field { base, field } => {
                let base_ty = self.check_lvalue_write(base, true);
                let brand = extract_root_ident_sym(base);
                self.resolve_field_type(base_ty, *field, true, brand, expr.span)
            }
            ExprKind::Index { base, index } => {
                let base_ty = self.check_lvalue_write(base, true);
                let idx_ty = self.check_expr(index, None);
                self.check_index_ty(base, idx_ty, index.span);
                self.index_elem_ty(base_ty, expr.span)
            }
            ExprKind::Deref(inner) => {
                let inner_ty = self.check_expr(inner, None);
                match self.env.get(inner_ty).clone() {
                    Ty::Borrow { is_mut: true, inner: crate::types::BorrowInner::Single(t) }
                    | Ty::RawPtr { is_mut: true, inner: t } => t,
                    Ty::Borrow { is_mut: false, .. } | Ty::RawPtr { is_mut: false, .. } => {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            "cannot assign through immutable borrow or pointer",
                        );
                        TyId::ERROR
                    }
                    Ty::Error => TyId::ERROR,
                    _ => {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot dereference `{}`", self.fmt_ty(inner_ty)),
                        );
                        TyId::ERROR
                    }
                }
            }
            ExprKind::Paren(inner) => self.check_lvalue_write(inner, needs_prior_val),
            _ => {
                self.diags.error(ErrorCode::E0230, expr.span, "invalid assignment target");
                TyId::ERROR
            }
        };
        self.table.insert(expr.span, res);
        res
    }

    fn check_expr_rvalue(&mut self, expr: &Expr, expected: Option<TyId>) -> TyId {
        // Explicit `move x` or ordinary non-trivial variable expression moves value on bind/call/return.
        let ty = match &expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(id) = self.lookup_local(*sym) {
                    self.read_local(id, expr.span, true)
                } else {
                    self.check_expr(expr, expected)
                }
            }
            ExprKind::Unary { op: UnaryOp::Move, expr: inner } => {
                if let ExprKind::Ident(sym) = &inner.kind
                    && let Some(id) = self.lookup_local(*sym)
                {
                    let t = self.read_local(id, inner.span, true);
                    if !self.locals[id.0].is_linear {
                        self.locals[id.0].moved = Some(expr.span);
                    }
                    t
                } else {
                    self.check_expr_rvalue(inner, expected)
                }
            }
            _ => self.check_expr(expr, expected),
        };
        self.table.insert(expr.span, ty);
        ty
    }

    fn check_expr(&mut self, expr: &Expr, expected: Option<TyId>) -> TyId {
        let ty = self.check_expr_inner(expr, expected);
        self.table.insert(expr.span, ty);
        ty
    }

    fn check_expr_inner(&mut self, expr: &Expr, expected: Option<TyId>) -> TyId {
        match &expr.kind {
            ExprKind::Error => TyId::ERROR,
            ExprKind::IntLit { raw, suffix } => self.check_int_lit(*raw, *suffix, false, expected, expr.span),
            ExprKind::FloatLit { raw, suffix } => self.check_float_lit(*raw, *suffix, expected, expr.span),
            ExprKind::StringLit(_) | ExprKind::RawStringLit(_) => {
                self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                })
            }
            ExprKind::ByteStringLit(bytes) => self.env.intern(Ty::Array {
                len: bytes.len() as u64,
                elem: TyId::U8,
            }),
            ExprKind::CharLit(_) => TyId::CHAR,
            ExprKind::ByteCharLit(_) => TyId::U8,
            ExprKind::BoolLit(_) => TyId::BOOL,
            ExprKind::Null => {
                if let Some(exp) = expected {
                    if matches!(self.env.get(exp), Ty::Optional(_)) {
                        return exp;
                    }
                    if let Ty::ErrorUnion { ok, .. } = self.env.get(exp)
                        && matches!(self.env.get(*ok), Ty::Optional(_))
                    {
                        return *ok;
                    }
                    if exp != TyId::ERROR {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("`null` cannot have non-optional type `{}`", self.fmt_ty(exp)),
                        );
                    }
                    return TyId::ERROR;
                }
                self.diags.error(
                    ErrorCode::E0230,
                    expr.span,
                    "`null` requires an optional `?T` type context",
                );
                TyId::ERROR
            }
            ExprKind::SelfVal => {
                let sym = self.interner.intern("self");
                if let Some(id) = self.lookup_local(sym) {
                    self.read_local(id, expr.span, false)
                } else {
                    self.generic_ctx.self_ty.unwrap_or_else(|| {
                        self.diags.error(ErrorCode::E0132, expr.span, "`self` is not bound here");
                        TyId::ERROR
                    })
                }
            }
            ExprKind::ReturnVal => self.sig.ret,
            ExprKind::Ident(sym) => {
                if let Some(id) = self.lookup_local(*sym) {
                    return self.read_local(id, expr.span, false);
                }
                // Fall back to module/prelude/type lookup.
                let seg = ast::IdentSpan {
                    span: expr.span,
                    sym: *sym,
                };
                self.resolve_value_path(&[seg], expr.span)
            }
            ExprKind::EnumLit(v) => self.check_enum_lit(*v, expected, expr.span),
            ExprKind::Paren(inner) => self.check_expr(inner, expected),
            ExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    return TyId::VOID;
                }
                let exp_elems = match expected.map(|e| self.env.get(e).clone()) {
                    Some(Ty::Tuple(ts)) if ts.len() == elems.len() => Some(ts),
                    _ => None,
                };
                let mut tys = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    let ex = exp_elems.as_ref().map(|ts| ts[i]);
                    tys.push(self.check_expr_rvalue(e, ex));
                }
                self.env.intern(Ty::Tuple(tys))
            }
            ExprKind::ArrayLit(elems) => {
                let exp_elem = match expected.map(|e| self.env.get(e).clone()) {
                    Some(Ty::Array { elem, .. }) => Some(elem),
                    Some(Ty::Borrow { inner: crate::types::BorrowInner::Slice(elem), .. }) => Some(elem),
                    _ => None,
                };
                let mut elem_ty = exp_elem;
                let mut inferred_tys = Vec::with_capacity(elems.len());
                for e in elems {
                    let t = self.check_expr_rvalue(e, elem_ty);
                    if elem_ty.is_none() && t != TyId::COMPTIME_INT && t != TyId::COMPTIME_FLOAT {
                        elem_ty = Some(t);
                    }
                    inferred_tys.push(t);
                }
                let final_elem = elem_ty.unwrap_or_else(|| {
                    if let Some(&first) = inferred_tys.first() {
                        self.check_concrete_binding_ty(first, expr.span)
                    } else {
                        TyId::ERROR
                    }
                });
                for (e, &t) in elems.iter().zip(&inferred_tys) {
                    if final_elem != TyId::ERROR {
                        let actual = if t == TyId::COMPTIME_INT || t == TyId::COMPTIME_FLOAT {
                            self.check_expr_rvalue(e, Some(final_elem))
                        } else {
                            t
                        };
                        self.expect_assignable(actual, final_elem, e.span);
                    }
                }
                self.env.intern(Ty::Array {
                    len: elems.len() as u64,
                    elem: final_elem,
                })
            }
            ExprKind::ArrayRepeat { val, count } => {
                let exp_elem = match expected.map(|e| self.env.get(e).clone()) {
                    Some(Ty::Array { elem, .. }) => Some(elem),
                    _ => None,
                };
                let val_ty = self.check_expr_rvalue(val, exp_elem);
                let final_elem = if let Some(el) = exp_elem {
                    self.expect_assignable(val_ty, el, val.span);
                    el
                } else {
                    self.check_concrete_binding_ty(val_ty, val.span)
                };
                let c_ty = self.check_expr(count, Some(TyId::USIZE));
                if c_ty != TyId::ERROR && !self.env.is_integer(c_ty) {
                    self.diags.error(ErrorCode::E0230, count.span, "array repeat count must be an integer");
                }
                let n = eval_const_int(
                    count,
                    self.mod_id,
                    &self.generic_ctx,
                    self.units,
                    self.prog,
                    self.env,
                    self.interner,
                )
                .unwrap_or(0)
                .max(0) as u64;
                self.env.intern(Ty::Array { len: n, elem: final_elem })
            }
            ExprKind::AnonStructLit(fields) => self.check_anon_struct_lit(fields, expected, expr.span),
            ExprKind::NamedStructLit { target, fields } => self.check_named_struct_lit(target, fields, expected, expr.span),
            ExprKind::Unary { op, expr: inner } => {
                if *op == UnaryOp::Neg
                    && let ExprKind::IntLit { raw, suffix } = &inner.kind
                {
                    return self.check_int_lit(*raw, *suffix, true, expected, expr.span);
                }
                self.check_unary(*op, inner, expected, expr.span)
            }
            ExprKind::Binary { op, lhs, rhs } => self.check_binary(*op, lhs, rhs, expected, expr.span),
            ExprKind::If { is_comptime, data } => self.check_if(*is_comptime, data, expected).0,
            ExprKind::Match { scrutinee, arms } => self.check_match(scrutinee, arms, expected, expr.span).0,
            ExprKind::Loop { label, body } => {
                let lbl = label.map(|l| l.sym);
                self.loop_stack.push((lbl, Vec::new()));
                let base_moves = self.snapshot_moves();
                self.push_scope();
                self.check_block_stmts(body);
                self.pop_scope();
                self.restore_moves(&base_moves);
                let (_, breaks) = self.loop_stack.pop().unwrap_or((None, Vec::new()));
                breaks.first().copied().unwrap_or(TyId::NEVER)
            }
            ExprKind::ComptimeBlock(b) => {
                let saved = self.in_comptime;
                self.in_comptime = true;
                let (ty, _) = self.check_block_expr(b, expected);
                self.in_comptime = saved;
                if !saved
                    && self.sig.generics.is_empty()
                    && let Some(val) = crate::comptime::eval_expr(
                        expr,
                        Some(ty),
                        self.mod_id,
                        &self.generic_ctx,
                        self.units,
                        self.prog,
                        self.env,
                        self.interner,
                        self.diags,
                        crate::comptime::DEFAULT_COMPTIME_FUEL,
                    )
                {
                    if let Some(iv) = val.as_i128() {
                        self.table.insert_const_int(expr.span, iv);
                    }
                    self.table.insert_comptime_val(expr.span, val);
                }
                ty
            }
            ExprKind::ComptimeExpr(e) => {
                let is_schedule_build = matches!(
                    &e.kind,
                    ExprKind::Call { callee, .. }
                        if matches!(
                            &callee.kind,
                            ExprKind::Field { base, field }
                                if matches!(&base.kind, ExprKind::Ident(s) if self.interner.get(*s) == "Schedule")
                                    && self.interner.get(field.sym) == "build"
                        )
                );
                let saved = self.in_comptime;
                self.in_comptime = true;
                let ty = self.check_expr(e, expected);
                self.in_comptime = saved;
                if !is_schedule_build
                    && !saved
                    && self.sig.generics.is_empty()
                    && let Some(val) = crate::comptime::eval_expr(
                        e,
                        Some(ty),
                        self.mod_id,
                        &self.generic_ctx,
                        self.units,
                        self.prog,
                        self.env,
                        self.interner,
                        self.diags,
                        crate::comptime::DEFAULT_COMPTIME_FUEL,
                    )
                {
                    if let Some(iv) = val.as_i128() {
                        self.table.insert_const_int(expr.span, iv);
                    }
                    self.table.insert_comptime_val(expr.span, val);
                }
                ty
            }
            ExprKind::Builtin { name, args } => self.check_builtin(*name, args, expected, expr.span),
            ExprKind::Is { expr: lhs, pat } => {
                let lt = self.check_expr(lhs, None);
                self.push_scope();
                self.bind_pattern(pat, lt, true);
                self.pop_scope();
                TyId::BOOL
            }
            ExprKind::Orelse { lhs, rhs } => self.check_orelse(lhs, rhs, expected),
            ExprKind::Catch { lhs, err_bind, rhs } => self.check_catch(lhs, *err_bind, rhs, expected),
            ExprKind::Field { base, field } => self.check_field_access(base, *field, expr.span),
            ExprKind::Deref(inner) => {
                let t = self.check_expr(inner, None);
                match self.env.get(t) {
                    Ty::Borrow { inner: crate::types::BorrowInner::Single(x), .. }
                    | Ty::RawPtr { inner: x, .. } => *x,
                    Ty::Error => TyId::ERROR,
                    _ => {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("cannot dereference value of type `{}`", self.fmt_ty(t)),
                        );
                        TyId::ERROR
                    }
                }
            }
            ExprKind::GenericApply { base, args } => self.check_generic_apply(base, args, expr.span),
            ExprKind::Call { callee, args } => self.check_call(callee, args, expected, expr.span),
            ExprKind::Index { base, index } => {
                let bt = self.check_expr(base, None);
                let it = self.check_expr(index, None);
                self.check_index_ty(base, it, index.span);
                self.index_elem_ty(bt, expr.span)
            }
            ExprKind::Slice { base, lo, hi } => {
                let bt = self.check_expr(base, None);
                if let Some(l) = lo {
                    let lt = self.check_expr(l, Some(TyId::USIZE));
                    if lt != TyId::ERROR && !self.env.is_integer(lt) {
                        self.diags.error(ErrorCode::E0230, l.span, "slice bound must be integer");
                    }
                }
                if let Some(h) = hi {
                    let ht = self.check_expr(h, Some(TyId::USIZE));
                    if ht != TyId::ERROR && !self.env.is_integer(ht) {
                        self.diags.error(ErrorCode::E0230, h.span, "slice bound must be integer");
                    }
                }
                self.slice_result_ty(bt, expr.span)
            }
            ExprKind::Closure { params, body } => self.check_closure(params, body, expected),
            ExprKind::Asm { operands, .. } => {
                for op in operands {
                    if let ast::AsmOperandKind::Named { arg, .. } = &op.kind
                        && let Some(e) = arg
                    {
                        self.check_expr(e, None);
                    } else if let ast::AsmOperandKind::Expr(e) = &op.kind {
                        self.check_expr(e, None);
                    }
                }
                TyId::VOID
            }
        }
    }

    fn check_int_lit(
        &mut self,
        raw: Symbol,
        suffix: Option<Symbol>,
        neg: bool,
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let raw_str = self.interner.get(raw).to_owned();
        let Some(mag) = parse_int_literal(&raw_str) else {
            self.diags.error(ErrorCode::E0201, span, "invalid integer literal");
            return TyId::ERROR;
        };

        if let Some(suf_sym) = suffix {
            let suf = self.interner.get(suf_sym).to_owned();
            let Some(ty) = self.prim_from_suffix(&suf) else {
                self.diags.error(ErrorCode::E0230, span, format!("unknown integer suffix `{}`", suf));
                return TyId::ERROR;
            };
            self.validate_int_fits(mag, neg, ty, span);
            return ty;
        }

        if let Some(exp) = expected
            && self.env.is_concrete_integer(exp)
        {
            self.validate_int_fits(mag, neg, exp, span);
            return exp;
        }
        TyId::COMPTIME_INT
    }

    fn prim_from_suffix(&mut self, suf: &str) -> Option<TyId> {
        let prim = match suf {
            "usize" => Some(PrimKind::USize),
            "isize" => Some(PrimKind::ISize),
            "u128" => Some(PrimKind::UInt(128)),
            "i8" => Some(PrimKind::IInt(8)),
            "i16" => Some(PrimKind::IInt(16)),
            "i32" => Some(PrimKind::IInt(32)),
            "i64" => Some(PrimKind::IInt(64)),
            "i128" => Some(PrimKind::IInt(128)),
            s if s.starts_with('u') => {
                let bits: u8 = s[1..].parse().ok()?;
                if (1..=64).contains(&bits) {
                    Some(PrimKind::UInt(bits))
                } else {
                    None
                }
            }
            s if s.starts_with('i') => {
                let bits: u8 = s[1..].parse().ok()?;
                if (1..=64).contains(&bits) {
                    Some(PrimKind::IInt(bits))
                } else {
                    None
                }
            }
            _ => None,
        }?;
        Some(prim_to_ty_id_in_env(prim, self.env))
    }

    fn validate_int_fits(&mut self, mag: u128, neg: bool, ty: TyId, span: Span) {
        let ty_val = self.env.get(ty).clone();
        match &ty_val {
            Ty::Int { signed: false, bits, is_ptr_sized } => {
                if neg {
                    self.diags.error(
                        ErrorCode::E0201,
                        span,
                        format!("negative literal cannot fit in unsigned type `{}`", self.fmt_ty(ty)),
                    );
                    return;
                }
                let actual_bits = if *is_ptr_sized { 64 } else { *bits };
                let fits = if actual_bits >= 128 {
                    true
                } else {
                    mag < (1u128 << actual_bits)
                };
                if !fits {
                    self.diags.error(
                        ErrorCode::E0201,
                        span,
                        format!("integer literal out of range for `{}`", self.fmt_ty(ty)),
                    );
                }
            }
            Ty::Int { signed: true, bits, is_ptr_sized } => {
                let actual_bits = if *is_ptr_sized { 64 } else { *bits };
                let max_pos: u128 = if actual_bits >= 128 {
                    i128::MAX as u128
                } else {
                    (1u128 << (actual_bits - 1)) - 1
                };
                let max_neg_mag: u128 = if actual_bits >= 128 {
                    (i128::MAX as u128) + 1
                } else {
                    1u128 << (actual_bits - 1)
                };
                let fits = if neg { mag <= max_neg_mag } else { mag <= max_pos };
                if !fits {
                    self.diags.error(
                        ErrorCode::E0201,
                        span,
                        format!("integer literal out of range for `{}`", self.fmt_ty(ty)),
                    );
                }
            }
            _ => {}
        }
    }

    fn check_float_lit(
        &mut self,
        raw: Symbol,
        suffix: Option<Symbol>,
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let raw_clean: String = self.interner.get(raw).chars().filter(|&c| c != '_').collect();
        let val: f64 = match raw_clean.parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => {
                self.diags.error(ErrorCode::E0202, span, "float literal overflows `f64`");
                return TyId::ERROR;
            }
        };
        let target = if let Some(s) = suffix {
            match self.interner.get(s) {
                "f16" => TyId::F16,
                "f32" => TyId::F32,
                "f64" => TyId::F64,
                other => {
                    self.diags.error(ErrorCode::E0230, span, format!("unknown float suffix `{}`", other));
                    return TyId::ERROR;
                }
            }
        } else if let Some(exp) = expected
            && self.env.is_float(exp)
            && exp != TyId::COMPTIME_FLOAT
        {
            exp
        } else {
            return TyId::COMPTIME_FLOAT;
        };

        if target == TyId::F32 && (val as f32).is_infinite() {
            self.diags.error(ErrorCode::E0202, span, "float literal overflows `f32`");
        } else if target == TyId::F16 && val.abs() > 65504.0 {
            self.diags.error(ErrorCode::E0202, span, "float literal overflows `f16`");
        }
        target
    }

    fn check_enum_lit(&mut self, v: ast::IdentSpan, expected: Option<TyId>, span: Span) -> TyId {
        let Some(exp) = expected else {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!("`.{}` requires an expected enum, union, or error type", self.interner.get(v.sym)),
            );
            return TyId::ERROR;
        };
        match self.env.get(exp).clone() {
            Ty::TypeType => TyId::TYPE,
            Ty::Enum(d) => {
                if let Some(meta) = self.env.enums.get(&d)
                    && meta.variants.iter().any(|(n, _)| *n == v.sym)
                {
                    return exp;
                }
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    format!("unknown enum variant `.{}` for `{}`", self.interner.get(v.sym), self.fmt_ty(exp)),
                );
                TyId::ERROR
            }
            Ty::Union { def, args } => {
                if let Some(um) = self.env.unions.get(&def)
                    && let Some(var) = um.variants.iter().find(|var| var.name == v.sym)
                {
                    if var.payload.is_some() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("union variant `.{}` requires a payload", self.interner.get(v.sym)),
                        );
                    }
                    let _ = args;
                    return exp;
                }
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    format!("unknown union variant `.{}` for `{}`", self.interner.get(v.sym), self.fmt_ty(exp)),
                );
                TyId::ERROR
            }
            Ty::ErrorSet(src) => {
                if self.error_union_has_variant(&[src], v.sym) {
                    exp
                } else {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("unknown error variant `.{}`", self.interner.get(v.sym)),
                    );
                    TyId::ERROR
                }
            }
            Ty::ErrorUnion { ok, errors } => {
                if self.error_union_has_variant(&errors, v.sym) {
                    exp
                } else if let Some(ok_ty) = self.try_check_enum_lit_in_ty(v, ok, span) {
                    ok_ty
                } else {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("unknown variant `.{}` for `{}`", self.interner.get(v.sym), self.fmt_ty(exp)),
                    );
                    TyId::ERROR
                }
            }
            Ty::Optional(inner) => {
                if let Some(inner_ty) = self.try_check_enum_lit_in_ty(v, inner, span) {
                    inner_ty
                } else {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("unknown variant `.{}` for `{}`", self.interner.get(v.sym), self.fmt_ty(exp)),
                    );
                    TyId::ERROR
                }
            }
            Ty::Builtin(BuiltinKind::Ordering, _) => {
                if matches!(
                    self.interner.get(v.sym),
                    "Less" | "Equal" | "Greater" | "Relaxed" | "Acquire" | "Release" | "AcqRel" | "SeqCst"
                ) {
                    exp
                } else {
                    self.diags.error(ErrorCode::E0230, span, "unknown `Ordering` variant");
                    TyId::ERROR
                }
            }
            Ty::Error => TyId::ERROR,
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    format!("`.{}` cannot construct `{}`", self.interner.get(v.sym), self.fmt_ty(exp)),
                );
                TyId::ERROR
            }
        }
    }

    fn try_check_enum_lit_in_ty(&mut self, v: ast::IdentSpan, target: TyId, span: Span) -> Option<TyId> {
        match self.env.get(target).clone() {
            Ty::Enum(d) => {
                if let Some(meta) = self.env.enums.get(&d)
                    && meta.variants.iter().any(|(n, _)| *n == v.sym)
                {
                    return Some(target);
                }
                None
            }
            Ty::Union { def, .. } => {
                if let Some(um) = self.env.unions.get(&def)
                    && let Some(var) = um.variants.iter().find(|var| var.name == v.sym)
                {
                    if var.payload.is_some() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("union variant `.{}` requires a payload", self.interner.get(v.sym)),
                        );
                    }
                    return Some(target);
                }
                None
            }
            Ty::Builtin(BuiltinKind::Ordering, _) => {
                if matches!(
                    self.interner.get(v.sym),
                    "Less" | "Equal" | "Greater" | "Relaxed" | "Acquire" | "Release" | "AcqRel" | "SeqCst"
                ) {
                    return Some(target);
                }
                None
            }
            Ty::Optional(inner) => self.try_check_enum_lit_in_ty(v, inner, span),
            _ => None,
        }
    }

    fn check_anon_struct_lit(
        &mut self,
        fields: &[ast::FieldInit],
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        if let Some(exp) = expected {
            match self.env.get(exp).clone() {
                Ty::Struct { def, args } => {
                    self.check_struct_fields_init(def, &args, fields, span);
                    return exp;
                }
                Ty::Pool { def, args } => {
                    self.check_pool_fields_init(def, &args, fields, span);
                    return exp;
                }
                Ty::Bitfield(def) => {
                    self.check_bitfield_fields_init(def, fields, span);
                    return exp;
                }
                Ty::Vec { elem, len } if fields.iter().all(|f| f.name.is_none()) => {
                    if fields.len() != len as usize {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("expected {} vector elements, found {}", len, fields.len()),
                        );
                    }
                    for f in fields {
                        let t = self.check_expr_rvalue(&f.value, Some(elem));
                        self.expect_assignable(t, elem, f.value.span);
                    }
                    return exp;
                }
                _ => {}
            }
        }
        let mut pairs = Vec::with_capacity(fields.len());
        for f in fields {
            let t = self.check_expr_rvalue(&f.value, None);
            let name = f.name.map(|n| n.sym).unwrap_or_else(|| self.interner.intern("_"));
            pairs.push((name, t));
        }
        self.env.intern(Ty::AnonStruct(pairs))
    }

    fn check_pool_fields_init(
        &mut self,
        pool_def: DefId,
        args: &[GenericVal],
        inits: &[ast::FieldInit],
        span: Span,
    ) {
        let Some(pmeta) = self.env.pools.get(&pool_def).cloned() else {
            return;
        };
        let mut seen = BTreeSet::new();
        for f in inits {
            let Some(fname) = f.name else {
                self.diags.error(
                    ErrorCode::E0230,
                    f.span,
                    "pool row initializers must specify column names",
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            seen.insert(fname.sym);
            let Some(col) = pmeta.columns.iter().find(|c| c.name == fname.sym) else {
                self.diags.error(
                    ErrorCode::E0132,
                    fname.span,
                    format!("pool has no column named `{}`", self.interner.get(fname.sym)),
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            let raw_cty = subst_ty(col.ty, args, self.env);
            let expected_cty = if let Some(kind) = self.pool_handle_kind {
                let pool_ty = self.env.intern(Ty::Pool {
                    def: pool_def,
                    args: args.to_vec(),
                });
                self.env.resolve_pool_link_ty(raw_cty, pool_ty, kind)
            } else {
                raw_cty
            };
            let got = self.check_expr_rvalue(&f.value, Some(expected_cty));
            self.expect_assignable(got, expected_cty, f.value.span);
        }
        for col in &pmeta.columns {
            if !seen.contains(&col.name) {
                self.diags.error(
                    ErrorCode::E0116,
                    span,
                    format!(
                        "ZM-16: missing pool column `{}` in row initializer",
                        self.interner.get(col.name)
                    ),
                );
            }
        }
    }

    fn check_named_struct_lit(
        &mut self,
        target: &Expr,
        fields: &[ast::FieldInit],
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let Some((def_id, mut args)) = self.resolve_type_target_expr(target) else {
            for f in fields {
                self.check_expr_rvalue(&f.value, None);
            }
            return TyId::ERROR;
        };
        if self.env.bitfields.contains_key(&def_id) {
            self.check_bitfield_fields_init(def_id, fields, span);
            return self.env.intern(Ty::Bitfield(def_id));
        }
        if !self.env.structs.contains_key(&def_id) {
            self.diags.error(
                ErrorCode::E0230,
                target.span,
                "struct literal requires a struct type",
            );
            return TyId::ERROR;
        }
        if args.is_empty()
            && let Some(exp) = expected
            && let Ty::Struct { def: edef, args: eargs } = self.env.get(exp).clone()
            && edef == def_id
        {
            args = eargs;
        }
        self.check_struct_fields_init(def_id, &args, fields, span);
        self.env.intern(Ty::Struct { def: def_id, args })
    }

    fn check_bitfield_fields_init(
        &mut self,
        def_id: DefId,
        inits: &[ast::FieldInit],
        _span: Span,
    ) {
        let Some(meta) = self.env.bitfields.get(&def_id).cloned() else {
            return;
        };
        let mut seen = BTreeSet::new();
        for f in inits {
            let Some(fname) = f.name else {
                self.diags.error(
                    ErrorCode::E0230,
                    f.span,
                    "bitfield literal initializers must specify member names",
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            if !seen.insert(fname.sym) {
                self.diags.error(
                    ErrorCode::E0132,
                    fname.span,
                    format!("duplicate bitfield member `{}` in initializer", self.interner.get(fname.sym)),
                );
            }
            let Some(mb) = meta.members.iter().find(|m| m.name == fname.sym) else {
                self.diags.error(
                    ErrorCode::E0132,
                    fname.span,
                    format!("bitfield has no member named `{}`", self.interner.get(fname.sym)),
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            let got = self.check_expr_rvalue(&f.value, Some(mb.ty));
            self.expect_assignable(got, mb.ty, f.value.span);
        }
    }

    fn check_struct_fields_init(
        &mut self,
        def_id: DefId,
        args: &[GenericVal],
        inits: &[ast::FieldInit],
        span: Span,
    ) {
        let Some(meta) = self.env.structs.get(&def_id).cloned() else {
            return;
        };
        let mut seen = BTreeSet::new();
        for f in inits {
            let Some(fname) = f.name else {
                self.diags.error(
                    ErrorCode::E0230,
                    f.span,
                    "struct literal initializers must specify field names",
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            if !seen.insert(fname.sym) {
                self.diags.error(
                    ErrorCode::E0132,
                    fname.span,
                    format!("duplicate field `{}` in struct literal", self.interner.get(fname.sym)),
                );
            }
            let Some(fmeta) = meta.fields.iter().find(|mf| mf.name == fname.sym) else {
                self.diags.error(
                    ErrorCode::E0132,
                    fname.span,
                    format!("struct has no field named `{}`", self.interner.get(fname.sym)),
                );
                let _ = self.check_expr_rvalue(&f.value, None);
                continue;
            };
            if !fmeta.is_pub && meta.module != self.mod_id {
                self.diags.error(
                    ErrorCode::E0133,
                    fname.span,
                    format!("field `{}` is private", self.interner.get(fname.sym)),
                );
            }
            let expected_fty = subst_ty(fmeta.ty, args, self.env);
            let got = self.check_expr_rvalue(&f.value, Some(expected_fty));
            self.expect_assignable(got, expected_fty, f.value.span);
        }
        for mf in &meta.fields {
            if !seen.contains(&mf.name) && !mf.has_default {
                self.diags.error(
                    ErrorCode::E0116,
                    span,
                    format!(
                        "ZM-16: missing struct field `{}` in initializer",
                        self.interner.get(mf.name)
                    ),
                );
            }
        }
    }

    fn resolve_type_target_expr(&mut self, expr: &Expr) -> Option<(DefId, Vec<GenericVal>)> {
        match &expr.kind {
            ExprKind::GenericApply { base, args } => {
                let (def_id, _) = self.resolve_type_target_expr(base)?;
                let mut gvals = Vec::with_capacity(args.len());
                for a in args {
                    gvals.push(self.eval_generic_arg(a));
                }
                Some((def_id, gvals))
            }
            _ => {
                let segs = self.extract_path_idents(expr)?;
                let target = resolve_path_segments(&segs, self.mod_id, self.prog, self.interner, self.diags)?;
                match target {
                    TopScopeTarget::Def(d) => Some((d, Vec::new())),
                    _ => {
                        self.diags.error(ErrorCode::E0230, expr.span, "expected user-defined type");
                        None
                    }
                }
            }
        }
    }

    fn extract_path_idents(&self, mut expr: &Expr) -> Option<Vec<ast::IdentSpan>> {
        let mut rev = Vec::new();
        loop {
            match &expr.kind {
                ExprKind::Ident(sym) => {
                    rev.push(ast::IdentSpan {
                        span: expr.span,
                        sym: *sym,
                    });
                    rev.reverse();
                    return Some(rev);
                }
                ExprKind::Field { base, field } => {
                    rev.push(*field);
                    expr = base;
                }
                _ => return None,
            }
        }
    }

    fn resolve_value_path(&mut self, segs: &[ast::IdentSpan], span: Span) -> TyId {
        // Primitive associated constant check: e.g. `u32.MAX`, `i64.MIN`.
        if segs.len() == 2 {
            let base_str = self.interner.get(segs[0].sym).to_owned();
            let mem_str = self.interner.get(segs[1].sym);
            if matches!(mem_str, "MIN" | "MAX")
                && let Some(prim_ty) = self.prim_from_suffix(&base_str)
            {
                return prim_ty;
            }
        }
        let Some(target) = resolve_path_segments(segs, self.mod_id, self.prog, self.interner, self.diags) else {
            return TyId::ERROR;
        };
        match target {
            TopScopeTarget::Def(d) => {
                if let Some(sig) = self.env.fns.get(&d) {
                    let params = sig.params.iter().map(|p| p.ty).collect();
                    let ret = sig.ret;
                    return self.env.intern(Ty::FnPtr { params, ret });
                }
                if let Some(c) = self.env.consts.get(&d) {
                    return c.ty;
                }
                if matches!(
                    self.prog.defs[d.0 as usize].kind,
                    TopDefKind::Struct(_) | TopDefKind::Enum(_) | TopDefKind::Union(_)
                ) {
                    return TyId::TYPE;
                }
                if matches!(self.prog.defs[d.0 as usize].kind, TopDefKind::Sys(_)) {
                    return TyId::VOID;
                }
                TyId::ERROR
            }
            TopScopeTarget::Builtin(bk) => match bk {
                BuiltinKind::Prim(_) | BuiltinKind::VecType | BuiltinKind::SimdAlias(_, _) => TyId::TYPE,
                _ => TyId::ERROR,
            },
            TopScopeTarget::Module(_) => {
                self.diags.error(ErrorCode::E0230, span, "expected value, found module");
                TyId::ERROR
            }
        }
    }

    fn check_unary(&mut self, op: UnaryOp, inner: &Expr, expected: Option<TyId>, span: Span) -> TyId {
        match op {
            UnaryOp::Neg => {
                let t = self.check_expr(inner, expected);
                if t == TyId::ERROR {
                    return TyId::ERROR;
                }
                let neg_ok = self.env.is_signed_integer(t)
                    || self.env.is_float(t)
                    || matches!(self.env.get(t), Ty::Vec { elem, .. } if self.env.is_signed_integer(*elem) || self.env.is_float(*elem));
                if !neg_ok {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("unary `-` requires signed numeric type, found `{}`", self.fmt_ty(t)),
                    );
                }
                t
            }
            UnaryOp::Not => {
                let t = self.check_expr(inner, Some(TyId::BOOL));
                if t != TyId::ERROR && t != TyId::BOOL {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("logical `!` requires `bool`, found `{}`", self.fmt_ty(t)),
                    );
                }
                TyId::BOOL
            }
            UnaryOp::BitNot => {
                let t = self.check_expr(inner, expected);
                if t != TyId::ERROR && !self.env.is_integer(t) {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("bitwise `~` requires integer type, found `{}`", self.fmt_ty(t)),
                    );
                }
                t
            }
            UnaryOp::Borrow | UnaryOp::BorrowMut => {
                let is_mut = op == UnaryOp::BorrowMut;
                self.check_borrow(inner, is_mut, expected, span)
            }
            UnaryOp::Move => self.check_expr_rvalue(inner, expected),
            UnaryOp::Copy => {
                let t = self.check_expr(inner, expected);
                if t != TyId::ERROR && self.env.is_linear(t) {
                    self.diags.error(
                        ErrorCode::E0320,
                        span,
                        format!("cannot `copy` a `linear` value of type `{}`", self.fmt_ty(t)),
                    );
                }
                t
            }
            UnaryOp::Try => {
                let t = self.check_expr(inner, None);
                match self.env.get(t).clone() {
                    Ty::ErrorUnion { ok, errors } => {
                        match self.env.get(self.sig.ret).clone() {
                            Ty::ErrorUnion { errors: ret_errs, .. } => {
                                let _ = ret_errs;
                                // Verify every error source could map into return error set when concrete.
                                for src in &errors {
                                    if !self.error_sources_compatible(*src, &ret_errs) {
                                        let src_ty = self.env.intern(Ty::ErrorSet(*src));
                                        self.diags.error(
                                            ErrorCode::E0230,
                                            span,
                                            format!(
                                                "`try` error `{}` cannot convert into function return type `{}`",
                                                self.fmt_ty(src_ty),
                                                self.fmt_ty(self.sig.ret)
                                            ),
                                        );
                                    }
                                }
                            }
                            Ty::Error => {}
                            _ => {
                                self.diags.error(
                                    ErrorCode::E0230,
                                    span,
                                    format!(
                                        "`try` used in function returning `{}` instead of an error union `... ! E`",
                                        self.fmt_ty(self.sig.ret)
                                    ),
                                );
                            }
                        }
                        ok
                    }
                    Ty::Error => TyId::ERROR,
                    _ => {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("`try` requires an error union `T ! E`, found `{}`", self.fmt_ty(t)),
                        );
                        TyId::ERROR
                    }
                }
            }
        }
    }

    fn error_sources_compatible(&self, src: ErrorSource, allowed: &[ErrorSource]) -> bool {
        if allowed.contains(&src) {
            return true;
        }
        if matches!(src, ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory))
            && allowed.contains(&ErrorSource::Builtin(BuiltinErrorKind::MemError))
        {
            return true;
        }
        if let ErrorSource::User(ud) = src
            && let Some(meta) = self.env.errors.get(&ud)
        {
            return meta
                .variants
                .iter()
                .all(|&v| self.error_union_has_variant(allowed, v));
        }
        false
    }

    fn check_borrow(&mut self, inner: &Expr, is_mut: bool, _expected: Option<TyId>, span: Span) -> TyId {
        match &inner.kind {
            ExprKind::Slice { base, lo, hi } => {
                if is_mut {
                    self.check_place_mutability(base, span);
                }
                let bt = self.check_expr(base, None);
                if let Some(l) = lo {
                    let lt = self.check_expr(l, Some(TyId::USIZE));
                    if lt != TyId::ERROR && !self.env.is_integer(lt) {
                        self.diags.error(ErrorCode::E0230, l.span, "slice bound must be integer");
                    }
                }
                if let Some(h) = hi {
                    let ht = self.check_expr(h, Some(TyId::USIZE));
                    if ht != TyId::ERROR && !self.env.is_integer(ht) {
                        self.diags.error(ErrorCode::E0230, h.span, "slice bound must be integer");
                    }
                }
                let raw_slice = self.slice_result_ty(bt, inner.span);
                match self.env.get(raw_slice).clone() {
                    Ty::Borrow { inner: crate::types::BorrowInner::Slice(el), .. } => self.env.intern(Ty::Borrow {
                        is_mut,
                        inner: crate::types::BorrowInner::Slice(el),
                    }),
                    other => self.env.intern(other),
                }
            }
            ExprKind::Field { base, field } => {
                if is_mut {
                    self.check_place_mutability(base, span);
                }
                let bt = self.check_expr(base, None);
                // Check `@packed` misaligned field borrow (`E0215`).
                let base_struct_ty = match self.env.get(bt) {
                    Ty::Borrow { inner: crate::types::BorrowInner::Single(s), .. } => *s,
                    _ => bt,
                };
                if let Ty::Struct { def, .. } = self.env.get(base_struct_ty).clone()
                    && let Some(meta) = self.env.structs.get(&def)
                    && meta.is_packed
                    && let Some(fmeta) = meta.fields.iter().find(|f| f.name == field.sym)
                    && fmeta.misaligned_in_packed
                {
                    self.diags.error(
                        ErrorCode::E0215,
                        span,
                        format!(
                            "cannot borrow misaligned field `{}` of `@packed` struct; copy it out by value instead",
                            self.interner.get(field.sym)
                        ),
                    );
                }
                let base_deref = match self.env.get(bt) {
                    Ty::Borrow { inner: crate::types::BorrowInner::Single(s), .. } => *s,
                    _ => bt,
                };
                let brand = extract_root_ident_sym(base);
                if matches!(self.env.get(base_deref), Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::SlotPool, _)) {
                    let fty = self.resolve_field_type(bt, *field, is_mut, brand, inner.span);
                    if let Ty::Borrow { inner: crate::types::BorrowInner::Slice(col_ty), .. } = self.env.get(fty).clone() {
                        return self.env.intern(Ty::Borrow {
                            is_mut,
                            inner: crate::types::BorrowInner::Slice(col_ty),
                        });
                    }
                    return fty;
                }
                let fty = self.resolve_field_type(bt, *field, is_mut, brand, inner.span);
                self.borrow_of(fty, is_mut)
            }
            _ => {
                if is_mut {
                    self.check_place_mutability(inner, span);
                }
                let t = self.check_expr(inner, None);
                if t == TyId::ERROR {
                    return TyId::ERROR;
                }
                self.borrow_of(t, is_mut)
            }
        }
    }

    fn borrow_of(&mut self, t: TyId, is_mut: bool) -> TyId {
        let inner = match self.env.get(t) {
            Ty::Str => crate::types::BorrowInner::Str,
            _ => crate::types::BorrowInner::Single(t),
        };
        self.env.intern(Ty::Borrow { is_mut, inner })
    }

    fn check_place_mutability(&mut self, expr: &Expr, span: Span) {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(id) = self.lookup_local(*sym) {
                    let (is_mut, ty, name) = {
                        let l = &mut self.locals[id.0];
                        l.used = true;
                        l.mutated = true;
                        (l.is_mut, l.ty, l.name_str.clone())
                    };
                    let mut_borrow = matches!(self.env.get(ty), Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. });
                    if !is_mut && !mut_borrow {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("cannot mutably borrow immutable binding `{}`", name),
                        );
                    }
                }
            }
            ExprKind::Field { base, .. } | ExprKind::Index { base, .. } | ExprKind::Paren(base) => {
                self.check_place_mutability(base, span);
            }
            _ => {}
        }
    }

    fn check_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        match op {
            BinaryOp::And | BinaryOp::Or => {
                let lt = self.check_expr(lhs, Some(TyId::BOOL));
                let rt = self.check_expr(rhs, Some(TyId::BOOL));
                if lt != TyId::ERROR && lt != TyId::BOOL {
                    self.diags.error(ErrorCode::E0230, lhs.span, "boolean operator requires `bool`");
                }
                if rt != TyId::ERROR && rt != TyId::BOOL {
                    self.diags.error(ErrorCode::E0230, rhs.span, "boolean operator requires `bool`");
                }
                TyId::BOOL
            }
            BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                let mut lt = self.check_expr(lhs, None);
                let hint = if !matches!(lt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT | TyId::ERROR) {
                    Some(lt)
                } else {
                    None
                };
                let rt = self.check_expr(rhs, hint);
                if matches!(lt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT)
                    && !matches!(rt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT | TyId::ERROR)
                {
                    lt = self.check_expr(lhs, Some(rt));
                }
                if lt != TyId::ERROR && rt != TyId::ERROR {
                    self.check_same_operand_types(lt, rt, span);
                }
                TyId::BOOL
            }
            BinaryOp::Shl | BinaryOp::Shr | BinaryOp::ShlWrap => {
                let lt = self.check_expr(lhs, expected);
                let rt = self.check_expr(rhs, None);
                if lt != TyId::ERROR && !self.env.is_integer(lt) {
                    self.diags.error(ErrorCode::E0230, lhs.span, "shift requires integer left operand");
                }
                if rt != TyId::ERROR && !self.env.is_integer(rt) {
                    self.diags.error(ErrorCode::E0230, rhs.span, "shift count must be an integer");
                }
                lt
            }
            BinaryOp::AddCheck
            | BinaryOp::SubCheck
            | BinaryOp::MulCheck
            | BinaryOp::DivCheck
            | BinaryOp::ModCheck => {
                let elem_exp = match expected.map(|e| self.env.get(e).clone()) {
                    Some(Ty::ErrorUnion { ok, .. }) => Some(ok),
                    _ => None,
                };
                let t = self.check_numeric_pair(op, lhs, rhs, elem_exp, span);
                self.env.intern(Ty::ErrorUnion {
                    ok: t,
                    errors: vec![ErrorSource::Builtin(BuiltinErrorKind::ArithError)],
                })
            }
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                let t = self.check_numeric_pair(op, lhs, rhs, expected, span);
                let bit_ok = self.env.is_integer(t)
                    || matches!(self.env.get(t), Ty::Vec { elem, .. } if self.env.is_integer(*elem));
                if t != TyId::ERROR && !bit_ok {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("bitwise operator requires integer type, found `{}`", self.fmt_ty(t)),
                    );
                }
                t
            }
            _ => self.check_numeric_pair(op, lhs, rhs, expected, span),
        }
    }

    fn check_numeric_pair(
        &mut self,
        _op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let mut lt = self.check_expr(lhs, expected);
        let rhs_hint = expected.or(
            if !matches!(lt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT | TyId::ERROR) {
                Some(lt)
            } else {
                None
            },
        );
        let mut rt = self.check_expr(rhs, rhs_hint);
        if matches!(lt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT)
            && !matches!(rt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT | TyId::ERROR)
        {
            lt = self.check_expr(lhs, Some(rt));
        }
        if lt == TyId::ERROR || rt == TyId::ERROR {
            return TyId::ERROR;
        }
        if self.env.is_address(lt) || self.env.is_address(rt) {
            self.diags.error(
                ErrorCode::E0211,
                span,
                format!(
                    "cannot perform direct arithmetic on address types `{}` and `{}`",
                    self.fmt_ty(lt),
                    self.fmt_ty(rt)
                ),
            );
            return TyId::ERROR;
        }
        if !self.check_same_operand_types(lt, rt, span) {
            return TyId::ERROR;
        }
        if lt == TyId::COMPTIME_INT && rt == TyId::COMPTIME_INT {
            return TyId::COMPTIME_INT;
        }
        if lt == TyId::COMPTIME_FLOAT && rt == TyId::COMPTIME_FLOAT {
            return TyId::COMPTIME_FLOAT;
        }
        if !self.env.is_numeric(lt) && !matches!(self.env.get(lt), Ty::Vec { .. }) {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!("operator requires numeric operands, found `{}`", self.fmt_ty(lt)),
            );
            return TyId::ERROR;
        }
        if matches!(rt, TyId::COMPTIME_INT | TyId::COMPTIME_FLOAT) {
            rt = lt;
        }
        let _ = rt;
        lt
    }

    fn check_same_operand_types(&mut self, lt: TyId, rt: TyId, span: Span) -> bool {
        if lt == rt {
            return true;
        }
        if (lt == TyId::COMPTIME_INT && self.env.is_concrete_integer(rt))
            || (rt == TyId::COMPTIME_INT && self.env.is_concrete_integer(lt))
            || (lt == TyId::COMPTIME_FLOAT && self.env.is_float(rt))
            || (rt == TyId::COMPTIME_FLOAT && self.env.is_float(lt))
        {
            return true;
        }
        if self.env.is_address(lt) || self.env.is_address(rt) {
            self.diags.error(
                ErrorCode::E0211,
                span,
                format!(
                    "incompatible address types `{}` and `{}`",
                    self.fmt_ty(lt),
                    self.fmt_ty(rt)
                ),
            );
            return false;
        }
        if self.env.is_numeric(lt) && self.env.is_numeric(rt) {
            self.diags.error(
                ErrorCode::E0105,
                span,
                format!(
                    "ZM-05: cannot mix `{}` and `{}` without an explicit conversion (.widen/.trunc/...)",
                    self.fmt_ty(lt),
                    self.fmt_ty(rt)
                ),
            );
            return false;
        }
        self.diags.error(
            ErrorCode::E0230,
            span,
            format!(
                "type mismatch between `{}` and `{}`",
                self.fmt_ty(lt),
                self.fmt_ty(rt)
            ),
        );
        false
    }

    fn check_if(&mut self, is_comptime: bool, data: &IfExpr, expected: Option<TyId>) -> (TyId, bool) {
        if is_comptime {
            self.push_scope();
            let taken = match &data.cond {
                IfCond::Expr(cond) => {
                    let saved = self.in_comptime;
                    self.in_comptime = true;
                    let ct = self.check_expr(cond, Some(TyId::BOOL));
                    self.in_comptime = saved;
                    if ct != TyId::ERROR && ct != TyId::BOOL {
                        self.diags.error(
                            ErrorCode::E0230,
                            cond.span,
                            format!("`comptime if` condition must be `bool`, found `{}`", self.fmt_ty(ct)),
                        );
                    }
                    crate::comptime::eval_const_bool(
                        cond,
                        self.mod_id,
                        &self.generic_ctx,
                        self.units,
                        self.prog,
                        self.env,
                        self.interner,
                        self.diags,
                    )
                    .unwrap_or(true)
                }
                IfCond::Let { .. } => false,
            };
            self.table.insert_comptime_branch(data.span, taken);
            let res = if taken {
                self.check_block_expr(&data.then_block, expected)
            } else if let Some(eb) = &data.else_branch {
                match eb {
                    ElseBranch::Block(b) => self.check_block_expr(b, expected),
                    ElseBranch::If { is_comptime: ic, data: nested } => {
                        self.check_if(*ic, nested, expected)
                    }
                }
            } else {
                (TyId::VOID, false)
            };
            self.pop_scope();
            return res;
        }
        self.push_scope();
        match &data.cond {
            IfCond::Expr(cond) => {
                let ct = self.check_expr(cond, Some(TyId::BOOL));
                if ct != TyId::ERROR && ct != TyId::BOOL {
                    self.diags.error(
                        ErrorCode::E0230,
                        cond.span,
                        format!("`if` condition must be `bool`, found `{}`", self.fmt_ty(ct)),
                    );
                }
            }
            IfCond::Let { pat, expr } => {
                let et = self.check_expr_rvalue(expr, None);
                let bound_ty = match self.env.get(et) {
                    Ty::Optional(inner) => *inner,
                    Ty::Error => TyId::ERROR,
                    _ => {
                        self.diags.error(
                            ErrorCode::E0230,
                            expr.span,
                            format!("`if let` requires optional `?T` or variant, found `{}`", self.fmt_ty(et)),
                        );
                        TyId::ERROR
                    }
                };
                self.bind_pattern(pat, bound_ty, true);
            }
        }

        let base_moves = self.snapshot_moves();
        let (then_ty, then_div) = self.check_block_expr(&data.then_block, expected);
        let then_moves = self.snapshot_moves();
        self.restore_moves(&base_moves);

        let (else_ty, else_div, else_moves) = if let Some(eb) = &data.else_branch {
            match eb {
                ElseBranch::Block(b) => {
                    let (t, d) = self.check_block_expr(b, expected.or(Some(then_ty)));
                    let m = self.snapshot_moves();
                    (t, d, m)
                }
                ElseBranch::If { is_comptime: ic, data: nested } => {
                    let (t, d) = self.check_if(*ic, nested, expected.or(Some(then_ty)));
                    let m = self.snapshot_moves();
                    (t, d, m)
                }
            }
        } else {
            (TyId::VOID, false, base_moves.clone())
        };

        self.merge_branch_moves(&base_moves, &[(then_div, then_moves), (else_div, else_moves)]);
        self.pop_scope();

        let res_ty = if then_ty == TyId::NEVER {
            else_ty
        } else if else_ty == TyId::NEVER {
            then_ty
        } else if data.else_branch.is_none() {
            if expected.is_some() && !matches!(then_ty, TyId::VOID | TyId::NEVER | TyId::ERROR) {
                self.diags.error(
                    ErrorCode::E0230,
                    data.span,
                    "`if` expression used as value must have an `else` branch",
                );
            }
            TyId::VOID
        } else {
            self.unify_branch_types(then_ty, else_ty, data.span)
        };
        (res_ty, then_div && else_div)
    }

    fn check_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        expected: Option<TyId>,
        span: Span,
    ) -> (TyId, bool) {
        let scrut_ty = self.check_expr(scrutinee, None);
        let base_moves = self.snapshot_moves();

        let mut cov = MatchCoverage::new(scrut_ty, self.env);
        let mut arm_types = Vec::new();
        let mut arm_results = Vec::new();
        let mut all_diverged = !arms.is_empty();

        for arm in arms {
            if cov.is_already_exhaustive() {
                self.diags.error(
                    ErrorCode::E0114,
                    arm.span,
                    "ZM-14: unreachable match arm",
                );
            } else if arm.guard.is_none() {
                cov.record_pattern(&arm.pat, self.env, self.interner, self.diags);
            }

            self.restore_moves(&base_moves);
            self.push_scope();
            let unwrap_opt = if matches!(self.env.get(scrut_ty), Ty::Optional(_)) && cov.seen_null {
                if let Ty::Optional(inner) = self.env.get(scrut_ty) {
                    *inner
                } else {
                    scrut_ty
                }
            } else {
                scrut_ty
            };
            self.bind_pattern(&arm.pat, unwrap_opt, true);
            if let Some(g) = &arm.guard {
                let gt = self.check_expr(g, Some(TyId::BOOL));
                if gt != TyId::ERROR && gt != TyId::BOOL {
                    self.diags.error(ErrorCode::E0230, g.span, "match guard must be `bool`");
                }
            }

            let exp_arm = expected.or_else(|| arm_types.first().copied());
            let (body_ty, arm_div) = match &arm.body {
                MatchArmBody::Block(b) => self.check_block_expr(b, exp_arm),
                MatchArmBody::Expr(e) => {
                    let t = self.check_expr_rvalue(e, exp_arm);
                    (t, t == TyId::NEVER)
                }
                MatchArmBody::Jump(j) => {
                    self.check_jump(j, arm.span);
                    (TyId::NEVER, true)
                }
            };
            self.pop_scope();
            all_diverged &= arm_div;
            arm_results.push((arm_div, self.snapshot_moves()));
            if body_ty != TyId::NEVER {
                arm_types.push(body_ty);
            }
        }

        self.merge_branch_moves(&base_moves, &arm_results);

        cov.verify_exhaustive(scrut_ty, self.env, self.interner, self.diags, span);

        if expected.is_none() && arms.iter().all(|a| matches!(a.body, MatchArmBody::Block(_))) && arm_types.iter().all(|&t| t == TyId::VOID) {
            return (TyId::VOID, all_diverged);
        }

        let Some(first) = arm_types.first().copied() else {
            return (TyId::NEVER, true);
        };
        let mut unified = first;
        for &t in &arm_types[1..] {
            unified = self.unify_branch_types(unified, t, span);
        }
        (unified, all_diverged)
    }

    fn check_orelse(&mut self, lhs: &Expr, rhs: &CatchBody, expected: Option<TyId>) -> TyId {
        let lt = self.check_expr_rvalue(lhs, None);
        let ok_ty = match self.env.get(lt) {
            Ty::Optional(inner) => *inner,
            Ty::Error => TyId::ERROR,
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    lhs.span,
                    format!("`orelse` requires an optional `?T`, found `{}`", self.fmt_ty(lt)),
                );
                TyId::ERROR
            }
        };
        let rt = self.check_catch_body(rhs, Some(ok_ty).or(expected), lhs.span);
        if rt != TyId::NEVER && ok_ty != TyId::ERROR {
            self.expect_assignable(rt, ok_ty, lhs.span);
        }
        ok_ty
    }

    fn check_catch(
        &mut self,
        lhs: &Expr,
        err_bind: Option<ast::IdentSpan>,
        rhs: &CatchBody,
        expected: Option<TyId>,
    ) -> TyId {
        let lt = self.check_expr_rvalue(lhs, None);
        let (ok_ty, err_sources) = match self.env.get(lt).clone() {
            Ty::ErrorUnion { ok, errors } => (ok, errors),
            Ty::Error => (TyId::ERROR, Vec::new()),
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    lhs.span,
                    format!("`catch` requires an error union `T ! E`, found `{}`", self.fmt_ty(lt)),
                );
                (TyId::ERROR, Vec::new())
            }
        };
        self.push_scope();
        if let Some(eb) = err_bind {
            let err_ty = err_sources
                .first()
                .map(|s| self.env.intern(Ty::ErrorSet(*s)))
                .unwrap_or(TyId::ERROR);
            self.bind_local(eb.sym, eb.span, err_ty, false, false, false, false);
        }
        let rt = self.check_catch_body(rhs, Some(ok_ty).or(expected), lhs.span);
        self.pop_scope();
        if rt != TyId::NEVER && ok_ty != TyId::ERROR {
            self.expect_assignable(rt, ok_ty, lhs.span);
        }
        ok_ty
    }

    fn check_catch_body(&mut self, body: &CatchBody, expected: Option<TyId>, span: Span) -> TyId {
        match body {
            CatchBody::Block(b) => self.check_block_expr(b, expected).0,
            CatchBody::Expr(e) => self.check_expr_rvalue(e, expected),
            CatchBody::Jump(j) => {
                self.check_jump(j, span);
                TyId::NEVER
            }
        }
    }

    fn check_builtin(&mut self, name: ast::IdentSpan, args: &[GenericArg], expected: Option<TyId>, span: Span) -> TyId {
        let bname = self.interner.get(name.sym).to_owned();
        match bname.as_str() {
            "sqrt" => {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, "`@sqrt` expects 1 float argument");
                    return TyId::ERROR;
                }
                let hint = expected.filter(|t| self.env.is_float(*t));
                let xt = if let GenericArgKind::Expr(e) = &args[0].kind {
                    self.check_expr_rvalue(e, hint)
                } else {
                    self.diags.error(ErrorCode::E0230, args[0].span, "`@sqrt` argument must be an expression");
                    return TyId::ERROR;
                };
                if xt == TyId::ERROR {
                    return TyId::ERROR;
                }
                if !self.env.is_float(xt) {
                    self.diags.error(ErrorCode::E0230, args[0].span, format!("`@sqrt` requires `f32` or `f64`, found `{}`", self.fmt_ty(xt)));
                    return TyId::ERROR;
                }
                if xt == TyId::COMPTIME_FLOAT {
                    return hint.unwrap_or(TyId::F64);
                }
                xt
            }
            "fma" => {
                if args.len() != 3 {
                    self.diags.error(ErrorCode::E0230, span, "`@fma` expects 3 float arguments (a, b, c)");
                    return TyId::ERROR;
                }
                let hint = expected.filter(|t| self.env.is_float(*t));
                let mut ts = Vec::with_capacity(3);
                for a in args {
                    if let GenericArgKind::Expr(e) = &a.kind {
                        ts.push((self.check_expr_rvalue(e, hint), a.span));
                    } else {
                        self.diags.error(ErrorCode::E0230, a.span, "`@fma` arguments must be expressions");
                        return TyId::ERROR;
                    }
                }
                let mut res_ty = hint.filter(|t| *t != TyId::COMPTIME_FLOAT).unwrap_or(TyId::ERROR);
                for &(t, sp) in &ts {
                    if t == TyId::ERROR {
                        return TyId::ERROR;
                    }
                    if !self.env.is_float(t) {
                        self.diags.error(ErrorCode::E0230, sp, format!("`@fma` requires `f32` or `f64`, found `{}`", self.fmt_ty(t)));
                        return TyId::ERROR;
                    }
                    if t != TyId::COMPTIME_FLOAT {
                        if res_ty == TyId::ERROR {
                            res_ty = t;
                        } else {
                            self.expect_assignable(t, res_ty, sp);
                        }
                    }
                }
                if res_ty == TyId::ERROR {
                    TyId::F64
                } else {
                    res_ty
                }
            }
            "lzcnt" | "tzcnt" | "popcnt" | "bswap" => {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, format!("`@{}` expects 1 integer argument", bname));
                    return TyId::ERROR;
                }
                let hint = expected.filter(|t| self.env.is_integer(*t));
                let xt = if let GenericArgKind::Expr(e) = &args[0].kind {
                    self.check_expr_rvalue(e, hint)
                } else {
                    self.diags.error(ErrorCode::E0230, args[0].span, format!("`@{}` argument must be an expression", bname));
                    return TyId::ERROR;
                };
                if xt == TyId::ERROR {
                    return TyId::ERROR;
                }
                if !self.env.is_integer(xt) {
                    self.diags.error(ErrorCode::E0230, args[0].span, format!("`@{}` requires an integer type, found `{}`", bname, self.fmt_ty(xt)));
                    return TyId::ERROR;
                }
                if xt == TyId::COMPTIME_INT {
                    return hint.unwrap_or(TyId::U64);
                }
                xt
            }
            "size_of" | "align_of" => {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, format!("`@{}` expects 1 type argument", bname));
                    return TyId::ERROR;
                }
                if let GenericVal::Ty(t) = self.eval_generic_arg(&args[0]) {
                    let lay = self.env.layout(t);
                    let v = if bname == "size_of" { lay.size } else { lay.align };
                    self.table.insert_const_int(span, v as i128);
                }
                TyId::USIZE
            }
            "type_info" => {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, "`@type_info` expects 1 type argument");
                    return TyId::ERROR;
                }
                let info_ty = crate::comptime::type_info_struct_ty(self.env, self.interner);
                if let GenericVal::Ty(t) = self.eval_generic_arg(&args[0]) {
                    let val = crate::comptime::make_type_info_val(t, self.env, self.interner);
                    self.table.insert_comptime_val(span, val);
                }
                info_ty
            }
            "target_arch" => {
                let sym = self.interner.intern(crate::comptime::target_arch_name());
                self.table.insert_comptime_val(span, crate::comptime::CtVal::EnumSym(sym, 0));
                TyId::TYPE
            }
            "target_os" => {
                let sym = self.interner.intern(crate::comptime::target_os_name());
                self.table.insert_comptime_val(span, crate::comptime::CtVal::EnumSym(sym, 0));
                TyId::TYPE
            }
            "comptime_budget" => {
                for a in args {
                    if let GenericArgKind::Expr(e) = &a.kind {
                        let _ = self.check_expr(e, Some(TyId::U64));
                    }
                }
                TyId::VOID
            }
            "compile_error" => {
                let msg = args
                    .first()
                    .and_then(|a| match &a.kind {
                        GenericArgKind::Expr(e) => match &e.kind {
                            ExprKind::StringLit(s) | ExprKind::RawStringLit(s) => Some(s.clone()),
                            _ => None,
                        },
                        _ => None,
                    })
                    .unwrap_or_else(|| "@compile_error triggered".into());
                if !self.in_comptime {
                    self.diags.error(ErrorCode::E0520, span, msg);
                }
                TyId::NEVER
            }
            "gpu_global_id" | "gpu_local_id" | "gpu_workgroup_id" => {
                self.env.intern(Ty::Vec { elem: TyId::U32, len: 3 })
            }
            "gpu_workgroup_barrier" => TyId::VOID,
            _ => {
                for a in args {
                    if let GenericArgKind::Expr(e) = &a.kind {
                        self.check_expr(e, None);
                    }
                }
                TyId::VOID
            }
        }
    }

    fn check_field_access(&mut self, base: &Expr, field: ast::IdentSpan, span: Span) -> TyId {
        // Check module path or primitive assoc constants (e.g., `u32.MAX`, `mod.CONST`).
        if let Some(segs) = self.extract_path_idents(base)
            && segs.iter().all(|s| self.lookup_local(s.sym).is_none())
        {
            let mut full = segs;
            full.push(field);
            if full.len() == 2 {
                let first_str = self.interner.get(full[0].sym).to_owned();
                let second_str = self.interner.get(full[1].sym);
                if matches!(second_str, "MIN" | "MAX")
                    && let Some(pt) = self.prim_from_suffix(&first_str)
                {
                    return pt;
                }
                if let Some(target) = self.prog.lookup_in_module(self.mod_id, full[0].sym) {
                    if let TopScopeTarget::Def(d) = target
                        && let Some(meta) = self.env.enums.get(&d)
                    {
                        self.prog.mark_module_binding_used(self.mod_id, full[0].sym);
                        self.prog.mark_def_used(d);
                        if meta.variants.iter().any(|(n, _)| *n == field.sym) {
                            return self.env.intern(Ty::Enum(d));
                        }
                    }
                    return self.resolve_value_path(&full, span);
                }
            } else {
                return self.resolve_value_path(&full, span);
            }
        }
        let bt = self.check_expr(base, None);
        let brand = extract_root_ident_sym(base);
        self.resolve_field_type(bt, field, false, brand, span)
    }

    fn resolve_field_type(
        &mut self,
        bt: TyId,
        field: ast::IdentSpan,
        _is_mut: bool,
        base_brand: Option<Symbol>,
        span: Span,
    ) -> TyId {
        let inner_ty = match self.env.get(bt) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => bt,
        };
        let fname = self.interner.get(field.sym).to_owned();
        match self.env.get(inner_ty).clone() {
            Ty::Error => TyId::ERROR,
            Ty::Struct { def, args } => {
                if let Some(meta) = self.env.structs.get(&def).cloned()
                    && let Some(f) = meta.fields.iter().find(|f| f.name == field.sym)
                {
                    if !f.is_pub && meta.module != self.mod_id {
                        self.diags.error(
                            ErrorCode::E0133,
                            field.span,
                            format!("field `{}` is private", fname),
                        );
                    }
                    return subst_ty(f.ty, &args, self.env);
                }
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("type `{}` has no field `{}`", self.fmt_ty(inner_ty), fname),
                );
                TyId::ERROR
            }
            Ty::Pool { def, args } => {
                if let Some(pmeta) = self.env.pools.get(&def).cloned()
                    && let Some(col) = pmeta.columns.iter().find(|c| c.name == field.sym)
                {
                    if !col.is_pub && pmeta.module != self.mod_id {
                        self.diags.error(
                            ErrorCode::E0133,
                            field.span,
                            format!("pool column `{}` is private", fname),
                        );
                    }
                    let sub_col = subst_ty(col.ty, &args, self.env);
                    return self.env.resolve_pool_link_ty(
                        sub_col,
                        inner_ty,
                        PoolHandleKind::SlotPool,
                    );
                }
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("pool `{}` has no column `{}`", self.fmt_ty(inner_ty), fname),
                );
                TyId::ERROR
            }
            Ty::AnonStruct(fields) => {
                if let Some((_, t)) = fields.iter().find(|(n, _)| *n == field.sym) {
                    return *t;
                }
                self.diags.error(ErrorCode::E0132, field.span, format!("unknown field `{}`", fname));
                TyId::ERROR
            }
            Ty::Tuple(elems) => {
                if let Ok(idx) = fname.parse::<usize>()
                    && let Some(&t) = elems.get(idx)
                {
                    return t;
                }
                self.diags.error(ErrorCode::E0132, field.span, format!("unknown tuple index `{}`", fname));
                TyId::ERROR
            }
            Ty::Bitfield(def) => {
                if let Some(meta) = self.env.bitfields.get(&def)
                    && let Some(m) = meta.members.iter().find(|m| m.name == field.sym)
                {
                    return m.ty;
                }
                self.diags.error(ErrorCode::E0132, field.span, format!("unknown bitfield member `{}`", fname));
                TyId::ERROR
            }
            Ty::Borrow { inner: crate::types::BorrowInner::Slice(_) | crate::types::BorrowInner::Str, .. }
            | Ty::Array { .. }
            | Ty::Str => {
                if fname == "len" {
                    return TyId::USIZE;
                }
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("type `{}` has no field `{}`", self.fmt_ty(inner_ty), fname),
                );
                TyId::ERROR
            }
            Ty::Vec { elem, len } => {
                // SIMD swizzle like `.x`..`.w`, `.xy`, `.zw`, `.yxwz`, `.r`..`.a`, `.rgba`
                let is_xyzw = fname.chars().all(|c| matches!(c, 'x' | 'y' | 'z' | 'w'));
                let is_rgba = fname.chars().all(|c| matches!(c, 'r' | 'g' | 'b' | 'a'));
                if !fname.is_empty() && (is_xyzw || is_rgba) {
                    let all_in_bounds = fname.chars().all(|c| {
                        let idx: u8 = match c {
                            'x' | 'r' => 0,
                            'y' | 'g' => 1,
                            'z' | 'b' => 2,
                            _ => 3,
                        };
                        idx < len
                    });
                    if all_in_bounds {
                        if fname.len() == 1 {
                            return elem;
                        }
                        let n = fname.len() as u8;
                        if (2..=4).contains(&n) {
                            return self.env.intern(Ty::Vec { elem, len: n });
                        }
                    }
                }
                self.diags.error(ErrorCode::E0132, field.span, format!("invalid vector field `{}`", fname));
                TyId::ERROR
            }
            Ty::Builtin(BuiltinKind::Env, _) => {
                let bk = match fname.as_str() {
                    "alloc" => Some(BuiltinKind::AllocCap),
                    "fs" => Some(BuiltinKind::FsCap),
                    "net" => Some(BuiltinKind::NetCap),
                    "clock" => Some(BuiltinKind::ClockCap),
                    "io" => Some(BuiltinKind::IoCap),
                    "args" => Some(BuiltinKind::ArgsCap),
                    _ => None,
                };
                if let Some(kind) = bk {
                    return self.env.intern(Ty::Builtin(kind, Vec::new()));
                }
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("`Env` has no capability field `{}`", fname),
                );
                TyId::ERROR
            }
            Ty::Builtin(bk @ (BuiltinKind::ArenaPool | BuiltinKind::SlotPool), args) => {
                let pool_is_mut = matches!(
                    self.env.get(bt),
                    Ty::Borrow { is_mut: true, .. }
                ) || !matches!(self.env.get(bt), Ty::Borrow { .. });
                if let Some(GenericVal::Ty(pool_ty)) = args.first().copied()
                    && let Ty::Pool { def: pool_def, args: pool_args } = self.env.get(pool_ty).clone()
                    && let Some(pmeta) = self.env.pools.get(&pool_def).cloned()
                    && let Some(col) = pmeta.columns.iter().find(|c| c.name == field.sym)
                {
                    if !col.is_pub && pmeta.module != self.mod_id {
                        self.diags.error(
                            ErrorCode::E0133,
                            field.span,
                            format!("pool column `{}` is private", fname),
                        );
                    }
                    let sub_col = subst_ty(col.ty, &pool_args, self.env);
                    let handle_kind = if bk == BuiltinKind::ArenaPool {
                        let brand = base_brand.unwrap_or_else(|| self.interner.intern("_pool"));
                        PoolHandleKind::ArenaPool(brand)
                    } else {
                        PoolHandleKind::SlotPool
                    };
                    let col_ty = self.env.resolve_pool_link_ty(sub_col, pool_ty, handle_kind);
                    return self.env.intern(Ty::Borrow {
                        is_mut: pool_is_mut,
                        inner: crate::types::BorrowInner::Slice(col_ty),
                    });
                }
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("pool `{}` has no column `{}`", self.fmt_ty(inner_ty), fname),
                );
                TyId::ERROR
            }
            _ => {
                let _ = span;
                self.diags.error(
                    ErrorCode::E0132,
                    field.span,
                    format!("type `{}` has no field `{}`", self.fmt_ty(inner_ty), fname),
                );
                TyId::ERROR
            }
        }
    }

    fn check_generic_apply(&mut self, base: &Expr, args: &[GenericArg], span: Span) -> TyId {
        if let Some(segs) = self.extract_path_idents(base)
            && let Some(TopScopeTarget::Def(d)) =
                resolve_path_segments(&segs, self.mod_id, self.prog, self.interner, self.diags)
            && let Some(sig) = self.env.fns.get(&d).cloned()
        {
            let gvals: Vec<GenericVal> = args.iter().map(|a| self.eval_generic_arg(a)).collect();
            let params = sig.params.iter().map(|p| subst_ty(p.ty, &gvals, self.env)).collect();
            let ret = subst_ty(sig.ret, &gvals, self.env);
            return self.env.intern(Ty::FnPtr { params, ret });
        }
        let _ = span;
        self.check_expr(base, None)
    }

    fn check_call(&mut self, callee: &Expr, args: &[Expr], expected: Option<TyId>, span: Span) -> TyId {
        // Case 0: Union variant constructor call like `.Click(payload)`
        if let ExprKind::EnumLit(v) = &callee.kind {
            let union_target = expected.and_then(|exp| match self.env.get(exp) {
                Ty::Union { .. } => Some(exp),
                Ty::Optional(inner) if matches!(self.env.get(*inner), Ty::Union { .. }) => Some(*inner),
                Ty::ErrorUnion { ok, .. } => match self.env.get(*ok) {
                    Ty::Union { .. } => Some(*ok),
                    Ty::Optional(inner) if matches!(self.env.get(*inner), Ty::Union { .. }) => Some(*inner),
                    _ => None,
                },
                _ => None,
            });
            if let Some(exp) = union_target
                && let Ty::Union { def, args: uargs } = self.env.get(exp).clone()
            {
                let v_payload = self.env.unions.get(&def).and_then(|um| {
                    um.variants
                        .iter()
                        .find(|uv| uv.name == v.sym)
                        .map(|uv| uv.payload)
                });
                match v_payload {
                    Some(Some(raw_pt)) => {
                        let pty = subst_ty(raw_pt, &uargs, self.env);
                        if args.len() != 1 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("variant `.{}` expects 1 payload argument", self.interner.get(v.sym)),
                            );
                        } else {
                            let got = self.check_expr_rvalue(&args[0], Some(pty));
                            self.expect_assignable(got, pty, args[0].span);
                        }
                        return exp;
                    }
                    Some(None) => {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("variant `.{}` takes no payload arguments", self.interner.get(v.sym)),
                        );
                        return exp;
                    }
                    None => {
                        self.diags.error(
                            ErrorCode::E0132,
                            v.span,
                            format!("unknown union variant `.{}`", self.interner.get(v.sym)),
                        );
                        return TyId::ERROR;
                    }
                }
            }
            for a in args {
                self.check_expr_rvalue(a, None);
            }
            self.diags.error(
                ErrorCode::E0230,
                callee.span,
                format!("`.{}` requires an expected union type", self.interner.get(v.sym)),
            );
            return TyId::ERROR;
        }

        // Case 1: Built-in generic conversion methods: `x.widen.[To]()`, `x.trunc.[To]()`, etc.
        if let ExprKind::GenericApply { base, args: gargs } = &callee.kind
            && let ExprKind::Field { base: recv, field } = &base.kind
        {
            let mname = self.interner.get(field.sym).to_owned();
            if matches!(
                mname.as_str(),
                "widen" | "trunc" | "clamp" | "try_into" | "bitcast" | "to_float" | "trunc_to_int"
            ) {
                return self.check_cast_method(&mname, recv, gargs, args, span);
            }
        }

        // Case 2: Method call `recv.method(args)` or `recv.method.[T](args)` vs module path call `mod.fn(args)`.
        let (method_recv_gargs, recv_expr, method_name) = match &callee.kind {
            ExprKind::GenericApply { base, args: ga } => {
                if let ExprKind::Field { base: r, field } = &base.kind {
                    if let Some(segs) = self.extract_path_idents(base)
                        && segs.first().is_some_and(|s| self.lookup_local(s.sym).is_none())
                        && matches!(
                            self.prog.lookup_in_module(self.mod_id, segs[0].sym),
                            Some(TopScopeTarget::Module(_))
                        )
                    {
                        return self.check_direct_fn_call(base, ga, args, span);
                    }
                    (ga.clone(), Some(r.as_ref()), Some(*field))
                } else {
                    return self.check_direct_fn_call(base, ga, args, span);
                }
            }
            ExprKind::Field { base: r, field } => {
                if let ExprKind::GenericApply { base: type_base, args: type_gargs } = &r.kind
                    && let Some(segs) = self.extract_path_idents(type_base)
                    && segs.first().is_some_and(|s| self.lookup_local(s.sym).is_none())
                    && let Some(target) =
                        resolve_path_segments(&segs, self.mod_id, self.prog, self.interner, self.diags)
                {
                    match target {
                        TopScopeTarget::Builtin(bk) => {
                            return self.check_builtin_assoc_call(bk, type_gargs, *field, args, expected, span);
                        }
                        TopScopeTarget::Def(type_def) => {
                            if let Some((impl_idx, sub_idx, sig)) =
                                self.find_type_method_with_key(MethodTarget::Def(type_def), field.sym)
                            {
                                self.prog.mark_module_binding_used(self.mod_id, segs[0].sym);
                                self.prog.mark_def_used(type_def);
                                let gvals: Vec<GenericVal> =
                                    type_gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
                                let fn_key = crate::cfg::CfgFnKey::ImplMethod { impl_idx, sub_idx };
                                return self.check_fn_sig_call(&sig, Some(fn_key), None, &gvals, args, span);
                            }
                        }
                        _ => {}
                    }
                }
                // Distinguish module-qualified call (`mod.func(...)`) or Type assoc call (`Type.init(...)` / `mod.Type.init(...)`) from value method call (`x.foo(...)`).
                if let Some(segs) = self.extract_path_idents(callee)
                    && segs.first().is_some_and(|s| self.lookup_local(s.sym).is_none())
                    && let Some(first_target) = self.prog.lookup_in_module(self.mod_id, segs[0].sym)
                {
                    match first_target {
                        TopScopeTarget::Module(_) => {
                            if segs.len() >= 3
                                && let Some(parent_target) = resolve_path_segments(
                                    &segs[..segs.len() - 1],
                                    self.mod_id,
                                    self.prog,
                                    self.interner,
                                    self.diags,
                                )
                            {
                                match parent_target {
                                    TopScopeTarget::Def(type_def) => {
                                        if let Some((impl_idx, sub_idx, sig)) =
                                            self.find_type_method_with_key(MethodTarget::Def(type_def), field.sym)
                                        {
                                            self.prog.mark_module_binding_used(self.mod_id, segs[0].sym);
                                            self.prog.mark_def_used(type_def);
                                            let fn_key = crate::cfg::CfgFnKey::ImplMethod { impl_idx, sub_idx };
                                            return self.check_fn_sig_call(&sig, Some(fn_key), None, &[], args, span);
                                        }
                                    }
                                    TopScopeTarget::Builtin(bk) => {
                                        return self.check_builtin_assoc_call(bk, &[], *field, args, expected, span);
                                    }
                                    _ => {}
                                }
                            }
                            return self.check_direct_fn_call(callee, &[], args, span);
                        }
                        TopScopeTarget::Def(type_def) if segs.len() == 2 => {
                            if let Some((impl_idx, sub_idx, sig)) = self.find_type_method_with_key(MethodTarget::Def(type_def), field.sym) {
                                self.prog.mark_module_binding_used(self.mod_id, segs[0].sym);
                                self.prog.mark_def_used(type_def);
                                let fn_key = crate::cfg::CfgFnKey::ImplMethod { impl_idx, sub_idx };
                                return self.check_fn_sig_call(&sig, Some(fn_key), None, &[], args, span);
                            }
                        }
                        TopScopeTarget::Builtin(bk) if segs.len() == 2 => {
                            return self.check_builtin_assoc_call(bk, &[], *field, args, expected, span);
                        }
                        _ => {}
                    }
                }
                (Vec::new(), Some(r.as_ref()), Some(*field))
            }
            _ => return self.check_direct_fn_call(callee, &[], args, span),
        };

        if let (Some(recv), Some(mfield)) = (recv_expr, method_name) {
            let rty = self.check_expr(recv, None);
            if rty == TyId::ERROR {
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                return TyId::ERROR;
            }
            let mname = self.interner.get(mfield.sym).to_owned();
            // Handle built-in slice/address/pool/arena helper methods cleanly.
            if let Some(builtin_ret) = self.try_builtin_method_call(recv, rty, &mname, args, span) {
                return builtin_ret;
            }
            let deref_ty = match self.env.get(rty) {
                Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
                _ => rty,
            };
            // Dynamic dispatch on `*dyn Iface` / `*mut dyn Iface`!
            if let Ty::Dyn(iface_def) = self.env.get(deref_ty).clone() {
                if let Some(im) = self.env.ifaces.get(&iface_def).cloned()
                    && let Some((midx, msig)) = im.methods.iter().enumerate().find(|(_, m)| m.name == mfield.sym)
                {
                    let needs_mut = msig.params.first().is_some_and(|p| {
                        p.is_self && matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. })
                    });
                    if needs_mut && !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. }) {
                        self.diags.error(
                            ErrorCode::E0230,
                            mfield.span,
                            format!("method `{}` requires `*mut dyn` receiver", mname),
                        );
                    }
                    self.table.insert_dyn_call(span, iface_def, midx);
                    return self.check_fn_sig_call(msig, None, Some(rty), &[], args, span);
                }
                self.diags.error(
                    ErrorCode::E0132,
                    mfield.span,
                    format!("interface `{}` has no method `{}`", self.fmt_ty(deref_ty), mname),
                );
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                return TyId::ERROR;
            }
            // Definition-site generic method check (`[T: Bound]`): verify method exists on declared bounds!
            if let Ty::GenericParam { idx, .. } = self.env.get(deref_ty).clone() {
                if let Some(gmeta) = self.sig.generics.get(idx as usize).cloned() {
                    if gmeta.builtin_ord && mname == "cmp" {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "`.cmp(&other)` expects 1 argument");
                        } else {
                            let expected_other = self.env.intern(Ty::Borrow {
                                is_mut: false,
                                inner: crate::types::BorrowInner::Single(deref_ty),
                            });
                            let at = self.check_expr_rvalue(&args[0], Some(expected_other));
                            self.expect_assignable(at, expected_other, args[0].span);
                        }
                        return TyId::ORDERING;
                    }
                    if gmeta.builtin_eq && mname == "eq" {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "`.eq(&other)` expects 1 argument");
                        } else {
                            let expected_other = self.env.intern(Ty::Borrow {
                                is_mut: false,
                                inner: crate::types::BorrowInner::Single(deref_ty),
                            });
                            let at = self.check_expr_rvalue(&args[0], Some(expected_other));
                            self.expect_assignable(at, expected_other, args[0].span);
                        }
                        return TyId::BOOL;
                    }
                    for &iface_def in &gmeta.bounds {
                        if let Some(im) = self.env.ifaces.get(&iface_def).cloned()
                            && let Some(msig) = im.methods.iter().find(|m| m.name == mfield.sym)
                        {
                            // Substitute `Ty::Dyn(iface_def)` (`Self`) -> `deref_ty` (`T`)!
                            let dyn_self = self.env.intern(Ty::Dyn(iface_def));
                            let mut sub_sig = msig.clone();
                            for p in &mut sub_sig.params {
                                p.ty = replace_ty(p.ty, dyn_self, deref_ty, self.env);
                            }
                            sub_sig.ret = replace_ty(sub_sig.ret, dyn_self, deref_ty, self.env);
                            return self.check_fn_sig_call(&sub_sig, None, Some(rty), &[], args, span);
                        }
                    }
                }
                self.diags.error(
                    ErrorCode::E0501,
                    mfield.span,
                    format!(
                        "generic type `{}` has no interface bound providing method `{}` (definition-site check)",
                        self.fmt_ty(deref_ty),
                        mname
                    ),
                );
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                return TyId::ERROR;
            }
            let target_key = match self.env.get(deref_ty) {
                Ty::Struct { def, .. }
                | Ty::Pool { def, .. }
                | Ty::Enum(def)
                | Ty::Union { def, .. }
                | Ty::Bitfield(def) => Some(MethodTarget::Def(*def)),
                Ty::Builtin(bk, _) => Some(MethodTarget::Builtin(*bk)),
                _ => None,
            };
            let struct_gargs: Vec<GenericVal> = match self.env.get(deref_ty) {
                Ty::Struct { args, .. } | Ty::Union { args, .. } | Ty::Pool { args, .. } => args.clone(),
                _ => Vec::new(),
            };
            if let Some(tk) = target_key
                && let Some((impl_idx, sub_idx, sig)) = self.find_type_method_with_key(tk, mfield.sym)
            {
                if sig.params.first().is_some_and(|p| {
                    p.is_self && matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. })
                }) && !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. })
                {
                    self.check_place_mutability(recv, recv.span);
                }
                let mut gvals = struct_gargs;
                for ga in &method_recv_gargs {
                    gvals.push(self.eval_generic_arg(ga));
                }
                let fn_key = crate::cfg::CfgFnKey::ImplMethod { impl_idx, sub_idx };
                return self.check_fn_sig_call(&sig, Some(fn_key), Some(rty), &gvals, args, span);
            }
            let _ = expected;
            self.diags.error(
                ErrorCode::E0132,
                mfield.span,
                format!("type `{}` has no method `{}`", self.fmt_ty(rty), mname),
            );
            for a in args {
                self.check_expr_rvalue(a, None);
            }
            return TyId::ERROR;
        }
        TyId::ERROR
    }

    fn check_cast_method(
        &mut self,
        kind: &str,
        recv: &Expr,
        gargs: &[GenericArg],
        args: &[Expr],
        span: Span,
    ) -> TyId {
        if !args.is_empty() {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!("`.{}()` takes no value arguments", kind),
            );
        }
        if gargs.len() != 1 {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!("`.{}.[To]()` requires 1 target type argument", kind),
            );
            return TyId::ERROR;
        }
        let to_ty = match self.eval_generic_arg(&gargs[0]) {
            GenericVal::Ty(t) => t,
            _ => {
                self.diags.error(ErrorCode::E0230, gargs[0].span, "expected target type");
                return TyId::ERROR;
            }
        };
        let from_ty = self.check_expr(recv, None);
        if from_ty == TyId::ERROR || to_ty == TyId::ERROR {
            return TyId::ERROR;
        }

        match kind {
            "widen" => {
                let ok = match (self.env.get(from_ty).clone(), self.env.get(to_ty).clone()) {
                    (Ty::Int { signed: s1, bits: b1, is_ptr_sized: p1 }, Ty::Int { signed: s2, bits: b2, is_ptr_sized: p2 }) => {
                        let bw1 = if p1 { 64 } else { b1 };
                        let bw2 = if p2 { 64 } else { b2 };
                        (s1 == s2 && bw2 >= bw1) || (!s1 && s2 && bw2 > bw1)
                    }
                    (Ty::Float(b1), Ty::Float(b2)) => b2 >= b1,
                    _ => false,
                };
                if !ok {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(
                            "cannot `.widen()` from `{}` to `{}`",
                            self.fmt_ty(from_ty),
                            self.fmt_ty(to_ty)
                        ),
                    );
                }
                to_ty
            }
            "trunc" | "clamp" => {
                if !self.env.is_integer(from_ty) || !self.env.is_integer(to_ty) {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(
                            "`.{}()` converts between integer types, found `{}` -> `{}`",
                            kind,
                            self.fmt_ty(from_ty),
                            self.fmt_ty(to_ty)
                        ),
                    );
                }
                to_ty
            }
            "try_into" => {
                if !self.env.is_integer(from_ty) || !self.env.is_integer(to_ty) {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(
                            "`.try_into()` converts between integer types, found `{}` -> `{}`",
                            self.fmt_ty(from_ty),
                            self.fmt_ty(to_ty)
                        ),
                    );
                }
                self.env.intern(Ty::ErrorUnion {
                    ok: to_ty,
                    errors: vec![ErrorSource::Builtin(BuiltinErrorKind::ArithError)],
                })
            }
            "bitcast" => {
                let l1 = self.env.layout(from_ty);
                let l2 = self.env.layout(to_ty);
                if l1.size != l2.size {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(
                            "`.bitcast()` requires equal byte sizes ({} != {})",
                            l1.size, l2.size
                        ),
                    );
                } else if self.env.has_niche_forbidden_for_bitcast(from_ty)
                    || self.env.has_niche_forbidden_for_bitcast(to_ty)
                {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(
                            "cannot `.bitcast()` between `{}` and `{}` because one contains disallowed niches/invariants",
                            self.fmt_ty(from_ty),
                            self.fmt_ty(to_ty)
                        ),
                    );
                }
                to_ty
            }
            "to_float" => {
                if !self.env.is_integer(from_ty) || !self.env.is_float(to_ty) {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        "`.to_float()` converts from integer to floating-point type",
                    );
                }
                to_ty
            }
            "trunc_to_int" => {
                if !self.env.is_float(from_ty) || !self.env.is_integer(to_ty) {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        "`.trunc_to_int()` converts from floating-point to integer type",
                    );
                }
                to_ty
            }
            _ => TyId::ERROR,
        }
    }

    fn try_builtin_method_call(
        &mut self,
        recv: &Expr,
        rty: TyId,
        mname: &str,
        args: &[Expr],
        span: Span,
    ) -> Option<TyId> {
        let deref = match self.env.get(rty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => rty,
        };
        if self.env.is_address(deref) && mname == "as_u64" && args.is_empty() {
            return Some(TyId::U64);
        }
        if matches!(self.env.get(deref), Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str)
            && mname == "as_bytes"
            && args.is_empty()
        {
            return Some(self.env.intern(Ty::Borrow {
                is_mut: false,
                inner: crate::types::BorrowInner::Slice(TyId::U8),
            }));
        }
        let needs_mut_self = matches!(
            mname,
            "alloc"
                | "reset"
                | "push"
                | "push_slice"
                | "insert"
                | "try_insert"
                | "remove"
                | "get_mut"
                | "at_mut"
                | "sub_mut"
        );
        if needs_mut_self
            && matches!(
                self.env.get(deref),
                Ty::Builtin(
                    BuiltinKind::Arena
                        | BuiltinKind::ArenaPool
                        | BuiltinKind::SlotPool
                        | BuiltinKind::LinearBuf
                        | BuiltinKind::Buf,
                    _
                )
            )
            && !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. })
        {
            self.check_place_mutability(recv, recv.span);
        }
        match self.env.get(rty).clone() {
            Ty::Borrow { is_mut, inner: crate::types::BorrowInner::Slice(elem) } => match mname {
                "len" => {
                    if !args.is_empty() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!(".len() takes 0 arguments, found {}", args.len()),
                        );
                    }
                    return Some(TyId::USIZE);
                }
                "as_str" if elem == TyId::U8 => {
                    if !args.is_empty() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!(".as_str() takes 0 arguments, found {}", args.len()),
                        );
                    }
                    let str_borrow = self.env.intern(Ty::Borrow {
                        is_mut: false,
                        inner: crate::types::BorrowInner::Str,
                    });
                    return Some(self.env.intern(Ty::ErrorUnion {
                        ok: str_borrow,
                        errors: vec![ErrorSource::Builtin(BuiltinErrorKind::Utf8Error)],
                    }));
                }
                "at" => return Some(self.check_slice_at_call(args, elem, false, span)),
                "at_mut" => {
                    if !is_mut {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.at_mut()` on immutable slice",
                        );
                    }
                    return Some(self.check_slice_at_call(args, elem, true, span));
                }
                "get" => return Some(self.check_slice_get_call(args, elem, false, span)),
                "get_mut" => {
                    if !is_mut {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.get_mut()` on immutable slice",
                        );
                    }
                    return Some(self.check_slice_get_call(args, elem, true, span));
                }
                "sub" => return Some(self.check_slice_sub_call(args, elem, false, span)),
                "sub_mut" => {
                    if !is_mut {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.sub_mut()` on immutable slice",
                        );
                    }
                    return Some(self.check_slice_sub_call(args, elem, true, span));
                }
                "each" => {
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, ".each() takes 0 arguments");
                    }
                    return Some(self.env.intern(Ty::Borrow {
                        is_mut: false,
                        inner: crate::types::BorrowInner::Slice(elem),
                    }));
                }
                "each_mut" => {
                    if !is_mut {
                        self.diags.error(ErrorCode::E0230, span, "cannot call `.each_mut()` on immutable slice");
                    }
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, ".each_mut() takes 0 arguments");
                    }
                    return Some(rty);
                }
                "par_reduce" => {
                    if args.len() != 2 {
                        self.diags.error(ErrorCode::E0230, span, ".par_reduce(op, init) expects 2 arguments");
                    } else {
                        if !matches!(args[0].kind, ExprKind::EnumLit(_)) {
                            let _ = self.check_expr_rvalue(&args[0], None);
                        }
                        let init_ty = self.check_expr_rvalue(&args[1], Some(elem));
                        self.expect_assignable(init_ty, elem, args[1].span);
                    }
                    return Some(elem);
                }
                _ => {}
            },
            Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } if mname == "len" => {
                if !args.is_empty() {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(".len() takes 0 arguments, found {}", args.len()),
                    );
                }
                return Some(TyId::USIZE);
            }
            _ => {}
        }
        match self.env.get(deref).clone() {
            Ty::Str if mname == "len" => {
                if !args.is_empty() {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!(".len() takes 0 arguments, found {}", args.len()),
                    );
                }
                return Some(TyId::USIZE);
            }
            Ty::Array { elem, .. } => match mname {
                "len" => {
                    if !args.is_empty() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!(".len() takes 0 arguments, found {}", args.len()),
                        );
                    }
                    return Some(TyId::USIZE);
                }
                "at" => return Some(self.check_slice_at_call(args, elem, false, span)),
                "at_mut" => {
                    if matches!(self.env.get(rty), Ty::Borrow { is_mut: false, .. }) {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.at_mut()` through immutable borrow",
                        );
                    } else if !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. }) {
                        self.check_place_mutability(recv, recv.span);
                    }
                    return Some(self.check_slice_at_call(args, elem, true, span));
                }
                "get" => return Some(self.check_slice_get_call(args, elem, false, span)),
                "get_mut" => {
                    if matches!(self.env.get(rty), Ty::Borrow { is_mut: false, .. }) {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.get_mut()` through immutable borrow",
                        );
                    } else if !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. }) {
                        self.check_place_mutability(recv, recv.span);
                    }
                    return Some(self.check_slice_get_call(args, elem, true, span));
                }
                "sub" => return Some(self.check_slice_sub_call(args, elem, false, span)),
                "sub_mut" => {
                    if matches!(self.env.get(rty), Ty::Borrow { is_mut: false, .. }) {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            "cannot call `.sub_mut()` through immutable borrow",
                        );
                    } else if !matches!(self.env.get(rty), Ty::Borrow { is_mut: true, .. }) {
                        self.check_place_mutability(recv, recv.span);
                    }
                    return Some(self.check_slice_sub_call(args, elem, true, span));
                }
                _ => {}
            },
            Ty::Vec { elem, len } => match mname {
                "len" => {
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, "vec.len() takes 0 arguments");
                    }
                    return Some(TyId::USIZE);
                }
                "cmpeq" | "cmpne" | "cmplt" | "cmple" | "cmpgt" | "cmpge" => {
                    if args.len() != 1 {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() expects 1 argument"));
                    } else {
                        let at = self.check_expr_rvalue(&args[0], Some(deref));
                        self.expect_assignable(at, deref, args[0].span);
                    }
                    return Some(self.env.intern(Ty::Vec { elem: TyId::BOOL, len }));
                }
                "any" | "all" => {
                    if elem != TyId::BOOL {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() requires vec[bool, N]"));
                    }
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() takes 0 arguments"));
                    }
                    return Some(TyId::BOOL);
                }
                "bitmask" => {
                    if elem != TyId::BOOL {
                        self.diags.error(ErrorCode::E0230, span, "vec.bitmask() requires vec[bool, N]");
                    }
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, "vec.bitmask() takes 0 arguments");
                    }
                    return Some(TyId::U64);
                }
                "select" => {
                    if elem != TyId::BOOL {
                        self.diags.error(ErrorCode::E0230, span, "vec.select() requires vec[bool, N] mask receiver");
                    }
                    if args.len() != 2 {
                        self.diags.error(ErrorCode::E0230, span, "vec.select(a, b) expects 2 vector arguments");
                        for a in args {
                            self.check_expr_rvalue(a, None);
                        }
                        return Some(TyId::ERROR);
                    }
                    let ta = self.check_expr_rvalue(&args[0], None);
                    let tb = self.check_expr_rvalue(&args[1], Some(ta));
                    if let Ty::Vec { len: alen, .. } = self.env.get(ta) {
                        if *alen != len {
                            self.diags.error(ErrorCode::E0230, args[0].span, format!("vec.select() mask length {len} != value length {alen}"));
                        }
                        self.expect_assignable(tb, ta, args[1].span);
                        return Some(ta);
                    }
                    self.diags.error(ErrorCode::E0230, args[0].span, "vec.select() arguments must be vectors");
                    return Some(TyId::ERROR);
                }
                "min" | "max" => {
                    if !self.env.is_numeric(elem) {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() requires numeric elements"));
                    }
                    if args.len() != 1 {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() expects 1 vector argument"));
                    } else {
                        let at = self.check_expr_rvalue(&args[0], Some(deref));
                        self.expect_assignable(at, deref, args[0].span);
                    }
                    return Some(deref);
                }
                "fma" => {
                    if !self.env.is_numeric(elem) {
                        self.diags.error(ErrorCode::E0230, span, "vec.fma() requires numeric elements");
                    }
                    if args.len() != 2 {
                        self.diags.error(ErrorCode::E0230, span, "vec.fma(b, c) expects 2 vector arguments");
                    } else {
                        let tb = self.check_expr_rvalue(&args[0], Some(deref));
                        self.expect_assignable(tb, deref, args[0].span);
                        let tc = self.check_expr_rvalue(&args[1], Some(deref));
                        self.expect_assignable(tc, deref, args[1].span);
                    }
                    return Some(deref);
                }
                "reduce_add" | "reduce_min" | "reduce_max" => {
                    if !self.env.is_numeric(elem) {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() requires numeric elements"));
                    }
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, format!("vec.{mname}() takes 0 arguments"));
                    }
                    return Some(elem);
                }
                "dot" => {
                    if !self.env.is_numeric(elem) {
                        self.diags.error(ErrorCode::E0230, span, "vec.dot() requires numeric elements");
                    }
                    if args.len() != 1 {
                        self.diags.error(ErrorCode::E0230, span, "vec.dot(other) expects 1 vector argument");
                    } else {
                        let at = self.check_expr_rvalue(&args[0], Some(deref));
                        self.expect_assignable(at, deref, args[0].span);
                    }
                    return Some(elem);
                }
                "shuffle" => {
                    if args.len() != 1 && args.len() != 2 {
                        self.diags.error(ErrorCode::E0230, span, "vec.shuffle expects (idx_vec) or (other, idx_vec)");
                        for a in args {
                            self.check_expr_rvalue(a, None);
                        }
                        return Some(TyId::ERROR);
                    }
                    let idx_arg = if args.len() == 2 {
                        let ot = self.check_expr_rvalue(&args[0], Some(deref));
                        self.expect_assignable(ot, deref, args[0].span);
                        &args[1]
                    } else {
                        &args[0]
                    };
                    let idx_ty = self.check_expr_rvalue(idx_arg, None);
                    if let Ty::Vec { elem: ie, len: ilen } = self.env.get(idx_ty).clone()
                        && self.env.is_integer(ie)
                    {
                        return Some(self.env.intern(Ty::Vec { elem, len: ilen }));
                    }
                    self.diags.error(ErrorCode::E0230, idx_arg.span, "vec.shuffle index argument must be an integer vector");
                    return Some(TyId::ERROR);
                }
                _ => {}
            },
            Ty::Builtin(BuiltinKind::Arena, _) => match mname {
                "alloc" => {
                    for a in args {
                        self.check_expr_rvalue(a, Some(TyId::USIZE));
                    }
                    let slice = self.env.intern(Ty::Borrow {
                        is_mut: true,
                        inner: crate::types::BorrowInner::Slice(TyId::U8),
                    });
                    return Some(self.env.intern(Ty::ErrorUnion {
                        ok: slice,
                        errors: vec![ErrorSource::Builtin(BuiltinErrorKind::MemError)],
                    }));
                }
                "reset" | "destroy" => {
                    for a in args {
                        self.check_expr_rvalue(a, None);
                    }
                    return Some(TyId::VOID);
                }
                "used_bytes" | "capacity" => {
                    if !args.is_empty() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("Arena.{}() takes 0 arguments, found {}", mname, args.len()),
                        );
                    }
                    return Some(TyId::USIZE);
                }
                _ => {}
            },
            Ty::Builtin(BuiltinKind::AllocCap, _) => match mname {
                "live_allocs" | "used_bytes" | "peak_bytes" => {
                    if !args.is_empty() {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("AllocCap.{}() takes 0 arguments, found {}", mname, args.len()),
                        );
                    }
                    return Some(TyId::USIZE);
                }
                "sub_budget" => {
                    if args.len() != 1 {
                        self.diags.error(
                            ErrorCode::E0230,
                            span,
                            format!("AllocCap.sub_budget() takes 1 argument, found {}", args.len()),
                        );
                    } else {
                        let lt = self.check_expr_rvalue(&args[0], Some(TyId::USIZE));
                        self.expect_assignable(lt, TyId::USIZE, args[0].span);
                    }
                    return Some(self.env.intern(Ty::Builtin(BuiltinKind::AllocCap, Vec::new())));
                }
                _ => {}
            },
            Ty::Builtin(BuiltinKind::ClockCap, _) => if mname == "now_ns" {
                if !args.is_empty() {
                    self.diags.error(ErrorCode::E0230, span, "ClockCap.now_ns() takes 0 arguments");
                }
                return Some(TyId::U64);
            },
            Ty::Builtin(BuiltinKind::ArgsCap, _) => match mname {
                "len" => {
                    if !args.is_empty() {
                        self.diags.error(ErrorCode::E0230, span, "ArgsCap.len() takes 0 arguments");
                    }
                    return Some(TyId::USIZE);
                }
                "get" => {
                    if args.len() != 1 {
                        self.diags.error(ErrorCode::E0230, span, "ArgsCap.get(idx) expects 1 index argument");
                    } else {
                        let it = self.check_expr_rvalue(&args[0], Some(TyId::USIZE));
                        self.expect_assignable(it, TyId::USIZE, args[0].span);
                    }
                    let str_borrow = self.env.intern(Ty::Borrow {
                        is_mut: false,
                        inner: crate::types::BorrowInner::Str,
                    });
                    return Some(self.env.intern(Ty::Optional(str_borrow)));
                }
                _ => {}
            },
            Ty::Builtin(BuiltinKind::FsCap, _) => {
                let str_borrow = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                match mname {
                    "open_dir" | "sub_dir" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, format!("FsCap.{} expects 1 path argument", mname));
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                        }
                        let ok = self.env.intern(Ty::Builtin(BuiltinKind::DirCap, Vec::new()));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "read_file" => {
                        let ac_mut = self.env.intern(Ty::Builtin(BuiltinKind::AllocCap, Vec::new()));
                        let ac_ptr = self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Single(ac_mut),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.read_file expects (alloc: *mut AllocCap, path: *str)");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let at = self.check_expr_rvalue(&args[0], Some(ac_ptr));
                            self.expect_assignable(at, ac_ptr, args[0].span);
                            let pt = self.check_expr_rvalue(&args[1], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[1].span);
                        }
                        let buf_u8 = self.env.intern(Ty::Builtin(BuiltinKind::Buf, vec![GenericVal::Ty(TyId::U8)]));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: buf_u8,
                            errors: vec![
                                ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory),
                                ErrorSource::Builtin(BuiltinErrorKind::IoError),
                            ],
                        }));
                    }
                    "write_file" => {
                        let slice_u8 = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(TyId::U8),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.write_file expects (path: *str, data: *[u8])");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                            let dt = self.check_expr_rvalue(&args[1], Some(slice_u8));
                            self.expect_assignable(dt, slice_u8, args[1].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "write_executable" => {
                        let slice_u8 = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(TyId::U8),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.write_executable expects (path: *str, data: *[u8])");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                            let dt = self.check_expr_rvalue(&args[1], Some(slice_u8));
                            self.expect_assignable(dt, slice_u8, args[1].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "is_dir" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.is_dir expects 1 path argument");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                        }
                        return Some(TyId::BOOL);
                    }
                    "mkdir_all" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.mkdir_all expects 1 path argument");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "list_dir" => {
                        let ac_mut = self.env.intern(Ty::Builtin(BuiltinKind::AllocCap, Vec::new()));
                        let ac_ptr = self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Single(ac_mut),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.list_dir expects (alloc: *mut AllocCap, path: *str)");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let at = self.check_expr_rvalue(&args[0], Some(ac_ptr));
                            self.expect_assignable(at, ac_ptr, args[0].span);
                            let pt = self.check_expr_rvalue(&args[1], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[1].span);
                        }
                        let buf_u8 = self.env.intern(Ty::Builtin(BuiltinKind::Buf, vec![GenericVal::Ty(TyId::U8)]));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: buf_u8,
                            errors: vec![
                                ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory),
                                ErrorSource::Builtin(BuiltinErrorKind::IoError),
                            ],
                        }));
                    }
                    "spawn_wait" => {
                        let slice_u8 = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(TyId::U8),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "FsCap.spawn_wait expects (path: *str, args_nul: *[u8])");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                            let dt = self.check_expr_rvalue(&args[1], Some(slice_u8));
                            self.expect_assignable(dt, slice_u8, args[1].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::I32,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::DirCap, _) => {
                let str_borrow = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                match mname {
                    "sub_dir" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "DirCap.sub_dir expects 1 path argument");
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                        }
                        let ok = self.env.intern(Ty::Builtin(BuiltinKind::DirCap, Vec::new()));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "read_file" => {
                        let ac_mut = self.env.intern(Ty::Builtin(BuiltinKind::AllocCap, Vec::new()));
                        let ac_ptr = self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Single(ac_mut),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "DirCap.read_file expects (alloc: *mut AllocCap, rel: *str)");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let at = self.check_expr_rvalue(&args[0], Some(ac_ptr));
                            self.expect_assignable(at, ac_ptr, args[0].span);
                            let pt = self.check_expr_rvalue(&args[1], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[1].span);
                        }
                        let buf_u8 = self.env.intern(Ty::Builtin(BuiltinKind::Buf, vec![GenericVal::Ty(TyId::U8)]));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: buf_u8,
                            errors: vec![
                                ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory),
                                ErrorSource::Builtin(BuiltinErrorKind::IoError),
                            ],
                        }));
                    }
                    "write_file" => {
                        let slice_u8 = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(TyId::U8),
                        });
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "DirCap.write_file expects (rel: *str, data: *[u8])");
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let pt = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(pt, str_borrow, args[0].span);
                            let dt = self.check_expr_rvalue(&args[1], Some(slice_u8));
                            self.expect_assignable(dt, slice_u8, args[1].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                        }));
                    }
                    "close" => {
                        if !args.is_empty() {
                            self.diags.error(ErrorCode::E0230, span, "DirCap.close() takes 0 arguments");
                        }
                        if matches!(self.env.get(rty), Ty::Borrow { .. }) {
                            self.diags.error(ErrorCode::E0109, span, "cannot consume borrowed `DirCap` via `.close()`; pass owned `DirCap`");
                        }
                        return Some(TyId::VOID);
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::NetCap, _) => if mname == "restrict_host" {
                let str_borrow = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, "NetCap.restrict_host expects 1 host argument");
                } else {
                    let ht = self.check_expr_rvalue(&args[0], Some(str_borrow));
                    self.expect_assignable(ht, str_borrow, args[0].span);
                }
                return Some(self.env.intern(Ty::Builtin(BuiltinKind::NetCap, Vec::new())));
            },
            Ty::Builtin(BuiltinKind::IoCap, _) => match mname {
                "nop_roundtrip" => {
                    if args.len() != 1 {
                        self.diags.error(ErrorCode::E0230, span, "IoCap.nop_roundtrip expects 1 count argument");
                    } else {
                        let ct = self.check_expr_rvalue(&args[0], Some(TyId::USIZE));
                        self.expect_assignable(ct, TyId::USIZE, args[0].span);
                    }
                    return Some(self.env.intern(Ty::ErrorUnion {
                        ok: TyId::USIZE,
                        errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                    }));
                }
                "run_fibers" => {
                    if args.len() != 2 {
                        self.diags.error(ErrorCode::E0230, span, "IoCap.run_fibers expects (count: usize, stack_bytes: usize)");
                        for a in args {
                            self.check_expr_rvalue(a, None);
                        }
                    } else {
                        let ct = self.check_expr_rvalue(&args[0], Some(TyId::USIZE));
                        self.expect_assignable(ct, TyId::USIZE, args[0].span);
                        let st = self.check_expr_rvalue(&args[1], Some(TyId::USIZE));
                        self.expect_assignable(st, TyId::USIZE, args[1].span);
                    }
                    return Some(self.env.intern(Ty::ErrorUnion {
                        ok: TyId::USIZE,
                        errors: vec![ErrorSource::Builtin(BuiltinErrorKind::IoError)],
                    }));
                }
                _ => {}
            },
            Ty::Builtin(BuiltinKind::BuildCap, _) => {
                let str_borrow = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                match mname {
                    "add_module" | "add_exe" | "add_shared_lib" | "add_shared" => {
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, format!("BuildCap.{} expects 2 string arguments", mname));
                            for a in args {
                                self.check_expr_rvalue(a, None);
                            }
                        } else {
                            let a0 = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(a0, str_borrow, args[0].span);
                            let a1 = self.check_expr_rvalue(&args[1], Some(str_borrow));
                            self.expect_assignable(a1, str_borrow, args[1].span);
                        }
                        return Some(TyId::VOID);
                    }
                    "set_opt_level" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "BuildCap.set_opt_level expects 1 u8 argument");
                        } else {
                            let lt = self.check_expr_rvalue(&args[0], Some(TyId::U8));
                            self.expect_assignable(lt, TyId::U8, args[0].span);
                        }
                        return Some(TyId::VOID);
                    }
                    "set_sanitize" | "set_audit_tcb" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, format!("BuildCap.{} expects 1 bool argument", mname));
                        } else {
                            let bt = self.check_expr_rvalue(&args[0], Some(TyId::BOOL));
                            self.expect_assignable(bt, TyId::BOOL, args[0].span);
                        }
                        return Some(TyId::VOID);
                    }
                    "link_lib" | "add_needed_lib" | "allow_hal" | "set_out_dir" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, format!("BuildCap.{} expects 1 string argument", mname));
                        } else {
                            let a0 = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(a0, str_borrow, args[0].span);
                        }
                        return Some(TyId::VOID);
                    }
                    "out_path" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "BuildCap.out_path expects 1 relative path argument");
                        } else {
                            let a0 = self.check_expr_rvalue(&args[0], Some(str_borrow));
                            self.expect_assignable(a0, str_borrow, args[0].span);
                        }
                        return Some(str_borrow);
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::Schedule, _) => {
                if mname == "run" {
                    for a in args {
                        self.check_expr_rvalue(a, None);
                    }
                    return Some(TyId::VOID);
                }
            }
            Ty::FnPtr { .. } if matches!(mname, "before" | "after") => {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, format!("sys.{}(other) expects 1 sys argument", mname));
                } else {
                    self.check_expr_rvalue(&args[0], None);
                }
                return Some(deref);
            }
            Ty::Builtin(BuiltinKind::ArenaPool, gargs) => {
                let pool_var = extract_root_ident_sym(recv)
                    .unwrap_or_else(|| self.interner.intern("_pool"));
                let elem_ty = match gargs.first() {
                    Some(GenericVal::Ty(t)) => Some(*t),
                    _ => None,
                };
                match mname {
                    "push" => {
                        if args.len() != 1 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("ArenaPool.push expects 1 argument, found {}", args.len()),
                            );
                        }
                        let prev = self.pool_handle_kind.replace(PoolHandleKind::ArenaPool(pool_var));
                        for a in args {
                            let got = self.check_expr_rvalue(a, elem_ty);
                            if let Some(et) = elem_ty {
                                self.expect_assignable(got, et, a.span);
                            }
                        }
                        self.pool_handle_kind = prev;
                        let id_ty = self.env.intern(Ty::Builtin(
                            BuiltinKind::IdBrand,
                            vec![GenericVal::Variant(pool_var)],
                        ));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: id_ty,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::MemError)],
                        }));
                    }
                    "id_at" => {
                        for a in args {
                            self.check_expr_rvalue(a, Some(TyId::USIZE));
                        }
                        let id_ty = self.env.intern(Ty::Builtin(
                            BuiltinKind::IdBrand,
                            vec![GenericVal::Variant(pool_var)],
                        ));
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: id_ty,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::BoundsError)],
                        }));
                    }
                    "reset" => {
                        return Some(self.env.intern(Ty::Builtin(BuiltinKind::ArenaPool, gargs)));
                    }
                    "len" => return Some(TyId::USIZE),
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::SlotPool, gargs) => {
                let elem_ty = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                let alloc_mut_cap = {
                    let ac = self.env.intern(Ty::Builtin(BuiltinKind::AllocCap, Vec::new()));
                    self.env.intern(Ty::Borrow {
                        is_mut: true,
                        inner: crate::types::BorrowInner::Single(ac),
                    })
                };
                let expected_gen_id = self.env.intern(Ty::Builtin(
                    BuiltinKind::GenId,
                    vec![GenericVal::Ty(elem_ty)],
                ));
                match mname {
                    "insert" => {
                        if args.len() != 2 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("SlotPool.insert expects 2 arguments, found {}", args.len()),
                            );
                        }
                        if let Some(a0) = args.first() {
                            let at0 = self.check_expr_rvalue(a0, Some(alloc_mut_cap));
                            self.expect_assignable(at0, alloc_mut_cap, a0.span);
                        }
                        let prev = self.pool_handle_kind.replace(PoolHandleKind::SlotPool);
                        for a in args.iter().skip(1) {
                            let got = self.check_expr_rvalue(a, Some(elem_ty));
                            self.expect_assignable(got, elem_ty, a.span);
                        }
                        self.pool_handle_kind = prev;
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: expected_gen_id,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory)],
                        }));
                    }
                    "try_insert" => {
                        if args.len() != 1 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("SlotPool.try_insert expects 1 argument, found {}", args.len()),
                            );
                        }
                        let prev = self.pool_handle_kind.replace(PoolHandleKind::SlotPool);
                        for a in args {
                            let got = self.check_expr_rvalue(a, Some(elem_ty));
                            self.expect_assignable(got, elem_ty, a.span);
                        }
                        self.pool_handle_kind = prev;
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: expected_gen_id,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory)],
                        }));
                    }
                    "remove" => {
                        self.check_slot_id_arg(args, expected_gen_id, mname, span);
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: elem_ty,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::StaleId)],
                        }));
                    }
                    "get" => {
                        self.check_slot_id_arg(args, expected_gen_id, mname, span);
                        let b = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Single(elem_ty),
                        });
                        return Some(self.env.intern(Ty::Optional(b)));
                    }
                    "get_mut" => {
                        self.check_slot_id_arg(args, expected_gen_id, mname, span);
                        let b = self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Single(elem_ty),
                        });
                        return Some(self.env.intern(Ty::Optional(b)));
                    }
                    "at" => {
                        self.check_slot_id_arg(args, expected_gen_id, mname, span);
                        let b = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Single(elem_ty),
                        });
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: b,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::StaleId)],
                        }));
                    }
                    "len" => return Some(TyId::USIZE),
                    "cap" => return Some(TyId::USIZE),
                    "is_alive" => {
                        for a in args {
                            let at = self.check_expr_rvalue(a, Some(TyId::USIZE));
                            self.expect_assignable(at, TyId::USIZE, a.span);
                        }
                        return Some(TyId::BOOL);
                    }
                    "id_at" => {
                        for a in args {
                            let at = self.check_expr_rvalue(a, Some(TyId::USIZE));
                            self.expect_assignable(at, TyId::USIZE, a.span);
                        }
                        return Some(self.env.intern(Ty::Optional(expected_gen_id)));
                    }
                    "each" | "each_mut" => {
                        if !args.is_empty() {
                            self.diags.error(ErrorCode::E0230, span, format!(".{mname}() takes 0 arguments"));
                        }
                        return Some(rty);
                    }
                    "destroy" => {
                        if args.len() != 1 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("SlotPool.destroy expects 1 argument, found {}", args.len()),
                            );
                        }
                        for a in args {
                            let at = self.check_expr_rvalue(a, Some(alloc_mut_cap));
                            self.expect_assignable(at, alloc_mut_cap, a.span);
                        }
                        return Some(TyId::VOID);
                    }
                    "drain" => {
                        let cb_ty = self.env.intern(Ty::FnPtr {
                            params: vec![elem_ty],
                            ret: TyId::VOID,
                        });
                        for (i, a) in args.iter().enumerate() {
                            let hint = if i + 1 == args.len() { Some(cb_ty) } else { None };
                            self.check_expr_rvalue(a, hint);
                        }
                        return Some(TyId::VOID);
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::LinearBuf, gargs) | Ty::Builtin(BuiltinKind::Buf, gargs) => {
                let elem_ty = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                match mname {
                    "push" => {
                        for a in args {
                            self.check_expr_rvalue(a, None);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory)],
                        }));
                    }
                    "push_slice" => {
                        let expected_slice = self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(elem_ty),
                        });
                        if args.len() != 1 && args.len() != 2 {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!("Buf.push_slice() expects 1 slice argument, found {}", args.len()),
                            );
                        } else {
                            let s_idx = args.len() - 1;
                            if args.len() == 2 {
                                self.check_expr_rvalue(&args[0], None);
                            }
                            let st = self.check_expr_rvalue(&args[s_idx], Some(expected_slice));
                            self.expect_assignable(st, expected_slice, args[s_idx].span);
                        }
                        return Some(self.env.intern(Ty::ErrorUnion {
                            ok: TyId::VOID,
                            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory)],
                        }));
                    }
                    "len" | "cap" => return Some(TyId::USIZE),
                    "clear" => return Some(TyId::VOID),
                    "as_slice" => {
                        return Some(self.env.intern(Ty::Borrow {
                            is_mut: false,
                            inner: crate::types::BorrowInner::Slice(elem_ty),
                        }));
                    }
                    "as_mut_slice" => {
                        return Some(self.env.intern(Ty::Borrow {
                            is_mut: true,
                            inner: crate::types::BorrowInner::Slice(elem_ty),
                        }));
                    }
                    "at" => return Some(self.check_slice_at_call(args, elem_ty, false, span)),
                    "at_mut" => return Some(self.check_slice_at_call(args, elem_ty, true, span)),
                    "get" => return Some(self.check_slice_get_call(args, elem_ty, false, span)),
                    "get_mut" => return Some(self.check_slice_get_call(args, elem_ty, true, span)),
                    "sub" => return Some(self.check_slice_sub_call(args, elem_ty, false, span)),
                    "sub_mut" => return Some(self.check_slice_sub_call(args, elem_ty, true, span)),
                    "pop" => return Some(self.env.intern(Ty::Optional(elem_ty))),
                    "destroy" => {
                        for a in args {
                            self.check_expr_rvalue(a, None);
                        }
                        return Some(TyId::VOID);
                    }
                    "drain" => {
                        let cb_ty = self.env.intern(Ty::FnPtr {
                            params: vec![elem_ty],
                            ret: TyId::VOID,
                        });
                        for (i, a) in args.iter().enumerate() {
                            let hint = if i + 1 == args.len() { Some(cb_ty) } else { None };
                            self.check_expr_rvalue(a, hint);
                        }
                        return Some(TyId::VOID);
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::Atomic, gargs) => {
                let elem_ty = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                match mname {
                    "load" => {
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "Atomic.load expects 1 Ordering argument");
                        } else {
                            let ot = self.check_expr_rvalue(&args[0], Some(TyId::ORDERING));
                            self.expect_assignable(ot, TyId::ORDERING, args[0].span);
                        }
                        return Some(elem_ty);
                    }
                    "store" => {
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, "Atomic.store expects (val, Ordering)");
                        } else {
                            let vt = self.check_expr_rvalue(&args[0], Some(elem_ty));
                            self.expect_assignable(vt, elem_ty, args[0].span);
                            let ot = self.check_expr_rvalue(&args[1], Some(TyId::ORDERING));
                            self.expect_assignable(ot, TyId::ORDERING, args[1].span);
                        }
                        return Some(TyId::VOID);
                    }
                    "fetch_add" | "fetch_sub" | "swap" => {
                        if args.len() != 2 {
                            self.diags.error(ErrorCode::E0230, span, format!("Atomic.{mname} expects (val, Ordering)"));
                        } else {
                            let vt = self.check_expr_rvalue(&args[0], Some(elem_ty));
                            self.expect_assignable(vt, elem_ty, args[0].span);
                            let ot = self.check_expr_rvalue(&args[1], Some(TyId::ORDERING));
                            self.expect_assignable(ot, TyId::ORDERING, args[1].span);
                        }
                        return Some(elem_ty);
                    }
                    "compare_exchange" => {
                        if args.len() != 4 {
                            self.diags.error(ErrorCode::E0230, span, "Atomic.compare_exchange expects (expected, desired, succ_ord, fail_ord)");
                        } else {
                            let e_ty = self.check_expr_rvalue(&args[0], Some(elem_ty));
                            self.expect_assignable(e_ty, elem_ty, args[0].span);
                            let d_ty = self.check_expr_rvalue(&args[1], Some(elem_ty));
                            self.expect_assignable(d_ty, elem_ty, args[1].span);
                            let s_ty = self.check_expr_rvalue(&args[2], Some(TyId::ORDERING));
                            self.expect_assignable(s_ty, TyId::ORDERING, args[2].span);
                            let f_ty = self.check_expr_rvalue(&args[3], Some(TyId::ORDERING));
                            self.expect_assignable(f_ty, TyId::ORDERING, args[3].span);
                        }
                        return Some(TyId::BOOL);
                    }
                    _ => {}
                }
            }
            Ty::Builtin(BuiltinKind::MmioReg, gargs) => {
                let elem_ty = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                let mode_str = match gargs.get(1) {
                    Some(GenericVal::Variant(sym)) => self.interner.get(*sym).to_owned(),
                    _ => "ReadWrite".to_owned(),
                };
                match mname {
                    "read" => {
                        if mode_str == "WriteOnly" {
                            self.diags.error(
                                ErrorCode::E0805,
                                span,
                                "cannot `.read()` from a `.WriteOnly` MmioReg",
                            );
                        }
                        if !args.is_empty() {
                            self.diags.error(ErrorCode::E0230, span, "MmioReg.read takes 0 arguments");
                        }
                        return Some(elem_ty);
                    }
                    "write" => {
                        if mode_str == "ReadOnly" {
                            self.diags.error(
                                ErrorCode::E0805,
                                span,
                                "cannot `.write()` to a `.ReadOnly` MmioReg",
                            );
                        }
                        if args.len() != 1 {
                            self.diags.error(ErrorCode::E0230, span, "MmioReg.write expects 1 value argument");
                        } else {
                            let vt = self.check_expr_rvalue(&args[0], Some(elem_ty));
                            self.expect_assignable(vt, elem_ty, args[0].span);
                        }
                        return Some(TyId::VOID);
                    }
                    _ => {}
                }
            }
            Ty::RawPtr { .. } | Ty::PhysAddr | Ty::VirtAddr | Ty::DmaAddr
                if mname == "as_usize" && args.is_empty() =>
            {
                return Some(TyId::USIZE);
            }
            _ => {}
        }
        let _ = span;
        None
    }

    fn check_slice_at_call(&mut self, args: &[Expr], elem: TyId, is_mut: bool, span: Span) -> TyId {
        if args.len() != 1 {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!(".at{}() expects 1 index argument, found {}", if is_mut { "_mut" } else { "" }, args.len()),
            );
        }
        for a in args {
            let it = self.check_expr_rvalue(a, Some(TyId::USIZE));
            self.expect_assignable(it, TyId::USIZE, a.span);
        }
        let ok = self.env.intern(Ty::Borrow {
            is_mut,
            inner: crate::types::BorrowInner::Single(elem),
        });
        self.env.intern(Ty::ErrorUnion {
            ok,
            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::BoundsError)],
        })
    }

    fn check_slice_get_call(&mut self, args: &[Expr], elem: TyId, is_mut: bool, span: Span) -> TyId {
        if args.len() != 1 {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!(".get{}() expects 1 index argument, found {}", if is_mut { "_mut" } else { "" }, args.len()),
            );
        }
        for a in args {
            let it = self.check_expr_rvalue(a, Some(TyId::USIZE));
            self.expect_assignable(it, TyId::USIZE, a.span);
        }
        let ptr = self.env.intern(Ty::Borrow {
            is_mut,
            inner: crate::types::BorrowInner::Single(elem),
        });
        self.env.intern(Ty::Optional(ptr))
    }

    fn check_slice_sub_call(&mut self, args: &[Expr], elem: TyId, is_mut: bool, span: Span) -> TyId {
        if args.len() != 2 {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!(".sub{}() expects 2 range arguments (lo, hi), found {}", if is_mut { "_mut" } else { "" }, args.len()),
            );
        }
        for a in args {
            let it = self.check_expr_rvalue(a, Some(TyId::USIZE));
            self.expect_assignable(it, TyId::USIZE, a.span);
        }
        let ok = self.env.intern(Ty::Borrow {
            is_mut,
            inner: crate::types::BorrowInner::Slice(elem),
        });
        self.env.intern(Ty::ErrorUnion {
            ok,
            errors: vec![ErrorSource::Builtin(BuiltinErrorKind::BoundsError)],
        })
    }

    fn check_slot_id_arg(&mut self, args: &[Expr], expected_gen_id: TyId, method: &str, span: Span) {
        if args.len() != 1 {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!("SlotPool.{} expects 1 `GenId` argument, found {}", method, args.len()),
            );
        }
        for a in args {
            let got = self.check_expr_rvalue(a, Some(expected_gen_id));
            self.expect_assignable(got, expected_gen_id, a.span);
        }
    }

    fn check_index_ty(&mut self, base: &Expr, idx_ty: TyId, span: Span) {
        if idx_ty == TyId::ERROR || self.env.is_integer(idx_ty) {
            return;
        }
        if let ExprKind::Field { base: pool_expr, .. } = &base.kind
            && let Some(raw_pool_ty) = self.table.get(pool_expr.span)
        {
            let pool_deref = match self.env.get(raw_pool_ty) {
                Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
                _ => raw_pool_ty,
            };
            match self.env.get(pool_deref).clone() {
                Ty::Builtin(BuiltinKind::ArenaPool, _) => {
                    if let Ty::Builtin(BuiltinKind::IdBrand, args) = self.env.get(idx_ty).clone() {
                        if let Some(GenericVal::Variant(id_brand)) = args.first().copied()
                            && let Some(pool_sym) = extract_root_ident_sym(pool_expr)
                            && pool_sym != id_brand
                        {
                            self.diags.error(
                                ErrorCode::E0330,
                                span,
                                format!(
                                    "brand mismatch: `Id[{}]` cannot index pool `{}`",
                                    self.interner.get(id_brand),
                                    self.interner.get(pool_sym)
                                ),
                            );
                        }
                        return;
                    }
                }
                Ty::Builtin(BuiltinKind::SlotPool, gargs) => {
                    let schema = gargs.first().and_then(|g| match g {
                        GenericVal::Ty(t) => Some(*t),
                        _ => None,
                    });
                    if let Ty::Builtin(BuiltinKind::GenId, _) = self.env.get(idx_ty) {
                        if let Some(expected_schema) = schema
                            && !self.env.is_gen_id_of(idx_ty, expected_schema)
                        {
                            self.diags.error(
                                ErrorCode::E0230,
                                span,
                                format!(
                                    "type mismatch: cannot index `SlotPool[{}]` column with `{}`",
                                    self.fmt_ty(expected_schema),
                                    self.fmt_ty(idx_ty)
                                ),
                            );
                        }
                        return;
                    }
                }
                _ => {}
            }
        }
        self.diags.error(ErrorCode::E0230, span, "index must be an integer or valid pool Id");
    }

    fn check_builtin_assoc_call(
        &mut self,
        bk: BuiltinKind,
        type_gargs: &[GenericArg],
        field: ast::IdentSpan,
        args: &[Expr],
        _expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let fname = self.interner.get(field.sym).to_owned();
        if fname == "build" && bk == BuiltinKind::Schedule {
            for a in args {
                self.check_expr_rvalue(a, None);
            }
            return self.env.intern(Ty::Builtin(BuiltinKind::Schedule, Vec::new()));
        }
        if fname == "new" && bk == BuiltinKind::Atomic {
            let gvals: Vec<GenericVal> = type_gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
            let elem_ty = match gvals.first() {
                Some(GenericVal::Ty(t)) => *t,
                _ => TyId::ERROR,
            };
            if args.len() != 1 {
                self.diags.error(ErrorCode::E0230, span, "Atomic.new expects 1 initial value argument");
            } else {
                let vt = self.check_expr_rvalue(&args[0], Some(elem_ty));
                self.expect_assignable(vt, elem_ty, args[0].span);
            }
            return self.env.intern(Ty::Builtin(BuiltinKind::Atomic, gvals));
        }
        if fname == "from_usize" {
            for a in args {
                let at = self.check_expr_rvalue(a, Some(TyId::USIZE));
                self.expect_assignable(at, TyId::USIZE, a.span);
            }
            match bk {
                BuiltinKind::RawPtr { is_mut } => {
                    let gvals: Vec<GenericVal> = type_gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
                    let inner = match gvals.first() {
                        Some(GenericVal::Ty(t)) => *t,
                        _ => TyId::VOID,
                    };
                    return self.env.intern(Ty::RawPtr { is_mut, inner });
                }
                BuiltinKind::Prim(PrimKind::PhysAddr) => return TyId::PHYS_ADDR,
                BuiltinKind::Prim(PrimKind::VirtAddr) => return TyId::VIRT_ADDR,
                BuiltinKind::Prim(PrimKind::DmaAddr) => return TyId::DMA_ADDR,
                _ => {}
            }
        }
        if fname == "splat" {
            let vec_ty = match bk {
                BuiltinKind::SimdAlias(pk, n) => {
                    let el = crate::types::prim_to_ty_id_in_env(pk, self.env);
                    Some((el, self.env.intern(Ty::Vec { elem: el, len: n })))
                }
                BuiltinKind::VecType => {
                    let gvals: Vec<GenericVal> = type_gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
                    if let (Some(GenericVal::Ty(el)), Some(GenericVal::Int(n))) = (gvals.first(), gvals.get(1))
                        && (2..=64).contains(n)
                        && (*n as u64).is_power_of_two()
                    {
                        Some((*el, self.env.intern(Ty::Vec { elem: *el, len: *n as u8 })))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some((el, vty)) = vec_ty {
                if args.len() != 1 {
                    self.diags.error(ErrorCode::E0230, span, "vec.splat expects 1 scalar argument");
                } else {
                    let vt = self.check_expr_rvalue(&args[0], Some(el));
                    self.expect_assignable(vt, el, args[0].span);
                }
                return vty;
            }
        }
        if fname == "init" {
            for a in args {
                self.check_expr_rvalue(a, None);
            }
            let gvals: Vec<GenericVal> = type_gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
            let err = match bk {
                BuiltinKind::Arena | BuiltinKind::ArenaPool => BuiltinErrorKind::MemError,
                _ => BuiltinErrorKind::OutOfMemory,
            };
            let ok = self.env.intern(Ty::Builtin(bk, gvals));
            return self.env.intern(Ty::ErrorUnion {
                ok,
                errors: vec![ErrorSource::Builtin(err)],
            });
        }
        let _ = span;
        for a in args {
            self.check_expr_rvalue(a, None);
        }
        TyId::ERROR
    }

    fn check_direct_fn_call(
        &mut self,
        callee: &Expr,
        gargs: &[GenericArg],
        args: &[Expr],
        span: Span,
    ) -> TyId {
        // Resolve path to named function def if applicable.
        if let Some(segs) = self.extract_path_idents(callee)
            && segs.first().is_some_and(|s| self.lookup_local(s.sym).is_none())
        {
            let Some(target) = resolve_path_segments(&segs, self.mod_id, self.prog, self.interner, self.diags) else {
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                return TyId::ERROR;
            };
            if let TopScopeTarget::Def(d) = target
                && let Some(sig) = self.env.fns.get(&d).cloned()
            {
                let gvals: Vec<GenericVal> = gargs.iter().map(|a| self.eval_generic_arg(a)).collect();
                return self.check_fn_sig_call(&sig, Some(crate::cfg::CfgFnKey::TopDef(d)), None, &gvals, args, span);
            }
        }

        let ct = self.check_expr(callee, None);
        match self.env.get(ct).clone() {
            Ty::FnPtr { params, ret } => {
                if args.len() != params.len() {
                    self.diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("expected {} arguments, found {}", params.len(), args.len()),
                    );
                }
                for (i, a) in args.iter().enumerate() {
                    let exp = params.get(i).copied();
                    let at = self.check_expr_rvalue(a, exp);
                    if let Some(e) = exp {
                        self.expect_assignable(at, e, a.span);
                    }
                }
                ret
            }
            Ty::Error => {
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                TyId::ERROR
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    callee.span,
                    format!("cannot call value of type `{}`", self.fmt_ty(ct)),
                );
                for a in args {
                    self.check_expr_rvalue(a, None);
                }
                TyId::ERROR
            }
        }
    }

    fn check_fn_sig_call(
        &mut self,
        sig: &FnSig,
        fn_key: Option<crate::cfg::CfgFnKey>,
        self_recv_ty: Option<TyId>,
        explicit_gargs: &[GenericVal],
        args: &[Expr],
        span: Span,
    ) -> TyId {
        let mut param_start = 0usize;
        if let (Some(rty), Some(first_p)) = (self_recv_ty, sig.params.first())
            && first_p.is_self
        {
            let p_ty = subst_ty(first_p.ty, explicit_gargs, self.env);
            self.expect_self_compatible(rty, p_ty, span);
            param_start = 1;
        }
        let expected_params = &sig.params[param_start..];
        if args.len() != expected_params.len() {
            self.diags.error(
                ErrorCode::E0230,
                span,
                format!(
                    "function `{}` expects {} arguments, found {}",
                    self.interner.get(sig.name),
                    expected_params.len(),
                    args.len()
                ),
            );
        }

        let mut inferred = explicit_gargs.to_vec();
        let mut param_arg_syms: Vec<(Symbol, Option<Symbol>)> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let p_sig = expected_params.get(i);
            let mut exp_ty = p_sig.map(|p| subst_ty(p.ty, &inferred, self.env));
            if let Some(et) = exp_ty
                && let Ty::Builtin(BuiltinKind::IdBrand, bargs) = self.env.get(et).clone()
                && let Some(GenericVal::Variant(formal_p)) = bargs.first().copied()
                && let Some(Some(actual_sym)) = param_arg_syms.iter().find(|(fp, _)| *fp == formal_p).map(|(_, a)| *a)
            {
                exp_ty = Some(self.env.intern(Ty::Builtin(
                    BuiltinKind::IdBrand,
                    vec![GenericVal::Variant(actual_sym)],
                )));
            }
            let arg_ty = if matches!(p_sig.and_then(|p| p.modifier), Some(ParamModifier::Mut)) {
                self.check_place_mutability(arg, arg.span);
                self.check_expr(arg, exp_ty)
            } else {
                self.check_expr_rvalue(arg, exp_ty)
            };
            if let Some(p) = p_sig {
                param_arg_syms.push((p.name, extract_root_ident_sym(arg)));
                self.unify_generic_arg(p.ty, arg_ty, &mut inferred);
                let mut resolved_p_ty = subst_ty(p.ty, &inferred, self.env);
                if let Ty::Builtin(BuiltinKind::IdBrand, bargs) = self.env.get(resolved_p_ty).clone()
                    && let Some(GenericVal::Variant(formal_p)) = bargs.first().copied()
                    && let Some(Some(actual_sym)) =
                        param_arg_syms.iter().find(|(fp, _)| *fp == formal_p).map(|(_, a)| *a)
                {
                    resolved_p_ty = self.env.intern(Ty::Builtin(
                        BuiltinKind::IdBrand,
                        vec![GenericVal::Variant(actual_sym)],
                    ));
                }
                self.expect_assignable(arg_ty, resolved_p_ty, arg.span);
            }
        }
        // Check generic interface bounds at call site & record concrete generic instantiation!
        if !sig.generics.is_empty() && !inferred.is_empty() {
            for (gi, gmeta) in sig.generics.iter().enumerate() {
                if let Some(GenericVal::Ty(actual_ty)) = inferred.get(gi).copied()
                    && actual_ty != TyId::ERROR
                    && !matches!(self.env.get(actual_ty), Ty::GenericParam { .. })
                {
                    for &iface_def in &gmeta.bounds {
                        if !self.type_implements_iface(actual_ty, iface_def) {
                            self.diags.error(
                                ErrorCode::E0501,
                                span,
                                format!(
                                    "type `{}` does not implement required interface `{}`",
                                    self.fmt_ty(actual_ty),
                                    self.interner.get(self.prog.defs[iface_def.0 as usize].name)
                                ),
                            );
                        }
                    }
                    if gmeta.builtin_ord && !self.type_supports_ord(actual_ty) {
                        self.diags.error(
                            ErrorCode::E0501,
                            span,
                            format!("type `{}` does not satisfy `Ord` bound", self.fmt_ty(actual_ty)),
                        );
                    }
                    if gmeta.builtin_eq && !self.type_supports_eq(actual_ty) {
                        self.diags.error(
                            ErrorCode::E0501,
                            span,
                            format!("type `{}` does not satisfy `Eq` bound", self.fmt_ty(actual_ty)),
                        );
                    }
                }
            }
        }
        if let Some(fk) = fn_key
            && !inferred.is_empty()
            && inferred.iter().all(|gv| match gv {
                GenericVal::Ty(t) => *t != TyId::ERROR && !ty_contains_generic_param(*t, self.env),
                _ => true,
            })
        {
            self.table.insert_call_generic_inst(span, fk, inferred.clone());
        }
        subst_ty(sig.ret, &inferred, self.env)
    }

    fn expect_self_compatible(&mut self, recv: TyId, param: TyId, span: Span) {
        if recv == param || recv == TyId::ERROR || param == TyId::ERROR {
            return;
        }
        // Auto-borrow: `T` matches `*T` and `&mut T` / `*mut T` matches `*T`.
        match self.env.get(param).clone() {
            Ty::Borrow { is_mut: false, inner: crate::types::BorrowInner::Single(t) } if t == recv => return,
            Ty::Borrow { is_mut: true, inner: crate::types::BorrowInner::Single(t) } if t == recv => return,
            _ => {}
        }
        if let Ty::Borrow { is_mut: true, inner: crate::types::BorrowInner::Single(r_in) } = self.env.get(recv).clone()
            && let Ty::Borrow { is_mut: false, inner: crate::types::BorrowInner::Single(p_in) } = self.env.get(param).clone()
            && r_in == p_in
        {
            return;
        }
        self.expect_assignable(recv, param, span);
    }

    fn unify_generic_arg(&self, pattern: TyId, actual: TyId, gvals: &mut Vec<GenericVal>) {
        if actual == TyId::ERROR || actual == TyId::COMPTIME_INT || actual == TyId::COMPTIME_FLOAT {
            return;
        }
        match self.env.get(pattern).clone() {
            Ty::GenericParam { idx, is_comptime: false, .. } => {
                let i = idx as usize;
                while gvals.len() <= i {
                    gvals.push(GenericVal::Ty(TyId::ERROR));
                }
                if gvals[i] == GenericVal::Ty(TyId::ERROR) {
                    gvals[i] = GenericVal::Ty(actual);
                }
            }
            Ty::Borrow { inner: crate::types::BorrowInner::Single(pi), .. } => {
                if let Ty::Borrow { inner: crate::types::BorrowInner::Single(ai), .. } = self.env.get(actual) {
                    self.unify_generic_arg(pi, *ai, gvals);
                }
            }
            Ty::Borrow { inner: crate::types::BorrowInner::Slice(pi), .. } => {
                if let Ty::Borrow { inner: crate::types::BorrowInner::Slice(ai), .. } = self.env.get(actual) {
                    self.unify_generic_arg(pi, *ai, gvals);
                }
            }
            Ty::Optional(pi) => {
                if let Ty::Optional(ai) = self.env.get(actual) {
                    self.unify_generic_arg(pi, *ai, gvals);
                }
            }
            _ => {}
        }
    }

    fn find_type_method_with_key(&self, target: MethodTarget, name: Symbol) -> Option<(usize, usize, FnSig)> {
        for (impl_idx, im) in self.env.impls.iter().enumerate() {
            if im.target == target
                && let Some((_, sub_idx, sig)) = im.methods.iter().find(|(_, _, m)| m.name == name)
            {
                return Some((impl_idx, *sub_idx, sig.clone()));
            }
        }
        None
    }

    fn type_implements_iface(&self, ty: TyId, iface_def: DefId) -> bool {
        let target = match self.env.get(ty) {
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def) => Some(MethodTarget::Def(*def)),
            Ty::Builtin(bk, _) => Some(MethodTarget::Builtin(*bk)),
            _ => None,
        };
        let Some(tk) = target else { return false };
        self.env.impls.iter().any(|im| im.target == tk && im.iface == Some(iface_def))
    }

    fn type_supports_ord(&self, ty: TyId) -> bool {
        self.env.is_integer(ty)
            || self.env.is_float(ty)
            || matches!(self.env.get(ty), Ty::Bool | Ty::Char | Ty::Str | Ty::Enum(_))
    }

    fn type_supports_eq(&self, ty: TyId) -> bool {
        self.type_supports_ord(ty)
            || matches!(self.env.get(ty), Ty::PhysAddr | Ty::VirtAddr | Ty::DmaAddr)
    }

    fn check_closure(&mut self, params: &[ast::Param], body: &Block, expected: Option<TyId>) -> TyId {
        let (exp_params, exp_ret) = match expected.map(|e| self.env.get(e).clone()) {
            Some(Ty::FnPtr { params: ep, ret }) => (Some(ep), Some(ret)),
            _ => (None, None),
        };
        self.push_scope();
        let mut p_tys = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let pty = if matches!(p.ty.kind, ast::TypeKind::Inferred) {
                exp_params.as_ref().and_then(|ep| ep.get(i).copied()).unwrap_or(TyId::ERROR)
            } else {
                self.resolve_ty(&p.ty)
            };
            p_tys.push(pty);
            self.table.insert(p.span, pty);
            let is_mut = matches!(p.modifier, Some(ParamModifier::Mut));
            self.bind_local(p.name.sym, p.span, pty, is_mut, true, p.is_comptime, p.is_self);
        }
        let saved_ret = self.sig.ret;
        self.sig.ret = exp_ret.unwrap_or(TyId::ERROR);
        let (ret_ty, _) = self.check_block_expr(body, exp_ret);
        self.sig.ret = saved_ret;
        self.pop_scope();
        let inferred_ret = if let Some(er) = exp_ret {
            er
        } else if ret_ty == TyId::NEVER {
            let mut found = None;
            for st in &body.stmts {
                if let StmtKind::Jump(JumpAction::Return(Some(rv))) = &st.kind {
                    found = self.table.get(rv.span);
                    break;
                }
            }
            found.unwrap_or(TyId::VOID)
        } else {
            ret_ty
        };
        self.env.intern(Ty::FnPtr {
            params: p_tys,
            ret: inferred_ret,
        })
    }

    fn index_elem_ty(&mut self, base_ty: TyId, span: Span) -> TyId {
        let inner = match self.env.get(base_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => base_ty,
        };
        match self.env.get(inner) {
            Ty::Array { elem, .. }
            | Ty::Borrow { inner: crate::types::BorrowInner::Slice(elem), .. }
            | Ty::Vec { elem, .. } => *elem,
            Ty::Builtin(BuiltinKind::GpuBuf, gargs) => match gargs.first() {
                Some(GenericVal::Ty(el)) => *el,
                _ => TyId::ERROR,
            },
            Ty::Str | Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    "cannot index `str` directly by byte offset; use `.as_bytes()` for UTF-8 byte access",
                );
                TyId::ERROR
            }
            Ty::Error => TyId::ERROR,
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    format!("cannot index value of type `{}`", self.fmt_ty(base_ty)),
                );
                TyId::ERROR
            }
        }
    }

    fn slice_result_ty(&mut self, base_ty: TyId, span: Span) -> TyId {
        let (is_mut, inner) = match self.env.get(base_ty) {
            Ty::Borrow { is_mut, inner: crate::types::BorrowInner::Single(t) } => (*is_mut, *t),
            Ty::Borrow { is_mut, inner: crate::types::BorrowInner::Slice(el) } => {
                return self.env.intern(Ty::Borrow {
                    is_mut: *is_mut,
                    inner: crate::types::BorrowInner::Slice(*el),
                });
            }
            Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    "cannot slice `str` directly (`s[lo..hi]` violates UTF-8 boundaries); use `.as_bytes()`",
                );
                return TyId::ERROR;
            }
            _ => (false, base_ty),
        };
        match self.env.get(inner) {
            Ty::Array { elem, .. } => self.env.intern(Ty::Borrow {
                is_mut,
                inner: crate::types::BorrowInner::Slice(*elem),
            }),
            Ty::Str => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    "cannot slice `str` directly (`s[lo..hi]` violates UTF-8 boundaries); use `.as_bytes()`",
                );
                TyId::ERROR
            }
            Ty::Error => TyId::ERROR,
            _ => {
                self.diags.error(
                    ErrorCode::E0230,
                    span,
                    format!("cannot slice value of type `{}`", self.fmt_ty(base_ty)),
                );
                TyId::ERROR
            }
        }
    }

    fn resolve_ty(&mut self, te: &ast::TypeExpr) -> TyId {
        resolve_type_expr(
            te,
            self.mod_id,
            &self.generic_ctx,
            self.units,
            self.prog,
            self.env,
            self.interner,
            self.diags,
        )
    }

    fn eval_generic_arg(&mut self, arg: &GenericArg) -> GenericVal {
        match &arg.kind {
            GenericArgKind::Type(t) => GenericVal::Ty(self.resolve_ty(t)),
            GenericArgKind::TypeOrConst(segs) => {
                if segs.len() == 1
                    && let Some(&val) = self.generic_ctx.comptime_vals.get(&segs[0].sym)
                {
                    return GenericVal::Int(val);
                }
                let tp = ast::TypePath {
                    segments: segs.clone(),
                    generics: Vec::new(),
                };
                let t = resolve_type_expr(
                    &ast::TypeExpr {
                        span: arg.span,
                        kind: ast::TypeKind::Path(tp),
                    },
                    self.mod_id,
                    &self.generic_ctx,
                    self.units,
                    self.prog,
                    self.env,
                    self.interner,
                    self.diags,
                );
                GenericVal::Ty(t)
            }
            GenericArgKind::Expr(e) => {
                if let ExprKind::EnumLit(v) = &e.kind {
                    return GenericVal::Variant(v.sym);
                }
                let v = eval_const_int(
                    e,
                    self.mod_id,
                    &self.generic_ctx,
                    self.units,
                    self.prog,
                    self.env,
                    self.interner,
                )
                .unwrap_or(0);
                GenericVal::Int(v)
            }
        }
    }

    fn bind_pattern(&mut self, pat: &Pat, ty: TyId, is_mut_allowed: bool) {
        let _ = is_mut_allowed;
        match &pat.kind {
            PatKind::Wildcard | PatKind::Error | PatKind::Null => {}
            PatKind::BoolLit(_) | PatKind::CharLit { .. } | PatKind::IntLit { .. } => {}
            PatKind::Ident(sym) => {
                self.bind_local(*sym, pat.span, ty, false, false, false, false);
            }
            PatKind::Tuple(ps) => {
                if let Ty::Tuple(elems) = self.env.get(ty).clone() {
                    if elems.len() != ps.len() {
                        self.diags.error(
                            ErrorCode::E0230,
                            pat.span,
                            format!("tuple pattern has {} elements, expected {}", ps.len(), elems.len()),
                        );
                    }
                    for (i, p) in ps.iter().enumerate() {
                        let ety = elems.get(i).copied().unwrap_or(TyId::ERROR);
                        self.bind_pattern(p, ety, is_mut_allowed);
                    }
                } else if ty != TyId::ERROR {
                    self.diags.error(
                        ErrorCode::E0230,
                        pat.span,
                        format!("cannot match tuple pattern against `{}`", self.fmt_ty(ty)),
                    );
                }
            }
            PatKind::Variant { name, args } => {
                match self.env.get(ty).clone() {
                    Ty::Union { def, args: uargs } => {
                        let raw_payload = self.env.unions.get(&def).and_then(|um| {
                            um.variants
                                .iter()
                                .find(|v| v.name == name.sym)
                                .map(|v| v.payload)
                        });
                        let payload = raw_payload.map(|opt_pt| opt_pt.map(|pt| subst_ty(pt, &uargs, self.env)));
                        match payload {
                            None => {
                                self.diags.error(
                                    ErrorCode::E0132,
                                    name.span,
                                    format!(
                                        "union `{}` has no variant `.{}`",
                                        self.fmt_ty(ty),
                                        self.interner.get(name.sym)
                                    ),
                                );
                            }
                            Some(None) => {
                                if !args.is_empty() {
                                    self.diags.error(
                                        ErrorCode::E0230,
                                        pat.span,
                                        format!("variant `.{}` has no payload", self.interner.get(name.sym)),
                                    );
                                }
                            }
                            Some(Some(pty)) => {
                                if args.len() != 1 {
                                    self.diags.error(
                                        ErrorCode::E0230,
                                        pat.span,
                                        format!("variant `.{}` expects 1 payload pattern", self.interner.get(name.sym)),
                                    );
                                } else {
                                    self.bind_pattern(&args[0], pty, is_mut_allowed);
                                }
                            }
                        }
                    }
                    Ty::Enum(d) => {
                        if !args.is_empty() {
                            self.diags.error(ErrorCode::E0230, pat.span, "enum variants have no payloads");
                        }
                        if let Some(meta) = self.env.enums.get(&d)
                            && !meta.variants.iter().any(|(n, _)| *n == name.sym)
                        {
                            self.diags.error(
                                ErrorCode::E0132,
                                name.span,
                                format!("unknown enum variant `.{}`", self.interner.get(name.sym)),
                            );
                        }
                    }
                    Ty::ErrorSet(src) => {
                        if !self.error_union_has_variant(&[src], name.sym) {
                            self.diags.error(
                                ErrorCode::E0132,
                                name.span,
                                format!("unknown error variant `.{}`", self.interner.get(name.sym)),
                            );
                        }
                    }
                    Ty::ErrorUnion { ok, errors } => {
                        let vname = self.interner.get(name.sym);
                        if vname == "Ok" {
                            if let Some(p) = args.first() {
                                self.bind_pattern(p, ok, is_mut_allowed);
                            }
                        } else if !self.error_union_has_variant(&errors, name.sym) {
                            self.diags.error(
                                ErrorCode::E0132,
                                name.span,
                                format!("unknown error variant `.{}`", vname),
                            );
                        }
                    }
                    Ty::Error => {}
                    _ => {
                        self.diags.error(
                            ErrorCode::E0230,
                            pat.span,
                            format!("variant pattern `.{}` cannot match `{}`", self.interner.get(name.sym), self.fmt_ty(ty)),
                        );
                    }
                }
            }
            PatKind::Struct { fields, .. } => {
                match self.env.get(ty).clone() {
                    Ty::Struct { def, args } => {
                        if let Some(meta) = self.env.structs.get(&def).cloned() {
                            for fp in fields {
                                if let Some(mf) = meta.fields.iter().find(|f| f.name == fp.name.sym) {
                                    let fty = subst_ty(mf.ty, &args, self.env);
                                    if let Some(sub) = &fp.pat {
                                        self.bind_pattern(sub, fty, is_mut_allowed);
                                    } else {
                                        self.bind_local(fp.name.sym, fp.name.span, fty, false, false, false, false);
                                    }
                                } else {
                                    self.diags.error(
                                        ErrorCode::E0132,
                                        fp.name.span,
                                        format!("unknown struct field `{}` in pattern", self.interner.get(fp.name.sym)),
                                    );
                                }
                            }
                        }
                    }
                    Ty::AnonStruct(mfields) => {
                        for fp in fields {
                            if let Some((_, fty)) = mfields.iter().find(|(n, _)| *n == fp.name.sym) {
                                if let Some(sub) = &fp.pat {
                                    self.bind_pattern(sub, *fty, is_mut_allowed);
                                } else {
                                    self.bind_local(fp.name.sym, fp.name.span, *fty, false, false, false, false);
                                }
                            } else {
                                self.diags.error(
                                    ErrorCode::E0132,
                                    fp.name.span,
                                    format!("unknown struct field `{}` in pattern", self.interner.get(fp.name.sym)),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn unify_branch_types(&mut self, a: TyId, b: TyId, span: Span) -> TyId {
        if a == b || b == TyId::NEVER || a == TyId::ERROR {
            return a;
        }
        if a == TyId::NEVER || b == TyId::ERROR {
            return b;
        }
        if a == TyId::COMPTIME_INT && self.env.is_concrete_integer(b) {
            return b;
        }
        if b == TyId::COMPTIME_INT && self.env.is_concrete_integer(a) {
            return a;
        }
        if a == TyId::COMPTIME_FLOAT && self.env.is_float(b) {
            return b;
        }
        if b == TyId::COMPTIME_FLOAT && self.env.is_float(a) {
            return a;
        }
        self.expect_assignable(b, a, span);
        a
    }

    fn expect_assignable(&mut self, from: TyId, to: TyId, span: Span) {
        if from == to || from == TyId::NEVER || from == TyId::ERROR || to == TyId::ERROR {
            return;
        }
        if from == TyId::COMPTIME_INT && (self.env.is_integer(to) || to == TyId::COMPTIME_INT) {
            return;
        }
        if from == TyId::COMPTIME_FLOAT && self.env.is_float(to) {
            return;
        }
        // `T` coerces into `?T` and into `T ! E` (including chained coercions like `*T -> *dyn Iface -> ?*dyn Iface`).
        match self.env.get(to).clone() {
            Ty::Optional(inner) => {
                if from == inner || (from == TyId::COMPTIME_INT && self.env.is_concrete_integer(inner)) {
                    return;
                }
                let prev = self.diags.diagnostics().len();
                self.expect_assignable(from, inner, span);
                if self.diags.diagnostics().len() == prev {
                    return;
                }
                self.diags.truncate(prev);
            }
            Ty::ErrorUnion { ok, .. }
                if from == ok || (from == TyId::COMPTIME_INT && self.env.is_concrete_integer(ok)) =>
            {
                return;
            }
            _ => {}
        }
        // `*mut T` coerces to `*T`; `*mut [T]` coerces to `*[T]`.
        if let (
            Ty::Borrow { is_mut: true, inner: f_in },
            Ty::Borrow { is_mut: false, inner: t_in },
        ) = (self.env.get(from), self.env.get(to))
            && f_in == t_in
        {
            return;
        }
        // `*T` -> `*dyn Iface` and `*mut T` -> `*mut dyn Iface` / `*dyn Iface` coercion!
        if let (
            Ty::Borrow { is_mut: fm, inner: crate::types::BorrowInner::Single(concrete_ty) },
            Ty::Borrow { is_mut: tm, inner: crate::types::BorrowInner::Single(dyn_inner) },
        ) = (self.env.get(from).clone(), self.env.get(to).clone())
            && let Ty::Dyn(iface_def) = self.env.get(dyn_inner).clone()
            && !matches!(self.env.get(concrete_ty), Ty::Dyn(_))
            && (!tm || fm)
        {
            if self.type_implements_iface(concrete_ty, iface_def) {
                self.table.insert_dyn_coercion(span, iface_def, concrete_ty);
                return;
            }
            self.diags.error(
                ErrorCode::E0501,
                span,
                format!(
                    "type `{}` does not implement interface `{}` required for `dyn` coercion",
                    self.fmt_ty(concrete_ty),
                    self.interner.get(self.prog.defs[iface_def.0 as usize].name)
                ),
            );
            return;
        }
        // `[N]T` borrow `* [N]T` -> `*[T]`.
        if let (
            Ty::Borrow { is_mut: fm, inner: crate::types::BorrowInner::Single(arr) },
            Ty::Borrow { is_mut: tm, inner: crate::types::BorrowInner::Slice(el) },
        ) = (self.env.get(from).clone(), self.env.get(to).clone())
            && let Ty::Array { elem, .. } = self.env.get(arr)
            && *elem == el
            && (!tm || fm)
        {
            return;
        }

        // Two distinct `Id[pool]` brands fail with `E0330` (including `?Id[pool]` targets).
        match (self.env.get(from).clone(), self.env.get(to).clone()) {
            (
                Ty::Builtin(BuiltinKind::IdBrand, a1),
                Ty::Builtin(BuiltinKind::IdBrand, a2),
            ) => {
                if a1 != a2 {
                    self.diags.error(
                        ErrorCode::E0330,
                        span,
                        format!(
                            "brand mismatch: `{}` cannot be assigned to `{}`",
                            self.fmt_ty(from),
                            self.fmt_ty(to)
                        ),
                    );
                }
                return;
            }
            (Ty::Builtin(BuiltinKind::IdBrand, a1), Ty::Optional(ti)) => {
                if let Ty::Builtin(BuiltinKind::IdBrand, a2) = self.env.get(ti).clone() {
                    if a1 != a2 {
                        self.diags.error(
                            ErrorCode::E0330,
                            span,
                            format!(
                                "brand mismatch: `{}` cannot be assigned to `{}`",
                                self.fmt_ty(from),
                                self.fmt_ty(to)
                            ),
                        );
                    }
                    return;
                }
            }
            (Ty::Optional(fi), Ty::Optional(ti)) => {
                if let (
                    Ty::Builtin(BuiltinKind::IdBrand, a1),
                    Ty::Builtin(BuiltinKind::IdBrand, a2),
                ) = (self.env.get(fi).clone(), self.env.get(ti).clone())
                {
                    if a1 != a2 {
                        self.diags.error(
                            ErrorCode::E0330,
                            span,
                            format!(
                                "brand mismatch: `{}` cannot be assigned to `{}`",
                                self.fmt_ty(from),
                                self.fmt_ty(to)
                            ),
                        );
                    }
                    return;
                }
            }
            _ => {}
        }

        // Distinguish ZM-05 numeric coercion (`E0105`), address mix (`E0211`), and general mismatch (`E0230`).
        if self.env.is_address(from) || self.env.is_address(to) {
            self.diags.error(
                ErrorCode::E0211,
                span,
                format!(
                    "incompatible address type: cannot assign `{}` to `{}`",
                    self.fmt_ty(from),
                    self.fmt_ty(to)
                ),
            );
            return;
        }
        if self.env.is_numeric(from) && self.env.is_numeric(to) {
            self.diags.error(
                ErrorCode::E0105,
                span,
                format!(
                    "ZM-05: implicit numeric conversion from `{}` to `{}` is forbidden; use .widen()/.trunc()/.to_float()",
                    self.fmt_ty(from),
                    self.fmt_ty(to)
                ),
            );
            return;
        }
        self.diags.error(
            ErrorCode::E0230,
            span,
            format!(
                "type mismatch: expected `{}`, found `{}`",
                self.fmt_ty(to),
                self.fmt_ty(from)
            ),
        );
    }
}

pub fn subst_ty(ty: TyId, args: &[GenericVal], env: &mut TypeEnv) -> TyId {
    if args.is_empty() {
        return ty;
    }
    match env.get(ty).clone() {
        Ty::GenericParam { idx, .. } => match args.get(idx as usize) {
            Some(GenericVal::Ty(t)) if *t != TyId::ERROR => *t,
            _ => ty,
        },
        Ty::Optional(inner) => {
            let i = subst_ty(inner, args, env);
            env.intern(Ty::Optional(i))
        }
        Ty::ErrorUnion { ok, errors } => {
            let o = subst_ty(ok, args, env);
            env.intern(Ty::ErrorUnion { ok: o, errors })
        }
        Ty::Borrow { is_mut, inner } => {
            let ni = match inner {
                crate::types::BorrowInner::Single(t) => crate::types::BorrowInner::Single(subst_ty(t, args, env)),
                crate::types::BorrowInner::Slice(t) => crate::types::BorrowInner::Slice(subst_ty(t, args, env)),
                crate::types::BorrowInner::Str => crate::types::BorrowInner::Str,
            };
            env.intern(Ty::Borrow { is_mut, inner: ni })
        }
        Ty::RawPtr { is_mut, inner } => {
            let i = subst_ty(inner, args, env);
            env.intern(Ty::RawPtr { is_mut, inner: i })
        }
        Ty::Array { len, elem } => {
            let e = subst_ty(elem, args, env);
            env.intern(Ty::Array { len, elem: e })
        }
        Ty::Tuple(elems) => {
            let ne = elems.into_iter().map(|t| subst_ty(t, args, env)).collect();
            env.intern(Ty::Tuple(ne))
        }
        Ty::FnPtr { params, ret } => {
            let np = params.into_iter().map(|t| subst_ty(t, args, env)).collect();
            let nr = subst_ty(ret, args, env);
            env.intern(Ty::FnPtr { params: np, ret: nr })
        }
        Ty::Struct { def, args: sa } => {
            let nsa = sa
                .into_iter()
                .map(|g| match g {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty(t, args, env)),
                    other => other,
                })
                .collect();
            env.intern(Ty::Struct { def, args: nsa })
        }
        Ty::Union { def, args: ua } => {
            let nua = ua
                .into_iter()
                .map(|g| match g {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty(t, args, env)),
                    other => other,
                })
                .collect();
            env.intern(Ty::Union { def, args: nua })
        }
        Ty::Pool { def, args: pa } => {
            let npa = pa
                .into_iter()
                .map(|g| match g {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty(t, args, env)),
                    other => other,
                })
                .collect();
            env.intern(Ty::Pool { def, args: npa })
        }
        Ty::Builtin(bk, ba) => {
            let nba = ba
                .into_iter()
                .map(|g| match g {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty(t, args, env)),
                    other => other,
                })
                .collect();
            env.intern(Ty::Builtin(bk, nba))
        }
        _ => ty,
    }
}

struct MatchCoverage {
    has_wild: bool,
    seen_true: bool,
    seen_false: bool,
    seen_null: bool,
    seen_some: bool,
    variants: BTreeSet<Symbol>,
    int_ranges: Vec<(i128, i128)>,
    all_variants: Vec<Symbol>,
    int_domain: Option<(i128, i128)>,
}

impl MatchCoverage {
    fn new(scrut: TyId, env: &TypeEnv) -> Self {
        let mut all_variants = Vec::new();
        let mut int_domain = None;
        match env.get(scrut) {
            Ty::Enum(d) => {
                if let Some(m) = env.enums.get(d) {
                    all_variants = m.variants.iter().map(|(n, _)| *n).collect();
                }
            }
            Ty::Union { def, .. } => {
                if let Some(m) = env.unions.get(def) {
                    all_variants = m.variants.iter().map(|v| v.name).collect();
                }
            }
            Ty::ErrorSet(ErrorSource::User(d)) => {
                if let Some(m) = env.errors.get(d) {
                    all_variants = m.variants.clone();
                }
            }
            Ty::Builtin(BuiltinKind::Ordering, _) => {
                // Represented by variants; checked dynamically if enum-like.
            }
            Ty::Char => {
                int_domain = Some((0, 0x10FFFF));
            }
            Ty::Int { .. } => {
                int_domain = int_bounds(env.get(scrut));
            }
            _ => {}
        }
        Self {
            has_wild: false,
            seen_true: false,
            seen_false: false,
            seen_null: false,
            seen_some: false,
            variants: BTreeSet::new(),
            int_ranges: Vec::new(),
            all_variants,
            int_domain,
        }
    }

    fn is_already_exhaustive(&self) -> bool {
        if self.has_wild {
            return true;
        }
        if !self.all_variants.is_empty()
            && self.all_variants.iter().all(|v| self.variants.contains(v))
        {
            return true;
        }
        false
    }

    fn record_pattern(&mut self, pat: &Pat, env: &TypeEnv, interner: &Interner, diags: &mut DiagBag) {
        match &pat.kind {
            PatKind::Wildcard => {
                self.has_wild = true;
            }
            PatKind::Ident(_) => {
                if self.seen_null && !self.seen_some {
                    self.seen_some = true;
                } else {
                    self.has_wild = true;
                }
            }
            PatKind::BoolLit(true) => {
                if self.seen_true {
                    diags.error(ErrorCode::E0114, pat.span, "ZM-14: duplicate `true` match pattern");
                }
                self.seen_true = true;
                if self.seen_false {
                    self.has_wild = true;
                }
            }
            PatKind::BoolLit(false) => {
                if self.seen_false {
                    diags.error(ErrorCode::E0114, pat.span, "ZM-14: duplicate `false` match pattern");
                }
                self.seen_false = true;
                if self.seen_true {
                    self.has_wild = true;
                }
            }
            PatKind::Null => {
                if self.seen_null {
                    diags.error(ErrorCode::E0114, pat.span, "ZM-14: duplicate `null` match pattern");
                }
                self.seen_null = true;
                if self.seen_some {
                    self.has_wild = true;
                }
            }
            PatKind::Variant { name, args } => {
                let _ = (env, args);
                if self.interner_is_some(name.sym, interner) {
                    self.seen_some = true;
                    if self.seen_null {
                        self.has_wild = true;
                    }
                } else if !self.variants.insert(name.sym) {
                    diags.error(
                        ErrorCode::E0114,
                        pat.span,
                        format!("ZM-14: unreachable duplicate variant `.{}`", interner.get(name.sym)),
                    );
                }
            }
            PatKind::IntLit { neg, raw, range_end } => {
                let lo = parse_signed_int(interner.get(*raw), *neg).unwrap_or(0);
                let hi = match range_end {
                    Some((RangeKind::Inclusive, eneg, eraw)) => {
                        parse_signed_int(interner.get(*eraw), *eneg).unwrap_or(lo)
                    }
                    Some((RangeKind::Exclusive, eneg, eraw)) => {
                        parse_signed_int(interner.get(*eraw), *eneg).unwrap_or(lo).saturating_sub(1)
                    }
                    None => lo,
                };
                if lo <= hi {
                    self.int_ranges.push((lo, hi));
                    if self.int_domain_covered() {
                        self.has_wild = true;
                    }
                }
            }
            PatKind::CharLit { ch, range_end } => {
                let lo = *ch as i128;
                let hi = match range_end {
                    Some((RangeKind::Inclusive, c2)) => *c2 as i128,
                    Some((RangeKind::Exclusive, c2)) => (*c2 as i128).saturating_sub(1),
                    None => lo,
                };
                if lo <= hi {
                    self.int_ranges.push((lo, hi));
                    if self.int_domain_covered() {
                        self.has_wild = true;
                    }
                }
            }
            PatKind::Struct { .. } | PatKind::Tuple(_) => {
                // Simple single-struct/tuple destructuring is irrefutable when inner patterns bind/wildcard.
                if self.pat_is_irrefutable(pat) {
                    self.has_wild = true;
                }
            }
            PatKind::Error => {
                self.has_wild = true;
            }
        }
    }

    fn pat_is_irrefutable(&self, pat: &Pat) -> bool {
        match &pat.kind {
            PatKind::Wildcard | PatKind::Ident(_) | PatKind::Error => true,
            PatKind::Tuple(ps) => ps.iter().all(|p| self.pat_is_irrefutable(p)),
            PatKind::Struct { fields, .. } => fields
                .iter()
                .all(|f| f.pat.as_ref().is_none_or(|p| self.pat_is_irrefutable(p))),
            _ => false,
        }
    }

    fn interner_is_some(&self, sym: Symbol, interner: &Interner) -> bool {
        interner.get(sym) == "Some"
    }

    fn int_domain_covered(&self) -> bool {
        let Some((dmin, dmax)) = self.int_domain else {
            return false;
        };
        if self.int_ranges.is_empty() {
            return false;
        }
        let mut sorted = self.int_ranges.clone();
        sorted.sort_unstable();
        let mut reach = dmin.saturating_sub(1);
        for (lo, hi) in sorted {
            if lo > reach.saturating_add(1) {
                return false;
            }
            reach = reach.max(hi);
            if reach >= dmax {
                return true;
            }
        }
        reach >= dmax
    }

    fn verify_exhaustive(
        &self,
        scrut: TyId,
        env: &TypeEnv,
        interner: &Interner,
        diags: &mut DiagBag,
        span: Span,
    ) {
        if self.has_wild {
            return;
        }
        match env.get(scrut) {
            Ty::Error => {}
            Ty::Bool => {
                if !(self.seen_true && self.seen_false) {
                    diags.error(
                        ErrorCode::E0104,
                        span,
                        "ZM-04: non-exhaustive `match` on `bool`",
                    );
                }
            }
            Ty::Optional(_) => {
                if !(self.seen_null && self.seen_some) {
                    diags.error(
                        ErrorCode::E0104,
                        span,
                        "ZM-04: non-exhaustive `match` on optional type (must handle `null` and value)",
                    );
                }
            }
            Ty::Enum(_) | Ty::Union { .. } | Ty::ErrorSet(_) => {
                let missing: Vec<&str> = self
                    .all_variants
                    .iter()
                    .filter(|v| !self.variants.contains(v))
                    .map(|v| interner.get(*v))
                    .collect();
                if !missing.is_empty() {
                    diags.error(
                        ErrorCode::E0104,
                        span,
                        format!(
                            "ZM-04: non-exhaustive `match`; missing variants: {}",
                            missing.join(", ")
                        ),
                    );
                }
            }
            Ty::Int { .. } | Ty::Char => {
                if !self.int_domain_covered() {
                    diags.error(
                        ErrorCode::E0104,
                        span,
                        "ZM-04: non-exhaustive `match` on scalar range (add a `_` arm)",
                    );
                }
            }
            _ => {
                diags.error(
                    ErrorCode::E0104,
                    span,
                    "ZM-04: non-exhaustive `match` (add a `_` catch-all arm)",
                );
            }
        }
    }
}

fn parse_signed_int(raw: &str, neg: bool) -> Option<i128> {
    let u = parse_int_literal(raw)?;
    if neg {
        if u == (i128::MAX as u128) + 1 {
            Some(i128::MIN)
        } else {
            Some(-(u as i128))
        }
    } else {
        i128::try_from(u).ok()
    }
}

fn extract_root_ident_sym(mut expr: &Expr) -> Option<Symbol> {
    loop {
        match &expr.kind {
            ExprKind::Ident(s) => return Some(*s),
            ExprKind::Unary { op: UnaryOp::Borrow | UnaryOp::BorrowMut, expr: inner }
            | ExprKind::Paren(inner)
            | ExprKind::Deref(inner) => expr = inner,
            ExprKind::Field { base, .. } => expr = base,
            _ => return None,
        }
    }
}

pub fn replace_ty(ty: TyId, from: TyId, to: TyId, env: &mut TypeEnv) -> TyId {
    if ty == from {
        return to;
    }
    match env.get(ty).clone() {
        Ty::Borrow { is_mut, inner } => {
            let new_inner = match inner {
                crate::types::BorrowInner::Single(t) => {
                    crate::types::BorrowInner::Single(replace_ty(t, from, to, env))
                }
                crate::types::BorrowInner::Slice(t) => {
                    crate::types::BorrowInner::Slice(replace_ty(t, from, to, env))
                }
                crate::types::BorrowInner::Str => crate::types::BorrowInner::Str,
            };
            env.intern(Ty::Borrow { is_mut, inner: new_inner })
        }
        Ty::Optional(t) => {
            let nt = replace_ty(t, from, to, env);
            env.intern(Ty::Optional(nt))
        }
        Ty::ErrorUnion { ok, errors } => {
            let nok = replace_ty(ok, from, to, env);
            env.intern(Ty::ErrorUnion { ok: nok, errors })
        }
        Ty::RawPtr { is_mut, inner } => {
            let ni = replace_ty(inner, from, to, env);
            env.intern(Ty::RawPtr { is_mut, inner: ni })
        }
        _ => ty,
    }
}

pub fn ty_contains_generic_param(ty: TyId, env: &TypeEnv) -> bool {
    match env.get(ty) {
        Ty::GenericParam { .. } => true,
        Ty::Borrow { inner: crate::types::BorrowInner::Single(t) | crate::types::BorrowInner::Slice(t), .. }
        | Ty::Optional(t)
        | Ty::RawPtr { inner: t, .. }
        | Ty::Array { elem: t, .. } => ty_contains_generic_param(*t, env),
        Ty::ErrorUnion { ok, .. } => ty_contains_generic_param(*ok, env),
        Ty::Struct { args, .. } | Ty::Union { args, .. } | Ty::Pool { args, .. } | Ty::Builtin(_, args) => {
            args.iter().any(|a| matches!(a, GenericVal::Ty(t) if ty_contains_generic_param(*t, env)))
        }
        Ty::Tuple(ts) => ts.iter().any(|&t| ty_contains_generic_param(t, env)),
        Ty::FnPtr { params, ret } => {
            params.iter().any(|&t| ty_contains_generic_param(t, env)) || ty_contains_generic_param(*ret, env)
        }
        _ => false,
    }
}
