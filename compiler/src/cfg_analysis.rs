#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{CompilationUnit, DeclKind};
use crate::borrow::{
    self, is_borrow_source, paths_conflict, view_source_index, AccessPath, PathIndexKey, PathSeg,
    ViewSourceError,
};
use crate::cfg::{
    BlockId, CfgBuiltinOp, CfgCallee, CfgConst, CfgFnKey, CfgFunction, CfgOperand, CfgPlace,
    CfgProgram, CfgProj, CfgRvalue, CfgStmt, CfgStmtKind, CfgTerm, LocalId,
};
use crate::check::TypeTable;
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{BuiltinKind, ResolvedProgram, TopDefKind};
use crate::span::{Interner, Span, Symbol};
use crate::types::{FnSig, GenericVal, Ty, TyId, TypeEnv};

/// Run ownership, borrow/NLL, linearity, view provenance, and pool epoch verification across the CFG.
pub fn analyze_program(
    units: &[CompilationUnit],
    prog: &ResolvedProgram,
    env: &TypeEnv,
    type_table: &TypeTable,
    cfg: &CfgProgram,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    for f in &cfg.functions {
        if matches!(f.key, CfgFnKey::GenericInst(_)) {
            continue;
        }
        if f.unsupported.is_empty() {
            analyze_function(f, cfg, env, interner, diags);
        } else if let Some((mod_id, sig, body)) = find_ast_fn(units, prog, env, f.key) {
            borrow::check_single_fn(mod_id, sig, body, prog, env, type_table, interner, diags);
        }
    }
}

fn find_ast_fn<'a>(
    units: &'a [CompilationUnit],
    prog: &ResolvedProgram,
    env: &'a TypeEnv,
    key: CfgFnKey,
) -> Option<(crate::resolve::ModuleId, &'a FnSig, &'a crate::ast::Block)> {
    match key {
        CfgFnKey::TopDef(def_id) => {
            let def = prog.defs.get(def_id.0 as usize)?;
            let ui = prog.modules[def.module.0 as usize].unit_idx?;
            let sig = env.fns.get(&def_id)?;
            match def.kind {
                TopDefKind::Fn(di) => {
                    if let DeclKind::Fn(f) = &units[ui].decls[di].kind {
                        return Some((def.module, sig, &f.body));
                    }
                }
                TopDefKind::Sys(di) => {
                    if let DeclKind::Sys(s) = &units[ui].decls[di].kind {
                        return Some((def.module, sig, &s.body));
                    }
                }
                _ => {}
            }
            None
        }
        CfgFnKey::ImplMethod { impl_idx, sub_idx } => {
            let im = env.impls.get(impl_idx)?;
            let ui = prog.modules[im.module.0 as usize].unit_idx?;
            let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
                return None;
            };
            let (_, _, sig) = im.methods.iter().find(|&&(_, si, _)| si == sub_idx)?;
            if let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind {
                return Some((im.module, sig, &f.body));
            }
            None
        }
        CfgFnKey::GenericInst(_) | CfgFnKey::Closure(_) => None,
    }
}

struct GraphInfo {
    rpo: Vec<BlockId>,
    rpo_idx: Vec<usize>,
    preds: Vec<Vec<BlockId>>,
    idom: Vec<Option<BlockId>>,
}

impl GraphInfo {
    fn build(f: &CfgFunction) -> Self {
        let n = f.blocks.len();
        let mut reachable = vec![false; n];
        let mut post = Vec::new();
        let mut visited = vec![false; n];

        fn dfs(
            u: BlockId,
            f: &CfgFunction,
            visited: &mut [bool],
            reachable: &mut [bool],
            post: &mut Vec<BlockId>,
        ) {
            visited[u.0 as usize] = true;
            reachable[u.0 as usize] = true;
            for v in f.block(u).successors() {
                if (v.0 as usize) < visited.len() && !visited[v.0 as usize] {
                    dfs(v, f, visited, reachable, post);
                }
            }
            post.push(u);
        }

        dfs(f.entry, f, &mut visited, &mut reachable, &mut post);
        post.reverse();
        let rpo = post;
        let mut rpo_idx = vec![usize::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_idx[b.0 as usize] = i;
        }

        let mut preds = vec![Vec::new(); n];
        for &u in &rpo {
            for v in f.block(u).successors() {
                if (v.0 as usize) < n && reachable[v.0 as usize] {
                    preds[v.0 as usize].push(u);
                }
            }
        }

        let mut idom = vec![None; n];
        idom[f.entry.0 as usize] = Some(f.entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new_idom: Option<BlockId> = None;
                for &p in &preds[b.0 as usize] {
                    if idom[p.0 as usize].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(&idom, &rpo_idx, p, cur),
                    });
                }
                if new_idom.is_some() && idom[b.0 as usize] != new_idom {
                    idom[b.0 as usize] = new_idom;
                    changed = true;
                }
            }
        }

        Self {
            rpo,
            rpo_idx,
            preds,
            idom,
        }
    }

    fn dominates(&self, a: BlockId, mut b: BlockId) -> bool {
        if a == b {
            return true;
        }
        while b != self.rpo[0] {
            let Some(p) = self.idom[b.0 as usize] else {
                return false;
            };
            if p == a {
                return true;
            }
            if p == b {
                break;
            }
            b = p;
        }
        a == b
    }
}

fn intersect(idom: &[Option<BlockId>], rpo_idx: &[usize], mut b1: BlockId, mut b2: BlockId) -> BlockId {
    while b1 != b2 {
        while rpo_idx[b1.0 as usize] > rpo_idx[b2.0 as usize] {
            b1 = idom[b1.0 as usize].unwrap_or(b1);
        }
        while rpo_idx[b2.0 as usize] > rpo_idx[b1.0 as usize] {
            b2 = idom[b2.0 as usize].unwrap_or(b2);
        }
    }
    b1
}

fn has_borrow(env: &TypeEnv, ty: TyId) -> bool {
    env.contains_second_class_ref(ty) || env.contains_borrowing_value(ty)
}

fn analyze_function(
    f: &CfgFunction,
    cfg: &CfgProgram,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    let g = GraphInfo::build(f);
    analyze_view_provenance(f, cfg, &g, env, interner, diags);
    analyze_linearity(f, &g, env, interner, diags);
    analyze_borrows_and_nll(f, cfg, &g, env, interner, diags);
    analyze_pool_epochs(f, &g, env, interner, diags);
}

// ============================================================================
// Pass 1: View Provenance (Forward Fixed-Point)
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Origin {
    Bottom,
    NoBorrow,
    ParamBorrow(Symbol),
    LocalStack(Symbol),
    Other,
}

impl Origin {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x,
            (Self::NoBorrow, x) | (x, Self::NoBorrow) => x,
            (a, b) if a == b => a,
            (Self::LocalStack(s), Self::ParamBorrow(_))
            | (Self::ParamBorrow(_), Self::LocalStack(s)) => Self::LocalStack(s),
            _ => Self::Other,
        }
    }
}

fn analyze_view_provenance(
    f: &CfgFunction,
    cfg: &CfgProgram,
    g: &GraphInfo,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    if !f.sig.is_view {
        return;
    }
    let expected_source = match view_source_index(env, &f.sig) {
        Ok(i) => f.sig.params[i].name,
        Err(err) => {
            let first_sp = f.sig.params.first().map(|p| p.span).unwrap_or(f.body_span);
            match err {
                ViewSourceError::MultipleMarkers => diags.error(
                    ErrorCode::E0310,
                    first_sp,
                    "`view fn` cannot have multiple `from` source parameters",
                ),
                ViewSourceError::NotBorrowed(sp) => diags.error(
                    ErrorCode::E0310,
                    sp,
                    "`from` must mark a reference or borrowing value",
                ),
                ViewSourceError::Missing => diags.error(
                    ErrorCode::E0310,
                    first_sp,
                    "`view fn` requires `self`, a single borrow parameter, or explicit `from` marker to anchor returned borrow",
                ),
            }
            return;
        }
    };

    if !has_borrow(env, f.sig.ret) {
        return;
    }

    let num_locals = f.locals.len();
    let num_blocks = f.blocks.len();
    let mut in_states = vec![vec![Origin::Bottom; num_locals]; num_blocks];
    let mut entry_state = vec![Origin::NoBorrow; num_locals];
    for p in &f.sig.params {
        if let Some(lid) = f.locals.iter().find(|l| l.is_param && l.name == Some(p.name)).map(|l| l.id)
            && is_borrow_source(env, p)
        {
            entry_state[lid.0 as usize] = Origin::ParamBorrow(p.name);
        }
    }
    in_states[f.entry.0 as usize] = entry_state;

    let mut changed = true;
    while changed {
        changed = false;
        for &bb in &g.rpo {
            let bi = bb.0 as usize;
            if bi != f.entry.0 as usize {
                let mut joined = vec![Origin::Bottom; num_locals];
                for &p in &g.preds[bi] {
                    if in_states[p.0 as usize].iter().all(|&o| o == Origin::Bottom) {
                        continue;
                    }
                    let mut out = in_states[p.0 as usize].clone();
                    for stmt in &f.block(p).stmts {
                        step_provenance_stmt(f, cfg, stmt, &mut out, env);
                    }
                    for k in 0..num_locals {
                        joined[k] = joined[k].join(out[k]);
                    }
                }
                if joined != in_states[bi] {
                    in_states[bi] = joined;
                    changed = true;
                }
            }
        }
    }

    // Verification pass (single pass over converged CFG)
    for &bb in &g.rpo {
        let mut st = in_states[bb.0 as usize].clone();
        if st.iter().all(|&o| o == Origin::Bottom) {
            continue;
        }
        let b = f.block(bb);
        for stmt in &b.stmts {
            step_provenance_stmt(f, cfg, stmt, &mut st, env);
        }
        if let CfgTerm::Return { value: Some(op), span } = &b.term {
            let is_err_ret = b.is_cleanup && b.error_site_span.is_some();
            if is_err_ret {
                continue;
            }
            let ret_orig = eval_operand_origin(f, op, &st, env);
            let source_name = interner.get(expected_source);
            let msg = match ret_orig {
                Origin::NoBorrow | Origin::Bottom => None,
                Origin::ParamBorrow(sym) if sym == expected_source => None,
                Origin::ParamBorrow(sym) => Some(format!(
                    "`view fn` returns a borrow from `{}`; expected a borrow derived from `{source_name}`",
                    interner.get(sym)
                )),
                Origin::LocalStack(sym) => Some(format!(
                    "`view fn` cannot return borrow derived from local variable `{}`",
                    interner.get(sym)
                )),
                Origin::Other => Some(format!(
                    "cannot prove `view fn` result is derived from `{source_name}`"
                )),
            };
            if let Some(m) = msg {
                diags.error(ErrorCode::E0310, *span, m);
            }
        }
    }
}

fn step_provenance_stmt(
    f: &CfgFunction,
    cfg: &CfgProgram,
    stmt: &CfgStmt,
    st: &mut [Origin],
    env: &TypeEnv,
) {
    match &stmt.kind {
        CfgStmtKind::StorageLive(id) => {
            if !f.local(*id).is_param {
                st[id.0 as usize] = Origin::NoBorrow;
            }
        }
        CfgStmtKind::Assign { dst, rvalue } => {
            let o = eval_rvalue_origin(f, rvalue, st, env);
            if dst.projs.is_empty() {
                st[dst.local.0 as usize] = o;
            }
        }
        CfgStmtKind::Call { dst: Some(dst), callee, args, recv_place, .. } => {
            let o = match callee {
                CfgCallee::Builtin(CfgBuiltinOp::StrAsBytes | CfgBuiltinOp::BytesAsStr | CfgBuiltinOp::ArgsGet)
                | CfgCallee::Builtin(CfgBuiltinOp::SliceAt { .. } | CfgBuiltinOp::SliceGet { .. } | CfgBuiltinOp::SliceSub { .. })
                | CfgCallee::Builtin(CfgBuiltinOp::BufAsSlice { .. } | CfgBuiltinOp::BufAt { .. } | CfgBuiltinOp::BufGet { .. } | CfgBuiltinOp::BufSub { .. })
                | CfgCallee::Builtin(CfgBuiltinOp::SlotPoolGet { .. } | CfgBuiltinOp::SlotPoolAt(_)) => {
                    if let Some(rp) = recv_place {
                        eval_place_address_origin(f, rp, st, env)
                    } else {
                        args.first().map_or(Origin::Other, |a| eval_operand_origin(f, a, st, env))
                    }
                }
                CfgCallee::Builtin(CfgBuiltinOp::ArrayAt { .. } | CfgBuiltinOp::ArrayGet { .. } | CfgBuiltinOp::ArraySub { .. }) => {
                    if let Some(rp) = recv_place {
                        eval_place_address_origin(f, rp, st, env)
                    } else {
                        args.first().map_or(Origin::Other, |a| eval_operand_origin(f, a, st, env))
                    }
                }
                CfgCallee::Builtin(CfgBuiltinOp::ArenaAlloc | CfgBuiltinOp::ArenaPoolInit(_)) => {
                    args.first().map_or(Origin::Other, |a| eval_operand_origin(f, a, st, env))
                }
                CfgCallee::DirectFn(def_id) => {
                    if let Some(sig) = env.fns.get(def_id)
                        && sig.is_view
                        && let Ok(idx) = view_source_index(env, sig)
                        && let Some(arg) = args.get(idx)
                    {
                        eval_operand_origin(f, arg, st, env)
                    } else if has_borrow(env, dst.ty) {
                        Origin::Other
                    } else {
                        Origin::NoBorrow
                    }
                }
                CfgCallee::Method { impl_idx, sub_idx, is_view } => {
                    if *is_view
                        && let Some(im) = env.impls.get(*impl_idx)
                        && let Some((_, _, sig)) = im.methods.iter().find(|&&(_, si, _)| si == *sub_idx)
                        && let Ok(idx) = view_source_index(env, sig)
                        && let Some(arg) = args.get(idx)
                    {
                        eval_operand_origin(f, arg, st, env)
                    } else if has_borrow(env, dst.ty) {
                        Origin::Other
                    } else {
                        Origin::NoBorrow
                    }
                }
                CfgCallee::GenericInst { inst_id, is_view } => {
                    if *is_view
                        && let Some(&fi) = cfg.by_key.get(&CfgFnKey::GenericInst(*inst_id))
                        && let Ok(idx) = view_source_index(env, &cfg.functions[fi].sig)
                        && let Some(arg) = args.get(idx)
                    {
                        eval_operand_origin(f, arg, st, env)
                    } else if has_borrow(env, dst.ty) {
                        Origin::Other
                    } else {
                        Origin::NoBorrow
                    }
                }
                _ => {
                    if has_borrow(env, dst.ty) {
                        Origin::Other
                    } else {
                        Origin::NoBorrow
                    }
                }
            };
            if dst.projs.is_empty() {
                st[dst.local.0 as usize] = o;
            }
        }
        _ => {}
    }
}

fn eval_rvalue_origin(f: &CfgFunction, rv: &CfgRvalue, st: &[Origin], env: &TypeEnv) -> Origin {
    match rv {
        CfgRvalue::Use(op)
        | CfgRvalue::UnwrapOptional(op)
        | CfgRvalue::UnwrapErrorOk(op)
        | CfgRvalue::FieldExtract { operand: op, .. }
        | CfgRvalue::MakeDyn { ptr: op, .. } => eval_operand_origin(f, op, st, env),
        CfgRvalue::MakeClosure { env_ptr, .. } => env_ptr
            .as_ref()
            .map_or(Origin::NoBorrow, |op| eval_operand_origin(f, op, st, env)),
        CfgRvalue::Borrow { place, .. } => eval_place_address_origin(f, place, st, env),
        CfgRvalue::Slice { base, .. } => {
            let bt = base.ty(f, env);
            if matches!(env.get(bt), Ty::Borrow { .. }) {
                eval_operand_origin(f, base, st, env)
            } else if let CfgOperand::Copy(p) | CfgOperand::Move(p) = base {
                eval_place_address_origin(f, p, st, env)
            } else {
                Origin::Other
            }
        }
        CfgRvalue::Cast { kind, operand, to_ty } => match kind {
            crate::cfg::CastKind::WrapErr => Origin::NoBorrow,
            _ if has_borrow(env, *to_ty) => eval_operand_origin(f, operand, st, env),
            _ => Origin::NoBorrow,
        },
        CfgRvalue::Aggregate { elems, .. } => {
            let mut acc = Origin::NoBorrow;
            for e in elems {
                acc = acc.join(eval_operand_origin(f, e, st, env));
            }
            acc
        }
        _ => Origin::NoBorrow,
    }
}

fn eval_operand_origin(f: &CfgFunction, op: &CfgOperand, st: &[Origin], env: &TypeEnv) -> Origin {
    let ty = op.ty(f, env);
    if !has_borrow(env, ty) {
        return Origin::NoBorrow;
    }
    match op {
        CfgOperand::Const(CfgConst::Null(_) | CfgConst::Str(_) | CfgConst::Bytes(_)) => Origin::NoBorrow,
        CfgOperand::Const(_) => Origin::Other,
        CfgOperand::Copy(p) | CfgOperand::Move(p) => {
            let loc_orig = st[p.local.0 as usize];
            if loc_orig == Origin::Bottom {
                Origin::NoBorrow
            } else {
                loc_orig
            }
        }
    }
}

fn eval_place_address_origin(f: &CfgFunction, p: &CfgPlace, st: &[Origin], env: &TypeEnv) -> Origin {
    let base_ty = f.local(p.local).ty;
    if matches!(
        env.get(base_ty),
        Ty::Borrow { .. } | Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::ArenaStr | BuiltinKind::Arena, _)
    ) || p.projs.iter().any(|pr| matches!(pr, CfgProj::Deref | CfgProj::PoolCol { .. }))
    {
        return st[p.local.0 as usize];
    }
    if let Some(sym) = f.local(p.local).name {
        Origin::LocalStack(sym)
    } else {
        Origin::Other
    }
}

// ============================================================================
// Pass 2: Linearity Dataflow (E0109, E0321, E0323, E0324, E0325)
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LinState {
    Bot,
    Uninit,
    Live,
    Consumed,
    Conflict,
}

impl LinState {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bot, x) | (x, Self::Bot) => x,
            (Self::Conflict, _) | (_, Self::Conflict) => Self::Conflict,
            (a, b) if a == b => a,
            (Self::Live, Self::Consumed) | (Self::Consumed, Self::Live) => Self::Conflict,
            (Self::Uninit, Self::Consumed) | (Self::Consumed, Self::Uninit) => Self::Consumed,
            (Self::Uninit, Self::Live) | (Self::Live, Self::Uninit) => Self::Conflict,
            _ => Self::Conflict,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum BranchPredKind {
    IsError(LocalId),
    IsNonNull(LocalId),
}

fn analyze_linearity(
    f: &CfgFunction,
    g: &GraphInfo,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    let has_discard = f.blocks.iter().any(|b| {
        b.stmts
            .iter()
            .any(|s| matches!(s.kind, CfgStmtKind::DiscardLinearError { .. }))
    });
    let lin_locals: Vec<LocalId> = f
        .locals
        .iter()
        .filter(|l| l.is_linear)
        .map(|l| l.id)
        .collect();
    if lin_locals.is_empty() && !has_discard {
        return;
    }

    let mut lin_idx_of: BTreeMap<LocalId, usize> = BTreeMap::new();
    for (i, &lid) in lin_locals.iter().enumerate() {
        lin_idx_of.insert(lid, i);
    }
    let m = lin_locals.len();

    // Detect branch condition predicates (`IsError(eu)` / `IsNonNull(opt)`) per block for edge refinement.
    let mut block_pred_cond: Vec<Option<BranchPredKind>> = vec![None; f.blocks.len()];
    for &bb in &g.rpo {
        if let CfgTerm::Branch { cond: CfgOperand::Copy(cp) | CfgOperand::Move(cp), .. } = &f.block(bb).term
            && cp.projs.is_empty()
        {
            for stmt in f.block(bb).stmts.iter().rev() {
                if let CfgStmtKind::Assign { dst, rvalue } = &stmt.kind
                    && dst.projs.is_empty()
                    && dst.local == cp.local
                {
                    match rvalue {
                        CfgRvalue::IsError(CfgOperand::Copy(p) | CfgOperand::Move(p)) if p.projs.is_empty() => {
                            block_pred_cond[bb.0 as usize] = Some(BranchPredKind::IsError(p.local));
                        }
                        CfgRvalue::IsNonNull(CfgOperand::Copy(p) | CfgOperand::Move(p)) if p.projs.is_empty() => {
                            block_pred_cond[bb.0 as usize] = Some(BranchPredKind::IsNonNull(p.local));
                        }
                        _ => {}
                    }
                    break;
                }
            }
        }
    }

    let mut in_state = vec![vec![LinState::Bot; m]; f.blocks.len()];
    let mut out_state = vec![vec![LinState::Bot; m]; f.blocks.len()];
    let mut in_move_spans: Vec<Vec<Option<Span>>> = vec![vec![None; m]; f.blocks.len()];
    let mut out_move_spans: Vec<Vec<Option<Span>>> = vec![vec![None; m]; f.blocks.len()];

    let mut entry = vec![LinState::Uninit; m];
    for (i, &lid) in lin_locals.iter().enumerate() {
        if f.local(lid).is_param {
            entry[i] = LinState::Live;
        }
    }
    in_state[f.entry.0 as usize] = entry;

    let edge_state = |p: BlockId, succ: BlockId, base_out: &[LinState]| -> Vec<LinState> {
        let mut s = base_out.to_vec();
        if let Some(pred_kind) = block_pred_cond[p.0 as usize]
            && let CfgTerm::Branch { then_bb, else_bb, .. } = f.block(p).term
        {
            match pred_kind {
                BranchPredKind::IsError(eu_loc) if succ == then_bb => {
                    if let Some(&k) = lin_idx_of.get(&eu_loc) {
                        s[k] = LinState::Consumed;
                    }
                }
                BranchPredKind::IsNonNull(opt_loc) if succ == else_bb => {
                    if let Some(&k) = lin_idx_of.get(&opt_loc) {
                        s[k] = LinState::Consumed;
                    }
                }
                _ => {}
            }
        }
        s
    };

    // Fixed-point iteration
    let mut changed = true;
    while changed {
        changed = false;
        for &bb in &g.rpo {
            let bi = bb.0 as usize;
            let b = f.block(bb);
            if bi != f.entry.0 as usize {
                let mut joined = vec![LinState::Bot; m];
                let mut jspans = vec![None; m];
                for &p in &g.preds[bi] {
                    if out_state[p.0 as usize].iter().all(|&s| s == LinState::Bot) {
                        continue;
                    }
                    let es = edge_state(p, bb, &out_state[p.0 as usize]);
                    for k in 0..m {
                        joined[k] = joined[k].join(es[k]);
                        if !b.is_loop_header {
                            jspans[k] = jspans[k].or(out_move_spans[p.0 as usize][k]);
                        }
                    }
                }
                if joined != in_state[bi] || jspans != in_move_spans[bi] {
                    in_state[bi] = joined;
                    in_move_spans[bi] = jspans;
                    changed = true;
                }
            }
            if in_state[bi].iter().all(|&s| s == LinState::Bot) {
                continue;
            }
            let mut st = in_state[bi].clone();
            let mut mspans = if b.is_loop_header {
                vec![None; m]
            } else {
                in_move_spans[bi].clone()
            };
            for stmt in &b.stmts {
                step_linearity_stmt(f, stmt, &lin_idx_of, &mut st, &mut mspans, b, None);
            }
            if let CfgTerm::Return { value: Some(op), .. } = &b.term {
                step_linearity_operand(
                    f,
                    op,
                    stmt_dummy_span(&b.term),
                    &lin_idx_of,
                    &mut st,
                    &mut mspans,
                    b,
                    None,
                );
            }
            if st != out_state[bi] || mspans != out_move_spans[bi] {
                out_state[bi] = st;
                out_move_spans[bi] = mspans;
                changed = true;
            }
        }
    }

    // Single verification pass on converged states
    let mut reported_conflict = vec![false; m];
    for &bb in &g.rpo {
        let bi = bb.0 as usize;
        let b = f.block(bb);
        if in_state[bi].iter().all(|&s| s == LinState::Bot) {
            continue;
        }

        // Check join / loop header errors ONCE per merge block.
        if bi != f.entry.0 as usize {
            if b.is_loop_header {
                let header_depth = b.loop_entry_depth.unwrap_or(0);
                let mut entry_preds = Vec::new();
                let mut back_preds = Vec::new();
                for &p in &g.preds[bi] {
                    if out_state[p.0 as usize].iter().all(|&s| s == LinState::Bot) {
                        continue;
                    }
                    if g.dominates(bb, p) || g.rpo_idx[p.0 as usize] >= g.rpo_idx[bi] {
                        back_preds.push(p);
                    } else {
                        entry_preds.push(p);
                    }
                }
                for (k, &lid) in lin_locals.iter().enumerate() {
                    let loc = f.local(lid);
                    if loc.is_temp || loc.scope_depth > header_depth || reported_conflict[k] {
                        continue;
                    }
                    let mut entry_st = LinState::Bot;
                    for &ep in &entry_preds {
                        let es = edge_state(ep, bb, &out_state[ep.0 as usize]);
                        entry_st = entry_st.join(es[k]);
                    }
                    if entry_st == LinState::Live {
                        for &bp in &back_preds {
                            let bs = edge_state(bp, bb, &out_state[bp.0 as usize]);
                            if bs[k] == LinState::Consumed || out_move_spans[bp.0 as usize][k].is_some() {
                                let name = loc.name.map(|s| interner.get(s)).unwrap_or("_");
                                let sp = out_move_spans[bp.0 as usize][k].unwrap_or(loc.span);
                                diags.error(
                                    ErrorCode::E0324,
                                    sp,
                                    format!(
                                        "cannot consume outer `linear` variable `{}` in loop body without exiting via `break` or `return`",
                                        name
                                    ),
                                );
                                reported_conflict[k] = true;
                                break;
                            }
                        }
                    }
                }
            } else if g.preds[bi].len() >= 2 {
                for (k, &lid) in lin_locals.iter().enumerate() {
                    let loc = f.local(lid);
                    if loc.is_temp || reported_conflict[k] {
                        continue;
                    }
                    let mut seen_live = false;
                    let mut seen_consumed = false;
                    for &p in &g.preds[bi] {
                        if out_state[p.0 as usize].iter().all(|&s| s == LinState::Bot) {
                            continue;
                        }
                        let es = edge_state(p, bb, &out_state[p.0 as usize]);
                        match es[k] {
                            LinState::Live => seen_live = true,
                            LinState::Consumed => seen_consumed = true,
                            _ => {}
                        }
                    }
                    if seen_live && seen_consumed {
                        let name = loc.name.map(|s| interner.get(s)).unwrap_or("_");
                        diags.error(
                            ErrorCode::E0323,
                            b.join_span.unwrap_or(loc.span),
                            format!(
                                "`linear` variable `{}` has inconsistent consumption state across branches",
                                name
                            ),
                        );
                        reported_conflict[k] = true;
                    }
                }
            }
        }

        let mut st = in_state[bi].clone();
        for (k, &rc) in reported_conflict.iter().enumerate() {
            if rc {
                st[k] = LinState::Conflict;
            }
        }
        let mut mspans = vec![None; m];
        for stmt in &b.stmts {
            if let CfgStmtKind::DiscardLinearError { local, span } = &stmt.kind {
                diags.error(
                    ErrorCode::E0109,
                    *span,
                    "ZM-09: cannot drop `linear` value with `let _ = ...`; destructure fields or call a consuming function",
                );
                if let Some(lid) = local
                    && let Some(&k) = lin_idx_of.get(lid)
                {
                    st[k] = LinState::Conflict;
                }
                continue;
            }
            step_linearity_stmt(f, stmt, &lin_idx_of, &mut st, &mut mspans, b, Some((env, interner, diags)));
        }
        if let CfgTerm::Return { value: Some(op), span } = &b.term {
            step_linearity_operand(f, op, *span, &lin_idx_of, &mut st, &mut mspans, b, Some((env, interner, diags)));
        }
    }
}

fn stmt_dummy_span(t: &CfgTerm) -> Span {
    match t {
        CfgTerm::Return { span, .. }
        | CfgTerm::Branch { span, .. }
        | CfgTerm::SwitchInt { span, .. }
        | CfgTerm::Unreachable { span, .. } => *span,
        CfgTerm::Goto(_) => Span::DUMMY,
    }
}

fn step_linearity_stmt(
    f: &CfgFunction,
    stmt: &CfgStmt,
    lin_idx_of: &BTreeMap<LocalId, usize>,
    st: &mut [LinState],
    mspans: &mut [Option<Span>],
    bb: &crate::cfg::CfgBlock,
    mut report: Option<(&TypeEnv, &Interner, &mut DiagBag)>,
) {
    match &stmt.kind {
        CfgStmtKind::StorageLive(lid) => {
            if let Some(&k) = lin_idx_of.get(lid) {
                st[k] = LinState::Uninit;
                mspans[k] = None;
            }
        }
        CfgStmtKind::StorageDead(lid) => {
            if let Some(&k) = lin_idx_of.get(lid)
                && st[k] == LinState::Live
            {
                if let Some((_, interner, ref mut diags)) = report {
                    let loc = f.local(*lid);
                    if !loc.is_temp {
                        let name = loc.name.map(|s| interner.get(s)).unwrap_or("_");
                        let sp = bb.error_site_span.unwrap_or(loc.span);
                        let msg = if bb.error_site_span.is_some() {
                            format!("ZM-09: `linear` value `{name}` is unconsumed on error path (`try` / error `return`); add `errdefer`")
                        } else {
                            format!("ZM-09: `linear` value `{name}` dropped at scope exit without being consumed")
                        };
                        diags.error(ErrorCode::E0109, sp, msg);
                    }
                }
                st[k] = LinState::Conflict;
            }
        }
        CfgStmtKind::ConsumeDestructured(lid) => {
            if let Some(&k) = lin_idx_of.get(lid) {
                st[k] = LinState::Consumed;
            }
        }
        CfgStmtKind::Assign { dst, rvalue } => {
            match rvalue {
                CfgRvalue::Use(op)
                | CfgRvalue::UnwrapOptional(op)
                | CfgRvalue::UnwrapErrorOk(op)
                | CfgRvalue::UnwrapErrorErr(op)
                | CfgRvalue::Unary { operand: op, .. }
                | CfgRvalue::Cast { operand: op, .. }
                | CfgRvalue::FieldExtract { operand: op, .. }
                | CfgRvalue::MakeDyn { ptr: op, .. } => {
                    let sub = report.as_mut().map(|(e, i, d)| (*e, *i, &mut **d));
                    step_linearity_operand(f, op, stmt.span, lin_idx_of, st, mspans, bb, sub);
                }
                CfgRvalue::MakeClosure { env_ptr: Some(op), .. } => {
                    let sub = report.as_mut().map(|(e, i, d)| (*e, *i, &mut **d));
                    step_linearity_operand(f, op, stmt.span, lin_idx_of, st, mspans, bb, sub);
                }
                CfgRvalue::Aggregate { elems, .. } => {
                    for e in elems {
                        let sub = report.as_mut().map(|(ev, i, d)| (*ev, *i, &mut **d));
                        step_linearity_operand(f, e, stmt.span, lin_idx_of, st, mspans, bb, sub);
                    }
                }
                _ => {}
            }
            if dst.projs.is_empty()
                && let Some(&k) = lin_idx_of.get(&dst.local)
            {
                if st[k] == LinState::Live
                    && !f.local(dst.local).is_temp
                    && let Some((_, interner, ref mut diags)) = report
                {
                    let name = f.local(dst.local).name.map(|s| interner.get(s)).unwrap_or("_");
                    diags.error(
                        ErrorCode::E0109,
                        stmt.span,
                        format!("ZM-09: overwriting unconsumed `linear` variable `{name}`"),
                    );
                }
                st[k] = LinState::Live;
                mspans[k] = None;
            }
        }
        CfgStmtKind::Call { dst, args, consumes_recv, recv_place, .. } => {
            let mut moved_recv = false;
            if *consumes_recv
                && let Some(rp) = recv_place
                && rp.projs.is_empty()
                && let Some(&k) = lin_idx_of.get(&rp.local)
            {
                moved_recv = true;
                let sub = report.as_mut().map(|(e, i, d)| (*e, *i, &mut **d));
                consume_lin_local(f, rp.local, k, stmt.span, st, mspans, bb, sub);
            }
            for (i, a) in args.iter().enumerate() {
                if moved_recv && i == 0 {
                    continue;
                }
                let sub = report.as_mut().map(|(e, iv, d)| (*e, *iv, &mut **d));
                step_linearity_operand(f, a, stmt.span, lin_idx_of, st, mspans, bb, sub);
            }
            if let Some(d) = dst
                && d.projs.is_empty()
                && let Some(&k) = lin_idx_of.get(&d.local)
            {
                st[k] = LinState::Live;
                mspans[k] = None;
            }
        }
        _ => {}
    }
}

fn step_linearity_operand(
    f: &CfgFunction,
    op: &CfgOperand,
    span: Span,
    lin_idx_of: &BTreeMap<LocalId, usize>,
    st: &mut [LinState],
    mspans: &mut [Option<Span>],
    bb: &crate::cfg::CfgBlock,
    report: Option<(&TypeEnv, &Interner, &mut DiagBag)>,
) {
    if let CfgOperand::Move(p) = op
        && p.projs.is_empty()
        && let Some(&k) = lin_idx_of.get(&p.local)
    {
        consume_lin_local(f, p.local, k, span, st, mspans, bb, report);
    }
}

fn consume_lin_local(
    f: &CfgFunction,
    lid: LocalId,
    k: usize,
    span: Span,
    st: &mut [LinState],
    mspans: &mut [Option<Span>],
    bb: &crate::cfg::CfgBlock,
    mut report: Option<(&TypeEnv, &Interner, &mut DiagBag)>,
) {
    if mspans[k].is_none() {
        mspans[k] = Some(span);
    }
    match st[k] {
        LinState::Live => {
            st[k] = LinState::Consumed;
        }
        LinState::Consumed => {
            if let Some((_, interner, ref mut diags)) = report {
                let name = f.local(lid).name.map(|s| interner.get(s)).unwrap_or("_");
                if bb.is_errdefer {
                    diags.error(
                        ErrorCode::E0325,
                        bb.error_site_span.unwrap_or(span),
                        format!("`{name}` was already consumed before `errdefer` ran on error path"),
                    );
                } else {
                    diags.error(
                        ErrorCode::E0321,
                        span,
                        format!("`linear` value `{name}` was already consumed"),
                    );
                }
            }
            st[k] = LinState::Conflict;
        }
        _ => {}
    }
}

// ============================================================================
// Pass 3: Backward NLL Liveness + AccessPath Exclusivity (E0118)
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Loan {
    holder: LocalId,
    path: AccessPath,
    is_mut: bool,
}

fn analyze_borrows_and_nll(
    f: &CfgFunction,
    cfg: &CfgProgram,
    g: &GraphInfo,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    // Step 3A: Backward NLL liveness per block/statement.
    let num_blocks = f.blocks.len();
    let mut live_in: Vec<BTreeSet<LocalId>> = vec![BTreeSet::new(); num_blocks];
    let mut live_out: Vec<BTreeSet<LocalId>> = vec![BTreeSet::new(); num_blocks];

    let mut changed = true;
    while changed {
        changed = false;
        for &bb in g.rpo.iter().rev() {
            let bi = bb.0 as usize;
            let mut out = BTreeSet::new();
            for s in f.block(bb).successors() {
                if (s.0 as usize) < num_blocks {
                    out.extend(&live_in[s.0 as usize]);
                }
            }
            let mut cur = out.clone();
            collect_term_uses(&f.block(bb).term, &mut cur);
            for stmt in f.block(bb).stmts.iter().rev() {
                step_liveness_back_stmt(stmt, &mut cur);
            }
            if out != live_out[bi] || cur != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = cur;
                changed = true;
            }
        }
    }

    // Compute live_before_stmt[bb][i] cleanly.
    let mut live_before_stmt: Vec<Vec<BTreeSet<LocalId>>> = Vec::with_capacity(num_blocks);
    for bi in 0..num_blocks {
        let stmts = &f.blocks[bi].stmts;
        let mut vec_b = vec![BTreeSet::new(); stmts.len()];
        let mut cur = live_out[bi].clone();
        collect_term_uses(&f.blocks[bi].term, &mut cur);
        for i in (0..stmts.len()).rev() {
            step_liveness_back_stmt(&stmts[i], &mut cur);
            vec_b[i] = cur.clone();
        }
        live_before_stmt.push(vec_b);
    }

    // Step 3B: Forward loan dataflow with NLL expiration at each statement.
    let mut loans_in: Vec<BTreeSet<Loan>> = vec![BTreeSet::new(); num_blocks];
    let mut loans_out: Vec<BTreeSet<Loan>> = vec![BTreeSet::new(); num_blocks];

    let mut lchanged = true;
    while lchanged {
        lchanged = false;
        for &bb in &g.rpo {
            let bi = bb.0 as usize;
            if bi != f.entry.0 as usize {
                let mut joined = BTreeSet::new();
                for &p in &g.preds[bi] {
                    for l in &loans_out[p.0 as usize] {
                        if live_in[bi].contains(&l.holder) {
                            joined.insert(l.clone());
                        }
                    }
                }
                if joined != loans_in[bi] {
                    loans_in[bi] = joined;
                    lchanged = true;
                }
            }
            let mut active = loans_in[bi].clone();
            for (si, stmt) in f.block(bb).stmts.iter().enumerate() {
                let live_set = &live_before_stmt[bi][si];
                active.retain(|l| live_set.contains(&l.holder));
                step_loans_stmt(f, cfg, stmt, &mut active, env, None, interner);
            }
            active.retain(|l| live_out[bi].contains(&l.holder));
            if active != loans_out[bi] {
                loans_out[bi] = active;
                lchanged = true;
            }
        }
    }

    // Step 3C: Single verification pass emitting E0118 on conflicts.
    for &bb in &g.rpo {
        let bi = bb.0 as usize;
        let mut active = loans_in[bi].clone();
        for (si, stmt) in f.block(bb).stmts.iter().enumerate() {
            let live_set = &live_before_stmt[bi][si];
            active.retain(|l| live_set.contains(&l.holder));
            step_loans_stmt(f, cfg, stmt, &mut active, env, Some(diags), interner);
        }
    }
}

fn collect_term_uses(term: &CfgTerm, set: &mut BTreeSet<LocalId>) {
    match term {
        CfgTerm::Branch { cond, .. } => collect_op_uses(cond, set),
        CfgTerm::SwitchInt { discr, .. } => collect_op_uses(discr, set),
        CfgTerm::Return { value: Some(op), .. } => collect_op_uses(op, set),
        _ => {}
    }
}

fn collect_op_uses(op: &CfgOperand, set: &mut BTreeSet<LocalId>) {
    match op {
        CfgOperand::Copy(p) | CfgOperand::Move(p) => collect_place_uses(p, set),
        CfgOperand::Const(_) => {}
    }
}

fn collect_place_uses(p: &CfgPlace, set: &mut BTreeSet<LocalId>) {
    set.insert(p.local);
    for pr in &p.projs {
        if let CfgProj::Index { index, .. } = pr {
            collect_op_uses(index, set);
        }
    }
}

fn collect_rvalue_uses(rv: &CfgRvalue, set: &mut BTreeSet<LocalId>) {
    match rv {
        CfgRvalue::Use(op)
        | CfgRvalue::UnwrapOptional(op)
        | CfgRvalue::UnwrapErrorOk(op)
        | CfgRvalue::UnwrapErrorErr(op)
        | CfgRvalue::IsNonNull(op)
        | CfgRvalue::IsError(op)
        | CfgRvalue::MatchTag(op)
        | CfgRvalue::Unary { operand: op, .. }
        | CfgRvalue::Cast { operand: op, .. }
        | CfgRvalue::FieldExtract { operand: op, .. }
        | CfgRvalue::MakeDyn { ptr: op, .. } => collect_op_uses(op, set),
        CfgRvalue::MakeClosure { env_ptr, .. } => {
            if let Some(op) = env_ptr {
                collect_op_uses(op, set);
            }
        }
        CfgRvalue::Borrow { place, .. } => collect_place_uses(place, set),
        CfgRvalue::Binary { lhs, rhs, .. } => {
            collect_op_uses(lhs, set);
            collect_op_uses(rhs, set);
        }
        CfgRvalue::Slice { base, lo, hi } => {
            collect_op_uses(base, set);
            if let Some(l) = lo { collect_op_uses(l, set); }
            if let Some(h) = hi { collect_op_uses(h, set); }
        }
        CfgRvalue::Aggregate { elems, .. } => {
            for e in elems { collect_op_uses(e, set); }
        }
        CfgRvalue::InlineAsm { inputs, .. } => {
            for (_, op) in inputs { collect_op_uses(op, set); }
        }
    }
}

fn step_liveness_back_stmt(stmt: &CfgStmt, live: &mut BTreeSet<LocalId>) {
    match &stmt.kind {
        CfgStmtKind::Assign { dst, rvalue } => {
            if dst.projs.is_empty() {
                live.remove(&dst.local);
            } else {
                collect_place_uses(dst, live);
            }
            collect_rvalue_uses(rvalue, live);
        }
        CfgStmtKind::Call { dst, callee, args, recv_place, .. } => {
            if let Some(d) = dst {
                if d.projs.is_empty() {
                    live.remove(&d.local);
                } else {
                    collect_place_uses(d, live);
                }
            }
            if let CfgCallee::FnPtr(op) = callee {
                collect_op_uses(op, live);
            }
            if let Some(rp) = recv_place {
                collect_place_uses(rp, live);
            }
            for a in args {
                collect_op_uses(a, live);
            }
        }
        CfgStmtKind::StorageLive(l) => {
            live.remove(l);
        }
        CfgStmtKind::StorageDead(_) | CfgStmtKind::DiscardLinearError { .. } | CfgStmtKind::ConsumeDestructured(_) => {}
        CfgStmtKind::PoolResetEpoch { pool_local } => {
            live.insert(*pool_local);
        }
    }
}

fn step_loans_stmt(
    f: &CfgFunction,
    cfg: &CfgProgram,
    stmt: &CfgStmt,
    active: &mut BTreeSet<Loan>,
    env: &TypeEnv,
    mut diags: Option<&mut DiagBag>,
    interner: &Interner,
) {
    match &stmt.kind {
        CfgStmtKind::Assign { dst, rvalue } => {
            // 1. Check place accesses in RHS / DST against active loans (including direct reassignment `x = ...` while `x` is borrowed!).
            if (!dst.projs.is_empty() || !f.local(dst.local).is_temp)
                && let Some(dp) = to_access_path(f, dst)
            {
                check_loan_conflict(f, active, &dp, true, dst.local, stmt.span, &mut diags, interner);
            }
            match rvalue {
                CfgRvalue::Borrow { is_mut, place } => {
                    if let Some(ap) = to_access_path(f, place) {
                        check_loan_conflict(f, active, &ap, *is_mut, place.local, stmt.span, &mut diags, interner);
                        if dst.projs.is_empty() {
                            active.insert(Loan {
                                holder: dst.local,
                                path: ap,
                                is_mut: *is_mut,
                            });
                        }
                    }
                    if dst.projs.is_empty() && place.projs.is_empty() {
                        let propagated: Vec<Loan> = active
                            .iter()
                            .filter(|l| l.holder == place.local)
                            .map(|l| Loan {
                                holder: dst.local,
                                path: l.path.clone(),
                                is_mut: l.is_mut,
                            })
                            .collect();
                        active.extend(propagated);
                    }
                }
                CfgRvalue::MakeClosure { env_ptr: Some(CfgOperand::Copy(p) | CfgOperand::Move(p)), .. } => {
                    if dst.projs.is_empty() {
                        let propagated: Vec<Loan> = active
                            .iter()
                            .filter(|l| l.holder == p.local)
                            .map(|l| Loan {
                                holder: dst.local,
                                path: l.path.clone(),
                                is_mut: l.is_mut,
                            })
                            .collect();
                        active.extend(propagated);
                    }
                }
                CfgRvalue::Use(CfgOperand::Copy(p) | CfgOperand::Move(p))
                | CfgRvalue::UnwrapOptional(CfgOperand::Copy(p) | CfgOperand::Move(p))
                | CfgRvalue::UnwrapErrorOk(CfgOperand::Copy(p) | CfgOperand::Move(p))
                | CfgRvalue::FieldExtract { operand: CfgOperand::Copy(p) | CfgOperand::Move(p), .. }
                | CfgRvalue::Cast { operand: CfgOperand::Copy(p) | CfgOperand::Move(p), .. }
                | CfgRvalue::MakeDyn { ptr: CfgOperand::Copy(p) | CfgOperand::Move(p), .. } => {
                    if let Some(ap) = to_access_path(f, p) {
                        let is_move = matches!(rvalue, CfgRvalue::Use(CfgOperand::Move(_)))
                            && !f.local(p.local).is_temp;
                        check_loan_conflict(f, active, &ap, is_move, p.local, stmt.span, &mut diags, interner);
                    }
                    // Propagate loan ownership if a borrowing value, reference, or `view struct` field is moved/extracted into `dst`.
                    if p.projs.is_empty() || has_borrow(env, dst.ty) {
                        let propagated: Vec<Loan> = active
                            .iter()
                            .filter(|l| l.holder == p.local)
                            .map(|l| Loan {
                                holder: dst.local,
                                path: l.path.clone(),
                                is_mut: l.is_mut,
                            })
                            .collect();
                        active.extend(propagated);
                    }
                }
                CfgRvalue::Aggregate { elems, .. } => {
                    // For `view struct` / tuple / optional aggregates carrying borrows,
                    // transfer active loans from every element operand to the aggregate holder.
                    for e in elems {
                        if let CfgOperand::Copy(p) | CfgOperand::Move(p) = e {
                            let propagated: Vec<Loan> = active
                                .iter()
                                .filter(|l| l.holder == p.local)
                                .map(|l| Loan {
                                    holder: dst.local,
                                    path: l.path.clone(),
                                    is_mut: l.is_mut,
                                })
                                .collect();
                            active.extend(propagated);
                        }
                    }
                }
                CfgRvalue::Slice { base: CfgOperand::Copy(p) | CfgOperand::Move(p), .. } => {
                    if let Some(ap) = to_access_path(f, p) {
                        check_loan_conflict(f, active, &ap, false, p.local, stmt.span, &mut diags, interner);
                        if dst.projs.is_empty() {
                            active.insert(Loan {
                                holder: dst.local,
                                path: ap,
                                is_mut: false,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        CfgStmtKind::Call { dst, callee, args, arg_spans, recv_place, recv_span, is_mut_recv, consumes_recv } => {
            // Check receiver access against active loans.
            if let Some(rp) = recv_place
                && let Some(ap) = to_access_path(f, rp)
            {
                let mut_acc = *is_mut_recv || *consumes_recv;
                check_loan_conflict(f, active, &ap, mut_acc, rp.local, recv_span.unwrap_or(stmt.span), &mut diags, interner);
            }
            // Also check ArenaPool.init's arena arg (&mut Arena)!
            if let CfgCallee::Builtin(CfgBuiltinOp::ArenaPoolInit(_)) = callee
                && let Some(CfgOperand::Copy(aplace) | CfgOperand::Move(aplace)) = args.first()
                && let Some(ap) = to_access_path(f, aplace)
            {
                check_loan_conflict(f, active, &ap, true, aplace.local, stmt.span, &mut diags, interner);
            }

            // Register new loan if call yields a borrowing value or view borrow.
            if let Some(d) = dst
                && d.projs.is_empty()
            {
                match callee {
                    CfgCallee::Builtin(CfgBuiltinOp::ArenaPoolInit(_)) => {
                        if let Some(CfgOperand::Copy(aplace) | CfgOperand::Move(aplace)) = args.first()
                            && let Some(ap) = to_access_path(f, aplace)
                        {
                            active.insert(Loan {
                                holder: d.local,
                                path: ap,
                                is_mut: true,
                            });
                        }
                    }
                    CfgCallee::Builtin(CfgBuiltinOp::ArenaAlloc) => {
                        if let Some(rp) = recv_place
                            && let Some(ap) = to_access_path(f, rp)
                        {
                            active.insert(Loan {
                                holder: d.local,
                                path: ap,
                                is_mut: true,
                            });
                        }
                    }
                    CfgCallee::Builtin(
                        CfgBuiltinOp::SlotPoolGet { is_mut, .. }
                        | CfgBuiltinOp::SliceAt { is_mut, .. }
                        | CfgBuiltinOp::ArrayAt { is_mut, .. }
                        | CfgBuiltinOp::BufAt { is_mut, .. }
                        | CfgBuiltinOp::SliceGet { is_mut, .. }
                        | CfgBuiltinOp::ArrayGet { is_mut, .. }
                        | CfgBuiltinOp::BufGet { is_mut, .. }
                        | CfgBuiltinOp::SliceSub { is_mut, .. }
                        | CfgBuiltinOp::ArraySub { is_mut, .. }
                        | CfgBuiltinOp::BufSub { is_mut, .. }
                        | CfgBuiltinOp::BufAsSlice { is_mut, .. },
                    ) => {
                        if let Some(rp) = recv_place
                            && let Some(ap) = to_access_path(f, rp)
                        {
                            active.insert(Loan {
                                holder: d.local,
                                path: ap,
                                is_mut: *is_mut,
                            });
                        }
                    }
                    CfgCallee::Builtin(CfgBuiltinOp::SlotPoolAt(_) | CfgBuiltinOp::StrAsBytes | CfgBuiltinOp::BytesAsStr) => {
                        if let Some(rp) = recv_place
                            && let Some(ap) = to_access_path(f, rp)
                        {
                            active.insert(Loan {
                                holder: d.local,
                                path: ap,
                                is_mut: false,
                            });
                        }
                    }
                    CfgCallee::DirectFn(def_id) => {
                        if let Some(sig) = env.fns.get(def_id)
                            && sig.is_view
                            && let Ok(idx) = view_source_index(env, sig)
                            && let Some(arg) = args.get(idx)
                        {
                            propagate_arg_loans_to_dst(f, arg, d.local, active);
                        }
                    }
                    CfgCallee::Method { impl_idx, sub_idx, is_view } if *is_view => {
                        if let Some(im) = env.impls.get(*impl_idx)
                            && let Some((_, _, sig)) = im.methods.iter().find(|&&(_, si, _)| si == *sub_idx)
                            && let Ok(idx) = view_source_index(env, sig)
                            && let Some(arg) = args.get(idx)
                        {
                            propagate_arg_loans_to_dst(f, arg, d.local, active);
                        }
                    }
                    CfgCallee::GenericInst { inst_id, is_view } if *is_view => {
                        if let Some(&fi) = cfg.by_key.get(&CfgFnKey::GenericInst(*inst_id))
                            && let Ok(idx) = view_source_index(env, &cfg.functions[fi].sig)
                            && let Some(arg) = args.get(idx)
                        {
                            propagate_arg_loans_to_dst(f, arg, d.local, active);
                        }
                    }
                    _ => {}
                }
            }
            let _ = arg_spans;
        }
        _ => {}
    }
}

fn propagate_arg_loans_to_dst(f: &CfgFunction, arg: &CfgOperand, dst: LocalId, active: &mut BTreeSet<Loan>) {
    if let CfgOperand::Copy(p) | CfgOperand::Move(p) = arg {
        if p.projs.is_empty() {
            let carried: Vec<Loan> = active
                .iter()
                .filter(|l| l.holder == p.local)
                .map(|l| Loan {
                    holder: dst,
                    path: l.path.clone(),
                    is_mut: l.is_mut,
                })
                .collect();
            if !carried.is_empty() {
                active.extend(carried);
                return;
            }
        }
        if let Some(ap) = to_access_path(f, p) {
            active.insert(Loan {
                holder: dst,
                path: ap,
                is_mut: false,
            });
        }
    }
}

fn check_loan_conflict(
    f: &CfgFunction,
    active: &BTreeSet<Loan>,
    access_path: &AccessPath,
    is_mut_access: bool,
    base_local: LocalId,
    span: Span,
    diags: &mut Option<&mut DiagBag>,
    interner: &Interner,
) {
    let Some(db) = diags.as_mut() else {
        return;
    };
    for loan in active {
        if loan.holder == base_local {
            continue;
        }
        if (loan.is_mut || is_mut_access) && paths_conflict(&loan.path, access_path) {
            let holder_name = f
                .local(loan.holder)
                .name
                .map(|s| interner.get(s))
                .unwrap_or("<borrow>");
            db.error(
                ErrorCode::E0118,
                span,
                format!(
                    "ZM-18: cannot access `{}` while borrowed by `{}`",
                    access_path.format(interner),
                    holder_name
                ),
            );
            return;
        }
    }
}

fn to_access_path(f: &CfgFunction, place: &CfgPlace) -> Option<AccessPath> {
    let root = f.local(place.local).name?;
    let mut segs = Vec::with_capacity(place.projs.len());
    for pr in &place.projs {
        match pr {
            CfgProj::Field { name, .. } | CfgProj::PoolCol { col: name, .. } => {
                segs.push(PathSeg::Field(*name));
            }
            CfgProj::Deref => segs.push(PathSeg::Deref),
            CfgProj::TupleField { index, .. } => {
                segs.push(PathSeg::Index(PathIndexKey::Const(*index as i128)));
            }
            CfgProj::Index { const_idx, key_sym, .. } => {
                let k = if let Some(c) = const_idx {
                    PathIndexKey::Const(*c)
                } else if let Some(s) = key_sym {
                    PathIndexKey::Ident(*s)
                } else {
                    PathIndexKey::Dynamic
                };
                segs.push(PathSeg::Index(k));
            }
        }
    }
    Some(AccessPath { root, segs })
}

// ============================================================================
// Pass 4: Pool Brand & Epoch Tracking (E0331)
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EpochVal {
    Bot,
    Initial(LocalId),
    ResetAt(u32, u32),
    Mixed,
}

impl EpochVal {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bot, x) | (x, Self::Bot) => x,
            (a, b) if a == b => a,
            _ => Self::Mixed,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IdOrigin {
    Bot,
    Valid(LocalId, EpochVal),
    Unknown,
}

impl IdOrigin {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bot, x) | (x, Self::Bot) => x,
            (Self::Valid(p1, e1), Self::Valid(p2, e2)) if p1 == p2 && e1 == e2 => {
                Self::Valid(p1, e1)
            }
            _ => Self::Unknown,
        }
    }
}

fn analyze_pool_epochs(
    f: &CfgFunction,
    g: &GraphInfo,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    let has_reset = f.blocks.iter().any(|b| {
        b.stmts
            .iter()
            .any(|s| matches!(s.kind, CfgStmtKind::PoolResetEpoch { .. }))
    });
    if !has_reset {
        return;
    }

    let n = f.locals.len();
    let mut in_pool_ep = vec![vec![EpochVal::Bot; n]; f.blocks.len()];
    let mut in_id_org = vec![vec![IdOrigin::Bot; n]; f.blocks.len()];

    for (i, loc) in f.locals.iter().enumerate() {
        let mut init_ep = EpochVal::Initial(loc.id);
        let deref = match env.get(loc.ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => loc.ty,
        };
        if matches!(env.get(deref), Ty::Builtin(BuiltinKind::ArenaPool, _)) {
            init_ep = EpochVal::Initial(loc.id);
        }
        in_pool_ep[f.entry.0 as usize][i] = init_ep;
        if loc.is_param
            && let Ty::Builtin(BuiltinKind::IdBrand, args) = env.get(loc.ty)
            && let Some(GenericVal::Variant(psym)) = args.first()
            && let Some(ploc) = f.locals.iter().find(|p| p.is_param && p.name == Some(*psym))
        {
            in_id_org[f.entry.0 as usize][i] = IdOrigin::Valid(ploc.id, EpochVal::Initial(ploc.id));
        }
    }

    let step_block = |bb: BlockId, pep: &mut Vec<EpochVal>, iorg: &mut Vec<IdOrigin>| {
        for (si, stmt) in f.block(bb).stmts.iter().enumerate() {
            match &stmt.kind {
                CfgStmtKind::PoolResetEpoch { pool_local } => {
                    pep[pool_local.0 as usize] = EpochVal::ResetAt(bb.0, si as u32);
                }
                CfgStmtKind::Call {
                    dst: Some(dst),
                    callee: CfgCallee::Builtin(CfgBuiltinOp::ArenaPoolPush(_) | CfgBuiltinOp::ArenaPoolIdAt(_)),
                    recv_place: Some(rp),
                    ..
                } if dst.projs.is_empty() => {
                    let pl = rp.local;
                    iorg[dst.local.0 as usize] = IdOrigin::Valid(pl, pep[pl.0 as usize]);
                }
                CfgStmtKind::Assign {
                    dst,
                    rvalue:
                        CfgRvalue::Use(CfgOperand::Copy(p) | CfgOperand::Move(p))
                        | CfgRvalue::UnwrapErrorOk(CfgOperand::Copy(p) | CfgOperand::Move(p))
                        | CfgRvalue::UnwrapOptional(CfgOperand::Copy(p) | CfgOperand::Move(p)),
                } if dst.projs.is_empty() => {
                    if p.projs.is_empty() {
                        iorg[dst.local.0 as usize] = iorg[p.local.0 as usize];
                    } else if let Some(CfgProj::PoolCol { is_slot_pool: false, .. }) = p.projs.first() {
                        iorg[dst.local.0 as usize] = IdOrigin::Valid(p.local, pep[p.local.0 as usize]);
                    }
                }
                _ => {}
            }
        }
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &bb in &g.rpo {
            let bi = bb.0 as usize;
            if bi != f.entry.0 as usize {
                let mut jep = vec![EpochVal::Bot; n];
                let mut jorg = vec![IdOrigin::Bot; n];
                for &p in &g.preds[bi] {
                    if in_pool_ep[p.0 as usize].iter().all(|&e| e == EpochVal::Bot) {
                        continue;
                    }
                    let mut pep = in_pool_ep[p.0 as usize].clone();
                    let mut iorg = in_id_org[p.0 as usize].clone();
                    step_block(p, &mut pep, &mut iorg);
                    for k in 0..n {
                        jep[k] = jep[k].join(pep[k]);
                        jorg[k] = jorg[k].join(iorg[k]);
                    }
                }
                if jep != in_pool_ep[bi] || jorg != in_id_org[bi] {
                    in_pool_ep[bi] = jep;
                    in_id_org[bi] = jorg;
                    changed = true;
                }
            }
        }
    }

    // Verification pass
    for &bb in &g.rpo {
        let bi = bb.0 as usize;
        if in_pool_ep[bi].iter().all(|&e| e == EpochVal::Bot) {
            continue;
        }
        let mut pep = in_pool_ep[bi].clone();
        let mut iorg = in_id_org[bi].clone();
        for (si, stmt) in f.block(bb).stmts.iter().enumerate() {
            check_stmt_id_uses(f, stmt, &pep, &iorg, interner, diags);
            match &stmt.kind {
                CfgStmtKind::PoolResetEpoch { pool_local } => {
                    pep[pool_local.0 as usize] = EpochVal::ResetAt(bb.0, si as u32);
                }
                CfgStmtKind::Call {
                    dst: Some(dst),
                    callee: CfgCallee::Builtin(CfgBuiltinOp::ArenaPoolPush(_) | CfgBuiltinOp::ArenaPoolIdAt(_)),
                    recv_place: Some(rp),
                    ..
                } if dst.projs.is_empty() => {
                    let pl = rp.local;
                    iorg[dst.local.0 as usize] = IdOrigin::Valid(pl, pep[pl.0 as usize]);
                }
                CfgStmtKind::Assign {
                    dst,
                    rvalue:
                        CfgRvalue::Use(CfgOperand::Copy(p) | CfgOperand::Move(p))
                        | CfgRvalue::UnwrapErrorOk(CfgOperand::Copy(p) | CfgOperand::Move(p))
                        | CfgRvalue::UnwrapOptional(CfgOperand::Copy(p) | CfgOperand::Move(p)),
                } if dst.projs.is_empty() => {
                    if p.projs.is_empty() {
                        iorg[dst.local.0 as usize] = iorg[p.local.0 as usize];
                    } else if let Some(CfgProj::PoolCol { is_slot_pool: false, .. }) = p.projs.first() {
                        iorg[dst.local.0 as usize] = IdOrigin::Valid(p.local, pep[p.local.0 as usize]);
                    }
                }
                _ => {}
            }
        }
    }
}

fn check_place_epochs(
    f: &CfgFunction,
    p: &CfgPlace,
    sp: Span,
    pep: &[EpochVal],
    iorg: &[IdOrigin],
    interner: &Interner,
    diags: &mut DiagBag,
) {
    for pr in &p.projs {
        if let CfgProj::Index { index: CfgOperand::Copy(ip) | CfgOperand::Move(ip), .. } = pr
            && ip.projs.is_empty()
            && let IdOrigin::Valid(pl, birth_ep) = iorg[ip.local.0 as usize]
            && birth_ep != pep[pl.0 as usize]
        {
            let pname = f.local(pl).name.map(|s| interner.get(s)).unwrap_or("pool");
            diags.error(
                ErrorCode::E0331,
                sp,
                format!("use of invalidated `Id[{pname}]` after `{pname}.reset()`"),
            );
        }
    }
}

fn check_stmt_id_uses(
    f: &CfgFunction,
    stmt: &CfgStmt,
    pep: &[EpochVal],
    iorg: &[IdOrigin],
    interner: &Interner,
    diags: &mut DiagBag,
) {
    match &stmt.kind {
        CfgStmtKind::Assign { dst, rvalue } => {
            check_place_epochs(f, dst, dst.span, pep, iorg, interner, diags);
            if let CfgRvalue::Use(CfgOperand::Copy(p) | CfgOperand::Move(p)) = rvalue {
                check_place_epochs(f, p, p.span, pep, iorg, interner, diags);
            }
        }
        CfgStmtKind::Call { args, recv_place, .. } => {
            if let Some(rp) = recv_place {
                check_place_epochs(f, rp, stmt.span, pep, iorg, interner, diags);
            }
            for a in args {
                if let CfgOperand::Copy(p) | CfgOperand::Move(p) = a {
                    check_place_epochs(f, p, p.span, pep, iorg, interner, diags);
                    if p.projs.is_empty()
                        && let IdOrigin::Valid(pl, birth_ep) = iorg[p.local.0 as usize]
                        && birth_ep != pep[pl.0 as usize]
                    {
                        let pname = f.local(pl).name.map(|s| interner.get(s)).unwrap_or("pool");
                        diags.error(
                            ErrorCode::E0331,
                            p.span,
                            format!("use of invalidated `Id[{pname}]` after `{pname}.reset()`"),
                        );
                    }
                }
            }
        }
        _ => {}
    }
}
