#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{
    self, AccessMode, Block, CompilationUnit, DeclKind, ElseBranch, Expr, ExprKind, IfCond,
    MatchArmBody, Pat, PatKind, Stmt, StmtKind, UnaryOp,
};
use crate::cfg::{
    CfgBuiltinOp, CfgCallee, CfgFnKey, CfgFunction, CfgProgram, CfgProj, CfgRvalue, CfgStmtKind,
};
use crate::diag::{DiagBag, Diagnostic, ErrorCode};
use crate::resolve::{BuiltinKind, DefId, ModuleId, ResolvedProgram, TopDefKind, TopScopeTarget};
use crate::span::{Interner, Span, Symbol};
use crate::types::{EffectSet, Ty, TypeEnv};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FnEffectsInfo {
    pub can_mutate: bool,
    pub can_alloc: bool,
    pub can_block: bool,
    pub uses_extern_c: bool,
    pub is_recursive: bool,
    pub calls_recursive: bool,
    pub own_stack_bytes: u64,
    pub stack_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct EffectsReport {
    pub functions: BTreeMap<CfgFnKey, FnEffectsInfo>,
}

#[derive(Clone, Debug, Default)]
struct IntrinsicEffects {
    mutates_mem: Option<Span>,
    allocates: Option<Span>,
    blocks: Option<Span>,
    uses_extern_c: Option<Span>,
}

/// Run Phase 3B static effect checks (`@pure`, `@noalloc`, `@noblock`, `@interrupt`, `@gpu`),
/// recursion checking (`@recursive` -> `E0113`), static stack analysis (`@stack(N)` -> `E0701`/`E0510`),
/// `sys` conflict graph checking (`E0601`), and `par for` capture checks (`E0602`/`E0603`).
pub fn check_effects_and_concurrency(
    units: &[CompilationUnit],
    prog: &ResolvedProgram,
    env: &TypeEnv,
    cfg: &CfgProgram,
    interner: &Interner,
    diags: &mut DiagBag,
) -> EffectsReport {
    // 1. Build call graph & intrinsic effects for every CFG function.
    let mut calls: BTreeMap<CfgFnKey, Vec<(CfgFnKey, Span)>> = BTreeMap::new();
    let mut dyn_stack_calls: BTreeMap<CfgFnKey, Vec<(Option<u64>, Span, String)>> = BTreeMap::new();
    let mut intrinsics: BTreeMap<CfgFnKey, IntrinsicEffects> = BTreeMap::new();
    let mut own_stack: BTreeMap<CfgFnKey, u64> = BTreeMap::new();
    let mut fn_by_key: BTreeMap<CfgFnKey, &CfgFunction> = BTreeMap::new();

    // Map impl methods to their interface method signatures (for covariance & E0510).
    let mut impl_iface_method: BTreeMap<CfgFnKey, (DefId, crate::types::FnSig)> = BTreeMap::new();
    for (impl_idx, im) in env.impls.iter().enumerate() {
        if let Some(iface_def) = im.iface
            && let Some(imeta) = env.ifaces.get(&iface_def)
        {
            for &(_, sub_idx, ref msig) in &im.methods {
                if let Some(isig) = imeta.methods.iter().find(|m| m.name == msig.name) {
                    impl_iface_method.insert(
                        CfgFnKey::ImplMethod { impl_idx, sub_idx },
                        (iface_def, isig.clone()),
                    );
                }
            }
        }
    }

    for f in &cfg.functions {
        if matches!(f.key, CfgFnKey::GenericInst(_)) {
            continue;
        }
        fn_by_key.insert(f.key, f);
        let mut fcalls = Vec::new();
        let mut fdyn_stacks = Vec::new();
        let mut eff = IntrinsicEffects::default();

        for b in &f.blocks {
            for stmt in &b.stmts {
                match &stmt.kind {
                    CfgStmtKind::Assign { dst, rvalue } => {
                        let is_closure_env_write = matches!(f.key, CfgFnKey::Closure(_))
                            && f.params.first() == Some(&dst.local);
                        // Writing through a deref/mut borrow or pool column mutates external memory.
                        if !is_closure_env_write && place_mutates_external(dst, f, env) && eff.mutates_mem.is_none() {
                            eff.mutates_mem = Some(stmt.span);
                        }
                        if let CfgRvalue::Borrow { is_mut: true, place } = rvalue
                            && !(matches!(f.key, CfgFnKey::Closure(_)) && f.params.first() == Some(&place.local))
                            && place_mutates_external(place, f, env)
                            && eff.mutates_mem.is_none()
                        {
                            eff.mutates_mem = Some(stmt.span);
                        }
                        if let CfgRvalue::MakeClosure { closure_id, .. } = rvalue {
                            fcalls.push((CfgFnKey::Closure(*closure_id), stmt.span));
                        }
                    }
                    CfgStmtKind::Call { callee, is_mut_recv, recv_place, .. } => {
                        match callee {
                            CfgCallee::DirectFn(d) => {
                                fcalls.push((CfgFnKey::TopDef(*d), stmt.span));
                            }
                            CfgCallee::Method { impl_idx, sub_idx, .. } => {
                                fcalls.push((
                                    CfgFnKey::ImplMethod {
                                        impl_idx: *impl_idx,
                                        sub_idx: *sub_idx,
                                    },
                                    stmt.span,
                                ));
                            }
                            CfgCallee::GenericInst { inst_id, .. } => {
                                if let Some(inst_fn) = cfg.functions.iter().find(|cf| cf.key == CfgFnKey::GenericInst(*inst_id))
                                    && let Some((base_key, _)) = &inst_fn.base_generic
                                {
                                    fcalls.push((*base_key, stmt.span));
                                }
                            }
                            CfgCallee::DynCall { iface_def, method_idx } => {
                                if let Some(im) = env.ifaces.get(iface_def)
                                    && let Some(msig) = im.methods.get(*method_idx)
                                {
                                    let me = &msig.effects;
                                    if !me.is_pure && eff.mutates_mem.is_none() {
                                        eff.mutates_mem = Some(stmt.span);
                                    }
                                    if !me.is_noalloc && eff.allocates.is_none() {
                                        eff.allocates = Some(stmt.span);
                                    }
                                    if !me.is_noblock && eff.blocks.is_none() {
                                        eff.blocks = Some(stmt.span);
                                    }
                                    fdyn_stacks.push((
                                        me.stack_limit,
                                        stmt.span,
                                        format!(
                                            "dyn {}.{}",
                                            interner.get(prog.defs[iface_def.0 as usize].name),
                                            interner.get(msig.name)
                                        ),
                                    ));
                                }
                            }
                            CfgCallee::ExternFn { .. } => {
                                if eff.uses_extern_c.is_none() {
                                    eff.uses_extern_c = Some(stmt.span);
                                }
                                if eff.mutates_mem.is_none() {
                                    eff.mutates_mem = Some(stmt.span);
                                }
                            }
                            CfgCallee::Builtin(bop) => {
                                match bop {
                                    CfgBuiltinOp::ArenaInit
                                    | CfgBuiltinOp::SlotPoolInit(_)
                                    | CfgBuiltinOp::SlotPoolInsert(_)
                                    | CfgBuiltinOp::FsReadFile
                                    | CfgBuiltinOp::FsListDir
                                    | CfgBuiltinOp::DirReadFile
                                    | CfgBuiltinOp::IoRunFibers => {
                                        if eff.allocates.is_none() {
                                            eff.allocates = Some(stmt.span);
                                        }
                                        if eff.mutates_mem.is_none() {
                                            eff.mutates_mem = Some(stmt.span);
                                        }
                                        if matches!(bop, CfgBuiltinOp::IoRunFibers | CfgBuiltinOp::FsReadFile | CfgBuiltinOp::FsListDir | CfgBuiltinOp::DirReadFile)
                                            && eff.blocks.is_none()
                                        {
                                            eff.blocks = Some(stmt.span);
                                        }
                                    }
                                    CfgBuiltinOp::FsOpenDir
                                    | CfgBuiltinOp::FsWriteFile
                                    | CfgBuiltinOp::FsWriteExecutable
                                    | CfgBuiltinOp::FsIsDir
                                    | CfgBuiltinOp::FsMkdirAll
                                    | CfgBuiltinOp::FsSpawnWait
                                    | CfgBuiltinOp::DirSubDir
                                    | CfgBuiltinOp::DirWriteFile
                                    | CfgBuiltinOp::DirClose
                                    | CfgBuiltinOp::IoNopRoundtrip => {
                                        if eff.mutates_mem.is_none() {
                                            eff.mutates_mem = Some(stmt.span);
                                        }
                                        if eff.blocks.is_none() {
                                            eff.blocks = Some(stmt.span);
                                        }
                                    }
                                    CfgBuiltinOp::ScheduleRun => {
                                        if eff.mutates_mem.is_none() {
                                            eff.mutates_mem = Some(stmt.span);
                                        }
                                    }
                                    CfgBuiltinOp::ArenaAlloc
                                    | CfgBuiltinOp::ArenaReset
                                    | CfgBuiltinOp::ArenaDestroy
                                    | CfgBuiltinOp::ArenaPoolInit(_)
                                    | CfgBuiltinOp::ArenaPoolPush(_)
                                    | CfgBuiltinOp::ArenaPoolReset(_)
                                    | CfgBuiltinOp::SlotPoolTryInsert(_)
                                    | CfgBuiltinOp::SlotPoolRemove(_)
                                    | CfgBuiltinOp::SlotPoolDestroy(_) => {
                                        if let Some(rp) = recv_place
                                            && place_mutates_external(rp, f, env)
                                            && eff.mutates_mem.is_none()
                                        {
                                            eff.mutates_mem = Some(stmt.span);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            CfgCallee::FnPtr(_) => {
                                if eff.mutates_mem.is_none() {
                                    eff.mutates_mem = Some(stmt.span);
                                }
                            }
                        }
                        if *is_mut_recv
                            && let Some(rp) = recv_place
                            && place_mutates_external(rp, f, env)
                            && eff.mutates_mem.is_none()
                        {
                            eff.mutates_mem = Some(stmt.span);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check `@interrupt` invariant: requires explicit `@stack(N)` and forbids `@recursive`.
        if f.sig.effects.is_interrupt {
            if f.sig.effects.stack_limit.is_none() {
                diags.error(
                    ErrorCode::E0112,
                    f.sig.effects.effect_span.unwrap_or(f.body_span),
                    "ZM-12: `@interrupt` function must declare an explicit `@stack(N)` budget",
                );
            }
            if f.sig.effects.is_recursive {
                diags.error(
                    ErrorCode::E0113,
                    f.sig.effects.effect_span.unwrap_or(f.body_span),
                    "ZM-13: `@interrupt` function cannot be `@recursive`",
                );
            }
        }
        if f.sig.effects.is_gpu && f.sig.effects.is_recursive {
            diags.error(
                ErrorCode::E0113,
                f.sig.effects.effect_span.unwrap_or(f.body_span),
                "ZM-13: `@gpu` function cannot be `@recursive`",
            );
        }
        if f.sig.effects.stack_limit.is_some() && f.sig.effects.is_recursive {
            diags.error(
                ErrorCode::E0113,
                f.sig.effects.stack_span.unwrap_or(f.body_span),
                "ZM-13: `@recursive` function cannot declare a static `@stack(N)` bound",
            );
        }

        calls.insert(f.key, fcalls);
        if !fdyn_stacks.is_empty() {
            dyn_stack_calls.insert(f.key, fdyn_stacks);
        }
        intrinsics.insert(f.key, eff);
        own_stack.insert(f.key, compute_own_frame_bytes(f, env));
    }

    // 2. Detect cycles (direct and mutual recursion) via DFS on the call graph.
    let mut in_cycle: BTreeSet<CfgFnKey> = BTreeSet::new();
    let mut cycle_call_span: BTreeMap<CfgFnKey, Span> = BTreeMap::new();
    {
        let mut color: BTreeMap<CfgFnKey, u8> = BTreeMap::new(); // 0=white, 1=gray, 2=black
        let mut stack: Vec<(CfgFnKey, Span)> = Vec::new();
        for &k in fn_by_key.keys() {
            if color.get(&k).copied().unwrap_or(0) == 0 {
                dfs_cycles(k, &calls, &mut color, &mut stack, &mut in_cycle, &mut cycle_call_span);
            }
        }
    }

    for &k in &in_cycle {
        if let Some(f) = fn_by_key.get(&k)
            && !f.sig.effects.is_recursive
        {
            let sp = cycle_call_span.get(&k).copied().unwrap_or(f.body_span);
            diags.error(
                ErrorCode::E0113,
                sp,
                format!(
                    "ZM-13: recursive call in `{}` requires explicit `@recursive` annotation",
                    interner.get(f.name)
                ),
            );
        }
    }

    // 3. Infer effective effects bottom-up for private functions and check contracts on all functions.
    // A function's effective flags: (can_mutate, can_alloc, can_block, can_recurse, uses_extern_c).
    let mut eff_flags: BTreeMap<CfgFnKey, (bool, bool, bool, bool, bool)> = BTreeMap::new();
    for (&k, f) in &fn_by_key {
        let ie = &intrinsics[&k];
        let is_rec = f.sig.effects.is_recursive || in_cycle.contains(&k);
        if f.sig.is_pub && !f.sig.is_comptime {
            // Public functions expose their explicit contract to callers (unannotated `pub fn` is conservatively impure/alloc/blocking).
            let can_mut = !f.sig.effects.is_pure;
            let can_alloc = !f.sig.effects.is_noalloc && !f.sig.effects.is_pure;
            let can_block = !f.sig.effects.is_noblock && !f.sig.effects.is_pure;
            eff_flags.insert(k, (can_mut, can_alloc, can_block, is_rec, ie.uses_extern_c.is_some()));
        } else {
            eff_flags.insert(
                k,
                (
                    ie.mutates_mem.is_some(),
                    ie.allocates.is_some(),
                    ie.blocks.is_some(),
                    is_rec,
                    ie.uses_extern_c.is_some(),
                ),
            );
        }
    }

    // Fixed-point propagate actual effects across private callees.
    for _ in 0..32 {
        let mut changed = false;
        let keys: Vec<CfgFnKey> = fn_by_key.keys().copied().collect();
        for k in keys {
            let f = fn_by_key[&k];
            if f.sig.is_pub && !f.sig.is_comptime {
                continue;
            }
            let mut cur = eff_flags[&k];
            if let Some(cs) = calls.get(&k) {
                for &(callee, _) in cs {
                    if let Some(&cf) = eff_flags.get(&callee) {
                        cur.0 |= cf.0;
                        cur.1 |= cf.1;
                        cur.2 |= cf.2;
                        cur.3 |= cf.3;
                        cur.4 |= cf.4;
                    }
                }
            }
            if cur != eff_flags[&k] {
                eff_flags.insert(k, cur);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Enforce declared effects on each function body & its callees.
    for (&k, f) in &fn_by_key {
        let req = &f.sig.effects;
        let ie = &intrinsics[&k];
        let fn_name = interner.get(f.name);

        if req.is_pure {
            if let Some(sp) = ie.mutates_mem {
                diags.error(
                    ErrorCode::E0112,
                    sp,
                    format!("ZM-12: `@pure` function `{fn_name}` cannot mutate external memory"),
                );
            }
            if let Some(sp) = ie.uses_extern_c {
                diags.error(
                    ErrorCode::E0112,
                    sp,
                    format!("ZM-12: `@pure` function `{fn_name}` cannot call `extern \"C\"`"),
                );
            }
        }
        if req.is_noalloc
            && let Some(sp) = ie.allocates
        {
            diags.error(
                ErrorCode::E0112,
                sp,
                format!("ZM-12: `@noalloc` function `{fn_name}` cannot perform heap allocation"),
            );
        }
        if req.is_gpu
            && let Some(sp) = ie.uses_extern_c
        {
            diags.error(
                ErrorCode::E0112,
                sp,
                format!("ZM-12: `@gpu` function `{fn_name}` cannot call CPU `extern \"C\"`"),
            );
        }

        if let Some(cs) = calls.get(&k) {
            for &(callee, sp) in cs {
                let Some(cf) = eff_flags.get(&callee).copied() else {
                    continue;
                };
                let callee_name = fn_by_key
                    .get(&callee)
                    .map(|cfn| interner.get(cfn.name))
                    .unwrap_or("<callee>");
                if req.is_pure && (cf.0 || cf.1 || cf.2) {
                    diags.error(
                        ErrorCode::E0112,
                        sp,
                        format!("ZM-12: `@pure` function `{fn_name}` cannot call impure `{callee_name}`"),
                    );
                } else if req.is_noalloc && cf.1 {
                    diags.error(
                        ErrorCode::E0112,
                        sp,
                        format!("ZM-12: `@noalloc` function `{fn_name}` cannot call allocating `{callee_name}`"),
                    );
                } else if req.is_noblock && cf.2 {
                    diags.error(
                        ErrorCode::E0112,
                        sp,
                        format!("ZM-12: `@noblock` function `{fn_name}` cannot call potentially blocking `{callee_name}`"),
                    );
                }
                if (req.stack_limit.is_some() || req.is_interrupt || req.is_gpu) && cf.3 {
                    diags.error(
                        ErrorCode::E0113,
                        sp,
                        format!(
                            "ZM-13: `{fn_name}` cannot call `@recursive` function `{callee_name}` under `@stack`/`@interrupt`/`@gpu`"
                        ),
                    );
                }
            }
        }
    }

    // 4. Static stack calculation (`@stack(N)` -> `E0701` / `E0510`), including `dyn` interface ceilings!
    let mut total_stack: BTreeMap<CfgFnKey, u64> = BTreeMap::new();
    for &k in fn_by_key.keys() {
        let mut visiting = BTreeSet::new();
        let sz = compute_total_stack(k, &own_stack, &calls, &dyn_stack_calls, &in_cycle, &fn_by_key, &mut total_stack, &mut visiting);
        let f = fn_by_key[&k];
        if let Some(limit) = f.sig.effects.stack_limit {
            if let Some(dcs) = dyn_stack_calls.get(&k) {
                for (lim_opt, sp, label) in dcs {
                    if lim_opt.is_none() {
                        diags.error(
                            ErrorCode::E0701,
                            *sp,
                            format!(
                                "cannot verify `@stack({limit})` across `{label}` call without `@stack(N)` on interface method"
                            ),
                        );
                    }
                }
            }
            if sz > limit {
            let is_iface_impl = impl_iface_method.contains_key(&k);
            let code = if is_iface_impl { ErrorCode::E0510 } else { ErrorCode::E0701 };
            let own = own_stack[&k];
            let callee_max = sz.saturating_sub(own);
            let mut d = Diagnostic::error(
                code,
                f.sig.effects.stack_span.unwrap_or(f.body_span),
                format!(
                    "static stack usage of `{}` is {} bytes, exceeding `@stack({})` budget",
                    interner.get(f.name),
                    sz,
                    limit
                ),
            );
            d.notes.push(format!(
                "stack breakdown: own frame = {} bytes, max callee chain = {} bytes",
                own, callee_max
            ));
            diags.emit(d);
            }
        }
        // Also check `impl Iface` covariance against `iface`'s `@stack(N)` and effects!
        if let Some((_, isig)) = impl_iface_method.get(&k) {
            check_iface_impl_covariance(f, isig, sz, &own_stack, interner, diags);
        }
    }

    // 5. Check `sys` conflict graph (`Schedule.build`) and `par for` capture/reduction rules across AST.
    check_ast_concurrency(units, prog, env, interner, diags);

    let mut report = EffectsReport::default();
    for (&k, f) in &fn_by_key {
        let ef = eff_flags
            .get(&k)
            .copied()
            .unwrap_or((false, false, false, false, false));
        let own = own_stack.get(&k).copied().unwrap_or(0);
        let total = total_stack.get(&k).copied().unwrap_or(own);
        report.functions.insert(
            k,
            FnEffectsInfo {
                can_mutate: ef.0,
                can_alloc: ef.1,
                can_block: ef.2,
                uses_extern_c: ef.4,
                is_recursive: f.sig.effects.is_recursive || in_cycle.contains(&k),
                calls_recursive: ef.3,
                own_stack_bytes: own,
                stack_bytes: total,
            },
        );
    }
    report
}

fn check_iface_impl_covariance(
    f: &CfgFunction,
    isig: &crate::types::FnSig,
    total_stack: u64,
    own_stack: &BTreeMap<CfgFnKey, u64>,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    let ie = &isig.effects;
    let fe = &f.sig.effects;
    if let Some(ilim) = ie.stack_limit {
        let violated = total_stack > ilim || fe.stack_limit.is_some_and(|fl| fl > ilim);
        if violated && fe.stack_limit.is_none_or(|fl| total_stack <= fl) {
            let own = own_stack.get(&f.key).copied().unwrap_or(0);
            let mut d = Diagnostic::error(
                ErrorCode::E0510,
                fe.stack_span.unwrap_or(f.body_span),
                format!(
                    "impl method `{}` requires {} bytes of stack, exceeding interface `@stack({})`",
                    interner.get(f.name),
                    total_stack,
                    ilim
                ),
            );
            d.notes.push(format!("own frame = {} bytes", own));
            diags.emit(d);
        }
    }
    if (ie.is_pure && !fe.is_pure)
        || (ie.is_noalloc && !fe.is_noalloc)
        || (ie.is_noblock && !fe.is_noblock)
    {
        diags.error(
            ErrorCode::E0112,
            f.body_span,
            format!(
                "ZM-12: impl method `{}` violates effect contract declared in interface",
                interner.get(f.name)
            ),
        );
    }
}

fn place_mutates_external(place: &crate::cfg::CfgPlace, f: &CfgFunction, env: &TypeEnv) -> bool {
    let root = f.local(place.local);
    if root.is_param {
        if matches!(env.get(root.ty), Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. }) {
            return true;
        }
        if matches!(root.ty, t if matches!(env.get(t), Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::SlotPool, _)))
            && f.sig.params.get(root.param_index.unwrap_or(usize::MAX)).is_some_and(|p| p.access == Some(AccessMode::Write))
        {
            return true;
        }
    }
    for pr in &place.projs {
        if matches!(pr, CfgProj::Deref) {
            return true;
        }
    }
    false
}

fn compute_own_frame_bytes(f: &CfgFunction, env: &TypeEnv) -> u64 {
    let mut bytes: u64 = 16; // ABI return address + saved frame pointer / alignment
    for loc in &f.locals {
        if loc.is_param {
            continue;
        }
        // Only count user locals and non-trivial aggregate temporaries (small scalar temps live in registers).
        let lay = env.layout(loc.ty);
        if !loc.is_temp || lay.size > 16 {
            let align = lay.align.max(1);
            bytes = (bytes + align - 1) & !(align - 1);
            bytes = bytes.saturating_add(lay.size);
        }
    }
    // Round frame size up to 16-byte ABI stack alignment.
    (bytes + 15) & !15
}

type DynStackCallMap = BTreeMap<CfgFnKey, Vec<(Option<u64>, Span, String)>>;

fn compute_total_stack(
    k: CfgFnKey,
    own_stack: &BTreeMap<CfgFnKey, u64>,
    calls: &BTreeMap<CfgFnKey, Vec<(CfgFnKey, Span)>>,
    dyn_stack_calls: &DynStackCallMap,
    in_cycle: &BTreeSet<CfgFnKey>,
    fn_by_key: &BTreeMap<CfgFnKey, &CfgFunction>,
    memo: &mut BTreeMap<CfgFnKey, u64>,
    visiting: &mut BTreeSet<CfgFnKey>,
) -> u64 {
    if let Some(&v) = memo.get(&k) {
        return v;
    }
    let own = own_stack.get(&k).copied().unwrap_or(16);
    if in_cycle.contains(&k) || !visiting.insert(k) {
        return own;
    }
    let mut max_callee = 0u64;
    if let Some(dcs) = dyn_stack_calls.get(&k) {
        for (lim_opt, _, _) in dcs {
            if let Some(lim) = lim_opt
                && *lim > max_callee
            {
                max_callee = *lim;
            }
        }
    }
    if let Some(cs) = calls.get(&k) {
        for &(c, _) in cs {
            let csz = if let Some(cf) = fn_by_key.get(&c) {
                if let Some(lim) = cf.sig.effects.stack_limit {
                    // Use computed stack (or declared ceiling for external/abstract calls).
                    let calc = compute_total_stack(c, own_stack, calls, dyn_stack_calls, in_cycle, fn_by_key, memo, visiting);
                    calc.min(lim).max(calc)
                } else {
                    compute_total_stack(c, own_stack, calls, dyn_stack_calls, in_cycle, fn_by_key, memo, visiting)
                }
            } else {
                0
            };
            if csz > max_callee {
                max_callee = csz;
            }
        }
    }
    visiting.remove(&k);
    let tot = own.saturating_add(max_callee);
    memo.insert(k, tot);
    tot
}

fn dfs_cycles(
    u: CfgFnKey,
    calls: &BTreeMap<CfgFnKey, Vec<(CfgFnKey, Span)>>,
    color: &mut BTreeMap<CfgFnKey, u8>,
    stack: &mut Vec<(CfgFnKey, Span)>,
    in_cycle: &mut BTreeSet<CfgFnKey>,
    cycle_call_span: &mut BTreeMap<CfgFnKey, Span>,
) {
    color.insert(u, 1);
    if let Some(cs) = calls.get(&u) {
        for &(v, sp) in cs {
            let vc = color.get(&v).copied().unwrap_or(0);
            if vc == 1 {
                in_cycle.insert(u);
                cycle_call_span.entry(u).or_insert(sp);
                in_cycle.insert(v);
                cycle_call_span.entry(v).or_insert(sp);
                if u != v {
                    for &(node, nsp) in stack.iter().rev() {
                        in_cycle.insert(node);
                        cycle_call_span.entry(node).or_insert(nsp);
                        if node == v {
                            break;
                        }
                    }
                }
            } else if vc == 0 && calls.contains_key(&v) {
                stack.push((u, sp));
                dfs_cycles(v, calls, color, stack, in_cycle, cycle_call_span);
                stack.pop();
            }
        }
    }
    color.insert(u, 2);
}

fn check_ast_concurrency(
    units: &[CompilationUnit],
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    for i in 0..prog.defs.len() {
        let def_mod = prog.defs[i].module;
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else {
            continue;
        };
        match prog.defs[i].kind {
            TopDefKind::Fn(decl_idx) => {
                if let DeclKind::Fn(f) = &units[ui].decls[decl_idx].kind {
                    let mut checker = ConcurrencyChecker::new(def_mod, prog, env, interner, diags);
                    for p in &f.params {
                        checker.bind_outer(p.name.sym);
                    }
                    checker.visit_block(&f.body);
                }
            }
            TopDefKind::Sys(decl_idx) => {
                if let DeclKind::Sys(s) = &units[ui].decls[decl_idx].kind {
                    let mut checker = ConcurrencyChecker::new(def_mod, prog, env, interner, diags);
                    for p in &s.params {
                        checker.bind_outer(p.name.sym);
                    }
                    checker.visit_block(&s.body);
                }
            }
            _ => {}
        }
    }
    for im in &env.impls {
        let Some(ui) = prog.modules[im.module.0 as usize].unit_idx else {
            continue;
        };
        let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
            continue;
        };
        for &(_, sub_idx, _) in &im.methods {
            if let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind {
                let mut checker = ConcurrencyChecker::new(im.module, prog, env, interner, diags);
                for p in &f.params {
                    checker.bind_outer(p.name.sym);
                }
                checker.visit_block(&f.body);
            }
        }
    }
}

struct ConcurrencyChecker<'a> {
    mod_id: ModuleId,
    prog: &'a ResolvedProgram,
    env: &'a TypeEnv,
    interner: &'a Interner,
    diags: &'a mut DiagBag,
    scopes: Vec<BTreeSet<Symbol>>,
    par_depth: usize,
    par_outer_vars: BTreeSet<Symbol>,
    par_locked_collection: Option<Symbol>,
    par_loop_var: Option<Symbol>,
    par_iter_is_each: bool,
}

impl<'a> ConcurrencyChecker<'a> {
    fn new(
        mod_id: ModuleId,
        prog: &'a ResolvedProgram,
        env: &'a TypeEnv,
        interner: &'a Interner,
        diags: &'a mut DiagBag,
    ) -> Self {
        Self {
            mod_id,
            prog,
            env,
            interner,
            diags,
            scopes: vec![BTreeSet::new()],
            par_depth: 0,
            par_outer_vars: BTreeSet::new(),
            par_locked_collection: None,
            par_loop_var: None,
            par_iter_is_each: false,
        }
    }

    fn is_disjoint_pool_col_access(&self, expr: &Expr) -> bool {
        if self.par_depth == 0 || !self.par_iter_is_each {
            return false;
        }
        let (Some(locked), Some(lv)) = (self.par_locked_collection, self.par_loop_var) else {
            return false;
        };
        if let ExprKind::Index { base, index } = &expr.kind
            && let ExprKind::Ident(idx_sym) = &index.kind
            && *idx_sym == lv
            && let ExprKind::Field { base: col_base, .. } = &base.kind
            && let Some(root_sym) = root_ident_of_lvalue(col_base)
            && root_sym == locked
        {
            return true;
        }
        false
    }

    fn bind_outer(&mut self, sym: Symbol) {
        if let Some(sc) = self.scopes.last_mut() {
            sc.insert(sym);
        }
    }

    fn bind_pat(&mut self, pat: &Pat) {
        match &pat.kind {
            PatKind::Ident(s) => self.bind_outer(*s),
            PatKind::Tuple(ps) | PatKind::Variant { args: ps, .. } => {
                for p in ps {
                    self.bind_pat(p);
                }
            }
            PatKind::Struct { fields, .. } => {
                for f in fields {
                    if let Some(p) = &f.pat {
                        self.bind_pat(p);
                    } else {
                        self.bind_outer(f.name.sym);
                    }
                }
            }
            _ => {}
        }
    }

    fn all_visible_vars(&self) -> BTreeSet<Symbol> {
        let mut s = BTreeSet::new();
        for sc in &self.scopes {
            for &v in sc {
                s.insert(v);
            }
        }
        s
    }

    fn visit_block(&mut self, b: &Block) {
        self.scopes.push(BTreeSet::new());
        for s in &b.stmts {
            self.visit_stmt(s);
        }
        if let Some(t) = &b.tail_expr {
            self.visit_expr(t);
        }
        self.scopes.pop();
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pat, init, .. } => {
                self.visit_expr(init);
                self.bind_pat(pat);
            }
            StmtKind::Var { name, init, .. } => {
                self.visit_expr(init);
                self.bind_outer(name.sym);
            }
            StmtKind::Assign { lhs, rhs, .. } => {
                if self.par_depth > 0
                    && !self.is_disjoint_pool_col_access(lhs)
                    && let Some(root_sym) = root_ident_of_lvalue(lhs)
                    && self.par_outer_vars.contains(&root_sym)
                {
                    self.diags.error(
                        ErrorCode::E0602,
                        stmt.span,
                        format!(
                            "cannot mutate outer variable `{}` inside `par for`",
                            self.interner.get(root_sym)
                        ),
                    );
                }
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            StmtKind::Expr(e) | StmtKind::Discard(e) => self.visit_expr(e),
            StmtKind::If { data, .. } => self.visit_if(data),
            StmtKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.scopes.push(BTreeSet::new());
                    self.bind_pat(&arm.pat);
                    if let Some(g) = &arm.guard {
                        self.visit_expr(g);
                    }
                    match &arm.body {
                        MatchArmBody::Block(b) => self.visit_block(b),
                        MatchArmBody::Expr(e) => self.visit_expr(e),
                        MatchArmBody::Jump(_) => {}
                    }
                    self.scopes.pop();
                }
            }
            StmtKind::While { cond, body, .. } => {
                self.visit_expr(cond);
                self.visit_block(body);
            }
            StmtKind::For { pat, iter_start, iter_end, body, .. } => {
                self.visit_expr(iter_start);
                if let Some((_, end)) = iter_end {
                    self.visit_expr(end);
                }
                self.scopes.push(BTreeSet::new());
                self.bind_pat(pat);
                self.visit_block(body);
                self.scopes.pop();
            }
            StmtKind::ParFor { pat, iter_start, iter_end, body } => {
                self.visit_expr(iter_start);
                if let Some((_, end)) = iter_end {
                    self.visit_expr(end);
                }
                let saved_outer = self.par_outer_vars.clone();
                let saved_locked = self.par_locked_collection;
                let saved_loop_var = self.par_loop_var;
                let saved_is_each = self.par_iter_is_each;
                let outer_now = self.all_visible_vars();
                self.par_outer_vars.extend(outer_now);
                self.par_locked_collection = iter_root_collection(iter_start);
                self.par_loop_var = match &pat.kind {
                    PatKind::Ident(s) => Some(*s),
                    _ => None,
                };
                self.par_iter_is_each = iter_is_pool_each(iter_start, self.interner);
                self.par_depth += 1;

                self.scopes.push(BTreeSet::new());
                self.bind_pat(pat);
                self.visit_block(body);
                self.scopes.pop();

                self.par_depth -= 1;
                self.par_outer_vars = saved_outer;
                self.par_locked_collection = saved_locked;
                self.par_loop_var = saved_loop_var;
                self.par_iter_is_each = saved_is_each;
            }
            StmtKind::Loop { body, .. }
            | StmtKind::ComptimeBlock(body)
            | StmtKind::RawBlock(body) => self.visit_block(body),
            StmtKind::Defer(d) | StmtKind::Errdefer { body: d, .. } => match d {
                ast::DeferBody::Block(b) => self.visit_block(b),
                ast::DeferBody::Expr(e) => self.visit_expr(e),
            },
            StmtKind::Jump(j) => match j {
                ast::JumpAction::Return(Some(e)) | ast::JumpAction::Break { value: Some(e), .. } => {
                    self.visit_expr(e);
                }
                _ => {}
            },
            StmtKind::Const(_) => {}
        }
    }

    fn visit_if(&mut self, data: &ast::IfExpr) {
        match &data.cond {
            IfCond::Expr(c) => self.visit_expr(c),
            IfCond::Let { pat, expr } => {
                self.visit_expr(expr);
                self.scopes.push(BTreeSet::new());
                self.bind_pat(pat);
                self.visit_block(&data.then_block);
                self.scopes.pop();
                if let Some(eb) = &data.else_branch {
                    match eb {
                        ElseBranch::Block(b) => self.visit_block(b),
                        ElseBranch::If { data: nested, .. } => self.visit_if(nested),
                    }
                }
                return;
            }
        }
        self.visit_block(&data.then_block);
        if let Some(eb) = &data.else_branch {
            match eb {
                ElseBranch::Block(b) => self.visit_block(b),
                ElseBranch::If { data: nested, .. } => self.visit_if(nested),
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if self.par_depth > 0 && self.par_locked_collection == Some(*sym) {
                    self.diags.error(
                        ErrorCode::E0602,
                        expr.span,
                        format!(
                            "iterated collection `{}` is locked inside `par for`; access elements only through the loop variable",
                            self.interner.get(*sym)
                        ),
                    );
                }
            }
            ExprKind::Unary { op: UnaryOp::BorrowMut, expr: inner } => {
                if self.par_depth > 0
                    && let Some(root_sym) = root_ident_of_lvalue(inner)
                    && self.par_outer_vars.contains(&root_sym)
                {
                    self.diags.error(
                        ErrorCode::E0602,
                        expr.span,
                        format!(
                            "cannot take `&mut` borrow of outer variable `{}` inside `par for`",
                            self.interner.get(root_sym)
                        ),
                    );
                }
                self.visit_expr(inner);
            }
            ExprKind::Unary { expr: inner, .. }
            | ExprKind::Paren(inner)
            | ExprKind::Deref(inner)
            | ExprKind::ComptimeExpr(inner) => self.visit_expr(inner),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::Field { base, .. } => self.visit_expr(base),
            ExprKind::Index { base, index } => {
                if self.is_disjoint_pool_col_access(expr) {
                    self.visit_expr(index);
                } else {
                    self.visit_expr(base);
                    self.visit_expr(index);
                }
            }
            ExprKind::Slice { base, lo, hi } => {
                self.visit_expr(base);
                if let Some(l) = lo { self.visit_expr(l); }
                if let Some(h) = hi { self.visit_expr(h); }
            }
            ExprKind::Call { callee, args } => {
                // Check `Schedule.build(.{ sys_a, sys_b, ... })` for `sys` column conflicts (`E0601`)!
                if let ExprKind::Field { base, field } = &callee.kind {
                    if let ExprKind::Ident(bsym) = &base.kind
                        && self.interner.get(*bsym) == "Schedule"
                        && self.interner.get(field.sym) == "build"
                    {
                        self.check_schedule_build(args, expr.span);
                    }
                    if self.interner.get(field.sym) == "par_reduce"
                        && let Some(arg0) = args.first()
                        && let ExprKind::EnumLit(ev) = &arg0.kind
                        && self.interner.get(ev.sym) == "Add"
                    {
                        self.diags.error(
                            ErrorCode::E0603,
                            arg0.span,
                            "non-deterministic `.Add` reduction in `par_reduce`; use `.AddWrap` (integers) or `.AddTree` (floats)",
                        );
                    }
                }
                self.visit_expr(callee);
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::If { data, .. } => self.visit_if(data),
            ExprKind::ComptimeBlock(b) | ExprKind::Loop { body: b, .. } => self.visit_block(b),
            ExprKind::Tuple(es) | ExprKind::ArrayLit(es) => {
                for e in es {
                    self.visit_expr(e);
                }
            }
            ExprKind::ArrayRepeat { val, count } => {
                self.visit_expr(val);
                self.visit_expr(count);
            }
            ExprKind::AnonStructLit(fs) | ExprKind::NamedStructLit { fields: fs, .. } => {
                for f in fs {
                    self.visit_expr(&f.value);
                }
            }
            ExprKind::Orelse { lhs, rhs } | ExprKind::Catch { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                match rhs {
                    ast::CatchBody::Block(b) => self.visit_block(b),
                    ast::CatchBody::Expr(e) => self.visit_expr(e),
                    ast::CatchBody::Jump(_) => {}
                }
            }
            _ => {}
        }
    }

    fn check_schedule_build(&mut self, args: &[Expr], span: Span) {
        let mut sys_defs: Vec<(DefId, Span)> = Vec::new();
        let mut seen_defs: BTreeSet<DefId> = BTreeSet::new();
        let mut edges: BTreeSet<(DefId, DefId)> = BTreeSet::new();

        for arg in args {
            match &arg.kind {
                ExprKind::AnonStructLit(fs) => {
                    for f in fs {
                        self.collect_schedule_entry(&f.value, &mut sys_defs, &mut seen_defs, &mut edges);
                    }
                }
                ExprKind::Tuple(es) | ExprKind::ArrayLit(es) => {
                    for e in es {
                        self.collect_schedule_entry(e, &mut sys_defs, &mut seen_defs, &mut edges);
                    }
                }
                _ => self.collect_schedule_entry(arg, &mut sys_defs, &mut seen_defs, &mut edges),
            }
        }

        // Cycle check + transitive ordering closure over directed `before`/`after` edges.
        let mut reach: BTreeMap<DefId, BTreeSet<DefId>> = BTreeMap::new();
        for &(d, _) in &sys_defs {
            let mut visited: BTreeSet<DefId> = BTreeSet::new();
            let mut stack: Vec<DefId> = edges
                .iter()
                .filter_map(|&(u, v)| if u == d { Some(v) } else { None })
                .collect();
            while let Some(cur) = stack.pop() {
                if !visited.insert(cur) {
                    continue;
                }
                for &(u, v) in &edges {
                    if u == cur && !visited.contains(&v) {
                        stack.push(v);
                    }
                }
            }
            if visited.contains(&d) {
                let sp = sys_defs
                    .iter()
                    .find(|(id, _)| *id == d)
                    .map(|(_, s)| *s)
                    .unwrap_or(span);
                let n = self.interner.get(self.prog.defs[d.0 as usize].name);
                self.diags.error(
                    ErrorCode::E0601,
                    if sp != Span::DUMMY { sp } else { span },
                    format!("cycle detected in `Schedule.build` ordering constraints involving `{n}`"),
                );
                break;
            }
            reach.insert(d, visited);
        }

        // Check every unordered pair `(s1, s2)` for write/read or write/write column conflicts.
        for i in 0..sys_defs.len() {
            for j in (i + 1)..sys_defs.len() {
                let (d1, sp1) = sys_defs[i];
                let (d2, _sp2) = sys_defs[j];
                let ordered = reach.get(&d1).is_some_and(|r| r.contains(&d2))
                    || reach.get(&d2).is_some_and(|r| r.contains(&d1));
                if ordered {
                    continue;
                }
                let Some(a1) = self.env.sys_access.get(&d1) else { continue };
                let Some(a2) = self.env.sys_access.get(&d2) else { continue };
                for x in a1 {
                    for y in a2 {
                        if x.pool_def != y.pool_def {
                            continue;
                        }
                        if x.mode == AccessMode::Read && y.mode == AccessMode::Read {
                            continue;
                        }
                        for &c1 in &x.columns {
                            if y.columns.contains(&c1) {
                                let n1 = self.interner.get(self.prog.defs[d1.0 as usize].name);
                                let n2 = self.interner.get(self.prog.defs[d2.0 as usize].name);
                                let pn = self.interner.get(self.prog.defs[x.pool_def.0 as usize].name);
                                let cn = self.interner.get(c1);
                                self.diags.error(
                                    ErrorCode::E0601,
                                    if sp1 != Span::DUMMY { sp1 } else { span },
                                    format!(
                                        "conflicting `sys` column access on `{pn}.{cn}` between `{n1}` and `{n2}` without explicit ordering (`before`/`after`)"
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn collect_schedule_entry(
        &self,
        expr: &Expr,
        sys_defs: &mut Vec<(DefId, Span)>,
        seen_defs: &mut BTreeSet<DefId>,
        edges: &mut BTreeSet<(DefId, DefId)>,
    ) {
        let mut push_node = |d: DefId, sp: Span| {
            if seen_defs.insert(d) {
                sys_defs.push((d, sp));
            }
        };
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(self.mod_id, *sym)
                    && matches!(self.prog.defs[d.0 as usize].kind, TopDefKind::Sys(_))
                {
                    push_node(d, expr.span);
                }
            }
            ExprKind::Call { callee, args } => {
                // `sys_a.before(sys_b)` => `sys_a -> sys_b`, or `sys_a.after(sys_b)` => `sys_b -> sys_a`
                if let ExprKind::Field { base, field } = &callee.kind
                    && let Some(lhs_def) = self.resolve_sys_ident(base)
                {
                    let fname = self.interner.get(field.sym);
                    push_node(lhs_def, base.span);
                    if let Some(a0) = args.first()
                        && let Some(rhs_def) = self.resolve_sys_ident(a0)
                    {
                        push_node(rhs_def, a0.span);
                        if fname == "before" {
                            edges.insert((lhs_def, rhs_def));
                        } else if fname == "after" {
                            edges.insert((rhs_def, lhs_def));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn resolve_sys_ident(&self, expr: &Expr) -> Option<DefId> {
        if let ExprKind::Ident(sym) = &expr.kind
            && let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(self.mod_id, *sym)
            && matches!(self.prog.defs[d.0 as usize].kind, TopDefKind::Sys(_))
        {
            return Some(d);
        }
        None
    }
}

fn root_ident_of_lvalue(mut e: &Expr) -> Option<Symbol> {
    loop {
        match &e.kind {
            ExprKind::Ident(sym) => return Some(*sym),
            ExprKind::Field { base, .. }
            | ExprKind::Index { base, .. }
            | ExprKind::Slice { base, .. }
            | ExprKind::Deref(base)
            | ExprKind::Paren(base) => e = base,
            _ => return None,
        }
    }
}

fn iter_root_collection(e: &Expr) -> Option<Symbol> {
    match &e.kind {
        ExprKind::Call { callee, .. } => {
            if let ExprKind::Field { base, .. } = &callee.kind {
                root_ident_of_lvalue(base)
            } else {
                None
            }
        }
        ExprKind::Ident(sym) => Some(*sym),
        _ => None,
    }
}

fn iter_is_pool_each(e: &Expr, interner: &Interner) -> bool {
    if let ExprKind::Call { callee, .. } = &e.kind
        && let ExprKind::Field { field, .. } = &callee.kind
    {
        return matches!(interner.get(field.sym), "each" | "each_mut");
    }
    false
}

#[allow(dead_code)]
fn _effect_set_ref(_e: &EffectSet) {}
