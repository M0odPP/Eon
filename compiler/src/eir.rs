#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::ast::{BinaryOp, Expr, UnaryOp};
use crate::cfg::{
    AggregateKind, BlockId, CastKind, CfgBuiltinOp, CfgCallee, CfgConst, CfgFnKey, CfgFunction,
    CfgOperand, CfgPlace, CfgProgram, CfgProj, CfgRvalue, CfgStmt, CfgStmtKind, CfgTerm, LocalId,
    ScheduleDag,
};
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{BuiltinKind, DefId, ModuleId, ResolvedProgram, TopDefKind};
use crate::span::{Interner, Span, Symbol};
use crate::types::{BorrowInner, FnSig, GenericVal, Ty, TyId, TypeEnv};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EirValue(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EirSlot(pub u32);

#[derive(Clone, Debug)]
pub struct EirParForRegion {
    pub header_bb: BlockId,
    pub body_bb: BlockId,
    pub step_bb: BlockId,
    pub exit_bb: BlockId,
    pub idx_slot: EirSlot,
    pub end_slot: EirSlot,
    pub chunk_size: u64,
}

#[derive(Clone, Debug)]
pub struct LayoutSharedGroup {
    pub canonical_key: CfgFnKey,
    pub layout_tag: String,
    pub member_insts: Vec<(CfgFnKey, Vec<GenericVal>)>,
}

#[derive(Clone, Debug)]
pub struct EirProgram {
    pub functions: Vec<EirFunction>,
    pub by_key: BTreeMap<CfgFnKey, usize>,
    pub extern_fns: Vec<DefId>,
    pub dyn_impl_pairs: Vec<(DefId, TyId)>,
    pub layout_shared_groups: Vec<LayoutSharedGroup>,
    pub schedules: Vec<ScheduleDag>,
}

#[derive(Clone, Debug)]
pub struct EirSlotInfo {
    pub id: EirSlot,
    pub ty: TyId,
    pub name: Option<Symbol>,
}

#[derive(Clone, Debug)]
pub struct EirPhi {
    pub dst: EirValue,
    pub ty: TyId,
    pub incoming: Vec<(BlockId, EirValue)>,
}

#[derive(Clone, Debug)]
pub struct EirFunction {
    pub key: CfgFnKey,
    pub module: ModuleId,
    pub name: Symbol,
    pub body_span: Span,
    pub is_sys: bool,
    pub sig: FnSig,
    pub params: Vec<EirValue>,
    pub param_names: Vec<Symbol>,
    pub slots: Vec<EirSlotInfo>,
    pub value_tys: Vec<TyId>,
    pub val_names: BTreeMap<EirValue, Symbol>,
    pub blocks: Vec<EirBlock>,
    pub entry: BlockId,
    pub rpo: Vec<BlockId>,
    pub preds: Vec<Vec<BlockId>>,
    pub idom: Vec<Option<BlockId>>,
    pub noalias_params: BTreeSet<usize>,
    pub value_ranges: BTreeMap<EirValue, (i128, i128)>,
    pub proven_safe_insts: BTreeSet<(BlockId, usize)>,
    pub where_clauses: Vec<Expr>,
    pub ensures_clauses: Vec<Expr>,
    pub loop_invariants: BTreeMap<BlockId, Vec<Expr>>,
    pub base_generic: Option<(CfgFnKey, Vec<GenericVal>)>,
    pub layout_shared_tag: Option<String>,
    pub par_for_regions: Vec<EirParForRegion>,
}

impl EirFunction {
    pub fn block(&self, id: BlockId) -> &EirBlock {
        &self.blocks[id.0 as usize]
    }
}

#[derive(Clone, Debug)]
pub struct EirBlock {
    pub id: BlockId,
    pub phis: Vec<EirPhi>,
    pub insts: Vec<EirInst>,
    pub term: EirTerm,
}

#[derive(Clone, Debug)]
pub enum EirPlaceBase {
    Slot(EirSlot),
    Val(EirValue),
}

#[derive(Clone, Debug)]
pub enum EirProj {
    Deref,
    Field {
        name: Symbol,
        field_idx: usize,
        ty: TyId,
    },
    TupleField {
        index: usize,
        ty: TyId,
    },
    Index {
        index: EirValue,
        ty: TyId,
    },
    PoolCol {
        col: Symbol,
        col_idx: usize,
        is_slot_pool: bool,
        elem_ty: TyId,
    },
}

#[derive(Clone, Debug)]
pub struct EirPlace {
    pub base: EirPlaceBase,
    pub projs: Vec<EirProj>,
    pub ty: TyId,
}

#[derive(Clone, Debug)]
pub struct EirInst {
    pub span: Span,
    pub kind: EirInstKind,
}

#[derive(Clone, Debug)]
pub enum EirInstKind {
    Const {
        dst: EirValue,
        ty: TyId,
        konst: CfgConst,
    },
    CopyVal {
        dst: EirValue,
        src: EirValue,
    },
    LoadSlot {
        dst: EirValue,
        slot: EirSlot,
    },
    StoreSlot {
        slot: EirSlot,
        val: EirValue,
    },
    SlotAddr {
        dst: EirValue,
        slot: EirSlot,
        is_mut: bool,
        ty: TyId,
    },
    LoadPlace {
        dst: EirValue,
        place: EirPlace,
    },
    StorePlace {
        place: EirPlace,
        val: EirValue,
    },
    PoolColumnLoad {
        dst: EirValue,
        pool: EirPlaceBase,
        pool_is_ptr: bool,
        col: Symbol,
        col_idx: usize,
        is_slot_pool: bool,
        index: EirValue,
        elem_ty: TyId,
    },
    PoolColumnStore {
        pool: EirPlaceBase,
        pool_is_ptr: bool,
        col: Symbol,
        col_idx: usize,
        is_slot_pool: bool,
        index: EirValue,
        val: EirValue,
        elem_ty: TyId,
    },
    PlaceAddr {
        dst: EirValue,
        place: EirPlace,
        is_mut: bool,
        ty: TyId,
    },
    Unary {
        dst: EirValue,
        op: UnaryOp,
        operand: EirValue,
        ty: TyId,
    },
    Binary {
        dst: EirValue,
        op: BinaryOp,
        lhs: EirValue,
        rhs: EirValue,
        op_ty: TyId,
        dst_ty: TyId,
    },
    Cast {
        dst: EirValue,
        kind: CastKind,
        operand: EirValue,
        from_ty: TyId,
        to_ty: TyId,
    },
    Slice {
        dst: EirValue,
        base: EirValue,
        base_ty: TyId,
        lo: Option<EirValue>,
        hi: Option<EirValue>,
        dst_ty: TyId,
    },
    Aggregate {
        dst: EirValue,
        kind: AggregateKind,
        elems: Vec<EirValue>,
        dst_ty: TyId,
    },
    UnwrapOptional {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
        dst_ty: TyId,
    },
    UnwrapErrorOk {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
        dst_ty: TyId,
    },
    UnwrapErrorErr {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
        dst_ty: TyId,
    },
    IsNonNull {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
    },
    IsError {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
    },
    MatchTag {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
    },
    FieldExtract {
        dst: EirValue,
        src: EirValue,
        src_ty: TyId,
        field_idx: usize,
        dst_ty: TyId,
    },
    MakeDyn {
        dst: EirValue,
        ptr: EirValue,
        iface_def: DefId,
        concrete_ty: TyId,
        is_mut: bool,
        dst_ty: TyId,
    },
    MakeClosure {
        dst: EirValue,
        closure_id: usize,
        env_ptr: Option<EirValue>,
        fn_ty: TyId,
    },
    InlineAsm {
        template: String,
        inputs: Vec<(String, EirValue)>,
    },
    Call {
        dst: Option<EirValue>,
        callee: EirCallee,
        args: Vec<EirValue>,
        ret_ty: TyId,
    },
    BuiltinCall {
        dst: Option<EirValue>,
        op: CfgBuiltinOp,
        args: Vec<EirValue>,
        arg_slots: Vec<Option<EirSlot>>,
        ret_ty: TyId,
    },
    // Pseudo-instructions eliminated during SSA mem2reg
    GetVar {
        dst: EirValue,
        local: LocalId,
    },
    SetVar {
        local: LocalId,
        val: EirValue,
    },
}

#[derive(Clone, Debug)]
pub enum EirCallee {
    DirectFn(DefId),
    Method { impl_idx: usize, sub_idx: usize },
    GenericInst { inst_id: usize },
    DynCall { iface_def: DefId, method_idx: usize },
    ExternFn { def: DefId, name: Symbol },
    FnPtr(EirValue),
}

#[derive(Clone, Debug)]
pub enum EirTerm {
    Goto(BlockId),
    Branch {
        cond: EirValue,
        then_bb: BlockId,
        else_bb: BlockId,
        span: Span,
        is_user_if: bool,
    },
    SwitchInt {
        discr: EirValue,
        targets: Vec<(i128, BlockId)>,
        otherwise: BlockId,
    },
    Return(Option<EirValue>),
    Unreachable {
        span: Span,
        explicit: bool,
    },
}

/// Lower CFG functions into verified SSA EIR.
pub fn lower_program(
    cfg: &CfgProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    diags: &mut DiagBag,
) -> EirProgram {
    lower_program_opts(cfg, prog, env, diags, true)
}

/// Lower CFG functions into SSA EIR for checking/verification (skips unsupported constructs without error).
pub fn lower_program_for_verify(
    cfg: &CfgProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    diags: &mut DiagBag,
) -> EirProgram {
    lower_program_opts(cfg, prog, env, diags, false)
}

fn lower_program_opts(
    cfg: &CfgProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    diags: &mut DiagBag,
    strict_unsupported: bool,
) -> EirProgram {
    let mut eir = EirProgram {
        functions: Vec::new(),
        by_key: BTreeMap::new(),
        extern_fns: Vec::new(),
        dyn_impl_pairs: cfg.dyn_impl_pairs.clone(),
        layout_shared_groups: Vec::new(),
        schedules: cfg.schedules.clone(),
    };

    for (i, d) in prog.defs.iter().enumerate() {
        if matches!(d.kind, TopDefKind::ExternFn(_)) {
            eir.extern_fns.push(DefId(i as u32));
        }
    }

    for f in &cfg.functions {
        if !f.sig.generics.is_empty() || f.sig.is_comptime || sig_has_generic_param(&f.sig, env) {
            continue;
        }
        if !f.unsupported.is_empty() {
            if strict_unsupported {
                for u in &f.unsupported {
                    diags.error(
                        ErrorCode::E0230,
                        u.span,
                        format!("cannot lower `{}` to EIR", u.kind),
                    );
                }
            }
            continue;
        }
        let func = lower_cfg_function(f, env);
        let idx = eir.functions.len();
        eir.by_key.insert(f.key, idx);
        eir.functions.push(func);
    }

    eir
}

fn sig_has_generic_param(sig: &FnSig, env: &TypeEnv) -> bool {
    sig.params.iter().any(|p| ty_has_generic_param(p.ty, env)) || ty_has_generic_param(sig.ret, env)
}

fn ty_has_generic_param(id: TyId, env: &TypeEnv) -> bool {
    match env.get(id) {
        Ty::GenericParam { .. } => true,
        Ty::Optional(i) | Ty::RawPtr { inner: i, .. } | Ty::Array { elem: i, .. } => {
            ty_has_generic_param(*i, env)
        }
        Ty::ErrorUnion { ok, .. } => ty_has_generic_param(*ok, env),
        Ty::Borrow { inner, .. } => match inner {
            crate::types::BorrowInner::Single(t) | crate::types::BorrowInner::Slice(t) => {
                ty_has_generic_param(*t, env)
            }
            crate::types::BorrowInner::Str => false,
        },
        Ty::Tuple(es) => es.iter().any(|&t| ty_has_generic_param(t, env)),
        Ty::Builtin(_, args) => args
            .iter()
            .any(|g| matches!(g, GenericVal::Ty(t) if ty_has_generic_param(*t, env))),
        _ => false,
    }
}

fn is_ssa_promotable_ty(ty: TyId, env: &TypeEnv) -> bool {
    match env.get(ty) {
        Ty::Void | Ty::Never => false,
        Ty::Bool
        | Ty::Char
        | Ty::Int { .. }
        | Ty::Float(_)
        | Ty::PhysAddr
        | Ty::VirtAddr
        | Ty::DmaAddr
        | Ty::Enum(_)
        | Ty::ErrorSet(_)
        | Ty::FnPtr { .. } => true,
        Ty::Borrow { .. } | Ty::RawPtr { .. } | Ty::Str => true,
        Ty::Builtin(BuiltinKind::IdBrand | BuiltinKind::GenId | BuiltinKind::Link | BuiltinKind::Ordering, _) => {
            true
        }
        Ty::Optional(inner) => matches!(
            env.get(*inner),
            Ty::Builtin(BuiltinKind::IdBrand | BuiltinKind::GenId | BuiltinKind::Link, _)
        ),
        _ => false,
    }
}

fn lower_cfg_function(f: &CfgFunction, env: &TypeEnv) -> EirFunction {
    // 1. Compute reachable blocks and dense block remapping in RPO.
    let n_orig = f.blocks.len();
    let mut visited = vec![false; n_orig];
    let mut post = Vec::new();
    fn dfs_rpo(u: BlockId, f: &CfgFunction, vis: &mut [bool], post: &mut Vec<BlockId>) {
        vis[u.0 as usize] = true;
        for v in f.block(u).successors() {
            if (v.0 as usize) < vis.len() && !vis[v.0 as usize] {
                dfs_rpo(v, f, vis, post);
            }
        }
        post.push(u);
    }
    dfs_rpo(f.entry, f, &mut visited, &mut post);
    post.reverse();

    let n_blocks = post.len();
    let mut orig_to_dense = vec![None; n_orig];
    for (i, &ob) in post.iter().enumerate() {
        orig_to_dense[ob.0 as usize] = Some(BlockId(i as u32));
    }

    // 2. Partition locals into SSA-promoted vs explicit EirSlots.
    let mut must_slot = vec![false; f.locals.len()];
    for (i, loc) in f.locals.iter().enumerate() {
        if !is_ssa_promotable_ty(loc.ty, env) {
            must_slot[i] = true;
        }
    }
    for &ob in &post {
        let b = f.block(ob);
        for s in &b.stmts {
            mark_non_promotable_stmt(s, &mut must_slot);
        }
    }
    for pf in &f.par_for_regions {
        must_slot[pf.idx_local.0 as usize] = true;
        must_slot[pf.end_local.0 as usize] = true;
    }

    let mut slots: Vec<EirSlotInfo> = Vec::new();
    let mut slot_of: Vec<Option<EirSlot>> = vec![None; f.locals.len()];
    let mut is_promoted = vec![false; f.locals.len()];
    for (i, loc) in f.locals.iter().enumerate() {
        if must_slot[i] {
            let sid = EirSlot(slots.len() as u32);
            slots.push(EirSlotInfo {
                id: sid,
                ty: loc.ty,
                name: loc.name,
            });
            slot_of[i] = Some(sid);
        } else if loc.ty != TyId::VOID && loc.ty != TyId::NEVER {
            is_promoted[i] = true;
        }
    }

    let mut bldr = EirBuilder {
        cfg_fn: f,
        env,
        orig_to_dense,
        slots,
        slot_of,
        is_promoted,
        value_tys: Vec::new(),
        blocks: (0..n_blocks)
            .map(|i| EirBlock {
                id: BlockId(i as u32),
                phis: Vec::new(),
                insts: Vec::new(),
                term: EirTerm::Unreachable {
                    span: Span::DUMMY,
                    explicit: false,
                },
            })
            .collect(),
        params: Vec::new(),
    };

    bldr.lower(post);
    bldr.run_mem2reg_and_cfg_meta()
}

fn mark_non_promotable_stmt(stmt: &CfgStmt, must_slot: &mut [bool]) {
    match &stmt.kind {
        CfgStmtKind::Assign { dst, rvalue } => {
            if !dst.projs.is_empty() {
                must_slot[dst.local.0 as usize] = true;
            }
            if let CfgRvalue::Borrow { place, .. } = rvalue {
                must_slot[place.local.0 as usize] = true;
            }
        }
        CfgStmtKind::Call {
            dst,
            recv_place,
            is_mut_recv,
            consumes_recv,
            ..
        } => {
            if let Some(d) = dst
                && !d.projs.is_empty()
            {
                must_slot[d.local.0 as usize] = true;
            }
            if (*is_mut_recv || *consumes_recv)
                && let Some(rp) = recv_place
            {
                must_slot[rp.local.0 as usize] = true;
            }
        }
        _ => {}
    }
}

struct EirBuilder<'a> {
    cfg_fn: &'a CfgFunction,
    env: &'a TypeEnv,
    orig_to_dense: Vec<Option<BlockId>>,
    slots: Vec<EirSlotInfo>,
    slot_of: Vec<Option<EirSlot>>,
    is_promoted: Vec<bool>,
    value_tys: Vec<TyId>,
    blocks: Vec<EirBlock>,
    params: Vec<EirValue>,
}

impl<'a> EirBuilder<'a> {
    fn new_val(&mut self, ty: TyId) -> EirValue {
        let id = EirValue(self.value_tys.len() as u32);
        self.value_tys.push(ty);
        id
    }

    fn map_bb(&self, orig: BlockId) -> BlockId {
        self.orig_to_dense[orig.0 as usize].unwrap_or(BlockId(0))
    }

    fn lower(&mut self, post: Vec<BlockId>) {
        // Entry block receives parameter definitions.
        let mut init_insts = Vec::new();
        for &plid in &self.cfg_fn.params {
            let pty = self.cfg_fn.local(plid).ty;
            let pv = self.new_val(pty);
            self.params.push(pv);
            if let Some(slot) = self.slot_of[plid.0 as usize] {
                init_insts.push(EirInst {
                    span: self.cfg_fn.local(plid).span,
                    kind: EirInstKind::StoreSlot { slot, val: pv },
                });
            } else if self.is_promoted[plid.0 as usize] {
                init_insts.push(EirInst {
                    span: self.cfg_fn.local(plid).span,
                    kind: EirInstKind::SetVar { local: plid, val: pv },
                });
            }
        }
        self.blocks[0].insts.extend(init_insts);

        for (bi, &ob) in post.iter().enumerate() {
            let cur_bb = BlockId(bi as u32);
            let cfg_b = self.cfg_fn.block(ob);
            for stmt in &cfg_b.stmts {
                self.lower_stmt(cur_bb, stmt);
            }
            let term = match &cfg_b.term {
                CfgTerm::Goto(t) => EirTerm::Goto(self.map_bb(*t)),
                CfgTerm::Branch { cond, then_bb, else_bb, span, is_user_if } => {
                    let cv = self.lower_operand(cur_bb, cond, *span);
                    EirTerm::Branch {
                        cond: cv,
                        then_bb: self.map_bb(*then_bb),
                        else_bb: self.map_bb(*else_bb),
                        span: *span,
                        is_user_if: *is_user_if,
                    }
                }
                CfgTerm::SwitchInt { discr, targets, otherwise, span } => {
                    let dv = self.lower_operand(cur_bb, discr, *span);
                    EirTerm::SwitchInt {
                        discr: dv,
                        targets: targets.iter().map(|&(v, b)| (v, self.map_bb(b))).collect(),
                        otherwise: self.map_bb(*otherwise),
                    }
                }
                CfgTerm::Return { value, span } => {
                    let rv = value.as_ref().map(|op| self.lower_operand(cur_bb, op, *span));
                    EirTerm::Return(rv)
                }
                CfgTerm::Unreachable { span, explicit } => EirTerm::Unreachable {
                    span: *span,
                    explicit: *explicit,
                },
            };
            self.blocks[bi].term = term;
        }
    }

    fn emit(&mut self, bb: BlockId, span: Span, kind: EirInstKind) {
        self.blocks[bb.0 as usize]
            .insts
            .push(EirInst { span, kind });
    }

    fn lower_stmt(&mut self, bb: BlockId, stmt: &CfgStmt) {
        match &stmt.kind {
            CfgStmtKind::StorageLive(lid) => {
                if self.is_promoted[lid.0 as usize] {
                    let lty = self.cfg_fn.local(*lid).ty;
                    let z = self.emit_zero_const(bb, lty, stmt.span);
                    self.emit(bb, stmt.span, EirInstKind::SetVar { local: *lid, val: z });
                }
            }
            CfgStmtKind::StorageDead(_)
            | CfgStmtKind::DiscardLinearError { .. }
            | CfgStmtKind::ConsumeDestructured(_)
            | CfgStmtKind::PoolResetEpoch { .. } => {}
            CfgStmtKind::Assign { dst, rvalue } => {
                let rv_val = self.lower_rvalue(bb, rvalue, Some(dst.ty), stmt.span);
                self.store_cfg_place(bb, dst, rv_val, stmt.span);
            }
            CfgStmtKind::Call {
                dst,
                callee,
                args,
                recv_place,
                ..
            } => {
                let ret_ty = dst
                    .as_ref()
                    .map_or(TyId::VOID, |d| d.ty);
                let dst_val = if ret_ty == TyId::VOID {
                    None
                } else {
                    Some(self.new_val(ret_ty))
                };

                let mut eargs = Vec::with_capacity(args.len());
                let mut arg_slots = Vec::with_capacity(args.len());
                for a in args {
                    arg_slots.push(self.operand_root_slot(a));
                    eargs.push(self.lower_operand(bb, a, stmt.span));
                }

                match callee {
                    CfgCallee::DirectFn(def_id) => {
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::DirectFn(*def_id),
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::Method { impl_idx, sub_idx, .. } => {
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::Method {
                                    impl_idx: *impl_idx,
                                    sub_idx: *sub_idx,
                                },
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::GenericInst { inst_id, .. } => {
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::GenericInst { inst_id: *inst_id },
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::DynCall { iface_def, method_idx } => {
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::DynCall {
                                    iface_def: *iface_def,
                                    method_idx: *method_idx,
                                },
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::ExternFn { def, name } => {
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::ExternFn {
                                    def: *def,
                                    name: *name,
                                },
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::FnPtr(op) => {
                        let fv = self.lower_operand(bb, op, stmt.span);
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::Call {
                                dst: dst_val,
                                callee: EirCallee::FnPtr(fv),
                                args: eargs,
                                ret_ty,
                            },
                        );
                    }
                    CfgCallee::Builtin(bop) => {
                        if let Some(rp) = recv_place
                            && rp.projs.is_empty()
                            && !matches!(self.env.get(rp.ty), Ty::Borrow { .. } | Ty::RawPtr { .. })
                            && arg_slots.first().is_some_and(Option::is_none)
                        {
                            arg_slots[0] = self.slot_of[rp.local.0 as usize];
                        }
                        self.emit(
                            bb,
                            stmt.span,
                            EirInstKind::BuiltinCall {
                                dst: dst_val,
                                op: bop.clone(),
                                args: eargs,
                                arg_slots,
                                ret_ty,
                            },
                        );
                    }
                }

                if let (Some(dplace), Some(dv)) = (dst, dst_val) {
                    self.store_cfg_place(bb, dplace, dv, stmt.span);
                }
            }
        }
    }

    fn operand_root_slot(&self, op: &CfgOperand) -> Option<EirSlot> {
        match op {
            CfgOperand::Copy(p) | CfgOperand::Move(p)
                if p.projs.is_empty()
                    && !matches!(self.env.get(p.ty), Ty::Borrow { .. } | Ty::RawPtr { .. }) =>
            {
                let s = self.slot_of[p.local.0 as usize];
                if s.is_some() {
                    return s;
                }
                None
            }
            _ => None,
        }
    }

    fn build_eir_place(&mut self, bb: BlockId, place: &CfgPlace, span: Span) -> EirPlace {
        let base = if let Some(slot) = self.slot_of[place.local.0 as usize] {
            EirPlaceBase::Slot(slot)
        } else {
            let v = self.new_val(self.cfg_fn.local(place.local).ty);
            self.emit(bb, span, EirInstKind::GetVar { dst: v, local: place.local });
            EirPlaceBase::Val(v)
        };
        let mut cur_ty = self.cfg_fn.local(place.local).ty;
        let mut projs = Vec::with_capacity(place.projs.len());
        for pr in &place.projs {
            match pr {
                CfgProj::Deref => {
                    cur_ty = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    projs.push(EirProj::Deref);
                }
                CfgProj::Field { name, field_idx, ty } => {
                    cur_ty = *ty;
                    projs.push(EirProj::Field {
                        name: *name,
                        field_idx: *field_idx,
                        ty: *ty,
                    });
                }
                CfgProj::TupleField { index, ty } => {
                    cur_ty = *ty;
                    projs.push(EirProj::TupleField {
                        index: *index,
                        ty: *ty,
                    });
                }
                CfgProj::PoolCol { col, col_idx, is_slot_pool, elem_ty } => {
                    cur_ty = *elem_ty;
                    projs.push(EirProj::PoolCol {
                        col: *col,
                        col_idx: *col_idx,
                        is_slot_pool: *is_slot_pool,
                        elem_ty: *elem_ty,
                    });
                }
                CfgProj::Index { index, ty, .. } => {
                    let iv = self.lower_operand(bb, index, span);
                    cur_ty = *ty;
                    projs.push(EirProj::Index {
                        index: iv,
                        ty: *ty,
                    });
                }
            }
        }
        EirPlace {
            base,
            projs,
            ty: place.ty,
        }
    }

    fn store_cfg_place(&mut self, bb: BlockId, place: &CfgPlace, val: EirValue, span: Span) {
        if place.projs.is_empty() {
            if let Some(slot) = self.slot_of[place.local.0 as usize] {
                self.emit(bb, span, EirInstKind::StoreSlot { slot, val });
            } else if self.is_promoted[place.local.0 as usize] {
                self.emit(bb, span, EirInstKind::SetVar { local: place.local, val });
            }
            return;
        }
        let ep = self.build_eir_place(bb, place, span);
        if ep.projs.len() == 2
            && let EirProj::PoolCol { col, col_idx, is_slot_pool, elem_ty } = ep.projs[0]
            && let EirProj::Index { index, .. } = ep.projs[1]
        {
            let pool_is_ptr = matches!(
                self.env.get(self.cfg_fn.local(place.local).ty),
                Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
            );
            self.emit(
                bb,
                span,
                EirInstKind::PoolColumnStore {
                    pool: ep.base,
                    pool_is_ptr,
                    col,
                    col_idx,
                    is_slot_pool,
                    index,
                    val,
                    elem_ty,
                },
            );
            return;
        }
        self.emit(bb, span, EirInstKind::StorePlace { place: ep, val });
    }

    fn lower_operand(&mut self, bb: BlockId, op: &CfgOperand, span: Span) -> EirValue {
        match op {
            CfgOperand::Const(c) => {
                let cty = c.ty(self.env, self.cfg_fn);
                let dst = self.new_val(cty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Const {
                        dst,
                        ty: cty,
                        konst: c.clone(),
                    },
                );
                dst
            }
            CfgOperand::Copy(p) | CfgOperand::Move(p) => {
                if p.projs.is_empty() {
                    let lty = if p.ty != TyId::ERROR {
                        p.ty
                    } else {
                        self.cfg_fn.locals[p.local.0 as usize].ty
                    };
                    if let Some(slot) = self.slot_of[p.local.0 as usize] {
                        let dst = self.new_val(lty);
                        self.emit(bb, span, EirInstKind::LoadSlot { dst, slot });
                        dst
                    } else if self.is_promoted[p.local.0 as usize] {
                        let dst = self.new_val(lty);
                        self.emit(
                            bb,
                            span,
                            EirInstKind::GetVar {
                                dst,
                                local: p.local,
                            },
                        );
                        dst
                    } else {
                        self.emit_zero_const(bb, lty, span)
                    }
                } else {
                    let ep = self.build_eir_place(bb, p, span);
                    let dst = self.new_val(p.ty);
                    if ep.projs.len() == 2
                        && let EirProj::PoolCol { col, col_idx, is_slot_pool, elem_ty } = ep.projs[0]
                        && let EirProj::Index { index, .. } = ep.projs[1]
                    {
                        let pool_is_ptr = matches!(
                            self.env.get(self.cfg_fn.local(p.local).ty),
                            Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                        );
                        self.emit(
                            bb,
                            span,
                            EirInstKind::PoolColumnLoad {
                                dst,
                                pool: ep.base,
                                pool_is_ptr,
                                col,
                                col_idx,
                                is_slot_pool,
                                index,
                                elem_ty,
                            },
                        );
                        return dst;
                    }
                    self.emit(bb, span, EirInstKind::LoadPlace { dst, place: ep });
                    dst
                }
            }
        }
    }

    fn lower_rvalue(&mut self, bb: BlockId, rv: &CfgRvalue, expected_dst: Option<TyId>, span: Span) -> EirValue {
        match rv {
            CfgRvalue::Use(op) => self.lower_operand(bb, op, span),
            CfgRvalue::Borrow { is_mut, place } => {
                let borrow_ty = expected_dst
                    .filter(|&t| t != TyId::ERROR)
                    .unwrap_or_else(|| self.make_borrow_ty(place.ty, *is_mut));
                let dst = self.new_val(borrow_ty);
                if place.projs.is_empty()
                    && let Some(slot) = self.slot_of[place.local.0 as usize]
                {
                    self.emit(
                        bb,
                        span,
                        EirInstKind::SlotAddr {
                            dst,
                            slot,
                            is_mut: *is_mut,
                            ty: borrow_ty,
                        },
                    );
                } else {
                    let ep = self.build_eir_place(bb, place, span);
                    self.emit(
                        bb,
                        span,
                        EirInstKind::PlaceAddr {
                            dst,
                            place: ep,
                            is_mut: *is_mut,
                            ty: borrow_ty,
                        },
                    );
                }
                dst
            }
            CfgRvalue::Unary { op, operand } => {
                let ot = operand.ty(self.cfg_fn, self.env);
                let ov = self.lower_operand(bb, operand, span);
                let dst_ty = if *op == UnaryOp::Not { TyId::BOOL } else { ot };
                let dst = self.new_val(dst_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Unary {
                        dst,
                        op: *op,
                        operand: ov,
                        ty: dst_ty,
                    },
                );
                dst
            }
            CfgRvalue::Binary { op, lhs, rhs } => {
                let op_ty = lhs.ty(self.cfg_fn, self.env);
                let lv = self.lower_operand(bb, lhs, span);
                let rv_v = self.lower_operand(bb, rhs, span);
                let dst_ty = match op {
                    BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::LtEq
                    | BinaryOp::Gt
                    | BinaryOp::GtEq
                    | BinaryOp::And
                    | BinaryOp::Or => TyId::BOOL,
                    BinaryOp::AddCheck
                    | BinaryOp::SubCheck
                    | BinaryOp::MulCheck
                    | BinaryOp::DivCheck
                    | BinaryOp::ModCheck => expected_dst.unwrap_or(op_ty),
                    _ => op_ty,
                };
                let dst = self.new_val(dst_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Binary {
                        dst,
                        op: *op,
                        lhs: lv,
                        rhs: rv_v,
                        op_ty,
                        dst_ty,
                    },
                );
                dst
            }
            CfgRvalue::Slice { base, lo, hi } => {
                let bty = base.ty(self.cfg_fn, self.env);
                let out_ty = expected_dst.filter(|&t| t != TyId::ERROR).unwrap_or(bty);
                let bv = self.lower_operand(bb, base, span);
                let lv = lo.as_ref().map(|x| self.lower_operand(bb, x, span));
                let hv = hi.as_ref().map(|x| self.lower_operand(bb, x, span));
                let dst = self.new_val(out_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Slice {
                        dst,
                        base: bv,
                        base_ty: bty,
                        lo: lv,
                        hi: hv,
                        dst_ty: out_ty,
                    },
                );
                dst
            }
            CfgRvalue::Cast { kind, operand, to_ty } => {
                let from_ty = operand.ty(self.cfg_fn, self.env);
                let ov = self.lower_operand(bb, operand, span);
                let dst = self.new_val(*to_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Cast {
                        dst,
                        kind: *kind,
                        operand: ov,
                        from_ty,
                        to_ty: *to_ty,
                    },
                );
                dst
            }
            CfgRvalue::Aggregate { kind, elems } => {
                let dst_ty = expected_dst
                    .filter(|&t| t != TyId::ERROR)
                    .unwrap_or_else(|| self.infer_agg_ty(kind, elems));
                let mut evs = Vec::with_capacity(elems.len());
                for e in elems {
                    evs.push(self.lower_operand(bb, e, span));
                }
                let dst = self.new_val(dst_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Aggregate {
                        dst,
                        kind: kind.clone(),
                        elems: evs,
                        dst_ty,
                    },
                );
                dst
            }
            CfgRvalue::UnwrapOptional(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let dt = match self.env.get(st) {
                    Ty::Optional(inner) => *inner,
                    _ => st,
                };
                let sv = self.lower_operand(bb, op, span);
                let dst = self.new_val(dt);
                self.emit(
                    bb,
                    span,
                    EirInstKind::UnwrapOptional {
                        dst,
                        src: sv,
                        src_ty: st,
                        dst_ty: dt,
                    },
                );
                dst
            }
            CfgRvalue::UnwrapErrorOk(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let dt = match self.env.get(st) {
                    Ty::ErrorUnion { ok, .. } => *ok,
                    _ => st,
                };
                let sv = self.lower_operand(bb, op, span);
                let dst = self.new_val(dt);
                self.emit(
                    bb,
                    span,
                    EirInstKind::UnwrapErrorOk {
                        dst,
                        src: sv,
                        src_ty: st,
                        dst_ty: dt,
                    },
                );
                dst
            }
            CfgRvalue::UnwrapErrorErr(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let sv = self.lower_operand(bb, op, span);
                let dt = expected_dst
                    .filter(|&t| t != TyId::ERROR)
                    .unwrap_or(TyId::U32);
                let dst = self.new_val(dt);
                self.emit(
                    bb,
                    span,
                    EirInstKind::UnwrapErrorErr {
                        dst,
                        src: sv,
                        src_ty: st,
                        dst_ty: dt,
                    },
                );
                dst
            }
            CfgRvalue::IsNonNull(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let sv = self.lower_operand(bb, op, span);
                let dst = self.new_val(TyId::BOOL);
                self.emit(
                    bb,
                    span,
                    EirInstKind::IsNonNull {
                        dst,
                        src: sv,
                        src_ty: st,
                    },
                );
                dst
            }
            CfgRvalue::IsError(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let sv = self.lower_operand(bb, op, span);
                let dst = self.new_val(TyId::BOOL);
                self.emit(
                    bb,
                    span,
                    EirInstKind::IsError {
                        dst,
                        src: sv,
                        src_ty: st,
                    },
                );
                dst
            }
            CfgRvalue::MatchTag(op) => {
                let st = op.ty(self.cfg_fn, self.env);
                let sv = self.lower_operand(bb, op, span);
                let dst = self.new_val(TyId::U32);
                self.emit(
                    bb,
                    span,
                    EirInstKind::MatchTag {
                        dst,
                        src: sv,
                        src_ty: st,
                    },
                );
                dst
            }
            CfgRvalue::FieldExtract {
                operand,
                field_idx,
                ty,
            } => {
                let st = operand.ty(self.cfg_fn, self.env);
                let sv = self.lower_operand(bb, operand, span);
                let dst = self.new_val(*ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::FieldExtract {
                        dst,
                        src: sv,
                        src_ty: st,
                        field_idx: *field_idx,
                        dst_ty: *ty,
                    },
                );
                dst
            }
            CfgRvalue::MakeDyn { ptr, iface_def, concrete_ty, is_mut } => {
                let pv = self.lower_operand(bb, ptr, span);
                let dst_ty = expected_dst.unwrap_or(TyId::ERROR);
                let dst = self.new_val(dst_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::MakeDyn {
                        dst,
                        ptr: pv,
                        iface_def: *iface_def,
                        concrete_ty: *concrete_ty,
                        is_mut: *is_mut,
                        dst_ty,
                    },
                );
                dst
            }
            CfgRvalue::MakeClosure { closure_id, env_ptr, fn_ty } => {
                let ev = env_ptr.as_ref().map(|op| self.lower_operand(bb, op, span));
                let dst = self.new_val(*fn_ty);
                self.emit(
                    bb,
                    span,
                    EirInstKind::MakeClosure {
                        dst,
                        closure_id: *closure_id,
                        env_ptr: ev,
                        fn_ty: *fn_ty,
                    },
                );
                dst
            }
            CfgRvalue::InlineAsm { template, inputs } => {
                let mut lowered = Vec::with_capacity(inputs.len());
                for (k, op) in inputs {
                    let v = self.lower_operand(bb, op, span);
                    lowered.push((k.clone(), v));
                }
                self.emit(
                    bb,
                    span,
                    EirInstKind::InlineAsm {
                        template: template.clone(),
                        inputs: lowered,
                    },
                );
                let dst = self.new_val(TyId::VOID);
                self.emit(
                    bb,
                    span,
                    EirInstKind::Const {
                        dst,
                        ty: TyId::VOID,
                        konst: CfgConst::Void,
                    },
                );
                dst
            }
        }
    }

    fn make_borrow_ty(&self, inner_ty: TyId, _is_mut: bool) -> TyId {
        inner_ty
    }

    fn infer_agg_ty(&self, kind: &AggregateKind, elems: &[CfgOperand]) -> TyId {
        let _ = elems;
        match kind {
            AggregateKind::Struct(_)
            | AggregateKind::PoolRow { .. }
            | AggregateKind::UnionVariant { .. }
            | AggregateKind::AnonStruct(_)
            | AggregateKind::Tuple
            | AggregateKind::Array(_)
            | AggregateKind::ArrayRepeat { .. }
            | AggregateKind::Bitfield(_)
            | AggregateKind::Vec { .. } => TyId::ERROR,
        }
    }

    fn emit_zero_const(&mut self, bb: BlockId, ty: TyId, span: Span) -> EirValue {
        let dst = self.new_val(ty);
        let konst = match self.env.get(ty) {
            Ty::Bool => CfgConst::Bool(false),
            Ty::Char => CfgConst::Char('\0'),
            Ty::Float(_) => CfgConst::Float("0.0".into(), ty),
            _ if self.env.is_integer(ty) || self.env.is_address(ty) => CfgConst::Int(0, ty),
            _ => CfgConst::Null(ty),
        };
        self.emit(bb, span, EirInstKind::Const { dst, ty, konst });
        dst
    }

    fn run_mem2reg_and_cfg_meta(mut self) -> EirFunction {
        // Patch `Aggregate` dst_ty and `Borrow` dst_ty from subsequent `StoreSlot` / `StorePlace` where possible.
        self.propagate_missing_tys();

        let n = self.blocks.len();
        let mut rpo = Vec::with_capacity(n);
        for i in 0..n {
            rpo.push(BlockId(i as u32));
        }

        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for (i, blk) in self.blocks.iter().enumerate().take(n) {
            let succs = term_succs(&blk.term);
            for s in succs {
                if !preds[s.0 as usize].contains(&BlockId(i as u32)) {
                    preds[s.0 as usize].push(BlockId(i as u32));
                }
            }
        }

        let mut idom = vec![None; n];
        idom[0] = Some(BlockId(0));
        let mut changed = true;
        while changed {
            changed = false;
            for bi in 1..n {
                let mut new_idom = None;
                for &p in &preds[bi] {
                    if idom[p.0 as usize].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect_dense(&idom, p, cur),
                    });
                }
                if new_idom.is_some() && idom[bi] != new_idom {
                    idom[bi] = new_idom;
                    changed = true;
                }
            }
        }

        // Compute dominance frontiers.
        let mut df: Vec<BTreeSet<BlockId>> = vec![BTreeSet::new(); n];
        for bi in 0..n {
            if preds[bi].len() >= 2 {
                let Some(id_b) = idom[bi] else { continue };
                for &p in &preds[bi] {
                    let mut runner = p;
                    while runner != id_b {
                        df[runner.0 as usize].insert(BlockId(bi as u32));
                        let Some(next) = idom[runner.0 as usize] else { break };
                        if next == runner {
                            break;
                        }
                        runner = next;
                    }
                }
            }
        }

        // Mem2Reg phi placement for promoted locals.
        let promoted: Vec<LocalId> = self
            .cfg_fn
            .locals
            .iter()
            .filter(|l| self.is_promoted[l.id.0 as usize])
            .map(|l| l.id)
            .collect();

        // Map (BlockId, phi_idx) -> LocalId so rename pushes to right stack.
        let mut phi_owner: BTreeMap<(u32, usize), LocalId> = BTreeMap::new();
        for &lid in &promoted {
            let mut def_blocks = BTreeSet::new();
            for (bi, blk) in self.blocks.iter().enumerate() {
                for inst in &blk.insts {
                    if let EirInstKind::SetVar { local, .. } = &inst.kind
                        && *local == lid
                    {
                        def_blocks.insert(BlockId(bi as u32));
                    }
                }
            }
            if def_blocks.is_empty() {
                continue;
            }
            let mut has_phi = BTreeSet::new();
            let mut work: Vec<BlockId> = def_blocks.iter().copied().collect();
            while let Some(b) = work.pop() {
                let frontier: Vec<BlockId> = df[b.0 as usize].iter().copied().collect();
                for d in frontier {
                    if has_phi.insert(d) {
                        let phi_ty = self.cfg_fn.local(lid).ty;
                        let phi_val = self.new_val(phi_ty);
                        let pidx = self.blocks[d.0 as usize].phis.len();
                        self.blocks[d.0 as usize].phis.push(EirPhi {
                            dst: phi_val,
                            ty: phi_ty,
                            incoming: Vec::new(),
                        });
                        phi_owner.insert((d.0, pidx), lid);
                        if !def_blocks.contains(&d) {
                            work.push(d);
                        }
                    }
                }
            }
        }

        // Build dominator tree children.
        let mut dom_children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for (bi, &pid) in idom.iter().enumerate().take(n).skip(1) {
            if let Some(p) = pid
                && p.0 as usize != bi
            {
                dom_children[p.0 as usize].push(BlockId(bi as u32));
            }
        }

        // Emit fallback zero-const in entry for any promoted local so stacks never underflow.
        let mut stacks: BTreeMap<LocalId, Vec<EirValue>> = BTreeMap::new();
        let mut entry_defaults = Vec::new();
        for &lid in &promoted {
            let lty = self.cfg_fn.local(lid).ty;
            let zv = self.new_val(lty);
            let konst = match self.env.get(lty) {
                Ty::Bool => CfgConst::Bool(false),
                Ty::Char => CfgConst::Char('\0'),
                Ty::Float(_) => CfgConst::Float("0.0".into(), lty),
                _ if self.env.is_integer(lty) || self.env.is_address(lty) => CfgConst::Int(0, lty),
                _ => CfgConst::Null(lty),
            };
            entry_defaults.push(EirInst {
                span: Span::DUMMY,
                kind: EirInstKind::Const {
                    dst: zv,
                    ty: lty,
                    konst,
                },
            });
            stacks.insert(lid, vec![zv]);
        }
        entry_defaults.append(&mut self.blocks[0].insts);
        self.blocks[0].insts = entry_defaults;

        let mut val_remap: BTreeMap<EirValue, EirValue> = BTreeMap::new();
        let mut val_names: BTreeMap<EirValue, Symbol> = BTreeMap::new();
        let mut param_names = Vec::with_capacity(self.params.len());
        for (pi, &pv) in self.params.iter().enumerate() {
            if let Some(ps) = self.cfg_fn.sig.params.get(pi) {
                param_names.push(ps.name);
                val_names.insert(pv, ps.name);
            }
        }
        let mut ctx = RenameCtx {
            blocks: &mut self.blocks,
            dom_children: &dom_children,
            phi_owner: &phi_owner,
            stacks: &mut stacks,
            val_remap: &mut val_remap,
            val_names: &mut val_names,
            cfg_fn: self.cfg_fn,
        };
        rename_block(BlockId(0), &mut ctx);

        // Compute proven `noalias` parameters from Eon's exclusivity rules:
        // Every `*mut T` / `*mut [T]` borrow parameter is guaranteed non-aliasing with all other
        // parameters; every `*T` / `*[T]` borrow parameter is non-aliasing with all `*mut` params.
        let mut noalias_params = BTreeSet::new();
        let any_mut_borrow = self
            .cfg_fn
            .sig
            .params
            .iter()
            .any(|p| matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. }));
        for (pi, ps) in self.cfg_fn.sig.params.iter().enumerate() {
            if let Ty::Borrow { is_mut, .. } = self.env.get(ps.ty)
                && (*is_mut || any_mut_borrow)
            {
                noalias_params.insert(pi);
            }
        }

        let mut loop_invariants = BTreeMap::new();
        for (&old_bb, invs) in &self.cfg_fn.loop_invariants {
            if let Some(new_bb) = self.orig_to_dense.get(old_bb.0 as usize).copied().flatten() {
                loop_invariants.insert(new_bb, invs.clone());
            }
        }

        let mut par_for_regions = Vec::new();
        for pf in &self.cfg_fn.par_for_regions {
            if let (Some(h), Some(b), Some(s), Some(e), Some(is), Some(es)) = (
                self.orig_to_dense.get(pf.header_bb.0 as usize).copied().flatten(),
                self.orig_to_dense.get(pf.body_bb.0 as usize).copied().flatten(),
                self.orig_to_dense.get(pf.step_bb.0 as usize).copied().flatten(),
                self.orig_to_dense.get(pf.exit_bb.0 as usize).copied().flatten(),
                self.slot_of.get(pf.idx_local.0 as usize).copied().flatten(),
                self.slot_of.get(pf.end_local.0 as usize).copied().flatten(),
            ) {
                par_for_regions.push(EirParForRegion {
                    header_bb: h,
                    body_bb: b,
                    step_bb: s,
                    exit_bb: e,
                    idx_slot: is,
                    end_slot: es,
                    chunk_size: pf.chunk_size,
                });
            }
        }

        EirFunction {
            key: self.cfg_fn.key,
            module: self.cfg_fn.module,
            name: self.cfg_fn.name,
            body_span: self.cfg_fn.body_span,
            is_sys: self.cfg_fn.is_sys,
            sig: self.cfg_fn.sig.clone(),
            params: self.params,
            param_names,
            slots: self.slots,
            value_tys: self.value_tys,
            val_names,
            blocks: self.blocks,
            entry: BlockId(0),
            rpo,
            preds,
            idom,
            noalias_params,
            value_ranges: BTreeMap::new(),
            proven_safe_insts: BTreeSet::new(),
            where_clauses: self.cfg_fn.where_clauses.clone(),
            ensures_clauses: self.cfg_fn.ensures_clauses.clone(),
            loop_invariants,
            base_generic: self.cfg_fn.base_generic.clone(),
            layout_shared_tag: None,
            par_for_regions,
        }
    }

    fn propagate_missing_tys(&mut self) {
        let mut expected_for_val: BTreeMap<EirValue, TyId> = BTreeMap::new();
        for blk in &self.blocks {
            for inst in &blk.insts {
                if let EirInstKind::StoreSlot { slot, val } = &inst.kind {
                    expected_for_val.insert(*val, self.slots[slot.0 as usize].ty);
                } else if let EirInstKind::StorePlace { place, val } = &inst.kind {
                    expected_for_val.insert(*val, place.ty);
                } else if let EirInstKind::SetVar { local, val } = &inst.kind {
                    expected_for_val.insert(*val, self.cfg_fn.local(*local).ty);
                }
            }
        }
        for blk in &mut self.blocks {
            for inst in &mut blk.insts {
                match &mut inst.kind {
                    EirInstKind::Aggregate { dst, dst_ty, .. }
                    | EirInstKind::SlotAddr { dst, ty: dst_ty, .. }
                    | EirInstKind::PlaceAddr { dst, ty: dst_ty, .. } => {
                        if *dst_ty == TyId::ERROR
                            && let Some(&et) = expected_for_val.get(dst)
                            && et != TyId::ERROR
                        {
                            *dst_ty = et;
                            self.value_tys[dst.0 as usize] = et;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

struct RenameCtx<'a> {
    blocks: &'a mut [EirBlock],
    dom_children: &'a [Vec<BlockId>],
    phi_owner: &'a BTreeMap<(u32, usize), LocalId>,
    stacks: &'a mut BTreeMap<LocalId, Vec<EirValue>>,
    val_remap: &'a mut BTreeMap<EirValue, EirValue>,
    val_names: &'a mut BTreeMap<EirValue, Symbol>,
    cfg_fn: &'a CfgFunction,
}

fn resolve_val(v: EirValue, remap: &BTreeMap<EirValue, EirValue>) -> EirValue {
    let mut cur = v;
    let mut guard = 0usize;
    while let Some(&next) = remap.get(&cur) {
        if next == cur || guard > 128 {
            break;
        }
        cur = next;
        guard += 1;
    }
    cur
}

fn rewrite_place(p: &mut EirPlace, remap: &BTreeMap<EirValue, EirValue>) {
    if let EirPlaceBase::Val(v) = &mut p.base {
        *v = resolve_val(*v, remap);
    }
    for pr in &mut p.projs {
        if let EirProj::Index { index, .. } = pr {
            *index = resolve_val(*index, remap);
        }
    }
}

fn rename_block(bb: BlockId, ctx: &mut RenameCtx<'_>) {
    let bi = bb.0 as usize;
    let mut pushed: Vec<LocalId> = Vec::new();

    for pi in 0..ctx.blocks[bi].phis.len() {
        if let Some(&lid) = ctx.phi_owner.get(&(bb.0, pi)) {
            let pval = ctx.blocks[bi].phis[pi].dst;
            if let Some(nm) = ctx.cfg_fn.local(lid).name {
                ctx.val_names.insert(pval, nm);
            }
            ctx.stacks.get_mut(&lid).unwrap().push(pval);
            pushed.push(lid);
        }
    }

    let old_insts = std::mem::take(&mut ctx.blocks[bi].insts);
    let mut new_insts = Vec::with_capacity(old_insts.len());
    for mut inst in old_insts {
        match &mut inst.kind {
            EirInstKind::GetVar { dst, local } => {
                let top = *ctx.stacks.get(local).and_then(|s| s.last()).unwrap_or(dst);
                ctx.val_remap.insert(*dst, top);
                continue;
            }
            EirInstKind::SetVar { local, val } => {
                let rv = resolve_val(*val, ctx.val_remap);
                if let Some(nm) = ctx.cfg_fn.local(*local).name {
                    ctx.val_names.entry(rv).or_insert(nm);
                }
                ctx.stacks.get_mut(local).unwrap().push(rv);
                pushed.push(*local);
                continue;
            }
            EirInstKind::Const { .. } | EirInstKind::SlotAddr { .. } | EirInstKind::LoadSlot { .. } => {}
            EirInstKind::CopyVal { src, .. } => *src = resolve_val(*src, ctx.val_remap),
            EirInstKind::StoreSlot { val, .. } => *val = resolve_val(*val, ctx.val_remap),
            EirInstKind::LoadPlace { place, .. } | EirInstKind::PlaceAddr { place, .. } => {
                rewrite_place(place, ctx.val_remap);
            }
            EirInstKind::StorePlace { place, val } => {
                rewrite_place(place, ctx.val_remap);
                *val = resolve_val(*val, ctx.val_remap);
            }
            EirInstKind::PoolColumnLoad { pool, index, .. } => {
                if let EirPlaceBase::Val(v) = pool {
                    *v = resolve_val(*v, ctx.val_remap);
                }
                *index = resolve_val(*index, ctx.val_remap);
            }
            EirInstKind::PoolColumnStore { pool, index, val, .. } => {
                if let EirPlaceBase::Val(v) = pool {
                    *v = resolve_val(*v, ctx.val_remap);
                }
                *index = resolve_val(*index, ctx.val_remap);
                *val = resolve_val(*val, ctx.val_remap);
            }
            EirInstKind::Unary { operand, .. }
            | EirInstKind::Cast { operand, .. } => {
                *operand = resolve_val(*operand, ctx.val_remap);
            }
            EirInstKind::Binary { lhs, rhs, .. } => {
                *lhs = resolve_val(*lhs, ctx.val_remap);
                *rhs = resolve_val(*rhs, ctx.val_remap);
            }
            EirInstKind::Slice { base, lo, hi, .. } => {
                *base = resolve_val(*base, ctx.val_remap);
                if let Some(l) = lo {
                    *l = resolve_val(*l, ctx.val_remap);
                }
                if let Some(h) = hi {
                    *h = resolve_val(*h, ctx.val_remap);
                }
            }
            EirInstKind::Aggregate { elems, .. } => {
                for e in elems {
                    *e = resolve_val(*e, ctx.val_remap);
                }
            }
            EirInstKind::UnwrapOptional { src, .. }
            | EirInstKind::UnwrapErrorOk { src, .. }
            | EirInstKind::UnwrapErrorErr { src, .. }
            | EirInstKind::IsNonNull { src, .. }
            | EirInstKind::IsError { src, .. }
            | EirInstKind::MatchTag { src, .. }
            | EirInstKind::FieldExtract { src, .. }
            | EirInstKind::MakeDyn { ptr: src, .. } => {
                *src = resolve_val(*src, ctx.val_remap);
            }
            EirInstKind::MakeClosure { env_ptr, .. } => {
                if let Some(ev) = env_ptr {
                    *ev = resolve_val(*ev, ctx.val_remap);
                }
            }
            EirInstKind::Call { callee, args, .. } => {
                if let EirCallee::FnPtr(v) = callee {
                    *v = resolve_val(*v, ctx.val_remap);
                }
                for a in args {
                    *a = resolve_val(*a, ctx.val_remap);
                }
            }
            EirInstKind::BuiltinCall { args, .. } => {
                for a in args {
                    *a = resolve_val(*a, ctx.val_remap);
                }
            }
            EirInstKind::InlineAsm { inputs, .. } => {
                for (_, a) in inputs {
                    *a = resolve_val(*a, ctx.val_remap);
                }
            }
        }
        new_insts.push(inst);
    }
    ctx.blocks[bi].insts = new_insts;

    match &mut ctx.blocks[bi].term {
        EirTerm::Branch { cond, .. } => *cond = resolve_val(*cond, ctx.val_remap),
        EirTerm::SwitchInt { discr, .. } => *discr = resolve_val(*discr, ctx.val_remap),
        EirTerm::Return(Some(v)) => *v = resolve_val(*v, ctx.val_remap),
        _ => {}
    }

    let succs = term_succs(&ctx.blocks[bi].term);
    let mut seen_succ = BTreeSet::new();
    for s in succs {
        if !seen_succ.insert(s) {
            continue;
        }
        for pi in 0..ctx.blocks[s.0 as usize].phis.len() {
            if let Some(&lid) = ctx.phi_owner.get(&(s.0, pi)) {
                let v = *ctx.stacks.get(&lid).and_then(|st| st.last()).unwrap();
                ctx.blocks[s.0 as usize].phis[pi].incoming.push((bb, v));
            }
        }
    }

    let children = ctx.dom_children[bi].clone();
    for ch in children {
        rename_block(ch, ctx);
    }

    for lid in pushed.into_iter().rev() {
        ctx.stacks.get_mut(&lid).unwrap().pop();
    }
}

fn intersect_dense(idom: &[Option<BlockId>], mut b1: BlockId, mut b2: BlockId) -> BlockId {
    while b1 != b2 {
        while b1.0 > b2.0 {
            b1 = idom[b1.0 as usize].unwrap_or(b1);
        }
        while b2.0 > b1.0 {
            b2 = idom[b2.0 as usize].unwrap_or(b2);
        }
    }
    b1
}

fn term_succs(term: &EirTerm) -> Vec<BlockId> {
    match term {
        EirTerm::Goto(t) => vec![*t],
        EirTerm::Branch { then_bb, else_bb, .. } => vec![*then_bb, *else_bb],
        EirTerm::SwitchInt { targets, otherwise, .. } => {
            let mut v = Vec::with_capacity(targets.len() + 1);
            for &(_, b) in targets {
                v.push(b);
            }
            v.push(*otherwise);
            v
        }
        EirTerm::Return(_) | EirTerm::Unreachable { .. } => Vec::new(),
    }
}

/// Verify structural SSA invariants of `EirProgram`:
/// single SSA definition, phi predecessor coverage, dominance of every use, and valid branch targets.
pub fn verify_eir(prog: &EirProgram) -> Result<(), String> {
    for f in &prog.functions {
        verify_eir_fn(f, None)?;
    }
    Ok(())
}

/// Full EIR-Lint pass: checks SSA dominance, value/slot bounds, operand/result type invariants,
/// absence of `Ty::Error`/`GenericParam`/`ComptimeInt` leaks, and slot definite-initialization.
pub fn lint_eir(prog: &EirProgram, env: &TypeEnv) -> Result<(), String> {
    for f in &prog.functions {
        verify_eir_fn(f, Some(env))?;
    }
    Ok(())
}

fn dominates_dense(idom: &[Option<BlockId>], dom: BlockId, mut cur: BlockId) -> bool {
    if dom == cur {
        return true;
    }
    while cur.0 > 0 {
        let Some(p) = idom[cur.0 as usize] else {
            return false;
        };
        if p == dom {
            return true;
        }
        if p == cur {
            break;
        }
        cur = p;
    }
    dom == cur
}

fn verify_eir_fn(f: &EirFunction, env: Option<&TypeEnv>) -> Result<(), String> {
    let n = f.blocks.len();
    if n == 0 {
        return Err(format!("EIR-Lint: fn {:?} has zero basic blocks", f.key));
    }
    if (f.entry.0 as usize) >= n {
        return Err(format!("EIR-Lint: fn {:?} entry block {:?} out of range", f.key, f.entry));
    }
    let nvals = f.value_tys.len();
    let nslots = f.slots.len();

    let check_val_bounds = |v: EirValue| -> Result<(), String> {
        if (v.0 as usize) >= nvals {
            return Err(format!(
                "EIR-Lint: value {:?} out of range (value_tys len = {}) in {:?}",
                v, nvals, f.key
            ));
        }
        Ok(())
    };
    let check_slot_bounds = |s: EirSlot| -> Result<(), String> {
        if (s.0 as usize) >= nslots {
            return Err(format!(
                "EIR-Lint: slot {:?} out of range (slots len = {}) in {:?}",
                s, nslots, f.key
            ));
        }
        Ok(())
    };

    for (vi, &vty) in f.value_tys.iter().enumerate() {
        if vty == TyId::ERROR {
            let def_inst = f.blocks.iter().flat_map(|b| b.insts.iter()).find(|i| inst_def(&i.kind) == Some(EirValue(vi as u32)));
            let slot_store = def_inst.and_then(|i| match &i.kind {
                EirInstKind::LoadSlot { slot, .. } => f
                    .blocks
                    .iter()
                    .flat_map(|b| b.insts.iter())
                    .find(|st| matches!(&st.kind, EirInstKind::StoreSlot { slot: s2, .. } if s2 == slot))
                    .map(|st| (st.span, &st.kind)),
                _ => None,
            });
            return Err(format!(
                "EIR-Lint: value v{vi} has TyId::ERROR in {:?} (def: {:?}, store: {:?})",
                f.key, def_inst.map(|i| (i.span, &i.kind)), slot_store
            ));
        }
    }

    // Track single definition site (block, inst_idx_or_0_for_param_or_phi).
    let mut def_site: BTreeMap<EirValue, (BlockId, usize)> = BTreeMap::new();
    let mut reg_def = |v: EirValue, bb: BlockId, pos: usize| -> Result<(), String> {
        check_val_bounds(v)?;
        if let Some(prev) = def_site.insert(v, (bb, pos)) {
            return Err(format!(
                "EIR-Lint: SSA value {:?} defined twice in {:?}: {:?} and {:?}",
                v, f.key, prev, (bb, pos)
            ));
        }
        Ok(())
    };

    for &pv in &f.params {
        reg_def(pv, f.entry, 0)?;
    }

    // Track slots whose address is taken or initialized directly.
    let mut slot_written: Vec<bool> = vec![false; nslots];

    for (bi, b) in f.blocks.iter().enumerate() {
        let bid = BlockId(bi as u32);
        for succ in term_succs(&b.term) {
            if (succ.0 as usize) >= n {
                return Err(format!(
                    "EIR-Lint: invalid terminator branch target {:?} in {:?}",
                    succ, f.key
                ));
            }
        }
        for (pi, phi) in b.phis.iter().enumerate() {
            reg_def(phi.dst, bid, pi)?;
            let mut inc_preds: BTreeSet<BlockId> = BTreeSet::new();
            for &(p, iv) in &phi.incoming {
                if (p.0 as usize) >= n {
                    return Err(format!(
                        "EIR-Lint: invalid phi predecessor {:?} in {:?}",
                        p, f.key
                    ));
                }
                check_val_bounds(iv)?;
                inc_preds.insert(p);
            }
            let expected_preds: BTreeSet<BlockId> = f.preds[bi].iter().copied().collect();
            if inc_preds != expected_preds {
                return Err(format!(
                    "EIR-Lint: phi predecessor mismatch in {:?} block {:?}: expected {:?}, got {:?}",
                    f.key, bid, expected_preds, inc_preds
                ));
            }
        }
        let base_pos = b.phis.len() + 1;
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Some(d) = inst_def(&inst.kind) {
                reg_def(d, bid, base_pos + ii)?;
            }
            match &inst.kind {
                EirInstKind::StoreSlot { slot, .. } => {
                    check_slot_bounds(*slot)?;
                    slot_written[slot.0 as usize] = true;
                }
                EirInstKind::SlotAddr { slot, .. } => {
                    check_slot_bounds(*slot)?;
                    slot_written[slot.0 as usize] = true;
                }
                EirInstKind::LoadSlot { slot, .. } => {
                    check_slot_bounds(*slot)?;
                }
                EirInstKind::StorePlace { place, .. } | EirInstKind::PlaceAddr { place, .. } => {
                    if let EirPlaceBase::Slot(s) = place.base {
                        check_slot_bounds(s)?;
                        slot_written[s.0 as usize] = true;
                    }
                }
                EirInstKind::LoadPlace { place, .. } => {
                    if let EirPlaceBase::Slot(s) = place.base {
                        check_slot_bounds(s)?;
                    }
                }
                EirInstKind::PoolColumnLoad { pool, .. }
                | EirInstKind::PoolColumnStore { pool, .. } => {
                    if let EirPlaceBase::Slot(s) = pool {
                        check_slot_bounds(*s)?;
                        if matches!(inst.kind, EirInstKind::PoolColumnStore { .. }) {
                            slot_written[s.0 as usize] = true;
                        }
                    }
                }
                EirInstKind::BuiltinCall { arg_slots, .. } => {
                    for s in arg_slots.iter().flatten() {
                        check_slot_bounds(*s)?;
                        slot_written[s.0 as usize] = true;
                    }
                }
                _ => {}
            }
        }
    }

    let check_use = |v: EirValue, use_bb: BlockId, use_pos: usize| -> Result<(), String> {
        check_val_bounds(v)?;
        let Some(&(def_bb, def_pos)) = def_site.get(&v) else {
            return Err(format!("EIR-Lint: use of undefined SSA value {:?} in {:?}", v, f.key));
        };
        if !dominates_dense(&f.idom, def_bb, use_bb) {
            return Err(format!(
                "EIR-Lint: SSA value {:?} defined at {:?} does not dominate use at {:?} in {:?}",
                v, def_bb, use_bb, f.key
            ));
        }
        if def_bb == use_bb && def_pos >= use_pos {
            return Err(format!(
                "EIR-Lint: SSA value {:?} used before its definition in {:?} block {:?}",
                v, f.key, use_bb
            ));
        }
        Ok(())
    };

    for (bi, b) in f.blocks.iter().enumerate() {
        let bid = BlockId(bi as u32);
        for phi in &b.phis {
            for &(pred_bb, val) in &phi.incoming {
                let end_pos = f.blocks[pred_bb.0 as usize].phis.len()
                    + f.blocks[pred_bb.0 as usize].insts.len()
                    + 2;
                check_use(val, pred_bb, end_pos)?;
            }
        }
        let base_pos = b.phis.len() + 1;
        for (ii, inst) in b.insts.iter().enumerate() {
            if matches!(
                inst.kind,
                EirInstKind::GetVar { .. } | EirInstKind::SetVar { .. }
            ) {
                return Err(format!("EIR-Lint: unlowered GetVar/SetVar remains in {:?}", f.key));
            }
            for u in inst_uses(&inst.kind) {
                check_use(u, bid, base_pos + ii)?;
            }
            if let EirInstKind::LoadSlot { slot, .. } = &inst.kind
                && !slot_written[slot.0 as usize]
            {
                return Err(format!(
                    "EIR-Lint: LoadSlot from never-initialized slot {:?} in {:?}",
                    slot, f.key
                ));
            }
        }
        let end_pos = base_pos + b.insts.len();
        for u in term_uses(&b.term) {
            check_use(u, bid, end_pos)?;
        }
    }

    // Extended type & invariant lint when TypeEnv is available.
    if let Some(env) = env {
        let check_concrete_ty = |ty: TyId, ctx: &str| -> Result<(), String> {
            match env.get(ty) {
                Ty::Error | Ty::GenericParam { .. } | Ty::ComptimeInt | Ty::ComptimeFloat => {
                    Err(format!(
                        "EIR-Lint: non-concrete type {:?} in {} of {:?}",
                        env.get(ty), ctx, f.key
                    ))
                }
                _ => Ok(()),
            }
        };
        for (si, slot) in f.slots.iter().enumerate() {
            check_concrete_ty(slot.ty, &format!("slot#{si}"))?;
        }
        for (bi, b) in f.blocks.iter().enumerate() {
            let bid = BlockId(bi as u32);
            for phi in &b.phis {
                check_concrete_ty(phi.ty, &format!("phi {:?} in {:?}", phi.dst, bid))?;
                for &(_, iv) in &phi.incoming {
                    let it = f.value_tys[iv.0 as usize];
                    if it != phi.ty && env.get(it) != env.get(phi.ty) {
                        // Allow compatible optional/error/int-sized coercions only when exact layout matches.
                        let ok = matches!(
                            (env.get(phi.ty), env.get(it)),
                            (Ty::Never, _)
                                | (_, Ty::Never)
                                | (Ty::Str, Ty::Borrow { is_mut: false, inner: BorrowInner::Str })
                                | (Ty::Borrow { is_mut: false, inner: BorrowInner::Str }, Ty::Str)
                        );
                        if !ok {
                            return Err(format!(
                                "EIR-Lint: phi {:?} type mismatch in {:?}: phi is {:?}, incoming {:?} is {:?}",
                                phi.dst, bid, env.get(phi.ty), iv, env.get(it)
                            ));
                        }
                    }
                }
            }
            for inst in &b.insts {
                match &inst.kind {
                    EirInstKind::Const { ty, konst, .. } => {
                        if !matches!(konst, CfgConst::Void) {
                            check_concrete_ty(*ty, &format!("const in {:?}", bid))?;
                        }
                    }
                    EirInstKind::IsNonNull { src_ty, .. }
                    | EirInstKind::UnwrapOptional { src_ty, .. } => {
                        if !matches!(env.get(*src_ty), Ty::Optional(_)) {
                            return Err(format!(
                                "EIR-Lint: Optional op on non-optional {:?} in {:?} {:?}",
                                env.get(*src_ty), f.key, bid
                            ));
                        }
                    }
                    EirInstKind::IsError { src_ty, .. }
                    | EirInstKind::UnwrapErrorOk { src_ty, .. }
                    | EirInstKind::UnwrapErrorErr { src_ty, .. }
                        if !matches!(env.get(*src_ty), Ty::ErrorUnion { .. }) =>
                    {
                        return Err(format!(
                            "EIR-Lint: ErrorUnion op on non-error-union {:?} in {:?} {:?}",
                            env.get(*src_ty), f.key, bid
                        ));
                    }
                    _ => {}
                }
            }
            match &b.term {
                EirTerm::Branch { cond, .. } => {
                    let cty = f.value_tys[cond.0 as usize];
                    if cty != TyId::BOOL {
                        return Err(format!(
                            "EIR-Lint: Branch condition {:?} has non-bool type {:?} in {:?} {:?}",
                            cond, env.get(cty), f.key, bid
                        ));
                    }
                }
                EirTerm::Return(Some(rv)) => {
                    let rty = f.value_tys[rv.0 as usize];
                    if f.sig.ret != TyId::VOID && f.sig.ret != TyId::NEVER {
                        check_concrete_ty(rty, &format!("return value in {:?}", bid))?;
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn inst_def(k: &EirInstKind) -> Option<EirValue> {
    Some(match k {
        EirInstKind::Const { dst, .. }
        | EirInstKind::CopyVal { dst, .. }
        | EirInstKind::LoadSlot { dst, .. }
        | EirInstKind::SlotAddr { dst, .. }
        | EirInstKind::LoadPlace { dst, .. }
        | EirInstKind::PoolColumnLoad { dst, .. }
        | EirInstKind::PlaceAddr { dst, .. }
        | EirInstKind::Unary { dst, .. }
        | EirInstKind::Binary { dst, .. }
        | EirInstKind::Cast { dst, .. }
        | EirInstKind::Slice { dst, .. }
        | EirInstKind::Aggregate { dst, .. }
        | EirInstKind::UnwrapOptional { dst, .. }
        | EirInstKind::UnwrapErrorOk { dst, .. }
        | EirInstKind::UnwrapErrorErr { dst, .. }
        | EirInstKind::IsNonNull { dst, .. }
        | EirInstKind::IsError { dst, .. }
        | EirInstKind::MatchTag { dst, .. }
        | EirInstKind::FieldExtract { dst, .. }
        | EirInstKind::MakeDyn { dst, .. }
        | EirInstKind::MakeClosure { dst, .. }
        | EirInstKind::GetVar { dst, .. } => *dst,
        EirInstKind::Call { dst, .. } | EirInstKind::BuiltinCall { dst, .. } => return *dst,
        EirInstKind::StoreSlot { .. }
        | EirInstKind::StorePlace { .. }
        | EirInstKind::PoolColumnStore { .. }
        | EirInstKind::InlineAsm { .. }
        | EirInstKind::SetVar { .. } => {
            return None
        }
    })
}

fn place_uses(p: &EirPlace, out: &mut Vec<EirValue>) {
    if let EirPlaceBase::Val(v) = p.base {
        out.push(v);
    }
    for pr in &p.projs {
        if let EirProj::Index { index, .. } = pr {
            out.push(*index);
        }
    }
}

fn inst_uses(k: &EirInstKind) -> Vec<EirValue> {
    let mut out = Vec::new();
    match k {
        EirInstKind::Const { .. } | EirInstKind::SlotAddr { .. } | EirInstKind::LoadSlot { .. } | EirInstKind::GetVar { .. } => {}
        EirInstKind::CopyVal { src, .. }
        | EirInstKind::StoreSlot { val: src, .. }
        | EirInstKind::Unary { operand: src, .. }
        | EirInstKind::Cast { operand: src, .. }
        | EirInstKind::UnwrapOptional { src, .. }
        | EirInstKind::UnwrapErrorOk { src, .. }
        | EirInstKind::UnwrapErrorErr { src, .. }
        | EirInstKind::IsNonNull { src, .. }
        | EirInstKind::IsError { src, .. }
        | EirInstKind::MatchTag { src, .. }
        | EirInstKind::FieldExtract { src, .. }
        | EirInstKind::MakeDyn { ptr: src, .. }
        | EirInstKind::SetVar { val: src, .. } => out.push(*src),
        EirInstKind::MakeClosure { env_ptr, .. } => {
            if let Some(ev) = env_ptr {
                out.push(*ev);
            }
        }
        EirInstKind::LoadPlace { place, .. } | EirInstKind::PlaceAddr { place, .. } => {
            place_uses(place, &mut out);
        }
        EirInstKind::StorePlace { place, val } => {
            place_uses(place, &mut out);
            out.push(*val);
        }
        EirInstKind::PoolColumnLoad { pool, index, .. } => {
            if let EirPlaceBase::Val(v) = pool {
                out.push(*v);
            }
            out.push(*index);
        }
        EirInstKind::PoolColumnStore { pool, index, val, .. } => {
            if let EirPlaceBase::Val(v) = pool {
                out.push(*v);
            }
            out.push(*index);
            out.push(*val);
        }
        EirInstKind::Binary { lhs, rhs, .. } => {
            out.push(*lhs);
            out.push(*rhs);
        }
        EirInstKind::Slice { base, lo, hi, .. } => {
            out.push(*base);
            if let Some(l) = lo { out.push(*l); }
            if let Some(h) = hi { out.push(*h); }
        }
        EirInstKind::Aggregate { elems, .. } => out.extend(elems),
        EirInstKind::Call { callee, args, .. } => {
            if let EirCallee::FnPtr(v) = callee { out.push(*v); }
            out.extend(args);
        }
        EirInstKind::BuiltinCall { args, .. } => out.extend(args),
        EirInstKind::InlineAsm { inputs, .. } => {
            for (_, v) in inputs {
                out.push(*v);
            }
        }
    }
    out
}

fn term_uses(t: &EirTerm) -> Vec<EirValue> {
    match t {
        EirTerm::Branch { cond, .. } => vec![*cond],
        EirTerm::SwitchInt { discr, .. } => vec![*discr],
        EirTerm::Return(Some(v)) => vec![*v],
        _ => Vec::new(),
    }
}

impl EirProgram {
    pub fn dump_canonical(&self, env: &TypeEnv, interner: &Interner) -> String {
        let mut entries: Vec<(String, String)> = Vec::with_capacity(self.functions.len());

        for f in &self.functions {
            let kind_str = match f.key {
                CfgFnKey::TopDef(_) => {
                    if f.is_sys {
                        "sys"
                    } else {
                        "fn"
                    }
                }
                CfgFnKey::ImplMethod { .. } => "method",
                CfgFnKey::GenericInst(_) => "inst",
                CfgFnKey::Closure(_) => "closure",
            };
            let name = interner.get(f.name);
            let gstr = match &f.base_generic {
                Some((_, args)) if !args.is_empty() => {
                    let parts: Vec<String> = args
                        .iter()
                        .map(|a| match a {
                            GenericVal::Ty(t) => env.format_ty(*t, interner),
                            GenericVal::Int(n) => n.to_string(),
                            GenericVal::Variant(s) => interner.get(*s).to_owned(),
                        })
                        .collect();
                    format!("[{}]", parts.join(", "))
                }
                _ => String::new(),
            };
            let mut params_str = String::new();
            for (i, &pv) in f.params.iter().enumerate() {
                if i > 0 {
                    params_str.push_str(", ");
                }
                let pty = f
                    .value_tys
                    .get(pv.0 as usize)
                    .copied()
                    .unwrap_or(TyId::ERROR);
                let _ = write!(params_str, "%v{}: {}", pv.0, env.format_ty(pty, interner));
            }
            let ret_str = env.format_ty(f.sig.ret, interner);

            let mut buf = String::new();
            let _ = writeln!(buf, "eir {kind_str} {name}{gstr}({params_str}) -> {ret_str}");

            for s in &f.slots {
                let sty = env.format_ty(s.ty, interner);
                if let Some(sn) = s.name {
                    let _ = writeln!(buf, "  slot s{}: {} ({})", s.id.0, sty, interner.get(sn));
                } else {
                    let _ = writeln!(buf, "  slot s{}: {}", s.id.0, sty);
                }
            }

            for b in &f.blocks {
                let _ = writeln!(buf, "  bb{}:", b.id.0);
                for phi in &b.phis {
                    let pty = env.format_ty(phi.ty, interner);
                    let _ = write!(buf, "    phi %v{}: {} = [", phi.dst.0, pty);
                    for (i, &(pred, val)) in phi.incoming.iter().enumerate() {
                        if i > 0 {
                            buf.push_str(", ");
                        }
                        let _ = write!(buf, "bb{}: %v{}", pred.0, val.0);
                    }
                    buf.push_str("]\n");
                }
                for inst in &b.insts {
                    buf.push_str("    ");
                    fmt_eir_inst(&inst.kind, f, self, env, interner, &mut buf);
                    buf.push('\n');
                }
                buf.push_str("    term ");
                fmt_eir_term(&b.term, &mut buf);
                buf.push('\n');
            }

            let sort_key = format!("{name}\t{kind_str}\t{gstr}\t{params_str}\t{ret_str}");
            entries.push((sort_key, buf));
        }

        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut out = String::new();
        for (_, text) in entries {
            out.push_str(&text);
        }
        out
    }
}

fn fmt_eir_place_base(base: &EirPlaceBase, out: &mut String) {
    match base {
        EirPlaceBase::Slot(s) => {
            let _ = write!(out, "s{}", s.0);
        }
        EirPlaceBase::Val(v) => {
            let _ = write!(out, "%v{}", v.0);
        }
    }
}

fn fmt_eir_place(place: &EirPlace, interner: &Interner, out: &mut String) {
    fmt_eir_place_base(&place.base, out);
    for pr in &place.projs {
        match pr {
            EirProj::Deref => out.push_str(".*"),
            EirProj::Field { name, field_idx, .. } => {
                let _ = write!(out, ".{}#{}", interner.get(*name), field_idx);
            }
            EirProj::TupleField { index, .. } => {
                let _ = write!(out, ".{index}");
            }
            EirProj::Index { index, .. } => {
                let _ = write!(out, "[%v{}]", index.0);
            }
            EirProj::PoolCol { col, col_idx, .. } => {
                let _ = write!(out, ".col:{}#{}", interner.get(*col), col_idx);
            }
        }
    }
}

fn fmt_eir_val_list(vals: &[EirValue], out: &mut String) {
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "%v{}", v.0);
    }
}

fn fmt_eir_const(k: &CfgConst, interner: &Interner, out: &mut String) {
    match k {
        CfgConst::Void => out.push_str("void"),
        CfgConst::Bool(b) => {
            let _ = write!(out, "bool({b})");
        }
        CfgConst::Char(c) => {
            let _ = write!(out, "char({})", *c as u32);
        }
        CfgConst::Int(n, _) => {
            let _ = write!(out, "int({n})");
        }
        CfgConst::Float(s, _) => {
            let _ = write!(out, "float({s})");
        }
        CfgConst::Str(s) => {
            let _ = write!(out, "str({:?})", s);
        }
        CfgConst::Bytes(bs) => {
            let _ = write!(out, "bytes({})", bs.len());
        }
        CfgConst::Null(_) => out.push_str("null"),
        CfgConst::EnumTag { variant, tag_val, .. } => {
            let _ = write!(out, "enum(.{}={tag_val})", interner.get(*variant));
        }
        CfgConst::ErrorTag { variant, tag_val, .. } => {
            let _ = write!(out, "err(.{}={tag_val})", interner.get(*variant));
        }
        CfgConst::FnRef(d, _) => {
            let _ = write!(out, "fnref(#{})", d.0);
        }
        CfgConst::ConstDef(d, _) => {
            let _ = write!(out, "constdef(#{})", d.0);
        }
    }
}

fn fmt_eir_agg_kind(kind: &AggregateKind, out: &mut String) {
    match kind {
        AggregateKind::Struct(_) => out.push_str("struct"),
        AggregateKind::AnonStruct(_) => out.push_str("anon_struct"),
        AggregateKind::PoolRow { is_slot_pool, .. } => {
            out.push_str(if *is_slot_pool { "slot_pool_row" } else { "arena_pool_row" });
        }
        AggregateKind::Tuple => out.push_str("tuple"),
        AggregateKind::Array(_) => out.push_str("array"),
        AggregateKind::ArrayRepeat { count, .. } => {
            let _ = write!(out, "array_repeat[{count}]");
        }
        AggregateKind::UnionVariant { variant_idx, .. } => {
            let _ = write!(out, "union_variant#{variant_idx}");
        }
        AggregateKind::Bitfield(_) => out.push_str("bitfield"),
        AggregateKind::Vec { len, .. } => {
            let _ = write!(out, "vec[{len}]");
        }
    }
}

fn fmt_eir_inst(
    kind: &EirInstKind,
    f: &EirFunction,
    prog: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    out: &mut String,
) {
    let val_ty = |v: EirValue| {
        let tid = f.value_tys.get(v.0 as usize).copied().unwrap_or(TyId::ERROR);
        env.format_ty(tid, interner)
    };
    match kind {
        EirInstKind::Const { dst, ty, konst } => {
            let _ = write!(out, "%v{}: {} = const ", dst.0, env.format_ty(*ty, interner));
            fmt_eir_const(konst, interner, out);
        }
        EirInstKind::CopyVal { dst, src } => {
            let _ = write!(out, "%v{}: {} = copy %v{}", dst.0, val_ty(*dst), src.0);
        }
        EirInstKind::LoadSlot { dst, slot } => {
            let _ = write!(out, "%v{}: {} = load_slot s{}", dst.0, val_ty(*dst), slot.0);
        }
        EirInstKind::StoreSlot { slot, val } => {
            let _ = write!(out, "store_slot s{} = %v{}", slot.0, val.0);
        }
        EirInstKind::SlotAddr { dst, slot, is_mut, ty } => {
            let m = if *is_mut { "_mut" } else { "" };
            let _ = write!(
                out,
                "%v{}: {} = slot_addr{m} s{}",
                dst.0,
                env.format_ty(*ty, interner),
                slot.0
            );
        }
        EirInstKind::LoadPlace { dst, place } => {
            let _ = write!(out, "%v{}: {} = load_place ", dst.0, val_ty(*dst));
            fmt_eir_place(place, interner, out);
        }
        EirInstKind::StorePlace { place, val } => {
            out.push_str("store_place ");
            fmt_eir_place(place, interner, out);
            let _ = write!(out, " = %v{}", val.0);
        }
        EirInstKind::PoolColumnLoad { dst, pool, col, index, elem_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = pool_col_load ",
                dst.0,
                env.format_ty(*elem_ty, interner)
            );
            fmt_eir_place_base(pool, out);
            let _ = write!(out, ".{}[%v{}]", interner.get(*col), index.0);
        }
        EirInstKind::PoolColumnStore { pool, col, index, val, elem_ty, .. } => {
            out.push_str("pool_col_store ");
            fmt_eir_place_base(pool, out);
            let _ = write!(
                out,
                ".{}[%v{}] = %v{}: {}",
                interner.get(*col),
                index.0,
                val.0,
                env.format_ty(*elem_ty, interner)
            );
        }
        EirInstKind::PlaceAddr { dst, place, is_mut, ty } => {
            let m = if *is_mut { "_mut" } else { "" };
            let _ = write!(
                out,
                "%v{}: {} = place_addr{m} ",
                dst.0,
                env.format_ty(*ty, interner)
            );
            fmt_eir_place(place, interner, out);
        }
        EirInstKind::Unary { dst, op, operand, ty } => {
            let _ = write!(
                out,
                "%v{}: {} = unary {:?} %v{}",
                dst.0,
                env.format_ty(*ty, interner),
                op,
                operand.0
            );
        }
        EirInstKind::Binary { dst, op, lhs, rhs, op_ty, dst_ty } => {
            let _ = write!(
                out,
                "%v{}: {} = bin {:?}({}) %v{}, %v{}",
                dst.0,
                env.format_ty(*dst_ty, interner),
                op,
                env.format_ty(*op_ty, interner),
                lhs.0,
                rhs.0
            );
        }
        EirInstKind::Cast { dst, kind, operand, from_ty, to_ty } => {
            let _ = write!(
                out,
                "%v{}: {} = cast {:?}({} -> {}) %v{}",
                dst.0,
                env.format_ty(*to_ty, interner),
                kind,
                env.format_ty(*from_ty, interner),
                env.format_ty(*to_ty, interner),
                operand.0
            );
        }
        EirInstKind::Slice { dst, base, lo, hi, dst_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = slice %v{}[",
                dst.0,
                env.format_ty(*dst_ty, interner),
                base.0
            );
            if let Some(l) = lo {
                let _ = write!(out, "%v{}", l.0);
            }
            out.push_str("..");
            if let Some(h) = hi {
                let _ = write!(out, "%v{}", h.0);
            }
            out.push(']');
        }
        EirInstKind::Aggregate { dst, kind, elems, dst_ty } => {
            let _ = write!(
                out,
                "%v{}: {} = aggregate ",
                dst.0,
                env.format_ty(*dst_ty, interner)
            );
            fmt_eir_agg_kind(kind, out);
            out.push('(');
            fmt_eir_val_list(elems, out);
            out.push(')');
        }
        EirInstKind::UnwrapOptional { dst, src, dst_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = unwrap_opt %v{}",
                dst.0,
                env.format_ty(*dst_ty, interner),
                src.0
            );
        }
        EirInstKind::UnwrapErrorOk { dst, src, dst_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = unwrap_ok %v{}",
                dst.0,
                env.format_ty(*dst_ty, interner),
                src.0
            );
        }
        EirInstKind::UnwrapErrorErr { dst, src, dst_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = unwrap_err %v{}",
                dst.0,
                env.format_ty(*dst_ty, interner),
                src.0
            );
        }
        EirInstKind::IsNonNull { dst, src, .. } => {
            let _ = write!(out, "%v{}: bool = is_non_null %v{}", dst.0, src.0);
        }
        EirInstKind::IsError { dst, src, .. } => {
            let _ = write!(out, "%v{}: bool = is_error %v{}", dst.0, src.0);
        }
        EirInstKind::MatchTag { dst, src, .. } => {
            let _ = write!(out, "%v{}: {} = match_tag %v{}", dst.0, val_ty(*dst), src.0);
        }
        EirInstKind::FieldExtract { dst, src, field_idx, dst_ty, .. } => {
            let _ = write!(
                out,
                "%v{}: {} = field_extract %v{}.{}",
                dst.0,
                env.format_ty(*dst_ty, interner),
                src.0,
                field_idx
            );
        }
        EirInstKind::MakeDyn { dst, ptr, concrete_ty, is_mut, dst_ty, .. } => {
            let m = if *is_mut { "_mut" } else { "" };
            let _ = write!(
                out,
                "%v{}: {} = make_dyn{m} %v{} ({})",
                dst.0,
                env.format_ty(*dst_ty, interner),
                ptr.0,
                env.format_ty(*concrete_ty, interner)
            );
        }
        EirInstKind::MakeClosure { dst, closure_id, env_ptr, fn_ty } => {
            let _ = write!(
                out,
                "%v{}: {} = make_closure cl{}(",
                dst.0,
                env.format_ty(*fn_ty, interner),
                closure_id
            );
            if let Some(ev) = env_ptr {
                let _ = write!(out, "%v{}", ev.0);
            }
            out.push(')');
        }
        EirInstKind::InlineAsm { template, inputs } => {
            let _ = write!(out, "inline_asm {:?} (", template);
            for (i, (c, v)) in inputs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{c}:%v{}", v.0);
            }
            out.push(')');
        }
        EirInstKind::Call { dst, callee, args, ret_ty } => {
            let rty = env.format_ty(*ret_ty, interner);
            if let Some(d) = dst {
                let _ = write!(out, "%v{}: {} = ", d.0, rty);
            }
            out.push_str("call ");
            match callee {
                EirCallee::DirectFn(d) => {
                    if let Some(&idx) = prog.by_key.get(&CfgFnKey::TopDef(*d)) {
                        let _ = write!(out, "fn:{}", interner.get(prog.functions[idx].name));
                    } else {
                        let _ = write!(out, "def#{}", d.0);
                    }
                }
                EirCallee::Method { impl_idx, sub_idx } => {
                    if let Some(&idx) = prog.by_key.get(&CfgFnKey::ImplMethod {
                        impl_idx: *impl_idx,
                        sub_idx: *sub_idx,
                    }) {
                        let _ = write!(out, "method:{}", interner.get(prog.functions[idx].name));
                    } else {
                        let _ = write!(out, "method#{impl_idx}.{sub_idx}");
                    }
                }
                EirCallee::GenericInst { inst_id } => {
                    if let Some(&idx) = prog.by_key.get(&CfgFnKey::GenericInst(*inst_id)) {
                        let _ = write!(out, "inst:{}", interner.get(prog.functions[idx].name));
                    } else {
                        let _ = write!(out, "inst#{inst_id}");
                    }
                }
                EirCallee::DynCall { iface_def, method_idx } => {
                    let _ = write!(out, "dyn#{}.{method_idx}", iface_def.0);
                }
                EirCallee::ExternFn { name, .. } => {
                    let _ = write!(out, "extern:{}", interner.get(*name));
                }
                EirCallee::FnPtr(v) => {
                    let _ = write!(out, "fnptr:%v{}", v.0);
                }
            }
            out.push('(');
            fmt_eir_val_list(args, out);
            let _ = write!(out, ") -> {rty}");
        }
        EirInstKind::BuiltinCall { dst, op, args, ret_ty, .. } => {
            let rty = env.format_ty(*ret_ty, interner);
            if let Some(d) = dst {
                let _ = write!(out, "%v{}: {} = ", d.0, rty);
            }
            let _ = write!(out, "builtin {op:?}(");
            fmt_eir_val_list(args, out);
            let _ = write!(out, ") -> {rty}");
        }
        EirInstKind::GetVar { dst, local } => {
            let _ = write!(out, "%v{}: {} = get_var l{}", dst.0, val_ty(*dst), local.0);
        }
        EirInstKind::SetVar { local, val } => {
            let _ = write!(out, "set_var l{} = %v{}", local.0, val.0);
        }
    }
}

fn fmt_eir_term(term: &EirTerm, out: &mut String) {
    match term {
        EirTerm::Goto(b) => {
            let _ = write!(out, "goto bb{}", b.0);
        }
        EirTerm::Branch { cond, then_bb, else_bb, .. } => {
            let _ = write!(out, "branch %v{} ? bb{} : bb{}", cond.0, then_bb.0, else_bb.0);
        }
        EirTerm::SwitchInt { discr, targets, otherwise } => {
            let _ = write!(out, "switch %v{} [", discr.0);
            for (i, &(val, bb)) in targets.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{val}->bb{}", bb.0);
            }
            let _ = write!(out, "] else bb{}", otherwise.0);
        }
        EirTerm::Return(Some(v)) => {
            let _ = write!(out, "return %v{}", v.0);
        }
        EirTerm::Return(None) => out.push_str("return void"),
        EirTerm::Unreachable { explicit: true, .. } => out.push_str("unreachable(explicit)"),
        EirTerm::Unreachable { explicit: false, .. } => out.push_str("unreachable"),
    }
}
