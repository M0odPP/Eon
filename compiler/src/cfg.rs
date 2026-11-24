#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{
    self, AssignOp, BinaryOp, Block, CatchBody, CompilationUnit, DeclKind, DeferBody, ElseBranch,
    Expr, ExprKind, GenericArg, IfCond, IfExpr, JumpAction, MatchArm, MatchArmBody, Pat, PatKind,
    RangeKind, Stmt, StmtKind, UnaryOp,
};
use crate::check::TypeTable;
use crate::resolve::{
    BuiltinKind, DefId, ModuleId, ResolvedProgram, TopDefKind, TopScopeTarget,
};
use crate::span::{Interner, Span, Symbol};
use crate::types::{
    parse_int_literal, ErrorSource, FnSig, GenericVal, MethodTarget, Ty, TyId, TypeEnv,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CfgFnKey {
    TopDef(DefId),
    ImplMethod { impl_idx: usize, sub_idx: usize },
    GenericInst(usize),
    Closure(usize),
}

#[derive(Clone, Debug)]
pub struct ParForRegion {
    pub header_bb: BlockId,
    pub body_bb: BlockId,
    pub step_bb: BlockId,
    pub exit_bb: BlockId,
    pub idx_local: LocalId,
    pub end_local: LocalId,
    pub chunk_size: u64,
}

#[derive(Clone, Debug)]
pub struct UnsupportedItem {
    pub kind: &'static str,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ScheduleWave {
    pub systems: Vec<DefId>,
}

#[derive(Clone, Debug, Default)]
pub struct ScheduleDag {
    pub waves: Vec<ScheduleWave>,
}

#[derive(Clone, Debug, Default)]
pub struct CfgProgram {
    pub functions: Vec<CfgFunction>,
    pub by_key: BTreeMap<CfgFnKey, usize>,
    pub generic_inst_map: BTreeMap<(CfgFnKey, Vec<GenericVal>), usize>,
    pub dyn_impl_pairs: Vec<(DefId, TyId)>,
    pub schedules: Vec<ScheduleDag>,
}

#[derive(Clone, Debug)]
pub struct CfgFunction {
    pub key: CfgFnKey,
    pub module: ModuleId,
    pub name: Symbol,
    pub body_span: Span,
    pub sig: FnSig,
    pub params: Vec<LocalId>,
    pub locals: Vec<CfgLocal>,
    pub blocks: Vec<CfgBlock>,
    pub entry: BlockId,
    pub unsupported: Vec<UnsupportedItem>,
    pub is_sys: bool,
    pub where_clauses: Vec<Expr>,
    pub ensures_clauses: Vec<Expr>,
    pub loop_invariants: BTreeMap<BlockId, Vec<Expr>>,
    pub base_generic: Option<(CfgFnKey, Vec<GenericVal>)>,
    pub par_for_regions: Vec<ParForRegion>,
}

impl CfgFunction {
    pub fn local(&self, id: LocalId) -> &CfgLocal {
        &self.locals[id.0 as usize]
    }

    pub fn block(&self, id: BlockId) -> &CfgBlock {
        &self.blocks[id.0 as usize]
    }
}

#[derive(Clone, Debug)]
pub struct CfgLocal {
    pub id: LocalId,
    pub name: Option<Symbol>,
    pub span: Span,
    pub ty: TyId,
    pub is_mut: bool,
    pub is_param: bool,
    pub is_self: bool,
    pub is_temp: bool,
    pub is_linear: bool,
    pub is_borrowing_val: bool,
    pub scope_depth: u32,
    pub param_index: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct CfgBlock {
    pub id: BlockId,
    pub stmts: Vec<CfgStmt>,
    pub term: CfgTerm,
    pub is_cleanup: bool,
    pub is_errdefer: bool,
    pub error_site_span: Option<Span>,
    pub is_loop_header: bool,
    pub loop_entry_depth: Option<u32>,
    pub join_span: Option<Span>,
}

impl CfgBlock {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            stmts: Vec::new(),
            term: CfgTerm::Unreachable { span: Span::DUMMY, explicit: false },
            is_cleanup: false,
            is_errdefer: false,
            error_site_span: None,
            is_loop_header: false,
            loop_entry_depth: None,
            join_span: None,
        }
    }

    pub fn successors(&self) -> Vec<BlockId> {
        match &self.term {
            CfgTerm::Goto(t) => vec![*t],
            CfgTerm::Branch { then_bb, else_bb, .. } => vec![*then_bb, *else_bb],
            CfgTerm::SwitchInt { targets, otherwise, .. } => {
                let mut v = Vec::with_capacity(targets.len() + 1);
                for &(_, b) in targets {
                    v.push(b);
                }
                v.push(*otherwise);
                v
            }
            CfgTerm::Return { .. } | CfgTerm::Unreachable { .. } => Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CfgPlace {
    pub local: LocalId,
    pub projs: Vec<CfgProj>,
    pub ty: TyId,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum CfgProj {
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
        index: CfgOperand,
        const_idx: Option<i128>,
        key_sym: Option<Symbol>,
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
pub enum CfgOperand {
    Copy(CfgPlace),
    Move(CfgPlace),
    Const(CfgConst),
}

impl CfgOperand {
    pub fn ty(&self, f: &CfgFunction, env: &TypeEnv) -> TyId {
        match self {
            CfgOperand::Copy(p) | CfgOperand::Move(p) => p.ty,
            CfgOperand::Const(c) => c.ty(env, f),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CfgConst {
    Void,
    Bool(bool),
    Char(char),
    Int(i128, TyId),
    Float(String, TyId),
    Str(String),
    Bytes(Vec<u8>),
    Null(TyId),
    EnumTag { ty: TyId, variant: Symbol, tag_val: i128 },
    ErrorTag { ty: TyId, variant: Symbol, tag_val: u32 },
    FnRef(DefId, TyId),
    ConstDef(DefId, TyId),
}

impl CfgConst {
    pub fn ty(&self, env: &TypeEnv, _f: &CfgFunction) -> TyId {
        match self {
            CfgConst::Void => TyId::VOID,
            CfgConst::Bool(_) => TyId::BOOL,
            CfgConst::Char(_) => TyId::CHAR,
            CfgConst::Int(_, ty) | CfgConst::Float(_, ty) | CfgConst::Null(ty) | CfgConst::ConstDef(_, ty) => *ty,
            CfgConst::Str(_) => TyId::STR,
            CfgConst::Bytes(_) => TyId::STR,
            CfgConst::EnumTag { ty, .. } => *ty,
            CfgConst::ErrorTag { ty, .. } => {
                if matches!(env.get(*ty), Ty::ErrorSet(_)) {
                    *ty
                } else {
                    TyId::U32
                }
            }
            CfgConst::FnRef(_, ty) => *ty,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CastKind {
    Widen,
    Trunc,
    Clamp,
    ToFloat,
    TruncToInt,
    Bitcast,
    AsU64,
    WrapOptional,
    WrapOk,
    WrapErr,
    MutToConst,
    ArrayBorrowToSlice,
}

#[derive(Clone, Debug)]
pub enum AggregateKind {
    Struct(DefId),
    AnonStruct(Vec<Symbol>),
    PoolRow { def: DefId, is_slot_pool: bool },
    Tuple,
    Array(TyId),
    ArrayRepeat { elem: TyId, count: u64 },
    UnionVariant { def: DefId, variant_idx: usize },
    Bitfield(DefId),
    Vec { elem: TyId, len: u8 },
}

#[derive(Clone, Debug)]
pub enum CfgRvalue {
    Use(CfgOperand),
    Borrow {
        is_mut: bool,
        place: CfgPlace,
    },
    Unary {
        op: UnaryOp,
        operand: CfgOperand,
    },
    Binary {
        op: BinaryOp,
        lhs: CfgOperand,
        rhs: CfgOperand,
    },
    Slice {
        base: CfgOperand,
        lo: Option<CfgOperand>,
        hi: Option<CfgOperand>,
    },
    Cast {
        kind: CastKind,
        operand: CfgOperand,
        to_ty: TyId,
    },
    Aggregate {
        kind: AggregateKind,
        elems: Vec<CfgOperand>,
    },
    UnwrapOptional(CfgOperand),
    UnwrapErrorOk(CfgOperand),
    UnwrapErrorErr(CfgOperand),
    IsNonNull(CfgOperand),
    IsError(CfgOperand),
    MatchTag(CfgOperand),
    FieldExtract {
        operand: CfgOperand,
        field_idx: usize,
        ty: TyId,
    },
    MakeDyn {
        ptr: CfgOperand,
        iface_def: DefId,
        concrete_ty: TyId,
        is_mut: bool,
    },
    MakeClosure {
        closure_id: usize,
        env_ptr: Option<CfgOperand>,
        fn_ty: TyId,
    },
    InlineAsm {
        template: String,
        inputs: Vec<(String, CfgOperand)>,
    },
}

#[derive(Clone, Debug)]
pub enum CfgBuiltinOp {
    ArenaInit,
    ArenaAlloc,
    ArenaReset,
    ArenaDestroy,
    ArenaUsedBytes,
    ArenaCapacity,
    AllocCapLiveAllocs,
    AllocCapUsedBytes,
    AllocCapPeakBytes,
    AllocCapSubBudget,
    ClockNowNs,
    ArgsLen,
    ArgsGet,
    FsOpenDir,
    FsReadFile,
    FsWriteFile,
    FsWriteExecutable,
    FsIsDir,
    FsMkdirAll,
    FsListDir,
    FsSpawnWait,
    DirSubDir,
    DirReadFile,
    DirWriteFile,
    DirClose,
    NetRestrictHost,
    IoNopRoundtrip,
    IoRunFibers,
    BuildAddModule,
    BuildAddExe,
    BuildAddSharedLib,
    BuildSetOutDir,
    BuildSetSanitize,
    BuildSetAuditTcb,
    BuildSetOptLevel,
    BuildLinkLib,
    BuildAllowHal,
    BuildOutPath,
    ScheduleRun,
    MathSqrt(TyId),
    MathFma(TyId),
    MathLzcnt(TyId),
    MathTzcnt(TyId),
    MathPopcnt(TyId),
    MathBswap(TyId),
    ArenaPoolInit(TyId),
    ArenaPoolPush(TyId),
    ArenaPoolIdAt(TyId),
    ArenaPoolReset(TyId),
    ArenaPoolLen,
    SlotPoolInit(TyId),
    SlotPoolInsert(TyId),
    SlotPoolTryInsert(TyId),
    SlotPoolRemove(TyId),
    SlotPoolGet { elem_ty: TyId, is_mut: bool },
    SlotPoolAt(TyId),
    SlotPoolLen,
    SlotPoolCap,
    SlotPoolIsAlive(TyId),
    SlotPoolIdAt(TyId),
    SlotPoolDestroy(TyId),
    BufInit(TyId),
    BufPush(TyId),
    BufPushSlice(TyId),
    BufPop(TyId),
    BufLen,
    BufCap,
    BufClear,
    BufAsSlice { elem_ty: TyId, is_mut: bool },
    BufAt { elem_ty: TyId, is_mut: bool },
    BufGet { elem_ty: TyId, is_mut: bool },
    BufSub { elem_ty: TyId, is_mut: bool },
    BufDestroy(TyId),
    AtomicNew(TyId),
    AtomicLoad(TyId),
    AtomicStore(TyId),
    AtomicFetchAdd(TyId),
    AtomicFetchSub(TyId),
    AtomicSwap(TyId),
    AtomicCas(TyId),
    MmioRead(TyId),
    MmioWrite(TyId),
    RawPtrFromUsize(TyId),
    RawPtrAsUsize,
    ScheduleBuild,
    ParReduce { elem_ty: TyId, op: Symbol },
    SliceLen,
    ArrayLen,
    StrLen,
    StrAsBytes,
    BytesAsStr,
    SliceAt { elem_ty: TyId, is_mut: bool },
    ArrayAt { elem_ty: TyId, is_mut: bool },
    SliceGet { elem_ty: TyId, is_mut: bool },
    ArrayGet { elem_ty: TyId, is_mut: bool },
    SliceSub { elem_ty: TyId, is_mut: bool },
    ArraySub { elem_ty: TyId, is_mut: bool },
    GpuGlobalId,
    GpuLocalId,
    GpuWorkgroupId,
    GpuWorkgroupBarrier,
    VecSplat { vec_ty: TyId },
    VecLen(u8),
    VecCmp { op: BinaryOp, vec_ty: TyId },
    VecMaskAny { len: u8 },
    VecMaskAll { len: u8 },
    VecBitmask { len: u8 },
    VecSelect { vec_ty: TyId },
    VecMin { vec_ty: TyId },
    VecMax { vec_ty: TyId },
    VecFma { vec_ty: TyId },
    VecReduceAdd { vec_ty: TyId },
    VecReduceMin { vec_ty: TyId },
    VecReduceMax { vec_ty: TyId },
    VecDot { vec_ty: TyId },
    VecShuffle { src_ty: TyId, idx_ty: TyId, two_src: bool },
}

#[derive(Clone, Debug)]
pub enum CfgCallee {
    DirectFn(DefId),
    Method { impl_idx: usize, sub_idx: usize, is_view: bool },
    GenericInst { inst_id: usize, is_view: bool },
    DynCall { iface_def: DefId, method_idx: usize },
    ExternFn { def: DefId, name: Symbol },
    FnPtr(CfgOperand),
    Builtin(CfgBuiltinOp),
}

#[derive(Clone, Debug)]
pub struct CfgStmt {
    pub span: Span,
    pub kind: CfgStmtKind,
}

#[derive(Clone, Debug)]
pub enum CfgStmtKind {
    StorageLive(LocalId),
    StorageDead(LocalId),
    Assign {
        dst: CfgPlace,
        rvalue: CfgRvalue,
    },
    Call {
        dst: Option<CfgPlace>,
        callee: CfgCallee,
        args: Vec<CfgOperand>,
        arg_spans: Vec<Span>,
        recv_place: Option<CfgPlace>,
        recv_span: Option<Span>,
        is_mut_recv: bool,
        consumes_recv: bool,
    },
    DiscardLinearError {
        local: Option<LocalId>,
        span: Span,
    },
    ConsumeDestructured(LocalId),
    PoolResetEpoch {
        pool_local: LocalId,
    },
}

#[derive(Clone, Debug)]
pub enum CfgTerm {
    Goto(BlockId),
    Branch {
        cond: CfgOperand,
        then_bb: BlockId,
        else_bb: BlockId,
        span: Span,
        is_user_if: bool,
    },
    SwitchInt {
        discr: CfgOperand,
        targets: Vec<(i128, BlockId)>,
        otherwise: BlockId,
        span: Span,
    },
    Return {
        value: Option<CfgOperand>,
        span: Span,
    },
    Unreachable {
        span: Span,
        explicit: bool,
    },
}

/// Lower every user-defined function, system, impl method, and concrete generic instantiation into a typed CFG.
pub fn lower_program(
    units: &[CompilationUnit],
    prog: &ResolvedProgram,
    env: &mut TypeEnv,
    type_table: &TypeTable,
    interner: &Interner,
) -> CfgProgram {
    let mut cfg = CfgProgram {
        dyn_impl_pairs: type_table.all_dyn_impl_pairs(),
        ..Default::default()
    };
    let mut generic_inst_map: BTreeMap<(CfgFnKey, Vec<GenericVal>), usize> = BTreeMap::new();
    let mut pending_insts: Vec<(CfgFnKey, Vec<GenericVal>, usize)> = Vec::new();
    let mut out_closures: Vec<CfgFunction> = Vec::new();
    let mut next_closure_id: usize = 0;
    for (fk, gargs) in type_table.all_generic_insts() {
        let next_id = generic_inst_map.len();
        if let std::collections::btree_map::Entry::Vacant(e) = generic_inst_map.entry((fk, gargs.clone())) {
            e.insert(next_id);
            pending_insts.push((fk, gargs, next_id));
        }
    }

    for i in 0..prog.defs.len() {
        let def_id = DefId(i as u32);
        let def_mod = prog.defs[i].module;
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else {
            continue;
        };
        match prog.defs[i].kind.clone() {
            TopDefKind::Fn(decl_idx) => {
                let Some(sig) = env.fns.get(&def_id).cloned() else {
                    continue;
                };
                if let DeclKind::Fn(f) = &units[ui].decls[decl_idx].kind {
                    let func = lower_fn_body(
                        CfgFnKey::TopDef(def_id),
                        def_mod,
                        sig,
                        &f.body,
                        false,
                        &f.where_clauses,
                        &f.ensures_clauses,
                        Vec::new(),
                        None,
                        &mut generic_inst_map,
                        &mut pending_insts,
                        &mut out_closures,
                        &mut next_closure_id,
                        &mut cfg.schedules,
                        prog,
                        env,
                        type_table,
                        interner,
                    );
                    let idx = cfg.functions.len();
                    cfg.by_key.insert(CfgFnKey::TopDef(def_id), idx);
                    cfg.functions.push(func);
                }
            }
            TopDefKind::Sys(decl_idx) => {
                let Some(sig) = env.fns.get(&def_id).cloned() else {
                    continue;
                };
                if let DeclKind::Sys(s) = &units[ui].decls[decl_idx].kind {
                    let func = lower_fn_body(
                        CfgFnKey::TopDef(def_id),
                        def_mod,
                        sig,
                        &s.body,
                        true,
                        &s.where_clauses,
                        &s.ensures_clauses,
                        Vec::new(),
                        None,
                        &mut generic_inst_map,
                        &mut pending_insts,
                        &mut out_closures,
                        &mut next_closure_id,
                        &mut cfg.schedules,
                        prog,
                        env,
                        type_table,
                        interner,
                    );
                    let idx = cfg.functions.len();
                    cfg.by_key.insert(CfgFnKey::TopDef(def_id), idx);
                    cfg.functions.push(func);
                }
            }
            _ => {}
        }
    }

    let impl_count = env.impls.len();
    for impl_idx in 0..impl_count {
        let im = env.impls[impl_idx].clone();
        let Some(ui) = prog.modules[im.module.0 as usize].unit_idx else {
            continue;
        };
        let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
            continue;
        };
        for &(_, sub_idx, ref sig) in &im.methods {
            if let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind {
                let key = CfgFnKey::ImplMethod { impl_idx, sub_idx };
                let func = lower_fn_body(
                    key,
                    im.module,
                    sig.clone(),
                    &f.body,
                    false,
                    &f.where_clauses,
                    &f.ensures_clauses,
                    Vec::new(),
                    None,
                    &mut generic_inst_map,
                    &mut pending_insts,
                    &mut out_closures,
                    &mut next_closure_id,
                    &mut cfg.schedules,
                    prog,
                    env,
                    type_table,
                    interner,
                );
                let idx = cfg.functions.len();
                cfg.by_key.insert(key, idx);
                cfg.functions.push(func);
            }
        }
    }

    // Drain concrete generic instantiations (including transitive calls inside generic bodies!).
    let mut cursor = 0usize;
    while cursor < pending_insts.len() {
        let (base_key, gargs, inst_id) = pending_insts[cursor].clone();
        cursor += 1;
        let target_info = match base_key {
            CfgFnKey::TopDef(def_id) => {
                let def_mod = prog.defs[def_id.0 as usize].module;
                prog.modules[def_mod.0 as usize].unit_idx.and_then(|ui| {
                    if let TopDefKind::Fn(decl_idx) = prog.defs[def_id.0 as usize].kind
                        && let DeclKind::Fn(f) = &units[ui].decls[decl_idx].kind
                    {
                        env.fns.get(&def_id).cloned().map(|sig| {
                            (def_mod, sig, &f.body, f.where_clauses.as_slice(), f.ensures_clauses.as_slice())
                        })
                    } else {
                        None
                    }
                })
            }
            CfgFnKey::ImplMethod { impl_idx, sub_idx } => {
                env.impls.get(impl_idx).cloned().and_then(|im| {
                    let ui = prog.modules[im.module.0 as usize].unit_idx?;
                    let DeclKind::Impl(impl_decl) = &units[ui].decls[im.impl_decl_idx].kind else {
                        return None;
                    };
                    let DeclKind::Fn(f) = &impl_decl.decls[sub_idx].kind else {
                        return None;
                    };
                    let sig = im.methods.iter().find(|(_, s, _)| *s == sub_idx)?.2.clone();
                    Some((im.module, sig, &f.body, f.where_clauses.as_slice(), f.ensures_clauses.as_slice()))
                })
            }
            CfgFnKey::GenericInst(_) | CfgFnKey::Closure(_) => None,
        };
        let Some((def_mod, mut sig, body, where_clauses, ensures_clauses)) = target_info else {
            continue;
        };
        for p in &mut sig.params {
            p.ty = crate::check::subst_ty(p.ty, &gargs, env);
        }
        sig.ret = crate::check::subst_ty(sig.ret, &gargs, env);
        sig.generics.clear();
        let key = CfgFnKey::GenericInst(inst_id);
        let func = lower_fn_body(
            key,
            def_mod,
            sig,
            body,
            false,
            where_clauses,
            ensures_clauses,
            gargs.clone(),
            Some((base_key, gargs)),
            &mut generic_inst_map,
            &mut pending_insts,
            &mut out_closures,
            &mut next_closure_id,
            &mut cfg.schedules,
            prog,
            env,
            type_table,
            interner,
        );
        let idx = cfg.functions.len();
        cfg.by_key.insert(key, idx);
        cfg.functions.push(func);
    }

    for cf in out_closures {
        let key = cf.key;
        let idx = cfg.functions.len();
        cfg.by_key.insert(key, idx);
        cfg.functions.push(cf);
    }

    cfg.generic_inst_map = generic_inst_map;
    cfg
}

#[allow(clippy::too_many_arguments)]
fn lower_fn_body(
    key: CfgFnKey,
    module: ModuleId,
    sig: FnSig,
    body: &Block,
    is_sys: bool,
    where_clauses: &[Expr],
    ensures_clauses: &[Expr],
    generic_subst: Vec<GenericVal>,
    base_generic: Option<(CfgFnKey, Vec<GenericVal>)>,
    generic_inst_map: &mut BTreeMap<(CfgFnKey, Vec<GenericVal>), usize>,
    pending_insts: &mut Vec<(CfgFnKey, Vec<GenericVal>, usize)>,
    out_closures: &mut Vec<CfgFunction>,
    next_closure_id: &mut usize,
    schedules: &mut Vec<ScheduleDag>,
    prog: &ResolvedProgram,
    env: &mut TypeEnv,
    type_table: &TypeTable,
    interner: &Interner,
) -> CfgFunction {
    let mut b = CfgBuilder::new(
        key,
        module,
        sig.clone(),
        body.span,
        generic_subst,
        base_generic,
        generic_inst_map,
        pending_insts,
        out_closures,
        next_closure_id,
        schedules,
        prog,
        env,
        type_table,
        interner,
    );
    b.func.is_sys = is_sys;
    b.func.where_clauses = where_clauses.to_vec();
    b.func.ensures_clauses = ensures_clauses.to_vec();
    b.bind_params();
    b.lower_body(body);
    b.finish()
}

#[derive(Clone)]
enum ScopeEvent {
    Local(LocalId),
    Defer { body: DeferBody, span: Span },
    Errdefer { err_bind: Option<ast::IdentSpan>, body: DeferBody, span: Span },
}

#[derive(Clone, Default)]
struct ScopeFrame {
    depth: u32,
    vars: BTreeMap<Symbol, LocalId>,
    events: Vec<ScopeEvent>,
}

#[derive(Clone)]
struct LoopTarget {
    label: Option<Symbol>,
    continue_bb: BlockId,
    exit_bb: BlockId,
    break_dst: Option<CfgPlace>,
    break_origin_ty: TyId,
    scope_len_at_entry: usize,
    any_break: bool,
}

struct CfgBuilder<'a> {
    module: ModuleId,
    prog: &'a ResolvedProgram,
    env: &'a mut TypeEnv,
    table: &'a TypeTable,
    interner: &'a Interner,
    generic_subst: Vec<GenericVal>,
    generic_inst_map: &'a mut BTreeMap<(CfgFnKey, Vec<GenericVal>), usize>,
    pending_insts: &'a mut Vec<(CfgFnKey, Vec<GenericVal>, usize)>,
    out_closures: &'a mut Vec<CfgFunction>,
    next_closure_id: &'a mut usize,
    schedules: &'a mut Vec<ScheduleDag>,
    captured_aliases: BTreeMap<Symbol, (LocalId, usize, TyId, TyId)>,
    captured_self: Option<(LocalId, usize, TyId, TyId)>,

    func: CfgFunction,
    cur_bb: Option<BlockId>,
    scopes: Vec<ScopeFrame>,
    loops: Vec<LoopTarget>,
    cleanup_depth: u32,
    in_errdefer: bool,
    active_err_span: Option<Span>,
    pool_row_is_slot: Option<bool>,
    consts: BTreeMap<Symbol, i128>,
}

impl<'a> CfgBuilder<'a> {
    fn new(
        key: CfgFnKey,
        module: ModuleId,
        sig: FnSig,
        body_span: Span,
        generic_subst: Vec<GenericVal>,
        base_generic: Option<(CfgFnKey, Vec<GenericVal>)>,
        generic_inst_map: &'a mut BTreeMap<(CfgFnKey, Vec<GenericVal>), usize>,
        pending_insts: &'a mut Vec<(CfgFnKey, Vec<GenericVal>, usize)>,
        out_closures: &'a mut Vec<CfgFunction>,
        next_closure_id: &'a mut usize,
        schedules: &'a mut Vec<ScheduleDag>,
        prog: &'a ResolvedProgram,
        env: &'a mut TypeEnv,
        table: &'a TypeTable,
        interner: &'a Interner,
    ) -> Self {
        let entry = BlockId(0);
        let name = sig.name;
        let func = CfgFunction {
            key,
            module,
            name,
            body_span,
            sig,
            params: Vec::new(),
            locals: Vec::new(),
            blocks: vec![CfgBlock::new(entry)],
            entry,
            unsupported: Vec::new(),
            is_sys: false,
            where_clauses: Vec::new(),
            ensures_clauses: Vec::new(),
            loop_invariants: BTreeMap::new(),
            base_generic,
            par_for_regions: Vec::new(),
        };
        Self {
            module,
            prog,
            env,
            table,
            interner,
            generic_subst,
            generic_inst_map,
            pending_insts,
            out_closures,
            next_closure_id,
            schedules,
            captured_aliases: BTreeMap::new(),
            captured_self: None,
            func,
            cur_bb: Some(entry),
            scopes: vec![ScopeFrame {
                depth: 0,
                vars: BTreeMap::new(),
                events: Vec::new(),
            }],
            loops: Vec::new(),
            cleanup_depth: 0,
            in_errdefer: false,
            active_err_span: None,
            pool_row_is_slot: None,
            consts: BTreeMap::new(),
        }
    }

    fn subst(&mut self, ty: TyId) -> TyId {
        if self.generic_subst.is_empty() {
            ty
        } else {
            crate::check::subst_ty(ty, &self.generic_subst, self.env)
        }
    }

    fn subst_gargs(&mut self, args: &[GenericVal]) -> Vec<GenericVal> {
        if self.generic_subst.is_empty() {
            return args.to_vec();
        }
        args.iter()
            .map(|a| match a {
                GenericVal::Ty(t) => GenericVal::Ty(crate::check::subst_ty(*t, &self.generic_subst, self.env)),
                other => *other,
            })
            .collect()
    }

    fn intern_generic_inst(&mut self, base_key: CfgFnKey, gargs: Vec<GenericVal>) -> usize {
        if let Some(&id) = self.generic_inst_map.get(&(base_key, gargs.clone())) {
            return id;
        }
        let id = self.generic_inst_map.len();
        self.generic_inst_map.insert((base_key, gargs.clone()), id);
        self.pending_insts.push((base_key, gargs, id));
        id
    }

    fn note_ty_unsupported(&mut self, _ty: TyId, _span: Span) {}

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.func.blocks.len() as u32);
        let mut bb = CfgBlock::new(id);
        if self.cleanup_depth > 0 {
            bb.is_cleanup = true;
            bb.is_errdefer = self.in_errdefer;
            bb.error_site_span = self.active_err_span;
        }
        self.func.blocks.push(bb);
        id
    }

    fn new_local(
        &mut self,
        name: Option<Symbol>,
        span: Span,
        ty: TyId,
        is_mut: bool,
        is_param: bool,
        is_self: bool,
        is_temp: bool,
        param_index: Option<usize>,
    ) -> LocalId {
        let ty = self.subst(ty);
        self.note_ty_unsupported(ty, span);
        let id = LocalId(self.func.locals.len() as u32);
        let is_linear = !matches!(self.env.get(ty), Ty::Borrow { .. }) && self.env.is_linear(ty);
        let is_borrowing_val = self.env.contains_borrowing_value(ty);
        let scope_depth = self.scopes.last().map_or(0, |s| s.depth);
        self.func.locals.push(CfgLocal {
            id,
            name,
            span,
            ty,
            is_mut,
            is_param,
            is_self,
            is_temp,
            is_linear,
            is_borrowing_val,
            scope_depth,
            param_index,
        });
        id
    }

    fn new_temp(&mut self, ty: TyId, span: Span) -> LocalId {
        let id = self.new_local(None, span, ty, true, false, false, true, None);
        self.push_stmt(span, CfgStmtKind::StorageLive(id));
        if let Some(frame) = self.scopes.last_mut() {
            frame.events.push(ScopeEvent::Local(id));
        }
        id
    }

    fn bind_user_local(&mut self, sym: Symbol, span: Span, ty: TyId, is_mut: bool) -> LocalId {
        if self.interner.get(sym) == "_" {
            return self.new_temp(ty, span);
        }
        let id = self.new_local(Some(sym), span, ty, is_mut, false, false, false, None);
        self.push_stmt(span, CfgStmtKind::StorageLive(id));
        if let Some(frame) = self.scopes.last_mut() {
            frame.vars.insert(sym, id);
            frame.events.push(ScopeEvent::Local(id));
        }
        id
    }

    fn bind_params(&mut self) {
        let params = self.func.sig.params.clone();
        for (i, p) in params.iter().enumerate() {
            let is_mut = matches!(p.modifier, Some(ast::ParamModifier::Mut))
                || matches!(p.access, Some(ast::AccessMode::Write));
            let id = self.new_local(
                Some(p.name),
                p.span,
                p.ty,
                is_mut,
                true,
                p.is_self,
                false,
                Some(i),
            );
            self.func.params.push(id);
            if self.interner.get(p.name) != "_"
                && let Some(frame) = self.scopes.last_mut()
            {
                frame.vars.insert(p.name, id);
                frame.events.push(ScopeEvent::Local(id));
            }
        }
    }

    fn lookup_local(&self, sym: Symbol) -> Option<LocalId> {
        for frame in self.scopes.iter().rev() {
            if let Some(&id) = frame.vars.get(&sym) {
                return Some(id);
            }
        }
        None
    }

    fn push_stmt(&mut self, span: Span, kind: CfgStmtKind) {
        if let Some(bb) = self.cur_bb {
            self.func.blocks[bb.0 as usize]
                .stmts
                .push(CfgStmt { span, kind });
        }
    }

    fn set_term(&mut self, term: CfgTerm) {
        if let Some(bb) = self.cur_bb {
            self.func.blocks[bb.0 as usize].term = term;
            self.cur_bb = None;
        }
    }

    fn push_scope(&mut self) {
        let depth = self.scopes.len() as u32;
        self.scopes.push(ScopeFrame {
            depth,
            vars: BTreeMap::new(),
            events: Vec::new(),
        });
    }

    fn unwind_events(&mut self, events: Vec<ScopeEvent>, include_errdefer: bool, err_val: Option<CfgOperand>) {
        for ev in events.into_iter().rev() {
            if self.cur_bb.is_none() {
                break;
            }
            match ev {
                ScopeEvent::Local(id) => {
                    let sp = self.func.locals[id.0 as usize].span;
                    self.push_stmt(sp, CfgStmtKind::StorageDead(id));
                }
                ScopeEvent::Defer { body, span } => {
                    self.lower_cleanup_body(&body, false, None, span);
                }
                ScopeEvent::Errdefer { err_bind, body, span } => {
                    if include_errdefer {
                        self.lower_cleanup_body(&body, true, err_bind.map(|eb| (eb, err_val.clone())), span);
                    }
                }
            }
        }
    }

    fn pop_scope_normal(&mut self) {
        let Some(frame) = self.scopes.pop() else {
            return;
        };
        if self.cur_bb.is_some() {
            self.unwind_events(frame.events, false, None);
        }
    }

    fn emit_unwind_to_scope(&mut self, target_len: usize, include_errdefer: bool, err_val: Option<CfgOperand>) {
        let mut all_events = Vec::new();
        for frame in self.scopes[target_len..].iter() {
            all_events.extend(frame.events.clone());
        }
        self.unwind_events(all_events, include_errdefer, err_val);
    }

    fn lower_cleanup_body(
        &mut self,
        body: &DeferBody,
        is_errdefer: bool,
        err_bind: Option<(ast::IdentSpan, Option<CfgOperand>)>,
        span: Span,
    ) {
        let saved_errdefer = self.in_errdefer;
        self.cleanup_depth += 1;
        if is_errdefer {
            self.in_errdefer = true;
        }
        let body_bb = self.new_block();
        self.set_term(CfgTerm::Goto(body_bb));
        self.cur_bb = Some(body_bb);

        self.push_scope();
        if let Some((eb, opt_val)) = err_bind {
            let ety = opt_val
                .as_ref()
                .map(|v| v.ty(&self.func, self.env))
                .filter(|&t| t != TyId::ERROR)
                .unwrap_or_else(|| self.enclosing_err_set_ty());
            let loc = self.bind_user_local(eb.sym, eb.span, ety, false);
            if let Some(ev) = opt_val {
                let place = CfgPlace {
                    local: loc,
                    projs: Vec::new(),
                    ty: ety,
                    span: eb.span,
                };
                self.push_stmt(eb.span, CfgStmtKind::Assign { dst: place, rvalue: CfgRvalue::Use(ev) });
            }
        }
        match body {
            DeferBody::Block(b) => {
                for s in &b.stmts {
                    if self.cur_bb.is_none() {
                        break;
                    }
                    self.lower_stmt(s);
                }
                if self.cur_bb.is_some()
                    && let Some(tail) = &b.tail_expr
                {
                    let _ = self.lower_expr_rvalue(tail, None);
                }
            }
            DeferBody::Expr(e) => {
                let _ = self.lower_expr_rvalue(e, None);
            }
        }
        self.pop_scope_normal();

        self.in_errdefer = saved_errdefer;
        self.cleanup_depth -= 1;
        if self.cur_bb.is_some() {
            let cont_bb = self.new_block();
            let _ = span;
            self.set_term(CfgTerm::Goto(cont_bb));
            self.cur_bb = Some(cont_bb);
        }
    }

    fn lower_body(&mut self, body: &Block) {
        let ret_ty = self.func.sig.ret;
        if ret_ty != TyId::VOID && ret_ty != TyId::NEVER && body.tail_expr.is_some() {
            let ret_tmp = self.new_temp(ret_ty, body.span);
            let ret_dst = CfgPlace { local: ret_tmp, projs: Vec::new(), ty: ret_ty, span: body.span };
            self.lower_block_into(body, Some(&ret_dst), Some(ret_ty));
            if self.cur_bb.is_some() {
                let rv = self.operand_for_local(ret_dst);
                self.emit_unwind_to_scope(0, false, None);
                self.set_term(CfgTerm::Return {
                    value: Some(rv),
                    span: body.span,
                });
            }
            let _ = self.scopes.pop();
            return;
        }
        self.lower_block_into(body, None, None);
        if self.cur_bb.is_some() {
            self.emit_unwind_to_scope(0, false, None);
            self.set_term(CfgTerm::Return {
                value: None,
                span: body.span,
            });
        }
        let _ = self.scopes.pop();
    }

    fn lower_block_into(&mut self, b: &Block, dst: Option<&CfgPlace>, expected_ty: Option<TyId>) {
        self.push_scope();
        for s in &b.stmts {
            if self.cur_bb.is_none() {
                break;
            }
            self.lower_stmt(s);
        }
        if self.cur_bb.is_some()
            && let Some(tail) = &b.tail_expr
        {
            if let Some(d) = dst {
                self.lower_expr_into(tail, d, expected_ty.unwrap_or(d.ty));
            } else {
                let _ = self.lower_expr_rvalue(tail, expected_ty);
            }
        }
        self.pop_scope_normal();
    }

    fn expr_ty(&mut self, e: &Expr, fallback: TyId) -> TyId {
        let raw = self
            .table
            .get(e.span)
            .filter(|&t| t != TyId::ERROR && t != TyId::COMPTIME_INT && t != TyId::COMPTIME_FLOAT)
            .unwrap_or(fallback);
        self.subst(raw)
    }

    fn raw_ty(&mut self, span: Span) -> TyId {
        let raw = self.table.get(span).unwrap_or(TyId::ERROR);
        self.subst(raw)
    }

    fn enclosing_err_set_ty(&mut self) -> TyId {
        let err_src = match self.env.get(self.func.sig.ret) {
            Ty::ErrorUnion { errors, .. } => errors.first().copied(),
            _ => None,
        };
        match err_src {
            Some(s) => self.env.intern(Ty::ErrorSet(s)),
            // Fallback: 32-bit unsigned discriminant (`UnwrapErrorErr`).
            None => TyId::U32,
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                let init_ty = self.expr_ty(init, TyId::ERROR);
                if matches!(pat.kind, PatKind::Wildcard) && self.env.is_linear(init_ty) {
                    let init_local = if let ExprKind::Ident(sym) = &init.kind {
                        self.lookup_local(*sym)
                    } else {
                        None
                    };
                    let _ = self.lower_expr_rvalue(init, None);
                    self.push_stmt(
                        stmt.span,
                        CfgStmtKind::DiscardLinearError {
                            local: init_local,
                            span: stmt.span,
                        },
                    );
                    return;
                }
                let decl_ty = ty
                    .as_ref()
                    .map(|t| self.raw_ty(t.span))
                    .filter(|&t| t != TyId::ERROR)
                    .map(|t| self.subst(t));
                let hint = decl_ty.or_else(|| {
                    let expected = self.raw_ty(init.span);
                    if expected != TyId::ERROR
                        && expected != TyId::COMPTIME_INT
                        && expected != TyId::COMPTIME_FLOAT
                    {
                        Some(expected)
                    } else {
                        None
                    }
                });
                let val_op = self.lower_expr_rvalue(init, hint);
                if self.cur_bb.is_none() {
                    return;
                }
                let val_op = if let Some(dt) = decl_ty {
                    self.coerce_operand(val_op, dt, init.span)
                } else {
                    val_op
                };
                let val_ty = decl_ty.unwrap_or_else(|| val_op.ty(&self.func, self.env));
                self.lower_bind_pattern(pat, val_op, val_ty, false);
            }
            StmtKind::Var { name, ty, init, .. } => {
                let decl_ty = ty
                    .as_ref()
                    .map(|t| self.raw_ty(t.span))
                    .filter(|&t| t != TyId::ERROR)
                    .map(|t| self.subst(t));
                let hint = decl_ty.or_else(|| {
                    let expected = self.raw_ty(init.span);
                    if expected != TyId::ERROR
                        && expected != TyId::COMPTIME_INT
                        && expected != TyId::COMPTIME_FLOAT
                    {
                        Some(expected)
                    } else {
                        None
                    }
                });
                let val_op = self.lower_expr_rvalue(init, hint);
                if self.cur_bb.is_none() {
                    return;
                }
                let val_op = if let Some(dt) = decl_ty {
                    self.coerce_operand(val_op, dt, init.span)
                } else {
                    val_op
                };
                let val_ty = decl_ty.unwrap_or_else(|| val_op.ty(&self.func, self.env));
                let vloc = self.bind_user_local(name.sym, name.span, val_ty, true);
                let dst = CfgPlace {
                    local: vloc,
                    projs: Vec::new(),
                    ty: val_ty,
                    span: name.span,
                };
                self.push_stmt(
                    stmt.span,
                    CfgStmtKind::Assign {
                        dst,
                        rvalue: CfgRvalue::Use(val_op),
                    },
                );
            }
            StmtKind::Const(c) => {
                let cty = self.raw_ty(c.value.span);
                if let Some(iv) = self.eval_simple_const_int(&c.value) {
                    self.consts.insert(c.name.sym, iv);
                }
                let val_op = self.lower_expr_rvalue(&c.value, Some(cty));
                if self.cur_bb.is_none() {
                    return;
                }
                let val_ty = if cty != TyId::ERROR && cty != TyId::COMPTIME_INT {
                    cty
                } else {
                    val_op.ty(&self.func, self.env)
                };
                let cloc = self.bind_user_local(c.name.sym, c.name.span, val_ty, false);
                let dst = CfgPlace {
                    local: cloc,
                    projs: Vec::new(),
                    ty: val_ty,
                    span: c.name.span,
                };
                self.push_stmt(stmt.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(val_op) });
            }
            StmtKind::Defer(body) => {
                if let Some(frame) = self.scopes.last_mut() {
                    frame.events.push(ScopeEvent::Defer {
                        body: body.clone(),
                        span: stmt.span,
                    });
                }
            }
            StmtKind::Errdefer { err_bind, body } => {
                if let Some(frame) = self.scopes.last_mut() {
                    frame.events.push(ScopeEvent::Errdefer {
                        err_bind: *err_bind,
                        body: body.clone(),
                        span: stmt.span,
                    });
                }
            }
            StmtKind::Jump(j) => {
                self.lower_jump(j, stmt.span);
            }
            StmtKind::If { is_comptime, data } => {
                self.lower_if(*is_comptime, data, None, None);
            }
            StmtKind::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms, None, None, stmt.span);
            }
            StmtKind::While { label, cond, where_clauses, body } => {
                self.lower_while(label.map(|l| l.sym), cond, where_clauses, body, stmt.span);
            }
            StmtKind::For { label, pat, iter_start, iter_end, body } => {
                self.lower_for(label.map(|l| l.sym), pat, iter_start, iter_end.as_ref(), body, stmt.span);
            }
            StmtKind::ParFor { pat, iter_start, iter_end, body } => {
                self.lower_for_inner(None, pat, iter_start, iter_end.as_ref(), body, stmt.span, true);
            }
            StmtKind::Loop { label, body } => {
                let _ = self.lower_loop(label.map(|l| l.sym), body, None, None, stmt.span);
            }
            StmtKind::ComptimeBlock(b) => {
                self.lower_block_into(b, None, None);
            }
            StmtKind::RawBlock(b) => {
                self.lower_block_into(b, None, None);
            }
            StmtKind::Assign { lhs, op, rhs } => {
                self.lower_assign(lhs, *op, rhs, stmt.span);
            }
            StmtKind::Discard(e) => {
                let ety = self.expr_ty(e, TyId::ERROR);
                if self.env.is_linear(ety) {
                    let loc = if let ExprKind::Ident(sym) = &e.kind {
                        self.lookup_local(*sym)
                    } else {
                        None
                    };
                    let _ = self.lower_expr_rvalue(e, None);
                    self.push_stmt(
                        stmt.span,
                        CfgStmtKind::DiscardLinearError {
                            local: loc,
                            span: stmt.span,
                        },
                    );
                } else {
                    let op = self.lower_expr_rvalue(e, None);
                    if self.cur_bb.is_some() && matches!(op, CfgOperand::Copy(_) | CfgOperand::Move(_)) {
                        let oty = op.ty(&self.func, self.env);
                        if oty != TyId::VOID && oty != TyId::NEVER {
                            let tmp = self.new_temp(oty, stmt.span);
                            let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: oty, span: stmt.span };
                            self.push_stmt(stmt.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(op) });
                        }
                    }
                }
            }
            StmtKind::Expr(e) => {
                let _ = self.lower_expr_rvalue(e, None);
            }
        }
    }

    fn lower_assign(&mut self, lhs: &Expr, op: AssignOp, rhs: &Expr, span: Span) {
        if op == AssignOp::Assign
            && self.is_pool_reset_assign(lhs, rhs)
            && let ExprKind::Ident(sym) = &lhs.kind
            && let Some(pool_local) = self.lookup_local(*sym)
        {
            let pool_ty = self.func.locals[pool_local.0 as usize].ty;
            let elem_ty = match self.env.get(pool_ty) {
                Ty::Builtin(BuiltinKind::ArenaPool, args) => match args.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                },
                _ => TyId::ERROR,
            };
            let pool_place = CfgPlace {
                local: pool_local,
                projs: Vec::new(),
                ty: pool_ty,
                span: lhs.span,
            };
            let tmp = self.new_temp(pool_ty, rhs.span);
            let tmp_place = CfgPlace {
                local: tmp,
                projs: Vec::new(),
                ty: pool_ty,
                span: rhs.span,
            };
            self.push_stmt(
                rhs.span,
                CfgStmtKind::Call {
                    dst: Some(tmp_place.clone()),
                    callee: CfgCallee::Builtin(CfgBuiltinOp::ArenaPoolReset(elem_ty)),
                    args: vec![CfgOperand::Copy(pool_place.clone())],
                    arg_spans: vec![lhs.span],
                    recv_place: Some(pool_place.clone()),
                    recv_span: Some(lhs.span),
                    is_mut_recv: true,
                    consumes_recv: false,
                },
            );
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: pool_place,
                    rvalue: CfgRvalue::Use(CfgOperand::Move(tmp_place)),
                },
            );
            self.push_stmt(span, CfgStmtKind::PoolResetEpoch { pool_local });
            return;
        }

        let lhs_ty = self.expr_ty(lhs, TyId::ERROR);
        if op == AssignOp::Assign {
            let rhs_raw = self.lower_expr_rvalue(rhs, Some(lhs_ty));
            if self.cur_bb.is_none() {
                return;
            }
            let rhs_op = self.coerce_operand(rhs_raw, lhs_ty, rhs.span);
            let lhs_place = self.lower_place(lhs);
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: lhs_place,
                    rvalue: CfgRvalue::Use(rhs_op),
                },
            );
        } else {
            let lhs_place = self.lower_place(lhs);
            let target_ty = lhs_place.ty;
            let cur_val = CfgOperand::Copy(lhs_place.clone());
            let rhs_raw = self.lower_expr_rvalue(rhs, Some(target_ty));
            if self.cur_bb.is_none() {
                return;
            }
            let rhs_op = self.coerce_operand(rhs_raw, target_ty, rhs.span);
            let bin = match op {
                AssignOp::AddAssign => BinaryOp::Add,
                AssignOp::SubAssign => BinaryOp::Sub,
                AssignOp::MulAssign => BinaryOp::Mul,
                AssignOp::AddWrapAssign => BinaryOp::AddWrap,
                AssignOp::SubWrapAssign => BinaryOp::SubWrap,
                AssignOp::MulWrapAssign => BinaryOp::MulWrap,
                AssignOp::AddSatAssign => BinaryOp::AddSat,
                AssignOp::SubSatAssign => BinaryOp::SubSat,
                AssignOp::MulSatAssign => BinaryOp::MulSat,
                AssignOp::BitAndAssign => BinaryOp::BitAnd,
                AssignOp::BitOrAssign => BinaryOp::BitOr,
                AssignOp::BitXorAssign => BinaryOp::BitXor,
                AssignOp::ShlAssign => BinaryOp::Shl,
                AssignOp::ShrAssign => BinaryOp::Shr,
                AssignOp::Assign => unreachable!(),
            };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: lhs_place,
                    rvalue: CfgRvalue::Binary {
                        op: bin,
                        lhs: cur_val,
                        rhs: rhs_op,
                    },
                },
            );
        }
    }

    fn is_pool_reset_assign(&self, lhs: &Expr, rhs: &Expr) -> bool {
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

    fn lower_jump(&mut self, j: &JumpAction, span: Span) {
        match j {
            JumpAction::Return(val_opt) => {
                let sig_ret = self.func.sig.ret;
                let (is_err_return, ret_val_span) = if let Some(v) = val_opt {
                    let is_err_lit = if let ExprKind::EnumLit(ev) = &v.kind
                        && let Ty::ErrorUnion { errors, .. } = self.env.get(sig_ret)
                    {
                        self.env.error_sources_have_variant(errors, ev.sym, self.interner)
                    } else {
                        false
                    };
                    (is_err_lit, v.span)
                } else {
                    (false, span)
                };

                let mut prepared = val_opt.as_ref().map(|val| {
                    let hint = if is_err_return {
                        Some(sig_ret)
                    } else if let Ty::ErrorUnion { ok, .. } = self.env.get(sig_ret) {
                        Some(*ok)
                    } else {
                        Some(sig_ret)
                    };
                    let op = self.lower_expr_rvalue(val, hint);
                    if is_err_return {
                        self.coerce_error_into_union(op, val.span, sig_ret)
                    } else {
                        self.coerce_operand(op, sig_ret, val.span)
                    }
                });

                if self.cur_bb.is_none() {
                    return;
                }
                if let Some(op) = prepared.take() {
                    let rty = op.ty(&self.func, self.env);
                    let rtmp = self.new_local(None, ret_val_span, rty, false, false, false, true, None);
                    self.push_stmt(ret_val_span, CfgStmtKind::StorageLive(rtmp));
                    let rplace = CfgPlace {
                        local: rtmp,
                        projs: Vec::new(),
                        ty: rty,
                        span: ret_val_span,
                    };
                    self.push_stmt(ret_val_span, CfgStmtKind::Assign { dst: rplace.clone(), rvalue: CfgRvalue::Use(op) });
                    prepared = Some(self.operand_for_local(rplace));
                }

                if is_err_return {
                    self.cleanup_depth += 1;
                    self.active_err_span = Some(span);
                    self.emit_unwind_to_scope(0, true, prepared.clone());
                    self.active_err_span = None;
                    self.cleanup_depth -= 1;
                } else {
                    self.emit_unwind_to_scope(0, false, None);
                }

                if self.cur_bb.is_some() {
                    self.set_term(CfgTerm::Return {
                        value: prepared,
                        span: ret_val_span,
                    });
                }
            }
            JumpAction::Break { label, value } => {
                let target_idx = match label {
                    Some(lbl) => self.loops.iter().rposition(|l| l.label == Some(lbl.sym)),
                    None => self.loops.len().checked_sub(1),
                };
                let Some(idx) = target_idx else {
                    self.set_term(CfgTerm::Unreachable { span, explicit: false });
                    return;
                };
                if let Some(val) = value {
                    let expect_ty = self.loops[idx].break_origin_ty;
                    let op = self.lower_expr_rvalue(val, if expect_ty != TyId::ERROR { Some(expect_ty) } else { None });
                    if self.cur_bb.is_none() {
                        return;
                    }
                    if let Some(dst) = self.loops[idx].break_dst.clone() {
                        let cop = self.coerce_operand(op, dst.ty, val.span);
                        self.push_stmt(
                            val.span,
                            CfgStmtKind::Assign {
                                dst,
                                rvalue: CfgRvalue::Use(cop),
                            },
                        );
                    }
                }
                self.loops[idx].any_break = true;
                let scope_len = self.loops[idx].scope_len_at_entry;
                let exit_bb = self.loops[idx].exit_bb;
                self.emit_unwind_to_scope(scope_len, false, None);
                if self.cur_bb.is_some() {
                    self.set_term(CfgTerm::Goto(exit_bb));
                }
            }
            JumpAction::Continue { label } => {
                let target_idx = match label {
                    Some(lbl) => self.loops.iter().rposition(|l| l.label == Some(lbl.sym)),
                    None => self.loops.len().checked_sub(1),
                };
                let Some(idx) = target_idx else {
                    self.set_term(CfgTerm::Unreachable { span, explicit: false });
                    return;
                };
                let scope_len = self.loops[idx].scope_len_at_entry;
                let cont_bb = self.loops[idx].continue_bb;
                self.emit_unwind_to_scope(scope_len, false, None);
                if self.cur_bb.is_some() {
                    self.set_term(CfgTerm::Goto(cont_bb));
                }
            }
            JumpAction::Unreachable => {
                self.set_term(CfgTerm::Unreachable { span, explicit: true });
            }
        }
    }

    fn lower_if(&mut self, is_comptime: bool, data: &IfExpr, dst: Option<&CfgPlace>, expected_ty: Option<TyId>) {
        if is_comptime {
            let taken = self.table.get_comptime_branch(data.span).unwrap_or(true);
            if taken {
                self.lower_block_into(&data.then_block, dst, expected_ty);
            } else if let Some(eb) = &data.else_branch {
                match eb {
                    ElseBranch::Block(b) => self.lower_block_into(b, dst, expected_ty),
                    ElseBranch::If { is_comptime: ic, data: nested } => {
                        self.lower_if(*ic, nested, dst, expected_ty);
                    }
                }
            }
            return;
        }
        let merge_bb = self.new_block();
        self.func.blocks[merge_bb.0 as usize].join_span = Some(data.span);

        let then_bb = self.new_block();
        let else_bb = self.new_block();
        let mut if_let_binding: Option<(Pat, CfgOperand, TyId)> = None;

        match &data.cond {
            IfCond::Expr(cond) => {
                self.lower_cond_branch(cond, then_bb, else_bb, cond.span, true);
            }
            IfCond::Let { pat, expr } => {
                let eop = self.lower_expr_rvalue(expr, None);
                if self.cur_bb.is_none() {
                    return;
                }
                let ety = eop.ty(&self.func, self.env);
                let opt_tmp = self.new_temp(ety, expr.span);
                let opt_place = CfgPlace {
                    local: opt_tmp,
                    projs: Vec::new(),
                    ty: ety,
                    span: expr.span,
                };
                self.push_stmt(expr.span, CfgStmtKind::Assign { dst: opt_place.clone(), rvalue: CfgRvalue::Use(eop) });
                let is_some_tmp = self.new_temp(TyId::BOOL, expr.span);
                let is_some_place = CfgPlace {
                    local: is_some_tmp,
                    projs: Vec::new(),
                    ty: TyId::BOOL,
                    span: expr.span,
                };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst: is_some_place.clone(),
                        rvalue: CfgRvalue::IsNonNull(CfgOperand::Copy(opt_place.clone())),
                    },
                );
                let inner_ty = match self.env.get(ety) {
                    Ty::Optional(inner) => *inner,
                    _ => TyId::ERROR,
                };
                let unwrap_op = if self.env.is_linear(inner_ty) {
                    CfgOperand::Move(opt_place)
                } else {
                    CfgOperand::Copy(opt_place)
                };
                if_let_binding = Some((pat.clone(), unwrap_op, inner_ty));
                self.set_term(CfgTerm::Branch {
                    cond: CfgOperand::Copy(is_some_place),
                    then_bb,
                    else_bb,
                    span: expr.span,
                    is_user_if: false,
                });
            }
        }

        let mut any_reach_merge = false;

        self.cur_bb = Some(then_bb);
        if let Some((pat, unwrap_src, inner_ty)) = if_let_binding {
            self.push_scope();
            let un_tmp = self.new_temp(inner_ty, pat.span);
            let un_place = CfgPlace {
                local: un_tmp,
                projs: Vec::new(),
                ty: inner_ty,
                span: pat.span,
            };
            self.push_stmt(
                pat.span,
                CfgStmtKind::Assign {
                    dst: un_place.clone(),
                    rvalue: CfgRvalue::UnwrapOptional(unwrap_src),
                },
            );
            let un_op = self.operand_for_local(un_place);
            self.lower_bind_pattern(&pat, un_op, inner_ty, false);
            self.lower_block_into(&data.then_block, dst, expected_ty);
            self.pop_scope_normal();
        } else {
            self.lower_block_into(&data.then_block, dst, expected_ty);
        }
        if self.cur_bb.is_some() {
            any_reach_merge = true;
            self.set_term(CfgTerm::Goto(merge_bb));
        }

        self.cur_bb = Some(else_bb);
        match &data.else_branch {
            Some(ElseBranch::Block(b)) => {
                self.lower_block_into(b, dst, expected_ty);
                if self.cur_bb.is_some() {
                    any_reach_merge = true;
                    self.set_term(CfgTerm::Goto(merge_bb));
                }
            }
            Some(ElseBranch::If { is_comptime: ic, data: nested }) => {
                self.lower_if(*ic, nested, dst, expected_ty);
                if self.cur_bb.is_some() {
                    any_reach_merge = true;
                    self.set_term(CfgTerm::Goto(merge_bb));
                }
            }
            None => {
                any_reach_merge = true;
                self.set_term(CfgTerm::Goto(merge_bb));
            }
        }

        if any_reach_merge {
            self.cur_bb = Some(merge_bb);
        } else {
            self.cur_bb = None;
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        dst: Option<&CfgPlace>,
        expected_ty: Option<TyId>,
        match_span: Span,
    ) {
        let scrut_op = self.lower_expr_rvalue(scrutinee, None);
        if self.cur_bb.is_none() {
            return;
        }
        let scrut_ty = scrut_op.ty(&self.func, self.env);
        let scrut_local = self.new_temp(scrut_ty, scrutinee.span);
        let scrut_place = CfgPlace {
            local: scrut_local,
            projs: Vec::new(),
            ty: scrut_ty,
            span: scrutinee.span,
        };
        self.push_stmt(
            scrutinee.span,
            CfgStmtKind::Assign {
                dst: scrut_place.clone(),
                rvalue: CfgRvalue::Use(scrut_op),
            },
        );

        let merge_bb = self.new_block();
        self.func.blocks[merge_bb.0 as usize].join_span = Some(match_span);
        let mut any_reach_merge = false;

        if arms.is_empty() {
            self.set_term(CfgTerm::Goto(merge_bb));
            self.cur_bb = Some(merge_bb);
            return;
        }

        for (i, arm) in arms.iter().enumerate() {
            if self.cur_bb.is_none() {
                break;
            }
            let is_last = i + 1 == arms.len();
            let arm_body_bb = self.new_block();
            let next_test_bb = if is_last { None } else { Some(self.new_block()) };

            let is_irrefutable = self.pat_is_irrefutable(&arm.pat) && arm.guard.is_none();
            if is_irrefutable || (is_last && arm.guard.is_none()) {
                self.set_term(CfgTerm::Goto(arm_body_bb));
                self.cur_bb = Some(arm_body_bb);
                self.push_scope();
                let sop = if self.env.is_linear(scrut_ty) {
                    CfgOperand::Move(scrut_place.clone())
                } else {
                    CfgOperand::Copy(scrut_place.clone())
                };
                self.lower_bind_pattern(&arm.pat, sop, scrut_ty, false);
                self.lower_arm_body(&arm.body, dst, expected_ty, arm.span);
                self.pop_scope_normal();
                if self.cur_bb.is_some() {
                    any_reach_merge = true;
                    self.set_term(CfgTerm::Goto(merge_bb));
                }
                if is_irrefutable {
                    break;
                }
                self.cur_bb = next_test_bb;
                continue;
            }

            let cond_op = self.lower_pat_test(&scrut_place, &arm.pat, scrut_ty, arm.span);
            if self.cur_bb.is_none() {
                break;
            }
            let bind_bb = self.new_block();
            let fail_bb = next_test_bb.unwrap_or_else(|| self.new_block());
            if next_test_bb.is_none() {
                self.func.blocks[fail_bb.0 as usize].term = CfgTerm::Unreachable { span: arm.span, explicit: false };
            }
            self.set_term(CfgTerm::Branch {
                cond: cond_op,
                then_bb: bind_bb,
                else_bb: fail_bb,
                span: arm.span,
                is_user_if: false,
            });

            self.cur_bb = Some(bind_bb);
            self.push_scope();
            let sop = if self.env.is_linear(scrut_ty) && is_last {
                CfgOperand::Move(scrut_place.clone())
            } else {
                CfgOperand::Copy(scrut_place.clone())
            };
            self.lower_bind_pattern(&arm.pat, sop, scrut_ty, false);
            if let Some(guard) = &arm.guard {
                let gop = self.lower_expr_rvalue(guard, Some(TyId::BOOL));
                if self.cur_bb.is_some() {
                    let guard_fail = self.new_block();
                    self.set_term(CfgTerm::Branch {
                        cond: gop,
                        then_bb: arm_body_bb,
                        else_bb: guard_fail,
                        span: guard.span,
                        is_user_if: false,
                    });
                    self.cur_bb = Some(guard_fail);
                    self.pop_scope_normal();
                    if self.cur_bb.is_some() {
                        self.set_term(CfgTerm::Goto(fail_bb));
                    }
                    self.cur_bb = Some(arm_body_bb);
                }
            } else {
                self.set_term(CfgTerm::Goto(arm_body_bb));
                self.cur_bb = Some(arm_body_bb);
            }

            if self.cur_bb.is_some() {
                self.lower_arm_body(&arm.body, dst, expected_ty, arm.span);
                self.pop_scope_normal();
                if self.cur_bb.is_some() {
                    any_reach_merge = true;
                    self.set_term(CfgTerm::Goto(merge_bb));
                }
            } else {
                let _ = self.scopes.pop();
            }

            self.cur_bb = next_test_bb;
        }

        if any_reach_merge {
            self.cur_bb = Some(merge_bb);
        } else {
            self.cur_bb = None;
        }
    }

    fn lower_arm_body(
        &mut self,
        body: &MatchArmBody,
        dst: Option<&CfgPlace>,
        expected_ty: Option<TyId>,
        span: Span,
    ) {
        match body {
            MatchArmBody::Block(b) => self.lower_block_into(b, dst, expected_ty),
            MatchArmBody::Expr(e) => {
                if let Some(d) = dst {
                    self.lower_expr_into(e, d, expected_ty.unwrap_or(d.ty));
                } else {
                    let _ = self.lower_expr_rvalue(e, expected_ty);
                }
            }
            MatchArmBody::Jump(j) => self.lower_jump(j, span),
        }
    }

    fn pat_is_irrefutable(&self, pat: &Pat) -> bool {
        matches!(pat.kind, PatKind::Wildcard | PatKind::Ident(_) | PatKind::Error)
    }

    fn lower_pat_test(&mut self, scrut: &CfgPlace, pat: &Pat, scrut_ty: TyId, span: Span) -> CfgOperand {
        match &pat.kind {
            PatKind::Wildcard | PatKind::Ident(_) | PatKind::Error => CfgOperand::Const(CfgConst::Bool(true)),
            PatKind::BoolLit(b) => self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Bool(*b)), TyId::BOOL, span),
            PatKind::Null => {
                let tmp = self.new_temp(TyId::BOOL, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: TyId::BOOL, span };
                self.push_stmt(span, CfgStmtKind::Assign { dst: dst.clone(), rvalue: CfgRvalue::IsNonNull(CfgOperand::Copy(scrut.clone())) });
                self.emit_unary_tmp(UnaryOp::Not, CfgOperand::Copy(dst), TyId::BOOL, span)
            }
            PatKind::CharLit { ch, range_end } => {
                if let Some((rk, end_ch)) = range_end {
                    let c1 = self.emit_bin_tmp(BinaryOp::GtEq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Char(*ch)), TyId::BOOL, span);
                    let cmp = if *rk == RangeKind::Inclusive { BinaryOp::LtEq } else { BinaryOp::Lt };
                    let c2 = self.emit_bin_tmp(cmp, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Char(*end_ch)), TyId::BOOL, span);
                    self.emit_bin_tmp(BinaryOp::And, c1, c2, TyId::BOOL, span)
                } else {
                    self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Char(*ch)), TyId::BOOL, span)
                }
            }
            PatKind::IntLit { neg, raw, range_end } => {
                let lo = parse_signed(*raw, *neg, self.interner).unwrap_or(0);
                if let Some((rk, e_neg, e_raw)) = range_end {
                    let hi = parse_signed(*e_raw, *e_neg, self.interner).unwrap_or(0);
                    let c1 = self.emit_bin_tmp(BinaryOp::GtEq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Int(lo, scrut_ty)), TyId::BOOL, span);
                    let cmp = if *rk == RangeKind::Inclusive { BinaryOp::LtEq } else { BinaryOp::Lt };
                    let c2 = self.emit_bin_tmp(cmp, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Int(hi, scrut_ty)), TyId::BOOL, span);
                    self.emit_bin_tmp(BinaryOp::And, c1, c2, TyId::BOOL, span)
                } else {
                    self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(CfgConst::Int(lo, scrut_ty)), TyId::BOOL, span)
                }
            }
            PatKind::Variant { name, .. } => {
                if matches!(self.env.get(scrut_ty), Ty::Union { .. }) {
                    let tag_tmp = self.new_temp(TyId::U32, span);
                    let tag_dst = CfgPlace { local: tag_tmp, projs: Vec::new(), ty: TyId::U32, span };
                    self.push_stmt(span, CfgStmtKind::Assign { dst: tag_dst.clone(), rvalue: CfgRvalue::MatchTag(CfgOperand::Copy(scrut.clone())) });
                    let tag_val = self.variant_tag_index(scrut_ty, name.sym).unwrap_or(0) as i128;
                    self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(tag_dst), CfgOperand::Const(CfgConst::Int(tag_val, TyId::U32)), TyId::BOOL, span)
                } else if matches!(self.env.get(scrut_ty), Ty::ErrorSet(_) | Ty::ErrorUnion { .. }) {
                    let tag_tmp = self.new_temp(TyId::U32, span);
                    let tag_dst = CfgPlace { local: tag_tmp, projs: Vec::new(), ty: TyId::U32, span };
                    self.push_stmt(span, CfgStmtKind::Assign { dst: tag_dst.clone(), rvalue: CfgRvalue::MatchTag(CfgOperand::Copy(scrut.clone())) });
                    let tag_val = self.error_variant_tag(scrut_ty, name.sym) as i128;
                    self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(tag_dst), CfgOperand::Const(CfgConst::Int(tag_val, TyId::U32)), TyId::BOOL, span)
                } else {
                    let val = self.variant_tag_index(scrut_ty, name.sym).unwrap_or(0) as i128;
                    let c = CfgConst::EnumTag { ty: scrut_ty, variant: name.sym, tag_val: val };
                    self.emit_bin_tmp(BinaryOp::Eq, CfgOperand::Copy(scrut.clone()), CfgOperand::Const(c), TyId::BOOL, span)
                }
            }
            PatKind::Tuple(_) | PatKind::Struct { .. } => CfgOperand::Const(CfgConst::Bool(true)),
        }
    }

    fn lower_bind_pattern(&mut self, pat: &Pat, val: CfgOperand, ty: TyId, is_mut: bool) {
        match &pat.kind {
            PatKind::Wildcard => {
                if matches!(val, CfgOperand::Copy(_) | CfgOperand::Move(_)) && ty != TyId::VOID && ty != TyId::NEVER {
                    let tmp = self.new_temp(ty, pat.span);
                    let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: pat.span };
                    self.push_stmt(pat.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(val) });
                }
            }
            PatKind::Error | PatKind::Null | PatKind::BoolLit(_) | PatKind::CharLit { .. } | PatKind::IntLit { .. } => {}
            PatKind::Ident(sym) => {
                if self.interner.get(*sym) == "_" {
                    if matches!(val, CfgOperand::Copy(_) | CfgOperand::Move(_)) && ty != TyId::VOID && ty != TyId::NEVER {
                        let tmp = self.new_temp(ty, pat.span);
                        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: pat.span };
                        self.push_stmt(pat.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(val) });
                    }
                    return;
                }
                let loc = self.bind_user_local(*sym, pat.span, ty, is_mut);
                let dst = CfgPlace { local: loc, projs: Vec::new(), ty, span: pat.span };
                self.push_stmt(pat.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(val) });
            }
            PatKind::Tuple(ps) => {
                let tmp = self.new_temp(ty, pat.span);
                let tmp_place = CfgPlace { local: tmp, projs: Vec::new(), ty, span: pat.span };
                self.push_stmt(pat.span, CfgStmtKind::Assign { dst: tmp_place.clone(), rvalue: CfgRvalue::Use(val) });
                let elems = match self.env.get(ty) {
                    Ty::Tuple(es) => es.clone(),
                    _ => Vec::new(),
                };
                for (i, p) in ps.iter().enumerate() {
                    let ety = elems.get(i).copied().unwrap_or(TyId::ERROR);
                    let mut field_place = tmp_place.clone();
                    field_place.projs.push(CfgProj::TupleField { index: i, ty: ety });
                    field_place.ty = ety;
                    let op = if self.env.is_linear(ety) {
                        CfgOperand::Move(field_place)
                    } else {
                        CfgOperand::Copy(field_place)
                    };
                    self.lower_bind_pattern(p, op, ety, is_mut);
                }
                self.push_stmt(pat.span, CfgStmtKind::ConsumeDestructured(tmp));
            }
            PatKind::Struct { path: _, fields } => {
                let tmp = self.new_temp(ty, pat.span);
                let tmp_place = CfgPlace { local: tmp, projs: Vec::new(), ty, span: pat.span };
                self.push_stmt(pat.span, CfgStmtKind::Assign { dst: tmp_place.clone(), rvalue: CfgRvalue::Use(val) });
                let (sdef, sgargs) = match self.env.get(ty) {
                    Ty::Struct { def, args } => (Some(*def), args.clone()),
                    _ => (None, Vec::new()),
                };
                let smeta = sdef.and_then(|d| self.env.structs.get(&d)).cloned();
                let eff_gargs = if !sgargs.is_empty() { sgargs } else { self.generic_subst.clone() };
                for fp in fields {
                    let (fidx, raw_fty) = smeta
                        .as_ref()
                        .and_then(|m| m.fields.iter().enumerate().find(|(_, f)| f.name == fp.name.sym))
                        .map(|(idx, f)| (idx, f.ty))
                        .unwrap_or((0, TyId::ERROR));
                    let fty = if !eff_gargs.is_empty() {
                        crate::check::subst_ty(raw_fty, &eff_gargs, self.env)
                    } else {
                        raw_fty
                    };
                    let mut fp_place = tmp_place.clone();
                    fp_place.projs.push(CfgProj::Field { name: fp.name.sym, field_idx: fidx, ty: fty });
                    fp_place.ty = fty;
                    let fop = if self.env.is_linear(fty) {
                        CfgOperand::Move(fp_place)
                    } else {
                        CfgOperand::Copy(fp_place)
                    };
                    if let Some(sub) = &fp.pat {
                        self.lower_bind_pattern(sub, fop, fty, is_mut);
                    } else if self.interner.get(fp.name.sym) != "_" {
                        let loc = self.bind_user_local(fp.name.sym, fp.name.span, fty, is_mut);
                        let dst = CfgPlace { local: loc, projs: Vec::new(), ty: fty, span: fp.name.span };
                        self.push_stmt(fp.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::Use(fop) });
                    }
                }
                self.push_stmt(pat.span, CfgStmtKind::ConsumeDestructured(tmp));
            }
            PatKind::Variant { name, args } => {
                if let Ty::Union { def, .. } = self.env.get(ty).clone()
                    && let Some(um) = self.env.unions.get(&def).cloned()
                    && let Some((vidx, var)) = um.variants.iter().enumerate().find(|(_, v)| v.name == name.sym)
                    && let Some(pty) = var.payload
                    && let Some(p0) = args.first()
                {
                    let ftmp = self.new_temp(pty, pat.span);
                    let dst = CfgPlace { local: ftmp, projs: Vec::new(), ty: pty, span: pat.span };
                    self.push_stmt(
                        pat.span,
                        CfgStmtKind::Assign {
                            dst: dst.clone(),
                            rvalue: CfgRvalue::FieldExtract {
                                operand: val,
                                field_idx: vidx,
                                ty: pty,
                            },
                        },
                    );
                    let popped = self.operand_for_local(dst);
                    self.lower_bind_pattern(p0, popped, pty, is_mut);
                }
            }
        }
    }

    fn lower_cond_branch(
        &mut self,
        cond: &Expr,
        then_bb: BlockId,
        else_bb: BlockId,
        span: Span,
        is_user_if: bool,
    ) {
        if self.cur_bb.is_none() {
            return;
        }
        match &cond.kind {
            ExprKind::Paren(inner) => {
                self.lower_cond_branch(inner, then_bb, else_bb, span, is_user_if);
            }
            ExprKind::Unary { op: UnaryOp::Not, expr: inner } => {
                self.lower_cond_branch(inner, else_bb, then_bb, span, is_user_if);
            }
            ExprKind::Binary { op: BinaryOp::And, lhs, rhs } => {
                let mid_bb = self.new_block();
                self.lower_cond_branch(lhs, mid_bb, else_bb, lhs.span, false);
                self.cur_bb = Some(mid_bb);
                self.lower_cond_branch(rhs, then_bb, else_bb, rhs.span, false);
            }
            ExprKind::Binary { op: BinaryOp::Or, lhs, rhs } => {
                let mid_bb = self.new_block();
                self.lower_cond_branch(lhs, then_bb, mid_bb, lhs.span, false);
                self.cur_bb = Some(mid_bb);
                self.lower_cond_branch(rhs, then_bb, else_bb, rhs.span, false);
            }
            _ => {
                let cop = self.lower_expr_rvalue(cond, Some(TyId::BOOL));
                if self.cur_bb.is_some() {
                    self.set_term(CfgTerm::Branch {
                        cond: cop,
                        then_bb,
                        else_bb,
                        span,
                        is_user_if,
                    });
                }
            }
        }
    }

    fn lower_while(&mut self, label: Option<Symbol>, cond: &Expr, where_clauses: &[Expr], body: &Block, span: Span) {
        let header_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();
        self.func.blocks[header_bb.0 as usize].is_loop_header = true;
        self.func.blocks[header_bb.0 as usize].loop_entry_depth = self.scopes.last().map(|s| s.depth);
        if !where_clauses.is_empty() {
            self.func.loop_invariants.insert(header_bb, where_clauses.to_vec());
        }

        self.set_term(CfgTerm::Goto(header_bb));
        self.cur_bb = Some(header_bb);

        self.lower_cond_branch(cond, body_bb, exit_bb, span, false);

        self.loops.push(LoopTarget {
            label,
            continue_bb: header_bb,
            exit_bb,
            break_dst: None,
            break_origin_ty: TyId::VOID,
            scope_len_at_entry: self.scopes.len(),
            any_break: false,
        });

        self.cur_bb = Some(body_bb);
        self.lower_block_into(body, None, None);
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(header_bb));
        }

        self.loops.pop();
        self.cur_bb = Some(exit_bb);
    }

    fn lower_for(
        &mut self,
        label: Option<Symbol>,
        pat: &Pat,
        iter_start: &Expr,
        iter_end: Option<&(RangeKind, Expr)>,
        body: &Block,
        span: Span,
    ) {
        self.lower_for_inner(label, pat, iter_start, iter_end, body, span, false);
    }

    fn lower_for_inner(
        &mut self,
        label: Option<Symbol>,
        pat: &Pat,
        iter_start: &Expr,
        iter_end: Option<&(RangeKind, Expr)>,
        body: &Block,
        span: Span,
        is_par: bool,
    ) {
        let Some((rk, end_expr)) = iter_end else {
            self.lower_for_seq(label, pat, iter_start, body, span, is_par);
            return;
        };
        let end_ty = self.expr_ty(end_expr, TyId::USIZE);
        let idx_ty = self.expr_ty(iter_start, end_ty);
        let idx_ty = if idx_ty == TyId::COMPTIME_INT { TyId::USIZE } else { idx_ty };
        let start_op = self.lower_expr_rvalue(iter_start, Some(idx_ty));
        let end_op = self.lower_expr_rvalue(end_expr, Some(idx_ty));
        if self.cur_bb.is_none() {
            return;
        }

        self.push_scope();
        let idx_local = self.new_temp(idx_ty, iter_start.span);
        let idx_place = CfgPlace {
            local: idx_local,
            projs: Vec::new(),
            ty: idx_ty,
            span: iter_start.span,
        };
        self.push_stmt(
            iter_start.span,
            CfgStmtKind::Assign {
                dst: idx_place.clone(),
                rvalue: CfgRvalue::Use(start_op),
            },
        );
        let end_local = self.new_temp(idx_ty, end_expr.span);
        let end_place = CfgPlace {
            local: end_local,
            projs: Vec::new(),
            ty: idx_ty,
            span: end_expr.span,
        };
        self.push_stmt(
            end_expr.span,
            CfgStmtKind::Assign {
                dst: end_place.clone(),
                rvalue: CfgRvalue::Use(end_op),
            },
        );

        let header_bb = self.new_block();
        let body_bb = self.new_block();
        let step_bb = self.new_block();
        let exit_bb = self.new_block();
        self.func.blocks[header_bb.0 as usize].is_loop_header = true;
        self.func.blocks[header_bb.0 as usize].loop_entry_depth = self.scopes.last().map(|s| s.depth);
        if is_par && *rk == RangeKind::Exclusive {
            self.func.par_for_regions.push(ParForRegion {
                header_bb,
                body_bb,
                step_bb,
                exit_bb,
                idx_local,
                end_local,
                chunk_size: 256,
            });
        }

        self.set_term(CfgTerm::Goto(header_bb));
        self.cur_bb = Some(header_bb);

        let cmp_op = if *rk == RangeKind::Inclusive {
            BinaryOp::LtEq
        } else {
            BinaryOp::Lt
        };
        let cond = self.emit_bin_tmp(
            cmp_op,
            CfgOperand::Copy(idx_place.clone()),
            CfgOperand::Copy(end_place),
            TyId::BOOL,
            span,
        );
        self.set_term(CfgTerm::Branch {
            cond,
            then_bb: body_bb,
            else_bb: exit_bb,
            span,
            is_user_if: false,
        });

        self.loops.push(LoopTarget {
            label,
            continue_bb: step_bb,
            exit_bb,
            break_dst: None,
            break_origin_ty: TyId::VOID,
            scope_len_at_entry: self.scopes.len(),
            any_break: false,
        });

        self.cur_bb = Some(body_bb);
        self.push_scope();
        self.lower_bind_pattern(pat, CfgOperand::Copy(idx_place.clone()), idx_ty, false);
        self.lower_block_into(body, None, None);
        self.pop_scope_normal();
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(step_bb));
        }

        self.cur_bb = Some(step_bb);
        let one = CfgOperand::Const(CfgConst::Int(1, idx_ty));
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: idx_place.clone(),
                rvalue: CfgRvalue::Binary {
                    op: BinaryOp::AddWrap,
                    lhs: CfgOperand::Copy(idx_place),
                    rhs: one,
                },
            },
        );
        self.set_term(CfgTerm::Goto(header_bb));

        self.loops.pop();
        self.cur_bb = Some(exit_bb);
        self.pop_scope_normal();
    }

    fn lower_for_seq(&mut self, label: Option<Symbol>, pat: &Pat, seq_expr: &Expr, body: &Block, span: Span, is_par: bool) {
        let (actual_seq_expr, borrow_mode) = match &seq_expr.kind {
            ExprKind::Call { callee, args } if args.is_empty() => match &callee.kind {
                ExprKind::Field { base, field } if self.interner.get(field.sym) == "each_mut" => {
                    (base.as_ref(), Some(true))
                }
                ExprKind::Field { base, field } if self.interner.get(field.sym) == "each" => {
                    (base.as_ref(), Some(false))
                }
                _ => (seq_expr, None),
            },
            _ => (seq_expr, None),
        };
        // Special case: iterating `SlotPool[T]` (`pool.each_mut()` / `pool.each()`):
        // lower to `for idx in 0..pool.cap() { if let id = pool.id_at(idx) { body } }`!
        let seq_expr_ty = self.expr_ty(actual_seq_expr, TyId::ERROR);
        let seq_deref_ty = match self.env.get(seq_expr_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => seq_expr_ty,
        };
        if let Ty::Builtin(BuiltinKind::SlotPool, gargs) = self.env.get(seq_deref_ty).clone() {
            let elem_ty = match gargs.first() {
                Some(GenericVal::Ty(t)) => *t,
                _ => TyId::ERROR,
            };
            let gen_id_ty = self.env.intern(Ty::Builtin(BuiltinKind::GenId, vec![GenericVal::Ty(elem_ty)]));
            let opt_gen_id_ty = self.env.intern(Ty::Optional(gen_id_ty));
            let pool_place = self.lower_place(actual_seq_expr);
            if self.cur_bb.is_none() {
                return;
            }
            self.push_scope();
            let cap_local = self.new_temp(TyId::USIZE, span);
            let cap_place = CfgPlace { local: cap_local, projs: Vec::new(), ty: TyId::USIZE, span };
            self.push_stmt(
                span,
                CfgStmtKind::Call {
                    dst: Some(cap_place.clone()),
                    callee: CfgCallee::Builtin(CfgBuiltinOp::SlotPoolCap),
                    args: vec![CfgOperand::Copy(pool_place.clone())],
                    arg_spans: vec![actual_seq_expr.span],
                    recv_place: Some(pool_place.clone()),
                    recv_span: Some(actual_seq_expr.span),
                    is_mut_recv: false,
                    consumes_recv: false,
                },
            );
            let idx_local = self.new_temp(TyId::USIZE, span);
            let idx_place = CfgPlace { local: idx_local, projs: Vec::new(), ty: TyId::USIZE, span };
            self.push_stmt(span, CfgStmtKind::Assign { dst: idx_place.clone(), rvalue: CfgRvalue::Use(CfgOperand::Const(CfgConst::Int(0, TyId::USIZE))) });

            let header_bb = self.new_block();
            let body_bb = self.new_block();
            let alive_bb = self.new_block();
            let step_bb = self.new_block();
            let exit_bb = self.new_block();
            self.func.blocks[header_bb.0 as usize].is_loop_header = true;
            self.func.blocks[header_bb.0 as usize].loop_entry_depth = self.scopes.last().map(|s| s.depth);
            if is_par {
                self.func.par_for_regions.push(ParForRegion {
                    header_bb,
                    body_bb,
                    step_bb,
                    exit_bb,
                    idx_local,
                    end_local: cap_local,
                    chunk_size: 256,
                });
            }

            self.set_term(CfgTerm::Goto(header_bb));
            self.cur_bb = Some(header_bb);
            let cond = self.emit_bin_tmp(BinaryOp::Lt, CfgOperand::Copy(idx_place.clone()), CfgOperand::Copy(cap_place), TyId::BOOL, span);
            self.set_term(CfgTerm::Branch { cond, then_bb: body_bb, else_bb: exit_bb, span, is_user_if: false });

            self.loops.push(LoopTarget {
                label,
                continue_bb: step_bb,
                exit_bb,
                break_dst: None,
                break_origin_ty: TyId::VOID,
                scope_len_at_entry: self.scopes.len(),
                any_break: false,
            });

            self.cur_bb = Some(body_bb);
            let opt_tmp = self.new_temp(opt_gen_id_ty, span);
            let opt_place = CfgPlace { local: opt_tmp, projs: Vec::new(), ty: opt_gen_id_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Call {
                    dst: Some(opt_place.clone()),
                    callee: CfgCallee::Builtin(CfgBuiltinOp::SlotPoolIdAt(elem_ty)),
                    args: vec![CfgOperand::Copy(pool_place.clone()), CfgOperand::Copy(idx_place.clone())],
                    arg_spans: vec![actual_seq_expr.span, span],
                    recv_place: Some(pool_place),
                    recv_span: Some(actual_seq_expr.span),
                    is_mut_recv: false,
                    consumes_recv: false,
                },
            );
            let is_some_tmp = self.new_temp(TyId::BOOL, span);
            let is_some_place = CfgPlace { local: is_some_tmp, projs: Vec::new(), ty: TyId::BOOL, span };
            self.push_stmt(span, CfgStmtKind::Assign { dst: is_some_place.clone(), rvalue: CfgRvalue::IsNonNull(CfgOperand::Copy(opt_place.clone())) });
            self.set_term(CfgTerm::Branch { cond: CfgOperand::Copy(is_some_place), then_bb: alive_bb, else_bb: step_bb, span, is_user_if: false });

            self.cur_bb = Some(alive_bb);
            self.push_scope();
            let id_tmp = self.new_temp(gen_id_ty, pat.span);
            let id_place = CfgPlace { local: id_tmp, projs: Vec::new(), ty: gen_id_ty, span: pat.span };
            self.push_stmt(pat.span, CfgStmtKind::Assign { dst: id_place.clone(), rvalue: CfgRvalue::UnwrapOptional(CfgOperand::Copy(opt_place)) });
            self.lower_bind_pattern(pat, CfgOperand::Copy(id_place), gen_id_ty, false);
            self.lower_block_into(body, None, None);
            self.pop_scope_normal();
            if self.cur_bb.is_some() {
                self.set_term(CfgTerm::Goto(step_bb));
            }

            self.cur_bb = Some(step_bb);
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: idx_place.clone(),
                    rvalue: CfgRvalue::Binary {
                        op: BinaryOp::AddWrap,
                        lhs: CfgOperand::Copy(idx_place),
                        rhs: CfgOperand::Const(CfgConst::Int(1, TyId::USIZE)),
                    },
                },
            );
            self.set_term(CfgTerm::Goto(header_bb));

            self.loops.pop();
            self.cur_bb = Some(exit_bb);
            self.pop_scope_normal();
            return;
        }
        let seq_op = self.lower_expr_rvalue(actual_seq_expr, None);
        if self.cur_bb.is_none() {
            return;
        }
        let seq_ty = seq_op.ty(&self.func, self.env);
        let (elem_ty, fixed_len) = match self.env.get(seq_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Slice(el), .. } => (*el, None),
            Ty::Array { elem, len } => (*elem, Some(*len)),
            Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str => (TyId::U8, None),
            _ => (TyId::ERROR, None),
        };

        self.push_scope();
        let seq_local = self.new_temp(seq_ty, seq_expr.span);
        let seq_place = CfgPlace { local: seq_local, projs: Vec::new(), ty: seq_ty, span: seq_expr.span };
        self.push_stmt(seq_expr.span, CfgStmtKind::Assign { dst: seq_place.clone(), rvalue: CfgRvalue::Use(seq_op) });

        let len_local = self.new_temp(TyId::USIZE, seq_expr.span);
        let len_place = CfgPlace { local: len_local, projs: Vec::new(), ty: TyId::USIZE, span: seq_expr.span };
        if let Some(n) = fixed_len {
            self.push_stmt(seq_expr.span, CfgStmtKind::Assign { dst: len_place.clone(), rvalue: CfgRvalue::Use(CfgOperand::Const(CfgConst::Int(n as i128, TyId::USIZE))) });
        } else {
            let op = match self.env.get(seq_ty) {
                Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str => CfgBuiltinOp::StrLen,
                _ => CfgBuiltinOp::SliceLen,
            };
            self.push_stmt(
                seq_expr.span,
                CfgStmtKind::Call {
                    dst: Some(len_place.clone()),
                    callee: CfgCallee::Builtin(op),
                    args: vec![CfgOperand::Copy(seq_place.clone())],
                    arg_spans: vec![seq_expr.span],
                    recv_place: Some(seq_place.clone()),
                    recv_span: Some(seq_expr.span),
                    is_mut_recv: false,
                    consumes_recv: false,
                },
            );
        }

        let idx_local = self.new_temp(TyId::USIZE, span);
        let idx_place = CfgPlace { local: idx_local, projs: Vec::new(), ty: TyId::USIZE, span };
        self.push_stmt(span, CfgStmtKind::Assign { dst: idx_place.clone(), rvalue: CfgRvalue::Use(CfgOperand::Const(CfgConst::Int(0, TyId::USIZE))) });

        let header_bb = self.new_block();
        let body_bb = self.new_block();
        let step_bb = self.new_block();
        let exit_bb = self.new_block();
        self.func.blocks[header_bb.0 as usize].is_loop_header = true;
        self.func.blocks[header_bb.0 as usize].loop_entry_depth = self.scopes.last().map(|s| s.depth);
        if is_par {
            self.func.par_for_regions.push(ParForRegion {
                header_bb,
                body_bb,
                step_bb,
                exit_bb,
                idx_local,
                end_local: len_local,
                chunk_size: 256,
            });
        }

        self.set_term(CfgTerm::Goto(header_bb));
        self.cur_bb = Some(header_bb);

        let cond = self.emit_bin_tmp(BinaryOp::Lt, CfgOperand::Copy(idx_place.clone()), CfgOperand::Copy(len_place), TyId::BOOL, span);
        self.set_term(CfgTerm::Branch { cond, then_bb: body_bb, else_bb: exit_bb, span, is_user_if: false });

        self.loops.push(LoopTarget {
            label,
            continue_bb: step_bb,
            exit_bb,
            break_dst: None,
            break_origin_ty: TyId::VOID,
            scope_len_at_entry: self.scopes.len(),
            any_break: false,
        });

        self.cur_bb = Some(body_bb);
        self.push_scope();
        let mut elem_place = seq_place;
        elem_place.projs.push(CfgProj::Index {
            index: CfgOperand::Copy(idx_place.clone()),
            const_idx: None,
            key_sym: None,
            ty: elem_ty,
        });
        elem_place.ty = elem_ty;
        if let Some(is_mut) = borrow_mode {
            let bind_ty = self.table.get(pat.span).unwrap_or(elem_ty);
            let ref_tmp = self.new_temp(bind_ty, pat.span);
            let ref_place = CfgPlace { local: ref_tmp, projs: Vec::new(), ty: bind_ty, span: pat.span };
            self.push_stmt(
                pat.span,
                CfgStmtKind::Assign {
                    dst: ref_place.clone(),
                    rvalue: CfgRvalue::Borrow {
                        is_mut,
                        place: elem_place,
                    },
                },
            );
            self.lower_bind_pattern(pat, CfgOperand::Copy(ref_place), bind_ty, false);
        } else {
            self.lower_bind_pattern(pat, CfgOperand::Copy(elem_place), elem_ty, false);
        }
        self.lower_block_into(body, None, None);
        self.pop_scope_normal();
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(step_bb));
        }

        self.cur_bb = Some(step_bb);
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: idx_place.clone(),
                rvalue: CfgRvalue::Binary {
                    op: BinaryOp::AddWrap,
                    lhs: CfgOperand::Copy(idx_place),
                    rhs: CfgOperand::Const(CfgConst::Int(1, TyId::USIZE)),
                },
            },
        );
        self.set_term(CfgTerm::Goto(header_bb));

        self.loops.pop();
        self.cur_bb = Some(exit_bb);
        self.pop_scope_normal();
    }

    fn lower_loop(
        &mut self,
        label: Option<Symbol>,
        body: &Block,
        dst: Option<&CfgPlace>,
        expected_ty: Option<TyId>,
        _span: Span,
    ) -> bool {
        let header_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();
        self.func.blocks[header_bb.0 as usize].is_loop_header = true;
        self.func.blocks[header_bb.0 as usize].loop_entry_depth = self.scopes.last().map(|s| s.depth);

        self.set_term(CfgTerm::Goto(header_bb));
        self.func.blocks[header_bb.0 as usize].term = CfgTerm::Goto(body_bb);

        self.loops.push(LoopTarget {
            label,
            continue_bb: header_bb,
            exit_bb,
            break_dst: dst.cloned(),
            break_origin_ty: expected_ty.unwrap_or(TyId::VOID),
            scope_len_at_entry: self.scopes.len(),
            any_break: false,
        });

        self.cur_bb = Some(body_bb);
        self.lower_block_into(body, None, None);
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(header_bb));
        }

        let target = self.loops.pop().unwrap();
        if target.any_break {
            self.cur_bb = Some(exit_bb);
            true
        } else {
            self.cur_bb = None;
            false
        }
    }

    fn lower_expr_into(&mut self, expr: &Expr, dst: &CfgPlace, expected_ty: TyId) {
        let op = self.lower_expr_rvalue(expr, Some(expected_ty));
        if self.cur_bb.is_none() {
            return;
        }
        let cop = self.coerce_operand(op, dst.ty, expr.span);
        self.push_stmt(
            expr.span,
            CfgStmtKind::Assign {
                dst: dst.clone(),
                rvalue: CfgRvalue::Use(cop),
            },
        );
    }

    fn operand_for_local(&self, place: CfgPlace) -> CfgOperand {
        if self.env.is_linear(place.ty) {
            CfgOperand::Move(place)
        } else {
            CfgOperand::Copy(place)
        }
    }

    fn lower_expr_rvalue(&mut self, expr: &Expr, expected: Option<TyId>) -> CfgOperand {
        let ty = self.expr_ty(expr, expected.unwrap_or(TyId::ERROR));
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(loc) = self.lookup_local(*sym) {
                    let lty = self.func.locals[loc.0 as usize].ty;
                    let p = CfgPlace { local: loc, projs: Vec::new(), ty: lty, span: expr.span };
                    return self.operand_for_place(&p);
                }
                if let Some(&(env_loc, fidx, ptr_ty, inner_ty)) = self.captured_aliases.get(sym) {
                    let p = CfgPlace {
                        local: env_loc,
                        projs: vec![
                            CfgProj::Deref,
                            CfgProj::TupleField { index: fidx, ty: ptr_ty },
                            CfgProj::Deref,
                        ],
                        ty: inner_ty,
                        span: expr.span,
                    };
                    return self.operand_for_place(&p);
                }
                if let Some(target) = self.prog.lookup_in_module(self.module, *sym) {
                    return self.lower_scope_target_const(target, expected.unwrap_or(ty));
                }
                CfgOperand::Const(CfgConst::Void)
            }
            ExprKind::SelfVal => {
                if let Some((env_loc, fidx, ptr_ty, inner_ty)) = self.captured_self {
                    let p = CfgPlace {
                        local: env_loc,
                        projs: vec![
                            CfgProj::Deref,
                            CfgProj::TupleField { index: fidx, ty: ptr_ty },
                            CfgProj::Deref,
                        ],
                        ty: inner_ty,
                        span: expr.span,
                    };
                    return self.operand_for_place(&p);
                }
                if let Some(&self_loc) = self.func.params.first() {
                    let lty = self.func.locals[self_loc.0 as usize].ty;
                    let p = CfgPlace { local: self_loc, projs: Vec::new(), ty: lty, span: expr.span };
                    return self.operand_for_place(&p);
                }
                CfgOperand::Const(CfgConst::Void)
            }
            ExprKind::IntLit { raw, .. } => {
                let n = parse_int_literal(self.interner.get(*raw)).unwrap_or(0) as i128;
                let final_ty = self.pick_concrete_int_ty(ty, expected);
                CfgOperand::Const(CfgConst::Int(n, final_ty))
            }
            ExprKind::FloatLit { raw, .. } => {
                let final_ty = if self.env.is_float(ty) {
                    ty
                } else if expected.is_some_and(|e| self.env.is_float(e)) {
                    expected.unwrap()
                } else {
                    TyId::F64
                };
                CfgOperand::Const(CfgConst::Float(self.interner.get(*raw).to_owned(), final_ty))
            }
            ExprKind::BoolLit(b) => CfgOperand::Const(CfgConst::Bool(*b)),
            ExprKind::CharLit(c) => CfgOperand::Const(CfgConst::Char(*c)),
            ExprKind::ByteCharLit(b) => CfgOperand::Const(CfgConst::Int(*b as i128, TyId::U8)),
            ExprKind::StringLit(s) | ExprKind::RawStringLit(s) => CfgOperand::Const(CfgConst::Str(s.clone())),
            ExprKind::ByteStringLit(bs) => CfgOperand::Const(CfgConst::Bytes(bs.clone())),
            ExprKind::Null => CfgOperand::Const(CfgConst::Null(if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::ERROR) })),
            ExprKind::EnumLit(v) => self.lower_enum_or_error_lit(*v, if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::ERROR) }),
            ExprKind::Paren(inner) => self.lower_expr_rvalue(inner, expected),
            ExprKind::ComptimeExpr(inner) => {
                if let Some(cv) = self.table.get_comptime_val(expr.span).cloned() {
                    return self.lower_ct_val(&cv, if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::ERROR) }, expr.span);
                }
                self.lower_expr_rvalue(inner, expected)
            }
            ExprKind::Field { base, field } => {
                // Use the expression's own (enum) type, not the expected type: for `return E.V;`
                // in a `-> ?E` function the expected type is `?E`, and typing the constant as `?E`
                // skipped the `WrapOptional` coercion (raw tag stored into an optional slot).
                let enum_ty = if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::ERROR) };
                if let Some(const_op) = self.try_lower_assoc_const_or_enum(base, *field, enum_ty) {
                    return const_op;
                }
                if let Some(crate::comptime::CtVal::Struct { fields, .. }) = self.table.get_comptime_val(base.span).cloned()
                    && let Some(fv) = fields.get(&field.sym)
                {
                    return self.lower_ct_val(fv, if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::ERROR) }, expr.span);
                }
                if self.interner.get(field.sym) == "len"
                    && let Some(len_op) = self.try_lower_len_field(base, expr.span)
                {
                    return len_op;
                }
                let p = self.lower_place(expr);
                self.operand_for_place(&p)
            }
            ExprKind::Deref(_) | ExprKind::Index { .. } => {
                let p = self.lower_place(expr);
                self.operand_for_place(&p)
            }
            ExprKind::Unary { op, expr: inner } => match op {
                UnaryOp::Neg => {
                    if let ExprKind::IntLit { raw, .. } = &inner.kind {
                        let u = parse_int_literal(self.interner.get(*raw)).unwrap_or(0);
                        let n = if u == (i128::MAX as u128) + 1 { i128::MIN } else { -(u as i128) };
                        let final_ty = self.pick_concrete_int_ty(ty, expected);
                        return CfgOperand::Const(CfgConst::Int(n, final_ty));
                    }
                    let op_val = self.lower_expr_rvalue(inner, Some(ty));
                    if self.cur_bb.is_none() {
                        return CfgOperand::Const(CfgConst::Void);
                    }
                    self.emit_unary_tmp(UnaryOp::Neg, op_val, ty, expr.span)
                }
                UnaryOp::Not | UnaryOp::BitNot => {
                    let op_val = self.lower_expr_rvalue(inner, Some(ty));
                    if self.cur_bb.is_none() {
                        return CfgOperand::Const(CfgConst::Void);
                    }
                    self.emit_unary_tmp(*op, op_val, ty, expr.span)
                }
                UnaryOp::Borrow | UnaryOp::BorrowMut => {
                    let is_mut = *op == UnaryOp::BorrowMut;
                    let place = self.lower_place(inner);
                    if self.cur_bb.is_none() {
                        return CfgOperand::Const(CfgConst::Void);
                    }
                    let res_ty = if ty != TyId::ERROR {
                        ty
                    } else {
                        place.ty
                    };
                    let tmp = self.new_temp(res_ty, expr.span);
                    let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: res_ty, span: expr.span };
                    self.push_stmt(
                        expr.span,
                        CfgStmtKind::Assign {
                            dst: dst.clone(),
                            rvalue: CfgRvalue::Borrow { is_mut, place },
                        },
                    );
                    CfgOperand::Copy(dst)
                }
                UnaryOp::Move => {
                    let place = self.lower_place(inner);
                    CfgOperand::Move(place)
                }
                UnaryOp::Copy => {
                    let place = self.lower_place(inner);
                    CfgOperand::Copy(place)
                }
                UnaryOp::Try => self.lower_try(inner, expr.span, ty),
            },
            ExprKind::Binary { op, lhs, rhs } => {
                if matches!(op, BinaryOp::And | BinaryOp::Or) && ty == TyId::BOOL {
                    return self.lower_short_circuit(*op, lhs, rhs, expr.span);
                }
                let lhs_ty = self.expr_ty(lhs, TyId::ERROR);
                let rhs_ty = self.expr_ty(rhs, TyId::ERROR);
                let op_ty = if lhs_ty != TyId::ERROR {
                    lhs_ty
                } else if rhs_ty != TyId::ERROR {
                    rhs_ty
                } else if !matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq) {
                    self.pick_concrete_int_ty(ty, expected)
                } else {
                    expected.unwrap_or(TyId::ERROR)
                };
                let l = self.lower_expr_rvalue(lhs, Some(op_ty));
                let l_concrete = if l.ty(&self.func, self.env) == TyId::COMPTIME_INT {
                    self.pick_concrete_int_ty(op_ty, expected)
                } else {
                    l.ty(&self.func, self.env)
                };
                let r = self.lower_expr_rvalue(rhs, Some(l_concrete));
                let l = self.coerce_operand(l, l_concrete, lhs.span);
                let r = self.coerce_operand(r, l_concrete, rhs.span);
                let out_ty = if matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq) {
                    TyId::BOOL
                } else if matches!(
                    op,
                    BinaryOp::AddCheck | BinaryOp::SubCheck | BinaryOp::MulCheck | BinaryOp::DivCheck | BinaryOp::ModCheck
                ) {
                    ty
                } else {
                    l_concrete
                };
                self.emit_bin_tmp(*op, l, r, out_ty, expr.span)
            }
            ExprKind::Slice { base, lo, hi } => {
                let b_op = self.lower_expr_rvalue(base, None);
                let l_op = lo.as_ref().map(|x| self.lower_expr_rvalue(x, Some(TyId::USIZE)));
                let h_op = hi.as_ref().map(|x| self.lower_expr_rvalue(x, Some(TyId::USIZE)));
                let tmp = self.new_temp(ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: expr.span };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Slice { base: b_op, lo: l_op, hi: h_op },
                    },
                );
                CfgOperand::Copy(dst)
            }
            ExprKind::If { is_comptime, data } => {
                let res_ty = if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::VOID) };
                let tmp = self.new_temp(res_ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: res_ty, span: expr.span };
                self.lower_if(*is_comptime, data, Some(&dst), Some(res_ty));
                self.operand_for_local(dst)
            }
            ExprKind::Match { scrutinee, arms } => {
                let res_ty = if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::VOID) };
                let tmp = self.new_temp(res_ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: res_ty, span: expr.span };
                self.lower_match(scrutinee, arms, Some(&dst), Some(res_ty), expr.span);
                self.operand_for_local(dst)
            }
            ExprKind::Loop { label, body } => {
                let res_ty = if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::VOID) };
                let tmp = self.new_temp(res_ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: res_ty, span: expr.span };
                let _ = self.lower_loop(label.map(|l| l.sym), body, Some(&dst), Some(res_ty), expr.span);
                self.operand_for_local(dst)
            }
            ExprKind::Orelse { lhs, rhs } => self.lower_orelse(lhs, rhs, ty, expr.span),
            ExprKind::Catch { lhs, err_bind, rhs } => self.lower_catch(lhs, *err_bind, rhs, ty, expr.span),
            ExprKind::Call { callee, args } => self.lower_call(callee, args, ty, expr.span),
            ExprKind::GenericApply { base, .. } => self.lower_expr_rvalue(base, expected),
            ExprKind::Tuple(es) => {
                let elem_tys = match self.env.get(ty) {
                    Ty::Tuple(ts) => ts.clone(),
                    _ => Vec::new(),
                };
                let mut ops = Vec::with_capacity(es.len());
                for (i, e) in es.iter().enumerate() {
                    let et = elem_tys.get(i).copied();
                    let op = self.lower_expr_rvalue(e, et);
                    let cop = if let Some(t) = et { self.coerce_operand(op, t, e.span) } else { op };
                    ops.push(cop);
                }
                let tmp = self.new_temp(ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: expr.span };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Tuple,
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            ExprKind::ArrayLit(es) => {
                let elem_ty = match self.env.get(ty) {
                    Ty::Array { elem, .. } => *elem,
                    _ => TyId::ERROR,
                };
                let mut ops = Vec::with_capacity(es.len());
                for e in es {
                    let op = self.lower_expr_rvalue(e, if elem_ty != TyId::ERROR { Some(elem_ty) } else { None });
                    let cop = if elem_ty != TyId::ERROR { self.coerce_operand(op, elem_ty, e.span) } else { op };
                    ops.push(cop);
                }
                let tmp = self.new_temp(ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: expr.span };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Array(elem_ty),
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            ExprKind::ArrayRepeat { val, count } => {
                let (elem_ty, len) = match self.env.get(ty) {
                    Ty::Array { elem, len } => (*elem, *len),
                    _ => (TyId::ERROR, 0),
                };
                let vop = self.lower_expr_rvalue(val, if elem_ty != TyId::ERROR { Some(elem_ty) } else { None });
                let cop = if elem_ty != TyId::ERROR { self.coerce_operand(vop, elem_ty, val.span) } else { vop };
                let _ = self.eval_simple_const_int(count);
                let tmp = self.new_temp(ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: expr.span };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::ArrayRepeat { elem: elem_ty, count: len },
                            elems: vec![cop],
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            ExprKind::AnonStructLit(fields) | ExprKind::NamedStructLit { fields, .. } => {
                self.lower_struct_lit(ty, fields, expr.span)
            }
            ExprKind::Is { expr: inner, pat } => {
                let sop = self.lower_expr_rvalue(inner, None);
                if self.cur_bb.is_none() {
                    return CfgOperand::Const(CfgConst::Bool(false));
                }
                let st = sop.ty(&self.func, self.env);
                let stmp = self.new_temp(st, inner.span);
                let splace = CfgPlace { local: stmp, projs: Vec::new(), ty: st, span: inner.span };
                self.push_stmt(inner.span, CfgStmtKind::Assign { dst: splace.clone(), rvalue: CfgRvalue::Use(sop) });
                self.lower_pat_test(&splace, pat, st, expr.span)
            }
            ExprKind::Builtin { name, args } => self.lower_builtin_expr(*name, args, ty, expr.span),
            ExprKind::ComptimeBlock(b) => {
                let res_ty = if ty != TyId::ERROR { ty } else { expected.unwrap_or(TyId::VOID) };
                if let Some(cv) = self.table.get_comptime_val(expr.span).cloned() {
                    return self.lower_ct_val(&cv, res_ty, expr.span);
                }
                let tmp = self.new_temp(res_ty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: res_ty, span: expr.span };
                self.lower_block_into(b, Some(&dst), Some(res_ty));
                self.operand_for_local(dst)
            }
            ExprKind::Closure { params, body } => {
                self.lower_closure(params, body, ty, expected, expr.span)
            }
            ExprKind::Asm { template, operands } => {
                let mut inputs = Vec::new();
                for op in operands {
                    match &op.kind {
                        ast::AsmOperandKind::Named { dir, constraint, arg } => {
                            let d = self.interner.get(dir.sym).to_owned();
                            let c = match &constraint.kind {
                                ExprKind::StringLit(s) | ExprKind::RawStringLit(s) => s.clone(),
                                _ => String::new(),
                            };
                            if let Some(a) = arg {
                                let v = self.lower_expr_rvalue(a, None);
                                inputs.push((format!("{d}:{c}"), v));
                            }
                        }
                        ast::AsmOperandKind::Expr(e) => {
                            let v = self.lower_expr_rvalue(e, None);
                            inputs.push(("expr".into(), v));
                        }
                    }
                }
                let tmp = self.new_temp(TyId::VOID, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: TyId::VOID, span: expr.span };
                self.push_stmt(
                    expr.span,
                    CfgStmtKind::Assign {
                        dst,
                        rvalue: CfgRvalue::InlineAsm {
                            template: template.clone(),
                            inputs,
                        },
                    },
                );
                CfgOperand::Const(CfgConst::Void)
            }
            ExprKind::ReturnVal | ExprKind::Error => CfgOperand::Const(CfgConst::Void),
        }
    }

    fn lower_struct_lit(&mut self, ty: TyId, fields: &[ast::FieldInit], span: Span) -> CfgOperand {
        match self.env.get(ty).clone() {
            Ty::Struct { def, .. } => {
                let meta = self.env.structs.get(&def).cloned();
                let mut ops = Vec::new();
                if let Some(m) = meta {
                    for mf in &m.fields {
                        if let Some(finit) = fields.iter().find(|f| f.name.is_some_and(|n| n.sym == mf.name)) {
                            let op = self.lower_expr_rvalue(&finit.value, Some(mf.ty));
                            ops.push(self.coerce_operand(op, mf.ty, finit.value.span));
                        } else {
                            ops.push(CfgOperand::Const(CfgConst::Null(mf.ty)));
                        }
                    }
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Struct(def),
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            Ty::Pool { def, .. } => {
                let is_slot_pool = self.pool_row_is_slot.unwrap_or(true);
                let meta = self.env.pools.get(&def).cloned();
                let mut ops = Vec::new();
                if let Some(m) = meta {
                    for col in &m.columns {
                        if let Some(finit) = fields.iter().find(|f| f.name.is_some_and(|n| n.sym == col.name)) {
                            let cty = self.raw_ty(finit.value.span);
                            let hint = if cty != TyId::ERROR { cty } else { col.ty };
                            let op = self.lower_expr_rvalue(&finit.value, Some(hint));
                            let target_ty = if !matches!(self.env.get(col.ty), Ty::Builtin(BuiltinKind::Link, _))
                                && !matches!(self.env.get(col.ty), Ty::Optional(inner) if matches!(self.env.get(*inner), Ty::Builtin(BuiltinKind::Link, _)))
                            {
                                col.ty
                            } else {
                                op.ty(&self.func, self.env)
                            };
                            ops.push(self.coerce_operand(op, target_ty, finit.value.span));
                        }
                    }
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::PoolRow { def, is_slot_pool },
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            Ty::AnonStruct(fs) => {
                let names: Vec<Symbol> = fs.iter().map(|(n, _)| *n).collect();
                let mut ops = Vec::new();
                for (name, fty) in fs {
                    if let Some(finit) = fields.iter().find(|f| f.name.is_some_and(|n| n.sym == name)) {
                        let op = self.lower_expr_rvalue(&finit.value, Some(fty));
                        ops.push(self.coerce_operand(op, fty, finit.value.span));
                    }
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::AnonStruct(names),
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            Ty::Bitfield(def) => {
                let meta = self.env.bitfields.get(&def).cloned();
                let mut ops = Vec::new();
                if let Some(m) = meta {
                    for mb in &m.members {
                        if let Some(finit) = fields.iter().find(|f| f.name.is_some_and(|n| n.sym == mb.name)) {
                            let op = self.lower_expr_rvalue(&finit.value, Some(mb.ty));
                            ops.push(self.coerce_operand(op, mb.ty, finit.value.span));
                        } else {
                            ops.push(CfgOperand::Const(CfgConst::Int(0, mb.ty)));
                        }
                    }
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Bitfield(def),
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            Ty::Vec { elem, len } => {
                let mut ops = Vec::with_capacity(fields.len());
                for finit in fields {
                    let op = self.lower_expr_rvalue(&finit.value, Some(elem));
                    ops.push(self.coerce_operand(op, elem, finit.value.span));
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Vec { elem, len },
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            _ => {
                for f in fields {
                    let _ = self.lower_expr_rvalue(&f.value, None);
                }
                CfgOperand::Const(CfgConst::Void)
            }
        }
    }

    fn lower_place(&mut self, expr: &Expr) -> CfgPlace {
        let expr_ty = self.expr_ty(expr, TyId::ERROR);
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(loc) = self.lookup_local(*sym) {
                    let lty = self.func.locals[loc.0 as usize].ty;
                    return CfgPlace { local: loc, projs: Vec::new(), ty: lty, span: expr.span };
                }
                if let Some(&(env_loc, fidx, ptr_ty, inner_ty)) = self.captured_aliases.get(sym) {
                    return CfgPlace {
                        local: env_loc,
                        projs: vec![
                            CfgProj::Deref,
                            CfgProj::TupleField { index: fidx, ty: ptr_ty },
                            CfgProj::Deref,
                        ],
                        ty: inner_ty,
                        span: expr.span,
                    };
                }
                if let Some(target) = self.prog.lookup_in_module(self.module, *sym) {
                    let cop = self.lower_scope_target_const(target, expr_ty);
                    let cty = cop.ty(&self.func, self.env);
                    let tmp = self.new_temp(if cty != TyId::ERROR { cty } else { expr_ty }, expr.span);
                    let place = CfgPlace { local: tmp, projs: Vec::new(), ty: if cty != TyId::ERROR { cty } else { expr_ty }, span: expr.span };
                    self.push_stmt(expr.span, CfgStmtKind::Assign { dst: place.clone(), rvalue: CfgRvalue::Use(cop) });
                    return place;
                }
                let dummy = self.new_temp(expr_ty, expr.span);
                CfgPlace { local: dummy, projs: Vec::new(), ty: expr_ty, span: expr.span }
            }
            ExprKind::SelfVal => {
                if let Some((env_loc, fidx, ptr_ty, inner_ty)) = self.captured_self {
                    return CfgPlace {
                        local: env_loc,
                        projs: vec![
                            CfgProj::Deref,
                            CfgProj::TupleField { index: fidx, ty: ptr_ty },
                            CfgProj::Deref,
                        ],
                        ty: inner_ty,
                        span: expr.span,
                    };
                }
                if let Some(&loc) = self.func.params.first() {
                    let lty = self.func.locals[loc.0 as usize].ty;
                    return CfgPlace { local: loc, projs: Vec::new(), ty: lty, span: expr.span };
                }
                let dummy = self.new_temp(expr_ty, expr.span);
                CfgPlace { local: dummy, projs: Vec::new(), ty: expr_ty, span: expr.span }
            }
            ExprKind::Paren(inner) => self.lower_place(inner),
            ExprKind::Deref(inner) => {
                let mut bp = self.lower_place(inner);
                let deref_ty = match self.env.get(bp.ty) {
                    Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. }
                    | Ty::RawPtr { inner: t, .. } => *t,
                    _ => expr_ty,
                };
                bp.projs.push(CfgProj::Deref);
                bp.ty = deref_ty;
                bp.span = expr.span;
                bp
            }
            ExprKind::Field { base, field } => {
                let mut bp = self.lower_place(base);
                let base_inner = match self.env.get(bp.ty) {
                    Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
                    _ => bp.ty,
                };
                match self.env.get(base_inner).clone() {
                    Ty::Struct { def, .. } => {
                        let (fidx, fty) = self
                            .env
                            .structs
                            .get(&def)
                            .and_then(|m| m.fields.iter().enumerate().find(|(_, f)| f.name == field.sym))
                            .map(|(i, f)| (i, f.ty))
                            .unwrap_or((0, expr_ty));
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: fty });
                        bp.ty = expr_ty;
                    }
                    Ty::Pool { def, .. } => {
                        let (fidx, fty) = self
                            .env
                            .pools
                            .get(&def)
                            .and_then(|m| m.columns.iter().enumerate().find(|(_, c)| c.name == field.sym))
                            .map(|(i, c)| (i, c.ty))
                            .unwrap_or((0, expr_ty));
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: fty });
                        bp.ty = expr_ty;
                    }
                    Ty::AnonStruct(fields) => {
                        let (fidx, fty) = fields
                            .iter()
                            .enumerate()
                            .find(|(_, (n, _))| *n == field.sym)
                            .map(|(i, (_, t))| (i, *t))
                            .unwrap_or((0, expr_ty));
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: fty });
                        bp.ty = expr_ty;
                    }
                    Ty::Tuple(_) => {
                        let idx = self.interner.get(field.sym).parse::<usize>().unwrap_or(0);
                        bp.projs.push(CfgProj::TupleField { index: idx, ty: expr_ty });
                        bp.ty = expr_ty;
                    }
                    Ty::Bitfield(def) => {
                        let (fidx, fty) = self
                            .env
                            .bitfields
                            .get(&def)
                            .and_then(|m| m.members.iter().enumerate().find(|(_, mb)| mb.name == field.sym))
                            .map(|(i, mb)| (i, mb.ty))
                            .unwrap_or((0, expr_ty));
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: fty });
                        bp.ty = expr_ty;
                    }
                    Ty::Vec { elem, .. } => {
                        let fname = self.interner.get(field.sym).to_owned();
                        if fname.len() >= 2 {
                            // Multi-lane SIMD swizzle is an rvalue: materialize into a fresh temp vector!
                            if matches!(
                                self.env.get(bp.ty),
                                Ty::Borrow { inner: crate::types::BorrowInner::Single(_), .. }
                            ) {
                                bp.projs.push(CfgProj::Deref);
                                bp.ty = base_inner;
                            }
                            let mut elems = Vec::with_capacity(fname.len());
                            for ch in fname.chars() {
                                let fidx = match ch {
                                    'x' | 'r' => 0,
                                    'y' | 'g' => 1,
                                    'z' | 'b' => 2,
                                    _ => 3,
                                };
                                let mut lane_p = bp.clone();
                                lane_p.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: elem });
                                lane_p.ty = elem;
                                elems.push(self.operand_for_place(&lane_p));
                            }
                            let out_ty = self.env.intern(Ty::Vec { elem, len: fname.len() as u8 });
                            let tmp = self.new_temp(out_ty, expr.span);
                            let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: out_ty, span: expr.span };
                            self.push_stmt(
                                expr.span,
                                CfgStmtKind::Assign {
                                    dst: dst.clone(),
                                    rvalue: CfgRvalue::Aggregate {
                                        kind: AggregateKind::Vec { elem, len: fname.len() as u8 },
                                        elems,
                                    },
                                },
                            );
                            return dst;
                        }
                        let fidx = match fname.as_str() {
                            "x" | "r" => 0,
                            "y" | "g" => 1,
                            "z" | "b" => 2,
                            "w" | "a" => 3,
                            _ => 0,
                        };
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: elem });
                        bp.ty = expr_ty;
                    }
                    Ty::Builtin(BuiltinKind::Env, _) => {
                        let fname = self.interner.get(field.sym);
                        let fidx = match fname {
                            "alloc" => 0,
                            "fs" => 1,
                            "net" => 2,
                            "clock" => 3,
                            "io" => 4,
                            "args" => 5,
                            _ => 0,
                        };
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: fidx, ty: expr_ty });
                        bp.ty = expr_ty;
                    }
                    Ty::Builtin(bk @ (BuiltinKind::ArenaPool | BuiltinKind::SlotPool), args) => {
                        let (cidx, cty) = if let Some(GenericVal::Ty(pt)) = args.first()
                            && let Ty::Pool { def: pdef, .. } = self.env.get(*pt)
                            && let Some(pm) = self.env.pools.get(pdef)
                            && let Some((ci, cm)) = pm.columns.iter().enumerate().find(|(_, c)| c.name == field.sym)
                        {
                            (ci, cm.ty)
                        } else {
                            (0, expr_ty)
                        };
                        let col_elem_ty = match self.env.get(expr_ty) {
                            Ty::Borrow { inner: crate::types::BorrowInner::Slice(el), .. } => *el,
                            _ => cty,
                        };
                        bp.projs.push(CfgProj::PoolCol {
                            col: field.sym,
                            col_idx: cidx,
                            is_slot_pool: bk == BuiltinKind::SlotPool,
                            elem_ty: col_elem_ty,
                        });
                        bp.ty = expr_ty;
                    }
                    _ => {
                        bp.projs.push(CfgProj::Field { name: field.sym, field_idx: 0, ty: expr_ty });
                        bp.ty = expr_ty;
                    }
                }
                bp.span = expr.span;
                bp
            }
            ExprKind::Index { base, index } => {
                let mut bp = self.lower_place(base);
                let idx_ty = self.expr_ty(index, TyId::USIZE);
                let idx_op = self.lower_expr_rvalue(index, Some(idx_ty));
                let const_idx = self.eval_simple_const_int(index);
                let key_sym = if let ExprKind::Ident(s) = &index.kind { Some(*s) } else { None };
                let elem_ty = match bp.projs.last() {
                    Some(CfgProj::PoolCol { elem_ty, .. }) => *elem_ty,
                    _ => match self.env.get(bp.ty) {
                        Ty::Array { elem, .. }
                        | Ty::Borrow { inner: crate::types::BorrowInner::Slice(elem), .. } => *elem,
                        Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str => TyId::U8,
                        Ty::Builtin(BuiltinKind::GpuBuf, gargs) => match gargs.first() {
                            Some(GenericVal::Ty(el)) => *el,
                            _ => expr_ty,
                        },
                        _ => expr_ty,
                    },
                };
                bp.projs.push(CfgProj::Index {
                    index: idx_op,
                    const_idx,
                    key_sym,
                    ty: elem_ty,
                });
                bp.ty = elem_ty;
                bp.span = expr.span;
                bp
            }
            _ => {
                let val_op = self.lower_expr_rvalue(expr, None);
                let vty = val_op.ty(&self.func, self.env);
                let tmp = self.new_temp(vty, expr.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: vty, span: expr.span };
                self.push_stmt(expr.span, CfgStmtKind::Assign { dst: dst.clone(), rvalue: CfgRvalue::Use(val_op) });
                dst
            }
        }
    }

    fn operand_for_place(&self, p: &CfgPlace) -> CfgOperand {
        if self.env.is_linear(p.ty) {
            CfgOperand::Move(p.clone())
        } else {
            CfgOperand::Copy(p.clone())
        }
    }

    fn emit_bin_tmp(&mut self, op: BinaryOp, lhs: CfgOperand, rhs: CfgOperand, ty: TyId, span: Span) -> CfgOperand {
        let tmp = self.new_temp(ty, span);
        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: dst.clone(),
                rvalue: CfgRvalue::Binary { op, lhs, rhs },
            },
        );
        CfgOperand::Copy(dst)
    }

    fn emit_unary_tmp(&mut self, op: UnaryOp, operand: CfgOperand, ty: TyId, span: Span) -> CfgOperand {
        let tmp = self.new_temp(ty, span);
        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: dst.clone(),
                rvalue: CfgRvalue::Unary { op, operand },
            },
        );
        CfgOperand::Copy(dst)
    }

    fn lower_short_circuit(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr, span: Span) -> CfgOperand {
        let res = self.new_temp(TyId::BOOL, span);
        let dst = CfgPlace { local: res, projs: Vec::new(), ty: TyId::BOOL, span };
        let l = self.lower_expr_rvalue(lhs, Some(TyId::BOOL));
        if self.cur_bb.is_none() {
            return CfgOperand::Copy(dst);
        }
        let rhs_bb = self.new_block();
        let short_bb = self.new_block();
        let merge_bb = self.new_block();

        let (then_bb, else_bb, short_val) = if op == BinaryOp::And {
            (rhs_bb, short_bb, false)
        } else {
            (short_bb, rhs_bb, true)
        };
        self.set_term(CfgTerm::Branch {
            cond: l,
            then_bb,
            else_bb,
            span,
            is_user_if: false,
        });

        self.cur_bb = Some(short_bb);
        self.push_stmt(span, CfgStmtKind::Assign { dst: dst.clone(), rvalue: CfgRvalue::Use(CfgOperand::Const(CfgConst::Bool(short_val))) });
        self.set_term(CfgTerm::Goto(merge_bb));

        self.cur_bb = Some(rhs_bb);
        let r = self.lower_expr_rvalue(rhs, Some(TyId::BOOL));
        if self.cur_bb.is_some() {
            self.push_stmt(rhs.span, CfgStmtKind::Assign { dst: dst.clone(), rvalue: CfgRvalue::Use(r) });
            self.set_term(CfgTerm::Goto(merge_bb));
        }

        self.cur_bb = Some(merge_bb);
        CfgOperand::Copy(dst)
    }

    fn lower_try(&mut self, inner: &Expr, span: Span, ok_ty: TyId) -> CfgOperand {
        let eu_ty = self.raw_ty(inner.span);
        let eu_op = self.lower_expr_rvalue(inner, if eu_ty != TyId::ERROR { Some(eu_ty) } else { None });
        if self.cur_bb.is_none() {
            return CfgOperand::Const(CfgConst::Void);
        }
        let actual_eu_ty = eu_op.ty(&self.func, self.env);
        let eu_tmp = self.new_temp(actual_eu_ty, inner.span);
        let eu_place = CfgPlace { local: eu_tmp, projs: Vec::new(), ty: actual_eu_ty, span: inner.span };
        self.push_stmt(inner.span, CfgStmtKind::Assign { dst: eu_place.clone(), rvalue: CfgRvalue::Use(eu_op) });

        let is_err_tmp = self.new_temp(TyId::BOOL, span);
        let is_err_place = CfgPlace { local: is_err_tmp, projs: Vec::new(), ty: TyId::BOOL, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: is_err_place.clone(),
                rvalue: CfgRvalue::IsError(CfgOperand::Copy(eu_place.clone())),
            },
        );

        let err_bb = self.new_block();
        let ok_bb = self.new_block();
        self.func.blocks[err_bb.0 as usize].is_cleanup = true;
        self.func.blocks[err_bb.0 as usize].error_site_span = Some(span);

        self.set_term(CfgTerm::Branch {
            cond: CfgOperand::Copy(is_err_place),
            then_bb: err_bb,
            else_bb: ok_bb,
            span,
            is_user_if: false,
        });

        // Build error cleanup & return path.
        self.cur_bb = Some(err_bb);
        self.cleanup_depth += 1;
        self.active_err_span = Some(span);

        let err_ext_ty = self.enclosing_err_set_ty();
        let ext_tmp = self.new_temp(err_ext_ty, span);
        let ext_place = CfgPlace { local: ext_tmp, projs: Vec::new(), ty: err_ext_ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: ext_place.clone(),
                rvalue: CfgRvalue::UnwrapErrorErr(CfgOperand::Copy(eu_place.clone())),
            },
        );
        let wrapped_err = self.coerce_error_into_union(CfgOperand::Copy(ext_place), span, self.func.sig.ret);

        self.emit_unwind_to_scope(0, true, Some(wrapped_err.clone()));
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Return {
                value: Some(wrapped_err),
                span,
            });
        }
        self.active_err_span = None;
        self.cleanup_depth -= 1;

        // Continue normal path in ok_bb.
        self.cur_bb = Some(ok_bb);
        let final_ok_ty = if ok_ty != TyId::ERROR {
            ok_ty
        } else if let Ty::ErrorUnion { ok, .. } = self.env.get(actual_eu_ty) {
            *ok
        } else {
            TyId::VOID
        };
        let ok_tmp = self.new_temp(final_ok_ty, span);
        let ok_dst = CfgPlace { local: ok_tmp, projs: Vec::new(), ty: final_ok_ty, span };
        let src_op = if self.env.is_linear(final_ok_ty) {
            CfgOperand::Move(eu_place)
        } else {
            CfgOperand::Copy(eu_place)
        };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: ok_dst.clone(),
                rvalue: CfgRvalue::UnwrapErrorOk(src_op),
            },
        );
        self.operand_for_local(ok_dst)
    }

    fn lower_orelse(&mut self, lhs: &Expr, rhs: &CatchBody, ty: TyId, span: Span) -> CfgOperand {
        let opt_ty = self.raw_ty(lhs.span);
        let lop = self.lower_expr_rvalue(lhs, if opt_ty != TyId::ERROR { Some(opt_ty) } else { None });
        if self.cur_bb.is_none() {
            return CfgOperand::Const(CfgConst::Void);
        }
        let actual_opt = lop.ty(&self.func, self.env);
        let opt_tmp = self.new_temp(actual_opt, lhs.span);
        let opt_place = CfgPlace { local: opt_tmp, projs: Vec::new(), ty: actual_opt, span: lhs.span };
        self.push_stmt(lhs.span, CfgStmtKind::Assign { dst: opt_place.clone(), rvalue: CfgRvalue::Use(lop) });

        let res_ty = if ty != TyId::ERROR { ty } else { match self.env.get(actual_opt) { Ty::Optional(i) => *i, _ => TyId::ERROR } };
        let res_tmp = self.new_temp(res_ty, span);
        let res_dst = CfgPlace { local: res_tmp, projs: Vec::new(), ty: res_ty, span };

        let cond_tmp = self.new_temp(TyId::BOOL, span);
        let cond_dst = CfgPlace { local: cond_tmp, projs: Vec::new(), ty: TyId::BOOL, span };
        self.push_stmt(span, CfgStmtKind::Assign { dst: cond_dst.clone(), rvalue: CfgRvalue::IsNonNull(CfgOperand::Copy(opt_place.clone())) });

        let some_bb = self.new_block();
        let none_bb = self.new_block();
        let merge_bb = self.new_block();
        self.func.blocks[merge_bb.0 as usize].join_span = Some(span);

        self.set_term(CfgTerm::Branch {
            cond: CfgOperand::Copy(cond_dst),
            then_bb: some_bb,
            else_bb: none_bb,
            span,
            is_user_if: false,
        });

        self.cur_bb = Some(some_bb);
        let unwrap_src = if self.env.is_linear(res_ty) {
            CfgOperand::Move(opt_place)
        } else {
            CfgOperand::Copy(opt_place)
        };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: res_dst.clone(),
                rvalue: CfgRvalue::UnwrapOptional(unwrap_src),
            },
        );
        self.set_term(CfgTerm::Goto(merge_bb));

        self.cur_bb = Some(none_bb);
        self.lower_catch_body(rhs, &res_dst, span);
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(merge_bb));
        }

        self.cur_bb = Some(merge_bb);
        self.operand_for_local(res_dst)
    }

    fn lower_catch(
        &mut self,
        lhs: &Expr,
        err_bind: Option<ast::IdentSpan>,
        rhs: &CatchBody,
        ty: TyId,
        span: Span,
    ) -> CfgOperand {
        let eu_ty = self.raw_ty(lhs.span);
        let lop = self.lower_expr_rvalue(lhs, if eu_ty != TyId::ERROR { Some(eu_ty) } else { None });
        if self.cur_bb.is_none() {
            return CfgOperand::Const(CfgConst::Void);
        }
        let actual_eu = lop.ty(&self.func, self.env);
        let eu_tmp = self.new_temp(actual_eu, lhs.span);
        let eu_place = CfgPlace { local: eu_tmp, projs: Vec::new(), ty: actual_eu, span: lhs.span };
        self.push_stmt(lhs.span, CfgStmtKind::Assign { dst: eu_place.clone(), rvalue: CfgRvalue::Use(lop) });

        let res_ty = if ty != TyId::ERROR { ty } else { match self.env.get(actual_eu) { Ty::ErrorUnion { ok, .. } => *ok, _ => TyId::ERROR } };
        let res_tmp = self.new_temp(res_ty, span);
        let res_dst = CfgPlace { local: res_tmp, projs: Vec::new(), ty: res_ty, span };

        let cond_tmp = self.new_temp(TyId::BOOL, span);
        let cond_dst = CfgPlace { local: cond_tmp, projs: Vec::new(), ty: TyId::BOOL, span };
        self.push_stmt(span, CfgStmtKind::Assign { dst: cond_dst.clone(), rvalue: CfgRvalue::IsError(CfgOperand::Copy(eu_place.clone())) });

        let err_bb = self.new_block();
        let ok_bb = self.new_block();
        let merge_bb = self.new_block();
        self.func.blocks[merge_bb.0 as usize].join_span = Some(span);

        self.set_term(CfgTerm::Branch {
            cond: CfgOperand::Copy(cond_dst),
            then_bb: err_bb,
            else_bb: ok_bb,
            span,
            is_user_if: false,
        });

        self.cur_bb = Some(ok_bb);
        let unwrap_src = if self.env.is_linear(res_ty) {
            CfgOperand::Move(eu_place.clone())
        } else {
            CfgOperand::Copy(eu_place.clone())
        };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: res_dst.clone(),
                rvalue: CfgRvalue::UnwrapErrorOk(unwrap_src),
            },
        );
        self.set_term(CfgTerm::Goto(merge_bb));

        self.cur_bb = Some(err_bb);
        self.push_scope();
        if let Some(eb) = err_bind {
            let err_src = match self.env.get(actual_eu) {
                Ty::ErrorUnion { errors, .. } => errors.first().copied(),
                _ => None,
            };
            let ety = match err_src {
                Some(s) => self.env.intern(Ty::ErrorSet(s)),
                None => self.enclosing_err_set_ty(),
            };
            let eloc = self.bind_user_local(eb.sym, eb.span, ety, false);
            let dst = CfgPlace { local: eloc, projs: Vec::new(), ty: ety, span: eb.span };
            self.push_stmt(eb.span, CfgStmtKind::Assign { dst, rvalue: CfgRvalue::UnwrapErrorErr(CfgOperand::Copy(eu_place)) });
        }
        self.lower_catch_body(rhs, &res_dst, span);
        self.pop_scope_normal();
        if self.cur_bb.is_some() {
            self.set_term(CfgTerm::Goto(merge_bb));
        }

        self.cur_bb = Some(merge_bb);
        self.operand_for_local(res_dst)
    }

    fn lower_catch_body(&mut self, body: &CatchBody, dst: &CfgPlace, span: Span) {
        match body {
            CatchBody::Block(b) => self.lower_block_into(b, Some(dst), Some(dst.ty)),
            CatchBody::Expr(e) => self.lower_expr_into(e, dst, dst.ty),
            CatchBody::Jump(j) => self.lower_jump(j, span),
        }
    }

    fn lower_call(&mut self, callee: &Expr, args: &[Expr], ret_ty: TyId, span: Span) -> CfgOperand {
        // Case 0: Union variant constructor `.Click(payload)`
        if let ExprKind::EnumLit(v) = &callee.kind
            && let Ty::Union { def, .. } = self.env.get(ret_ty).clone()
        {
            let vidx = self.variant_tag_index(ret_ty, v.sym).unwrap_or(0);
            let pty = self
                .env
                .unions
                .get(&def)
                .and_then(|um| um.variants.get(vidx).and_then(|uv| uv.payload))
                .unwrap_or(TyId::ERROR);
            let mut elems = Vec::new();
            if let Some(a0) = args.first() {
                let op = self.lower_expr_rvalue(a0, Some(pty));
                elems.push(self.coerce_operand(op, pty, a0.span));
            }
            let tmp = self.new_temp(ret_ty, span);
            let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: dst.clone(),
                    rvalue: CfgRvalue::Aggregate {
                        kind: AggregateKind::UnionVariant { def, variant_idx: vidx },
                        elems,
                    },
                },
            );
            return self.operand_for_local(dst);
        }

        // Case 1: Conversion casts `x.widen.[To]()`, etc.
        if let ExprKind::GenericApply { base, .. } = &callee.kind
            && let ExprKind::Field { base: recv, field } = &base.kind
        {
            let mname = self.interner.get(field.sym);
            if mname == "try_into" {
                let rop = self.lower_expr_rvalue(recv, None);
                return self.lower_try_into(rop, ret_ty, field.sym, span);
            }
            let ck = match mname {
                "widen" => Some(CastKind::Widen),
                "trunc" => Some(CastKind::Trunc),
                "clamp" => Some(CastKind::Clamp),
                "to_float" => Some(CastKind::ToFloat),
                "trunc_to_int" => Some(CastKind::TruncToInt),
                "bitcast" => Some(CastKind::Bitcast),
                _ => None,
            };
            if let Some(kind) = ck {
                let rop = self.lower_expr_rvalue(recv, None);
                let tmp = self.new_temp(ret_ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Cast {
                            kind,
                            operand: rop,
                            to_ty: ret_ty,
                        },
                    },
                );
                return self.operand_for_local(dst);
            }
        }

        // Case 2: Builtin type assoc call (`ArenaPool.[T].init`, `Arena.init`, etc.) or method call.
        let inner_callee = match &callee.kind {
            ExprKind::GenericApply { base, .. } => base.as_ref(),
            _ => callee,
        };

        if let ExprKind::Field { base: recv, field } = &inner_callee.kind {
            let mname = self.interner.get(field.sym).to_owned();
            // Builtin type assoc call like `ArenaPool.[Node].init(...)`, `Arena.init(...)`, `Atomic.[u64].new(...)`, `rawptr.[u8].from_usize(...)`, `Schedule.build(...)`
            if let Some(bk) = self.extract_builtin_assoc_kind(recv) {
                if mname == "init" {
                    return self.lower_builtin_init_call(bk, ret_ty, args, span);
                }
                if mname == "new" && bk == BuiltinKind::Atomic {
                    return self.lower_builtin_assoc_op(CfgBuiltinOp::AtomicNew(self.extract_first_type_arg(ret_ty)), ret_ty, args, span);
                }
                if mname == "from_usize" {
                    let bop = CfgBuiltinOp::RawPtrFromUsize(ret_ty);
                    return self.lower_builtin_assoc_op(bop, ret_ty, args, span);
                }
                if mname == "splat" && matches!(bk, BuiltinKind::SimdAlias(_, _) | BuiltinKind::VecType) {
                    let bop = CfgBuiltinOp::VecSplat { vec_ty: ret_ty };
                    return self.lower_builtin_assoc_op(bop, ret_ty, args, span);
                }
                if mname == "build" && bk == BuiltinKind::Schedule {
                    return self.lower_schedule_build(args, ret_ty);
                }
            }

            // Distinguish module/type static call vs method call on value receiver.
            let is_static_path = self.is_module_or_type_static_path(recv);
            if !is_static_path {
                let rty = self.expr_ty(recv, TyId::ERROR);
                let rplace = self.lower_place(recv);
                if self.cur_bb.is_none() {
                    return CfgOperand::Const(CfgConst::Void);
                }
                if (mname == "as_u64" || mname == "as_usize") && args.is_empty() && (self.env.is_address(rty) || matches!(self.env.get(rty), Ty::RawPtr { .. })) {
                    let tmp = self.new_temp(ret_ty, span);
                    let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span };
                    self.push_stmt(
                        span,
                        CfgStmtKind::Assign {
                            dst: dst.clone(),
                            rvalue: CfgRvalue::Cast {
                                kind: CastKind::AsU64,
                                operand: CfgOperand::Copy(rplace),
                                to_ty: ret_ty,
                            },
                        },
                    );
                    return CfgOperand::Copy(dst);
                }

                // Dynamic dispatch call on `*dyn Iface` / `*mut dyn Iface`!
                if let Some((iface_def, method_idx)) = self.table.get_dyn_call(span)
                    && let Some(msig) = self.env.ifaces.get(&iface_def).and_then(|im| im.methods.get(method_idx)).cloned()
                {
                    let is_mut_recv = msig.params.first().is_some_and(|p| {
                        p.is_self && matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. })
                    });
                    let mut cargs = vec![CfgOperand::Copy(rplace.clone())];
                    let mut arg_spans = vec![recv.span];
                    for (i, a) in args.iter().enumerate() {
                        let pty = msig.params.get(i + 1).map(|p| p.ty);
                        let op = self.lower_expr_rvalue(a, pty);
                        let cop = if let Some(pt) = pty { self.coerce_operand(op, pt, a.span) } else { op };
                        cargs.push(cop);
                        arg_spans.push(a.span);
                    }
                    let dst_place = if ret_ty == TyId::VOID {
                        None
                    } else {
                        let tmp = self.new_temp(ret_ty, span);
                        Some(CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span })
                    };
                    self.push_stmt(
                        span,
                        CfgStmtKind::Call {
                            dst: dst_place.clone(),
                            callee: CfgCallee::DynCall { iface_def, method_idx },
                            args: cargs,
                            arg_spans,
                            recv_place: Some(rplace),
                            recv_span: Some(recv.span),
                            is_mut_recv,
                            consumes_recv: false,
                        },
                    );
                    return dst_place.map_or(CfgOperand::Const(CfgConst::Void), |d| self.operand_for_local(d));
                }

                if mname == "drain" {
                    let deref_rty = match self.env.get(rty) {
                        Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
                        _ => rty,
                    };
                    if let Ty::Builtin(BuiltinKind::Buf | BuiltinKind::LinearBuf, gargs) = self.env.get(deref_rty).clone() {
                        let elem_ty = match gargs.first() {
                            Some(GenericVal::Ty(t)) => *t,
                            _ => TyId::U64,
                        };
                        return self.lower_buf_drain(&rplace, recv.span, elem_ty, args, span);
                    }
                }

                if let Some((bop, is_mut_recv, consumes_recv)) = self.match_builtin_method(rty, &mname, args) {
                    return self.lower_builtin_method_call(bop, &rplace, recv.span, is_mut_recv, consumes_recv, args, ret_ty, span);
                }

                // User `impl` method call (handles concrete AND generic struct/method instantiations!)
                if let Some((impl_idx, sub_idx, msig)) = self.find_method_entry(rty, field.sym) {
                    let struct_gargs: Vec<GenericVal> = {
                        let d = match self.env.get(rty) {
                            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
                            _ => rty,
                        };
                        match self.env.get(d) {
                            Ty::Struct { args, .. } | Ty::Union { args, .. } | Ty::Pool { args, .. } => args.clone(),
                            _ => Vec::new(),
                        }
                    };
                    let inst_gargs: Vec<GenericVal> = if let Some((_, cg)) = self.table.get_call_generic_inst(span).cloned() {
                        self.subst_gargs(&cg)
                    } else {
                        self.subst_gargs(&struct_gargs)
                    };
                    let mut sub_sig = msig.clone();
                    if !inst_gargs.is_empty() {
                        for p in &mut sub_sig.params {
                            p.ty = crate::check::subst_ty(p.ty, &inst_gargs, self.env);
                        }
                        sub_sig.ret = crate::check::subst_ty(sub_sig.ret, &inst_gargs, self.env);
                    }
                    let is_mut_recv = sub_sig.params.first().is_some_and(|p| {
                        p.is_self && matches!(self.env.get(p.ty), Ty::Borrow { is_mut: true, .. })
                    });
                    let consumes_recv = sub_sig.params.first().is_some_and(|p| {
                        p.is_self && !matches!(self.env.get(p.ty), Ty::Borrow { .. }) && self.env.is_linear(p.ty)
                    });
                    let mut created_self_borrow = false;
                    let recv_arg = if let Some(p0) = sub_sig.params.first().filter(|p| p.is_self) {
                        if let Ty::Borrow { is_mut, .. } = self.env.get(p0.ty).clone()
                            && !matches!(self.env.get(rplace.ty), Ty::Borrow { .. })
                        {
                            created_self_borrow = true;
                            let bt = p0.ty;
                            let btmp = self.new_temp(bt, recv.span);
                            let bdst = CfgPlace { local: btmp, projs: Vec::new(), ty: bt, span: recv.span };
                            self.push_stmt(
                                recv.span,
                                CfgStmtKind::Assign {
                                    dst: bdst.clone(),
                                    rvalue: CfgRvalue::Borrow {
                                        is_mut,
                                        place: rplace.clone(),
                                    },
                                },
                            );
                            CfgOperand::Copy(bdst)
                        } else if consumes_recv {
                            CfgOperand::Move(rplace.clone())
                        } else {
                            CfgOperand::Copy(rplace.clone())
                        }
                    } else {
                        CfgOperand::Copy(rplace.clone())
                    };
                    let has_self = sub_sig.params.first().is_some_and(|p| p.is_self);
                    let p_offset = usize::from(has_self);
                    let mut cargs = Vec::new();
                    let mut arg_spans = Vec::new();
                    if has_self {
                        cargs.push(recv_arg);
                        arg_spans.push(recv.span);
                    }
                    for (i, a) in args.iter().enumerate() {
                        let pty = sub_sig.params.get(i + p_offset).map(|p| p.ty);
                        let op = self.lower_expr_rvalue(a, pty);
                        let cop = if let Some(pt) = pty { self.coerce_operand(op, pt, a.span) } else { op };
                        cargs.push(cop);
                        arg_spans.push(a.span);
                    }
                    let dst_place = if ret_ty == TyId::VOID {
                        None
                    } else {
                        let tmp = self.new_temp(ret_ty, span);
                        Some(CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span })
                    };
                    let callee = if !inst_gargs.is_empty() {
                        let base_key = CfgFnKey::ImplMethod { impl_idx, sub_idx };
                        let inst_id = self.intern_generic_inst(base_key, inst_gargs);
                        CfgCallee::GenericInst { inst_id, is_view: sub_sig.is_view }
                    } else {
                        CfgCallee::Method { impl_idx, sub_idx, is_view: sub_sig.is_view }
                    };
                    self.push_stmt(
                        span,
                        CfgStmtKind::Call {
                            dst: dst_place.clone(),
                            callee,
                            args: cargs,
                            arg_spans,
                            recv_place: if created_self_borrow { None } else { Some(rplace) },
                            recv_span: Some(recv.span),
                            is_mut_recv,
                            consumes_recv,
                        },
                    );
                    return dst_place.map_or(CfgOperand::Const(CfgConst::Void), |d| self.operand_for_local(d));
                }
            }
        }

        // Case 3: Direct or module-qualified function call, generic static method call, or fn pointer call.
        let (mut callee_kind, mut sig_opt) = self.resolve_direct_callee(callee);
        if let Some((base_key, cg)) = self.table.get_call_generic_inst(span).cloned() {
            let inst_gargs = self.subst_gargs(&cg);
            if !inst_gargs.is_empty()
                && inst_gargs.iter().all(|gv| match gv {
                    GenericVal::Ty(t) => *t != TyId::ERROR && !crate::check::ty_contains_generic_param(*t, self.env),
                    _ => true,
                })
            {
                let is_view = sig_opt.as_ref().is_some_and(|s| s.is_view);
                if let Some(ref mut s) = sig_opt {
                    for p in &mut s.params {
                        p.ty = crate::check::subst_ty(p.ty, &inst_gargs, self.env);
                    }
                    s.ret = crate::check::subst_ty(s.ret, &inst_gargs, self.env);
                }
                let inst_id = self.intern_generic_inst(base_key, inst_gargs);
                callee_kind = CfgCallee::GenericInst { inst_id, is_view };
            }
        }
        let mut cargs = Vec::with_capacity(args.len());
        let mut arg_spans = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let pty = sig_opt.as_ref().and_then(|s| s.params.get(i)).map(|p| p.ty);
            let op = self.lower_expr_rvalue(a, pty);
            let cop = if let Some(pt) = pty { self.coerce_operand(op, pt, a.span) } else { op };
            cargs.push(cop);
            arg_spans.push(a.span);
        }
        if self.cur_bb.is_none() {
            return CfgOperand::Const(CfgConst::Void);
        }
        let actual_ret = if ret_ty != TyId::ERROR {
            ret_ty
        } else {
            sig_opt.as_ref().map_or(TyId::VOID, |s| s.ret)
        };
        let dst_place = if actual_ret == TyId::VOID {
            None
        } else {
            let tmp = self.new_temp(actual_ret, span);
            Some(CfgPlace { local: tmp, projs: Vec::new(), ty: actual_ret, span })
        };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: dst_place.clone(),
                callee: callee_kind,
                args: cargs,
                arg_spans,
                recv_place: None,
                recv_span: None,
                is_mut_recv: false,
                consumes_recv: false,
            },
        );
        dst_place.map_or(CfgOperand::Const(CfgConst::Void), |d| self.operand_for_local(d))
    }

    fn lower_builtin_init_call(&mut self, bk: BuiltinKind, ret_ty: TyId, args: &[Expr], span: Span) -> CfgOperand {
        let mut cargs = Vec::with_capacity(args.len());
        let mut arg_spans = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let hint = if i == 1 { Some(TyId::USIZE) } else { None };
            let op = self.lower_expr_rvalue(a, hint);
            let cop = if i == 1 { self.coerce_operand(op, TyId::USIZE, a.span) } else { op };
            cargs.push(cop);
            arg_spans.push(a.span);
        }
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => ret_ty,
        };
        let elem_ty = match self.env.get(ok_ty) {
            Ty::Builtin(_, gargs) => match gargs.first() {
                Some(GenericVal::Ty(t)) => *t,
                _ => TyId::ERROR,
            },
            _ => TyId::ERROR,
        };
        let bop = match bk {
            BuiltinKind::Arena => CfgBuiltinOp::ArenaInit,
            BuiltinKind::ArenaPool => CfgBuiltinOp::ArenaPoolInit(elem_ty),
            BuiltinKind::SlotPool => CfgBuiltinOp::SlotPoolInit(elem_ty),
            BuiltinKind::Buf | BuiltinKind::LinearBuf => CfgBuiltinOp::BufInit(elem_ty),
            _ => CfgBuiltinOp::ArenaInit,
        };
        let tmp = self.new_temp(ret_ty, span);
        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: Some(dst.clone()),
                callee: CfgCallee::Builtin(bop),
                args: cargs,
                arg_spans,
                recv_place: None,
                recv_span: None,
                is_mut_recv: false,
                consumes_recv: false,
            },
        );
        self.operand_for_local(dst)
    }

    fn extract_first_type_arg(&self, ty: TyId) -> TyId {
        match self.env.get(ty) {
            Ty::Builtin(_, gargs) => match gargs.first() {
                Some(GenericVal::Ty(t)) => *t,
                _ => TyId::U64,
            },
            Ty::RawPtr { inner, .. } => *inner,
            _ => TyId::U64,
        }
    }

    fn lower_builtin_assoc_op(&mut self, bop: CfgBuiltinOp, ret_ty: TyId, args: &[Expr], span: Span) -> CfgOperand {
        let hint = match &bop {
            CfgBuiltinOp::AtomicNew(el) => Some(*el),
            CfgBuiltinOp::RawPtrFromUsize(_) => Some(TyId::USIZE),
            CfgBuiltinOp::VecSplat { vec_ty } => match self.env.get(*vec_ty) {
                Ty::Vec { elem, .. } => Some(*elem),
                _ => None,
            },
            _ => None,
        };
        let mut cargs = Vec::with_capacity(args.len());
        let mut arg_spans = Vec::with_capacity(args.len());
        for a in args {
            let op = self.lower_expr_rvalue(a, hint);
            let cop = if let Some(ht) = hint { self.coerce_operand(op, ht, a.span) } else { op };
            cargs.push(cop);
            arg_spans.push(a.span);
        }
        let tmp = self.new_temp(ret_ty, span);
        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: Some(dst.clone()),
                callee: CfgCallee::Builtin(bop),
                args: cargs,
                arg_spans,
                recv_place: None,
                recv_span: None,
                is_mut_recv: false,
                consumes_recv: false,
            },
        );
        self.operand_for_local(dst)
    }

    fn lower_builtin_method_call(
        &mut self,
        bop: CfgBuiltinOp,
        rplace: &CfgPlace,
        recv_span: Span,
        is_mut_recv: bool,
        consumes_recv: bool,
        args: &[Expr],
        ret_ty: TyId,
        span: Span,
    ) -> CfgOperand {
        let mut cargs = Vec::with_capacity(args.len() + 1);
        let mut arg_spans = Vec::with_capacity(args.len() + 1);
        let mut created_recv_borrow = false;
        let rop = if consumes_recv {
            CfgOperand::Move(rplace.clone())
        } else if (!rplace.projs.is_empty()
            && matches!(
                self.env.get(rplace.ty),
                Ty::Builtin(
                    BuiltinKind::Arena
                        | BuiltinKind::ArenaPool
                        | BuiltinKind::SlotPool
                        | BuiltinKind::Buf
                        | BuiltinKind::LinearBuf
                        | BuiltinKind::Atomic
                        | BuiltinKind::MmioReg,
                    _
                ) | Ty::Array { .. }
            ))
            || matches!(
                self.env.get(rplace.ty),
                Ty::Builtin(
                    BuiltinKind::AllocCap
                        | BuiltinKind::ArgsCap
                        | BuiltinKind::BuildCap
                        | BuiltinKind::FsCap
                        | BuiltinKind::DirCap
                        | BuiltinKind::NetCap
                        | BuiltinKind::ClockCap
                        | BuiltinKind::IoCap,
                    _
                )
            )
        {
            created_recv_borrow = true;
            let bty = self.env.intern(Ty::Borrow {
                is_mut: is_mut_recv,
                inner: crate::types::BorrowInner::Single(rplace.ty),
            });
            let btmp = self.new_temp(bty, recv_span);
            let bdst = CfgPlace { local: btmp, projs: Vec::new(), ty: bty, span: recv_span };
            self.push_stmt(
                recv_span,
                CfgStmtKind::Assign {
                    dst: bdst.clone(),
                    rvalue: CfgRvalue::Borrow {
                        is_mut: is_mut_recv,
                        place: rplace.clone(),
                    },
                },
            );
            CfgOperand::Copy(bdst)
        } else if matches!(
            bop,
            CfgBuiltinOp::VecLen(_)
                | CfgBuiltinOp::VecCmp { .. }
                | CfgBuiltinOp::VecMaskAny { .. }
                | CfgBuiltinOp::VecMaskAll { .. }
                | CfgBuiltinOp::VecBitmask { .. }
                | CfgBuiltinOp::VecSelect { .. }
                | CfgBuiltinOp::VecMin { .. }
                | CfgBuiltinOp::VecMax { .. }
                | CfgBuiltinOp::VecFma { .. }
                | CfgBuiltinOp::VecReduceAdd { .. }
                | CfgBuiltinOp::VecReduceMin { .. }
                | CfgBuiltinOp::VecReduceMax { .. }
                | CfgBuiltinOp::VecDot { .. }
                | CfgBuiltinOp::VecShuffle { .. }
        ) && let Ty::Borrow { inner: crate::types::BorrowInner::Single(inner), .. } =
            self.env.get(rplace.ty).clone()
        {
            let mut dp = rplace.clone();
            dp.projs.push(CfgProj::Deref);
            dp.ty = inner;
            CfgOperand::Copy(dp)
        } else {
            CfgOperand::Copy(rplace.clone())
        };
        cargs.push(rop);
        arg_spans.push(recv_span);

        let expected_args: Vec<Option<TyId>> = match &bop {
            CfgBuiltinOp::ArenaAlloc
            | CfgBuiltinOp::ArenaPoolIdAt(_)
            | CfgBuiltinOp::SlotPoolIsAlive(_)
            | CfgBuiltinOp::SlotPoolIdAt(_)
            | CfgBuiltinOp::SliceAt { .. }
            | CfgBuiltinOp::ArrayAt { .. }
            | CfgBuiltinOp::BufAt { .. }
            | CfgBuiltinOp::SliceGet { .. }
            | CfgBuiltinOp::ArrayGet { .. }
            | CfgBuiltinOp::BufGet { .. } => vec![Some(TyId::USIZE)],
            CfgBuiltinOp::SliceSub { .. }
            | CfgBuiltinOp::ArraySub { .. }
            | CfgBuiltinOp::BufSub { .. } => vec![Some(TyId::USIZE), Some(TyId::USIZE)],
            CfgBuiltinOp::ArenaPoolPush(el) | CfgBuiltinOp::SlotPoolTryInsert(el) => vec![Some(*el)],
            CfgBuiltinOp::SlotPoolInsert(el) | CfgBuiltinOp::BufPush(el) => {
                if args.len() == 2 { vec![None, Some(*el)] } else { vec![Some(*el)] }
            }
            CfgBuiltinOp::BufPushSlice(el) => {
                let slice_ty = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Slice(*el),
                });
                if args.len() == 2 { vec![None, Some(slice_ty)] } else { vec![Some(slice_ty)] }
            }
            CfgBuiltinOp::AtomicLoad(_) => vec![Some(TyId::ORDERING)],
            CfgBuiltinOp::AtomicStore(el) | CfgBuiltinOp::AtomicFetchAdd(el) | CfgBuiltinOp::AtomicFetchSub(el) | CfgBuiltinOp::AtomicSwap(el) => {
                vec![Some(*el), Some(TyId::ORDERING)]
            }
            CfgBuiltinOp::AtomicCas(el) => vec![Some(*el), Some(*el), Some(TyId::ORDERING), Some(TyId::ORDERING)],
            CfgBuiltinOp::MmioWrite(el) => vec![Some(*el)],
            CfgBuiltinOp::SlotPoolRemove(_) | CfgBuiltinOp::SlotPoolGet { .. } | CfgBuiltinOp::SlotPoolAt(_) => vec![None],
            CfgBuiltinOp::VecCmp { vec_ty, .. }
            | CfgBuiltinOp::VecMin { vec_ty }
            | CfgBuiltinOp::VecMax { vec_ty }
            | CfgBuiltinOp::VecDot { vec_ty } => vec![Some(*vec_ty)],
            CfgBuiltinOp::VecFma { vec_ty } | CfgBuiltinOp::VecSelect { vec_ty } => {
                vec![Some(*vec_ty), Some(*vec_ty)]
            }
            CfgBuiltinOp::VecShuffle { src_ty, idx_ty, two_src } => {
                if *two_src { vec![Some(*src_ty), Some(*idx_ty)] } else { vec![Some(*idx_ty)] }
            }
            CfgBuiltinOp::FsWriteExecutable | CfgBuiltinOp::FsSpawnWait => {
                let str_ty = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                let slice_u8 = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Slice(TyId::U8),
                });
                vec![Some(str_ty), Some(slice_u8)]
            }
            CfgBuiltinOp::FsIsDir | CfgBuiltinOp::FsMkdirAll => {
                let str_ty = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                vec![Some(str_ty)]
            }
            CfgBuiltinOp::FsListDir => {
                let str_ty = self.env.intern(Ty::Borrow {
                    is_mut: false,
                    inner: crate::types::BorrowInner::Str,
                });
                vec![None, Some(str_ty)]
            }
            _ => vec![None; args.len()],
        };
        let saved_row_kind = match &bop {
            CfgBuiltinOp::ArenaPoolPush(_) => self.pool_row_is_slot.replace(false),
            CfgBuiltinOp::SlotPoolInsert(_) | CfgBuiltinOp::SlotPoolTryInsert(_) => {
                self.pool_row_is_slot.replace(true)
            }
            _ => None,
        };
        for (i, a) in args.iter().enumerate() {
            let hint = expected_args.get(i).copied().flatten();
            let op = self.lower_expr_rvalue(a, hint);
            let cop = if let Some(ht) = hint {
                if matches!(self.env.get(ht), Ty::Pool { .. }) {
                    op
                } else {
                    self.coerce_operand(op, ht, a.span)
                }
            } else {
                op
            };
            cargs.push(cop);
            arg_spans.push(a.span);
        }
        if matches!(
            bop,
            CfgBuiltinOp::ArenaPoolPush(_)
                | CfgBuiltinOp::SlotPoolInsert(_)
                | CfgBuiltinOp::SlotPoolTryInsert(_)
        ) {
            self.pool_row_is_slot = saved_row_kind;
        }

        let dst_place = if ret_ty == TyId::VOID {
            None
        } else {
            let tmp = self.new_temp(ret_ty, span);
            Some(CfgPlace { local: tmp, projs: Vec::new(), ty: ret_ty, span })
        };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: dst_place.clone(),
                callee: CfgCallee::Builtin(bop),
                args: cargs,
                arg_spans,
                recv_place: if created_recv_borrow { None } else { Some(rplace.clone()) },
                recv_span: Some(recv_span),
                is_mut_recv,
                consumes_recv,
            },
        );
        dst_place.map_or(CfgOperand::Const(CfgConst::Void), |d| self.operand_for_local(d))
    }

    fn match_builtin_method(&mut self, rty: TyId, mname: &str, args: &[Expr]) -> Option<(CfgBuiltinOp, bool, bool)> {
        let deref = match self.env.get(rty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => rty,
        };
        if matches!(rty, _ if matches!(self.env.get(rty), Ty::Borrow { inner: crate::types::BorrowInner::Slice(_), .. })) {
            let Ty::Borrow { inner: crate::types::BorrowInner::Slice(el), .. } = self.env.get(rty) else {
                unreachable!();
            };
            return match mname {
                "len" => Some((CfgBuiltinOp::SliceLen, false, false)),
                "as_str" if *el == TyId::U8 => Some((CfgBuiltinOp::BytesAsStr, false, false)),
                "at" => Some((CfgBuiltinOp::SliceAt { elem_ty: *el, is_mut: false }, false, false)),
                "at_mut" => Some((CfgBuiltinOp::SliceAt { elem_ty: *el, is_mut: true }, true, false)),
                "get" => Some((CfgBuiltinOp::SliceGet { elem_ty: *el, is_mut: false }, false, false)),
                "get_mut" => Some((CfgBuiltinOp::SliceGet { elem_ty: *el, is_mut: true }, true, false)),
                "sub" => Some((CfgBuiltinOp::SliceSub { elem_ty: *el, is_mut: false }, false, false)),
                "sub_mut" => Some((CfgBuiltinOp::SliceSub { elem_ty: *el, is_mut: true }, true, false)),
                "par_reduce" => {
                    let op_sym = if let Some(Expr { kind: ExprKind::EnumLit(v), .. }) = args.first() {
                        v.sym
                    } else {
                        Symbol(0)
                    };
                    Some((CfgBuiltinOp::ParReduce { elem_ty: *el, op: op_sym }, false, false))
                }
                _ => None,
            };
        }
        if matches!(self.env.get(deref), Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str) {
            return match mname {
                "len" => Some((CfgBuiltinOp::StrLen, false, false)),
                "as_bytes" => Some((CfgBuiltinOp::StrAsBytes, false, false)),
                _ => None,
            };
        }
        match self.env.get(deref).clone() {
            Ty::Array { elem, .. } => match mname {
                "len" => Some((CfgBuiltinOp::ArrayLen, false, false)),
                "at" => Some((CfgBuiltinOp::ArrayAt { elem_ty: elem, is_mut: false }, false, false)),
                "at_mut" => Some((CfgBuiltinOp::ArrayAt { elem_ty: elem, is_mut: true }, true, false)),
                "get" => Some((CfgBuiltinOp::ArrayGet { elem_ty: elem, is_mut: false }, false, false)),
                "get_mut" => Some((CfgBuiltinOp::ArrayGet { elem_ty: elem, is_mut: true }, true, false)),
                "sub" => Some((CfgBuiltinOp::ArraySub { elem_ty: elem, is_mut: false }, false, false)),
                "sub_mut" => Some((CfgBuiltinOp::ArraySub { elem_ty: elem, is_mut: true }, true, false)),
                _ => None,
            },
            Ty::Vec { len, .. } => match mname {
                "len" => Some((CfgBuiltinOp::VecLen(len), false, false)),
                "cmpeq" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::Eq, vec_ty: deref }, false, false)),
                "cmpne" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::NotEq, vec_ty: deref }, false, false)),
                "cmplt" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::Lt, vec_ty: deref }, false, false)),
                "cmple" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::LtEq, vec_ty: deref }, false, false)),
                "cmpgt" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::Gt, vec_ty: deref }, false, false)),
                "cmpge" => Some((CfgBuiltinOp::VecCmp { op: BinaryOp::GtEq, vec_ty: deref }, false, false)),
                "any" => Some((CfgBuiltinOp::VecMaskAny { len }, false, false)),
                "all" => Some((CfgBuiltinOp::VecMaskAll { len }, false, false)),
                "bitmask" => Some((CfgBuiltinOp::VecBitmask { len }, false, false)),
                "select" => {
                    let vty = args.first().map_or(TyId::ERROR, |a| self.expr_ty(a, TyId::ERROR));
                    Some((CfgBuiltinOp::VecSelect { vec_ty: vty }, false, false))
                }
                "min" => Some((CfgBuiltinOp::VecMin { vec_ty: deref }, false, false)),
                "max" => Some((CfgBuiltinOp::VecMax { vec_ty: deref }, false, false)),
                "fma" => Some((CfgBuiltinOp::VecFma { vec_ty: deref }, false, false)),
                "reduce_add" => Some((CfgBuiltinOp::VecReduceAdd { vec_ty: deref }, false, false)),
                "reduce_min" => Some((CfgBuiltinOp::VecReduceMin { vec_ty: deref }, false, false)),
                "reduce_max" => Some((CfgBuiltinOp::VecReduceMax { vec_ty: deref }, false, false)),
                "dot" => Some((CfgBuiltinOp::VecDot { vec_ty: deref }, false, false)),
                "shuffle" => {
                    let two_src = args.len() == 2;
                    let idx_arg = if two_src { args.get(1) } else { args.first() };
                    let idx_ty = idx_arg.map_or(TyId::ERROR, |a| self.expr_ty(a, TyId::ERROR));
                    Some((CfgBuiltinOp::VecShuffle { src_ty: deref, idx_ty, two_src }, false, false))
                }
                _ => None,
            },
            Ty::Builtin(BuiltinKind::Arena, _) => match mname {
                "alloc" => Some((CfgBuiltinOp::ArenaAlloc, true, false)),
                "reset" => Some((CfgBuiltinOp::ArenaReset, true, false)),
                "destroy" => Some((CfgBuiltinOp::ArenaDestroy, true, true)),
                "used_bytes" => Some((CfgBuiltinOp::ArenaUsedBytes, false, false)),
                "capacity" => Some((CfgBuiltinOp::ArenaCapacity, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::AllocCap, _) => match mname {
                "live_allocs" => Some((CfgBuiltinOp::AllocCapLiveAllocs, false, false)),
                "used_bytes" => Some((CfgBuiltinOp::AllocCapUsedBytes, false, false)),
                "peak_bytes" => Some((CfgBuiltinOp::AllocCapPeakBytes, false, false)),
                "sub_budget" => Some((CfgBuiltinOp::AllocCapSubBudget, true, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::ClockCap, _) => match mname {
                "now_ns" => Some((CfgBuiltinOp::ClockNowNs, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::ArgsCap, _) => match mname {
                "len" => Some((CfgBuiltinOp::ArgsLen, false, false)),
                "get" => Some((CfgBuiltinOp::ArgsGet, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::FsCap, _) => match mname {
                "open_dir" | "sub_dir" => Some((CfgBuiltinOp::FsOpenDir, false, false)),
                "read_file" => Some((CfgBuiltinOp::FsReadFile, false, false)),
                "write_file" => Some((CfgBuiltinOp::FsWriteFile, false, false)),
                "write_executable" => Some((CfgBuiltinOp::FsWriteExecutable, false, false)),
                "is_dir" => Some((CfgBuiltinOp::FsIsDir, false, false)),
                "mkdir_all" => Some((CfgBuiltinOp::FsMkdirAll, false, false)),
                "list_dir" => Some((CfgBuiltinOp::FsListDir, false, false)),
                "spawn_wait" => Some((CfgBuiltinOp::FsSpawnWait, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::DirCap, _) => match mname {
                "sub_dir" => Some((CfgBuiltinOp::DirSubDir, false, false)),
                "read_file" => Some((CfgBuiltinOp::DirReadFile, false, false)),
                "write_file" => Some((CfgBuiltinOp::DirWriteFile, false, false)),
                "close" => Some((CfgBuiltinOp::DirClose, true, true)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::NetCap, _) => match mname {
                "restrict_host" => Some((CfgBuiltinOp::NetRestrictHost, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::IoCap, _) => match mname {
                "nop_roundtrip" => Some((CfgBuiltinOp::IoNopRoundtrip, false, false)),
                "run_fibers" => Some((CfgBuiltinOp::IoRunFibers, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::BuildCap, _) => match mname {
                "add_module" => Some((CfgBuiltinOp::BuildAddModule, true, false)),
                "add_exe" => Some((CfgBuiltinOp::BuildAddExe, true, false)),
                "add_shared_lib" | "add_shared" => Some((CfgBuiltinOp::BuildAddSharedLib, true, false)),
                "set_out_dir" => Some((CfgBuiltinOp::BuildSetOutDir, true, false)),
                "set_sanitize" => Some((CfgBuiltinOp::BuildSetSanitize, true, false)),
                "set_audit_tcb" => Some((CfgBuiltinOp::BuildSetAuditTcb, true, false)),
                "set_opt_level" => Some((CfgBuiltinOp::BuildSetOptLevel, true, false)),
                "link_lib" | "add_needed_lib" => Some((CfgBuiltinOp::BuildLinkLib, true, false)),
                "allow_hal" => Some((CfgBuiltinOp::BuildAllowHal, true, false)),
                "out_path" => Some((CfgBuiltinOp::BuildOutPath, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::Schedule, _) => match mname {
                "run" => Some((CfgBuiltinOp::ScheduleRun, false, false)),
                _ => None,
            },
            Ty::Builtin(BuiltinKind::ArenaPool, gargs) => {
                let elem = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                match mname {
                    "push" => Some((CfgBuiltinOp::ArenaPoolPush(elem), true, false)),
                    "id_at" => Some((CfgBuiltinOp::ArenaPoolIdAt(elem), false, false)),
                    "reset" => Some((CfgBuiltinOp::ArenaPoolReset(elem), true, false)),
                    "len" => Some((CfgBuiltinOp::ArenaPoolLen, false, false)),
                    _ => None,
                }
            }
            Ty::Builtin(BuiltinKind::SlotPool, gargs) => {
                let elem = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                match mname {
                    "insert" => Some((CfgBuiltinOp::SlotPoolInsert(elem), true, false)),
                    "try_insert" => Some((CfgBuiltinOp::SlotPoolTryInsert(elem), true, false)),
                    "remove" => Some((CfgBuiltinOp::SlotPoolRemove(elem), true, false)),
                    "get" => Some((CfgBuiltinOp::SlotPoolGet { elem_ty: elem, is_mut: false }, false, false)),
                    "get_mut" => Some((CfgBuiltinOp::SlotPoolGet { elem_ty: elem, is_mut: true }, true, false)),
                    "at" => Some((CfgBuiltinOp::SlotPoolAt(elem), false, false)),
                    "len" => Some((CfgBuiltinOp::SlotPoolLen, false, false)),
                    "cap" => Some((CfgBuiltinOp::SlotPoolCap, false, false)),
                    "is_alive" => Some((CfgBuiltinOp::SlotPoolIsAlive(elem), false, false)),
                    "id_at" => Some((CfgBuiltinOp::SlotPoolIdAt(elem), false, false)),
                    "destroy" => Some((CfgBuiltinOp::SlotPoolDestroy(elem), true, true)),
                    _ => None,
                }
            }
            Ty::Builtin(BuiltinKind::Buf | BuiltinKind::LinearBuf, gargs) => {
                let elem = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::ERROR,
                };
                match mname {
                    "push" => Some((CfgBuiltinOp::BufPush(elem), true, false)),
                    "push_slice" => Some((CfgBuiltinOp::BufPushSlice(elem), true, false)),
                    "pop" => Some((CfgBuiltinOp::BufPop(elem), true, false)),
                    "len" => Some((CfgBuiltinOp::BufLen, false, false)),
                    "cap" => Some((CfgBuiltinOp::BufCap, false, false)),
                    "clear" => Some((CfgBuiltinOp::BufClear, true, false)),
                    "as_slice" => Some((CfgBuiltinOp::BufAsSlice { elem_ty: elem, is_mut: false }, false, false)),
                    "as_mut_slice" => Some((CfgBuiltinOp::BufAsSlice { elem_ty: elem, is_mut: true }, true, false)),
                    "at" => Some((CfgBuiltinOp::BufAt { elem_ty: elem, is_mut: false }, false, false)),
                    "at_mut" => Some((CfgBuiltinOp::BufAt { elem_ty: elem, is_mut: true }, true, false)),
                    "get" => Some((CfgBuiltinOp::BufGet { elem_ty: elem, is_mut: false }, false, false)),
                    "get_mut" => Some((CfgBuiltinOp::BufGet { elem_ty: elem, is_mut: true }, true, false)),
                    "sub" => Some((CfgBuiltinOp::BufSub { elem_ty: elem, is_mut: false }, false, false)),
                    "sub_mut" => Some((CfgBuiltinOp::BufSub { elem_ty: elem, is_mut: true }, true, false)),
                    "destroy" => Some((CfgBuiltinOp::BufDestroy(elem), true, true)),
                    _ => None,
                }
            }
            Ty::Builtin(BuiltinKind::Atomic, gargs) => {
                let elem = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::U64,
                };
                match mname {
                    "load" => Some((CfgBuiltinOp::AtomicLoad(elem), false, false)),
                    "store" => Some((CfgBuiltinOp::AtomicStore(elem), false, false)),
                    "fetch_add" => Some((CfgBuiltinOp::AtomicFetchAdd(elem), false, false)),
                    "fetch_sub" => Some((CfgBuiltinOp::AtomicFetchSub(elem), false, false)),
                    "swap" => Some((CfgBuiltinOp::AtomicSwap(elem), false, false)),
                    "compare_exchange" => Some((CfgBuiltinOp::AtomicCas(elem), false, false)),
                    _ => None,
                }
            }
            Ty::Builtin(BuiltinKind::MmioReg, gargs) => {
                let elem = match gargs.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::U32,
                };
                match mname {
                    "read" => Some((CfgBuiltinOp::MmioRead(elem), false, false)),
                    "write" => Some((CfgBuiltinOp::MmioWrite(elem), true, false)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn extract_builtin_assoc_kind(&self, expr: &Expr) -> Option<BuiltinKind> {
        let base = match &expr.kind {
            ExprKind::GenericApply { base, .. } => base.as_ref(),
            _ => expr,
        };
        let ExprKind::Ident(sym) = &base.kind else {
            return None;
        };
        if self.lookup_local(*sym).is_some() {
            return None;
        }
        match self.prog.lookup_in_module(self.module, *sym) {
            Some(TopScopeTarget::Builtin(bk)) => Some(bk),
            _ => None,
        }
    }

    fn is_module_or_type_static_path(&self, expr: &Expr) -> bool {
        let mut cur = expr;
        let mut segs = Vec::new();
        loop {
            match &cur.kind {
                ExprKind::Ident(s) => {
                    segs.push(*s);
                    break;
                }
                ExprKind::Field { base, field } => {
                    segs.push(field.sym);
                    cur = base;
                }
                ExprKind::GenericApply { base, .. } => {
                    cur = base;
                }
                _ => return false,
            }
        }
        segs.reverse();
        if segs.first().is_some_and(|s| self.lookup_local(*s).is_none())
            && let Some(t) = self.prog.lookup_in_module(self.module, segs[0])
        {
            return matches!(t, TopScopeTarget::Module(_) | TopScopeTarget::Def(_) | TopScopeTarget::Builtin(_));
        }
        false
    }

    fn find_method_entry(&self, rty: TyId, name: Symbol) -> Option<(usize, usize, FnSig)> {
        let deref = match self.env.get(rty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => rty,
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
        for (ii, im) in self.env.impls.iter().enumerate() {
            if im.target == target
                && let Some(&(_, sub_idx, ref sig)) = im.methods.iter().find(|(_, _, m)| m.name == name)
            {
                return Some((ii, sub_idx, sig.clone()));
            }
        }
        None
    }

    fn resolve_direct_callee(&mut self, callee: &Expr) -> (CfgCallee, Option<FnSig>) {
        let inner = match &callee.kind {
            ExprKind::GenericApply { base, .. } => base.as_ref(),
            _ => callee,
        };
        match &inner.kind {
            ExprKind::Ident(sym) => {
                if self.lookup_local(*sym).is_none()
                    && !self.captured_aliases.contains_key(sym)
                    && let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(self.module, *sym)
                    && let Some(sig) = self.env.fns.get(&d).cloned()
                {
                    if sig.is_extern {
                        return (CfgCallee::ExternFn { def: d, name: sig.name }, Some(sig));
                    }
                    return (CfgCallee::DirectFn(d), Some(sig));
                }
            }
            ExprKind::Field { base, field } => {
                if let Some(target) = self.resolve_static_expr_path(base) {
                    match target {
                        TopScopeTarget::Module(m) => {
                            if let Some(b) = self.prog.modules[m.0 as usize].exports.get(&field.sym)
                                && let TopScopeTarget::Def(d) = b.target
                                && let Some(sig) = self.env.fns.get(&d).cloned()
                            {
                                if sig.is_extern {
                                    return (CfgCallee::ExternFn { def: d, name: sig.name }, Some(sig));
                                }
                                return (CfgCallee::DirectFn(d), Some(sig));
                            }
                        }
                        TopScopeTarget::Def(d) => {
                            for (ii, im) in self.env.impls.iter().enumerate() {
                                if im.target == MethodTarget::Def(d)
                                    && let Some(&(_, sub_idx, ref sig)) = im.methods.iter().find(|(_, _, m)| m.name == field.sym)
                                {
                                    return (CfgCallee::Method { impl_idx: ii, sub_idx, is_view: sig.is_view }, Some(sig.clone()));
                                }
                            }
                        }
                        TopScopeTarget::Builtin(_) => {}
                    }
                }
            }
            _ => {}
        }
        let op = self.lower_expr_rvalue(inner, None);
        let sig = match self.env.get(op.ty(&self.func, self.env)) {
            Ty::FnPtr { params, ret } => Some(FnSig {
                def: None,
                module: self.module,
                name: Symbol(0),
                is_pub: false,
                is_view: false,
                is_comptime: false,
                is_extern: false,
                generics: Vec::new(),
                params: params
                    .iter()
                    .map(|&t| crate::types::ParamSig {
                        name: Symbol(0),
                        span: Span::DUMMY,
                        is_self: false,
                        is_comptime: false,
                        modifier: None,
                        access: None,
                        ty: t,
                        gpu_binding: None,
                        is_push_const: false,
                    })
                    .collect(),
                ret: *ret,
                has_gpu_attr: false,
                effects: crate::types::EffectSet::default(),
            }),
            _ => None,
        };
        (CfgCallee::FnPtr(op), sig)
    }

    fn resolve_static_expr_path(&self, expr: &Expr) -> Option<TopScopeTarget> {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if self.lookup_local(*sym).is_some() || self.captured_aliases.contains_key(sym) {
                    return None;
                }
                self.prog.lookup_in_module(self.module, *sym)
            }
            ExprKind::GenericApply { base, .. } => self.resolve_static_expr_path(base),
            ExprKind::Field { base, field } => {
                let TopScopeTarget::Module(m) = self.resolve_static_expr_path(base)? else {
                    return None;
                };
                self.prog.modules[m.0 as usize].exports.get(&field.sym).map(|b| b.target)
            }
            _ => None,
        }
    }

    fn lower_schedule_build(&mut self, args: &[Expr], ret_ty: TyId) -> CfgOperand {
        let mut nodes: Vec<DefId> = Vec::new();
        let mut seen: BTreeSet<DefId> = BTreeSet::new();
        let mut edges: BTreeSet<(DefId, DefId)> = BTreeSet::new();

        for arg in args {
            match &arg.kind {
                ExprKind::AnonStructLit(fs) => {
                    for f in fs {
                        self.collect_schedule_dag_entry(&f.value, &mut nodes, &mut seen, &mut edges);
                    }
                }
                ExprKind::Tuple(es) | ExprKind::ArrayLit(es) => {
                    for e in es {
                        self.collect_schedule_dag_entry(e, &mut nodes, &mut seen, &mut edges);
                    }
                }
                _ => self.collect_schedule_dag_entry(arg, &mut nodes, &mut seen, &mut edges),
            }
        }

        let mut in_deg: BTreeMap<DefId, usize> = nodes.iter().map(|&d| (d, 0)).collect();
        for &(_, v) in &edges {
            if let Some(c) = in_deg.get_mut(&v) {
                *c += 1;
            }
        }
        let mut waves: Vec<ScheduleWave> = Vec::new();
        let mut remaining: BTreeSet<DefId> = nodes.iter().copied().collect();
        while !remaining.is_empty() {
            let ready: Vec<DefId> = nodes
                .iter()
                .copied()
                .filter(|d| remaining.contains(d) && in_deg.get(d).copied().unwrap_or(0) == 0)
                .collect();
            if ready.is_empty() {
                break;
            }
            for &d in &ready {
                remaining.remove(&d);
                for &(u, v) in &edges {
                    if u == d
                        && let Some(c) = in_deg.get_mut(&v)
                    {
                        *c = c.saturating_sub(1);
                    }
                }
            }
            waves.push(ScheduleWave { systems: ready });
        }

        let idx = self.schedules.len();
        self.schedules.push(ScheduleDag { waves });
        CfgOperand::Const(CfgConst::Int(idx as i128, ret_ty))
    }

    fn collect_schedule_dag_entry(
        &self,
        expr: &Expr,
        nodes: &mut Vec<DefId>,
        seen: &mut BTreeSet<DefId>,
        edges: &mut BTreeSet<(DefId, DefId)>,
    ) {
        let resolve_sys = |e: &Expr| -> Option<DefId> {
            if let ExprKind::Ident(sym) = &e.kind
                && let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(self.module, *sym)
                && matches!(self.prog.defs[d.0 as usize].kind, TopDefKind::Sys(_))
            {
                return Some(d);
            }
            None
        };
        let mut push_node = |d: DefId| {
            if seen.insert(d) {
                nodes.push(d);
            }
        };
        match &expr.kind {
            ExprKind::Ident(_) => {
                if let Some(d) = resolve_sys(expr) {
                    push_node(d);
                }
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::Field { base, field } = &callee.kind
                    && let Some(lhs) = resolve_sys(base)
                {
                    push_node(lhs);
                    let fname = self.interner.get(field.sym);
                    if let Some(a0) = args.first()
                        && let Some(rhs) = resolve_sys(a0)
                    {
                        push_node(rhs);
                        if fname == "before" {
                            edges.insert((lhs, rhs));
                        } else if fname == "after" {
                            edges.insert((rhs, lhs));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn lower_builtin_expr(&mut self, name: ast::IdentSpan, args: &[GenericArg], ty: TyId, span: Span) -> CfgOperand {
        let bname = self.interner.get(name.sym);
        match bname {
            "sqrt" | "fma" | "lzcnt" | "tzcnt" | "popcnt" | "bswap" => {
                let bop = match bname {
                    "sqrt" => CfgBuiltinOp::MathSqrt(ty),
                    "fma" => CfgBuiltinOp::MathFma(ty),
                    "lzcnt" => CfgBuiltinOp::MathLzcnt(ty),
                    "tzcnt" => CfgBuiltinOp::MathTzcnt(ty),
                    "popcnt" => CfgBuiltinOp::MathPopcnt(ty),
                    _ => CfgBuiltinOp::MathBswap(ty),
                };
                let mut cargs = Vec::with_capacity(args.len());
                let mut arg_spans = Vec::with_capacity(args.len());
                for a in args {
                    if let ast::GenericArgKind::Expr(e) = &a.kind {
                        cargs.push(self.lower_expr_rvalue(e, Some(ty)));
                        arg_spans.push(a.span);
                    }
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Call {
                        dst: Some(dst.clone()),
                        callee: CfgCallee::Builtin(bop),
                        args: cargs,
                        arg_spans,
                        recv_place: None,
                        recv_span: None,
                        is_mut_recv: false,
                        consumes_recv: false,
                    },
                );
                CfgOperand::Copy(dst)
            }
            "size_of" | "align_of" => {
                let val = self.table.get_const_int(span).unwrap_or(0);
                CfgOperand::Const(CfgConst::Int(val, TyId::USIZE))
            }
            "type_info" => {
                if let Some(cv) = self.table.get_comptime_val(span).cloned() {
                    return self.lower_ct_val(&cv, ty, span);
                }
                CfgOperand::Const(CfgConst::Void)
            }
            "gpu_global_id" | "gpu_local_id" | "gpu_workgroup_id" => {
                let bop = match bname {
                    "gpu_global_id" => CfgBuiltinOp::GpuGlobalId,
                    "gpu_local_id" => CfgBuiltinOp::GpuLocalId,
                    _ => CfgBuiltinOp::GpuWorkgroupId,
                };
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Call {
                        dst: Some(dst.clone()),
                        callee: CfgCallee::Builtin(bop),
                        args: Vec::new(),
                        arg_spans: Vec::new(),
                        recv_place: None,
                        recv_span: None,
                        is_mut_recv: false,
                        consumes_recv: false,
                    },
                );
                CfgOperand::Copy(dst)
            }
            "gpu_workgroup_barrier" => {
                self.push_stmt(
                    span,
                    CfgStmtKind::Call {
                        dst: None,
                        callee: CfgCallee::Builtin(CfgBuiltinOp::GpuWorkgroupBarrier),
                        args: Vec::new(),
                        arg_spans: Vec::new(),
                        recv_place: None,
                        recv_span: None,
                        is_mut_recv: false,
                        consumes_recv: false,
                    },
                );
                CfgOperand::Const(CfgConst::Void)
            }
            _ => CfgOperand::Const(CfgConst::Void),
        }
    }

    fn try_lower_len_field(&mut self, base: &Expr, span: Span) -> Option<CfgOperand> {
        let base_ty = self.expr_ty(base, TyId::ERROR);
        let deref = match self.env.get(base_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Single(t), .. } => *t,
            _ => base_ty,
        };
        let op = match self.env.get(base_ty) {
            Ty::Borrow { inner: crate::types::BorrowInner::Slice(_), .. } => Some(CfgBuiltinOp::SliceLen),
            Ty::Borrow { inner: crate::types::BorrowInner::Str, .. } | Ty::Str => Some(CfgBuiltinOp::StrLen),
            _ => match self.env.get(deref) {
                Ty::Array { .. } => Some(CfgBuiltinOp::ArrayLen),
                Ty::Builtin(BuiltinKind::ArenaPool, _) => Some(CfgBuiltinOp::ArenaPoolLen),
                Ty::Builtin(BuiltinKind::SlotPool, _) => Some(CfgBuiltinOp::SlotPoolLen),
                _ => None,
            },
        }?;
        let (recv_op, recv_place) = if matches!(
            op,
            CfgBuiltinOp::ArrayLen | CfgBuiltinOp::ArenaPoolLen | CfgBuiltinOp::SlotPoolLen
        ) && !matches!(self.env.get(base_ty), Ty::Borrow { .. })
        {
            let p = self.lower_place(base);
            (CfgOperand::Copy(p.clone()), Some(p))
        } else {
            let op_v = self.lower_expr_rvalue(base, None);
            let rp = match &op_v {
                CfgOperand::Copy(p) | CfgOperand::Move(p) => Some(p.clone()),
                _ => None,
            };
            (op_v, rp)
        };
        let dst_local = self.new_temp(TyId::USIZE, span);
        let dst_place = CfgPlace {
            local: dst_local,
            projs: Vec::new(),
            ty: TyId::USIZE,
            span,
        };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: Some(dst_place.clone()),
                callee: CfgCallee::Builtin(op),
                args: vec![recv_op],
                arg_spans: vec![base.span],
                recv_place,
                recv_span: Some(base.span),
                is_mut_recv: false,
                consumes_recv: false,
            },
        );
        Some(CfgOperand::Copy(dst_place))
    }

    fn lower_ct_val(&mut self, cv: &crate::comptime::CtVal, ty: TyId, span: Span) -> CfgOperand {
        match cv {
            crate::comptime::CtVal::Void => CfgOperand::Const(CfgConst::Void),
            crate::comptime::CtVal::Bool(b) => CfgOperand::Const(CfgConst::Bool(*b)),
            crate::comptime::CtVal::Char(c) => CfgOperand::Const(CfgConst::Char(*c)),
            crate::comptime::CtVal::Int(n, ct_ty) => {
                let final_ty = if self.env.is_integer(ty) && ty != TyId::COMPTIME_INT {
                    ty
                } else if self.env.is_integer(*ct_ty) && *ct_ty != TyId::COMPTIME_INT {
                    *ct_ty
                } else {
                    TyId::USIZE
                };
                CfgOperand::Const(CfgConst::Int(*n, final_ty))
            }
            crate::comptime::CtVal::Float(f, ct_ty) => {
                let final_ty = if self.env.is_float(ty) { ty } else { *ct_ty };
                CfgOperand::Const(CfgConst::Float(format!("{f}"), final_ty))
            }
            crate::comptime::CtVal::Str(s) => CfgOperand::Const(CfgConst::Str(s.clone())),
            crate::comptime::CtVal::EnumSym(sym, tag_val) => {
                CfgOperand::Const(CfgConst::EnumTag {
                    ty,
                    variant: *sym,
                    tag_val: *tag_val,
                })
            }
            crate::comptime::CtVal::Array(elems, elem_ty) => {
                let mut ops = Vec::with_capacity(elems.len());
                for e in elems {
                    ops.push(self.lower_ct_val(e, *elem_ty, span));
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Array(*elem_ty),
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            crate::comptime::CtVal::Tuple(elems) => {
                let etys = match self.env.get(ty) {
                    Ty::Tuple(ts) => ts.clone(),
                    _ => Vec::new(),
                };
                let mut ops = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    ops.push(self.lower_ct_val(e, etys.get(i).copied().unwrap_or(TyId::ERROR), span));
                }
                let tmp = self.new_temp(ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::Tuple,
                            elems: ops,
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            crate::comptime::CtVal::Struct { ty: st_ty, fields } => {
                let use_ty = if ty != TyId::ERROR { ty } else { *st_ty };
                match self.env.get(use_ty).clone() {
                    Ty::AnonStruct(fs) => {
                        let mut ops = Vec::with_capacity(fs.len());
                        for (fname, fty) in &fs {
                            if let Some(fv) = fields.get(fname) {
                                ops.push(self.lower_ct_val(fv, *fty, span));
                            } else {
                                ops.push(CfgOperand::Const(CfgConst::Null(*fty)));
                            }
                        }
                        let tmp = self.new_temp(use_ty, span);
                        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: use_ty, span };
                        self.push_stmt(
                            span,
                            CfgStmtKind::Assign {
                                dst: dst.clone(),
                                rvalue: CfgRvalue::Aggregate {
                                    kind: AggregateKind::AnonStruct(fs.iter().map(|(n, _)| *n).collect()),
                                    elems: ops,
                                },
                            },
                        );
                        self.operand_for_local(dst)
                    }
                    Ty::Struct { def, .. } => {
                        let mfs = self.env.structs.get(&def).map(|m| m.fields.clone()).unwrap_or_default();
                        let mut ops = Vec::with_capacity(mfs.len());
                        for mf in &mfs {
                            if let Some(fv) = fields.get(&mf.name) {
                                ops.push(self.lower_ct_val(fv, mf.ty, span));
                            } else {
                                ops.push(CfgOperand::Const(CfgConst::Null(mf.ty)));
                            }
                        }
                        let tmp = self.new_temp(use_ty, span);
                        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: use_ty, span };
                        self.push_stmt(
                            span,
                            CfgStmtKind::Assign {
                                dst: dst.clone(),
                                rvalue: CfgRvalue::Aggregate {
                                    kind: AggregateKind::Struct(def),
                                    elems: ops,
                                },
                            },
                        );
                        self.operand_for_local(dst)
                    }
                    _ => CfgOperand::Const(CfgConst::Void),
                }
            }
            crate::comptime::CtVal::TypeVal(_) => CfgOperand::Const(CfgConst::Void),
        }
    }

    fn try_lower_assoc_const_or_enum(&mut self, base: &Expr, field: ast::IdentSpan, fallback_ty: TyId) -> Option<CfgOperand> {
        let segs = self.extract_static_idents(base)?;
        if segs.len() == 1 {
            let bname = self.interner.get(segs[0]);
            let fname = self.interner.get(field.sym);
            if matches!(fname, "MIN" | "MAX")
                && let Some(pt) = self.prim_ty_by_name(bname)
                && let Some((lo, hi)) = crate::types::int_bounds(self.env.get(pt))
            {
                let v = if fname == "MIN" { lo } else { hi };
                return Some(CfgOperand::Const(CfgConst::Int(v, pt)));
            }
            match self.prog.lookup_in_module(self.module, segs[0]) {
                Some(TopScopeTarget::Def(d)) => {
                    if let Some(meta) = self.env.enums.get(&d)
                        && let Some(&(_, tv)) = meta.variants.iter().find(|(n, _)| *n == field.sym)
                    {
                        return Some(CfgOperand::Const(CfgConst::EnumTag {
                            ty: fallback_ty,
                            variant: field.sym,
                            tag_val: tv,
                        }));
                    }
                }
                Some(TopScopeTarget::Module(mod_id)) => {
                    if let Some(target @ TopScopeTarget::Def(d)) = self.prog.lookup_in_module(mod_id, field.sym)
                        && (self.env.consts.contains_key(&d) || self.env.fns.contains_key(&d))
                    {
                        return Some(self.lower_scope_target_const(target, fallback_ty));
                    }
                }
                _ => {}
            }
        } else if segs.len() == 2
            && let Some(TopScopeTarget::Module(mod_id)) = self.prog.lookup_in_module(self.module, segs[0])
            && let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(mod_id, segs[1])
            && let Some(meta) = self.env.enums.get(&d)
            && let Some(&(_, tv)) = meta.variants.iter().find(|(n, _)| *n == field.sym)
        {
            return Some(CfgOperand::Const(CfgConst::EnumTag {
                ty: fallback_ty,
                variant: field.sym,
                tag_val: tv,
            }));
        }
        None
    }

    fn eval_simple_const_int(&self, expr: &Expr) -> Option<i128> {
        match &expr.kind {
            ExprKind::IntLit { raw, .. } => {
                parse_int_literal(self.interner.get(*raw)).and_then(|u| i128::try_from(u).ok())
            }
            ExprKind::Paren(inner) | ExprKind::ComptimeExpr(inner) => self.eval_simple_const_int(inner),
            ExprKind::Ident(s) => {
                if let Some(&v) = self.consts.get(s) {
                    return Some(v);
                }
                if let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(self.module, *s) {
                    return self.env.consts.get(&d).and_then(|c| c.int_val);
                }
                None
            }
            ExprKind::Field { base, field } => {
                let segs = self.extract_static_idents(base)?;
                if segs.len() == 1
                    && let Some(TopScopeTarget::Module(mod_id)) = self.prog.lookup_in_module(self.module, segs[0])
                    && let Some(TopScopeTarget::Def(d)) = self.prog.lookup_in_module(mod_id, field.sym)
                {
                    return self.env.consts.get(&d).and_then(|c| c.int_val);
                }
                None
            }
            ExprKind::Unary { op: UnaryOp::Neg, expr: inner } => {
                self.eval_simple_const_int(inner).and_then(|v| v.checked_neg())
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_simple_const_int(lhs)?;
                let r = self.eval_simple_const_int(rhs)?;
                match op {
                    BinaryOp::Add | BinaryOp::AddWrap | BinaryOp::AddCheck => l.checked_add(r),
                    BinaryOp::Sub | BinaryOp::SubWrap | BinaryOp::SubCheck => l.checked_sub(r),
                    BinaryOp::Mul | BinaryOp::MulWrap | BinaryOp::MulCheck => l.checked_mul(r),
                    BinaryOp::Div | BinaryOp::DivCheck if r != 0 => l.checked_div(r),
                    BinaryOp::Mod | BinaryOp::ModCheck if r != 0 => l.checked_rem(r),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn extract_static_idents(&self, mut expr: &Expr) -> Option<Vec<Symbol>> {
        let mut out = Vec::new();
        loop {
            match &expr.kind {
                ExprKind::Ident(s) => {
                    if self.lookup_local(*s).is_some() {
                        return None;
                    }
                    out.push(*s);
                    break;
                }
                ExprKind::Field { base, field } => {
                    out.push(field.sym);
                    expr = base;
                }
                _ => return None,
            }
        }
        out.reverse();
        Some(out)
    }

    fn prim_ty_by_name(&self, s: &str) -> Option<TyId> {
        Some(match s {
            "u8" => TyId::U8,
            "u16" => TyId::U16,
            "u32" => TyId::U32,
            "u64" => TyId::U64,
            "u128" => TyId::U128,
            "usize" => TyId::USIZE,
            "i8" => TyId::I8,
            "i16" => TyId::I16,
            "i32" => TyId::I32,
            "i64" => TyId::I64,
            "i128" => TyId::I128,
            "isize" => TyId::ISIZE,
            _ => return None,
        })
    }

    fn lower_scope_target_const(&mut self, target: TopScopeTarget, expected_ty: TyId) -> CfgOperand {
        match target {
            TopScopeTarget::Def(d) => {
                if let Some(c) = self.env.consts.get(&d) {
                    if let Some(iv) = c.int_val {
                        return CfgOperand::Const(CfgConst::Int(iv, c.ty));
                    }
                    if let Some(crate::comptime::CtVal::Bool(b)) = c.ct_val {
                        return CfgOperand::Const(CfgConst::Bool(b));
                    }
                    if let Some(crate::comptime::CtVal::Str(ref s)) = c.ct_val {
                        return CfgOperand::Const(CfgConst::Str(s.clone()));
                    }
                    if c.ct_val.is_some() {
                        return CfgOperand::Const(CfgConst::ConstDef(d, c.ty));
                    }
                }
                if let Some(sig) = self.env.fns.get(&d).cloned() {
                    let fn_ty = if matches!(self.env.get(expected_ty), Ty::FnPtr { .. }) {
                        expected_ty
                    } else {
                        let params: Vec<TyId> = sig.params.iter().map(|p| p.ty).collect();
                        self.env.intern(Ty::FnPtr { params, ret: sig.ret })
                    };
                    return CfgOperand::Const(CfgConst::FnRef(d, fn_ty));
                }
                CfgOperand::Const(CfgConst::Null(expected_ty))
            }
            _ => CfgOperand::Const(CfgConst::Void),
        }
    }

    fn lower_enum_or_error_lit(&mut self, v: ast::IdentSpan, ty: TyId) -> CfgOperand {
        match self.env.get(ty).clone() {
            Ty::Enum(d) => {
                let tag_val = self
                    .env
                    .enums
                    .get(&d)
                    .and_then(|m| m.variants.iter().find(|(n, _)| *n == v.sym).map(|(_, x)| *x))
                    .unwrap_or(0);
                CfgOperand::Const(CfgConst::EnumTag { ty, variant: v.sym, tag_val })
            }
            Ty::Optional(inner) => {
                let inner_op = self.lower_enum_or_error_lit(v, inner);
                self.coerce_operand(inner_op, ty, v.span)
            }
            Ty::ErrorSet(_) => {
                let tag_val = self.error_variant_tag(ty, v.sym);
                CfgOperand::Const(CfgConst::ErrorTag { ty, variant: v.sym, tag_val })
            }
            Ty::ErrorUnion { ok, errors } => {
                if self.env.error_sources_have_variant(&errors, v.sym, self.interner) {
                    let tag_val = self.error_variant_tag(ty, v.sym);
                    CfgOperand::Const(CfgConst::ErrorTag { ty, variant: v.sym, tag_val })
                } else {
                    let ok_op = self.lower_enum_or_error_lit(v, ok);
                    self.coerce_operand(ok_op, ty, v.span)
                }
            }
            Ty::Union { def, .. } => {
                let vidx = self.variant_tag_index(ty, v.sym).unwrap_or(0);
                let tmp = self.new_temp(ty, v.span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty, span: v.span };
                self.push_stmt(
                    v.span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Aggregate {
                            kind: AggregateKind::UnionVariant { def, variant_idx: vidx },
                            elems: Vec::new(),
                        },
                    },
                );
                self.operand_for_local(dst)
            }
            Ty::Builtin(BuiltinKind::Ordering, _) => {
                let tv = match self.interner.get(v.sym) {
                    "Less" | "Relaxed" => 0,
                    "Equal" | "Acquire" => 1,
                    "Greater" | "Release" => 2,
                    "AcqRel" => 3,
                    "SeqCst" => 4,
                    _ => 4,
                };
                CfgOperand::Const(CfgConst::EnumTag { ty, variant: v.sym, tag_val: tv })
            }
            _ => CfgOperand::Const(CfgConst::Void),
        }
    }

    fn variant_tag_index(&self, ty: TyId, variant: Symbol) -> Option<usize> {
        match self.env.get(ty) {
            Ty::Enum(d) => self.env.enums.get(d).and_then(|m| m.variants.iter().position(|(n, _)| *n == variant)),
            Ty::Union { def, .. } => self.env.unions.get(def).and_then(|m| m.variants.iter().position(|v| v.name == variant)),
            Ty::Builtin(BuiltinKind::Ordering, _) => Some(match self.interner.get(variant) {
                "Less" => 0,
                "Equal" => 1,
                _ => 2,
            }),
            _ => None,
        }
    }

    fn error_variant_tag(&self, ty: TyId, variant: Symbol) -> u32 {
        let name = self.interner.get(variant);
        match name {
            "Overflow" => 1,
            "DivByZero" => 2,
            "Underflow" => 3,
            "OutOfBounds" => 4,
            "InvalidUtf8" => 5,
            "UnexpectedEof" => 6,
            "OutOfMemory" => 7,
            "InvalidSize" => 8,
            "BadAlignment" => 9,
            "AllocFailed" => 10,
            "StaleId" => 11,
            "NotFound" => 12,
            "PermissionDenied" => 13,
            "PathEscape" => 14,
            "AlreadyExists" => 15,
            "InvalidInput" => 16,
            "Other" => 17,
            _ => {
                let srcs: &[ErrorSource] = match self.env.get(ty) {
                    Ty::ErrorSet(s) => std::slice::from_ref(s),
                    Ty::ErrorUnion { errors, .. } => errors.as_slice(),
                    _ => &[],
                };
                for src in srcs {
                    if let ErrorSource::User(d) = src
                        && let Some(meta) = self.env.errors.get(d)
                        && let Some(idx) = meta.variants.iter().position(|&v| v == variant)
                    {
                        return 100 + (d.0 * 64) + (idx as u32);
                    }
                }
                if let Ty::ErrorUnion { errors, .. } = self.env.get(self.func.sig.ret) {
                    for src in errors {
                        if let ErrorSource::User(d) = src
                            && let Some(meta) = self.env.errors.get(d)
                            && let Some(idx) = meta.variants.iter().position(|&v| v == variant)
                        {
                            return 100 + (d.0 * 64) + (idx as u32);
                        }
                    }
                }
                99
            }
        }
    }

    fn coerce_error_into_union(&mut self, op: CfgOperand, span: Span, target_union: TyId) -> CfgOperand {
        let tmp = self.new_temp(target_union, span);
        let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: target_union, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: dst.clone(),
                rvalue: CfgRvalue::Cast {
                    kind: CastKind::WrapErr,
                    operand: op,
                    to_ty: target_union,
                },
            },
        );
        self.operand_for_local(dst)
    }

    /// Lower `x.try_into.[T]()` (type `T ! ArithError`) into explicit range checks:
    /// `if x >= T.MIN && x <= T.MAX { ok(cast(x)) } else { err(.Overflow) }`.
    /// Checks implied by the source type's own range are omitted, so `u8 -> u32` is a plain cast.
    /// Backends need nothing new: only compares, branches, casts and `WrapOk`/`WrapErr`.
    fn lower_try_into(&mut self, rop: CfgOperand, ret_ty: TyId, sym_for_tag: Symbol, span: Span) -> CfgOperand {
        let from_ty = rop.ty(&self.func, self.env);
        let to_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => ret_ty,
        };
        let (Some((flo, fhi)), Some((tlo, thi))) = (
            crate::types::int_bounds(self.env.get(from_ty)),
            crate::types::int_bounds(self.env.get(to_ty)),
        ) else {
            return CfgOperand::Const(CfgConst::Void);
        };
        // Range sizes are exact powers of two (`2^bits`), so the width is the trailing-zero count.
        let from_bits = ((fhi - flo) as u128 + 1).trailing_zeros();
        let to_bits = ((thi - tlo) as u128 + 1).trailing_zeros();
        let cast_kind = if to_bits > from_bits { CastKind::Widen } else { CastKind::Trunc };

        let res = self.new_temp(ret_ty, span);
        let res_place = CfgPlace { local: res, projs: Vec::new(), ty: ret_ty, span };
        let ok_bb = self.new_block();
        let err_bb = self.new_block();
        let merge_bb = self.new_block();

        // Range checks against the bounds of `T` that the source type can actually exceed.
        let mut checks: Vec<(BinaryOp, i128)> = Vec::new();
        if tlo > flo {
            checks.push((BinaryOp::GtEq, tlo));
        }
        if thi < fhi {
            checks.push((BinaryOp::LtEq, thi));
        }
        if checks.is_empty() {
            self.set_term(CfgTerm::Goto(ok_bb));
        }
        let n_checks = checks.len();
        for (i, (op, bound)) in checks.into_iter().enumerate() {
            let c = self.new_temp(TyId::BOOL, span);
            let c_place = CfgPlace { local: c, projs: Vec::new(), ty: TyId::BOOL, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: c_place.clone(),
                    rvalue: CfgRvalue::Binary {
                        op,
                        lhs: rop.clone(),
                        rhs: CfgOperand::Const(CfgConst::Int(bound, from_ty)),
                    },
                },
            );
            let next_bb = if i + 1 == n_checks { ok_bb } else { self.new_block() };
            self.set_term(CfgTerm::Branch {
                cond: CfgOperand::Copy(c_place),
                then_bb: next_bb,
                else_bb: err_bb,
                span,
                is_user_if: false,
            });
            self.cur_bb = Some(next_bb);
        }

        // ok: res = WrapOk(cast(x))
        self.cur_bb = Some(ok_bb);
        let v = self.new_temp(to_ty, span);
        let v_place = CfgPlace { local: v, projs: Vec::new(), ty: to_ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: v_place.clone(),
                rvalue: CfgRvalue::Cast { kind: cast_kind, operand: rop, to_ty },
            },
        );
        let wrapped = self.coerce_operand(CfgOperand::Copy(v_place), ret_ty, span);
        self.push_stmt(span, CfgStmtKind::Assign { dst: res_place.clone(), rvalue: CfgRvalue::Use(wrapped) });
        self.set_term(CfgTerm::Goto(merge_bb));

        // err: res = WrapErr(ArithError.Overflow). Backends only read `tag_val`; the builtin
        // `Overflow` tag is fixed by `error_variant_tag`.
        self.cur_bb = Some(err_bb);
        let tag = CfgOperand::Const(CfgConst::ErrorTag { ty: ret_ty, variant: sym_for_tag, tag_val: 1 });
        // Explicit `WrapErr`: `coerce_operand` would see the tag's `u32` type and pick `WrapOk`
        // whenever `T == u32`.
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: res_place.clone(),
                rvalue: CfgRvalue::Cast { kind: CastKind::WrapErr, operand: tag, to_ty: ret_ty },
            },
        );
        self.set_term(CfgTerm::Goto(merge_bb));

        self.cur_bb = Some(merge_bb);
        CfgOperand::Copy(res_place)
    }

    fn coerce_operand(&mut self, op: CfgOperand, to_ty: TyId, span: Span) -> CfgOperand {
        if to_ty == TyId::ERROR || to_ty == TyId::VOID {
            return op;
        }
        let from_ty = op.ty(&self.func, self.env);
        if from_ty == to_ty || from_ty == TyId::ERROR || from_ty == TyId::NEVER {
            return op;
        }
        if matches!(op, CfgOperand::Const(CfgConst::Int(_, _)))
            && self.env.is_integer(to_ty)
            && let CfgOperand::Const(CfgConst::Int(v, _)) = op
        {
            return CfgOperand::Const(CfgConst::Int(v, to_ty));
        }
        if matches!(op, CfgOperand::Const(CfgConst::Float(_, _)))
            && self.env.is_float(to_ty)
            && let CfgOperand::Const(CfgConst::Float(ref s, _)) = op
        {
            return CfgOperand::Const(CfgConst::Float(s.clone(), to_ty));
        }
        if let (
            Ty::Borrow { inner: crate::types::BorrowInner::Single(concrete_ty), .. },
            Ty::Borrow { is_mut, inner: crate::types::BorrowInner::Single(dyn_ty) },
        ) = (self.env.get(from_ty).clone(), self.env.get(to_ty).clone())
            && let Ty::Dyn(iface_def) = self.env.get(dyn_ty).clone()
            && !matches!(self.env.get(concrete_ty), Ty::Dyn(_))
        {
            let tmp = self.new_temp(to_ty, span);
            let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: to_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: dst.clone(),
                    rvalue: CfgRvalue::MakeDyn {
                        ptr: op,
                        iface_def,
                        concrete_ty,
                        is_mut,
                    },
                },
            );
            return self.operand_for_local(dst);
        }
        // Handle chained coercions into `T ! E` or `?T` (e.g. `Id[ast] -> ?Id[ast] -> ?Id[ast] ! E`, `*mut [N]T -> *[T] -> ?*[T]`).
        if let Ty::ErrorUnion { ok, .. } = self.env.get(to_ty).clone()
            && from_ty != ok
            && !matches!(op, CfgOperand::Const(CfgConst::ErrorTag { .. }))
            && !matches!(self.env.get(from_ty), Ty::ErrorSet(_))
        {
            let coerced_ok = self.coerce_operand(op, ok, span);
            if coerced_ok.ty(&self.func, self.env) == ok {
                let tmp = self.new_temp(to_ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: to_ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Cast { kind: CastKind::WrapOk, operand: coerced_ok, to_ty },
                    },
                );
                return self.operand_for_local(dst);
            }
            return coerced_ok;
        }
        if let Ty::Optional(inner) = self.env.get(to_ty).clone()
            && from_ty != inner
            && !matches!(op, CfgOperand::Const(CfgConst::Null(_)))
        {
            let coerced_inner = self.coerce_operand(op, inner, span);
            if coerced_inner.ty(&self.func, self.env) == inner {
                let tmp = self.new_temp(to_ty, span);
                let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: to_ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: dst.clone(),
                        rvalue: CfgRvalue::Cast { kind: CastKind::WrapOptional, operand: coerced_inner, to_ty },
                    },
                );
                return self.operand_for_local(dst);
            }
            return coerced_inner;
        }
        let cast_kind = match (self.env.get(from_ty), self.env.get(to_ty)) {
            (_, Ty::Optional(inner)) if from_ty == *inner => Some(CastKind::WrapOptional),
            (_, Ty::ErrorUnion { ok, .. }) if from_ty == *ok => Some(CastKind::WrapOk),
            (_, Ty::ErrorUnion { .. }) if matches!(op, CfgOperand::Const(CfgConst::ErrorTag { .. })) || matches!(self.env.get(from_ty), Ty::ErrorSet(_)) => {
                Some(CastKind::WrapErr)
            }
            (
                Ty::Borrow { is_mut: true, inner: fi },
                Ty::Borrow { is_mut: false, inner: ti },
            ) if fi == ti => Some(CastKind::MutToConst),
            (
                Ty::Borrow { inner: crate::types::BorrowInner::Single(arr), .. },
                Ty::Borrow { inner: crate::types::BorrowInner::Slice(_), .. },
            ) if matches!(self.env.get(*arr), Ty::Array { .. }) => Some(CastKind::ArrayBorrowToSlice),
            _ => None,
        };
        if let Some(kind) = cast_kind {
            let tmp = self.new_temp(to_ty, span);
            let dst = CfgPlace { local: tmp, projs: Vec::new(), ty: to_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: dst.clone(),
                    rvalue: CfgRvalue::Cast { kind, operand: op, to_ty },
                },
            );
            return self.operand_for_local(dst);
        }
        op
    }

    fn pick_concrete_int_ty(&self, ty: TyId, expected: Option<TyId>) -> TyId {
        if self.env.is_concrete_integer(ty) {
            ty
        } else if let Some(exp) = expected
            && self.env.is_concrete_integer(exp)
        {
            exp
        } else if let Some(exp) = expected {
            match self.env.get(exp) {
                Ty::Optional(inner) | Ty::ErrorUnion { ok: inner, .. } if self.env.is_concrete_integer(*inner) => *inner,
                _ => TyId::I64,
            }
        } else {
            TyId::I64
        }
    }

    fn lower_buf_drain(
        &mut self,
        rplace: &CfgPlace,
        recv_span: Span,
        elem_ty: TyId,
        args: &[Expr],
        span: Span,
    ) -> CfgOperand {
        let alloc_op = if args.len() >= 2 {
            Some((self.lower_expr_rvalue(&args[0], None), args[0].span))
        } else {
            None
        };
        let cb_ty = self.env.intern(Ty::FnPtr {
            params: vec![elem_ty],
            ret: TyId::VOID,
        });
        let (cb_op, cb_span) = if let Some(cb_expr) = args.last() {
            let op = self.lower_expr_rvalue(cb_expr, Some(cb_ty));
            let cop = self.coerce_operand(op, cb_ty, cb_expr.span);
            (cop, cb_expr.span)
        } else {
            return CfgOperand::Const(CfgConst::Void);
        };
        let cb_tmp = self.new_temp(cb_ty, cb_span);
        let cb_place = CfgPlace { local: cb_tmp, projs: Vec::new(), ty: cb_ty, span: cb_span };
        self.push_stmt(
            cb_span,
            CfgStmtKind::Assign {
                dst: cb_place.clone(),
                rvalue: CfgRvalue::Use(cb_op),
            },
        );

        let opt_ty = self.env.intern(Ty::Optional(elem_ty));
        let opt_loc = self.new_temp(opt_ty, span);
        let opt_place = CfgPlace { local: opt_loc, projs: Vec::new(), ty: opt_ty, span };
        let has_loc = self.new_temp(TyId::BOOL, span);
        let has_place = CfgPlace { local: has_loc, projs: Vec::new(), ty: TyId::BOOL, span };
        let item_loc = self.new_temp(elem_ty, span);
        let item_place = CfgPlace { local: item_loc, projs: Vec::new(), ty: elem_ty, span };

        let header_bb = self.new_block();
        self.func.blocks[header_bb.0 as usize].is_loop_header = true;
        let body_bb = self.new_block();
        let done_bb = self.new_block();

        self.set_term(CfgTerm::Goto(header_bb));
        self.cur_bb = Some(header_bb);
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: Some(opt_place.clone()),
                callee: CfgCallee::Builtin(CfgBuiltinOp::BufPop(elem_ty)),
                args: vec![CfgOperand::Copy(rplace.clone())],
                arg_spans: vec![recv_span],
                recv_place: Some(rplace.clone()),
                recv_span: Some(recv_span),
                is_mut_recv: true,
                consumes_recv: false,
            },
        );
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: has_place.clone(),
                rvalue: CfgRvalue::IsNonNull(CfgOperand::Copy(opt_place.clone())),
            },
        );
        self.set_term(CfgTerm::Branch {
            cond: CfgOperand::Copy(has_place),
            then_bb: body_bb,
            else_bb: done_bb,
            span,
            is_user_if: false,
        });

        self.cur_bb = Some(body_bb);
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: item_place.clone(),
                rvalue: CfgRvalue::UnwrapOptional(CfgOperand::Move(opt_place)),
            },
        );
        let item_op = if self.env.is_linear(elem_ty) {
            CfgOperand::Move(item_place)
        } else {
            CfgOperand::Copy(item_place)
        };
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: None,
                callee: CfgCallee::FnPtr(CfgOperand::Copy(cb_place)),
                args: vec![item_op],
                arg_spans: vec![span],
                recv_place: None,
                recv_span: None,
                is_mut_recv: false,
                consumes_recv: false,
            },
        );
        self.set_term(CfgTerm::Goto(header_bb));

        self.cur_bb = Some(done_bb);
        let mut d_args = vec![CfgOperand::Copy(rplace.clone())];
        let mut d_spans = vec![recv_span];
        if let Some((aop, aspan)) = alloc_op {
            d_args.push(aop);
            d_spans.push(aspan);
        }
        self.push_stmt(
            span,
            CfgStmtKind::Call {
                dst: None,
                callee: CfgCallee::Builtin(CfgBuiltinOp::BufDestroy(elem_ty)),
                args: d_args,
                arg_spans: d_spans,
                recv_place: Some(rplace.clone()),
                recv_span: Some(recv_span),
                is_mut_recv: true,
                consumes_recv: true,
            },
        );
        CfgOperand::Const(CfgConst::Void)
    }

    fn lower_closure(
        &mut self,
        params: &[ast::Param],
        body: &Block,
        ty: TyId,
        expected: Option<TyId>,
        span: Span,
    ) -> CfgOperand {
        let raw_fn_ty = if matches!(self.env.get(ty), Ty::FnPtr { .. }) {
            ty
        } else if let Some(exp) = expected
            && matches!(self.env.get(exp), Ty::FnPtr { .. })
        {
            exp
        } else {
            ty
        };
        let fn_ty = self.subst(raw_fn_ty);
        let (param_tys, ret_ty) = match self.env.get(fn_ty).clone() {
            Ty::FnPtr { params: ps, ret } => (ps, ret),
            _ => {
                let ps: Vec<TyId> = params
                    .iter()
                    .map(|p| {
                        let pt = self.raw_ty(p.span);
                        self.subst(if pt != TyId::ERROR { pt } else { TyId::U64 })
                    })
                    .collect();
                let rt = TyId::VOID;
                (ps, rt)
            }
        };
        let final_fn_ty = self.env.intern(Ty::FnPtr {
            params: param_tys.clone(),
            ret: ret_ty,
        });

        let mut bound = BTreeSet::new();
        for p in params {
            bound.insert(p.name.sym);
        }
        let mut free_vars: BTreeMap<Symbol, bool> = BTreeMap::new();
        let mut uses_self = false;
        collect_free_vars_block(body, &bound, &mut free_vars, &mut uses_self, self.interner);

        // Build captures list from outer locals (or outer captured_aliases for nested closures).
        enum CapSrc {
            Sym(Symbol),
            SelfCap,
        }
        let mut cap_entries: Vec<(CapSrc, CfgPlace, bool, TyId, TyId)> = Vec::new();
        if uses_self {
            if let Some((env_loc, fidx, ptr_ty, inner_ty)) = self.captured_self {
                let outer_place = CfgPlace {
                    local: env_loc,
                    projs: vec![
                        CfgProj::Deref,
                        CfgProj::TupleField { index: fidx, ty: ptr_ty },
                        CfgProj::Deref,
                    ],
                    ty: inner_ty,
                    span,
                };
                let is_mut = matches!(self.env.get(ptr_ty), Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. });
                let cap_ptr_ty = if matches!(self.env.get(inner_ty), Ty::Pool { .. }) {
                    self.env.intern(Ty::RawPtr { is_mut, inner: inner_ty })
                } else {
                    self.env.intern(Ty::Borrow {
                        is_mut,
                        inner: crate::types::BorrowInner::Single(inner_ty),
                    })
                };
                cap_entries.push((CapSrc::SelfCap, outer_place, is_mut, cap_ptr_ty, inner_ty));
            } else if let Some(&self_loc) = self.func.params.first()
                && self.func.locals[self_loc.0 as usize].is_self
            {
                let inner_ty = self.func.locals[self_loc.0 as usize].ty;
                let is_mut = self.func.locals[self_loc.0 as usize].is_mut;
                let outer_place = CfgPlace { local: self_loc, projs: Vec::new(), ty: inner_ty, span };
                let cap_ptr_ty = if matches!(self.env.get(inner_ty), Ty::Pool { .. }) {
                    self.env.intern(Ty::RawPtr { is_mut, inner: inner_ty })
                } else {
                    self.env.intern(Ty::Borrow {
                        is_mut,
                        inner: crate::types::BorrowInner::Single(inner_ty),
                    })
                };
                cap_entries.push((CapSrc::SelfCap, outer_place, is_mut, cap_ptr_ty, inner_ty));
            }
        }
        for (sym, is_mutated) in free_vars {
            if let Some(outer_loc) = self.lookup_local(sym) {
                let inner_ty = self.func.locals[outer_loc.0 as usize].ty;
                let is_mut = self.func.locals[outer_loc.0 as usize].is_mut && is_mutated;
                let outer_place = CfgPlace { local: outer_loc, projs: Vec::new(), ty: inner_ty, span };
                let cap_ptr_ty = if matches!(self.env.get(inner_ty), Ty::Pool { .. }) {
                    self.env.intern(Ty::RawPtr { is_mut, inner: inner_ty })
                } else {
                    self.env.intern(Ty::Borrow {
                        is_mut,
                        inner: crate::types::BorrowInner::Single(inner_ty),
                    })
                };
                cap_entries.push((CapSrc::Sym(sym), outer_place, is_mut, cap_ptr_ty, inner_ty));
            } else if let Some(&(env_loc, fidx, ptr_ty, inner_ty)) = self.captured_aliases.get(&sym) {
                let outer_place = CfgPlace {
                    local: env_loc,
                    projs: vec![
                        CfgProj::Deref,
                        CfgProj::TupleField { index: fidx, ty: ptr_ty },
                        CfgProj::Deref,
                    ],
                    ty: inner_ty,
                    span,
                };
                let outer_mut = matches!(self.env.get(ptr_ty), Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. });
                let is_mut = outer_mut && is_mutated;
                let cap_ptr_ty = if matches!(self.env.get(inner_ty), Ty::Pool { .. }) {
                    self.env.intern(Ty::RawPtr { is_mut, inner: inner_ty })
                } else {
                    self.env.intern(Ty::Borrow {
                        is_mut,
                        inner: crate::types::BorrowInner::Single(inner_ty),
                    })
                };
                cap_entries.push((CapSrc::Sym(sym), outer_place, is_mut, cap_ptr_ty, inner_ty));
            }
        }

        let cid = *self.next_closure_id;
        *self.next_closure_id += 1;

        let (env_ptr_ty, env_ptr_op) = if cap_entries.is_empty() {
            let void_ptr = self.env.intern(Ty::RawPtr { is_mut: true, inner: TyId::VOID });
            (void_ptr, None)
        } else {
            let mut ptr_tys = Vec::with_capacity(cap_entries.len());
            let mut ptr_ops = Vec::with_capacity(cap_entries.len());
            for (_, outer_place, is_mut, cap_ptr_ty, _) in &cap_entries {
                ptr_tys.push(*cap_ptr_ty);
                let ptmp = self.new_temp(*cap_ptr_ty, span);
                let pdst = CfgPlace { local: ptmp, projs: Vec::new(), ty: *cap_ptr_ty, span };
                self.push_stmt(
                    span,
                    CfgStmtKind::Assign {
                        dst: pdst.clone(),
                        rvalue: CfgRvalue::Borrow {
                            is_mut: *is_mut,
                            place: outer_place.clone(),
                        },
                    },
                );
                ptr_ops.push(CfgOperand::Copy(pdst));
            }
            let env_tup_ty = self.env.intern(Ty::Tuple(ptr_tys));
            let env_tup_loc = self.new_temp(env_tup_ty, span);
            let env_tup_place = CfgPlace { local: env_tup_loc, projs: Vec::new(), ty: env_tup_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: env_tup_place.clone(),
                    rvalue: CfgRvalue::Aggregate {
                        kind: AggregateKind::Tuple,
                        elems: ptr_ops,
                    },
                },
            );
            let env_ptr_ty = self.env.intern(Ty::Borrow {
                is_mut: true,
                inner: crate::types::BorrowInner::Single(env_tup_ty),
            });
            let env_ptr_loc = self.new_temp(env_ptr_ty, span);
            let env_ptr_place = CfgPlace { local: env_ptr_loc, projs: Vec::new(), ty: env_ptr_ty, span };
            self.push_stmt(
                span,
                CfgStmtKind::Assign {
                    dst: env_ptr_place.clone(),
                    rvalue: CfgRvalue::Borrow {
                        is_mut: true,
                        place: env_tup_place,
                    },
                },
            );
            (env_ptr_ty, Some(CfgOperand::Copy(env_ptr_place)))
        };

        // Construct closure FnSig: param 0 is hidden `_env`, params 1.. are closure params.
        let mut cl_params = Vec::with_capacity(params.len() + 1);
        cl_params.push(crate::types::ParamSig {
            name: Symbol(0),
            span,
            is_self: false,
            is_comptime: false,
            modifier: None,
            access: None,
            ty: env_ptr_ty,
            gpu_binding: None,
            is_push_const: false,
        });
        for (i, p) in params.iter().enumerate() {
            let pty = param_tys.get(i).copied().unwrap_or(TyId::U64);
            cl_params.push(crate::types::ParamSig {
                name: p.name.sym,
                span: p.span,
                is_self: false,
                is_comptime: false,
                modifier: p.modifier,
                access: None,
                ty: pty,
                gpu_binding: None,
                is_push_const: false,
            });
        }
        let cl_sig = FnSig {
            def: None,
            module: self.module,
            name: Symbol(0),
            is_pub: false,
            is_view: false,
            is_comptime: false,
            is_extern: false,
            generics: Vec::new(),
            params: cl_params,
            ret: ret_ty,
            has_gpu_attr: false,
            effects: self.func.sig.effects.clone(),
        };

        let mut cb = CfgBuilder::new(
            CfgFnKey::Closure(cid),
            self.module,
            cl_sig,
            body.span,
            self.generic_subst.clone(),
            None,
            self.generic_inst_map,
            self.pending_insts,
            self.out_closures,
            self.next_closure_id,
            self.schedules,
            self.prog,
            self.env,
            self.table,
            self.interner,
        );
        cb.bind_params();
        let env_local = cb.func.params[0];
        for (fidx, (src, _, _, cap_ptr_ty, inner_ty)) in cap_entries.iter().enumerate() {
            match src {
                CapSrc::Sym(s) => {
                    cb.captured_aliases.insert(*s, (env_local, fidx, *cap_ptr_ty, *inner_ty));
                }
                CapSrc::SelfCap => {
                    cb.captured_self = Some((env_local, fidx, *cap_ptr_ty, *inner_ty));
                }
            }
        }
        cb.lower_body(body);
        let cl_func = cb.finish();
        self.out_closures.push(cl_func);

        let dst_loc = self.new_temp(final_fn_ty, span);
        let dst = CfgPlace { local: dst_loc, projs: Vec::new(), ty: final_fn_ty, span };
        self.push_stmt(
            span,
            CfgStmtKind::Assign {
                dst: dst.clone(),
                rvalue: CfgRvalue::MakeClosure {
                    closure_id: cid,
                    env_ptr: env_ptr_op,
                    fn_ty: final_fn_ty,
                },
            },
        );
        self.operand_for_local(dst)
    }

    fn finish(self) -> CfgFunction {
        self.func
    }
}

fn collect_pat_bindings(pat: &Pat, bound: &mut BTreeSet<Symbol>) {
    match &pat.kind {
        PatKind::Ident(s) => {
            bound.insert(*s);
        }
        PatKind::Variant { args, .. } | PatKind::Tuple(args) => {
            for a in args {
                collect_pat_bindings(a, bound);
            }
        }
        PatKind::Struct { fields, .. } => {
            for f in fields {
                if let Some(p) = &f.pat {
                    collect_pat_bindings(p, bound);
                } else {
                    bound.insert(f.name.sym);
                }
            }
        }
        _ => {}
    }
}

fn collect_free_vars_block(
    block: &Block,
    outer_bound: &BTreeSet<Symbol>,
    out: &mut BTreeMap<Symbol, bool>,
    uses_self: &mut bool,
    interner: &Interner,
) {
    let mut bound = outer_bound.clone();
    for stmt in &block.stmts {
        collect_free_vars_stmt(stmt, &mut bound, out, uses_self, interner);
    }
    if let Some(tail) = &block.tail_expr {
        collect_free_vars_expr(tail, &bound, false, out, uses_self, interner);
    }
}

fn collect_free_vars_stmt(
    stmt: &Stmt,
    bound: &mut BTreeSet<Symbol>,
    out: &mut BTreeMap<Symbol, bool>,
    uses_self: &mut bool,
    interner: &Interner,
) {
    match &stmt.kind {
        StmtKind::Let { pat, init, .. } => {
            collect_free_vars_expr(init, bound, false, out, uses_self, interner);
            collect_pat_bindings(pat, bound);
        }
        StmtKind::Var { name, init, .. } => {
            collect_free_vars_expr(init, bound, false, out, uses_self, interner);
            bound.insert(name.sym);
        }
        StmtKind::Const(c) => {
            collect_free_vars_expr(&c.value, bound, false, out, uses_self, interner);
            bound.insert(c.name.sym);
        }
        StmtKind::Defer(body) => match body {
            DeferBody::Block(b) => collect_free_vars_block(b, bound, out, uses_self, interner),
            DeferBody::Expr(e) => collect_free_vars_expr(e, bound, false, out, uses_self, interner),
        },
        StmtKind::Errdefer { err_bind, body } => {
            let mut inner = bound.clone();
            if let Some(eb) = err_bind {
                inner.insert(eb.sym);
            }
            match body {
                DeferBody::Block(b) => collect_free_vars_block(b, &inner, out, uses_self, interner),
                DeferBody::Expr(e) => collect_free_vars_expr(e, &inner, false, out, uses_self, interner),
            }
        }
        StmtKind::Jump(j) => match j {
            JumpAction::Return(Some(e)) | JumpAction::Break { value: Some(e), .. } => {
                collect_free_vars_expr(e, bound, false, out, uses_self, interner);
            }
            _ => {}
        },
        StmtKind::If { data, .. } => collect_free_vars_if(data, bound, out, uses_self, interner),
        StmtKind::Match { scrutinee, arms } => {
            collect_free_vars_expr(scrutinee, bound, false, out, uses_self, interner);
            for arm in arms {
                let mut inner = bound.clone();
                collect_pat_bindings(&arm.pat, &mut inner);
                if let Some(g) = &arm.guard {
                    collect_free_vars_expr(g, &inner, false, out, uses_self, interner);
                }
                match &arm.body {
                    MatchArmBody::Block(b) => collect_free_vars_block(b, &inner, out, uses_self, interner),
                    MatchArmBody::Expr(e) => collect_free_vars_expr(e, &inner, false, out, uses_self, interner),
                    MatchArmBody::Jump(JumpAction::Return(Some(e)))
                    | MatchArmBody::Jump(JumpAction::Break { value: Some(e), .. }) => {
                        collect_free_vars_expr(e, &inner, false, out, uses_self, interner);
                    }
                    MatchArmBody::Jump(_) => {}
                }
            }
        }
        StmtKind::While { cond, where_clauses, body, .. } => {
            collect_free_vars_expr(cond, bound, false, out, uses_self, interner);
            for w in where_clauses {
                collect_free_vars_expr(w, bound, false, out, uses_self, interner);
            }
            collect_free_vars_block(body, bound, out, uses_self, interner);
        }
        StmtKind::For { pat, iter_start, iter_end, body, .. }
        | StmtKind::ParFor { pat, iter_start, iter_end, body } => {
            collect_free_vars_expr(iter_start, bound, false, out, uses_self, interner);
            if let Some((_, end_e)) = iter_end {
                collect_free_vars_expr(end_e, bound, false, out, uses_self, interner);
            }
            let mut inner = bound.clone();
            collect_pat_bindings(pat, &mut inner);
            collect_free_vars_block(body, &inner, out, uses_self, interner);
        }
        StmtKind::Loop { body, .. }
        | StmtKind::ComptimeBlock(body)
        | StmtKind::RawBlock(body) => {
            collect_free_vars_block(body, bound, out, uses_self, interner);
        }
        StmtKind::Assign { lhs, rhs, .. } => {
            collect_free_vars_expr(lhs, bound, true, out, uses_self, interner);
            collect_free_vars_expr(rhs, bound, false, out, uses_self, interner);
        }
        StmtKind::Discard(e) | StmtKind::Expr(e) => {
            collect_free_vars_expr(e, bound, false, out, uses_self, interner);
        }
    }
}

fn collect_free_vars_if(
    data: &IfExpr,
    bound: &BTreeSet<Symbol>,
    out: &mut BTreeMap<Symbol, bool>,
    uses_self: &mut bool,
    interner: &Interner,
) {
    let mut then_bound = bound.clone();
    match &data.cond {
        IfCond::Expr(e) => collect_free_vars_expr(e, bound, false, out, uses_self, interner),
        IfCond::Let { pat, expr } => {
            collect_free_vars_expr(expr, bound, false, out, uses_self, interner);
            collect_pat_bindings(pat, &mut then_bound);
        }
    }
    collect_free_vars_block(&data.then_block, &then_bound, out, uses_self, interner);
    if let Some(eb) = &data.else_branch {
        match eb {
            ElseBranch::Block(b) => collect_free_vars_block(b, bound, out, uses_self, interner),
            ElseBranch::If { data: next, .. } => collect_free_vars_if(next, bound, out, uses_self, interner),
        }
    }
}

fn collect_free_vars_expr(
    expr: &Expr,
    bound: &BTreeSet<Symbol>,
    mutated: bool,
    out: &mut BTreeMap<Symbol, bool>,
    uses_self: &mut bool,
    interner: &Interner,
) {
    match &expr.kind {
        ExprKind::Ident(sym) => {
            if !bound.contains(sym) {
                out.entry(*sym).and_modify(|m| *m |= mutated).or_insert(mutated);
            }
        }
        ExprKind::SelfVal => {
            *uses_self = true;
        }
        ExprKind::AnonStructLit(fields) => {
            for f in fields {
                collect_free_vars_expr(&f.value, bound, false, out, uses_self, interner);
            }
        }
        ExprKind::NamedStructLit { target, fields } => {
            collect_free_vars_expr(target, bound, false, out, uses_self, interner);
            for f in fields {
                collect_free_vars_expr(&f.value, bound, false, out, uses_self, interner);
            }
        }
        ExprKind::ArrayLit(es) | ExprKind::Tuple(es) => {
            for e in es {
                collect_free_vars_expr(e, bound, false, out, uses_self, interner);
            }
        }
        ExprKind::ArrayRepeat { val, count } => {
            collect_free_vars_expr(val, bound, false, out, uses_self, interner);
            collect_free_vars_expr(count, bound, false, out, uses_self, interner);
        }
        ExprKind::Paren(inner) | ExprKind::Deref(inner) | ExprKind::ComptimeExpr(inner) => {
            collect_free_vars_expr(inner, bound, mutated, out, uses_self, interner);
        }
        ExprKind::Closure { params, body } => {
            let mut inner = bound.clone();
            for p in params {
                inner.insert(p.name.sym);
            }
            collect_free_vars_block(body, &inner, out, uses_self, interner);
        }
        ExprKind::If { data, .. } => collect_free_vars_if(data, bound, out, uses_self, interner),
        ExprKind::Match { scrutinee, arms } => {
            collect_free_vars_expr(scrutinee, bound, false, out, uses_self, interner);
            for arm in arms {
                let mut inner = bound.clone();
                collect_pat_bindings(&arm.pat, &mut inner);
                if let Some(g) = &arm.guard {
                    collect_free_vars_expr(g, &inner, false, out, uses_self, interner);
                }
                match &arm.body {
                    MatchArmBody::Block(b) => collect_free_vars_block(b, &inner, out, uses_self, interner),
                    MatchArmBody::Expr(e) => collect_free_vars_expr(e, &inner, false, out, uses_self, interner),
                    MatchArmBody::Jump(JumpAction::Return(Some(e)))
                    | MatchArmBody::Jump(JumpAction::Break { value: Some(e), .. }) => {
                        collect_free_vars_expr(e, &inner, false, out, uses_self, interner);
                    }
                    MatchArmBody::Jump(_) => {}
                }
            }
        }
        ExprKind::Loop { body, .. } | ExprKind::ComptimeBlock(body) => {
            collect_free_vars_block(body, bound, out, uses_self, interner);
        }
        ExprKind::Builtin { args, .. } => {
            for a in args {
                if let ast::GenericArgKind::Expr(e) = &a.kind {
                    collect_free_vars_expr(e, bound, false, out, uses_self, interner);
                }
            }
        }
        ExprKind::Asm { operands, .. } => {
            for op in operands {
                match &op.kind {
                    ast::AsmOperandKind::Named { arg: Some(e), .. }
                    | ast::AsmOperandKind::Expr(e) => {
                        collect_free_vars_expr(e, bound, false, out, uses_self, interner);
                    }
                    _ => {}
                }
            }
        }
        ExprKind::Unary { op, expr: inner } => {
            let m = mutated || *op == UnaryOp::BorrowMut;
            collect_free_vars_expr(inner, bound, m, out, uses_self, interner);
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_free_vars_expr(lhs, bound, false, out, uses_self, interner);
            collect_free_vars_expr(rhs, bound, false, out, uses_self, interner);
        }
        ExprKind::Is { expr: inner, .. } => {
            collect_free_vars_expr(inner, bound, false, out, uses_self, interner);
        }
        ExprKind::Orelse { lhs, rhs } => {
            collect_free_vars_expr(lhs, bound, false, out, uses_self, interner);
            match rhs {
                CatchBody::Block(b) => collect_free_vars_block(b, bound, out, uses_self, interner),
                CatchBody::Expr(e) => collect_free_vars_expr(e, bound, false, out, uses_self, interner),
                CatchBody::Jump(JumpAction::Return(Some(e)))
                | CatchBody::Jump(JumpAction::Break { value: Some(e), .. }) => {
                    collect_free_vars_expr(e, bound, false, out, uses_self, interner);
                }
                CatchBody::Jump(_) => {}
            }
        }
        ExprKind::Catch { lhs, err_bind, rhs } => {
            collect_free_vars_expr(lhs, bound, false, out, uses_self, interner);
            let mut inner = bound.clone();
            if let Some(eb) = err_bind {
                inner.insert(eb.sym);
            }
            match rhs {
                CatchBody::Block(b) => collect_free_vars_block(b, &inner, out, uses_self, interner),
                CatchBody::Expr(e) => collect_free_vars_expr(e, &inner, false, out, uses_self, interner),
                CatchBody::Jump(JumpAction::Return(Some(e)))
                | CatchBody::Jump(JumpAction::Break { value: Some(e), .. }) => {
                    collect_free_vars_expr(e, &inner, false, out, uses_self, interner);
                }
                CatchBody::Jump(_) => {}
            }
        }
        ExprKind::Field { base, .. } => {
            collect_free_vars_expr(base, bound, mutated, out, uses_self, interner);
        }
        ExprKind::GenericApply { base, args } => {
            collect_free_vars_expr(base, bound, false, out, uses_self, interner);
            for a in args {
                if let ast::GenericArgKind::Expr(e) = &a.kind {
                    collect_free_vars_expr(e, bound, false, out, uses_self, interner);
                }
            }
        }
        ExprKind::Call { callee, args } => {
            if let ExprKind::Field { base, field } = &callee.kind {
                let mname = interner.get(field.sym);
                let readonly = matches!(
                    mname,
                    "len" | "cap" | "as_slice" | "get" | "at" | "sub" | "is_alive" | "as_bytes" | "as_str" | "as_u64" | "as_usize"
                );
                collect_free_vars_expr(base, bound, !readonly, out, uses_self, interner);
            } else {
                collect_free_vars_expr(callee, bound, false, out, uses_self, interner);
            }
            for a in args {
                collect_free_vars_expr(a, bound, false, out, uses_self, interner);
            }
        }
        ExprKind::Index { base, index } => {
            collect_free_vars_expr(base, bound, mutated, out, uses_self, interner);
            collect_free_vars_expr(index, bound, false, out, uses_self, interner);
        }
        ExprKind::Slice { base, lo, hi } => {
            collect_free_vars_expr(base, bound, mutated, out, uses_self, interner);
            if let Some(l) = lo {
                collect_free_vars_expr(l, bound, false, out, uses_self, interner);
            }
            if let Some(h) = hi {
                collect_free_vars_expr(h, bound, false, out, uses_self, interner);
            }
        }
        _ => {}
    }
}

fn parse_signed(raw: Symbol, neg: bool, interner: &Interner) -> Option<i128> {
    let u = parse_int_literal(interner.get(raw))?;
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
