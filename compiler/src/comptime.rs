#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use crate::ast::{
    self, AssignOp, BinaryOp, Block, CompilationUnit, DeclKind, ElseBranch, Expr, ExprKind,
    GenericArg, GenericArgKind, IfCond, IfExpr, JumpAction, MatchArmBody, Pat, PatKind, RangeKind,
    Stmt, StmtKind, UnaryOp,
};
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{BuiltinKind, DefId, ModuleId, ResolvedProgram, TopDefKind, TopScopeTarget};
use crate::span::{Interner, Span, Symbol};
use crate::types::{
    int_bounds, parse_int_literal, resolve_path_segments, resolve_type_expr, GenericCtx, Ty, TyId,
    TypeEnv,
};

pub const DEFAULT_COMPTIME_FUEL: u64 = 10_000_000;

#[derive(Clone, Debug, PartialEq)]
pub enum CtVal {
    Void,
    Bool(bool),
    Char(char),
    Int(i128, TyId),
    Float(f64, TyId),
    Str(String),
    EnumSym(Symbol, i128),
    Array(Vec<CtVal>, TyId),
    Tuple(Vec<CtVal>),
    Struct {
        ty: TyId,
        fields: BTreeMap<Symbol, CtVal>,
    },
    TypeVal(TyId),
}

impl CtVal {
    pub fn as_i128(&self) -> Option<i128> {
        match self {
            CtVal::Int(v, _) | CtVal::EnumSym(_, v) => Some(*v),
            CtVal::Char(c) => Some(*c as i128),
            CtVal::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            CtVal::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

pub fn target_arch_name() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "X86_64"
    } else if cfg!(target_arch = "aarch64") {
        "Arm64"
    } else if cfg!(target_arch = "riscv64") {
        "RiscV64"
    } else {
        "X86_64"
    }
}

pub fn target_os_name() -> &'static str {
    if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "MacOs"
    } else {
        "Linux"
    }
}

pub fn type_info_struct_ty(env: &mut TypeEnv, interner: &mut Interner) -> TyId {
    let str_ty = env.intern(Ty::Borrow {
        is_mut: false,
        inner: crate::types::BorrowInner::Str,
    });
    let fields = vec![
        (interner.intern("size"), TyId::USIZE),
        (interner.intern("align"), TyId::USIZE),
        (interner.intern("field_count"), TyId::USIZE),
        (interner.intern("is_linear"), TyId::BOOL),
        (interner.intern("is_trivial_copy"), TyId::BOOL),
        (interner.intern("name"), str_ty),
        (interner.intern("kind"), str_ty),
    ];
    env.intern(Ty::AnonStruct(fields))
}

pub fn make_type_info_val(ty: TyId, env: &mut TypeEnv, interner: &mut Interner) -> CtVal {
    let lay = env.layout(ty);
    let is_linear = env.is_linear(ty);
    let is_trivial_copy = env.is_trivial_copy(ty);
    let (kind_str, field_count, name_str) = match env.get(ty).clone() {
        Ty::Struct { def, .. } => {
            let fc = env.structs.get(&def).map_or(0, |m| m.fields.len());
            ("struct", fc, env.format_ty(ty, interner))
        }
        Ty::Pool { def, .. } => {
            let fc = env.pools.get(&def).map_or(0, |m| m.columns.len());
            ("pool", fc, env.format_ty(ty, interner))
        }
        Ty::Enum(def) => {
            let vc = env.enums.get(&def).map_or(0, |m| m.variants.len());
            ("enum", vc, env.format_ty(ty, interner))
        }
        Ty::Union { def, .. } => {
            let vc = env.unions.get(&def).map_or(0, |m| m.variants.len());
            ("union", vc, env.format_ty(ty, interner))
        }
        Ty::Bitfield(def) => {
            let mc = env.bitfields.get(&def).map_or(0, |m| m.members.len());
            ("bitfield", mc, env.format_ty(ty, interner))
        }
        Ty::Array { len, .. } => ("array", len as usize, env.format_ty(ty, interner)),
        Ty::Tuple(es) => ("tuple", es.len(), env.format_ty(ty, interner)),
        Ty::AnonStruct(fs) => ("struct", fs.len(), env.format_ty(ty, interner)),
        Ty::Int { .. } => ("int", 0, env.format_ty(ty, interner)),
        Ty::Float(_) => ("float", 0, env.format_ty(ty, interner)),
        Ty::Bool => ("bool", 0, "bool".to_owned()),
        _ => ("other", 0, env.format_ty(ty, interner)),
    };
    let info_ty = type_info_struct_ty(env, interner);
    let mut fields = BTreeMap::new();
    fields.insert(interner.intern("size"), CtVal::Int(lay.size as i128, TyId::USIZE));
    fields.insert(interner.intern("align"), CtVal::Int(lay.align as i128, TyId::USIZE));
    fields.insert(
        interner.intern("field_count"),
        CtVal::Int(field_count as i128, TyId::USIZE),
    );
    fields.insert(interner.intern("is_linear"), CtVal::Bool(is_linear));
    fields.insert(interner.intern("is_trivial_copy"), CtVal::Bool(is_trivial_copy));
    fields.insert(interner.intern("name"), CtVal::Str(name_str));
    fields.insert(interner.intern("kind"), CtVal::Str(kind_str.to_owned()));
    CtVal::Struct { ty: info_ty, fields }
}

type Reg = usize;

#[derive(Clone, Debug)]
enum CtOp {
    LoadConst { dst: Reg, val: CtVal },
    Mov { dst: Reg, src: Reg },
    Unary { dst: Reg, op: UnaryOp, src: Reg, span: Span },
    Binary { dst: Reg, op: BinaryOp, lhs: Reg, rhs: Reg, span: Span },
    Cast { dst: Reg, kind: CtCastKind, src: Reg, to_ty: TyId, span: Span },
    ArrayInit { dst: Reg, elems: Vec<Reg>, elem_ty: TyId },
    ArrayRepeat { dst: Reg, val: Reg, count: usize, elem_ty: TyId, span: Span },
    ArrayLoad { dst: Reg, arr: Reg, idx: Reg, span: Span },
    ArrayStore { arr: Reg, idx: Reg, val: Reg, span: Span },
    FieldLoad { dst: Reg, base: Reg, field: Symbol, span: Span },
    FieldStore { base: Reg, field: Symbol, val: Reg, span: Span },
    StructInit { dst: Reg, ty: TyId, fields: Vec<(Symbol, Reg)> },
    TupleInit { dst: Reg, elems: Vec<Reg> },
    Jump(usize),
    JumpIfFalse { cond: Reg, target: usize },
    JumpIfTrue { cond: Reg, target: usize },
    CallFn { dst: Reg, def: DefId, args: Vec<Reg>, span: Span },
    SetBudget { limit: Reg, span: Span },
    CompileError { msg: Reg, span: Span },
    Ret(Option<Reg>),
}

#[derive(Copy, Clone, Debug)]
enum CtCastKind {
    Widen,
    Trunc,
    Clamp,
    ToFloat,
    TruncToInt,
    Bitcast,
}

struct LoopPatch {
    label: Option<Symbol>,
    continue_jumps: Vec<usize>,
    break_jumps: Vec<usize>,
    break_dst: Option<Reg>,
}

struct CtCompiler<'a> {
    mod_id: ModuleId,
    ctx: &'a GenericCtx,
    units: &'a [CompilationUnit],
    prog: &'a mut ResolvedProgram,
    env: &'a mut TypeEnv,
    interner: &'a mut Interner,
    diags: &'a mut DiagBag,
    ops: Vec<CtOp>,
    num_regs: usize,
    scopes: Vec<BTreeMap<Symbol, Reg>>,
    loops: Vec<LoopPatch>,
    failed: bool,
}

impl<'a> CtCompiler<'a> {
    fn new(
        mod_id: ModuleId,
        ctx: &'a GenericCtx,
        units: &'a [CompilationUnit],
        prog: &'a mut ResolvedProgram,
        env: &'a mut TypeEnv,
        interner: &'a mut Interner,
        diags: &'a mut DiagBag,
    ) -> Self {
        Self {
            mod_id,
            ctx,
            units,
            prog,
            env,
            interner,
            diags,
            ops: Vec::new(),
            num_regs: 0,
            scopes: vec![BTreeMap::new()],
            loops: Vec::new(),
            failed: false,
        }
    }

    fn alloc_reg(&mut self) -> Reg {
        let r = self.num_regs;
        self.num_regs += 1;
        r
    }

    fn emit(&mut self, op: CtOp) -> usize {
        let pc = self.ops.len();
        self.ops.push(op);
        pc
    }

    fn lookup_var(&self, sym: Symbol) -> Option<Reg> {
        for sc in self.scopes.iter().rev() {
            if let Some(&r) = sc.get(&sym) {
                return Some(r);
            }
        }
        None
    }

    fn bind_pat(&mut self, pat: &Pat, val_reg: Reg) {
        match &pat.kind {
            PatKind::Wildcard => {}
            PatKind::Ident(sym) => {
                let r = self.alloc_reg();
                self.emit(CtOp::Mov { dst: r, src: val_reg });
                if let Some(sc) = self.scopes.last_mut() {
                    sc.insert(*sym, r);
                }
            }
            PatKind::Tuple(ps) => {
                for (i, p) in ps.iter().enumerate() {
                    let elem_r = self.alloc_reg();
                    let sym = self.interner.intern(&i.to_string());
                    self.emit(CtOp::FieldLoad {
                        dst: elem_r,
                        base: val_reg,
                        field: sym,
                        span: p.span,
                    });
                    self.bind_pat(p, elem_r);
                }
            }
            _ => {}
        }
    }

    fn compile_block(&mut self, block: &Block, expected_ty: Option<TyId>) -> Reg {
        self.scopes.push(BTreeMap::new());
        for s in &block.stmts {
            self.compile_stmt(s);
        }
        let res = if let Some(tail) = &block.tail_expr {
            self.compile_expr(tail, expected_ty)
        } else {
            let r = self.alloc_reg();
            self.emit(CtOp::LoadConst {
                dst: r,
                val: CtVal::Void,
            });
            r
        };
        self.scopes.pop();
        res
    }

    fn compile_jump(&mut self, j: &JumpAction, span: Span) {
        match j {
            JumpAction::Return(val_opt) => {
                let rr = val_opt.as_ref().map(|v| self.compile_expr(v, None));
                self.emit(CtOp::Ret(rr));
            }
            JumpAction::Break { label, value } => {
                let val_r = value.as_ref().map(|v| self.compile_expr(v, None));
                let target_idx = self.find_loop_idx(label.map(|l| l.sym));
                if let Some(li) = target_idx {
                    if let (Some(vr), Some(dst)) = (val_r, self.loops[li].break_dst) {
                        self.emit(CtOp::Mov { dst, src: vr });
                    }
                    let jpc = self.emit(CtOp::Jump(0));
                    self.loops[li].break_jumps.push(jpc);
                }
            }
            JumpAction::Continue { label } => {
                if let Some(li) = self.find_loop_idx(label.map(|l| l.sym)) {
                    let jpc = self.emit(CtOp::Jump(0));
                    self.loops[li].continue_jumps.push(jpc);
                }
            }
            JumpAction::Unreachable => {
                let msg = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: msg,
                    val: CtVal::Str("reached `unreachable;` in comptime evaluation".into()),
                });
                self.emit(CtOp::CompileError { msg, span });
            }
        }
    }

    fn compile_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                let exp = ty.as_ref().map(|t| {
                    resolve_type_expr(
                        t,
                        self.mod_id,
                        self.ctx,
                        self.units,
                        self.prog,
                        self.env,
                        self.interner,
                        self.diags,
                    )
                });
                let vr = self.compile_expr(init, exp);
                self.bind_pat(pat, vr);
            }
            StmtKind::Var { name, ty, init, .. } => {
                let exp = ty.as_ref().map(|t| {
                    resolve_type_expr(
                        t,
                        self.mod_id,
                        self.ctx,
                        self.units,
                        self.prog,
                        self.env,
                        self.interner,
                        self.diags,
                    )
                });
                let vr = self.compile_expr(init, exp);
                let r = self.alloc_reg();
                self.emit(CtOp::Mov { dst: r, src: vr });
                if let Some(sc) = self.scopes.last_mut() {
                    sc.insert(name.sym, r);
                }
            }
            StmtKind::Assign { lhs, op, rhs } => {
                let rhs_r = self.compile_expr(rhs, None);
                self.compile_assign(lhs, *op, rhs_r, stmt.span);
            }
            StmtKind::Expr(e) | StmtKind::Discard(e) => {
                let _ = self.compile_expr(e, None);
            }
            StmtKind::ComptimeBlock(b) => {
                let _ = self.compile_block(b, None);
            }
            StmtKind::If { is_comptime, data } => {
                let _ = self.compile_if(*is_comptime, data, None);
            }
            StmtKind::While { label, cond, body, .. } => {
                let loop_start = self.ops.len();
                let cond_r = self.compile_expr(cond, Some(TyId::BOOL));
                let j_exit = self.emit(CtOp::JumpIfFalse {
                    cond: cond_r,
                    target: 0,
                });
                self.loops.push(LoopPatch {
                    label: label.map(|l| l.sym),
                    continue_jumps: Vec::new(),
                    break_jumps: vec![j_exit],
                    break_dst: None,
                });
                let _ = self.compile_block(body, None);
                self.emit(CtOp::Jump(loop_start));
                let after = self.ops.len();
                if let Some(lp) = self.loops.pop() {
                    for cj in lp.continue_jumps {
                        self.patch_jump(cj, loop_start);
                    }
                    for bj in lp.break_jumps {
                        self.patch_jump(bj, after);
                    }
                }
            }
            StmtKind::Loop { label, body } => {
                let loop_start = self.ops.len();
                self.loops.push(LoopPatch {
                    label: label.map(|l| l.sym),
                    continue_jumps: Vec::new(),
                    break_jumps: Vec::new(),
                    break_dst: None,
                });
                let _ = self.compile_block(body, None);
                self.emit(CtOp::Jump(loop_start));
                let after = self.ops.len();
                if let Some(lp) = self.loops.pop() {
                    for cj in lp.continue_jumps {
                        self.patch_jump(cj, loop_start);
                    }
                    for bj in lp.break_jumps {
                        self.patch_jump(bj, after);
                    }
                }
            }
            StmtKind::For {
                label,
                pat,
                iter_start,
                iter_end: Some((rk, end_expr)),
                body,
            } => {
                let start_r = self.compile_expr(iter_start, Some(TyId::USIZE));
                let end_r = self.compile_expr(end_expr, Some(TyId::USIZE));
                let idx_r = self.alloc_reg();
                self.emit(CtOp::Mov {
                    dst: idx_r,
                    src: start_r,
                });
                let header_pc = self.ops.len();
                let cmp_r = self.alloc_reg();
                let cmp_op = if *rk == RangeKind::Inclusive {
                    BinaryOp::LtEq
                } else {
                    BinaryOp::Lt
                };
                self.emit(CtOp::Binary {
                    dst: cmp_r,
                    op: cmp_op,
                    lhs: idx_r,
                    rhs: end_r,
                    span: stmt.span,
                });
                let j_exit = self.emit(CtOp::JumpIfFalse {
                    cond: cmp_r,
                    target: 0,
                });
                self.loops.push(LoopPatch {
                    label: label.map(|l| l.sym),
                    continue_jumps: Vec::new(),
                    break_jumps: vec![j_exit],
                    break_dst: None,
                });
                self.scopes.push(BTreeMap::new());
                self.bind_pat(pat, idx_r);
                let _ = self.compile_block(body, None);
                self.scopes.pop();

                let step_pc = self.ops.len();
                let one_r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: one_r,
                    val: CtVal::Int(1, TyId::COMPTIME_INT),
                });
                self.emit(CtOp::Binary {
                    dst: idx_r,
                    op: BinaryOp::AddWrap,
                    lhs: idx_r,
                    rhs: one_r,
                    span: stmt.span,
                });
                self.emit(CtOp::Jump(header_pc));
                let after = self.ops.len();
                if let Some(lp) = self.loops.pop() {
                    for cj in lp.continue_jumps {
                        self.patch_jump(cj, step_pc);
                    }
                    for bj in lp.break_jumps {
                        self.patch_jump(bj, after);
                    }
                }
            }
            StmtKind::Jump(j) => self.compile_jump(j, stmt.span),
            StmtKind::Match { scrutinee, arms } => {
                let _ = self.compile_match(scrutinee, arms, None);
            }
            _ => {}
        }
    }

    fn find_loop_idx(&self, label: Option<Symbol>) -> Option<usize> {
        if let Some(lbl) = label {
            self.loops.iter().rposition(|l| l.label == Some(lbl))
        } else {
            self.loops.len().checked_sub(1)
        }
    }

    fn patch_jump(&mut self, pc: usize, target: usize) {
        match &mut self.ops[pc] {
            CtOp::Jump(t)
            | CtOp::JumpIfFalse { target: t, .. }
            | CtOp::JumpIfTrue { target: t, .. } => {
                *t = target;
            }
            _ => {}
        }
    }

    fn compile_assign(&mut self, lhs: &Expr, op: AssignOp, rhs_r: Reg, span: Span) {
        let bin_op = match op {
            AssignOp::Assign => None,
            AssignOp::AddAssign => Some(BinaryOp::Add),
            AssignOp::SubAssign => Some(BinaryOp::Sub),
            AssignOp::MulAssign => Some(BinaryOp::Mul),
            AssignOp::AddWrapAssign => Some(BinaryOp::AddWrap),
            AssignOp::SubWrapAssign => Some(BinaryOp::SubWrap),
            AssignOp::MulWrapAssign => Some(BinaryOp::MulWrap),
            AssignOp::AddSatAssign => Some(BinaryOp::AddSat),
            AssignOp::SubSatAssign => Some(BinaryOp::SubSat),
            AssignOp::MulSatAssign => Some(BinaryOp::MulSat),
            AssignOp::BitAndAssign => Some(BinaryOp::BitAnd),
            AssignOp::BitOrAssign => Some(BinaryOp::BitOr),
            AssignOp::BitXorAssign => Some(BinaryOp::BitXor),
            AssignOp::ShlAssign => Some(BinaryOp::Shl),
            AssignOp::ShrAssign => Some(BinaryOp::Shr),
        };
        match &lhs.kind {
            ExprKind::Ident(sym) => {
                if let Some(dst) = self.lookup_var(*sym) {
                    if let Some(bop) = bin_op {
                        self.emit(CtOp::Binary {
                            dst,
                            op: bop,
                            lhs: dst,
                            rhs: rhs_r,
                            span,
                        });
                    } else {
                        self.emit(CtOp::Mov { dst, src: rhs_r });
                    }
                }
            }
            ExprKind::Index { base, index } => {
                if let ExprKind::Ident(sym) = &base.kind
                    && let Some(arr_r) = self.lookup_var(*sym)
                {
                    let idx_r = self.compile_expr(index, Some(TyId::USIZE));
                    let val_r = if let Some(bop) = bin_op {
                        let cur_r = self.alloc_reg();
                        self.emit(CtOp::ArrayLoad {
                            dst: cur_r,
                            arr: arr_r,
                            idx: idx_r,
                            span,
                        });
                        let nxt = self.alloc_reg();
                        self.emit(CtOp::Binary {
                            dst: nxt,
                            op: bop,
                            lhs: cur_r,
                            rhs: rhs_r,
                            span,
                        });
                        nxt
                    } else {
                        rhs_r
                    };
                    self.emit(CtOp::ArrayStore {
                        arr: arr_r,
                        idx: idx_r,
                        val: val_r,
                        span,
                    });
                }
            }
            ExprKind::Field { base, field } => {
                if let ExprKind::Ident(sym) = &base.kind
                    && let Some(base_r) = self.lookup_var(*sym)
                {
                    let val_r = if let Some(bop) = bin_op {
                        let cur_r = self.alloc_reg();
                        self.emit(CtOp::FieldLoad {
                            dst: cur_r,
                            base: base_r,
                            field: field.sym,
                            span,
                        });
                        let nxt = self.alloc_reg();
                        self.emit(CtOp::Binary {
                            dst: nxt,
                            op: bop,
                            lhs: cur_r,
                            rhs: rhs_r,
                            span,
                        });
                        nxt
                    } else {
                        rhs_r
                    };
                    self.emit(CtOp::FieldStore {
                        base: base_r,
                        field: field.sym,
                        val: val_r,
                        span,
                    });
                }
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0520,
                    span,
                    "unsupported assignment target in `comptime`",
                );
                self.failed = true;
            }
        }
    }

    fn compile_if(&mut self, _is_comptime: bool, data: &IfExpr, expected_ty: Option<TyId>) -> Reg {
        let cond_r = match &data.cond {
            IfCond::Expr(e) => self.compile_expr(e, Some(TyId::BOOL)),
            IfCond::Let { .. } => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Bool(false),
                });
                r
            }
        };
        let out_r = self.alloc_reg();
        let j_else = self.emit(CtOp::JumpIfFalse {
            cond: cond_r,
            target: 0,
        });
        let then_r = self.compile_block(&data.then_block, expected_ty);
        self.emit(CtOp::Mov {
            dst: out_r,
            src: then_r,
        });
        let j_end = self.emit(CtOp::Jump(0));
        let else_pc = self.ops.len();
        self.patch_jump(j_else, else_pc);
        if let Some(eb) = &data.else_branch {
            let else_r = match eb {
                ElseBranch::Block(b) => self.compile_block(b, expected_ty),
                ElseBranch::If { is_comptime, data } => {
                    self.compile_if(*is_comptime, data, expected_ty)
                }
            };
            self.emit(CtOp::Mov {
                dst: out_r,
                src: else_r,
            });
        } else {
            self.emit(CtOp::LoadConst {
                dst: out_r,
                val: CtVal::Void,
            });
        }
        let end_pc = self.ops.len();
        self.patch_jump(j_end, end_pc);
        out_r
    }

    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[ast::MatchArm],
        expected_ty: Option<TyId>,
    ) -> Reg {
        let scr_r = self.compile_expr(scrutinee, None);
        let out_r = self.alloc_reg();
        self.emit(CtOp::LoadConst {
            dst: out_r,
            val: CtVal::Void,
        });
        let mut end_jumps = Vec::new();
        for arm in arms {
            let match_r = self.compile_pat_test(scr_r, &arm.pat, arm.span);
            let j_next = self.emit(CtOp::JumpIfFalse {
                cond: match_r,
                target: 0,
            });
            self.scopes.push(BTreeMap::new());
            self.bind_pat(&arm.pat, scr_r);
            match &arm.body {
                MatchArmBody::Block(b) => {
                    let body_r = self.compile_block(b, expected_ty);
                    self.emit(CtOp::Mov {
                        dst: out_r,
                        src: body_r,
                    });
                }
                MatchArmBody::Expr(e) => {
                    let body_r = self.compile_expr(e, expected_ty);
                    self.emit(CtOp::Mov {
                        dst: out_r,
                        src: body_r,
                    });
                }
                MatchArmBody::Jump(j) => {
                    self.compile_jump(j, arm.span);
                }
            }
            self.scopes.pop();
            end_jumps.push(self.emit(CtOp::Jump(0)));
            let next_pc = self.ops.len();
            self.patch_jump(j_next, next_pc);
        }
        let end_pc = self.ops.len();
        for ej in end_jumps {
            self.patch_jump(ej, end_pc);
        }
        out_r
    }

    fn compile_pat_test(&mut self, scr_r: Reg, pat: &Pat, span: Span) -> Reg {
        match &pat.kind {
            PatKind::Wildcard | PatKind::Ident(_) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Bool(true),
                });
                r
            }
            PatKind::BoolLit(b) => {
                let lit_r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: lit_r,
                    val: CtVal::Bool(*b),
                });
                let r = self.alloc_reg();
                self.emit(CtOp::Binary {
                    dst: r,
                    op: BinaryOp::Eq,
                    lhs: scr_r,
                    rhs: lit_r,
                    span,
                });
                r
            }
            PatKind::IntLit { neg, raw, range_end } => {
                let lo_v = parse_int_literal(self.interner.get(*raw)).unwrap_or(0) as i128;
                let lo_v = if *neg { -lo_v } else { lo_v };
                let lo_r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: lo_r,
                    val: CtVal::Int(lo_v, TyId::COMPTIME_INT),
                });
                if let Some((rk, rneg, rraw)) = range_end {
                    let hi_v = parse_int_literal(self.interner.get(*rraw)).unwrap_or(0) as i128;
                    let hi_v = if *rneg { -hi_v } else { hi_v };
                    let hi_r = self.alloc_reg();
                    self.emit(CtOp::LoadConst {
                        dst: hi_r,
                        val: CtVal::Int(hi_v, TyId::COMPTIME_INT),
                    });
                    let c1 = self.alloc_reg();
                    self.emit(CtOp::Binary {
                        dst: c1,
                        op: BinaryOp::GtEq,
                        lhs: scr_r,
                        rhs: lo_r,
                        span,
                    });
                    let c2 = self.alloc_reg();
                    let hi_op = if *rk == RangeKind::Inclusive {
                        BinaryOp::LtEq
                    } else {
                        BinaryOp::Lt
                    };
                    self.emit(CtOp::Binary {
                        dst: c2,
                        op: hi_op,
                        lhs: scr_r,
                        rhs: hi_r,
                        span,
                    });
                    let res = self.alloc_reg();
                    self.emit(CtOp::Binary {
                        dst: res,
                        op: BinaryOp::And,
                        lhs: c1,
                        rhs: c2,
                        span,
                    });
                    res
                } else {
                    let r = self.alloc_reg();
                    self.emit(CtOp::Binary {
                        dst: r,
                        op: BinaryOp::Eq,
                        lhs: scr_r,
                        rhs: lo_r,
                        span,
                    });
                    r
                }
            }
            PatKind::Variant { name, .. } => {
                let lit_r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: lit_r,
                    val: CtVal::EnumSym(name.sym, 0),
                });
                let r = self.alloc_reg();
                self.emit(CtOp::Binary {
                    dst: r,
                    op: BinaryOp::Eq,
                    lhs: scr_r,
                    rhs: lit_r,
                    span,
                });
                r
            }
            _ => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Bool(false),
                });
                r
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr, expected_ty: Option<TyId>) -> Reg {
        match &expr.kind {
            ExprKind::IntLit { raw, suffix } => {
                let v = parse_int_literal(self.interner.get(*raw)).unwrap_or(0);
                let ty = suffix
                    .and_then(|s| self.resolve_int_suffix(s))
                    .or(expected_ty.filter(|&t| self.env.is_integer(t)))
                    .unwrap_or(TyId::COMPTIME_INT);
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Int(v as i128, ty),
                });
                r
            }
            ExprKind::FloatLit { raw, suffix } => {
                let s = self.interner.get(*raw).replace('_', "");
                let v: f64 = s.parse().unwrap_or(0.0);
                let ty = match suffix.map(|sym| self.interner.get(sym)) {
                    Some("f32") => TyId::F32,
                    Some("f64") => TyId::F64,
                    _ => expected_ty.filter(|&t| self.env.is_float(t)).unwrap_or(TyId::F64),
                };
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Float(v, ty),
                });
                r
            }
            ExprKind::BoolLit(b) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Bool(*b),
                });
                r
            }
            ExprKind::CharLit(c) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Char(*c),
                });
                r
            }
            ExprKind::ByteCharLit(b) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Int(*b as i128, TyId::U8),
                });
                r
            }
            ExprKind::StringLit(s) | ExprKind::RawStringLit(s) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Str(s.clone()),
                });
                r
            }
            ExprKind::EnumLit(v) => {
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::EnumSym(v.sym, 0),
                });
                r
            }
            ExprKind::Paren(inner) | ExprKind::ComptimeExpr(inner) => {
                self.compile_expr(inner, expected_ty)
            }
            ExprKind::ComptimeBlock(b) => self.compile_block(b, expected_ty),
            ExprKind::Ident(sym) => {
                if let Some(r) = self.lookup_var(*sym) {
                    return r;
                }
                if let Some(&cv) = self.ctx.comptime_vals.get(sym) {
                    let r = self.alloc_reg();
                    self.emit(CtOp::LoadConst {
                        dst: r,
                        val: CtVal::Int(cv, TyId::COMPTIME_INT),
                    });
                    return r;
                }
                let seg = ast::IdentSpan {
                    span: expr.span,
                    sym: *sym,
                };
                if let Some(TopScopeTarget::Def(d)) = resolve_path_segments(
                    &[seg],
                    self.mod_id,
                    self.prog,
                    self.interner,
                    self.diags,
                ) && let Some(cm) = self.env.consts.get(&d).cloned()
                {
                    let val = if let Some(cv) = cm.ct_val {
                        cv
                    } else if let Some(iv) = cm.int_val {
                        CtVal::Int(iv, cm.ty)
                    } else {
                        CtVal::Void
                    };
                    let r = self.alloc_reg();
                    self.emit(CtOp::LoadConst { dst: r, val });
                    return r;
                }
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Void,
                });
                r
            }
            ExprKind::Unary { op, expr: inner } => {
                if *op == UnaryOp::Neg
                    && let ExprKind::IntLit { raw, suffix } = &inner.kind
                {
                    let v = parse_int_literal(self.interner.get(*raw)).unwrap_or(0);
                    let iv = if v == (1u128 << 127) {
                        i128::MIN
                    } else {
                        -(v as i128)
                    };
                    let ty = suffix
                        .and_then(|s| self.resolve_int_suffix(s))
                        .or(expected_ty.filter(|&t| self.env.is_integer(t)))
                        .unwrap_or(TyId::COMPTIME_INT);
                    let r = self.alloc_reg();
                    self.emit(CtOp::LoadConst {
                        dst: r,
                        val: CtVal::Int(iv, ty),
                    });
                    return r;
                }
                let sr = self.compile_expr(inner, expected_ty);
                let dst = self.alloc_reg();
                self.emit(CtOp::Unary {
                    dst,
                    op: *op,
                    src: sr,
                    span: expr.span,
                });
                dst
            }
            ExprKind::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::And {
                    let lr = self.compile_expr(lhs, Some(TyId::BOOL));
                    let dst = self.alloc_reg();
                    self.emit(CtOp::Mov { dst, src: lr });
                    let j_false = self.emit(CtOp::JumpIfFalse {
                        cond: lr,
                        target: 0,
                    });
                    let rr = self.compile_expr(rhs, Some(TyId::BOOL));
                    self.emit(CtOp::Mov { dst, src: rr });
                    let end = self.ops.len();
                    self.patch_jump(j_false, end);
                    return dst;
                }
                if *op == BinaryOp::Or {
                    let lr = self.compile_expr(lhs, Some(TyId::BOOL));
                    let dst = self.alloc_reg();
                    self.emit(CtOp::Mov { dst, src: lr });
                    let j_true = self.emit(CtOp::JumpIfTrue {
                        cond: lr,
                        target: 0,
                    });
                    let rr = self.compile_expr(rhs, Some(TyId::BOOL));
                    self.emit(CtOp::Mov { dst, src: rr });
                    let end = self.ops.len();
                    self.patch_jump(j_true, end);
                    return dst;
                }
                let op_exp = if matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::NotEq
                        | BinaryOp::Lt
                        | BinaryOp::LtEq
                        | BinaryOp::Gt
                        | BinaryOp::GtEq
                ) {
                    None
                } else {
                    expected_ty
                };
                let lr = self.compile_expr(lhs, op_exp);
                let rr = self.compile_expr(rhs, op_exp);
                let dst = self.alloc_reg();
                self.emit(CtOp::Binary {
                    dst,
                    op: *op,
                    lhs: lr,
                    rhs: rr,
                    span: expr.span,
                });
                dst
            }
            ExprKind::ArrayLit(elems) => {
                let elem_ty = match expected_ty.map(|t| self.env.get(t).clone()) {
                    Some(Ty::Array { elem, .. }) => elem,
                    _ => TyId::COMPTIME_INT,
                };
                let mut regs = Vec::with_capacity(elems.len());
                for e in elems {
                    regs.push(self.compile_expr(e, Some(elem_ty)));
                }
                let dst = self.alloc_reg();
                self.emit(CtOp::ArrayInit {
                    dst,
                    elems: regs,
                    elem_ty,
                });
                dst
            }
            ExprKind::ArrayRepeat { val, count } => {
                let elem_ty = match expected_ty.map(|t| self.env.get(t).clone()) {
                    Some(Ty::Array { elem, .. }) => elem,
                    _ => TyId::COMPTIME_INT,
                };
                let vr = self.compile_expr(val, Some(elem_ty));
                let cnt = crate::types::eval_const_int(
                    count,
                    self.mod_id,
                    self.ctx,
                    self.units,
                    self.prog,
                    self.env,
                    self.interner,
                )
                .unwrap_or(0) as usize;
                let dst = self.alloc_reg();
                self.emit(CtOp::ArrayRepeat {
                    dst,
                    val: vr,
                    count: cnt,
                    elem_ty,
                    span: expr.span,
                });
                dst
            }
            ExprKind::Index { base, index } => {
                let ar = self.compile_expr(base, None);
                let ir = self.compile_expr(index, Some(TyId::USIZE));
                let dst = self.alloc_reg();
                self.emit(CtOp::ArrayLoad {
                    dst,
                    arr: ar,
                    idx: ir,
                    span: expr.span,
                });
                dst
            }
            ExprKind::Field { base, field } => {
                // Handle primitive constants like `u32.MAX`, `usize.MIN`
                if let ExprKind::Ident(bsym) = &base.kind
                    && self.lookup_var(*bsym).is_none()
                    && let Some(cval) = self.try_prim_assoc_const(*bsym, field.sym)
                {
                    let r = self.alloc_reg();
                    self.emit(CtOp::LoadConst { dst: r, val: cval });
                    return r;
                }
                let br = self.compile_expr(base, None);
                let dst = self.alloc_reg();
                self.emit(CtOp::FieldLoad {
                    dst,
                    base: br,
                    field: field.sym,
                    span: expr.span,
                });
                dst
            }
            ExprKind::AnonStructLit(fields) | ExprKind::NamedStructLit { fields, .. } => {
                let st_ty = expected_ty.unwrap_or(TyId::ERROR);
                let mut fregs = Vec::with_capacity(fields.len());
                for f in fields {
                    if let Some(name) = f.name {
                        let vr = self.compile_expr(&f.value, None);
                        fregs.push((name.sym, vr));
                    }
                }
                let dst = self.alloc_reg();
                self.emit(CtOp::StructInit {
                    dst,
                    ty: st_ty,
                    fields: fregs,
                });
                dst
            }
            ExprKind::Tuple(elems) => {
                let mut regs = Vec::with_capacity(elems.len());
                for e in elems {
                    regs.push(self.compile_expr(e, None));
                }
                let dst = self.alloc_reg();
                self.emit(CtOp::TupleInit { dst, elems: regs });
                dst
            }
            ExprKind::If { is_comptime, data } => {
                self.compile_if(*is_comptime, data, expected_ty)
            }
            ExprKind::Match { scrutinee, arms } => self.compile_match(scrutinee, arms, expected_ty),
            ExprKind::Loop { label, body } => {
                let loop_start = self.ops.len();
                let out_r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: out_r,
                    val: CtVal::Void,
                });
                self.loops.push(LoopPatch {
                    label: label.map(|l| l.sym),
                    continue_jumps: Vec::new(),
                    break_jumps: Vec::new(),
                    break_dst: Some(out_r),
                });
                let _ = self.compile_block(body, None);
                self.emit(CtOp::Jump(loop_start));
                let after = self.ops.len();
                if let Some(lp) = self.loops.pop() {
                    for cj in lp.continue_jumps {
                        self.patch_jump(cj, loop_start);
                    }
                    for bj in lp.break_jumps {
                        self.patch_jump(bj, after);
                    }
                }
                out_r
            }
            ExprKind::Call { callee, args } => {
                // Method calls like `x.trunc.[u32]()`, `x.widen.[u64]()`, `arr.len()`
                if let ExprKind::GenericApply { base, args: gargs } = &callee.kind
                    && let ExprKind::Field { base: recv, field } = &base.kind
                {
                    let fname = self.interner.get(field.sym).to_owned();
                    let kind = match fname.as_str() {
                        "widen" => Some(CtCastKind::Widen),
                        "trunc" => Some(CtCastKind::Trunc),
                        "clamp" => Some(CtCastKind::Clamp),
                        "to_float" => Some(CtCastKind::ToFloat),
                        "trunc_to_int" => Some(CtCastKind::TruncToInt),
                        "bitcast" => Some(CtCastKind::Bitcast),
                        _ => None,
                    };
                    if let Some(ck) = kind
                        && let Some(ga) = gargs.first()
                    {
                        let to_ty = self.resolve_generic_ty_arg(ga);
                        let sr = self.compile_expr(recv, None);
                        let dst = self.alloc_reg();
                        self.emit(CtOp::Cast {
                            dst,
                            kind: ck,
                            src: sr,
                            to_ty,
                            span: expr.span,
                        });
                        return dst;
                    }
                }
                if let ExprKind::Field { base: recv, field } = &callee.kind
                    && self.interner.get(field.sym) == "len"
                    && args.is_empty()
                {
                    let br = self.compile_expr(recv, None);
                    let dst = self.alloc_reg();
                    let lsym = self.interner.intern("len");
                    self.emit(CtOp::FieldLoad {
                        dst,
                        base: br,
                        field: lsym,
                        span: expr.span,
                    });
                    return dst;
                }
                // Direct function call (comptime fn or pure fn)
                if let Some(segs) = extract_expr_path(callee)
                    && let Some(TopScopeTarget::Def(d)) = resolve_path_segments(
                        &segs,
                        self.mod_id,
                        self.prog,
                        self.interner,
                        self.diags,
                    )
                    && matches!(self.prog.defs[d.0 as usize].kind, TopDefKind::Fn(_))
                {
                    let mut aregs = Vec::with_capacity(args.len());
                    for a in args {
                        aregs.push(self.compile_expr(a, None));
                    }
                    let dst = self.alloc_reg();
                    self.emit(CtOp::CallFn {
                        dst,
                        def: d,
                        args: aregs,
                        span: expr.span,
                    });
                    return dst;
                }
                self.diags.error(
                    ErrorCode::E0520,
                    expr.span,
                    "call target is not allowed or unsupported in `comptime`",
                );
                self.failed = true;
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Void,
                });
                r
            }
            ExprKind::Builtin { name, args } => {
                self.compile_builtin(name.sym, args, expr.span)
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0520,
                    expr.span,
                    "unsupported expression in `comptime`",
                );
                self.failed = true;
                let r = self.alloc_reg();
                self.emit(CtOp::LoadConst {
                    dst: r,
                    val: CtVal::Void,
                });
                r
            }
        }
    }

    fn compile_builtin(&mut self, sym: Symbol, args: &[GenericArg], span: Span) -> Reg {
        let bname = self.interner.get(sym).to_owned();
        let dst = self.alloc_reg();
        match bname.as_str() {
            "size_of" | "align_of" => {
                let ty = args
                    .first()
                    .map(|a| self.resolve_generic_ty_arg(a))
                    .unwrap_or(TyId::ERROR);
                let lay = self.env.layout(ty);
                let v = if bname == "size_of" { lay.size } else { lay.align };
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::Int(v as i128, TyId::USIZE),
                });
            }
            "type_info" => {
                let ty = args
                    .first()
                    .map(|a| self.resolve_generic_ty_arg(a))
                    .unwrap_or(TyId::ERROR);
                let info = make_type_info_val(ty, self.env, self.interner);
                self.emit(CtOp::LoadConst { dst, val: info });
            }
            "target_arch" => {
                let asym = self.interner.intern(target_arch_name());
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::EnumSym(asym, 0),
                });
            }
            "target_os" => {
                let osym = self.interner.intern(target_os_name());
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::EnumSym(osym, 0),
                });
            }
            "comptime_budget" => {
                if let Some(GenericArg {
                    kind: GenericArgKind::Expr(e),
                    ..
                }) = args.first()
                {
                    let lim_r = self.compile_expr(e, Some(TyId::U64));
                    self.emit(CtOp::SetBudget { limit: lim_r, span });
                }
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::Void,
                });
            }
            "compile_error" => {
                let msg_r = if let Some(GenericArg {
                    kind: GenericArgKind::Expr(e),
                    ..
                }) = args.first()
                {
                    self.compile_expr(e, None)
                } else {
                    let r = self.alloc_reg();
                    self.emit(CtOp::LoadConst {
                        dst: r,
                        val: CtVal::Str("@compile_error triggered".into()),
                    });
                    r
                };
                self.emit(CtOp::CompileError { msg: msg_r, span });
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::Void,
                });
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0520,
                    span,
                    format!("builtin `@{bname}` is not supported in `comptime`"),
                );
                self.failed = true;
                self.emit(CtOp::LoadConst {
                    dst,
                    val: CtVal::Void,
                });
            }
        }
        dst
    }

    fn resolve_generic_ty_arg(&mut self, arg: &GenericArg) -> TyId {
        match &arg.kind {
            GenericArgKind::Type(t) => resolve_type_expr(
                t,
                self.mod_id,
                self.ctx,
                self.units,
                self.prog,
                self.env,
                self.interner,
                self.diags,
            ),
            GenericArgKind::TypeOrConst(segs) => {
                let tp = ast::TypePath {
                    segments: segs.clone(),
                    generics: Vec::new(),
                };
                let te = ast::TypeExpr {
                    span: arg.span,
                    kind: ast::TypeKind::Path(tp),
                };
                resolve_type_expr(
                    &te,
                    self.mod_id,
                    self.ctx,
                    self.units,
                    self.prog,
                    self.env,
                    self.interner,
                    self.diags,
                )
            }
            GenericArgKind::Expr(_) => TyId::ERROR,
        }
    }

    fn resolve_int_suffix(&mut self, sym: Symbol) -> Option<TyId> {
        let s = self.interner.get(sym);
        match s {
            "u8" => Some(TyId::U8),
            "u16" => Some(TyId::U16),
            "u32" => Some(TyId::U32),
            "u64" => Some(TyId::U64),
            "u128" => Some(TyId::U128),
            "usize" => Some(TyId::USIZE),
            "i8" => Some(TyId::I8),
            "i16" => Some(TyId::I16),
            "i32" => Some(TyId::I32),
            "i64" => Some(TyId::I64),
            "i128" => Some(TyId::I128),
            "isize" => Some(TyId::ISIZE),
            _ => None,
        }
    }

    fn try_prim_assoc_const(&self, type_sym: Symbol, field_sym: Symbol) -> Option<CtVal> {
        let tn = self.interner.get(type_sym);
        let fnm = self.interner.get(field_sym);
        let ty = match tn {
            "u8" => TyId::U8,
            "u16" => TyId::U16,
            "u32" => TyId::U32,
            "u64" => TyId::U64,
            "usize" => TyId::USIZE,
            "i8" => TyId::I8,
            "i16" => TyId::I16,
            "i32" => TyId::I32,
            "i64" => TyId::I64,
            "isize" => TyId::ISIZE,
            _ => return None,
        };
        let (lo, hi) = int_bounds(self.env.get(ty))?;
        match fnm {
            "MIN" => Some(CtVal::Int(lo, ty)),
            "MAX" => Some(CtVal::Int(hi, ty)),
            _ => None,
        }
    }
}

fn extract_expr_path(e: &Expr) -> Option<Vec<ast::IdentSpan>> {
    match &e.kind {
        ExprKind::Ident(sym) => Some(vec![ast::IdentSpan {
            span: e.span,
            sym: *sym,
        }]),
        ExprKind::Field { base, field } => {
            let mut v = extract_expr_path(base)?;
            v.push(*field);
            Some(v)
        }
        _ => None,
    }
}

struct CtVm<'a> {
    units: &'a [CompilationUnit],
    prog: &'a mut ResolvedProgram,
    env: &'a mut TypeEnv,
    interner: &'a mut Interner,
    diags: &'a mut DiagBag,
    steps: u64,
    fuel_limit: u64,
    call_depth: usize,
}

impl<'a> CtVm<'a> {
    fn new(
        units: &'a [CompilationUnit],
        prog: &'a mut ResolvedProgram,
        env: &'a mut TypeEnv,
        interner: &'a mut Interner,
        diags: &'a mut DiagBag,
        fuel_limit: u64,
    ) -> Self {
        Self {
            units,
            prog,
            env,
            interner,
            diags,
            steps: 0,
            fuel_limit,
            call_depth: 0,
        }
    }

    fn exec(&mut self, ops: &[CtOp], num_regs: usize, ret_reg: Reg, root_span: Span) -> Result<CtVal, ()> {
        let mut regs = vec![CtVal::Void; num_regs];
        let mut pc = 0usize;
        while pc < ops.len() {
            self.steps += 1;
            if self.steps > self.fuel_limit {
                self.diags.error(
                    ErrorCode::E0520,
                    root_span,
                    format!(
                        "comptime VM exceeded instruction budget of {} steps (possible infinite loop; increase with `@comptime_budget(...)`)",
                        self.fuel_limit
                    ),
                );
                return Err(());
            }
            match &ops[pc] {
                CtOp::LoadConst { dst, val } => {
                    regs[*dst] = val.clone();
                    pc += 1;
                }
                CtOp::Mov { dst, src } => {
                    regs[*dst] = regs[*src].clone();
                    pc += 1;
                }
                CtOp::Unary { dst, op, src, span } => {
                    let v = self.eval_unary(*op, &regs[*src], *span)?;
                    regs[*dst] = v;
                    pc += 1;
                }
                CtOp::Binary { dst, op, lhs, rhs, span } => {
                    let v = self.eval_binary(*op, &regs[*lhs], &regs[*rhs], *span)?;
                    regs[*dst] = v;
                    pc += 1;
                }
                CtOp::Cast { dst, kind, src, to_ty, span } => {
                    let v = self.eval_cast(*kind, &regs[*src], *to_ty, *span)?;
                    regs[*dst] = v;
                    pc += 1;
                }
                CtOp::ArrayInit { dst, elems, elem_ty } => {
                    let vs: Vec<CtVal> = elems.iter().map(|&r| regs[r].clone()).collect();
                    regs[*dst] = CtVal::Array(vs, *elem_ty);
                    pc += 1;
                }
                CtOp::ArrayRepeat { dst, val, count, elem_ty, span } => {
                    if *count > 1_048_576 {
                        self.diags.error(
                            ErrorCode::E0520,
                            *span,
                            "comptime array repeat size exceeds maximum limit (1M elements)",
                        );
                        return Err(());
                    }
                    regs[*dst] = CtVal::Array(vec![regs[*val].clone(); *count], *elem_ty);
                    pc += 1;
                }
                CtOp::ArrayLoad { dst, arr, idx, span } => {
                    let Some(i) = regs[*idx].as_i128() else {
                        self.diags.error(ErrorCode::E0520, *span, "comptime array index is not an integer");
                        return Err(());
                    };
                    match &regs[*arr] {
                        CtVal::Array(elems, _) => {
                            if i < 0 || (i as usize) >= elems.len() {
                                self.diags.error(
                                    ErrorCode::E0520,
                                    *span,
                                    format!("comptime array index {} out of bounds (len = {})", i, elems.len()),
                                );
                                return Err(());
                            }
                            regs[*dst] = elems[i as usize].clone();
                        }
                        CtVal::Str(s) => {
                            let bs = s.as_bytes();
                            if i < 0 || (i as usize) >= bs.len() {
                                self.diags.error(
                                    ErrorCode::E0520,
                                    *span,
                                    format!("comptime string index {} out of bounds (len = {})", i, bs.len()),
                                );
                                return Err(());
                            }
                            regs[*dst] = CtVal::Int(bs[i as usize] as i128, TyId::U8);
                        }
                        _ => {
                            self.diags.error(ErrorCode::E0520, *span, "comptime indexing non-array value");
                            return Err(());
                        }
                    }
                    pc += 1;
                }
                CtOp::ArrayStore { arr, idx, val, span } => {
                    let Some(i) = regs[*idx].as_i128() else {
                        self.diags.error(ErrorCode::E0520, *span, "comptime array index is not an integer");
                        return Err(());
                    };
                    let new_v = regs[*val].clone();
                    match &mut regs[*arr] {
                        CtVal::Array(elems, elem_ty) => {
                            if i < 0 || (i as usize) >= elems.len() {
                                self.diags.error(
                                    ErrorCode::E0520,
                                    *span,
                                    format!("comptime array store index {} out of bounds (len = {})", i, elems.len()),
                                );
                                return Err(());
                            }
                            let stored = match new_v {
                                CtVal::Int(n, TyId::COMPTIME_INT) if self.env.is_integer(*elem_ty) => {
                                    CtVal::Int(n, *elem_ty)
                                }
                                other => other,
                            };
                            elems[i as usize] = stored;
                        }
                        _ => {
                            self.diags.error(ErrorCode::E0520, *span, "comptime array store on non-array value");
                            return Err(());
                        }
                    }
                    pc += 1;
                }
                CtOp::FieldLoad { dst, base, field, span } => {
                    let fname = self.interner.get(*field);
                    match &regs[*base] {
                        CtVal::Array(elems, _) if fname == "len" => {
                            regs[*dst] = CtVal::Int(elems.len() as i128, TyId::USIZE);
                        }
                        CtVal::Str(s) if fname == "len" => {
                            regs[*dst] = CtVal::Int(s.len() as i128, TyId::USIZE);
                        }
                        CtVal::Struct { fields, .. } => {
                            if let Some(v) = fields.get(field) {
                                regs[*dst] = v.clone();
                            } else {
                                self.diags.error(
                                    ErrorCode::E0520,
                                    *span,
                                    format!("comptime struct has no field `{}`", fname),
                                );
                                return Err(());
                            }
                        }
                        CtVal::Tuple(elems) => {
                            if let Ok(idx) = fname.parse::<usize>()
                                && idx < elems.len()
                            {
                                regs[*dst] = elems[idx].clone();
                            } else {
                                self.diags.error(ErrorCode::E0520, *span, "invalid comptime tuple field");
                                return Err(());
                            }
                        }
                        _ => {
                            self.diags.error(
                                ErrorCode::E0520,
                                *span,
                                format!("cannot access field `{}` in comptime", fname),
                            );
                            return Err(());
                        }
                    }
                    pc += 1;
                }
                CtOp::FieldStore { base, field, val, span } => {
                    let nv = regs[*val].clone();
                    if let CtVal::Struct { fields, .. } = &mut regs[*base] {
                        fields.insert(*field, nv);
                    } else {
                        self.diags.error(ErrorCode::E0520, *span, "comptime field store on non-struct");
                        return Err(());
                    }
                    pc += 1;
                }
                CtOp::StructInit { dst, ty, fields } => {
                    let mut map = BTreeMap::new();
                    for &(sym, r) in fields {
                        map.insert(sym, regs[r].clone());
                    }
                    regs[*dst] = CtVal::Struct { ty: *ty, fields: map };
                    pc += 1;
                }
                CtOp::TupleInit { dst, elems } => {
                    regs[*dst] = CtVal::Tuple(elems.iter().map(|&r| regs[r].clone()).collect());
                    pc += 1;
                }
                CtOp::Jump(target) => {
                    pc = *target;
                }
                CtOp::JumpIfFalse { cond, target } => {
                    if !regs[*cond].as_bool().unwrap_or(false) {
                        pc = *target;
                    } else {
                        pc += 1;
                    }
                }
                CtOp::JumpIfTrue { cond, target } => {
                    if regs[*cond].as_bool().unwrap_or(false) {
                        pc = *target;
                    } else {
                        pc += 1;
                    }
                }
                CtOp::CallFn { dst, def, args, span } => {
                    if self.call_depth >= 64 {
                        self.diags.error(
                            ErrorCode::E0520,
                            *span,
                            "comptime VM recursion depth exceeded (max 64)",
                        );
                        return Err(());
                    }
                    let arg_vals: Vec<CtVal> = args.iter().map(|&r| regs[r].clone()).collect();
                    let val = self.eval_fn_call(*def, &arg_vals, *span)?;
                    regs[*dst] = val;
                    pc += 1;
                }
                CtOp::SetBudget { limit, span } => {
                    if let Some(n) = regs[*limit].as_i128()
                        && n > 0
                    {
                        self.fuel_limit = n as u64;
                    } else {
                        self.diags.error(
                            ErrorCode::E0520,
                            *span,
                            "`@comptime_budget` expects a positive integer",
                        );
                        return Err(());
                    }
                    pc += 1;
                }
                CtOp::CompileError { msg, span } => {
                    let text = match &regs[*msg] {
                        CtVal::Str(s) => s.clone(),
                        _ => "@compile_error triggered".into(),
                    };
                    self.diags.error(ErrorCode::E0520, *span, text);
                    return Err(());
                }
                CtOp::Ret(reg_opt) => {
                    return Ok(reg_opt.map_or(CtVal::Void, |r| regs[r].clone()));
                }
            }
        }
        Ok(regs[ret_reg].clone())
    }

    fn eval_fn_call(&mut self, def: DefId, arg_vals: &[CtVal], span: Span) -> Result<CtVal, ()> {
        let def_info = &self.prog.defs[def.0 as usize];
        let fn_mod = def_info.module;
        let TopDefKind::Fn(decl_idx) = def_info.kind else {
            self.diags.error(ErrorCode::E0520, span, "cannot call non-function in comptime");
            return Err(());
        };
        let Some(ui) = self.prog.modules[fn_mod.0 as usize].unit_idx else {
            self.diags.error(ErrorCode::E0520, span, "cannot call virtual builtin in comptime");
            return Err(());
        };
        let DeclKind::Fn(fdecl) = &self.units[ui].decls[decl_idx].kind else {
            return Err(());
        };
        let fdecl_params: Vec<Symbol> = fdecl.params.iter().map(|p| p.name.sym).collect();
        let body = &fdecl.body;

        let ctx = GenericCtx::empty();
        let mut comp = CtCompiler::new(
            fn_mod,
            &ctx,
            self.units,
            self.prog,
            self.env,
            self.interner,
            self.diags,
        );
        for (psym, aval) in fdecl_params.iter().zip(arg_vals.iter()) {
            let pr = comp.alloc_reg();
            comp.emit(CtOp::LoadConst {
                dst: pr,
                val: aval.clone(),
            });
            if let Some(sc) = comp.scopes.last_mut() {
                sc.insert(*psym, pr);
            }
        }
        let ret_r = comp.compile_block(body, None);
        if comp.failed {
            return Err(());
        }
        let ops = comp.ops;
        let nregs = comp.num_regs;
        self.call_depth += 1;
        let res = self.exec(&ops, nregs, ret_r, span);
        self.call_depth -= 1;
        res
    }

    fn eval_unary(&mut self, op: UnaryOp, v: &CtVal, span: Span) -> Result<CtVal, ()> {
        match (op, v) {
            (UnaryOp::Not, CtVal::Bool(b)) => Ok(CtVal::Bool(!b)),
            (UnaryOp::Neg, CtVal::Int(n, ty)) => {
                let Some(neg) = n.checked_neg() else {
                    self.diags.error(ErrorCode::E0520, span, "comptime integer negation overflow");
                    return Err(());
                };
                self.check_int_range(neg, *ty, span)?;
                Ok(CtVal::Int(neg, *ty))
            }
            (UnaryOp::Neg, CtVal::Float(f, ty)) => Ok(CtVal::Float(-f, *ty)),
            (UnaryOp::BitNot, CtVal::Int(n, ty)) => {
                let res = wrap_int(!n, self.env.get(*ty));
                Ok(CtVal::Int(res, *ty))
            }
            _ => {
                self.diags.error(ErrorCode::E0520, span, "invalid unary operation in comptime");
                Err(())
            }
        }
    }

    fn eval_binary(&mut self, op: BinaryOp, l: &CtVal, r: &CtVal, span: Span) -> Result<CtVal, ()> {
        match (l, r) {
            (CtVal::Bool(a), CtVal::Bool(b)) => match op {
                BinaryOp::Eq => Ok(CtVal::Bool(a == b)),
                BinaryOp::NotEq => Ok(CtVal::Bool(a != b)),
                BinaryOp::And => Ok(CtVal::Bool(*a && *b)),
                BinaryOp::Or => Ok(CtVal::Bool(*a || *b)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid bool binary op in comptime");
                    Err(())
                }
            },
            (CtVal::EnumSym(a, _), CtVal::EnumSym(b, _)) => match op {
                BinaryOp::Eq => Ok(CtVal::Bool(a == b)),
                BinaryOp::NotEq => Ok(CtVal::Bool(a != b)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid enum binary op in comptime");
                    Err(())
                }
            },
            (CtVal::Str(a), CtVal::Str(b)) => match op {
                BinaryOp::Eq => Ok(CtVal::Bool(a == b)),
                BinaryOp::NotEq => Ok(CtVal::Bool(a != b)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid string binary op in comptime");
                    Err(())
                }
            },
            (CtVal::Float(a, ty), CtVal::Float(b, _)) => match op {
                BinaryOp::Add => Ok(CtVal::Float(a + b, *ty)),
                BinaryOp::Sub => Ok(CtVal::Float(a - b, *ty)),
                BinaryOp::Mul => Ok(CtVal::Float(a * b, *ty)),
                BinaryOp::Div => Ok(CtVal::Float(a / b, *ty)),
                BinaryOp::Eq => Ok(CtVal::Bool(a == b)),
                BinaryOp::NotEq => Ok(CtVal::Bool(a != b)),
                BinaryOp::Lt => Ok(CtVal::Bool(a < b)),
                BinaryOp::LtEq => Ok(CtVal::Bool(a <= b)),
                BinaryOp::Gt => Ok(CtVal::Bool(a > b)),
                BinaryOp::GtEq => Ok(CtVal::Bool(a >= b)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid float op in comptime");
                    Err(())
                }
            },
            (CtVal::Int(a, lt), CtVal::Int(b, rt)) => {
                let ty = if *lt != TyId::COMPTIME_INT { *lt } else { *rt };
                let bits = int_bitwidth(self.env.get(ty));
                match op {
                    BinaryOp::Eq => Ok(CtVal::Bool(a == b)),
                    BinaryOp::NotEq => Ok(CtVal::Bool(a != b)),
                    BinaryOp::Lt => Ok(CtVal::Bool(a < b)),
                    BinaryOp::LtEq => Ok(CtVal::Bool(a <= b)),
                    BinaryOp::Gt => Ok(CtVal::Bool(a > b)),
                    BinaryOp::GtEq => Ok(CtVal::Bool(a >= b)),
                    BinaryOp::Add => {
                        let Some(res) = a.checked_add(*b) else {
                            self.diags.error(ErrorCode::E0520, span, "comptime integer addition overflow");
                            return Err(());
                        };
                        self.check_int_range(res, ty, span)?;
                        Ok(CtVal::Int(res, ty))
                    }
                    BinaryOp::Sub => {
                        let Some(res) = a.checked_sub(*b) else {
                            self.diags.error(ErrorCode::E0520, span, "comptime integer subtraction overflow");
                            return Err(());
                        };
                        self.check_int_range(res, ty, span)?;
                        Ok(CtVal::Int(res, ty))
                    }
                    BinaryOp::Mul => {
                        let Some(res) = a.checked_mul(*b) else {
                            self.diags.error(ErrorCode::E0520, span, "comptime integer multiplication overflow");
                            return Err(());
                        };
                        self.check_int_range(res, ty, span)?;
                        Ok(CtVal::Int(res, ty))
                    }
                    BinaryOp::Div => {
                        if *b == 0 {
                            self.diags.error(ErrorCode::E0520, span, "comptime division by zero");
                            return Err(());
                        }
                        let Some(res) = a.checked_div(*b) else {
                            self.diags.error(ErrorCode::E0520, span, "comptime division overflow");
                            return Err(());
                        };
                        self.check_int_range(res, ty, span)?;
                        Ok(CtVal::Int(res, ty))
                    }
                    BinaryOp::Mod => {
                        if *b == 0 {
                            self.diags.error(ErrorCode::E0520, span, "comptime modulo by zero");
                            return Err(());
                        }
                        let Some(res) = a.checked_rem(*b) else {
                            self.diags.error(ErrorCode::E0520, span, "comptime modulo overflow");
                            return Err(());
                        };
                        self.check_int_range(res, ty, span)?;
                        Ok(CtVal::Int(res, ty))
                    }
                    BinaryOp::AddWrap => Ok(CtVal::Int(wrap_int(a.wrapping_add(*b), self.env.get(ty)), ty)),
                    BinaryOp::SubWrap => Ok(CtVal::Int(wrap_int(a.wrapping_sub(*b), self.env.get(ty)), ty)),
                    BinaryOp::MulWrap => Ok(CtVal::Int(wrap_int(a.wrapping_mul(*b), self.env.get(ty)), ty)),
                    BinaryOp::AddSat => Ok(CtVal::Int(sat_int(a.saturating_add(*b), self.env.get(ty)), ty)),
                    BinaryOp::SubSat => Ok(CtVal::Int(sat_int(a.saturating_sub(*b), self.env.get(ty)), ty)),
                    BinaryOp::MulSat => Ok(CtVal::Int(sat_int(a.saturating_mul(*b), self.env.get(ty)), ty)),
                    BinaryOp::BitAnd => Ok(CtVal::Int(wrap_int(a & b, self.env.get(ty)), ty)),
                    BinaryOp::BitOr => Ok(CtVal::Int(wrap_int(a | b, self.env.get(ty)), ty)),
                    BinaryOp::BitXor => Ok(CtVal::Int(wrap_int(a ^ b, self.env.get(ty)), ty)),
                    BinaryOp::Shl => {
                        if *b < 0 || (*b as u32) >= bits {
                            self.diags.error(
                                ErrorCode::E0520,
                                span,
                                format!("comptime left shift amount {} >= bitwidth {}", b, bits),
                            );
                            return Err(());
                        }
                        Ok(CtVal::Int(wrap_int(a.wrapping_shl(*b as u32), self.env.get(ty)), ty))
                    }
                    BinaryOp::ShlWrap => {
                        let mask = (bits - 1) as i128;
                        let sh = (*b & mask) as u32;
                        Ok(CtVal::Int(wrap_int(a.wrapping_shl(sh), self.env.get(ty)), ty))
                    }
                    BinaryOp::Shr => {
                        if *b < 0 || (*b as u32) >= bits {
                            self.diags.error(
                                ErrorCode::E0520,
                                span,
                                format!("comptime right shift amount {} >= bitwidth {}", b, bits),
                            );
                            return Err(());
                        }
                        let is_signed = matches!(self.env.get(ty), Ty::Int { signed: true, .. });
                        let res = if is_signed {
                            a >> (*b as u32)
                        } else {
                            let u = (*a as u128) & bitmask_u128(bits);
                            (u >> (*b as u32)) as i128
                        };
                        Ok(CtVal::Int(res, ty))
                    }
                    _ => {
                        self.diags.error(ErrorCode::E0520, span, "unsupported binary op in comptime");
                        Err(())
                    }
                }
            }
            _ => {
                self.diags.error(ErrorCode::E0520, span, "type mismatch in comptime binary op");
                Err(())
            }
        }
    }

    fn eval_cast(&mut self, kind: CtCastKind, v: &CtVal, to_ty: TyId, span: Span) -> Result<CtVal, ()> {
        match kind {
            CtCastKind::Widen => match v {
                CtVal::Int(n, _) => Ok(CtVal::Int(*n, to_ty)),
                CtVal::Float(f, _) => Ok(CtVal::Float(*f, to_ty)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid comptime .widen()");
                    Err(())
                }
            },
            CtCastKind::Trunc => match v {
                CtVal::Int(n, _) => Ok(CtVal::Int(wrap_int(*n, self.env.get(to_ty)), to_ty)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid comptime .trunc()");
                    Err(())
                }
            },
            CtCastKind::Clamp => match v {
                CtVal::Int(n, _) => Ok(CtVal::Int(sat_int(*n, self.env.get(to_ty)), to_ty)),
                _ => {
                    self.diags.error(ErrorCode::E0520, span, "invalid comptime .clamp()");
                    Err(())
                }
            },
            CtCastKind::ToFloat => match v {
                CtVal::Int(n, _) => Ok(CtVal::Float(*n as f64, to_ty)),
                CtVal::Float(f, _) => Ok(CtVal::Float(*f, to_ty)),
                _ => Err(()),
            },
            CtCastKind::TruncToInt => match v {
                CtVal::Float(f, _) => Ok(CtVal::Int(sat_int(*f as i128, self.env.get(to_ty)), to_ty)),
                _ => Err(()),
            },
            CtCastKind::Bitcast => match v {
                CtVal::Int(n, _) => Ok(CtVal::Int(wrap_int(*n, self.env.get(to_ty)), to_ty)),
                _ => Ok(v.clone()),
            },
        }
    }

    fn check_int_range(&mut self, v: i128, ty: TyId, span: Span) -> Result<(), ()> {
        if let Some((lo, hi)) = int_bounds(self.env.get(ty))
            && (v < lo || v > hi)
        {
            self.diags.error(
                ErrorCode::E0520,
                span,
                format!(
                    "comptime integer value {} out of range [{}..={}] for `{}`",
                    v,
                    lo,
                    hi,
                    self.env.format_ty(ty, self.interner)
                ),
            );
            return Err(());
        }
        Ok(())
    }
}

fn int_bitwidth(ty: &Ty) -> u32 {
    match ty {
        Ty::Int {
            bits,
            is_ptr_sized,
            ..
        } => {
            if *is_ptr_sized {
                64
            } else {
                *bits as u32
            }
        }
        _ => 64,
    }
}

fn bitmask_u128(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn wrap_int(v: i128, ty: &Ty) -> i128 {
    match ty {
        Ty::Int {
            signed,
            bits,
            is_ptr_sized,
        } => {
            let bw = if *is_ptr_sized { 64 } else { *bits as u32 };
            if bw >= 128 {
                return v;
            }
            let u = (v as u128) & bitmask_u128(bw);
            if *signed {
                let sign_bit = 1u128 << (bw - 1);
                if (u & sign_bit) != 0 {
                    (u | !bitmask_u128(bw)) as i128
                } else {
                    u as i128
                }
            } else {
                u as i128
            }
        }
        _ => v,
    }
}

fn sat_int(v: i128, ty: &Ty) -> i128 {
    if let Some((lo, hi)) = int_bounds(ty) {
        v.clamp(lo, hi)
    } else {
        v
    }
}

pub fn eval_expr(
    expr: &Expr,
    expected_ty: Option<TyId>,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
    fuel_limit: u64,
) -> Option<CtVal> {
    let mut comp = CtCompiler::new(mod_id, ctx, units, prog, env, interner, diags);
    let ret_r = comp.compile_expr(expr, expected_ty);
    if comp.failed {
        return None;
    }
    let ops = comp.ops;
    let nregs = comp.num_regs;
    let mut vm = CtVm::new(units, prog, env, interner, diags, fuel_limit);
    vm.exec(&ops, nregs, ret_r, expr.span).ok()
}

pub fn eval_const_bool(
    expr: &Expr,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Option<bool> {
    eval_expr(
        expr,
        Some(TyId::BOOL),
        mod_id,
        ctx,
        units,
        prog,
        env,
        interner,
        diags,
        DEFAULT_COMPTIME_FUEL,
    )
    .and_then(|v| v.as_bool())
}

pub fn eval_const_decl(
    expr: &Expr,
    ty: TyId,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Option<CtVal> {
    eval_expr(
        expr,
        Some(ty),
        mod_id,
        ctx,
        units,
        prog,
        env,
        interner,
        diags,
        DEFAULT_COMPTIME_FUEL,
    )
}

pub fn is_comptime_const_kind(ty: TyId, env: &TypeEnv) -> bool {
    matches!(
        env.get(ty),
        Ty::Int { .. }
            | Ty::Float(_)
            | Ty::Bool
            | Ty::Char
            | Ty::Array { .. }
            | Ty::Tuple(_)
            | Ty::Struct { .. }
            | Ty::AnonStruct(_)
            | Ty::Str
            | Ty::Borrow { inner: crate::types::BorrowInner::Str, .. }
            | Ty::Builtin(BuiltinKind::Ordering, _)
    )
}
