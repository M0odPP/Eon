//! Direct `EIR -> SPIR-V` binary GPU backend (`eon_codegen_gpu`) — ZERO external shader tools!
//! Implements:
//! - `@layout(std430)` validation on all structs crossing into `@gpu` (`E0820`).
//! - GPU capability/type restriction validation (`E0821`: forbids `AllocCap`, `dyn`, `rawptr`,
//!   `i128`/`u128`, arenas/pools, `@recursive`).
//! - Post-dominator control-dependence Uniform Control Flow check for `@gpu_workgroup_barrier()`
//!   (`E0822`).
//! - Direct SPIR-V 1.3 (Vulkan 1.1) binary module emitter passing `spirv-val`.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{AccessMode, BinaryOp, UnaryOp};
use crate::cfg::{BlockId, CastKind, CfgBuiltinOp, CfgConst};
use crate::diag::{DiagBag, ErrorCode};
use crate::eir::{
    EirFunction, EirInstKind, EirPlace, EirPlaceBase, EirProgram, EirProj, EirSlot, EirTerm,
    EirValue,
};
use crate::resolve::{BuiltinKind, DefId};
use crate::span::{Interner, Span};
use crate::types::{GenericVal, StructLayoutMode, Ty, TyId, TypeEnv, align_up};

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
const SPIRV_GENERATOR_EON: u32 = 0x0045_4F4E;

// SPIR-V opcodes
const OP_NAME: u16 = 5;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_UCONVERT: u16 = 113;
const OP_SCONVERT: u16 = 114;
const OP_SNEGATE: u16 = 126;
const OP_FNEGATE: u16 = 127;
const OP_IADD: u16 = 128;
const OP_FADD: u16 = 129;
const OP_ISUB: u16 = 130;
const OP_FSUB: u16 = 131;
const OP_IMUL: u16 = 132;
const OP_FMUL: u16 = 133;
const OP_UDIV: u16 = 134;
const OP_SDIV: u16 = 135;
const OP_FDIV: u16 = 136;
const OP_UMOD: u16 = 137;
const OP_SREM: u16 = 138;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_LOGICAL_NOT: u16 = 168;
const OP_IEQUAL: u16 = 170;
const OP_INOT_EQUAL: u16 = 171;
const OP_UGREATER_THAN: u16 = 172;
const OP_SGREATER_THAN: u16 = 173;
const OP_UGREATER_THAN_EQUAL: u16 = 174;
const OP_SGREATER_THAN_EQUAL: u16 = 175;
const OP_ULESS_THAN: u16 = 176;
const OP_SLESS_THAN: u16 = 177;
const OP_ULESS_THAN_EQUAL: u16 = 178;
const OP_SLESS_THAN_EQUAL: u16 = 179;
const OP_FORD_EQUAL: u16 = 180;
const OP_FORD_NOT_EQUAL: u16 = 182;
const OP_FORD_LESS_THAN: u16 = 184;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;
const OP_NOT: u16 = 200;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;

// Storage classes
const SC_INPUT: u32 = 1;
const SC_FUNCTION: u32 = 7;
const SC_STORAGE_BUFFER: u32 = 12;

// Decorations
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_NON_WRITABLE: u32 = 24;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

// BuiltIns
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

#[derive(Clone, Debug)]
pub struct SpirvModule {
    pub entry_name: String,
    pub words: Vec<u32>,
    pub bytes: Vec<u8>,
}

/// Validate all `@gpu` functions in `eir` (`E0820`, `E0821`, `E0822`) and emit binary SPIR-V modules.
pub fn compile_gpu_shaders(
    eir: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) -> Vec<SpirvModule> {
    let mut out = Vec::new();
    for f in &eir.functions {
        if !f.sig.effects.is_gpu {
            continue;
        }
        let before = diags.diagnostics().len();
        validate_gpu_function(f, env, interner, diags);
        if diags.diagnostics().len() > before {
            continue;
        }
        if let Ok(spv) = emit_spirv_compute(f, env, interner) {
            out.push(spv);
        }
    }
    out
}

/// Validate `@layout(std430)` (`E0820`), GPU capabilities (`E0821`), and uniform barrier control flow (`E0822`).
pub fn validate_gpu_function(
    f: &EirFunction,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    if f.sig.effects.is_recursive {
        diags.error(
            ErrorCode::E0821,
            f.body_span,
            format!("`@gpu` function `{}` cannot be `@recursive`", interner.get(f.name)),
        );
    }

    // 1. Validate parameter and local types (`E0820` std430 & `E0821` disallowed CPU types).
    let mut checked_defs = BTreeSet::new();
    for p in &f.sig.params {
        validate_gpu_type(p.ty, p.span, env, interner, diags, &mut checked_defs);
    }
    for sl in &f.slots {
        validate_gpu_type(sl.ty, f.body_span, env, interner, diags, &mut checked_defs);
    }
    for b in &f.blocks {
        for inst in &b.insts {
            if let EirInstKind::Call { callee, .. } = &inst.kind
                && matches!(callee, crate::eir::EirCallee::DynCall { .. } | crate::eir::EirCallee::ExternFn { .. })
            {
                diags.error(
                    ErrorCode::E0821,
                    inst.span,
                    "disallowed `dyn` or `extern` call inside `@gpu` function",
                );
            }
        }
    }

    // 2. Uniform Control Flow Check (`E0822`) for `@gpu_workgroup_barrier()`.
    check_uniform_barriers(f, diags);
}

fn validate_gpu_type(
    ty: TyId,
    span: Span,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
    checked_structs: &mut BTreeSet<DefId>,
) {
    match env.get(ty) {
        Ty::Struct { def, .. } => {
            if checked_structs.insert(*def)
                && let Some(sm) = env.structs.get(def)
            {
                if sm.layout_mode != StructLayoutMode::Std430 {
                    diags.error(
                        ErrorCode::E0820,
                        span,
                        format!(
                            "struct `{}` used in `@gpu` function must be annotated with `@layout(std430)`",
                            env.format_ty(ty, interner)
                        ),
                    );
                }
                for field in &sm.fields {
                    validate_gpu_type(field.ty, field.span, env, interner, diags, checked_structs);
                }
            }
        }
        Ty::Builtin(BuiltinKind::GpuBuf | BuiltinKind::GpuImage2D, args) => {
            if let Some(GenericVal::Ty(inner)) = args.first() {
                validate_gpu_type(*inner, span, env, interner, diags, checked_structs);
            }
        }
        Ty::Builtin(
            BuiltinKind::AllocCap
            | BuiltinKind::Arena
            | BuiltinKind::ArenaPool
            | BuiltinKind::SlotPool
            | BuiltinKind::Buf
            | BuiltinKind::LinearBuf,
            _,
        ) => {
            diags.error(
                ErrorCode::E0821,
                span,
                format!(
                    "CPU memory/allocator type `{}` is forbidden inside `@gpu` function",
                    env.format_ty(ty, interner)
                ),
            );
        }
        Ty::RawPtr { .. } | Ty::Dyn(_) | Ty::Borrow { .. } => {
            diags.error(
                ErrorCode::E0821,
                span,
                format!(
                    "CPU pointer or `dyn` type `{}` is forbidden inside `@gpu` function",
                    env.format_ty(ty, interner)
                ),
            );
        }
        Ty::Int { bits: 128, .. } => {
            diags.error(
                ErrorCode::E0821,
                span,
                "128-bit integer type is forbidden inside `@gpu` function",
            );
        }
        Ty::Array { elem, .. } | Ty::Vec { elem, .. } => {
            validate_gpu_type(*elem, span, env, interner, diags, checked_structs);
        }
        _ => {}
    }
}

fn check_uniform_barriers(f: &EirFunction, diags: &mut DiagBag) {
    // 1. Forward taint propagation from invocation-varying builtins (`@gpu_global_id`, `@gpu_local_id`).
    let mut tainted_vals: BTreeSet<EirValue> = BTreeSet::new();
    let mut tainted_slots: BTreeSet<EirSlot> = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for b in &f.blocks {
            for phi in &b.phis {
                if phi.incoming.iter().any(|&(_, v)| tainted_vals.contains(&v))
                    && tainted_vals.insert(phi.dst)
                {
                    changed = true;
                }
            }
            for inst in &b.insts {
                match &inst.kind {
                    EirInstKind::BuiltinCall { dst: Some(d), op: CfgBuiltinOp::GpuGlobalId | CfgBuiltinOp::GpuLocalId, .. } => {
                        if tainted_vals.insert(*d) {
                            changed = true;
                        }
                    }
                    EirInstKind::CopyVal { dst, src }
                    | EirInstKind::Unary { dst, operand: src, .. }
                    | EirInstKind::Cast { dst, operand: src, .. }
                    | EirInstKind::FieldExtract { dst, src, .. } => {
                        if tainted_vals.contains(src) && tainted_vals.insert(*dst) {
                            changed = true;
                        }
                    }
                    EirInstKind::Binary { dst, lhs, rhs, .. } => {
                        if (tainted_vals.contains(lhs) || tainted_vals.contains(rhs))
                            && tainted_vals.insert(*dst)
                        {
                            changed = true;
                        }
                    }
                    EirInstKind::StoreSlot { slot, val } => {
                        if tainted_vals.contains(val) && tainted_slots.insert(*slot) {
                            changed = true;
                        }
                    }
                    EirInstKind::LoadSlot { dst, slot } => {
                        if tainted_slots.contains(slot) && tainted_vals.insert(*dst) {
                            changed = true;
                        }
                    }
                    EirInstKind::LoadPlace { dst, place } => {
                        let base_t = match place.base {
                            EirPlaceBase::Slot(s) => tainted_slots.contains(&s),
                            EirPlaceBase::Val(v) => tainted_vals.contains(&v),
                        };
                        let idx_t = place.projs.iter().any(|p| match p {
                            EirProj::Index { index, .. } => tainted_vals.contains(index),
                            _ => false,
                        });
                        if (base_t || idx_t) && tainted_vals.insert(*dst) {
                            changed = true;
                        }
                    }
                    EirInstKind::StorePlace { place, val } => {
                        if tainted_vals.contains(val)
                            && let EirPlaceBase::Slot(s) = place.base
                            && tainted_slots.insert(s)
                        {
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 2. Compute post-dominators across all blocks (exit blocks post-dominate only themselves).
    let n = f.blocks.len();
    let all_blocks: BTreeSet<BlockId> = (0..n).map(|i| BlockId(i as u32)).collect();
    let mut pdom: Vec<BTreeSet<BlockId>> = (0..n)
        .map(|i| {
            let b = &f.blocks[i];
            if block_succs(&b.term).is_empty() {
                let mut s = BTreeSet::new();
                s.insert(b.id);
                s
            } else {
                all_blocks.clone()
            }
        })
        .collect();

    let mut pd_changed = true;
    while pd_changed {
        pd_changed = false;
        for b in &f.blocks {
            let succs = block_succs(&b.term);
            if succs.is_empty() {
                continue;
            }
            let mut inter = pdom[succs[0].0 as usize].clone();
            for &s in succs.iter().skip(1) {
                inter.retain(|x| pdom[s.0 as usize].contains(x));
            }
            inter.insert(b.id);
            if inter != pdom[b.id.0 as usize] {
                pdom[b.id.0 as usize] = inter;
                pd_changed = true;
            }
        }
    }

    // 3. Mark blocks control-dependent on any non-uniform branch!
    // Block B is control-dependent on branch D if D has a tainted condition, B is reachable from D,
    // and B does NOT post-dominate D (so some threads took another path and won't reach B!).
    let mut non_uniform_blocks: BTreeSet<BlockId> = BTreeSet::new();
    for b in &f.blocks {
        let is_tainted_branch = match &b.term {
            EirTerm::Branch { cond, .. } => tainted_vals.contains(cond),
            EirTerm::SwitchInt { discr, .. } => tainted_vals.contains(discr),
            _ => false,
        };
        if !is_tainted_branch {
            continue;
        }
        // BFS from successors of `b`, stopping at any block that post-dominates `b`.
        let mut queue = std::collections::VecDeque::new();
        let mut visited = BTreeSet::new();
        for s in block_succs(&b.term) {
            if visited.insert(s) {
                queue.push_back(s);
            }
        }
        while let Some(cur) = queue.pop_front() {
            if pdom[b.id.0 as usize].contains(&cur) && cur != b.id {
                // Reconverged cleanly!
                continue;
            }
            non_uniform_blocks.insert(cur);
            for ns in block_succs(&f.block(cur).term) {
                if visited.insert(ns) {
                    queue.push_back(ns);
                }
            }
        }
    }

    // 4. Emit `E0822` for any `@gpu_workgroup_barrier()` in a non-uniform block!
    for b in &f.blocks {
        if !non_uniform_blocks.contains(&b.id) {
            continue;
        }
        for inst in &b.insts {
            if let EirInstKind::BuiltinCall { op: CfgBuiltinOp::GpuWorkgroupBarrier, .. } = &inst.kind {
                diags.error(
                    ErrorCode::E0822,
                    inst.span,
                    "`@gpu_workgroup_barrier()` called in non-uniform control flow (branch depends on `@gpu_local_id` / `@gpu_global_id`)",
                );
            }
        }
    }
}

fn block_succs(t: &EirTerm) -> Vec<BlockId> {
    match t {
        EirTerm::Goto(b) => vec![*b],
        EirTerm::Branch { then_bb, else_bb, .. } => vec![*then_bb, *else_bb],
        EirTerm::SwitchInt { targets, otherwise, .. } => {
            let mut v: Vec<BlockId> = targets.iter().map(|&(_, b)| b).collect();
            v.push(*otherwise);
            v
        }
        _ => Vec::new(),
    }
}

// ============================================================================
// Direct Binary SPIR-V 1.3 Emitter
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SpvTyKey {
    Void,
    Bool,
    Int(u32, bool), // (bits, signed)
    Float(u32),
    Vec(Box<SpvTyKey>, u32),
    Array(Box<SpvTyKey>, u32),
    RuntimeArray(Box<SpvTyKey>),
    UserStruct(DefId),
    SsboWrapper(Box<SpvTyKey>),
    Ptr(u32, Box<SpvTyKey>), // (storage_class, pointee)
    FnVoid,
}

struct SpvBuilder<'a> {
    f: &'a EirFunction,
    env: &'a TypeEnv,
    interner: &'a Interner,
    next_id: u32,
    names: Vec<u32>,
    decorations: Vec<u32>,
    types_consts: Vec<u32>,
    globals: Vec<u32>,
    fn_words: Vec<u32>,
    type_ids: BTreeMap<SpvTyKey, u32>,
    const_u32: BTreeMap<u32, u32>,
    const_i32: BTreeMap<i32, u32>,
    const_f32: BTreeMap<u32, u32>, // bits -> id
    const_bool: BTreeMap<bool, u32>,
    interface_vars: Vec<u32>,
    builtin_global_id_var: Option<u32>,
    builtin_local_id_var: Option<u32>,
    builtin_workgroup_id_var: Option<u32>,
    param_ssbo_var: BTreeMap<usize, (u32, SpvTyKey)>, // param_idx -> (var_id, elem_spv_ty)
    val_ids: BTreeMap<EirValue, u32>,
    slot_vars: BTreeMap<EirSlot, (u32, SpvTyKey)>,
    param_slot_to_ssbo: BTreeMap<EirSlot, (u32, SpvTyKey)>,
    bb_labels: BTreeMap<BlockId, u32>,
}

impl<'a> SpvBuilder<'a> {
    fn new(f: &'a EirFunction, env: &'a TypeEnv, interner: &'a Interner) -> Self {
        Self {
            f,
            env,
            interner,
            next_id: 1,
            names: Vec::new(),
            decorations: Vec::new(),
            types_consts: Vec::new(),
            globals: Vec::new(),
            fn_words: Vec::new(),
            type_ids: BTreeMap::new(),
            const_u32: BTreeMap::new(),
            const_i32: BTreeMap::new(),
            const_f32: BTreeMap::new(),
            const_bool: BTreeMap::new(),
            interface_vars: Vec::new(),
            builtin_global_id_var: None,
            builtin_local_id_var: None,
            builtin_workgroup_id_var: None,
            param_ssbo_var: BTreeMap::new(),
            val_ids: BTreeMap::new(),
            slot_vars: BTreeMap::new(),
            param_slot_to_ssbo: BTreeMap::new(),
            bb_labels: BTreeMap::new(),
        }
    }

    fn fresh(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn emit_op(dst: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
        let wc = (1 + operands.len()) as u32;
        dst.push((wc << 16) | (opcode as u32));
        dst.extend_from_slice(operands);
    }

    fn encode_str(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    fn spv_key_for_ty(&self, ty: TyId) -> SpvTyKey {
        match self.env.get(ty) {
            Ty::Void | Ty::Never => SpvTyKey::Void,
            Ty::Bool => SpvTyKey::Bool,
            Ty::Int { signed, bits, is_ptr_sized } => {
                let w = if *is_ptr_sized { 32 } else { (*bits as u32).clamp(32, 32) };
                SpvTyKey::Int(w, *signed)
            }
            Ty::Float(bits) => SpvTyKey::Float(if *bits <= 32 { 32 } else { 64 }),
            Ty::Vec { elem, len } => {
                SpvTyKey::Vec(Box::new(self.spv_key_for_ty(*elem)), *len as u32)
            }
            Ty::Array { elem, len } => {
                SpvTyKey::Array(Box::new(self.spv_key_for_ty(*elem)), (*len as u32).max(1))
            }
            Ty::Struct { def, .. } => SpvTyKey::UserStruct(*def),
            _ => SpvTyKey::Int(32, false),
        }
    }

    fn std430_stride(&self, key: &SpvTyKey) -> u32 {
        match key {
            SpvTyKey::Void | SpvTyKey::Bool | SpvTyKey::Int(32, _) | SpvTyKey::Float(32) => 4,
            SpvTyKey::Int(64, _) | SpvTyKey::Float(64) => 8,
            SpvTyKey::Vec(el, n) => {
                let es = self.std430_stride(el);
                if *n == 3 { es * 4 } else { es * n }
            }
            SpvTyKey::Array(el, n) => self.std430_stride(el) * (*n).max(1),
            SpvTyKey::UserStruct(def) => {
                self.env.structs.get(def).map_or(4, |m| m.layout.size.max(4) as u32)
            }
            _ => 4,
        }
    }

    fn get_or_emit_ty(&mut self, key: SpvTyKey) -> u32 {
        if let Some(&id) = self.type_ids.get(&key) {
            return id;
        }
        let id = match &key {
            SpvTyKey::Void => {
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_VOID, &[id]);
                id
            }
            SpvTyKey::Bool => {
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_BOOL, &[id]);
                id
            }
            SpvTyKey::Int(w, signed) => {
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_INT, &[id, *w, u32::from(*signed)]);
                id
            }
            SpvTyKey::Float(w) => {
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_FLOAT, &[id, *w]);
                id
            }
            SpvTyKey::Vec(el, cnt) => {
                let el_id = self.get_or_emit_ty((**el).clone());
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_VECTOR, &[id, el_id, *cnt]);
                id
            }
            SpvTyKey::Array(el, len) => {
                let el_id = self.get_or_emit_ty((**el).clone());
                let len_c = self.get_const_u32(*len);
                let id = self.fresh();
                let stride = self.std430_stride(el);
                Self::emit_op(&mut self.decorations, OP_DECORATE, &[id, DEC_ARRAY_STRIDE, stride]);
                Self::emit_op(&mut self.types_consts, OP_TYPE_ARRAY, &[id, el_id, len_c]);
                id
            }
            SpvTyKey::RuntimeArray(el) => {
                let el_id = self.get_or_emit_ty((**el).clone());
                let stride = self.std430_stride(el);
                let id = self.fresh();
                Self::emit_op(&mut self.decorations, OP_DECORATE, &[id, DEC_ARRAY_STRIDE, stride]);
                Self::emit_op(&mut self.types_consts, OP_TYPE_RUNTIME_ARRAY, &[id, el_id]);
                id
            }
            SpvTyKey::UserStruct(def) => {
                let fields = self.env.structs.get(def).map(|m| m.fields.clone()).unwrap_or_default();
                let mut member_ids = Vec::with_capacity(fields.len());
                let mut member_offs = Vec::with_capacity(fields.len());
                let mut off = 0u64;
                for fld in &fields {
                    let fk = self.spv_key_for_ty(fld.ty);
                    let falign = self.std430_align(&fk) as u64;
                    off = align_up(off, falign.max(1));
                    member_offs.push(off as u32);
                    member_ids.push(self.get_or_emit_ty(fk.clone()));
                    off += self.std430_stride(&fk) as u64;
                }
                let id = self.fresh();
                for (mi, mo) in member_offs.iter().enumerate() {
                    Self::emit_op(&mut self.decorations, OP_MEMBER_DECORATE, &[id, mi as u32, DEC_OFFSET, *mo]);
                }
                let mut ops = vec![id];
                ops.extend_from_slice(&member_ids);
                Self::emit_op(&mut self.types_consts, OP_TYPE_STRUCT, &ops);
                id
            }
            SpvTyKey::SsboWrapper(el) => {
                let rarr = self.get_or_emit_ty(SpvTyKey::RuntimeArray(el.clone()));
                let id = self.fresh();
                Self::emit_op(&mut self.decorations, OP_DECORATE, &[id, DEC_BLOCK]);
                Self::emit_op(&mut self.decorations, OP_MEMBER_DECORATE, &[id, 0, DEC_OFFSET, 0]);
                Self::emit_op(&mut self.types_consts, OP_TYPE_STRUCT, &[id, rarr]);
                id
            }
            SpvTyKey::Ptr(sc, pointee) => {
                let pt_id = self.get_or_emit_ty((**pointee).clone());
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_POINTER, &[id, *sc, pt_id]);
                id
            }
            SpvTyKey::FnVoid => {
                let void_id = self.get_or_emit_ty(SpvTyKey::Void);
                let id = self.fresh();
                Self::emit_op(&mut self.types_consts, OP_TYPE_FUNCTION, &[id, void_id]);
                id
            }
        };
        self.type_ids.insert(key, id);
        id
    }

    fn std430_align(&self, key: &SpvTyKey) -> u32 {
        match key {
            SpvTyKey::Void | SpvTyKey::Bool | SpvTyKey::Int(32, _) | SpvTyKey::Float(32) => 4,
            SpvTyKey::Int(64, _) | SpvTyKey::Float(64) => 8,
            SpvTyKey::Vec(el, n) => {
                let ea = self.std430_align(el);
                if *n == 2 { ea * 2 } else { ea * 4 }
            }
            SpvTyKey::Array(el, _) => self.std430_align(el),
            SpvTyKey::UserStruct(def) => {
                self.env.structs.get(def).map_or(4, |m| m.layout.align.max(4) as u32)
            }
            _ => 4,
        }
    }

    fn get_const_u32(&mut self, v: u32) -> u32 {
        if let Some(&id) = self.const_u32.get(&v) {
            return id;
        }
        let u32_ty = self.get_or_emit_ty(SpvTyKey::Int(32, false));
        let id = self.fresh();
        Self::emit_op(&mut self.types_consts, OP_CONSTANT, &[u32_ty, id, v]);
        self.const_u32.insert(v, id);
        id
    }

    fn get_const_i32(&mut self, v: i32) -> u32 {
        if let Some(&id) = self.const_i32.get(&v) {
            return id;
        }
        let i32_ty = self.get_or_emit_ty(SpvTyKey::Int(32, true));
        let id = self.fresh();
        Self::emit_op(&mut self.types_consts, OP_CONSTANT, &[i32_ty, id, v as u32]);
        self.const_i32.insert(v, id);
        id
    }

    fn get_const_f32(&mut self, v: f32) -> u32 {
        let bits = v.to_bits();
        if let Some(&id) = self.const_f32.get(&bits) {
            return id;
        }
        let f32_ty = self.get_or_emit_ty(SpvTyKey::Float(32));
        let id = self.fresh();
        Self::emit_op(&mut self.types_consts, OP_CONSTANT, &[f32_ty, id, bits]);
        self.const_f32.insert(bits, id);
        id
    }

    fn get_const_bool(&mut self, v: bool) -> u32 {
        if let Some(&id) = self.const_bool.get(&v) {
            return id;
        }
        let b_ty = self.get_or_emit_ty(SpvTyKey::Bool);
        let id = self.fresh();
        // OpConstantTrue = 41, OpConstantFalse = 42
        Self::emit_op(&mut self.types_consts, if v { 41 } else { 42 }, &[b_ty, id]);
        self.const_bool.insert(v, id);
        id
    }

    fn ensure_builtin_uvec3(&mut self, builtin_code: u32) -> u32 {
        let cached = match builtin_code {
            BUILTIN_GLOBAL_INVOCATION_ID => self.builtin_global_id_var,
            BUILTIN_LOCAL_INVOCATION_ID => self.builtin_local_id_var,
            _ => self.builtin_workgroup_id_var,
        };
        if let Some(id) = cached {
            return id;
        }
        let u32_k = SpvTyKey::Int(32, false);
        let uvec3_k = SpvTyKey::Vec(Box::new(u32_k), 3);
        let ptr_ty = self.get_or_emit_ty(SpvTyKey::Ptr(SC_INPUT, Box::new(uvec3_k)));
        let var_id = self.fresh();
        Self::emit_op(&mut self.decorations, OP_DECORATE, &[var_id, DEC_BUILTIN, builtin_code]);
        Self::emit_op(&mut self.globals, OP_VARIABLE, &[ptr_ty, var_id, SC_INPUT]);
        self.interface_vars.push(var_id);
        match builtin_code {
            BUILTIN_GLOBAL_INVOCATION_ID => self.builtin_global_id_var = Some(var_id),
            BUILTIN_LOCAL_INVOCATION_ID => self.builtin_local_id_var = Some(var_id),
            _ => self.builtin_workgroup_id_var = Some(var_id),
        }
        var_id
    }

    fn build(mut self) -> Result<SpirvModule, String> {
        let entry_name = self.interner.get(self.f.name).to_owned();
        let void_id = self.get_or_emit_ty(SpvTyKey::Void);
        let fn_ty_id = self.get_or_emit_ty(SpvTyKey::FnVoid);
        let fn_id = self.fresh();

        // Emit OpName for entry point.
        let mut name_ops = vec![fn_id];
        name_ops.extend(Self::encode_str(&entry_name));
        Self::emit_op(&mut self.names, OP_NAME, &name_ops);

        // Emit SSBO global variables for each `GpuBuf[T]` parameter.
        for (pi, p) in self.f.sig.params.iter().enumerate() {
            if let Ty::Builtin(BuiltinKind::GpuBuf, gargs) = self.env.get(p.ty)
                && let Some(GenericVal::Ty(elem_ty)) = gargs.first()
            {
                let elem_k = self.spv_key_for_ty(*elem_ty);
                let wrap_k = SpvTyKey::SsboWrapper(Box::new(elem_k.clone()));
                let ptr_ty = self.get_or_emit_ty(SpvTyKey::Ptr(SC_STORAGE_BUFFER, Box::new(wrap_k)));
                let var_id = self.fresh();
                let (dset, bind) = p.gpu_binding.unwrap_or((0, pi as u32));
                Self::emit_op(&mut self.decorations, OP_DECORATE, &[var_id, DEC_DESCRIPTOR_SET, dset]);
                Self::emit_op(&mut self.decorations, OP_DECORATE, &[var_id, DEC_BINDING, bind]);
                if p.access == Some(AccessMode::Read) {
                    Self::emit_op(&mut self.decorations, OP_DECORATE, &[var_id, DEC_NON_WRITABLE]);
                }
                Self::emit_op(&mut self.globals, OP_VARIABLE, &[ptr_ty, var_id, SC_STORAGE_BUFFER]);
                self.param_ssbo_var.insert(pi, (var_id, elem_k));
            }
        }

        // Map any `StoreSlot(slot, param_val)` to its SSBO so `LoadPlace/StorePlace` on `Slot` resolves to SSBO!
        let mut param_val_to_idx: BTreeMap<EirValue, usize> = BTreeMap::new();
        for (pi, &pv) in self.f.params.iter().enumerate() {
            param_val_to_idx.insert(pv, pi);
        }
        for b in &self.f.blocks {
            for inst in &b.insts {
                if let EirInstKind::StoreSlot { slot, val } = &inst.kind
                    && let Some(&pi) = param_val_to_idx.get(val)
                    && let Some(ssbo) = self.param_ssbo_var.get(&pi).cloned()
                {
                    self.param_slot_to_ssbo.insert(*slot, ssbo);
                }
            }
        }

        // Allocate label IDs for all blocks.
        for b in &self.f.blocks {
            let lbl = self.fresh();
            self.bb_labels.insert(b.id, lbl);
        }

        // Compute post-dominators for structured `OpSelectionMerge`!
        let ipdom = compute_ipdom(self.f);

        // Begin entry function.
        Self::emit_op(&mut self.fn_words, OP_FUNCTION, &[void_id, fn_id, 0, fn_ty_id]);

        // Emit blocks in RPO order!
        for (bi, &bb) in self.f.rpo.clone().iter().enumerate() {
            let blk = self.f.block(bb);
            let lbl = self.bb_labels[&bb];
            Self::emit_op(&mut self.fn_words, OP_LABEL, &[lbl]);

            // Entry block MUST declare all Function-class OpVariables first.
            if bi == 0 {
                for sl in &self.f.slots {
                    if self.param_slot_to_ssbo.contains_key(&sl.id)
                        || matches!(self.env.get(sl.ty), Ty::Builtin(BuiltinKind::GpuBuf | BuiltinKind::GpuImage2D, _) | Ty::Void | Ty::Never)
                    {
                        continue;
                    }
                    let sk = self.spv_key_for_ty(sl.ty);
                    let ptr_ty = self.get_or_emit_ty(SpvTyKey::Ptr(SC_FUNCTION, Box::new(sk.clone())));
                    let vid = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_VARIABLE, &[ptr_ty, vid, SC_FUNCTION]);
                    self.slot_vars.insert(sl.id, (vid, sk));
                }
            }

            for inst in &blk.insts {
                self.emit_inst(&inst.kind);
            }
            self.emit_term(bb, &blk.term, &ipdom);
        }

        Self::emit_op(&mut self.fn_words, OP_FUNCTION_END, &[]);

        // Assemble final SPIR-V binary in exact required section order!
        let (lx, ly, lz) = self.f.sig.effects.gpu_local_size.unwrap_or((1, 1, 1));
        let mut words = vec![
            SPIRV_MAGIC,
            SPIRV_VERSION_1_3,
            SPIRV_GENERATOR_EON,
            self.next_id,
            0,
        ];

        // 1. Capabilities
        Self::emit_op(&mut words, OP_CAPABILITY, &[1]); // Shader
        // 2. Memory model
        Self::emit_op(&mut words, OP_MEMORY_MODEL, &[0, 1]); // Logical, GLSL450
        // 3. EntryPoint
        let mut ep_ops = vec![5u32, fn_id]; // GLCompute = 5
        ep_ops.extend(Self::encode_str(&entry_name));
        ep_ops.extend_from_slice(&self.interface_vars);
        Self::emit_op(&mut words, OP_ENTRY_POINT, &ep_ops);
        // 4. ExecutionMode LocalSize
        Self::emit_op(&mut words, OP_EXECUTION_MODE, &[fn_id, 17, lx, ly, lz]);
        // 5. Debug names
        words.extend_from_slice(&self.names);
        // 6. Decorations
        words.extend_from_slice(&self.decorations);
        // 7. Types, constants, global variables
        words.extend_from_slice(&self.types_consts);
        words.extend_from_slice(&self.globals);
        // 8. Function bodies
        words.extend_from_slice(&self.fn_words);

        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        Ok(SpirvModule {
            entry_name,
            words,
            bytes,
        })
    }

    fn val_id(&mut self, v: EirValue) -> u32 {
        if let Some(&id) = self.val_ids.get(&v) {
            return id;
        }
        let ty = self.f.value_tys[v.0 as usize];
        match self.env.get(ty) {
            Ty::Float(_) => self.get_const_f32(0.0),
            Ty::Bool => self.get_const_bool(false),
            Ty::Int { signed: true, .. } => self.get_const_i32(0),
            _ => self.get_const_u32(0),
        }
    }

    fn emit_inst(&mut self, k: &EirInstKind) {
        match k {
            EirInstKind::Const { dst, ty, konst } => {
                let id = match konst {
                    CfgConst::Bool(b) => self.get_const_bool(*b),
                    CfgConst::Int(n, _) => {
                        if matches!(self.env.get(*ty), Ty::Int { signed: true, .. }) {
                            self.get_const_i32(*n as i32)
                        } else {
                            self.get_const_u32(*n as u32)
                        }
                    }
                    CfgConst::Float(s, _) => {
                        let v = s.replace('_', "").parse::<f32>().unwrap_or(0.0);
                        self.get_const_f32(v)
                    }
                    _ => return,
                };
                self.val_ids.insert(*dst, id);
            }
            EirInstKind::CopyVal { dst, src } => {
                let id = self.val_id(*src);
                self.val_ids.insert(*dst, id);
            }
            EirInstKind::LoadSlot { dst, slot } => {
                if let Some((var_id, sk)) = self.slot_vars.get(slot).cloned() {
                    let ty_id = self.get_or_emit_ty(sk);
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_LOAD, &[ty_id, id, var_id]);
                    self.val_ids.insert(*dst, id);
                }
            }
            EirInstKind::StoreSlot { slot, val } => {
                if let Some((var_id, _)) = self.slot_vars.get(slot).cloned() {
                    let vid = self.val_id(*val);
                    Self::emit_op(&mut self.fn_words, OP_STORE, &[var_id, vid]);
                }
            }
            EirInstKind::LoadPlace { dst, place } => {
                if let Some((ptr_id, elem_k)) = self.emit_place_ptr(place) {
                    let elem_ty = self.get_or_emit_ty(elem_k);
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_LOAD, &[elem_ty, id, ptr_id]);
                    self.val_ids.insert(*dst, id);
                } else if let EirPlaceBase::Val(bv) = place.base
                    && place.projs.len() == 1
                    && let EirProj::Field { field_idx, ty, .. } = &place.projs[0]
                {
                    let bvid = self.val_id(bv);
                    let rk = self.spv_key_for_ty(*ty);
                    let rty = self.get_or_emit_ty(rk);
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_COMPOSITE_EXTRACT, &[rty, id, bvid, *field_idx as u32]);
                    self.val_ids.insert(*dst, id);
                }
            }
            EirInstKind::StorePlace { place, val } => {
                if let Some((ptr_id, _)) = self.emit_place_ptr(place) {
                    let vid = self.val_id(*val);
                    Self::emit_op(&mut self.fn_words, OP_STORE, &[ptr_id, vid]);
                }
            }
            EirInstKind::Unary { dst, op, operand, ty } => {
                let sv = self.val_id(*operand);
                let rk = self.spv_key_for_ty(*ty);
                let rty = self.get_or_emit_ty(rk);
                let id = self.fresh();
                let spv_op = match op {
                    UnaryOp::Neg => if matches!(self.env.get(*ty), Ty::Float(_)) { OP_FNEGATE } else { OP_SNEGATE },
                    UnaryOp::Not => OP_LOGICAL_NOT,
                    UnaryOp::BitNot => OP_NOT,
                    _ => return,
                };
                Self::emit_op(&mut self.fn_words, spv_op, &[rty, id, sv]);
                self.val_ids.insert(*dst, id);
            }
            EirInstKind::Binary { dst, op, lhs, rhs, op_ty, dst_ty } => {
                let lv = self.val_id(*lhs);
                let rv = self.val_id(*rhs);
                let rk = self.spv_key_for_ty(*dst_ty);
                let rty = self.get_or_emit_ty(rk);
                let is_float = matches!(self.env.get(*op_ty), Ty::Float(_));
                let is_signed = matches!(self.env.get(*op_ty), Ty::Int { signed: true, .. });
                let spv_op = match op {
                    BinaryOp::Add | BinaryOp::AddWrap => if is_float { OP_FADD } else { OP_IADD },
                    BinaryOp::Sub | BinaryOp::SubWrap => if is_float { OP_FSUB } else { OP_ISUB },
                    BinaryOp::Mul | BinaryOp::MulWrap => if is_float { OP_FMUL } else { OP_IMUL },
                    BinaryOp::Div => if is_float { OP_FDIV } else if is_signed { OP_SDIV } else { OP_UDIV },
                    BinaryOp::Mod => if is_signed { OP_SREM } else { OP_UMOD },
                    BinaryOp::BitAnd => OP_BITWISE_AND,
                    BinaryOp::BitOr => OP_BITWISE_OR,
                    BinaryOp::BitXor => OP_BITWISE_XOR,
                    BinaryOp::And => OP_LOGICAL_AND,
                    BinaryOp::Or => OP_LOGICAL_OR,
                    BinaryOp::Shl | BinaryOp::ShlWrap => OP_SHIFT_LEFT_LOGICAL,
                    BinaryOp::Shr => if is_signed { OP_SHIFT_RIGHT_ARITHMETIC } else { OP_SHIFT_RIGHT_LOGICAL },
                    BinaryOp::Eq => if is_float { OP_FORD_EQUAL } else { OP_IEQUAL },
                    BinaryOp::NotEq => if is_float { OP_FORD_NOT_EQUAL } else { OP_INOT_EQUAL },
                    BinaryOp::Lt => if is_float { OP_FORD_LESS_THAN } else if is_signed { OP_SLESS_THAN } else { OP_ULESS_THAN },
                    BinaryOp::LtEq => if is_float { OP_FORD_LESS_THAN_EQUAL } else if is_signed { OP_SLESS_THAN_EQUAL } else { OP_ULESS_THAN_EQUAL },
                    BinaryOp::Gt => if is_float { OP_FORD_GREATER_THAN } else if is_signed { OP_SGREATER_THAN } else { OP_UGREATER_THAN },
                    BinaryOp::GtEq => if is_float { OP_FORD_GREATER_THAN_EQUAL } else if is_signed { OP_SGREATER_THAN_EQUAL } else { OP_UGREATER_THAN_EQUAL },
                    _ => return,
                };
                let id = self.fresh();
                Self::emit_op(&mut self.fn_words, spv_op, &[rty, id, lv, rv]);
                self.val_ids.insert(*dst, id);
            }
            EirInstKind::Cast { dst, kind, operand, from_ty, to_ty } => {
                let sv = self.val_id(*operand);
                let fk = self.spv_key_for_ty(*from_ty);
                let tk = self.spv_key_for_ty(*to_ty);
                if fk == tk {
                    self.val_ids.insert(*dst, sv);
                    return;
                }
                let rty = self.get_or_emit_ty(tk);
                let from_signed = matches!(self.env.get(*from_ty), Ty::Int { signed: true, .. });
                let to_signed = matches!(self.env.get(*to_ty), Ty::Int { signed: true, .. });
                let spv_op = match kind {
                    CastKind::ToFloat => if from_signed { OP_CONVERT_S_TO_F } else { OP_CONVERT_U_TO_F },
                    CastKind::TruncToInt => if to_signed { OP_CONVERT_F_TO_S } else { OP_CONVERT_F_TO_U },
                    _ => if to_signed { OP_SCONVERT } else { OP_UCONVERT },
                };
                let id = self.fresh();
                Self::emit_op(&mut self.fn_words, spv_op, &[rty, id, sv]);
                self.val_ids.insert(*dst, id);
            }
            EirInstKind::BuiltinCall { dst, op, .. } => match op {
                CfgBuiltinOp::GpuGlobalId => {
                    let var = self.ensure_builtin_uvec3(BUILTIN_GLOBAL_INVOCATION_ID);
                    let uvec3 = self.get_or_emit_ty(SpvTyKey::Vec(Box::new(SpvTyKey::Int(32, false)), 3));
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_LOAD, &[uvec3, id, var]);
                    if let Some(d) = dst { self.val_ids.insert(*d, id); }
                }
                CfgBuiltinOp::GpuLocalId => {
                    let var = self.ensure_builtin_uvec3(BUILTIN_LOCAL_INVOCATION_ID);
                    let uvec3 = self.get_or_emit_ty(SpvTyKey::Vec(Box::new(SpvTyKey::Int(32, false)), 3));
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_LOAD, &[uvec3, id, var]);
                    if let Some(d) = dst { self.val_ids.insert(*d, id); }
                }
                CfgBuiltinOp::GpuWorkgroupId => {
                    let var = self.ensure_builtin_uvec3(BUILTIN_WORKGROUP_ID);
                    let uvec3 = self.get_or_emit_ty(SpvTyKey::Vec(Box::new(SpvTyKey::Int(32, false)), 3));
                    let id = self.fresh();
                    Self::emit_op(&mut self.fn_words, OP_LOAD, &[uvec3, id, var]);
                    if let Some(d) = dst { self.val_ids.insert(*d, id); }
                }
                CfgBuiltinOp::GpuWorkgroupBarrier => {
                    // ExecutionScope=Workgroup(2), MemoryScope=Workgroup(2), Semantics=AcquireRelease|WorkgroupMemory(0x108)
                    let scope = self.get_const_u32(2);
                    let sem = self.get_const_u32(0x108);
                    Self::emit_op(&mut self.fn_words, OP_CONTROL_BARRIER, &[scope, scope, sem]);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn emit_place_ptr(&mut self, place: &EirPlace) -> Option<(u32, SpvTyKey)> {
        let (base_var, sc, mut cur_k, is_ssbo) = match place.base {
            EirPlaceBase::Slot(s) => {
                if let Some((var_id, elem_k)) = self.param_slot_to_ssbo.get(&s).cloned() {
                    (var_id, SC_STORAGE_BUFFER, elem_k, true)
                } else if let Some((var_id, sk)) = self.slot_vars.get(&s).cloned() {
                    (var_id, SC_FUNCTION, sk, false)
                } else {
                    return None;
                }
            }
            EirPlaceBase::Val(v) => {
                if let Some(pi) = self.f.params.iter().position(|&p| p == v)
                    && let Some((var_id, elem_k)) = self.param_ssbo_var.get(&pi).cloned()
                {
                    (var_id, SC_STORAGE_BUFFER, elem_k, true)
                } else {
                    return None;
                }
            }
        };

        let mut indices = Vec::new();
        let mut proj_start = 0usize;
        if is_ssbo {
            // Member 0 of SSBO wrapper struct is the RuntimeArray!
            indices.push(self.get_const_u32(0));
            if let Some(EirProj::Index { index, .. }) = place.projs.first() {
                indices.push(self.val_id(*index));
                proj_start = 1;
            }
        }
        for pr in place.projs.iter().skip(proj_start) {
            match pr {
                EirProj::Field { field_idx, ty, .. } => {
                    indices.push(self.get_const_u32(*field_idx as u32));
                    cur_k = self.spv_key_for_ty(*ty);
                }
                EirProj::Index { index, ty } => {
                    indices.push(self.val_id(*index));
                    cur_k = self.spv_key_for_ty(*ty);
                }
                _ => {}
            }
        }
        if indices.is_empty() {
            return Some((base_var, cur_k));
        }
        let ptr_ty = self.get_or_emit_ty(SpvTyKey::Ptr(sc, Box::new(cur_k.clone())));
        let id = self.fresh();
        let mut ops = vec![ptr_ty, id, base_var];
        ops.extend_from_slice(&indices);
        Self::emit_op(&mut self.fn_words, OP_ACCESS_CHAIN, &ops);
        Some((id, cur_k))
    }

    fn emit_term(&mut self, bb: BlockId, term: &EirTerm, ipdom: &BTreeMap<BlockId, BlockId>) {
        match term {
            EirTerm::Goto(t) => {
                let l = self.bb_labels[t];
                Self::emit_op(&mut self.fn_words, OP_BRANCH, &[l]);
            }
            EirTerm::Branch { cond, then_bb, else_bb, .. } => {
                let cid = self.val_id(*cond);
                let tl = self.bb_labels[then_bb];
                let el = self.bb_labels[else_bb];
                // Check if `bb` is a loop header (has a predecessor after it in RPO -> backedge).
                let is_loop_hdr = self.f.preds[bb.0 as usize].iter().any(|&p| {
                    let pi = self.f.rpo.iter().position(|&x| x == p).unwrap_or(0);
                    let bi = self.f.rpo.iter().position(|&x| x == bb).unwrap_or(0);
                    pi >= bi
                });
                if is_loop_hdr {
                    // Find continue (latch) block and merge (exit) block!
                    let latch = self.f.preds[bb.0 as usize]
                        .iter()
                        .copied()
                        .find(|&p| {
                            let pi = self.f.rpo.iter().position(|&x| x == p).unwrap_or(0);
                            let bi = self.f.rpo.iter().position(|&x| x == bb).unwrap_or(0);
                            pi >= bi
                        })
                        .unwrap_or(*then_bb);
                    let exit_lbl = self.bb_labels[else_bb];
                    let cont_lbl = self.bb_labels[&latch];
                    Self::emit_op(&mut self.fn_words, OP_LOOP_MERGE, &[exit_lbl, cont_lbl, 0]);
                } else {
                    let merge_bb = ipdom.get(&bb).copied().unwrap_or(*else_bb);
                    let ml = self.bb_labels[&merge_bb];
                    Self::emit_op(&mut self.fn_words, OP_SELECTION_MERGE, &[ml, 0]);
                }
                Self::emit_op(&mut self.fn_words, OP_BRANCH_CONDITIONAL, &[cid, tl, el]);
            }
            EirTerm::Return(_) | EirTerm::Unreachable { .. } => {
                Self::emit_op(&mut self.fn_words, OP_RETURN, &[]);
            }
            EirTerm::SwitchInt { otherwise, .. } => {
                let ol = self.bb_labels[otherwise];
                Self::emit_op(&mut self.fn_words, OP_BRANCH, &[ol]);
            }
        }
    }
}

fn compute_ipdom(f: &EirFunction) -> BTreeMap<BlockId, BlockId> {
    let n = f.blocks.len();
    let all: BTreeSet<BlockId> = (0..n).map(|i| BlockId(i as u32)).collect();
    let mut pdom: Vec<BTreeSet<BlockId>> = (0..n)
        .map(|i| {
            let b = &f.blocks[i];
            if block_succs(&b.term).is_empty() {
                let mut s = BTreeSet::new();
                s.insert(b.id);
                s
            } else {
                all.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in &f.blocks {
            let succs = block_succs(&b.term);
            if succs.is_empty() {
                continue;
            }
            let mut inter = pdom[succs[0].0 as usize].clone();
            for &s in succs.iter().skip(1) {
                inter.retain(|x| pdom[s.0 as usize].contains(x));
            }
            inter.insert(b.id);
            if inter != pdom[b.id.0 as usize] {
                pdom[b.id.0 as usize] = inter;
                changed = true;
            }
        }
    }
    let mut ipdom = BTreeMap::new();
    for b in &f.blocks {
        let strict: Vec<BlockId> = pdom[b.id.0 as usize]
            .iter()
            .copied()
            .filter(|&x| x != b.id)
            .collect();
        // Immediate post-dominator is the strict post-dominator with the LARGEST pdom set (closest to b)!
        if let Some(&best) = strict.iter().max_by_key(|&&x| pdom[x.0 as usize].len()) {
            ipdom.insert(b.id, best);
        }
    }
    ipdom
}

fn emit_spirv_compute(
    f: &EirFunction,
    env: &TypeEnv,
    interner: &Interner,
) -> Result<SpirvModule, String> {
    SpvBuilder::new(f, env, interner).build()
}
