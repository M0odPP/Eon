//! Direct `EIR -> x86_64` native backend (`eon_codegen_x64`) — ZERO C compiler, ZERO LLVM!
//! Implements:
//! - Full `EonCall` ABI (8 GP arg regs `rdi, rsi, rdx, rcx, r8, r9, r10, r11`, 2-reg slices/fat-ptrs,
//!   hidden `sret_ptr` for >16B returns, error return in dedicated `r15` + `CF` carry flag).
//! - `@extern("C")` / `extern "C" fn` System V AMD64 calling convention + Win64 ABI trampoline.
//! - `@naked` (zero prologue/epilogue) and `@interrupt` (full GPR save/restore + `iretq`).
//! - FastISel + Linear Scan Register Allocation over callee-saved GPRs `[rbx, r12, r13, r14]`.
//! - Inline `asm(...)` via built-in x86_64 mini-assembler (`x64_asm::encode_inline_asm`).
//! - Per-function `CompiledFnBlob` output for direct `.emi` caching & incremental rebuilds!

#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinaryOp, UnaryOp};
use crate::cfg::{AggregateKind, BlockId, CastKind, CfgBuiltinOp, CfgConst, CfgFnKey};
use crate::eir::{
    EirBlock, EirCallee, EirFunction, EirInst, EirInstKind, EirPlace, EirPlaceBase, EirProgram,
    EirProj, EirSlot, EirTerm, EirValue,
};
use crate::elf64::{
    ElfLinkInput, ElfMode, ElfReloc, ElfSymbolDef, LineEntry, LinkedElf, RelocKind, RelocSection,
    link_elf64,
};
use crate::resolve::{BuiltinKind, DefId, ResolvedProgram, TopDefKind};
use crate::span::{FileId, Interner, SourceMap};
use crate::types::{BorrowInner, FnSig, GenericVal, Ty, TyId, TypeEnv};
use crate::x64_asm::{
    Cond, FnReloc, Label, Reg, X64Buf, Xmm, emit_rt_helper_by_name, encode_inline_asm,
    rt_bss_layout, rt_helper_deps,
};

#[derive(Clone, Debug)]
pub struct X64Options {
    pub mode: ElfMode,
    pub sanitize: bool,
    pub needed_libs: Vec<String>,
}

impl Default for X64Options {
    fn default() -> Self {
        Self {
            mode: ElfMode::LinuxPie,
            sanitize: false,
            needed_libs: Vec::new(),
        }
    }
}

pub(crate) fn is_builtin_rt_helper(name: &str) -> bool {
    matches!(
        name,
        "eon_alloc_raw"
            | "eon_free_raw"
            | "eon_alloc_raw_san"
            | "eon_free_raw_san"
            | "eon_arena_alloc_aligned"
            | "eon_rt_arena_reset_san"
            | "eon_rt_install_san_handler"
            | "eon_rt_tls_init"
            | "eon_rt_parse_vdso"
            | "eon_rt_now_ns"
            | "eon_rt_openat2_beneath"
            | "eon_rt_read_fd"
            | "eon_rt_write_fd"
            | "eon_rt_close_fd"
            | "eon_rt_print_str"
            | "eon_rt_print_bytes"
            | "eon_rt_print_u64"
            | "eon_rt_print_i64"
            | "eon_rt_print_f32"
            | "eon_rt_print_nl"
            | "eon_rt_eprint_str"
            | "eon_rt_eprint_bytes"
            | "eon_rt_eprint_nl"
            | "eon_rt_validate_utf8"
            | "eon_rt_is_dir"
            | "eon_rt_mkdir_all"
            | "eon_rt_spawn_wait"
            | "eon_f16_to_f32"
            | "eon_f32_to_f16"
            | "eon_sched_init"
            | "eon_sched_shutdown"
            | "eon_par_for"
            | "eon_sched_run_dag"
            | "eon_fiber_switch"
            | "eon_iouring_create"
            | "eon_iouring_destroy"
            | "eon_run_fibers_iouring"
    )
}

pub const EON_GP_ARG_REGS: [Reg; 8] = [
    Reg::Rdi,
    Reg::Rsi,
    Reg::Rdx,
    Reg::Rcx,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
];

pub const SYSV_GP_ARG_REGS: [Reg; 6] = [
    Reg::Rdi,
    Reg::Rsi,
    Reg::Rdx,
    Reg::Rcx,
    Reg::R8,
    Reg::R9,
];

pub const FLOAT_ARG_REGS: [Xmm; 8] = [
    Xmm::Xmm0,
    Xmm::Xmm1,
    Xmm::Xmm2,
    Xmm::Xmm3,
    Xmm::Xmm4,
    Xmm::Xmm5,
    Xmm::Xmm6,
    Xmm::Xmm7,
];

const LINEAR_SCAN_REGS: [Reg; 4] = [Reg::Rbx, Reg::R12, Reg::R13, Reg::R14];

#[derive(Clone, Debug)]
pub struct RodataItem {
    pub sym: String,
    pub align: usize,
    pub bytes: Vec<u8>,
}

/// Self-contained compiled machine code for one EIR function (directly cacheable in `.emi`).
#[derive(Clone, Debug)]
pub struct CompiledFnBlob {
    pub sym_name: String,
    pub code: Vec<u8>,
    pub relocs: Vec<FnReloc>,
    pub rodata: Vec<RodataItem>,
    pub line_entries: Vec<LineEntry>,
    pub used_helpers: BTreeSet<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct X64Layout {
    pub size: usize,
    pub align: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ArgClass {
    Void,
    Float32,
    Float64,
    Reg1(usize),
    Reg2(usize),
    Indirect(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RetClass {
    Void,
    Float32,
    Float64,
    Reg1(usize),
    Reg2(usize),
    Sret(usize),
    ErrorUnion { ok_ty: TyId, ok_size: usize, ok_is_f32: bool, ok_is_f64: bool },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ValLoc {
    Reg(Reg),
    Stack(i32), // offset relative to RBP (always negative: [rbp + off])
}

pub fn fn_sym_name(key: CfgFnKey, eir: &EirProgram, interner: &Interner) -> String {
    if let Some(&idx) = eir.by_key.get(&key) {
        let f = &eir.functions[idx];
        if let Some(tag) = &f.layout_shared_tag {
            return format!("eon_fn_{}__{}", sanitize_sym(interner.get(f.name)), tag);
        }
        if f.sig.effects.is_export_c {
            return sanitize_sym(interner.get(f.name));
        }
        // Keep the source name in the symbol (same scheme as the C11 backend) so `gdb`/`perf`
        // backtraces show `eon_fn_6_arr_sum` instead of an opaque `eon_fn_6`.
        if let CfgFnKey::TopDef(d) = key {
            return format!("eon_fn_{}_{}", d.0, sanitize_sym(interner.get(f.name)));
        }
    }
    match key {
        CfgFnKey::TopDef(d) => format!("eon_fn_{}", d.0),
        CfgFnKey::ImplMethod { impl_idx, sub_idx } => format!("eon_m_{}_{}", impl_idx, sub_idx),
        CfgFnKey::GenericInst(id) => format!("eon_g_{}", id),
        CfgFnKey::Closure(id) => format!("eon_cl_{}", id),
    }
}

fn sanitize_sym(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn align_up(v: usize, a: usize) -> usize {
    if a <= 1 { v } else { (v + a - 1) & !(a - 1) }
}

pub fn subst_ty_id(env: &TypeEnv, ty: TyId, gargs: &[GenericVal]) -> TyId {
    match env.get(ty) {
        Ty::GenericParam { idx, .. } => match gargs.get(*idx as usize) {
            Some(GenericVal::Ty(ct)) => *ct,
            _ => ty,
        },
        Ty::Struct { def, args } if !args.is_empty() => {
            let sub: Vec<GenericVal> = args
                .iter()
                .map(|gv| match gv {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty_id(env, *t, gargs)),
                    other => *other,
                })
                .collect();
            env.find_interned(&Ty::Struct { def: *def, args: sub }).unwrap_or(ty)
        }
        Ty::RawPtr { is_mut, inner } => {
            let si = subst_ty_id(env, *inner, gargs);
            env.find_interned(&Ty::RawPtr { is_mut: *is_mut, inner: si }).unwrap_or(ty)
        }
        Ty::Borrow { is_mut, inner: BorrowInner::Single(t) } => {
            let si = subst_ty_id(env, *t, gargs);
            env.find_interned(&Ty::Borrow { is_mut: *is_mut, inner: BorrowInner::Single(si) }).unwrap_or(ty)
        }
        Ty::Builtin(bk, args) if !args.is_empty() => {
            let sub: Vec<GenericVal> = args
                .iter()
                .map(|gv| match gv {
                    GenericVal::Ty(t) => GenericVal::Ty(subst_ty_id(env, *t, gargs)),
                    other => *other,
                })
                .collect();
            env.find_interned(&Ty::Builtin(*bk, sub)).unwrap_or(ty)
        }
        _ => ty,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum OptNiche {
    None,
    IdBrand,
    GenId,
    Link,
    NullPtr,
    FatPtrNullFirst,
    SliceMaxLen,
    BoolTag,
    ScalarSentinel { size: usize, sentinel: u64 },
}

pub(crate) fn opt_niche(env: &TypeEnv, inner: TyId) -> OptNiche {
    match env.get(inner) {
        Ty::Builtin(BuiltinKind::IdBrand, _) => OptNiche::IdBrand,
        Ty::Builtin(BuiltinKind::GenId, _) => OptNiche::GenId,
        Ty::Builtin(BuiltinKind::Link, _) => OptNiche::Link,
        Ty::Bool => OptNiche::BoolTag,
        Ty::Char => OptNiche::ScalarSentinel {
            size: 4,
            sentinel: u32::MAX as u64,
        },
        Ty::ErrorSet(_) => OptNiche::ScalarSentinel {
            size: 4,
            sentinel: 0,
        },
        Ty::Enum(d) => {
            if let Some(meta) = env.enums.get(d)
                && meta.has_niche
            {
                let mut cand = u32::MAX;
                while meta.variants.iter().any(|(_, v)| (*v as u32) == cand) {
                    cand = cand.wrapping_sub(1);
                }
                OptNiche::ScalarSentinel {
                    size: 4,
                    sentinel: cand as u64,
                }
            } else {
                OptNiche::None
            }
        }
        Ty::Int { bits, is_ptr_sized, .. }
            if !is_ptr_sized && !matches!(*bits, 8 | 16 | 32 | 64 | 128) =>
        {
            let sz = match *bits {
                1..=8 => 1,
                9..=16 => 2,
                17..=32 => 4,
                _ => 8,
            };
            let sentinel = 1u64 << (sz * 8 - 1);
            OptNiche::ScalarSentinel { size: sz, sentinel }
        }
        Ty::Borrow { inner: bi, .. } => match bi {
            BorrowInner::Single(t) => match env.get(*t) {
                Ty::Pool { .. } | Ty::Dyn(_) => OptNiche::FatPtrNullFirst,
                _ => OptNiche::NullPtr,
            },
            BorrowInner::Slice(_) | BorrowInner::Str => OptNiche::SliceMaxLen,
        },
        Ty::RawPtr { .. } => OptNiche::NullPtr,
        Ty::FnPtr { .. } => OptNiche::FatPtrNullFirst,
        _ => OptNiche::None,
    }
}

pub fn is_niche_id_opt(env: &TypeEnv, inner: TyId) -> bool {
    opt_niche(env, inner) != OptNiche::None
}

pub fn is_link_or_opt_link(env: &TypeEnv, ty: TyId) -> bool {
    match env.get(ty) {
        Ty::Builtin(BuiltinKind::Link, _) => true,
        Ty::Optional(inner) => matches!(env.get(*inner), Ty::Builtin(BuiltinKind::Link, _)),
        _ => false,
    }
}

pub fn x64_layout(env: &TypeEnv, ty: TyId) -> X64Layout {
    x64_layout_impl(env, ty, false)
}

/// Encode a compile-time `CtVal` constant into little-endian x86_64 target bytes matching `x64_layout`.
pub(crate) fn encode_ct_val_bytes(val: &crate::comptime::CtVal, ty: TyId, env: &TypeEnv) -> Option<Vec<u8>> {
    use crate::comptime::CtVal;
    let sz = x64_layout(env, ty).size;
    if sz == 0 {
        return Some(Vec::new());
    }
    match val {
        CtVal::Void => Some(vec![0u8; sz]),
        CtVal::Bool(b) => {
            let mut out = vec![0u8; sz];
            out[0] = u8::from(*b);
            Some(out)
        }
        CtVal::Char(c) => {
            let mut out = vec![0u8; sz];
            let cb = (*c as u32).to_le_bytes();
            let n = sz.min(4);
            out[..n].copy_from_slice(&cb[..n]);
            Some(out)
        }
        CtVal::Int(n, _) => {
            let bs = (*n).to_le_bytes();
            let mut out = vec![0u8; sz];
            let k = sz.min(16);
            out[..k].copy_from_slice(&bs[..k]);
            Some(out)
        }
        CtVal::Float(f, _) => {
            let mut out = vec![0u8; sz];
            match sz {
                2 => out.copy_from_slice(&crate::eir_interp::f32_to_f16_bits(*f as f32).to_le_bytes()),
                4 => out.copy_from_slice(&(*f as f32).to_le_bytes()),
                _ => out[..sz.min(8)].copy_from_slice(&(*f).to_le_bytes()[..sz.min(8)]),
            }
            Some(out)
        }
        CtVal::EnumSym(_, v) => {
            let bs = (*v).to_le_bytes();
            let mut out = vec![0u8; sz];
            let k = sz.min(16);
            out[..k].copy_from_slice(&bs[..k]);
            Some(out)
        }
        CtVal::Array(elems, ct_elem_ty) => {
            let elem_ty = match env.get(ty) {
                Ty::Array { elem, .. } | Ty::Vec { elem, .. } => *elem,
                _ => *ct_elem_ty,
            };
            let esz = x64_layout(env, elem_ty).size.max(1);
            let mut out = vec![0u8; sz];
            for (i, ev) in elems.iter().enumerate() {
                let eb = encode_ct_val_bytes(ev, elem_ty, env)?;
                let start = i * esz;
                if start + eb.len() <= out.len() {
                    out[start..start + eb.len()].copy_from_slice(&eb);
                }
            }
            Some(out)
        }
        CtVal::Tuple(elems) => {
            let mut out = vec![0u8; sz];
            for (i, ev) in elems.iter().enumerate() {
                let (foff, fty) = struct_field_offset(env, ty, i);
                let eb = encode_ct_val_bytes(ev, fty, env)?;
                if foff + eb.len() <= out.len() {
                    out[foff..foff + eb.len()].copy_from_slice(&eb);
                }
            }
            Some(out)
        }
        CtVal::Struct { ty: sty, fields } => {
            let mut out = vec![0u8; sz];
            match env.get(*sty) {
                Ty::Struct { def, .. } => {
                    let meta = env.structs.get(def)?;
                    for (i, fm) in meta.fields.iter().enumerate() {
                        let (foff, fty) = struct_field_offset(env, *sty, i);
                        if let Some(fv) = fields.get(&fm.name) {
                            let eb = encode_ct_val_bytes(fv, fty, env)?;
                            if foff + eb.len() <= out.len() {
                                out[foff..foff + eb.len()].copy_from_slice(&eb);
                            }
                        }
                    }
                    Some(out)
                }
                Ty::AnonStruct(fs) => {
                    for (i, (fname, _)) in fs.iter().enumerate() {
                        let (foff, fty) = struct_field_offset(env, *sty, i);
                        if let Some(fv) = fields.get(fname) {
                            let eb = encode_ct_val_bytes(fv, fty, env)?;
                            if foff + eb.len() <= out.len() {
                                out[foff..foff + eb.len()].copy_from_slice(&eb);
                            }
                        }
                    }
                    Some(out)
                }
                _ => None,
            }
        }
        CtVal::Str(_) | CtVal::TypeVal(_) => None,
    }
}

pub fn x64_layout_impl(env: &TypeEnv, ty: TyId, link_is_slot: bool) -> X64Layout {
    match env.get(ty) {
        Ty::Void | Ty::Never => X64Layout { size: 0, align: 1 },
        Ty::Bool => X64Layout { size: 1, align: 1 },
        Ty::Char | Ty::Enum(_) | Ty::ErrorSet(_) => X64Layout { size: 4, align: 4 },
        Ty::Int { bits, is_ptr_sized, .. } => {
            let sz = if *is_ptr_sized {
                8
            } else {
                match *bits {
                    0..=8 => 1,
                    9..=16 => 2,
                    17..=32 => 4,
                    33..=64 => 8,
                    _ => 16,
                }
            };
            X64Layout { size: sz, align: sz.min(8) }
        }
        Ty::Float(bits) => {
            let sz = match *bits {
                16 => 2,
                32 => 4,
                _ => 8,
            };
            X64Layout { size: sz, align: sz }
        }
        Ty::PhysAddr | Ty::VirtAddr | Ty::DmaAddr | Ty::RawPtr { .. } => {
            X64Layout { size: 8, align: 8 }
        }
        Ty::FnPtr { .. } => X64Layout { size: 16, align: 8 },
        Ty::Str | Ty::Dyn(_) => X64Layout { size: 16, align: 8 },
        Ty::Borrow { inner, .. } => match inner {
            BorrowInner::Str | BorrowInner::Slice(_) => X64Layout { size: 16, align: 8 },
            BorrowInner::Single(t) => match env.get(*t) {
                Ty::Pool { .. } | Ty::Dyn(_) => X64Layout { size: 16, align: 8 },
                _ => X64Layout { size: 8, align: 8 },
            },
        },
        Ty::Optional(inner) => {
            if is_niche_id_opt(env, *inner) {
                x64_layout_impl(env, *inner, link_is_slot)
            } else {
                let il = x64_layout_impl(env, *inner, link_is_slot);
                let align = il.align.max(1);
                let val_off = align_up(1, align);
                let size = align_up(val_off + il.size.max(1), align);
                X64Layout { size, align }
            }
        }
        Ty::ErrorUnion { ok, .. } => {
            let ol = if *ok == TyId::VOID || *ok == TyId::NEVER {
                X64Layout { size: 1, align: 1 }
            } else {
                x64_layout_impl(env, *ok, link_is_slot)
            };
            let align = ol.align.max(4);
            let ok_off = align_up(4, ol.align.max(1));
            let size = align_up(ok_off + ol.size.max(1), align);
            X64Layout { size, align }
        }
        Ty::Array { elem, len } => {
            let el = x64_layout(env, *elem);
            let count = (*len as usize).max(1);
            X64Layout {
                size: el.size.max(1) * count,
                align: el.align.max(1),
            }
        }
        Ty::Tuple(elems) => {
            if elems.is_empty() {
                return X64Layout { size: 1, align: 1 };
            }
            let mut off = 0usize;
            let mut max_a = 1usize;
            for &e in elems {
                let l = x64_layout(env, e);
                let a = l.align.max(1);
                max_a = max_a.max(a);
                off = align_up(off, a) + l.size.max(1);
            }
            X64Layout { size: align_up(off, max_a), align: max_a }
        }
        Ty::AnonStruct(fs) => {
            if fs.is_empty() {
                return X64Layout { size: 1, align: 1 };
            }
            let mut off = 0usize;
            let mut max_a = 1usize;
            for &(_, ft) in fs {
                let l = x64_layout(env, ft);
                let a = l.align.max(1);
                max_a = max_a.max(a);
                off = align_up(off, a) + l.size.max(1);
            }
            X64Layout { size: align_up(off, max_a), align: max_a }
        }
        Ty::Struct { def, args } => {
            let Some(m) = env.structs.get(def) else {
                return X64Layout { size: 1, align: 1 };
            };
            if m.fields.is_empty() {
                return X64Layout { size: 1, align: 1 };
            }
            let mut off = 0usize;
            let mut max_a = 1usize;
            for f in &m.fields {
                let fty = if args.is_empty() { f.ty } else { subst_ty_id(env, f.ty, args) };
                let l = x64_layout(env, fty);
                let a = if m.is_packed { 1 } else { l.align.max(1) };
                max_a = max_a.max(a);
                off = align_up(off, a) + l.size.max(1);
            }
            X64Layout { size: align_up(off, max_a), align: max_a }
        }
        Ty::Pool { def, .. } => {
            let Some(m) = env.pools.get(def) else {
                return X64Layout { size: 1, align: 1 };
            };
            if m.columns.is_empty() {
                return X64Layout { size: 1, align: 1 };
            }
            let mut off = 0usize;
            let mut max_a = 1usize;
            for c in &m.columns {
                // Row struct stores `Link` as 8B `EonGenId` (matches codegen_c.rs).
                let l = x64_layout_impl(env, c.ty, true);
                let a = l.align.max(1);
                max_a = max_a.max(a);
                off = align_up(off, a) + l.size.max(1);
            }
            X64Layout { size: align_up(off, max_a), align: max_a }
        }
        Ty::Union { def, .. } => {
            let Some(m) = env.unions.get(def) else {
                return X64Layout { size: 4, align: 4 };
            };
            let mut max_psz = 1usize;
            let mut max_pa = 1usize;
            for v in &m.variants {
                if let Some(pt) = v.payload {
                    let l = x64_layout(env, pt);
                    max_psz = max_psz.max(l.size.max(1));
                    max_pa = max_pa.max(l.align.max(1));
                }
            }
            let align = max_pa.max(4);
            let p_off = align_up(4, max_pa);
            X64Layout { size: align_up(p_off + max_psz, align), align }
        }
        Ty::Bitfield(d) => {
            let bty = env.bitfields.get(d).map_or(TyId::U32, |m| m.base_ty);
            x64_layout(env, bty)
        }
        Ty::Vec { elem, len } => {
            let el = x64_layout(env, *elem);
            X64Layout { size: el.size.max(1) * (*len as usize).max(1), align: el.align.max(4) }
        }
        Ty::Builtin(bk, args) => match bk {
            BuiltinKind::Arena | BuiltinKind::Buf | BuiltinKind::LinearBuf => {
                X64Layout { size: 24, align: 8 }
            }
            BuiltinKind::AllocCap => X64Layout { size: 40, align: 8 },
            BuiltinKind::Env => X64Layout { size: 88, align: 8 },
            BuiltinKind::ArgsCap | BuiltinKind::BuildCap => X64Layout { size: 16, align: 8 },
            BuiltinKind::FsCap
            | BuiltinKind::DirCap
            | BuiltinKind::NetCap
            | BuiltinKind::ClockCap
            | BuiltinKind::IoCap
            | BuiltinKind::Schedule => X64Layout { size: 8, align: 8 },
            BuiltinKind::ArenaStr | BuiltinKind::GpuBuf | BuiltinKind::GpuImage2D => {
                X64Layout { size: 16, align: 8 }
            }
            BuiltinKind::Ordering | BuiltinKind::IdBrand => X64Layout { size: 4, align: 4 },
            BuiltinKind::GenId => X64Layout { size: 8, align: 4 },
            BuiltinKind::Link => if link_is_slot {
                X64Layout { size: 8, align: 4 }
            } else {
                X64Layout { size: 4, align: 4 }
            },
            BuiltinKind::ArenaPool => {
                let ncols = pool_col_count(env, args);
                X64Layout { size: (ncols + 2) * 8, align: 8 }
            }
            BuiltinKind::SlotPool => {
                let ncols = pool_col_count(env, args);
                X64Layout { size: (ncols + 5) * 8, align: 8 }
            }
            BuiltinKind::Atomic | BuiltinKind::MmioReg => {
                let el = match args.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => TyId::U64,
                };
                x64_layout(env, el)
            }
            _ => X64Layout { size: 8, align: 8 },
        },
        _ => X64Layout { size: 8, align: 8 },
    }
}

pub(crate) fn pool_def_of_gargs(env: &TypeEnv, args: &[GenericVal]) -> Option<DefId> {
    if let Some(GenericVal::Ty(pt)) = args.first()
        && let Ty::Pool { def, .. } = env.get(*pt)
    {
        return Some(*def);
    }
    None
}

pub(crate) fn pool_col_count(env: &TypeEnv, args: &[GenericVal]) -> usize {
    pool_def_of_gargs(env, args)
        .and_then(|d| env.pools.get(&d))
        .map_or(0, |m| m.columns.len())
}

pub(crate) fn struct_field_offset(env: &TypeEnv, ty: TyId, field_idx: usize) -> (usize, TyId) {
    match env.get(ty) {
        Ty::Struct { def, args } => {
            let Some(m) = env.structs.get(def) else { return (0, TyId::U64); };
            let mut off = 0usize;
            for (i, f) in m.fields.iter().enumerate() {
                let fty = if args.is_empty() { f.ty } else { subst_ty_id(env, f.ty, args) };
                let l = x64_layout(env, fty);
                let a = if m.is_packed { 1 } else { l.align.max(1) };
                off = align_up(off, a);
                if i == field_idx {
                    return (off, fty);
                }
                off += l.size.max(1);
            }
            (0, TyId::U64)
        }
        Ty::Pool { def, .. } => {
            let Some(m) = env.pools.get(def) else { return (0, TyId::U64); };
            let mut off = 0usize;
            for (i, c) in m.columns.iter().enumerate() {
                let l = x64_layout_impl(env, c.ty, true);
                off = align_up(off, l.align.max(1));
                if i == field_idx {
                    return (off, c.ty);
                }
                off += l.size.max(1);
            }
            (0, TyId::U64)
        }
        Ty::AnonStruct(fs) => {
            let mut off = 0usize;
            for (i, &(_, ft)) in fs.iter().enumerate() {
                let l = x64_layout(env, ft);
                off = align_up(off, l.align.max(1));
                if i == field_idx {
                    return (off, ft);
                }
                off += l.size.max(1);
            }
            (0, TyId::U64)
        }
        Ty::Tuple(es) => {
            let mut off = 0usize;
            for (i, &et) in es.iter().enumerate() {
                let l = x64_layout(env, et);
                off = align_up(off, l.align.max(1));
                if i == field_idx {
                    return (off, et);
                }
                off += l.size.max(1);
            }
            (0, TyId::U64)
        }
        Ty::Vec { elem, .. } => {
            let el = x64_layout(env, *elem);
            (field_idx * el.size.max(1), *elem)
        }
        Ty::Builtin(BuiltinKind::Env, _) => {
            let (off, bk) = match field_idx {
                0 => (0usize, BuiltinKind::AllocCap),
                1 => (40usize, BuiltinKind::FsCap),
                2 => (48usize, BuiltinKind::NetCap),
                3 => (56usize, BuiltinKind::ClockCap),
                4 => (64usize, BuiltinKind::IoCap),
                5 => (72usize, BuiltinKind::ArgsCap),
                _ => (0usize, BuiltinKind::AllocCap),
            };
            let ty = env
                .find_interned(&Ty::Builtin(bk, Vec::new()))
                .unwrap_or(TyId::U64);
            (off, ty)
        }
        _ => (0, TyId::U64),
    }
}

pub(crate) fn opt_val_offset(env: &TypeEnv, inner: TyId) -> usize {
    let il = x64_layout(env, inner);
    align_up(1, il.align.max(1))
}

pub(crate) fn eu_ok_offset(env: &TypeEnv, ok: TyId) -> usize {
    if ok == TyId::VOID || ok == TyId::NEVER {
        4
    } else {
        let ol = x64_layout(env, ok);
        align_up(4, ol.align.max(1))
    }
}

pub(crate) fn union_payload_offset(env: &TypeEnv, def: DefId) -> usize {
    let Some(m) = env.unions.get(&def) else { return 4; };
    let mut max_a = 1usize;
    for v in &m.variants {
        if let Some(pt) = v.payload {
            max_a = max_a.max(x64_layout(env, pt).align.max(1));
        }
    }
    align_up(4, max_a)
}

pub(crate) fn is_signed_int(env: &TypeEnv, ty: TyId) -> bool {
    matches!(env.get(ty), Ty::Int { signed: true, .. })
}

pub(crate) fn int_bits(env: &TypeEnv, ty: TyId) -> u8 {
    match env.get(ty) {
        Ty::Int { bits, is_ptr_sized, .. } => if *is_ptr_sized { 64 } else { *bits },
        Ty::Bool => 8,
        Ty::Char | Ty::Enum(_) | Ty::ErrorSet(_) => 32,
        Ty::Bitfield(d) => {
            let bty = env.bitfields.get(d).map_or(TyId::U32, |m| m.base_ty);
            int_bits(env, bty)
        }
        _ => 64,
    }
}

fn classify_arg(env: &TypeEnv, ty: TyId) -> ArgClass {
    match env.get(ty) {
        Ty::Void | Ty::Never => ArgClass::Void,
        Ty::Float(16) => ArgClass::Reg1(2),
        Ty::Float(32) => ArgClass::Float32,
        Ty::Float(_) => ArgClass::Float64,
        _ => {
            let sz = x64_layout(env, ty).size;
            if sz == 0 {
                ArgClass::Void
            } else if sz <= 8 {
                ArgClass::Reg1(sz)
            } else if sz <= 16 {
                ArgClass::Reg2(sz)
            } else {
                ArgClass::Indirect(sz)
            }
        }
    }
}

fn classify_ret(env: &TypeEnv, ty: TyId, is_extern_c: bool) -> RetClass {
    match env.get(ty) {
        Ty::Void | Ty::Never => RetClass::Void,
        Ty::Float(16) => RetClass::Reg1(2),
        Ty::Float(32) => RetClass::Float32,
        Ty::Float(_) => RetClass::Float64,
        Ty::ErrorUnion { ok, .. } if !is_extern_c => {
            let ok_is_f32 = matches!(env.get(*ok), Ty::Float(32));
            let ok_is_f64 = matches!(env.get(*ok), Ty::Float(64));
            let ok_size = if *ok == TyId::VOID || *ok == TyId::NEVER {
                0
            } else {
                x64_layout(env, *ok).size
            };
            RetClass::ErrorUnion {
                ok_ty: *ok,
                ok_size,
                ok_is_f32,
                ok_is_f64,
            }
        }
        _ => {
            let sz = x64_layout(env, ty).size;
            if sz == 0 {
                RetClass::Void
            } else if sz <= 8 {
                RetClass::Reg1(sz)
            } else if sz <= 16 {
                RetClass::Reg2(sz)
            } else {
                RetClass::Sret(sz)
            }
        }
    }
}

fn ret_uses_sret(rc: RetClass) -> bool {
    match rc {
        RetClass::Sret(_) => true,
        RetClass::ErrorUnion { ok_size, .. } => ok_size > 16,
        _ => false,
    }
}

// ============================================================================
// Linear Scan Register Allocator + Stack Frame Layout
// ============================================================================

struct FnFramePlan {
    val_locs: Vec<ValLoc>,
    slot_offs: Vec<i32>, // [rbp + off] (negative)
    used_callee_regs: Vec<Reg>,
    sret_off: Option<i32>,
    temp_copy_off: i32,
    frame_size: i32, // bytes to subtract from rsp after pushing `used_callee_regs`
    total_aligned: usize,
}

fn build_frame_plan(f: &EirFunction, env: &TypeEnv) -> FnFramePlan {
    let nvals = f.value_tys.len();
    let mut start = vec![usize::MAX; nvals];
    let mut end = vec![0usize; nvals];
    let mut eligible = vec![false; nvals];
    let allow_regalloc = f.par_for_regions.is_empty();

    for (vi, &ty) in f.value_tys.iter().enumerate() {
        let lay = x64_layout(env, ty);
        let is_float = matches!(env.get(ty), Ty::Float(_));
        let is_agg = matches!(
            env.get(ty),
            Ty::Struct { .. }
                | Ty::AnonStruct(_)
                | Ty::Tuple(_)
                | Ty::Array { .. }
                | Ty::Union { .. }
                | Ty::Pool { .. }
                | Ty::ErrorUnion { .. }
                | Ty::Optional(_)
                | Ty::Vec { .. }
                | Ty::Builtin(BuiltinKind::GenId | BuiltinKind::Link, _)
        );
        if allow_regalloc && lay.size >= 1 && lay.size <= 8 && !is_float && !is_agg {
            eligible[vi] = true;
        }
    }

    // Walk RPO blocks to compute instruction-indexed live intervals [start, end].
    let mut bb_start_pos: BTreeMap<BlockId, usize> = BTreeMap::new();
    let mut bb_end_pos: BTreeMap<BlockId, usize> = BTreeMap::new();
    let mut pos = 1usize;

    for &p in &f.params {
        start[p.0 as usize] = 0;
        end[p.0 as usize] = 1;
    }

    for &bb in &f.rpo {
        bb_start_pos.insert(bb, pos);
        let blk = f.block(bb);
        for phi in &blk.phis {
            let vi = phi.dst.0 as usize;
            start[vi] = start[vi].min(pos);
            end[vi] = end[vi].max(pos);
            for &(_, iv) in &phi.incoming {
                let ii = iv.0 as usize;
                start[ii] = start[ii].min(pos);
                end[ii] = end[ii].max(pos);
            }
            pos += 1;
        }
        for inst in &blk.insts {
            if let Some(d) = eir_inst_dst(&inst.kind) {
                let vi = d.0 as usize;
                start[vi] = start[vi].min(pos);
                end[vi] = end[vi].max(pos);
            }
            for u in eir_inst_uses(&inst.kind) {
                let vi = u.0 as usize;
                start[vi] = start[vi].min(pos);
                end[vi] = end[vi].max(pos);
            }
            pos += 1;
        }
        for u in eir_term_uses(&blk.term) {
            let vi = u.0 as usize;
            start[vi] = start[vi].min(pos);
            end[vi] = end[vi].max(pos);
        }
        bb_end_pos.insert(bb, pos);
        pos += 1;
    }

    // Extend intervals across loop backedges so values live at loop header stay live through latch!
    let mut rpo_idx: BTreeMap<BlockId, usize> = BTreeMap::new();
    for (i, &bb) in f.rpo.iter().enumerate() {
        rpo_idx.insert(bb, i);
    }
    for &bb in &f.rpo {
        let blk = f.block(bb);
        let b_rpo = rpo_idx[&bb];
        let b_end = bb_end_pos[&bb];
        for succ in term_succs(&blk.term) {
            if let Some(&s_rpo) = rpo_idx.get(&succ)
                && s_rpo <= b_rpo
            {
                let h_start = bb_start_pos[&succ];
                for vi in 0..nvals {
                    if start[vi] <= h_start && end[vi] >= h_start && end[vi] < b_end {
                        end[vi] = b_end;
                    }
                }
            }
        }
        // Also ensure phi incoming values are live through the end of their predecessor block.
        for succ in term_succs(&blk.term) {
            for phi in &f.block(succ).phis {
                for &(pbb, iv) in &phi.incoming {
                    if pbb == bb {
                        end[iv.0 as usize] = end[iv.0 as usize].max(b_end);
                    }
                }
            }
        }
    }

    // Run deterministic Linear Scan over `[RBX, R12, R13, R14]`.
    let mut intervals: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, val_idx)
    for vi in 0..nvals {
        if eligible[vi] && start[vi] != usize::MAX {
            intervals.push((start[vi], end[vi], vi));
        }
    }
    intervals.sort_by_key(|&(s, e, vi)| (s, e, vi));

    let mut assigned_reg: Vec<Option<Reg>> = vec![None; nvals];
    let mut active: Vec<(usize, usize, Reg)> = Vec::new(); // (end, val_idx, reg)
    let mut free_regs: Vec<Reg> = LINEAR_SCAN_REGS.iter().rev().copied().collect();
    let mut used_set: BTreeSet<Reg> = BTreeSet::new();

    for (iv_s, iv_e, vi) in intervals {
        // Expire old intervals.
        active.retain(|&(ae, _, r)| {
            if ae < iv_s {
                free_regs.push(r);
                false
            } else {
                true
            }
        });
        free_regs.sort_by_key(|r| LINEAR_SCAN_REGS.iter().position(|x| x == r).unwrap_or(99));
        if let Some(reg) = free_regs.first().copied() {
            free_regs.remove(0);
            assigned_reg[vi] = Some(reg);
            used_set.insert(reg);
            active.push((iv_e, vi, reg));
        } else {
            // Spill the interval with the furthest end point.
            active.sort_by_key(|&(ae, _, _)| ae);
            if let Some(&(last_e, last_vi, last_reg)) = active.last()
                && last_e > iv_e
            {
                active.pop();
                assigned_reg[last_vi] = None;
                assigned_reg[vi] = Some(last_reg);
                active.push((iv_e, vi, last_reg));
            }
        }
    }

    let used_callee_regs: Vec<Reg> = LINEAR_SCAN_REGS
        .iter()
        .copied()
        .filter(|r| used_set.contains(r))
        .collect();

    // Assign stack offsets below pushed callee-saved regs `[rbp - k*8]`.
    let saved_bytes = used_callee_regs.len() * 8;
    let mut cur_depth = saved_bytes;

    let mut alloc_stack = |size: usize, align: usize| -> i32 {
        let sz = size.max(8);
        let al = align.max(8);
        cur_depth = align_up(cur_depth + sz, al);
        -(cur_depth as i32)
    };

    let ret_cls = classify_ret(env, f.sig.ret, f.sig.effects.is_export_c);
    let sret_off = if ret_uses_sret(ret_cls) {
        Some(alloc_stack(8, 8))
    } else {
        None
    };

    let mut slot_offs = Vec::with_capacity(f.slots.len());
    for sl in &f.slots {
        let lay = x64_layout(env, sl.ty);
        slot_offs.push(alloc_stack(lay.size.max(8), lay.align.max(8)));
    }

    let mut val_locs = Vec::with_capacity(nvals);
    for (vi, &ty) in f.value_tys.iter().enumerate() {
        if let Some(r) = assigned_reg[vi] {
            val_locs.push(ValLoc::Reg(r));
        } else {
            let lay = x64_layout(env, ty);
            if lay.size == 0 {
                val_locs.push(ValLoc::Stack(0));
            } else {
                val_locs.push(ValLoc::Stack(alloc_stack(lay.size.max(8), lay.align.max(8))));
            }
        }
    }

    // Reserve a scratch buffer for indirect call args, parallel phi copies, and large struct/pool ops.
    let mut max_temp = 128usize;
    for b in &f.blocks {
        let phi_bytes: usize = b.phis.iter().map(|p| align_up(x64_layout(env, p.ty).size.max(8), 8)).sum();
        max_temp = max_temp.max(phi_bytes);
        for inst in &b.insts {
            match &inst.kind {
                EirInstKind::Call { args, .. } => {
                    let mut call_bytes = 0usize;
                    for &a in args {
                        let sz = x64_layout(env, f.value_tys[a.0 as usize]).size;
                        if sz > 16 {
                            call_bytes += align_up(sz, 16);
                        }
                    }
                    max_temp = max_temp.max(call_bytes);
                }
                EirInstKind::BuiltinCall { op: CfgBuiltinOp::ScheduleRun, args, .. } => {
                    let mut call_bytes = 0usize;
                    for &a in args.iter().skip(1) {
                        let ty = f.value_tys[a.0 as usize];
                        let sz = x64_layout(env, ty).size;
                        if sz > 16 {
                            call_bytes += align_up(sz, 16);
                        } else if let Ty::Borrow { inner: BorrowInner::Single(inner), .. } = env.get(ty) {
                            let isz = x64_layout(env, *inner).size;
                            if isz > 16 {
                                call_bytes += align_up(isz, 16);
                            }
                        }
                    }
                    max_temp = max_temp.max(call_bytes);
                }
                _ => {}
            }
        }
    }
    let temp_copy_off = alloc_stack(max_temp, 16);

    // Ensure total stack depth `(saved_bytes + frame_size)` is 16-byte aligned!
    let total_aligned = align_up(cur_depth, 16);
    let frame_size = (total_aligned - saved_bytes) as i32;

    FnFramePlan {
        val_locs,
        slot_offs,
        used_callee_regs,
        sret_off,
        temp_copy_off,
        frame_size,
        total_aligned,
    }
}

// ============================================================================
// Per-Function EIR -> x86_64 Compiler
// ============================================================================

#[derive(Clone, Debug)]
struct ParForOutlinePlan {
    header_bb: BlockId,
    step_bb: BlockId,
    exit_bb: BlockId,
    idx_slot: EirSlot,
    end_slot: EirSlot,
    chunk_size: u64,
    body_bbs: Vec<BlockId>,
    inner_val_defs: BTreeSet<EirValue>,
    cap_slots: BTreeSet<EirSlot>,
    worker_lbl: Label,
}

#[derive(Clone, Debug)]
struct ActiveWorker {
    header_bb: BlockId,
    step_bb: BlockId,
    exit_bb: BlockId,
    inner_val_defs: BTreeSet<EirValue>,
    cap_slots: BTreeSet<EirSlot>,
    worker_step_lbl: Label,
    worker_done_lbl: Label,
}

#[derive(Clone, Debug)]
enum SchedArgPass {
    /// Load value from `(bundle + off)` as scalar into GP/XMM arg reg (`bytes` wide).
    BundleScalar { off: usize, bytes: usize },
    /// Load 16B scalar pair from `(bundle + off .. off+16)` into two GP arg regs.
    BundlePair { off: usize },
    /// Load `ptr` = `[bundle + off]` and copy `bytes` from `[ptr]` into wrapper temp stack if >16B, or pass `ptr`.
    DerefIndirect { off: usize, bytes: usize },
    /// Deref `[bundle + off]` scalar (`bytes` wide).
    DerefScalar { off: usize, bytes: usize },
    /// Pass pointer `(bundle + off)` directly (for `&val` or Indirect arg stored in bundle).
    BundlePtr { off: usize },
}

#[derive(Clone, Debug)]
struct DeferredSchedWrapper {
    lbl: Label,
    sys_sym: String,
    params: Vec<(ArgClass, SchedArgPass)>,
}

pub fn compile_function_x64(
    f: &EirFunction,
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    sm: &SourceMap,
    interner: &Interner,
    file_index_map: &BTreeMap<FileId, u32>,
    sanitize: bool,
    is_pie_or_shared: bool,
) -> Result<CompiledFnBlob, String> {
    let sym_name = fn_sym_name(f.key, eir, interner);
    let plan = build_frame_plan(f, env);
    let mut cg = FnCodegen {
        f,
        eir,
        env,
        sm,
        interner,
        file_index_map,
        sym_name: sym_name.clone(),
        plan,
        buf: X64Buf::new(),
        rodata: Vec::new(),
        rodata_counter: 0,
        bb_labels: BTreeMap::new(),
        line_entries: Vec::new(),
        last_line: (u32::MAX, u32::MAX),
        used_helpers: BTreeSet::new(),
        sanitize,
        is_pie_or_shared,
        par_for_plans: Vec::new(),
        outlined_skip_bbs: BTreeSet::new(),
        active_worker: None,
        deferred_sched_wrappers: Vec::new(),
    };
    let _ = prog;
    cg.emit_function()?;
    let (code, relocs) = cg.buf.finish()?;
    Ok(CompiledFnBlob {
        sym_name,
        code,
        relocs,
        rodata: cg.rodata,
        line_entries: cg.line_entries,
        used_helpers: cg.used_helpers,
    })
}

struct FnCodegen<'a> {
    f: &'a EirFunction,
    eir: &'a EirProgram,
    env: &'a TypeEnv,
    sm: &'a SourceMap,
    interner: &'a Interner,
    file_index_map: &'a BTreeMap<FileId, u32>,
    sym_name: String,
    plan: FnFramePlan,
    buf: X64Buf,
    rodata: Vec<RodataItem>,
    rodata_counter: usize,
    bb_labels: BTreeMap<BlockId, Label>,
    line_entries: Vec<LineEntry>,
    last_line: (u32, u32),
    used_helpers: BTreeSet<String>,
    sanitize: bool,
    is_pie_or_shared: bool,
    par_for_plans: Vec<ParForOutlinePlan>,
    outlined_skip_bbs: BTreeSet<BlockId>,
    active_worker: Option<ActiveWorker>,
    deferred_sched_wrappers: Vec<DeferredSchedWrapper>,
}

impl<'a> FnCodegen<'a> {
    fn record_line(&mut self, span: crate::span::Span) {
        if span.file == FileId::DUMMY {
            return;
        }
        let Some(&fidx) = self.file_index_map.get(&span.file) else { return; };
        let file = self.sm.file(span.file);
        let (line, col) = file.line_col(span.start);
        if self.last_line == (fidx, line) {
            return;
        }
        self.last_line = (fidx, line);
        self.line_entries.push(LineEntry {
            code_offset: self.buf.pos() as u32,
            file_idx: fidx,
            line,
            col,
        });
    }

    fn helper_call(&mut self, helper: &str) {
        let actual = match helper {
            "eon_alloc_raw" if self.sanitize => "eon_alloc_raw_san",
            "eon_free_raw" if self.sanitize => "eon_free_raw_san",
            other => other,
        };
        self.used_helpers.insert(actual.to_owned());
        self.buf.call_sym(actual);
    }

    fn intern_rodata(&mut self, bytes: Vec<u8>, align: usize) -> String {
        let idx = self.rodata_counter;
        self.rodata_counter += 1;
        let sym = format!("__ro_{}_{}", self.sym_name, idx);
        self.rodata.push(RodataItem {
            sym: sym.clone(),
            align,
            bytes,
        });
        sym
    }

    fn emit_align_check(&mut self, addr_reg: Reg, align: usize) {
        if !self.sanitize || align <= 1 {
            return;
        }
        let l_ok = self.buf.new_label();
        self.buf.mov_ri64(Reg::Rax, (align - 1) as u64);
        self.buf.test_rr(addr_reg, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_ok);
        self.buf.ud2();
        self.buf.bind_label(l_ok);
    }

    /// Return `(base_reg, offset)` for reading/writing `v`. Inside an outlined `par for`
    /// worker, captured outer values live on parent frame `R14`, while inner values live on `Rbp`.
    fn val_mem(&self, v: EirValue) -> (Reg, i32) {
        let off = self.val_stack_off(v);
        let base = match &self.active_worker {
            Some(w) if !w.inner_val_defs.contains(&v) => Reg::R14,
            _ => Reg::Rbp,
        };
        (base, off)
    }

    fn slot_mem(&self, s: EirSlot) -> (Reg, i32) {
        let off = self.plan.slot_offs[s.0 as usize];
        let base = match &self.active_worker {
            Some(w) if w.cap_slots.contains(&s) => Reg::R14,
            _ => Reg::Rbp,
        };
        (base, off)
    }

    fn build_par_for_plans(&mut self) {
        if self.f.par_for_regions.is_empty()
            || self.f.sig.effects.is_naked
            || self.f.sig.effects.is_interrupt
        {
            return;
        }
        let mut claimed = BTreeSet::new();
        for pf in &self.f.par_for_regions {
            if claimed.contains(&pf.header_bb) {
                continue;
            }
            let mut body_bbs = Vec::new();
            let mut body_set = BTreeSet::new();
            let mut queue = std::collections::VecDeque::new();
            body_set.insert(pf.body_bb);
            queue.push_back(pf.body_bb);
            let mut can_outline = true;
            while let Some(cur) = queue.pop_front() {
                body_bbs.push(cur);
                let blk = self.f.block(cur);
                if !blk.phis.is_empty() {
                    can_outline = false;
                    break;
                }
                match &blk.term {
                    EirTerm::Return(_) => {
                        can_outline = false;
                        break;
                    }
                    EirTerm::Goto(t) => {
                        if *t != pf.step_bb
                            && *t != pf.header_bb
                            && *t != pf.exit_bb
                            && body_set.insert(*t)
                        {
                            queue.push_back(*t);
                        }
                    }
                    EirTerm::Branch { then_bb, else_bb, .. } => {
                        for &t in &[then_bb, else_bb] {
                            if *t != pf.step_bb
                                && *t != pf.header_bb
                                && *t != pf.exit_bb
                                && body_set.insert(*t)
                            {
                                queue.push_back(*t);
                            }
                        }
                    }
                    EirTerm::SwitchInt { targets, otherwise, .. } => {
                        for &(_, t) in targets {
                            if t != pf.step_bb
                                && t != pf.header_bb
                                && t != pf.exit_bb
                                && body_set.insert(t)
                            {
                                queue.push_back(t);
                            }
                        }
                        if *otherwise != pf.step_bb
                            && *otherwise != pf.header_bb
                            && *otherwise != pf.exit_bb
                            && body_set.insert(*otherwise)
                        {
                            queue.push_back(*otherwise);
                        }
                    }
                    EirTerm::Unreachable { .. } => {}
                }
            }
            if !can_outline {
                continue;
            }
            if body_set.iter().any(|b| claimed.contains(b)) || claimed.contains(&pf.step_bb) {
                continue;
            }
            // Don't outline an outer loop if an un-outlined inner par-for shares nothing weird;
            // actually inner par-for headers inside body_set will simply run sequentially inside worker
            // if they appear in `claimed`!
            let mut inner_val_defs = BTreeSet::new();
            let mut inner_slots_used = BTreeSet::new();
            let mut outer_slots_used = BTreeSet::new();
            for (bi, blk) in self.f.blocks.iter().enumerate() {
                let b_id = BlockId(bi as u32);
                let is_inner = body_set.contains(&b_id);
                for inst in &blk.insts {
                    let mut slots_here = Vec::new();
                    eir_inst_slots(&inst.kind, &mut slots_here);
                    if is_inner {
                        if let Some(d) = eir_inst_dst(&inst.kind) {
                            inner_val_defs.insert(d);
                        }
                        for s in slots_here {
                            inner_slots_used.insert(s);
                        }
                    } else if b_id != pf.header_bb && b_id != pf.step_bb {
                        for s in slots_here {
                            outer_slots_used.insert(s);
                        }
                    }
                }
            }
            let cap_slots: BTreeSet<EirSlot> = inner_slots_used
                .iter()
                .copied()
                .filter(|&s| {
                    s != pf.idx_slot
                        && s != pf.end_slot
                        && outer_slots_used.contains(&s)
                        && self.f.slots[s.0 as usize].ty != TyId::VOID
                })
                .collect();

            for &bb in &body_bbs {
                claimed.insert(bb);
                self.outlined_skip_bbs.insert(bb);
            }
            claimed.insert(pf.header_bb);
            claimed.insert(pf.step_bb);
            self.outlined_skip_bbs.insert(pf.step_bb);

            let worker_lbl = self.buf.new_label();
            self.par_for_plans.push(ParForOutlinePlan {
                header_bb: pf.header_bb,
                step_bb: pf.step_bb,
                exit_bb: pf.exit_bb,
                idx_slot: pf.idx_slot,
                end_slot: pf.end_slot,
                chunk_size: pf.chunk_size,
                body_bbs,
                inner_val_defs,
                cap_slots,
                worker_lbl,
            });
        }
    }

    fn emit_par_for_launch(&mut self, plan: &ParForOutlinePlan) {
        let idx_off = self.plan.slot_offs[plan.idx_slot.0 as usize];
        let end_off = self.plan.slot_offs[plan.end_slot.0 as usize];
        let idx_ty = self.f.slots[plan.idx_slot.0 as usize].ty;
        let idx_sz = x64_layout(self.env, idx_ty).size.clamp(1, 8);
        let idx_signed = is_signed_int(self.env, idx_ty);
        // Compute trip count n = max(0, end - start)
        self.buf.load_mem(Reg::Rax, Reg::Rbp, idx_off, idx_sz, idx_signed);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, end_off, idx_sz, idx_signed);
        let l_done = self.buf.new_label();
        self.buf.cmp_rr(Reg::Rdi, Reg::Rax);
        self.buf.jcc_label(if idx_signed { Cond::Le } else { Cond::Be }, l_done);
        self.buf.sub_rr(Reg::Rdi, Reg::Rax); // rdi = n
        self.buf.mov_ri64(Reg::Rsi, plan.chunk_size.max(1));
        self.buf.lea_rip_label(Reg::Rdx, plan.worker_lbl);
        self.buf.mov_rr(Reg::Rcx, Reg::Rbp);
        self.helper_call("eon_par_for");
        // Set idx_slot = end for any post-loop inspection.
        self.buf.load_mem(Reg::Rax, Reg::Rbp, end_off, idx_sz, false);
        self.buf.store_mem(Reg::Rbp, idx_off, Reg::Rax, idx_sz);
        self.buf.bind_label(l_done);
        self.emit_phi_edge(plan.header_bb, plan.exit_bb);
        self.buf.jmp_label(self.bb_labels[&plan.exit_bb]);
    }

    fn emit_par_for_worker(&mut self, plan: &ParForOutlinePlan) -> Result<(), String> {
        while !self.buf.pos().is_multiple_of(16) {
            self.buf.nop();
        }
        self.buf.bind_label(plan.worker_lbl);
        // System V ABI: rdi = _start, rsi = _end, rdx = ctx (parent RBP)
        // Push [rbx, r12, r13, r14] (32B) BEFORE `push rbp` so worker `[rbp + off]`
        // matches `plan.slot_offs`/`plan.val_locs` identically and stack stays 16B-aligned!
        self.buf.push(Reg::Rbx);
        self.buf.push(Reg::R12);
        self.buf.push(Reg::R13);
        self.buf.push(Reg::R14);
        self.buf.push(Reg::Rbp);
        self.buf.mov_rr(Reg::Rbp, Reg::Rsp);
        let frame_alloc = align_up(self.plan.total_aligned, 16);
        if frame_alloc > 0 {
            self.buf.sub_ri32(Reg::Rsp, frame_alloc as i32);
        }

        let idx_off = self.plan.slot_offs[plan.idx_slot.0 as usize];
        let idx_ty = self.f.slots[plan.idx_slot.0 as usize].ty;
        let idx_sz = x64_layout(self.env, idx_ty).size.clamp(1, 8);
        let idx_signed = is_signed_int(self.env, idx_ty);

        self.buf.mov_rr(Reg::R12, Reg::Rdi); // _pi = _start
        self.buf.mov_rr(Reg::R13, Reg::Rsi); // _end
        self.buf.mov_rr(Reg::R14, Reg::Rdx); // parent_rbp
        self.buf.load_mem(Reg::Rbx, Reg::R14, idx_off, idx_sz, idx_signed); // base_start

        let l_loop = self.buf.new_label();
        let l_step = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.active_worker = Some(ActiveWorker {
            header_bb: plan.header_bb,
            step_bb: plan.step_bb,
            exit_bb: plan.exit_bb,
            inner_val_defs: plan.inner_val_defs.clone(),
            cap_slots: plan.cap_slots.clone(),
            worker_step_lbl: l_step,
            worker_done_lbl: l_done,
        });

        self.buf.bind_label(l_loop);
        self.buf.cmp_rr(Reg::R12, Reg::R13);
        self.buf.jcc_label(Cond::Ae, l_done);
        // idx_slot = base_start + _pi (stored in thread-private [rbp + idx_off])
        self.buf.mov_rr(Reg::Rax, Reg::Rbx);
        self.buf.add_rr(Reg::Rax, Reg::R12);
        self.buf.store_mem(Reg::Rbp, idx_off, Reg::Rax, idx_sz);

        for &bb in &plan.body_bbs {
            let lbl = self.bb_labels[&bb];
            self.buf.bind_label(lbl);
            let blk = self.f.block(bb);
            for inst in &blk.insts {
                self.record_line(inst.span);
                self.emit_inst(inst)?;
            }
            self.emit_term(blk);
        }

        self.buf.bind_label(l_step);
        self.buf.add_ri32(Reg::R12, 1);
        self.buf.jmp_label(l_loop);

        self.buf.bind_label(l_done);
        self.active_worker = None;
        self.buf.mov_rr(Reg::Rsp, Reg::Rbp);
        self.buf.pop(Reg::Rbp);
        self.buf.pop(Reg::R14);
        self.buf.pop(Reg::R13);
        self.buf.pop(Reg::R12);
        self.buf.pop(Reg::Rbx);
        self.buf.ret();
        Ok(())
    }

    fn emit_sched_wrapper(&mut self, w: &DeferredSchedWrapper) {
        while !self.buf.pos().is_multiple_of(16) {
            self.buf.nop();
        }
        self.buf.bind_label(w.lbl);
        // SysV wrapper: rdi = arg_bundle pointer -> move to R10 (never in EON_GP_ARG_REGS[0..6])
        self.buf.push(Reg::Rbp);
        self.buf.mov_rr(Reg::Rbp, Reg::Rsp);
        let mut copy_bytes = 0usize;
        for (_, pass) in &w.params {
            if let SchedArgPass::DerefIndirect { bytes, .. } = pass {
                copy_bytes += align_up(*bytes, 16);
            }
        }
        let aligned_copy = align_up(copy_bytes, 16);
        if aligned_copy > 0 {
            self.buf.sub_ri32(Reg::Rsp, aligned_copy as i32);
        }
        self.buf.mov_rr(Reg::R10, Reg::Rdi); // bundle_ptr (note EonCall arg6=R10, arg7=R11)

        // First perform any DerefIndirect copies into wrapper stack frame [rbp - copy_off].
        let mut tmp_off = 0i32;
        let mut deref_stack_offs: Vec<i32> = Vec::with_capacity(w.params.len());
        for (_, pass) in &w.params {
            if let SchedArgPass::DerefIndirect { off, bytes } = pass {
                tmp_off -= align_up(*bytes, 16) as i32;
                self.buf.load_mem(Reg::R9, Reg::R10, *off as i32, 8, false);
                if *bytes <= 64 {
                    self.copy_mem_using_rax(Reg::Rbp, tmp_off, Reg::R9, 0, *bytes);
                } else {
                    self.buf.push(Reg::R10);
                    self.buf.push(Reg::R9);
                    self.buf.lea_mem(Reg::Rdi, Reg::Rbp, tmp_off);
                    self.buf.pop(Reg::Rsi);
                    self.buf.mov_ri64(Reg::Rcx, *bytes as u64);
                    self.buf.rep_movsb();
                    self.buf.pop(Reg::R10);
                }
                deref_stack_offs.push(tmp_off);
            } else {
                deref_stack_offs.push(0);
            }
        }

        // Preserve bundle_ptr in Rbx if >6 GP args needed (`EonCall` has r10/r11 as GP6/GP7!).
        let need_stack_or_hi_gp = w.params.len() > 6;
        if need_stack_or_hi_gp {
            self.buf.push(Reg::Rbx);
            self.buf.push(Reg::R12); // keep rsp 16B aligned
            self.buf.mov_rr(Reg::Rbx, Reg::R10);
        }
        let bundle_reg = if need_stack_or_hi_gp { Reg::Rbx } else { Reg::R10 };

        // Count stack args if >8 GP args or >8 XMM args.
        let mut gp_i = 0usize;
        let mut xmm_i = 0usize;
        enum WArg {
            Gp1 { reg: Reg, pass_idx: usize },
            Gp2 { r0: Reg, r1: Reg, off: usize, bytes: usize },
            Float { xmm: Xmm, off: usize, is_f64: bool },
            Stack1 { pass_idx: usize },
            Stack2 { off: usize },
        }
        let mut wargs = Vec::new();
        for (pi, (ac, _)) in w.params.iter().enumerate() {
            match ac {
                ArgClass::Void => {}
                ArgClass::Float32 => {
                    if let SchedArgPass::BundleScalar { off, .. } = w.params[pi].1 {
                        if xmm_i < FLOAT_ARG_REGS.len() {
                            wargs.push(WArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], off, is_f64: false });
                            xmm_i += 1;
                        } else {
                            wargs.push(WArg::Stack1 { pass_idx: pi });
                        }
                    }
                }
                ArgClass::Float64 => {
                    if let SchedArgPass::BundleScalar { off, .. } = w.params[pi].1 {
                        if xmm_i < FLOAT_ARG_REGS.len() {
                            wargs.push(WArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], off, is_f64: true });
                            xmm_i += 1;
                        } else {
                            wargs.push(WArg::Stack1 { pass_idx: pi });
                        }
                    }
                }
                ArgClass::Reg1(_) | ArgClass::Indirect(_) => {
                    if gp_i < EON_GP_ARG_REGS.len() {
                        wargs.push(WArg::Gp1 { reg: EON_GP_ARG_REGS[gp_i], pass_idx: pi });
                        gp_i += 1;
                    } else {
                        wargs.push(WArg::Stack1 { pass_idx: pi });
                    }
                }
                ArgClass::Reg2(sz) => {
                    if let SchedArgPass::BundlePair { off } = w.params[pi].1 {
                        if gp_i + 1 < EON_GP_ARG_REGS.len() {
                            wargs.push(WArg::Gp2 {
                                r0: EON_GP_ARG_REGS[gp_i],
                                r1: EON_GP_ARG_REGS[gp_i + 1],
                                off,
                                bytes: *sz,
                            });
                            gp_i += 2;
                        } else {
                            wargs.push(WArg::Stack2 { off });
                        }
                    }
                }
            }
        }
        let stack_bytes: usize = wargs
            .iter()
            .filter_map(|a| match a {
                WArg::Stack1 { .. } => Some(8usize),
                WArg::Stack2 { .. } => Some(16usize),
                _ => None,
            })
            .sum();
        let aligned_stack = align_up(stack_bytes, 16);
        if aligned_stack > 0 {
            self.buf.sub_ri32(Reg::Rsp, aligned_stack as i32);
            let mut sp = 0i32;
            for wa in &wargs {
                match wa {
                    WArg::Stack1 { pass_idx } => {
                        match &w.params[*pass_idx].1 {
                            SchedArgPass::BundleScalar { off, bytes } => {
                                self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32, (*bytes).clamp(1, 8), false);
                            }
                            SchedArgPass::DerefScalar { off, bytes } => {
                                self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32, 8, false);
                                self.buf.load_mem(Reg::Rax, Reg::Rax, 0, (*bytes).clamp(1, 8), false);
                            }
                            SchedArgPass::BundlePtr { off } => {
                                self.buf.lea_mem(Reg::Rax, bundle_reg, *off as i32);
                            }
                            SchedArgPass::DerefIndirect { .. } => {
                                self.buf.lea_mem(Reg::Rax, Reg::Rbp, deref_stack_offs[*pass_idx]);
                            }
                            SchedArgPass::BundlePair { off, .. } => {
                                self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32, 8, false);
                            }
                        }
                        self.buf.store_mem(Reg::Rsp, sp, Reg::Rax, 8);
                        sp += 8;
                    }
                    WArg::Stack2 { off } => {
                        self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32, 8, false);
                        self.buf.store_mem(Reg::Rsp, sp, Reg::Rax, 8);
                        self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32 + 8, 8, false);
                        self.buf.store_mem(Reg::Rsp, sp + 8, Reg::Rax, 8);
                        sp += 16;
                    }
                    _ => {}
                }
            }
        }
        for wa in &wargs {
            match wa {
                WArg::Gp1 { reg, pass_idx } => match &w.params[*pass_idx].1 {
                    SchedArgPass::BundleScalar { off, bytes } => {
                        self.buf.load_mem(*reg, bundle_reg, *off as i32, (*bytes).clamp(1, 8), false);
                    }
                    SchedArgPass::DerefScalar { off, bytes } => {
                        self.buf.load_mem(Reg::Rax, bundle_reg, *off as i32, 8, false);
                        self.buf.load_mem(*reg, Reg::Rax, 0, (*bytes).clamp(1, 8), false);
                    }
                    SchedArgPass::BundlePtr { off } => {
                        self.buf.lea_mem(*reg, bundle_reg, *off as i32);
                    }
                    SchedArgPass::DerefIndirect { .. } => {
                        self.buf.lea_mem(*reg, Reg::Rbp, deref_stack_offs[*pass_idx]);
                    }
                    SchedArgPass::BundlePair { off, .. } => {
                        self.buf.load_mem(*reg, bundle_reg, *off as i32, 8, false);
                    }
                },
                WArg::Gp2 { r0, r1, off, bytes } => {
                    self.buf.load_mem(*r0, bundle_reg, *off as i32, 8, false);
                    if *bytes > 8 {
                        self.buf.load_mem(*r1, bundle_reg, *off as i32 + 8, 8, false);
                    }
                }
                WArg::Float { xmm, off, is_f64 } => {
                    if *is_f64 {
                        self.buf.movsd_load(*xmm, bundle_reg, *off as i32);
                    } else {
                        self.buf.movss_load(*xmm, bundle_reg, *off as i32);
                    }
                }
                _ => {}
            }
        }
        self.buf.call_sym(w.sys_sym.clone());
        if aligned_stack > 0 {
            self.buf.add_ri32(Reg::Rsp, aligned_stack as i32);
        }
        if need_stack_or_hi_gp {
            self.buf.pop(Reg::R12);
            self.buf.pop(Reg::Rbx);
        }
        self.buf.mov_rr(Reg::Rsp, Reg::Rbp);
        self.buf.pop(Reg::Rbp);
        self.buf.ret();
    }

    fn emit_function(&mut self) -> Result<(), String> {
        for b in &self.f.blocks {
            let l = self.buf.new_label();
            self.bb_labels.insert(b.id, l);
        }
        self.build_par_for_plans();

        // 1. Prologue (`@naked` emits ZERO prologue; `@interrupt` saves all GPRs!)
        if self.f.sig.effects.is_naked {
            // Raw machine code only!
        } else if self.f.sig.effects.is_interrupt {
            for r in [
                Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi,
                Reg::R8, Reg::R9, Reg::R10, Reg::R11, Reg::R15,
            ] {
                self.buf.push(r);
            }
            self.buf.push(Reg::Rbp);
            self.buf.mov_rr(Reg::Rbp, Reg::Rsp);
            for &r in &self.plan.used_callee_regs {
                self.buf.push(r);
            }
            if self.plan.frame_size > 0 {
                self.buf.sub_ri32(Reg::Rsp, self.plan.frame_size);
            }
        } else {
            self.buf.push(Reg::Rbp);
            self.buf.mov_rr(Reg::Rbp, Reg::Rsp);
            for &r in &self.plan.used_callee_regs {
                self.buf.push(r);
            }
            if self.plan.frame_size > 0 {
                self.buf.sub_ri32(Reg::Rsp, self.plan.frame_size);
            }
        }

        // 2. Receive incoming arguments according to EonCall / System V AMD64 ABI.
        if !self.f.sig.effects.is_naked {
            self.receive_params();
        }

        // 3. Emit blocks in RPO order!
        for &bb in &self.f.rpo.clone() {
            if self.outlined_skip_bbs.contains(&bb) {
                continue;
            }
            let lbl = self.bb_labels[&bb];
            self.buf.bind_label(lbl);
            if let Some(plan) = self.par_for_plans.iter().find(|p| p.header_bb == bb).cloned() {
                self.emit_par_for_launch(&plan);
                continue;
            }
            let blk = self.f.block(bb);
            for inst in &blk.insts {
                self.record_line(inst.span);
                self.emit_inst(inst)?;
            }
            self.emit_term(blk);
        }

        // 4. Emit outlined parallel `par for` workers & schedule wrappers at end of blob.
        let plans = self.par_for_plans.clone();
        for plan in &plans {
            self.emit_par_for_worker(plan)?;
        }
        let sched_wrappers = self.deferred_sched_wrappers.clone();
        for w in &sched_wrappers {
            self.emit_sched_wrapper(w);
        }
        Ok(())
    }

    fn receive_params(&mut self) {
        let gp_regs: &[Reg] = if self.f.sig.effects.is_export_c {
            &SYSV_GP_ARG_REGS
        } else {
            &EON_GP_ARG_REGS
        };
        let mut gp_i = 0usize;
        let mut xmm_i = 0usize;
        let mut stack_arg_off = 16i32; // [rbp + 16] is first stack arg

        if let Some(sret_off) = self.plan.sret_off {
            self.buf.store_mem(Reg::Rbp, sret_off, gp_regs[0], 8);
            gp_i = 1;
        }

        for &pv in &self.f.params {
            let pty = self.f.value_tys[pv.0 as usize];
            match classify_arg(self.env, pty) {
                ArgClass::Void => {}
                ArgClass::Float32 => {
                    let off = self.val_stack_off(pv);
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        self.buf.movss_store(Reg::Rbp, off, FLOAT_ARG_REGS[xmm_i]);
                        xmm_i += 1;
                    } else {
                        self.buf.load_mem(Reg::Rax, Reg::Rbp, stack_arg_off, 4, false);
                        self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 4);
                        stack_arg_off += 8;
                    }
                }
                ArgClass::Float64 => {
                    let off = self.val_stack_off(pv);
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        self.buf.movsd_store(Reg::Rbp, off, FLOAT_ARG_REGS[xmm_i]);
                        xmm_i += 1;
                    } else {
                        self.buf.load_mem(Reg::Rax, Reg::Rbp, stack_arg_off, 8, false);
                        self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                        stack_arg_off += 8;
                    }
                }
                ArgClass::Reg1(sz) => {
                    let src_reg = if gp_i < gp_regs.len() {
                        let r = gp_regs[gp_i];
                        gp_i += 1;
                        r
                    } else {
                        self.buf.load_mem(Reg::Rax, Reg::Rbp, stack_arg_off, sz, false);
                        stack_arg_off += 8;
                        Reg::Rax
                    };
                    self.store_scalar_val(pv, src_reg, sz);
                }
                ArgClass::Reg2(sz) => {
                    let off = self.val_stack_off(pv);
                    if gp_i + 1 < gp_regs.len() {
                        let r0 = gp_regs[gp_i];
                        let r1 = gp_regs[gp_i + 1];
                        gp_i += 2;
                        self.buf.store_mem(Reg::Rbp, off, r0, 8);
                        if sz > 8 {
                            self.buf.store_mem(Reg::Rbp, off + 8, r1, 8);
                        }
                    } else {
                        self.buf.load_mem(Reg::Rax, Reg::Rbp, stack_arg_off, 8, false);
                        self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                        self.buf.load_mem(Reg::Rax, Reg::Rbp, stack_arg_off + 8, 8, false);
                        self.buf.store_mem(Reg::Rbp, off + 8, Reg::Rax, 8);
                        stack_arg_off += 16;
                    }
                }
                ArgClass::Indirect(sz) => {
                    // Caller passed pointer to argument copy in GP reg (or stack slot).
                    let ptr_reg = if gp_i < gp_regs.len() {
                        let r = gp_regs[gp_i];
                        gp_i += 1;
                        r
                    } else {
                        // All GP argument registers have already been consumed above, so `Rdi` is
                        // free and avoids colliding with `Rax` inside `copy_mem_using_rax`!
                        self.buf.load_mem(Reg::Rdi, Reg::Rbp, stack_arg_off, 8, false);
                        stack_arg_off += 8;
                        Reg::Rdi
                    };
                    let off = self.val_stack_off(pv);
                    // Save remaining arg regs? No, emit_memcpy_inline ONLY uses rdi/rsi/rcx when sz > 32;
                    // copy via rax using simple unrolled/loop copy on r10/r11/rax? Wait: r10/r11 might hold later args!
                    // Push gp_regs[gp_i..] or copy using ONLY rax! Actually push/pop is 2 bytes and 100% safe:
                    // Even simpler: copy `ptr_reg -> [rbp + off]` using ONLY `rax` as temp!
                    // Wait: we can't mutate `ptr_reg` if it's still needed, no it's already consumed!
                    // Copy 8 bytes at a time using `rax` and constant displacements from `ptr_reg` & `rbp`!
                    self.copy_mem_using_rax(Reg::Rbp, off, ptr_reg, 0, sz);
                }
            }
        }
    }

    fn copy_mem_using_rax(&mut self, dst_base: Reg, dst_disp: i32, src_base: Reg, src_disp: i32, sz: usize) {
        let mut i = 0usize;
        while i + 8 <= sz {
            self.buf.load_mem(Reg::Rax, src_base, src_disp + i as i32, 8, false);
            self.buf.store_mem(dst_base, dst_disp + i as i32, Reg::Rax, 8);
            i += 8;
        }
        while i + 4 <= sz {
            self.buf.load_mem(Reg::Rax, src_base, src_disp + i as i32, 4, false);
            self.buf.store_mem(dst_base, dst_disp + i as i32, Reg::Rax, 4);
            i += 4;
        }
        while i < sz {
            self.buf.load_mem(Reg::Rax, src_base, src_disp + i as i32, 1, false);
            self.buf.store_mem(dst_base, dst_disp + i as i32, Reg::Rax, 1);
            i += 1;
        }
    }

    /// Copy `sz` bytes from `[src_ptr + 0]` to `[dst_ptr + 0]`. Clobbers `rdi, rsi, rcx, rax`.
    fn emit_memcpy_ptrs(&mut self, dst_ptr: Reg, src_ptr: Reg, sz: usize) {
        if sz == 0 {
            return;
        }
        if sz <= 64 {
            // Avoid clobbering dst_ptr/src_ptr if one is rax by moving to r10/r11 if needed.
            let d = if dst_ptr == Reg::Rax {
                self.buf.mov_rr(Reg::R10, dst_ptr);
                Reg::R10
            } else {
                dst_ptr
            };
            let s = if src_ptr == Reg::Rax {
                self.buf.mov_rr(Reg::R11, src_ptr);
                Reg::R11
            } else {
                src_ptr
            };
            self.copy_mem_using_rax(d, 0, s, 0, sz);
        } else {
            // Move into rdi/rsi safely even if swapped.
            if dst_ptr == Reg::Rsi && src_ptr == Reg::Rdi {
                self.buf.mov_rr(Reg::Rax, Reg::Rdi);
                self.buf.mov_rr(Reg::Rdi, Reg::Rsi);
                self.buf.mov_rr(Reg::Rsi, Reg::Rax);
            } else if dst_ptr == Reg::Rsi {
                self.buf.mov_rr(Reg::Rdi, dst_ptr);
                self.buf.mov_rr(Reg::Rsi, src_ptr);
            } else {
                self.buf.mov_rr(Reg::Rsi, src_ptr);
                self.buf.mov_rr(Reg::Rdi, dst_ptr);
            }
            self.buf.mov_ri64(Reg::Rcx, sz as u64);
            self.buf.rep_movsb();
        }
    }

    fn emit_memzero(&mut self, base: Reg, disp: i32, sz: usize) {
        if sz == 0 {
            return;
        }
        if sz <= 64 {
            self.buf.xor_rr32(Reg::Rax, Reg::Rax);
            let mut i = 0usize;
            while i + 8 <= sz {
                self.buf.store_mem(base, disp + i as i32, Reg::Rax, 8);
                i += 8;
            }
            while i + 4 <= sz {
                self.buf.store_mem(base, disp + i as i32, Reg::Rax, 4);
                i += 4;
            }
            while i < sz {
                self.buf.store_mem(base, disp + i as i32, Reg::Rax, 1);
                i += 1;
            }
        } else {
            self.buf.lea_mem(Reg::Rdi, base, disp);
            self.buf.xor_rr32(Reg::Rax, Reg::Rax);
            self.buf.mov_ri64(Reg::Rcx, sz as u64);
            self.buf.rep_stosb();
        }
    }

    fn val_stack_off(&self, v: EirValue) -> i32 {
        match self.plan.val_locs[v.0 as usize] {
            ValLoc::Stack(off) => off,
            ValLoc::Reg(_) => self.plan.temp_copy_off, // fallback for unexpected stack access
        }
    }

    fn load_scalar_val(&mut self, dst_reg: Reg, v: EirValue) {
        let ty = self.f.value_tys[v.0 as usize];
        let sz = x64_layout(self.env, ty).size.clamp(1, 8);
        let signed = is_signed_int(self.env, ty);
        match self.plan.val_locs[v.0 as usize] {
            ValLoc::Reg(r) => {
                self.buf.mov_rr(dst_reg, r);
                if sz < 8 {
                    self.buf.extend_reg(dst_reg, dst_reg, (sz * 8) as u8, signed);
                }
            }
            ValLoc::Stack(off) => {
                let base = self.val_mem(v).0;
                self.buf.load_mem(dst_reg, base, off, sz, signed);
            }
        }
    }

    fn load_scalar_raw(&mut self, dst_reg: Reg, v: EirValue, bytes: usize) {
        match self.plan.val_locs[v.0 as usize] {
            ValLoc::Reg(r) => self.buf.mov_rr(dst_reg, r),
            ValLoc::Stack(off) => {
                let base = self.val_mem(v).0;
                self.buf.load_mem(dst_reg, base, off, bytes.clamp(1, 8), false);
            }
        }
    }

    fn store_scalar_val(&mut self, v: EirValue, src_reg: Reg, bytes: usize) {
        let sz = bytes.clamp(1, 8);
        match self.plan.val_locs[v.0 as usize] {
            ValLoc::Reg(r) => {
                self.buf.mov_rr(r, src_reg);
                if sz < 8 {
                    let signed = is_signed_int(self.env, self.f.value_tys[v.0 as usize]);
                    self.buf.extend_reg(r, r, (sz * 8) as u8, signed);
                }
            }
            ValLoc::Stack(off) => {
                self.buf.store_mem(Reg::Rbp, off, src_reg, sz);
            }
        }
    }

    fn load_float_val(&mut self, dst: Xmm, v: EirValue, is_f64: bool) {
        let (base, off) = self.val_mem(v);
        if is_f64 {
            self.buf.movsd_load(dst, base, off);
        } else {
            self.buf.movss_load(dst, base, off);
        }
    }

    fn store_float_val(&mut self, v: EirValue, src: Xmm, is_f64: bool) {
        let off = self.val_stack_off(v);
        if is_f64 {
            self.buf.movsd_store(Reg::Rbp, off, src);
        } else {
            self.buf.movss_store(Reg::Rbp, off, src);
        }
    }

    /// Put address of value `v` (in stack memory) into `dst_reg`. If `v` lives in a register,
    /// spills it to `temp_copy_off` first.
    fn val_addr_into(&mut self, dst_reg: Reg, v: EirValue) {
        match self.plan.val_locs[v.0 as usize] {
            ValLoc::Stack(off) => {
                let base = self.val_mem(v).0;
                self.buf.lea_mem(dst_reg, base, off);
            }
            ValLoc::Reg(r) => {
                self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, r, 8);
                self.buf.lea_mem(dst_reg, Reg::Rbp, self.plan.temp_copy_off);
            }
        }
    }

    /// Copy any value `src` into any value `dst` using the target type's natural size.
    fn copy_val_to_val(&mut self, dst: EirValue, src: EirValue) {
        let ty = self.f.value_tys[dst.0 as usize];
        let sz = x64_layout(self.env, ty).size;
        self.copy_val_to_val_sz(dst, src, sz);
    }

    /// Copy `sz` bytes from `src` into `dst`.
    fn copy_val_to_val_sz(&mut self, dst: EirValue, src: EirValue, sz: usize) {
        let ty = self.f.value_tys[dst.0 as usize];
        if sz == 0 {
            return;
        }
        match (self.plan.val_locs[dst.0 as usize], self.plan.val_locs[src.0 as usize]) {
            (ValLoc::Reg(dr), ValLoc::Reg(sr)) => self.buf.mov_rr(dr, sr),
            (ValLoc::Reg(dr), ValLoc::Stack(so)) => {
                let sbase = self.val_mem(src).0;
                self.buf.load_mem(dr, sbase, so, sz.clamp(1, 8), is_signed_int(self.env, ty));
            }
            (ValLoc::Stack(doff), ValLoc::Reg(sr)) => {
                self.buf.store_mem(Reg::Rbp, doff, sr, sz.clamp(1, 8));
            }
            (ValLoc::Stack(doff), ValLoc::Stack(soff)) => {
                let sbase = self.val_mem(src).0;
                if sbase != Reg::Rbp || doff != soff {
                    self.copy_mem_using_rax(Reg::Rbp, doff, sbase, soff, sz);
                }
            }
        }
    }

    /// Store value `src` into `[dst_ptr + dst_disp]`. Clobbers `rax, rdi, rsi, rcx` if sz > 64.
    fn store_val_to_mem(&mut self, dst_ptr: Reg, dst_disp: i32, src: EirValue, ty: TyId) {
        let sz = x64_layout(self.env, ty).size;
        self.store_val_to_mem_sz(dst_ptr, dst_disp, src, sz);
    }

    /// Store `sz` bytes of value `src` into `[dst_ptr + dst_disp]`.
    fn store_val_to_mem_sz(&mut self, dst_ptr: Reg, dst_disp: i32, src: EirValue, sz: usize) {
        if sz == 0 {
            return;
        }
        match self.plan.val_locs[src.0 as usize] {
            ValLoc::Reg(sr) => self.buf.store_mem(dst_ptr, dst_disp, sr, sz.clamp(1, 8)),
            ValLoc::Stack(soff) => {
                let sbase = self.val_mem(src).0;
                if sz <= 64 {
                    self.copy_mem_using_rax(dst_ptr, dst_disp, sbase, soff, sz);
                } else {
                    self.buf.lea_mem(Reg::Rdi, dst_ptr, dst_disp);
                    self.buf.lea_mem(Reg::Rsi, sbase, soff);
                    self.buf.mov_ri64(Reg::Rcx, sz as u64);
                    self.buf.rep_movsb();
                }
            }
        }
    }

    /// Load value `dst` from `[src_ptr + src_disp]`.
    fn load_val_from_mem(&mut self, dst: EirValue, src_ptr: Reg, src_disp: i32, ty: TyId) {
        let sz = x64_layout(self.env, ty).size;
        self.load_val_from_mem_sz(dst, src_ptr, src_disp, ty, sz);
    }

    /// Load `sz` bytes into `dst` from `[src_ptr + src_disp]`.
    fn load_val_from_mem_sz(&mut self, dst: EirValue, src_ptr: Reg, src_disp: i32, ty: TyId, sz: usize) {
        if sz == 0 {
            return;
        }
        match self.plan.val_locs[dst.0 as usize] {
            ValLoc::Reg(dr) => {
                self.buf.load_mem(dr, src_ptr, src_disp, sz.clamp(1, 8), is_signed_int(self.env, ty));
            }
            ValLoc::Stack(doff) => {
                if sz <= 64 {
                    self.copy_mem_using_rax(Reg::Rbp, doff, src_ptr, src_disp, sz);
                } else {
                    self.buf.lea_mem(Reg::Rsi, src_ptr, src_disp);
                    self.buf.lea_mem(Reg::Rdi, Reg::Rbp, doff);
                    self.buf.mov_ri64(Reg::Rcx, sz as u64);
                    self.buf.rep_movsb();
                }
            }
        }
    }

    // ========================================================================
    // Place Address Resolution
    // ========================================================================

    /// Computes address of `place` into `Reg::R8`! Returns metadata including whether a
    /// bare `PoolCol` slice `{ col_ptr, len }` was resolved and whether pointer/packed access occurred.
    fn compute_place_addr(&mut self, p: &EirPlace) -> (Option<(usize, bool, TyId)>, bool, bool) {
        let mut cur_ty = match p.base {
            EirPlaceBase::Slot(s) => {
                let (base, off) = self.slot_mem(s);
                self.buf.lea_mem(Reg::R8, base, off);
                self.f.slots[s.0 as usize].ty
            }
            EirPlaceBase::Val(v) => {
                let vty = self.f.value_tys[v.0 as usize];
                self.val_addr_into(Reg::R8, v);
                vty
            }
        };

        let mut has_ptr_deref = false;
        let mut in_packed = false;
        let mut i = 0usize;
        while i < p.projs.len() {
            match &p.projs[i] {
                EirProj::Deref => {
                    let inner = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if !matches!(self.env.get(inner), Ty::Pool { .. }) {
                        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                    }
                    has_ptr_deref = true;
                    in_packed = false;
                    cur_ty = inner;
                }
                EirProj::Field { field_idx, ty, .. } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if matches!(self.env.get(deref_t), Ty::Pool { .. }) && is_ptr {
                        // RowView { pool_ptr (0), index (8) }!
                        self.buf.load_mem(Reg::R9, Reg::R8, 0, 8, false); // pool_ptr
                        self.buf.load_mem(Reg::R10, Reg::R8, 8, 4, false); // index (u32)
                        self.buf.load_mem(Reg::R8, Reg::R9, (*field_idx as i32) * 8, 8, false); // col_ptr
                        let elem_sz = x64_layout_impl(self.env, *ty, true).size.max(1);
                        self.buf.imul_rri32(Reg::R10, Reg::R10, elem_sz as i32);
                        self.buf.add_rr(Reg::R8, Reg::R10);
                        has_ptr_deref = true;
                        in_packed = false;
                    } else {
                        if is_ptr {
                            self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                            has_ptr_deref = true;
                            in_packed = false;
                        }
                        if let Ty::Struct { def, .. } = self.env.get(deref_t)
                            && self.env.structs.get(def).is_some_and(|m| m.is_packed)
                        {
                            in_packed = true;
                        }
                        let (foff, _) = struct_field_offset(self.env, deref_t, *field_idx);
                        if foff != 0 {
                            self.buf.add_ri32(Reg::R8, foff as i32);
                        }
                    }
                    cur_ty = *ty;
                }
                EirProj::TupleField { index, ty } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if is_ptr {
                        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                        has_ptr_deref = true;
                        in_packed = false;
                    }
                    let (foff, _) = struct_field_offset(self.env, deref_t, *index);
                    if foff != 0 {
                        self.buf.add_ri32(Reg::R8, foff as i32);
                    }
                    cur_ty = *ty;
                }
                EirProj::PoolCol { col_idx, is_slot_pool, elem_ty, .. } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    if is_ptr {
                        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                    }
                    if let Some(EirProj::Index { index, ty }) = p.projs.get(i + 1) {
                        // Load col_ptr = [pool + col_idx * 8]
                        self.buf.load_mem(Reg::R8, Reg::R8, (*col_idx as i32) * 8, 8, false);
                        // Load index (if slot pool, low 32 bits of GenId is `.index`!)
                        if *is_slot_pool {
                            self.load_scalar_raw(Reg::R9, *index, 4);
                        } else {
                            self.load_scalar_val(Reg::R9, *index);
                        }
                        let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                        self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32);
                        self.buf.add_rr(Reg::R8, Reg::R9);
                        cur_ty = *ty;
                        i += 2;
                        continue;
                    } else {
                        // Bare pool column yields a slice `{ col_ptr, len }`!
                        return (Some((*col_idx, *is_slot_pool, cur_ty)), has_ptr_deref, in_packed);
                    }
                }
                EirProj::Index { index, ty } => {
                    let is_ptr_single = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if is_ptr_single {
                        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                        has_ptr_deref = true;
                        in_packed = false;
                    }
                    match self.env.get(deref_t) {
                        Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str => {
                            // Slice / str is `{ ptr (0), len (8) }` -> load `.ptr`!
                            self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                            has_ptr_deref = true;
                            in_packed = false;
                        }
                        Ty::Array { .. } | Ty::Vec { .. } => {
                            // R8 already points to element 0!
                        }
                        _ => {}
                    }
                    self.load_scalar_val(Reg::R9, *index);
                    let esz = x64_layout(self.env, *ty).size.max(1);
                    if esz != 1 {
                        self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32);
                    }
                    self.buf.add_rr(Reg::R8, Reg::R9);
                    cur_ty = *ty;
                }
            }
            i += 1;
        }
        (None, has_ptr_deref, in_packed)
    }

    // ========================================================================
    // Instruction Lowering (FastISel)
    // ========================================================================

    fn emit_inst(&mut self, inst: &EirInst) -> Result<(), String> {
        match &inst.kind {
            EirInstKind::Const { dst, ty, konst } => self.emit_const(*dst, *ty, konst),
            EirInstKind::CopyVal { dst, src } => self.copy_val_to_val(*dst, *src),
            EirInstKind::LoadSlot { dst, slot } => {
                let (base, off) = self.slot_mem(*slot);
                let ty = self.f.value_tys[dst.0 as usize];
                self.load_val_from_mem(*dst, base, off, ty);
            }
            EirInstKind::StoreSlot { slot, val } => {
                let (base, off) = self.slot_mem(*slot);
                let ty = self.f.slots[slot.0 as usize].ty;
                self.store_val_to_mem(base, off, *val, ty);
            }
            EirInstKind::SlotAddr { dst, slot, .. } => {
                let (base, off) = self.slot_mem(*slot);
                self.buf.lea_mem(Reg::Rax, base, off);
                self.store_scalar_val(*dst, Reg::Rax, 8);
            }
            EirInstKind::LoadPlace { dst, place } => {
                if let Some((bf_def, fidx, base_place)) = self.split_bitfield_tail(place) {
                    let (_, has_ptr, in_packed) = self.compute_place_addr(&base_place);
                    let (base_sz, lo, width) = self.bitfield_member_layout(bf_def, fidx);
                    if has_ptr && !in_packed {
                        self.emit_align_check(Reg::R8, base_sz);
                    }
                    self.buf.load_mem(Reg::Rax, Reg::R8, 0, base_sz, false);
                    if lo > 0 {
                        self.buf.shr_imm(Reg::Rax, lo);
                    }
                    let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                    if width < 64 {
                        self.buf.mov_ri64(Reg::Rcx, mask);
                        self.buf.and_rr(Reg::Rax, Reg::Rcx);
                    }
                    let dst_ty = self.f.value_tys[dst.0 as usize];
                    let dsz = x64_layout(self.env, dst_ty).size.clamp(1, 8);
                    self.store_scalar_val(*dst, Reg::Rax, dsz);
                } else {
                    let (bare_col, has_ptr, in_packed) = self.compute_place_addr(place);
                    if let Some((col_idx, is_slot, pool_ty)) = bare_col {
                        // Bare PoolCol -> materialize slice `{ pool->col_i, pool->len }`
                        let deref = match self.env.get(pool_ty) {
                            Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t,
                            _ => pool_ty,
                        };
                        let ncols = match self.env.get(deref) {
                            Ty::Builtin(_, args) => pool_col_count(self.env, args),
                            _ => 0,
                        };
                        let len_off = if is_slot { (ncols + 2) * 8 } else { ncols * 8 };
                        self.buf.load_mem(Reg::Rax, Reg::R8, (col_idx * 8) as i32, 8, false);
                        self.buf.load_mem(Reg::Rdx, Reg::R8, len_off as i32, 8, false);
                        let doff = self.val_stack_off(*dst);
                        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                        self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
                    } else {
                        let ty = self.f.value_tys[dst.0 as usize];
                        if has_ptr && !in_packed {
                            let align = x64_layout(self.env, place.ty).align.clamp(1, 8);
                            self.emit_align_check(Reg::R8, align);
                        }
                        self.load_val_from_mem(*dst, Reg::R8, 0, ty);
                    }
                }
            }
            EirInstKind::StorePlace { place, val } => {
                if let Some((bf_def, fidx, base_place)) = self.split_bitfield_tail(place) {
                    let (_, has_ptr, in_packed) = self.compute_place_addr(&base_place);
                    let (base_sz, lo, width) = self.bitfield_member_layout(bf_def, fidx);
                    if has_ptr && !in_packed {
                        self.emit_align_check(Reg::R8, base_sz);
                    }
                    let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                    let shifted_mask = if lo >= 64 { 0 } else { mask.wrapping_shl(lo as u32) };
                    self.buf.load_mem(Reg::Rax, Reg::R8, 0, base_sz, false);
                    self.buf.mov_ri64(Reg::Rdx, !shifted_mask);
                    self.buf.and_rr(Reg::Rax, Reg::Rdx);
                    self.load_scalar_val(Reg::Rcx, *val);
                    if width < 64 {
                        self.buf.mov_ri64(Reg::Rdx, mask);
                        self.buf.and_rr(Reg::Rcx, Reg::Rdx);
                    }
                    if lo > 0 {
                        self.buf.shl_imm(Reg::Rcx, lo);
                    }
                    self.buf.or_rr(Reg::Rax, Reg::Rcx);
                    self.buf.store_mem(Reg::R8, 0, Reg::Rax, base_sz);
                } else {
                    let (_, has_ptr, in_packed) = self.compute_place_addr(place);
                    if has_ptr && !in_packed {
                        let align = x64_layout(self.env, place.ty).align.clamp(1, 8);
                        self.emit_align_check(Reg::R8, align);
                    }
                    self.store_val_to_mem(Reg::R8, 0, *val, place.ty);
                }
            }
            EirInstKind::PlaceAddr { dst, place, .. } => {
                let _ = self.compute_place_addr(place);
                self.store_scalar_val(*dst, Reg::R8, 8);
            }
            EirInstKind::PoolColumnLoad {
                dst,
                pool,
                pool_is_ptr,
                col_idx,
                is_slot_pool,
                index,
                elem_ty,
                ..
            } => {
                self.load_pool_base_into_r8(pool, *pool_is_ptr);
                self.buf.load_mem(Reg::R8, Reg::R8, (*col_idx * 8) as i32, 8, false);
                if *is_slot_pool {
                    self.load_scalar_raw(Reg::R9, *index, 4);
                } else {
                    self.load_scalar_val(Reg::R9, *index);
                }
                let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                if esz != 1 {
                    self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32);
                }
                self.buf.add_rr(Reg::R8, Reg::R9);
                let dst_ty = self.f.value_tys[dst.0 as usize];
                self.load_val_from_mem_sz(*dst, Reg::R8, 0, dst_ty, esz);
            }
            EirInstKind::PoolColumnStore {
                pool,
                pool_is_ptr,
                col_idx,
                is_slot_pool,
                index,
                val,
                elem_ty,
                ..
            } => {
                self.load_pool_base_into_r8(pool, *pool_is_ptr);
                self.buf.load_mem(Reg::R8, Reg::R8, (*col_idx * 8) as i32, 8, false);
                if *is_slot_pool {
                    self.load_scalar_raw(Reg::R9, *index, 4);
                } else {
                    self.load_scalar_val(Reg::R9, *index);
                }
                let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                if esz != 1 {
                    self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32);
                }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.store_val_to_mem_sz(Reg::R8, 0, *val, esz);
            }
            EirInstKind::Unary { dst, op, operand, ty } => self.emit_unary(*dst, *op, *operand, *ty)?,
            EirInstKind::Binary { dst, op, lhs, rhs, op_ty, dst_ty } => {
                self.emit_binary(*dst, *op, *lhs, *rhs, *op_ty, *dst_ty)?;
            }
            EirInstKind::Cast { dst, kind, operand, from_ty, to_ty } => {
                self.emit_cast(*dst, *kind, *operand, *from_ty, *to_ty)?;
            }
            EirInstKind::Slice { dst, base, base_ty, lo, hi, dst_ty } => {
                self.emit_slice(*dst, *base, *base_ty, *lo, *hi, *dst_ty);
            }
            EirInstKind::Aggregate { dst, kind, elems, dst_ty } => {
                self.emit_aggregate(*dst, kind, elems, *dst_ty);
            }
            EirInstKind::UnwrapOptional { dst, src, src_ty, dst_ty } => {
                if *dst_ty == TyId::VOID {
                    return Ok(());
                }
                let inner = match self.env.get(*src_ty) { Ty::Optional(i) => *i, _ => *dst_ty };
                if is_niche_id_opt(self.env, inner) {
                    self.copy_val_to_val(*dst, *src);
                } else {
                    let voff = opt_val_offset(self.env, inner);
                    let (sbase, soff) = self.val_mem(*src);
                    self.load_val_from_mem(*dst, sbase, soff + voff as i32, *dst_ty);
                }
            }
            EirInstKind::UnwrapErrorOk { dst, src, dst_ty, .. } => {
                if *dst_ty == TyId::VOID {
                    return Ok(());
                }
                let ok_off = eu_ok_offset(self.env, *dst_ty);
                let (sbase, soff) = self.val_mem(*src);
                self.load_val_from_mem(*dst, sbase, soff + ok_off as i32, *dst_ty);
            }
            EirInstKind::UnwrapErrorErr { dst, src, .. } => {
                let (sbase, soff) = self.val_mem(*src);
                self.buf.load_mem(Reg::Rax, sbase, soff, 4, false);
                self.store_scalar_val(*dst, Reg::Rax, 4);
            }
            EirInstKind::IsNonNull { dst, src, src_ty } => {
                let inner = match self.env.get(*src_ty) { Ty::Optional(i) => *i, _ => *src_ty };
                let (sbase, soff) = self.val_mem(*src);
                match opt_niche(self.env, inner) {
                    OptNiche::IdBrand => {
                        self.buf.load_mem(Reg::Rax, sbase, soff, 4, false);
                        self.buf.mov_ri64(Reg::Rcx, u32::MAX as u64);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::GenId | OptNiche::Link => {
                        self.buf.load_mem(Reg::Rax, sbase, soff + 4, 4, false); // .gen
                        self.buf.test_rr32(Reg::Rax, Reg::Rax);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::NullPtr | OptNiche::FatPtrNullFirst => {
                        self.buf.load_mem(Reg::Rax, sbase, soff, 8, false);
                        self.buf.test_rr(Reg::Rax, Reg::Rax);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::SliceMaxLen => {
                        self.buf.load_mem(Reg::Rax, sbase, soff + 8, 8, false); // .len
                        self.buf.mov_ri64(Reg::Rcx, u64::MAX);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::BoolTag => {
                        self.buf.load_mem(Reg::Rax, sbase, soff, 1, false);
                        self.buf.mov_ri64(Reg::Rcx, 2);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::ScalarSentinel { size, sentinel } => {
                        self.buf.load_mem(Reg::Rax, sbase, soff, size, false);
                        self.buf.mov_ri64(Reg::Rcx, sentinel);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.buf.setcc(Cond::Ne, Reg::Rax);
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                    OptNiche::None => {
                        self.buf.load_mem(Reg::Rax, sbase, soff, 1, false); // .has
                        self.store_scalar_val(*dst, Reg::Rax, 1);
                    }
                }
            }
            EirInstKind::IsError { dst, src, .. } => {
                let (sbase, soff) = self.val_mem(*src);
                self.buf.load_mem(Reg::Rax, sbase, soff, 4, false);
                self.buf.test_rr32(Reg::Rax, Reg::Rax);
                self.buf.setcc(Cond::Ne, Reg::Rax);
                self.store_scalar_val(*dst, Reg::Rax, 1);
            }
            EirInstKind::MatchTag { dst, src, src_ty } => match self.env.get(*src_ty) {
                Ty::ErrorUnion { .. } | Ty::Union { .. } => {
                    let (sbase, soff) = self.val_mem(*src);
                    self.buf.load_mem(Reg::Rax, sbase, soff, 4, false);
                    self.store_scalar_val(*dst, Reg::Rax, 4);
                }
                _ => {
                    self.load_scalar_raw(Reg::Rax, *src, 4);
                    self.store_scalar_val(*dst, Reg::Rax, 4);
                }
            },
            EirInstKind::FieldExtract { dst, src, src_ty, dst_ty, .. } => {
                if *dst_ty == TyId::VOID {
                    return Ok(());
                }
                let p_off = match self.env.get(*src_ty) {
                    Ty::Union { def, .. } => union_payload_offset(self.env, *def),
                    _ => 4,
                };
                let (sbase, soff) = self.val_mem(*src);
                self.load_val_from_mem(*dst, sbase, soff + p_off as i32, *dst_ty);
            }
            EirInstKind::MakeDyn { dst, ptr, iface_def, concrete_ty, .. } => {
                self.load_scalar_val(Reg::Rax, *ptr);
                let vt_sym = format!("eon_vtable_{}_{}", iface_def.0, concrete_ty.0);
                self.buf.lea_rip_sym(Reg::Rdx, vt_sym, 0);
                let doff = self.val_stack_off(*dst);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
            }
            EirInstKind::MakeClosure { dst, closure_id, env_ptr, .. } => {
                let csym = fn_sym_name(CfgFnKey::Closure(*closure_id), self.eir, self.interner);
                self.buf.lea_rip_sym(Reg::Rax, csym, 0);
                if let Some(ev) = env_ptr {
                    self.load_scalar_val(Reg::Rdx, *ev);
                } else {
                    self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                }
                let doff = self.val_stack_off(*dst);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
            }
            EirInstKind::InlineAsm { template, .. } => {
                encode_inline_asm(&mut self.buf, template)?;
            }
            EirInstKind::Call { dst, callee, args, ret_ty } => {
                self.emit_call(*dst, callee, args, *ret_ty);
            }
            EirInstKind::BuiltinCall { dst, op, args, arg_slots, ret_ty } => {
                self.emit_builtin_call(*dst, op, args, arg_slots, *ret_ty)?;
            }
            EirInstKind::GetVar { .. } | EirInstKind::SetVar { .. } => {
                return Err(format!(
                    "ICE [codegen_x64]: unlowered pseudo-instruction {:?} reached native x86_64 codegen in `{}`",
                    inst.kind,
                    self.interner.get(self.f.name)
                ));
            }
        }
        Ok(())
    }

    fn load_pool_base_into_r8(&mut self, pool: &EirPlaceBase, pool_is_ptr: bool) {
        match pool {
            EirPlaceBase::Slot(s) => {
                let (base, off) = self.slot_mem(*s);
                if pool_is_ptr {
                    self.buf.load_mem(Reg::R8, base, off, 8, false);
                } else {
                    self.buf.lea_mem(Reg::R8, base, off);
                }
            }
            EirPlaceBase::Val(v) => {
                if pool_is_ptr {
                    self.load_scalar_val(Reg::R8, *v);
                } else {
                    self.val_addr_into(Reg::R8, *v);
                }
            }
        }
    }

    fn emit_const(&mut self, dst: EirValue, ty: TyId, konst: &CfgConst) {
        let lay = x64_layout(self.env, ty);
        if lay.size == 0 {
            return;
        }
        match konst {
            CfgConst::Void => {}
            CfgConst::Bool(b) => {
                self.buf.mov_ri64(Reg::Rax, u64::from(*b));
                self.store_scalar_val(dst, Reg::Rax, 1);
            }
            CfgConst::Char(c) => {
                self.buf.mov_ri64(Reg::Rax, *c as u64);
                self.store_scalar_val(dst, Reg::Rax, 4);
            }
            CfgConst::Int(n, _) => {
                self.buf.mov_ri64(Reg::Rax, (*n as i64) as u64);
                self.store_scalar_val(dst, Reg::Rax, lay.size);
            }
            CfgConst::Float(s, fty) => {
                let clean = s.replace('_', "");
                let clean = clean
                    .trim_end_matches("_f16")
                    .trim_end_matches("_f32")
                    .trim_end_matches("_f64");
                let off = self.val_stack_off(dst);
                if matches!(self.env.get(*fty), Ty::Float(16)) {
                    let v = clean.parse::<f32>().unwrap_or(0.0);
                    let bits16 = crate::eir_interp::f32_to_f16_bits(v);
                    self.buf.mov_ri64(Reg::Rax, bits16 as u64);
                    self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 2);
                } else if matches!(self.env.get(*fty), Ty::Float(32)) {
                    let v = clean.parse::<f32>().unwrap_or(0.0);
                    self.buf.mov_ri64(Reg::Rax, v.to_bits() as u64);
                    self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 4);
                } else {
                    let v = clean.parse::<f64>().unwrap_or(0.0);
                    self.buf.mov_ri64(Reg::Rax, v.to_bits());
                    self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                }
            }
            CfgConst::Str(s) => {
                let mut bytes = s.as_bytes().to_vec();
                bytes.push(0); // NUL terminator for safety
                let rsym = self.intern_rodata(bytes, 1);
                let off = self.val_stack_off(dst);
                self.buf.lea_rip_sym(Reg::Rax, rsym, 0);
                self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                self.buf.mov_ri64(Reg::Rax, s.len() as u64);
                self.buf.store_mem(Reg::Rbp, off + 8, Reg::Rax, 8);
            }
            CfgConst::Bytes(bs) => {
                let rsym = self.intern_rodata(bs.clone(), 1);
                let off = self.val_stack_off(dst);
                self.buf.lea_rip_sym(Reg::Rax, rsym, 0);
                self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                self.buf.mov_ri64(Reg::Rax, bs.len() as u64);
                self.buf.store_mem(Reg::Rbp, off + 8, Reg::Rax, 8);
            }
            CfgConst::Null(_) => self.emit_opt_null(dst, ty),
            CfgConst::EnumTag { tag_val, .. } => {
                self.buf.mov_ri64(Reg::Rax, (*tag_val as i64) as u64);
                self.store_scalar_val(dst, Reg::Rax, 4);
            }
            CfgConst::ErrorTag { tag_val, .. } => {
                self.buf.mov_ri64(Reg::Rax, *tag_val as u64);
                self.store_scalar_val(dst, Reg::Rax, 4);
            }
            CfgConst::FnRef(d, _) => {
                let tsym = format!("eon_fnref_thunk_{}", d.0);
                self.buf.lea_rip_sym(Reg::Rax, tsym, 0);
                self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                let doff = self.val_stack_off(dst);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
            }
            CfgConst::ConstDef(d, _) => {
                if let Some(cm) = self.env.consts.get(d)
                    && let Some(ref ct) = cm.ct_val
                    && let Some(bytes) = encode_ct_val_bytes(ct, ty, self.env)
                {
                    if bytes.len() <= 8 {
                        let mut raw = [0u8; 8];
                        raw[..bytes.len()].copy_from_slice(&bytes);
                        self.buf.mov_ri64(Reg::Rax, u64::from_le_bytes(raw));
                        self.store_scalar_val(dst, Reg::Rax, lay.size);
                    } else {
                        let rsym = self.intern_rodata(bytes.clone(), lay.align.clamp(1, 16));
                        let doff = self.val_stack_off(dst);
                        self.buf.lea_rip_sym(Reg::Rsi, rsym, 0);
                        self.buf.lea_mem(Reg::Rdi, Reg::Rbp, doff);
                        self.buf.mov_ri64(Reg::Rcx, bytes.len() as u64);
                        self.buf.rep_movsb();
                    }
                } else if let Some(cm) = self.env.consts.get(d)
                    && let Some(iv) = cm.int_val
                {
                    self.buf.mov_ri64(Reg::Rax, (iv as i64) as u64);
                    self.store_scalar_val(dst, Reg::Rax, lay.size);
                } else {
                    match self.plan.val_locs[dst.0 as usize] {
                        ValLoc::Reg(r) => self.buf.xor_rr32(r, r),
                        ValLoc::Stack(off) => self.emit_memzero(Reg::Rbp, off, lay.size),
                    }
                }
            }
        }
    }

    fn emit_unary(&mut self, dst: EirValue, op: UnaryOp, operand: EirValue, ty: TyId) -> Result<(), String> {
        if let Ty::Vec { elem, len } = self.env.get(ty).clone() {
            let esz = x64_layout(self.env, elem).size.max(1);
            let (sbase, soff) = self.val_mem(operand);
            let doff = self.val_stack_off(dst);
            for lane in 0..(len as usize) {
                let off_s = soff + (lane * esz) as i32;
                let off_d = doff + (lane * esz) as i32;
                if let Ty::Float(bits) = self.env.get(elem) {
                    if op != UnaryOp::Neg {
                        return Err(format!("ICE [codegen_x64]: unsupported vec float unary op {op:?}"));
                    }
                    let (bytes, mask) = match *bits {
                        16 => (2, 0x8000u64),
                        32 => (4, 1u64 << 31),
                        _ => (8, 1u64 << 63),
                    };
                    self.buf.load_mem(Reg::Rax, sbase, off_s, bytes, false);
                    self.buf.mov_ri64(Reg::Rcx, mask);
                    self.buf.xor_rr(Reg::Rax, Reg::Rcx);
                    self.buf.store_mem(Reg::Rbp, off_d, Reg::Rax, bytes);
                } else {
                    self.buf.load_mem(Reg::Rax, sbase, off_s, esz.clamp(1, 8), is_signed_int(self.env, elem));
                    match op {
                        UnaryOp::Neg => self.buf.neg_r(Reg::Rax),
                        UnaryOp::Not => self.buf.xor_ri32(Reg::Rax, 1),
                        UnaryOp::BitNot => self.buf.not_r(Reg::Rax),
                        other => return Err(format!("ICE [codegen_x64]: unsupported vec unary op {other:?}")),
                    }
                    self.emit_norm_subword_int(Reg::Rax, elem);
                    self.buf.store_mem(Reg::Rbp, off_d, Reg::Rax, esz.clamp(1, 8));
                }
            }
            return Ok(());
        }
        if let Ty::Float(bits) = self.env.get(ty) {
            if op != UnaryOp::Neg {
                return Err(format!(
                    "ICE [codegen_x64]: unsupported float unary op {:?} in `{}`",
                    op,
                    self.interner.get(self.f.name)
                ));
            }
            let (sbase, off) = self.val_mem(operand);
            let doff = self.val_stack_off(dst);
            match *bits {
                16 => {
                    self.buf.load_mem(Reg::Rax, sbase, off, 2, false);
                    self.buf.mov_ri64(Reg::Rcx, 0x8000);
                    self.buf.xor_rr(Reg::Rax, Reg::Rcx);
                    self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 2);
                }
                32 => {
                    self.buf.load_mem(Reg::Rax, sbase, off, 4, false);
                    self.buf.mov_ri64(Reg::Rcx, 1u64 << 31);
                    self.buf.xor_rr(Reg::Rax, Reg::Rcx);
                    self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                }
                _ => {
                    self.buf.load_mem(Reg::Rax, sbase, off, 8, false);
                    self.buf.mov_ri64(Reg::Rcx, 1u64 << 63);
                    self.buf.xor_rr(Reg::Rax, Reg::Rcx);
                    self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                }
            }
            return Ok(());
        }
        let sz = x64_layout(self.env, ty).size.clamp(1, 8);
        self.load_scalar_val(Reg::Rax, operand);
        match op {
            UnaryOp::Neg => self.buf.neg_r(Reg::Rax),
            UnaryOp::Not => self.buf.xor_ri32(Reg::Rax, 1),
            UnaryOp::BitNot => self.buf.not_r(Reg::Rax),
            other => {
                return Err(format!(
                    "ICE [codegen_x64]: unsupported scalar unary op {:?} in `{}`",
                    other,
                    self.interner.get(self.f.name)
                ));
            }
        }
        self.emit_norm_subword_int(Reg::Rax, ty);
        self.store_scalar_val(dst, Reg::Rax, sz);
        Ok(())
    }

    fn emit_binary(
        &mut self,
        dst: EirValue,
        op: BinaryOp,
        lhs: EirValue,
        rhs: EirValue,
        op_ty: TyId,
        dst_ty: TyId,
    ) -> Result<(), String> {
        // SIMD Vector operations (`vec[T, N]`, with native SSE2 packed fast paths)
        if let Ty::Vec { elem, len } = self.env.get(op_ty).clone() {
            let esz = x64_layout(self.env, elem).size.max(1);
            let (lbase, loff) = self.val_mem(lhs);
            let (rbase, roff) = self.val_mem(rhs);
            // SSE2 packed fast path: `f32x4` (`addps`/`subps`/`mulps`/`divps`)
            if elem == TyId::F32 && len == 4 && matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div) {
                let doff = self.val_stack_off(dst);
                let opcode = match op {
                    BinaryOp::Add => 0x58,
                    BinaryOp::Sub => 0x5C,
                    BinaryOp::Mul => 0x59,
                    _ => 0x5E,
                };
                self.buf.movups_load(Xmm::Xmm0, lbase, loff);
                self.buf.movups_load(Xmm::Xmm1, rbase, roff);
                self.buf.sse_ps_bin(opcode, Xmm::Xmm0, Xmm::Xmm1);
                self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                return Ok(());
            }
            // SSE2 packed fast path: `f64x2` (`addpd`/`subpd`/`mulpd`/`divpd`)
            if elem == TyId::F64 && len == 2 && matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div) {
                let doff = self.val_stack_off(dst);
                let opcode = match op {
                    BinaryOp::Add => 0x58,
                    BinaryOp::Sub => 0x5C,
                    BinaryOp::Mul => 0x59,
                    _ => 0x5E,
                };
                self.buf.movups_load(Xmm::Xmm0, lbase, loff);
                self.buf.movups_load(Xmm::Xmm1, rbase, roff);
                self.buf.sse_66_bin(opcode, Xmm::Xmm0, Xmm::Xmm1);
                self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                return Ok(());
            }
            // SSE2 packed fast path: 128-bit integer vectors (`i32x4`/`u32x4`/`u8x16`/`i8x16`)
            if esz * (len as usize) == 16
                && matches!(elem, TyId::I32 | TyId::U32 | TyId::I8 | TyId::U8)
                && matches!(
                    op,
                    BinaryOp::Add | BinaryOp::AddWrap | BinaryOp::Sub | BinaryOp::SubWrap | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                )
            {
                let doff = self.val_stack_off(dst);
                let opcode = match (op, esz) {
                    (BinaryOp::Add | BinaryOp::AddWrap, 4) => 0xFE, // paddd
                    (BinaryOp::Sub | BinaryOp::SubWrap, 4) => 0xFA, // psubd
                    (BinaryOp::Add | BinaryOp::AddWrap, 1) => 0xFC, // paddb
                    (BinaryOp::Sub | BinaryOp::SubWrap, 1) => 0xF8, // psubb
                    (BinaryOp::BitAnd, _) => 0xDB,                  // pand
                    (BinaryOp::BitOr, _) => 0xEB,                   // por
                    _ => 0xEF,                                      // pxor
                };
                self.buf.movups_load(Xmm::Xmm0, lbase, loff);
                self.buf.movups_load(Xmm::Xmm1, rbase, roff);
                self.buf.sse_66_bin(opcode, Xmm::Xmm0, Xmm::Xmm1);
                self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                return Ok(());
            }
            if matches!(op, BinaryOp::Eq | BinaryOp::NotEq) && dst_ty == TyId::BOOL {
                let l_diff = self.buf.new_label();
                let l_done = self.buf.new_label();
                for lane in 0..(len as usize) {
                    let lo = loff + (lane * esz) as i32;
                    let ro = roff + (lane * esz) as i32;
                    if let Ty::Float(bits) = self.env.get(elem) {
                        let is_f16 = *bits == 16;
                        let is_f64 = *bits > 32;
                        if is_f16 {
                            self.buf.load_mem(Reg::Rdi, rbase, ro, 2, false);
                            self.helper_call("eon_f16_to_f32");
                            self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                            self.buf.load_mem(Reg::Rdi, lbase, lo, 2, false);
                            self.helper_call("eon_f16_to_f32");
                            self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                            self.buf.ucomi(false, Xmm::Xmm0, Xmm::Xmm1);
                        } else if is_f64 {
                            self.buf.movsd_load(Xmm::Xmm0, lbase, lo);
                            self.buf.movsd_load(Xmm::Xmm1, rbase, ro);
                            self.buf.ucomi(true, Xmm::Xmm0, Xmm::Xmm1);
                        } else {
                            self.buf.movss_load(Xmm::Xmm0, lbase, lo);
                            self.buf.movss_load(Xmm::Xmm1, rbase, ro);
                            self.buf.ucomi(false, Xmm::Xmm0, Xmm::Xmm1);
                        }
                        self.buf.jcc_label(Cond::P, l_diff);
                        self.buf.jcc_label(Cond::Ne, l_diff);
                    } else {
                        let sgn = is_signed_int(self.env, elem);
                        self.buf.load_mem(Reg::Rax, lbase, lo, esz.clamp(1, 8), sgn);
                        self.buf.load_mem(Reg::Rcx, rbase, ro, esz.clamp(1, 8), sgn);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.buf.jcc_label(Cond::Ne, l_diff);
                    }
                }
                self.buf.mov_ri64(Reg::Rax, u64::from(op == BinaryOp::Eq));
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_diff);
                self.buf.mov_ri64(Reg::Rax, u64::from(op == BinaryOp::NotEq));
                self.buf.bind_label(l_done);
                self.store_scalar_val(dst, Reg::Rax, 1);
                return Ok(());
            }
            let doff = self.val_stack_off(dst);
            let sgn = is_signed_int(self.env, elem);
            for lane in 0..(len as usize) {
                let lo = loff + (lane * esz) as i32;
                let ro = roff + (lane * esz) as i32;
                let dd = doff + (lane * esz) as i32;
                if let Ty::Float(bits) = self.env.get(elem) {
                    let is_f16 = *bits == 16;
                    let is_f64 = *bits > 32;
                    if is_f16 {
                        self.buf.load_mem(Reg::Rdi, rbase, ro, 2, false);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                        self.buf.load_mem(Reg::Rdi, lbase, lo, 2, false);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                    } else if is_f64 {
                        self.buf.movsd_load(Xmm::Xmm0, lbase, lo);
                        self.buf.movsd_load(Xmm::Xmm1, rbase, ro);
                    } else {
                        self.buf.movss_load(Xmm::Xmm0, lbase, lo);
                        self.buf.movss_load(Xmm::Xmm1, rbase, ro);
                    }
                    let opcode = match op {
                        BinaryOp::Add => 0x58,
                        BinaryOp::Sub => 0x5C,
                        BinaryOp::Mul => 0x59,
                        BinaryOp::Div => 0x5E,
                        other => return Err(format!("ICE [codegen_x64]: unsupported float vec op {other:?}")),
                    };
                    self.buf.sse_bin(!is_f16 && is_f64, opcode, Xmm::Xmm0, Xmm::Xmm1);
                    if is_f16 {
                        self.helper_call("eon_f32_to_f16");
                        self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, 2);
                    } else if is_f64 {
                        self.buf.movsd_store(Reg::Rbp, dd, Xmm::Xmm0);
                    } else {
                        self.buf.movss_store(Reg::Rbp, dd, Xmm::Xmm0);
                    }
                } else {
                    let lane_sz = esz.clamp(1, 8);
                    self.buf.load_mem(Reg::Rax, lbase, lo, lane_sz, sgn);
                    self.buf.load_mem(Reg::Rcx, rbase, ro, lane_sz, sgn);
                    match op {
                        BinaryOp::Add | BinaryOp::AddWrap => self.buf.add_rr(Reg::Rax, Reg::Rcx),
                        BinaryOp::Sub | BinaryOp::SubWrap => self.buf.sub_rr(Reg::Rax, Reg::Rcx),
                        BinaryOp::Mul | BinaryOp::MulWrap => self.buf.imul_rr(Reg::Rax, Reg::Rcx),
                        BinaryOp::Div => {
                            if sgn { self.buf.cqo(); self.buf.idiv_r(Reg::Rcx); }
                            else { self.buf.xor_rr32(Reg::Rdx, Reg::Rdx); self.buf.div_r(Reg::Rcx); }
                        }
                        BinaryOp::Mod => {
                            if sgn { self.buf.cqo(); self.buf.idiv_r(Reg::Rcx); }
                            else { self.buf.xor_rr32(Reg::Rdx, Reg::Rdx); self.buf.div_r(Reg::Rcx); }
                            self.buf.mov_rr(Reg::Rax, Reg::Rdx);
                        }
                        BinaryOp::BitAnd | BinaryOp::And => self.buf.and_rr(Reg::Rax, Reg::Rcx),
                        BinaryOp::BitOr | BinaryOp::Or => self.buf.or_rr(Reg::Rax, Reg::Rcx),
                        BinaryOp::BitXor => self.buf.xor_rr(Reg::Rax, Reg::Rcx),
                        other => return Err(format!("ICE [codegen_x64]: unsupported int vec op {other:?}")),
                    }
                    self.emit_norm_subword_int(Reg::Rax, elem);
                    self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, lane_sz);
                }
            }
            return Ok(());
        }

        // Float operations
        if let Ty::Float(bits) = self.env.get(op_ty) {
            let is_f16 = *bits == 16;
            let is_f64 = *bits > 32;
            if is_f16 {
                // Convert rhs -> f32 into temp stack slot, then lhs -> f32 in xmm0, load rhs into xmm1.
                self.load_scalar_raw(Reg::Rdi, rhs, 2);
                self.helper_call("eon_f16_to_f32");
                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                self.load_scalar_raw(Reg::Rdi, lhs, 2);
                self.helper_call("eon_f16_to_f32");
                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
            } else {
                self.load_float_val(Xmm::Xmm0, lhs, is_f64);
                self.load_float_val(Xmm::Xmm1, rhs, is_f64);
            }
            let sse_is_f64 = !is_f16 && is_f64;
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    let opcode = match op {
                        BinaryOp::Add => 0x58,
                        BinaryOp::Sub => 0x5C,
                        BinaryOp::Mul => 0x59,
                        _ => 0x5E,
                    };
                    self.buf.sse_bin(sse_is_f64, opcode, Xmm::Xmm0, Xmm::Xmm1);
                    if is_f16 {
                        self.helper_call("eon_f32_to_f16");
                        self.store_scalar_val(dst, Reg::Rax, 2);
                    } else {
                        self.store_float_val(dst, Xmm::Xmm0, is_f64);
                    }
                }
                BinaryOp::Eq => {
                    // IEEE-754 ordered equality: ZF=1 && PF=0 (NaN == NaN is false)
                    self.buf.ucomi(sse_is_f64, Xmm::Xmm0, Xmm::Xmm1);
                    self.buf.setcc(Cond::E, Reg::Rax);
                    self.buf.setcc(Cond::Np, Reg::Rcx);
                    self.buf.and_rr(Reg::Rax, Reg::Rcx);
                    self.store_scalar_val(dst, Reg::Rax, 1);
                }
                BinaryOp::NotEq => {
                    // IEEE-754 unordered inequality: ZF=0 || PF=1 (NaN != NaN is true)
                    self.buf.ucomi(sse_is_f64, Xmm::Xmm0, Xmm::Xmm1);
                    self.buf.setcc(Cond::Ne, Reg::Rax);
                    self.buf.setcc(Cond::P, Reg::Rcx);
                    self.buf.or_rr(Reg::Rax, Reg::Rcx);
                    self.store_scalar_val(dst, Reg::Rax, 1);
                }
                BinaryOp::Lt | BinaryOp::LtEq => {
                    // Swap operands so `ucomi(rhs, lhs)` with `A`/`Ae` returns false on NaN (CF=1).
                    self.buf.ucomi(sse_is_f64, Xmm::Xmm1, Xmm::Xmm0);
                    self.buf.setcc(if op == BinaryOp::Lt { Cond::A } else { Cond::Ae }, Reg::Rax);
                    self.store_scalar_val(dst, Reg::Rax, 1);
                }
                BinaryOp::Gt | BinaryOp::GtEq => {
                    self.buf.ucomi(sse_is_f64, Xmm::Xmm0, Xmm::Xmm1);
                    self.buf.setcc(if op == BinaryOp::Gt { Cond::A } else { Cond::Ae }, Reg::Rax);
                    self.store_scalar_val(dst, Reg::Rax, 1);
                }
                other => {
                    return Err(format!(
                        "ICE [codegen_x64]: unsupported float binary op {:?} in `{}`",
                        other,
                        self.interner.get(self.f.name)
                    ));
                }
            }
            return Ok(());
        }

        // GenId / Link equality comparison (8 bytes)
        if matches!(op, BinaryOp::Eq | BinaryOp::NotEq)
            && matches!(self.env.get(op_ty), Ty::Builtin(BuiltinKind::GenId | BuiltinKind::Link, _))
        {
            let (lb, lo) = self.val_mem(lhs);
            let (rb, ro) = self.val_mem(rhs);
            let sz = x64_layout(self.env, op_ty).size.clamp(4, 8);
            self.buf.load_mem(Reg::Rax, lb, lo, sz, false);
            self.buf.load_mem(Reg::Rcx, rb, ro, sz, false);
            self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
            self.buf.setcc(if op == BinaryOp::Eq { Cond::E } else { Cond::Ne }, Reg::Rax);
            self.store_scalar_val(dst, Reg::Rax, 1);
            return Ok(());
        }

        let signed = is_signed_int(self.env, op_ty);
        let bits = int_bits(self.env, op_ty);
        let sz = x64_layout(self.env, op_ty).size.clamp(1, 8);

        // Checked arithmetic returning ErrorUnion { err, ok }
        if matches!(
            op,
            BinaryOp::AddCheck | BinaryOp::SubCheck | BinaryOp::MulCheck | BinaryOp::DivCheck | BinaryOp::ModCheck
        ) {
            self.emit_checked_binary(dst, op, lhs, rhs, op_ty, dst_ty, signed, bits)?;
            return Ok(());
        }

        // Saturating arithmetic
        if matches!(op, BinaryOp::AddSat | BinaryOp::SubSat | BinaryOp::MulSat) {
            self.emit_sat_binary(dst, op, lhs, rhs, signed, bits, sz);
            return Ok(());
        }

        self.load_scalar_val(Reg::Rax, lhs);
        self.load_scalar_val(Reg::Rcx, rhs);

        match op {
            BinaryOp::Add | BinaryOp::AddWrap => {
                self.buf.add_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Sub | BinaryOp::SubWrap => {
                self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Mul | BinaryOp::MulWrap => {
                self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Div => {
                if signed {
                    self.buf.cqo();
                    self.buf.idiv_r(Reg::Rcx);
                } else {
                    self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                    self.buf.div_r(Reg::Rcx);
                }
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Mod => {
                if signed {
                    self.buf.cqo();
                    self.buf.idiv_r(Reg::Rcx);
                } else {
                    self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                    self.buf.div_r(Reg::Rcx);
                }
                self.emit_norm_subword_int(Reg::Rdx, op_ty);
                self.store_scalar_val(dst, Reg::Rdx, sz);
            }
            BinaryOp::BitAnd | BinaryOp::And => {
                self.buf.and_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::BitOr | BinaryOp::Or => {
                self.buf.or_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::BitXor => {
                self.buf.xor_rr(Reg::Rax, Reg::Rcx);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Shl | BinaryOp::ShlWrap => {
                if op == BinaryOp::ShlWrap && bits < 64 {
                    self.buf.and_ri32(Reg::Rcx, (bits.saturating_sub(1)) as i32);
                }
                self.buf.shl_cl(Reg::Rax);
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Shr => {
                if signed {
                    self.buf.sar_cl(Reg::Rax);
                } else {
                    self.buf.shr_cl(Reg::Rax);
                }
                self.emit_norm_subword_int(Reg::Rax, op_ty);
                self.store_scalar_val(dst, Reg::Rax, sz);
            }
            BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                let cc = match (op, signed) {
                    (BinaryOp::Eq, _) => Cond::E,
                    (BinaryOp::NotEq, _) => Cond::Ne,
                    (BinaryOp::Lt, true) => Cond::L,
                    (BinaryOp::Lt, false) => Cond::B,
                    (BinaryOp::LtEq, true) => Cond::Le,
                    (BinaryOp::LtEq, false) => Cond::Be,
                    (BinaryOp::Gt, true) => Cond::G,
                    (BinaryOp::Gt, false) => Cond::A,
                    (BinaryOp::GtEq, true) => Cond::Ge,
                    (BinaryOp::GtEq, false) => Cond::Ae,
                    _ => Cond::E,
                };
                self.buf.setcc(cc, Reg::Rax);
                self.store_scalar_val(dst, Reg::Rax, 1);
            }
            other => {
                return Err(format!(
                    "ICE [codegen_x64]: unsupported binary op {:?} in `{}`",
                    other,
                    self.interner.get(self.f.name)
                ));
            }
        }
        Ok(())
    }

    fn emit_checked_binary(
        &mut self,
        dst: EirValue,
        op: BinaryOp,
        lhs: EirValue,
        rhs: EirValue,
        op_ty: TyId,
        _dst_ty: TyId,
        signed: bool,
        bits: u8,
    ) -> Result<(), String> {
        let doff = self.val_stack_off(dst);
        let ok_off = eu_ok_offset(self.env, op_ty) as i32;
        let ok_sz = x64_layout(self.env, op_ty).size.clamp(1, 8);
        self.load_scalar_val(Reg::Rax, lhs);
        self.load_scalar_val(Reg::Rcx, rhs);

        let l_ovf = self.buf.new_label();
        let l_divz = self.buf.new_label();
        let l_ok = self.buf.new_label();
        let l_done = self.buf.new_label();

        match op {
            BinaryOp::DivCheck | BinaryOp::ModCheck => {
                self.buf.test_rr(Reg::Rcx, Reg::Rcx);
                self.buf.jcc_label(Cond::E, l_divz);
                if signed {
                    let (min_v, _) = int_type_bounds(bits, true);
                    let l_safe = self.buf.new_label();
                    self.buf.cmp_ri32(Reg::Rcx, -1);
                    self.buf.jcc_label(Cond::Ne, l_safe);
                    self.buf.mov_ri64(Reg::Rdx, min_v);
                    self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                    self.buf.jcc_label(Cond::E, l_ovf);
                    self.buf.bind_label(l_safe);
                    self.buf.cqo();
                    self.buf.idiv_r(Reg::Rcx);
                } else {
                    self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                    self.buf.div_r(Reg::Rcx);
                }
                if op == BinaryOp::ModCheck {
                    self.buf.mov_rr(Reg::Rax, Reg::Rdx);
                }
                self.buf.jmp_label(l_ok);
            }
            BinaryOp::AddCheck | BinaryOp::SubCheck | BinaryOp::MulCheck => {
                if bits < 64 {
                    if !signed && op == BinaryOp::SubCheck {
                        self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                        self.buf.jcc_label(Cond::B, l_ovf);
                        self.buf.jmp_label(l_ok);
                    } else {
                        // Exact wider 64-bit computation + range check!
                        match op {
                            BinaryOp::AddCheck => self.buf.add_rr(Reg::Rax, Reg::Rcx),
                            BinaryOp::SubCheck => self.buf.sub_rr(Reg::Rax, Reg::Rcx),
                            _ => self.buf.imul_rr(Reg::Rax, Reg::Rcx),
                        }
                        let (lo, hi) = int_type_bounds(bits, signed);
                        if signed {
                            self.buf.mov_ri64(Reg::Rdx, lo);
                            self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                            self.buf.jcc_label(Cond::L, l_ovf);
                        }
                        self.buf.mov_ri64(Reg::Rdx, hi);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                        self.buf.jcc_label(if signed { Cond::G } else { Cond::A }, l_ovf);
                        self.buf.jmp_label(l_ok);
                    }
                } else {
                    match op {
                        BinaryOp::AddCheck => {
                            self.buf.add_rr(Reg::Rax, Reg::Rcx);
                            self.buf.jcc_label(if signed { Cond::O } else { Cond::B }, l_ovf);
                        }
                        BinaryOp::SubCheck => {
                            self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                            self.buf.jcc_label(if signed { Cond::O } else { Cond::B }, l_ovf);
                        }
                        _ => {
                            if signed {
                                self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                                self.buf.jcc_label(Cond::O, l_ovf);
                            } else {
                                self.buf.mul_r(Reg::Rcx);
                                self.buf.jcc_label(Cond::O, l_ovf);
                            }
                        }
                    }
                    self.buf.jmp_label(l_ok);
                }
            }
            other => {
                return Err(format!(
                    "ICE [codegen_x64]: unexpected checked binary op {:?} in `{}`",
                    other,
                    self.interner.get(self.f.name)
                ));
            }
        }

        self.buf.bind_label(l_divz);
        self.buf.mov_ri64(Reg::Rdx, 2);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rdx, ok_sz);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_ovf);
        self.buf.mov_ri64(Reg::Rdx, 1);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rdx, ok_sz);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_ok);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, ok_sz);

        self.buf.bind_label(l_done);
        Ok(())
    }

    fn emit_sat_binary(
        &mut self,
        dst: EirValue,
        op: BinaryOp,
        lhs: EirValue,
        rhs: EirValue,
        signed: bool,
        bits: u8,
        sz: usize,
    ) {
        self.load_scalar_val(Reg::Rax, lhs);
        self.load_scalar_val(Reg::Rcx, rhs);
        let (lo, hi) = int_type_bounds(bits, signed);

        if bits < 64 {
            if !signed && op == BinaryOp::SubSat {
                self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                self.buf.cmovcc(Cond::B, Reg::Rax, Reg::Rdx);
            } else {
                match op {
                    BinaryOp::AddSat => self.buf.add_rr(Reg::Rax, Reg::Rcx),
                    BinaryOp::SubSat => self.buf.sub_rr(Reg::Rax, Reg::Rcx),
                    _ => self.buf.imul_rr(Reg::Rax, Reg::Rcx),
                }
                if signed {
                    self.buf.mov_ri64(Reg::Rdx, lo);
                    self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                    self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                }
                self.buf.mov_ri64(Reg::Rdx, hi);
                self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                self.buf.cmovcc(if signed { Cond::G } else { Cond::A }, Reg::Rax, Reg::Rdx);
            }
        } else if !signed {
            match op {
                BinaryOp::AddSat => {
                    self.buf.mov_ri64(Reg::Rdx, u64::MAX);
                    self.buf.add_rr(Reg::Rax, Reg::Rcx);
                    self.buf.cmovcc(Cond::B, Reg::Rax, Reg::Rdx);
                }
                BinaryOp::SubSat => {
                    self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                    self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                    self.buf.cmovcc(Cond::B, Reg::Rax, Reg::Rdx);
                }
                _ => {
                    self.buf.mul_r(Reg::Rcx);
                    self.buf.mov_ri64(Reg::Rdx, u64::MAX);
                    self.buf.cmovcc(Cond::O, Reg::Rax, Reg::Rdx);
                }
            }
        } else {
            // 64-bit signed saturating
            let l_done = self.buf.new_label();
            self.buf.mov_rr(Reg::R8, Reg::Rax);
            match op {
                BinaryOp::AddSat => {
                    self.buf.add_rr(Reg::Rax, Reg::Rcx);
                    self.buf.jcc_label(Cond::No, l_done);
                    self.buf.mov_ri64(Reg::Rax, i64::MAX as u64);
                    self.buf.mov_ri64(Reg::Rdx, i64::MIN as u64);
                    self.buf.cmp_ri32(Reg::R8, 0);
                    self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                }
                BinaryOp::SubSat => {
                    self.buf.sub_rr(Reg::Rax, Reg::Rcx);
                    self.buf.jcc_label(Cond::No, l_done);
                    self.buf.mov_ri64(Reg::Rax, i64::MAX as u64);
                    self.buf.mov_ri64(Reg::Rdx, i64::MIN as u64);
                    self.buf.cmp_ri32(Reg::R8, 0);
                    self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                }
                _ => {
                    self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                    self.buf.jcc_label(Cond::No, l_done);
                    self.buf.xor_rr(Reg::R8, Reg::Rcx);
                    self.buf.mov_ri64(Reg::Rax, i64::MAX as u64);
                    self.buf.mov_ri64(Reg::Rdx, i64::MIN as u64);
                    self.buf.cmp_ri32(Reg::R8, 0);
                    self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                }
            }
            self.buf.bind_label(l_done);
        }
        self.store_scalar_val(dst, Reg::Rax, sz);
    }

    fn emit_cast(&mut self, dst: EirValue, kind: CastKind, operand: EirValue, from_ty: TyId, to_ty: TyId) -> Result<(), String> {
        match kind {
            CastKind::Widen | CastKind::Trunc | CastKind::AsU64 | CastKind::MutToConst => {
                if let (Ty::Float(fb), Ty::Float(tb)) = (self.env.get(from_ty), self.env.get(to_ty)) {
                    let (fb, tb) = (*fb, *tb);
                    if fb == tb {
                        self.copy_val_to_val(dst, operand);
                    } else {
                        match fb {
                            16 => {
                                self.load_scalar_raw(Reg::Rdi, operand, 2);
                                self.helper_call("eon_f16_to_f32");
                                if tb == 64 {
                                    self.buf.cvtss2sd(Xmm::Xmm0, Xmm::Xmm0);
                                    self.store_float_val(dst, Xmm::Xmm0, true);
                                } else {
                                    self.store_float_val(dst, Xmm::Xmm0, false);
                                }
                            }
                            32 => {
                                self.load_float_val(Xmm::Xmm0, operand, false);
                                if tb == 16 {
                                    self.helper_call("eon_f32_to_f16");
                                    self.store_scalar_val(dst, Reg::Rax, 2);
                                } else {
                                    self.buf.cvtss2sd(Xmm::Xmm0, Xmm::Xmm0);
                                    self.store_float_val(dst, Xmm::Xmm0, true);
                                }
                            }
                            _ => {
                                self.load_float_val(Xmm::Xmm0, operand, true);
                                self.buf.cvtsd2ss(Xmm::Xmm0, Xmm::Xmm0);
                                if tb == 16 {
                                    self.helper_call("eon_f32_to_f16");
                                    self.store_scalar_val(dst, Reg::Rax, 2);
                                } else {
                                    self.store_float_val(dst, Xmm::Xmm0, false);
                                }
                            }
                        }
                    }
                } else if x64_layout(self.env, from_ty).size == x64_layout(self.env, to_ty).size
                    && x64_layout(self.env, to_ty).size > 8
                {
                    self.copy_val_to_val(dst, operand);
                } else {
                    self.load_scalar_val(Reg::Rax, operand);
                    self.emit_norm_subword_int(Reg::Rax, to_ty);
                    let to_sz = x64_layout(self.env, to_ty).size.clamp(1, 8);
                    self.store_scalar_val(dst, Reg::Rax, to_sz);
                }
            }
            CastKind::Clamp => {
                self.load_scalar_val(Reg::Rax, operand);
                let from_signed = is_signed_int(self.env, from_ty);
                let to_signed = is_signed_int(self.env, to_ty);
                let to_bits = int_bits(self.env, to_ty);
                let (lo, hi) = int_type_bounds(to_bits, to_signed);
                match (from_signed, to_signed) {
                    (true, true) => {
                        self.buf.mov_ri64(Reg::Rdx, lo);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                        self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                        self.buf.mov_ri64(Reg::Rdx, hi);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                        self.buf.cmovcc(Cond::G, Reg::Rax, Reg::Rdx);
                    }
                    (true, false) => {
                        // Negative values clamp to 0; non-negative values then compare unsigned vs `hi`.
                        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                        self.buf.cmp_ri32(Reg::Rax, 0);
                        self.buf.cmovcc(Cond::L, Reg::Rax, Reg::Rdx);
                        self.buf.mov_ri64(Reg::Rdx, hi);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                        self.buf.cmovcc(Cond::A, Reg::Rax, Reg::Rdx);
                    }
                    (false, true) | (false, false) => {
                        // Unsigned source is always >= 0 >= lo, so only upper-bound clamp is needed.
                        self.buf.mov_ri64(Reg::Rdx, hi);
                        self.buf.cmp_rr(Reg::Rax, Reg::Rdx);
                        self.buf.cmovcc(Cond::A, Reg::Rax, Reg::Rdx);
                    }
                }
                self.emit_norm_subword_int(Reg::Rax, to_ty);
                let to_sz = x64_layout(self.env, to_ty).size.clamp(1, 8);
                self.store_scalar_val(dst, Reg::Rax, to_sz);
            }
            CastKind::ToFloat => {
                self.load_scalar_val(Reg::Rax, operand);
                match self.env.get(to_ty) {
                    Ty::Float(16) => {
                        self.buf.cvtsi2f(false, Xmm::Xmm0, Reg::Rax);
                        self.helper_call("eon_f32_to_f16");
                        self.store_scalar_val(dst, Reg::Rax, 2);
                    }
                    Ty::Float(64) => {
                        self.buf.cvtsi2f(true, Xmm::Xmm0, Reg::Rax);
                        self.store_float_val(dst, Xmm::Xmm0, true);
                    }
                    _ => {
                        self.buf.cvtsi2f(false, Xmm::Xmm0, Reg::Rax);
                        self.store_float_val(dst, Xmm::Xmm0, false);
                    }
                }
            }
            CastKind::TruncToInt => {
                match self.env.get(from_ty) {
                    Ty::Float(16) => {
                        self.load_scalar_raw(Reg::Rdi, operand, 2);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.cvttf2si(false, Reg::Rax, Xmm::Xmm0);
                    }
                    Ty::Float(64) => {
                        self.load_float_val(Xmm::Xmm0, operand, true);
                        self.buf.cvttf2si(true, Reg::Rax, Xmm::Xmm0);
                    }
                    _ => {
                        self.load_float_val(Xmm::Xmm0, operand, false);
                        self.buf.cvttf2si(false, Reg::Rax, Xmm::Xmm0);
                    }
                }
                self.emit_norm_subword_int(Reg::Rax, to_ty);
                let to_sz = x64_layout(self.env, to_ty).size.clamp(1, 8);
                self.store_scalar_val(dst, Reg::Rax, to_sz);
            }
            CastKind::Bitcast => {
                let sz = x64_layout(self.env, to_ty).size;
                if sz <= 8 {
                    self.load_scalar_raw(Reg::Rax, operand, sz);
                    self.store_scalar_val(dst, Reg::Rax, sz);
                } else {
                    self.copy_val_to_val(dst, operand);
                }
            }
            CastKind::WrapOptional => {
                let inner = match self.env.get(to_ty) { Ty::Optional(i) => *i, _ => from_ty };
                if is_niche_id_opt(self.env, inner) {
                    let sz = x64_layout(self.env, from_ty)
                        .size
                        .max(x64_layout(self.env, to_ty).size);
                    self.copy_val_to_val_sz(dst, operand, sz);
                } else {
                    let doff = self.val_stack_off(dst);
                    self.buf.mov_ri64(Reg::Rax, 1);
                    self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 1);
                    let voff = opt_val_offset(self.env, inner) as i32;
                    self.store_val_to_mem(Reg::Rbp, doff + voff, operand, inner);
                }
            }
            CastKind::WrapOk => {
                let doff = self.val_stack_off(dst);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                let ok_ty = match self.env.get(to_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => from_ty };
                if ok_ty != TyId::VOID && ok_ty != TyId::NEVER {
                    let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                    self.store_val_to_mem(Reg::Rbp, doff + ok_off, operand, ok_ty);
                }
            }
            CastKind::WrapErr => {
                let doff = self.val_stack_off(dst);
                let sz = x64_layout(self.env, to_ty).size;
                self.emit_memzero(Reg::Rbp, doff, sz);
                self.load_scalar_raw(Reg::Rax, operand, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
            }
            CastKind::ArrayBorrowToSlice => {
                let arr_len = match self.env.get(from_ty) {
                    Ty::Borrow { inner: BorrowInner::Single(at), .. } => match self.env.get(*at) {
                        Ty::Array { len, .. } => *len,
                        _ => 0,
                    },
                    _ => 0,
                };
                self.load_scalar_val(Reg::Rax, operand);
                let doff = self.val_stack_off(dst);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.mov_ri64(Reg::Rax, arr_len);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rax, 8);
            }
        }
        Ok(())
    }

    fn emit_slice(
        &mut self,
        dst: EirValue,
        base: EirValue,
        base_ty: TyId,
        lo: Option<EirValue>,
        hi: Option<EirValue>,
        dst_ty: TyId,
    ) {
        let elem_sz = match self.env.get(dst_ty) {
            Ty::Borrow { inner: BorrowInner::Slice(el), .. } => x64_layout(self.env, *el).size.max(1),
            _ => 1,
        };
        // Load base_ptr -> R8, base_len -> R9
        match self.env.get(base_ty) {
            Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str => {
                let (bbase, boff) = self.val_mem(base);
                self.buf.load_mem(Reg::R8, bbase, boff, 8, false);
                self.buf.load_mem(Reg::R9, bbase, boff + 8, 8, false);
            }
            Ty::Borrow { inner: BorrowInner::Single(at), .. } => {
                self.load_scalar_val(Reg::R8, base);
                match self.env.get(*at) {
                    Ty::Array { len, .. } => self.buf.mov_ri64(Reg::R9, *len),
                    _ => {
                        self.buf.load_mem(Reg::R9, Reg::R8, 8, 8, false);
                        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                    }
                }
            }
            Ty::Array { len, .. } => {
                self.val_addr_into(Reg::R8, base);
                self.buf.mov_ri64(Reg::R9, *len);
            }
            _ => {
                let (bbase, boff) = self.val_mem(base);
                self.buf.load_mem(Reg::R8, bbase, boff, 8, false);
                self.buf.load_mem(Reg::R9, bbase, boff + 8, 8, false);
            }
        }
        // lo -> R10, hi -> R11
        if let Some(lv) = lo {
            self.load_scalar_val(Reg::R10, lv);
        } else {
            self.buf.xor_rr32(Reg::R10, Reg::R10);
        }
        if let Some(hv) = hi {
            self.load_scalar_val(Reg::R11, hv);
        } else {
            self.buf.mov_rr(Reg::R11, Reg::R9);
        }
        // new_len = hi - lo
        self.buf.sub_rr(Reg::R11, Reg::R10);
        // new_ptr = ptr + lo * elem_sz
        if elem_sz != 1 {
            self.buf.imul_rri32(Reg::R10, Reg::R10, elem_sz as i32);
        }
        self.buf.add_rr(Reg::R8, Reg::R10);
        let doff = self.val_stack_off(dst);
        self.buf.store_mem(Reg::Rbp, doff, Reg::R8, 8);
        self.buf.store_mem(Reg::Rbp, doff + 8, Reg::R11, 8);
    }

    fn emit_norm_subword_int(&mut self, reg: Reg, ty: TyId) {
        if let Ty::Int { signed, bits, is_ptr_sized } = self.env.get(ty)
            && !is_ptr_sized
            && *bits >= 1
            && *bits < 64
            && !matches!(*bits, 8 | 16 | 32)
        {
            if *signed {
                let sh = 64 - *bits;
                self.buf.shl_imm(reg, sh);
                self.buf.sar_imm(reg, sh);
            } else {
                let mask = (1u64 << *bits) - 1;
                if *bits <= 31 {
                    self.buf.and_ri32(reg, mask as i32);
                } else {
                    let scratch = if reg == Reg::Rdx { Reg::R11 } else { Reg::Rdx };
                    self.buf.mov_ri64(scratch, mask);
                    self.buf.and_rr(reg, scratch);
                }
            }
        }
    }

    fn split_bitfield_tail(&self, place: &EirPlace) -> Option<(DefId, usize, EirPlace)> {
        let last = place.projs.last()?;
        let EirProj::Field { field_idx, .. } = last else {
            return None;
        };
        let mut cur_ty = match place.base {
            EirPlaceBase::Slot(s) => self.f.slots[s.0 as usize].ty,
            EirPlaceBase::Val(v) => self.f.value_tys[v.0 as usize],
        };
        for pr in &place.projs[..place.projs.len() - 1] {
            cur_ty = match pr {
                EirProj::Deref => match self.env.get(cur_ty) {
                    Ty::Borrow { inner: BorrowInner::Single(t), .. }
                    | Ty::RawPtr { inner: t, .. } => *t,
                    _ => cur_ty,
                },
                EirProj::Field { ty, .. }
                | EirProj::TupleField { ty, .. }
                | EirProj::Index { ty, .. }
                | EirProj::PoolCol { elem_ty: ty, .. } => *ty,
            };
        }
        let deref_t = match self.env.get(cur_ty) {
            Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
            _ => cur_ty,
        };
        let Ty::Bitfield(bf_def) = self.env.get(deref_t) else {
            return None;
        };
        let mut base_projs = place.projs[..place.projs.len() - 1].to_vec();
        if matches!(self.env.get(cur_ty), Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }) {
            base_projs.push(EirProj::Deref);
        }
        Some((
            *bf_def,
            *field_idx,
            EirPlace {
                base: place.base.clone(),
                projs: base_projs,
                ty: deref_t,
            },
        ))
    }

    fn bitfield_member_layout(&self, bf_def: DefId, field_idx: usize) -> (usize, u8, u8) {
        let Some(meta) = self.env.bitfields.get(&bf_def) else {
            return (4, 0, 32);
        };
        let base_sz = x64_layout(self.env, meta.base_ty).size.clamp(1, 8);
        let (lo, hi) = meta
            .members
            .get(field_idx)
            .map(|m| (m.lo, m.hi))
            .unwrap_or((0, meta.base_bits));
        (base_sz, lo, hi.saturating_sub(lo).max(1))
    }

    fn emit_aggregate(&mut self, dst: EirValue, kind: &AggregateKind, elems: &[EirValue], dst_ty: TyId) {
        let lay = x64_layout(self.env, dst_ty);
        if let AggregateKind::Bitfield(def) = kind {
            let meta = self.env.bitfields.get(def).cloned();
            self.buf.xor_rr32(Reg::R10, Reg::R10);
            if let Some(m) = meta {
                for (i, &ev) in elems.iter().enumerate() {
                    if let Some(mb) = m.members.get(i) {
                        let width = mb.hi.saturating_sub(mb.lo).max(1);
                        let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                        self.load_scalar_val(Reg::Rax, ev);
                        if width < 64 {
                            self.buf.mov_ri64(Reg::Rcx, mask);
                            self.buf.and_rr(Reg::Rax, Reg::Rcx);
                        }
                        if mb.lo > 0 {
                            self.buf.shl_imm(Reg::Rax, mb.lo);
                        }
                        self.buf.or_rr(Reg::R10, Reg::Rax);
                    }
                }
            }
            self.store_scalar_val(dst, Reg::R10, lay.size.clamp(1, 8));
            return;
        }
        let doff = self.val_stack_off(dst);
        self.emit_memzero(Reg::Rbp, doff, lay.size);
        match kind {
            AggregateKind::Bitfield(_) => unreachable!(),
            AggregateKind::Struct(_) | AggregateKind::AnonStruct(_) | AggregateKind::Tuple => {
                for (i, &ev) in elems.iter().enumerate() {
                    let (foff, fty) = struct_field_offset(self.env, dst_ty, i);
                    self.store_val_to_mem(Reg::Rbp, doff + foff as i32, ev, fty);
                }
            }
            AggregateKind::PoolRow { def, is_slot_pool } => {
                let cols = self.env.pools.get(def).map(|m| m.columns.clone()).unwrap_or_default();
                for (i, &ev) in elems.iter().enumerate() {
                    let (foff, fty) = struct_field_offset(self.env, dst_ty, i);
                    let is_link = cols.get(i).is_some_and(|c| is_link_or_opt_link(self.env, c.ty));
                    let ev_ty = self.f.value_tys[ev.0 as usize];
                    let ev_sz = x64_layout(self.env, ev_ty).size;
                    if !*is_slot_pool && is_link && ev_sz == 4 {
                        // Store 4-byte Id into `.index` (offset +0) and 0 into `.gen` (offset +4).
                        self.load_scalar_raw(Reg::Rax, ev, 4);
                        self.buf.store_mem(Reg::Rbp, doff + foff as i32, Reg::Rax, 4);
                        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                        self.buf.store_mem(Reg::Rbp, doff + foff as i32 + 4, Reg::Rax, 4);
                    } else {
                        let store_ty = if *is_slot_pool && is_link { ev_ty } else { fty };
                        self.store_val_to_mem(Reg::Rbp, doff + foff as i32, ev, store_ty);
                    }
                }
            }
            AggregateKind::Array(elem_ty) | AggregateKind::Vec { elem: elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                for (i, &ev) in elems.iter().enumerate() {
                    self.store_val_to_mem(Reg::Rbp, doff + (i * esz) as i32, ev, *elem_ty);
                }
            }
            AggregateKind::ArrayRepeat { elem, count } => {
                if let Some(&ev) = elems.first() {
                    let esz = x64_layout(self.env, *elem).size.max(1);
                    if *count <= 16 {
                        for i in 0..(*count as usize) {
                            self.store_val_to_mem(Reg::Rbp, doff + (i * esz) as i32, ev, *elem);
                        }
                    } else {
                        // Loop `r10 = 0..count` storing to `[r8]`, `r8 += esz`.
                        self.buf.lea_mem(Reg::R8, Reg::Rbp, doff);
                        self.buf.mov_ri64(Reg::R10, *count);
                        let l_lp = self.buf.new_label();
                        self.buf.bind_label(l_lp);
                        self.store_val_to_mem(Reg::R8, 0, ev, *elem);
                        self.buf.add_ri32(Reg::R8, esz as i32);
                        self.buf.sub_ri32(Reg::R10, 1);
                        self.buf.jcc_label(Cond::Ne, l_lp);
                    }
                }
            }
            AggregateKind::UnionVariant { def, variant_idx } => {
                self.buf.mov_ri64(Reg::Rax, *variant_idx as u64);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                if let Some(&pv) = elems.first() {
                    let poff = union_payload_offset(self.env, *def) as i32;
                    let pty = self.f.value_tys[pv.0 as usize];
                    self.store_val_to_mem(Reg::Rbp, doff + poff, pv, pty);
                }
            }
        }
    }

    // ========================================================================
    // Call Lowering (`EonCall` ABI + `extern "C"` + `dyn` vtable dispatch)
    // ========================================================================

    fn emit_call(&mut self, dst: Option<EirValue>, callee: &EirCallee, args: &[EirValue], ret_ty: TyId) {
        let is_extern_c = match callee {
            EirCallee::ExternFn { .. } => true,
            EirCallee::DirectFn(d) => self.env.fns.get(d).is_some_and(|s| s.effects.is_export_c),
            _ => false,
        };
        let gp_regs: &[Reg] = if is_extern_c { &SYSV_GP_ARG_REGS } else { &EON_GP_ARG_REGS };
        let ret_cls = classify_ret(self.env, ret_ty, is_extern_c);

        // If callee is a Layout-Shared canonical function, check its actual param types so float/int matches!
        let target_sig: Option<&FnSig> = match callee {
            EirCallee::DirectFn(d) => self.eir.by_key.get(&CfgFnKey::TopDef(*d)).map(|&i| &self.eir.functions[i].sig),
            EirCallee::Method { impl_idx, sub_idx } => self
                .eir
                .by_key
                .get(&CfgFnKey::ImplMethod { impl_idx: *impl_idx, sub_idx: *sub_idx })
                .map(|&i| &self.eir.functions[i].sig),
            EirCallee::GenericInst { inst_id } => self
                .eir
                .by_key
                .get(&CfgFnKey::GenericInst(*inst_id))
                .map(|&i| &self.eir.functions[i].sig),
            _ => None,
        };

        // Plan all argument placements BEFORE loading into volatile registers!
        let mut gp_i = 0usize;
        let mut xmm_i = 0usize;
        let mut sret_target: Option<i32> = None;
        if ret_uses_sret(ret_cls) {
            let doff = dst.map_or(self.plan.temp_copy_off, |d| {
                let base = self.val_stack_off(d);
                match ret_cls {
                    RetClass::ErrorUnion { ok_ty, .. } => base + eu_ok_offset(self.env, ok_ty) as i32,
                    _ => base,
                }
            });
            sret_target = Some(doff);
            gp_i = 1;
        }

        // First copy any indirect (>16B) arguments into our staging area at `temp_copy_off`.
        let mut cur_tmp_off = self.plan.temp_copy_off;
        enum PreparedArg {
            Gp1 { reg: Reg, val: EirValue, sz: usize },
            Gp1Ptr { reg: Reg, stack_off: i32 },
            Gp1DynData { reg: Reg, fat_val: EirValue },
            Gp1ClosureEnv { reg: Reg, fat_val: EirValue },
            Gp2 { r0: Reg, r1: Reg, val: EirValue, sz: usize },
            Float { xmm: Xmm, val: EirValue, is_f64: bool },
            StackScalar { val: EirValue, sz: usize },
            StackPair { val: EirValue },
            StackPtr { stack_off: i32 },
        }
        let mut prepared: Vec<PreparedArg> = Vec::new();
        let is_dyn_call = matches!(callee, EirCallee::DynCall { .. });
        if let EirCallee::FnPtr(fpv) = callee {
            prepared.push(PreparedArg::Gp1ClosureEnv { reg: gp_regs[gp_i], fat_val: *fpv });
            gp_i += 1;
        }

        for (ai, &av) in args.iter().enumerate() {
            if is_dyn_call && ai == 0 {
                // arg 0 of DynCall is `*dyn Iface` `{ data_ptr, vtable_ptr }` -> pass `data_ptr` in `gp_regs[gp_i]`!
                prepared.push(PreparedArg::Gp1DynData { reg: gp_regs[gp_i], fat_val: av });
                gp_i += 1;
                continue;
            }
            let aty = target_sig
                .and_then(|s| s.params.get(ai))
                .map(|p| p.ty)
                .unwrap_or_else(|| self.f.value_tys[av.0 as usize]);
            let val_ty = self.f.value_tys[av.0 as usize];
            let actual_sz = x64_layout(self.env, val_ty).size;
            match classify_arg(self.env, aty) {
                ArgClass::Void => {}
                ArgClass::Float32 => {
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        prepared.push(PreparedArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], val: av, is_f64: false });
                        xmm_i += 1;
                    } else {
                        prepared.push(PreparedArg::StackScalar { val: av, sz: 4 });
                    }
                }
                ArgClass::Float64 => {
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        prepared.push(PreparedArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], val: av, is_f64: true });
                        xmm_i += 1;
                    } else {
                        prepared.push(PreparedArg::StackScalar { val: av, sz: 8 });
                    }
                }
                ArgClass::Reg1(sz) => {
                    if gp_i < gp_regs.len() {
                        prepared.push(PreparedArg::Gp1 { reg: gp_regs[gp_i], val: av, sz: actual_sz.min(sz).max(1) });
                        gp_i += 1;
                    } else {
                        prepared.push(PreparedArg::StackScalar { val: av, sz });
                    }
                }
                ArgClass::Reg2(sz) => {
                    if gp_i + 1 < gp_regs.len() {
                        prepared.push(PreparedArg::Gp2 { r0: gp_regs[gp_i], r1: gp_regs[gp_i + 1], val: av, sz });
                        gp_i += 2;
                    } else {
                        prepared.push(PreparedArg::StackPair { val: av });
                    }
                }
                ArgClass::Indirect(sz) => {
                    let copy_off = cur_tmp_off;
                    cur_tmp_off += align_up(sz, 16) as i32;
                    let (sbase, soff) = self.val_mem(av);
                    self.copy_mem_using_rax(Reg::Rbp, copy_off, sbase, soff, sz);
                    if gp_i < gp_regs.len() {
                        prepared.push(PreparedArg::Gp1Ptr { reg: gp_regs[gp_i], stack_off: copy_off });
                        gp_i += 1;
                    } else {
                        prepared.push(PreparedArg::StackPtr { stack_off: copy_off });
                    }
                }
            }
        }

        // Count stack args and maintain 16-byte stack alignment!
        let mut stack_slots: Vec<&PreparedArg> = Vec::new();
        for pa in &prepared {
            if matches!(pa, PreparedArg::StackScalar { .. } | PreparedArg::StackPair { .. } | PreparedArg::StackPtr { .. }) {
                stack_slots.push(pa);
            }
        }
        let stack_bytes: usize = stack_slots
            .iter()
            .map(|p| if matches!(p, PreparedArg::StackPair { .. }) { 16 } else { 8 })
            .sum();
        let aligned_stack_bytes = align_up(stack_bytes, 16);
        if aligned_stack_bytes > 0 {
            self.buf.sub_ri32(Reg::Rsp, aligned_stack_bytes as i32);
            let mut sp_off = 0i32;
            for p in &stack_slots {
                match p {
                    PreparedArg::StackScalar { val, sz } => {
                        self.load_scalar_raw(Reg::Rax, *val, *sz);
                        self.buf.store_mem(Reg::Rsp, sp_off, Reg::Rax, 8);
                        sp_off += 8;
                    }
                    PreparedArg::StackPair { val } => {
                        let (vbase, voff) = self.val_mem(*val);
                        self.buf.load_mem(Reg::Rax, vbase, voff, 8, false);
                        self.buf.store_mem(Reg::Rsp, sp_off, Reg::Rax, 8);
                        self.buf.load_mem(Reg::Rax, vbase, voff + 8, 8, false);
                        self.buf.store_mem(Reg::Rsp, sp_off + 8, Reg::Rax, 8);
                        sp_off += 16;
                    }
                    PreparedArg::StackPtr { stack_off } => {
                        self.buf.lea_mem(Reg::Rax, Reg::Rbp, *stack_off);
                        self.buf.store_mem(Reg::Rsp, sp_off, Reg::Rax, 8);
                        sp_off += 8;
                    }
                    _ => {}
                }
            }
        }

        // Load register arguments (note: values live in RBX/R12..R14 or stack [rbp+off], never clobbered by GP arg regs!).
        if let Some(sret_off) = sret_target {
            self.buf.lea_mem(gp_regs[0], Reg::Rbp, sret_off);
        }
        for pa in &prepared {
            match pa {
                PreparedArg::Gp1 { reg, val, sz } => {
                    self.load_scalar_raw(*reg, *val, *sz);
                }
                PreparedArg::Gp1Ptr { reg, stack_off } => {
                    self.buf.lea_mem(*reg, Reg::Rbp, *stack_off);
                }
                PreparedArg::Gp1DynData { reg, fat_val } => {
                    let (fbase, foff) = self.val_mem(*fat_val);
                    self.buf.load_mem(*reg, fbase, foff, 8, false);
                }
                PreparedArg::Gp1ClosureEnv { reg, fat_val } => {
                    let (fbase, foff) = self.val_mem(*fat_val);
                    self.buf.load_mem(*reg, fbase, foff + 8, 8, false);
                }
                PreparedArg::Gp2 { r0, r1, val, sz } => {
                    let (vbase, voff) = self.val_mem(*val);
                    self.buf.load_mem(*r0, vbase, voff, 8, false);
                    if *sz > 8 {
                        self.buf.load_mem(*r1, vbase, voff + 8, 8, false);
                    }
                }
                PreparedArg::Float { xmm, val, is_f64 } => {
                    self.load_float_val(*xmm, *val, *is_f64);
                }
                _ => {}
            }
        }

        // Emit call instruction!
        match callee {
            EirCallee::DirectFn(d) => {
                let sym = fn_sym_name(CfgFnKey::TopDef(*d), self.eir, self.interner);
                self.buf.call_sym(sym);
            }
            EirCallee::Method { impl_idx, sub_idx } => {
                let sym = fn_sym_name(
                    CfgFnKey::ImplMethod { impl_idx: *impl_idx, sub_idx: *sub_idx },
                    self.eir,
                    self.interner,
                );
                self.buf.call_sym(sym);
            }
            EirCallee::GenericInst { inst_id } => {
                let sym = fn_sym_name(CfgFnKey::GenericInst(*inst_id), self.eir, self.interner);
                self.buf.call_sym(sym);
            }
            EirCallee::ExternFn { name, .. } => {
                let s = self.interner.get(*name).to_owned();
                if is_builtin_rt_helper(&s) {
                    self.helper_call(&s);
                } else if self.is_pie_or_shared {
                    self.buf.call_plt_sym(s);
                } else {
                    self.buf.call_sym(s);
                }
            }
            EirCallee::DynCall { method_idx, .. } => {
                // Load vtable ptr from `args[0]` (`*dyn Iface`), then load `[vtable + method_idx * 8]` into Rax!
                let (fbase, foff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::Rax, fbase, foff + 8, 8, false);
                self.buf.load_mem(Reg::Rax, Reg::Rax, (*method_idx * 8) as i32, 8, false);
                self.buf.call_reg(Reg::Rax);
            }
            EirCallee::FnPtr(fpv) => {
                let (fbase, foff) = self.val_mem(*fpv);
                self.buf.load_mem(Reg::Rax, fbase, foff, 8, false);
                self.buf.call_reg(Reg::Rax);
            }
        }

        if aligned_stack_bytes > 0 {
            // Preserve flags (CF!) across stack cleanup? Use `lea rsp, [rsp + aligned_stack_bytes]` — LEA DOES NOT AFFECT FLAGS (CF stays intact for EonCall!!)!
            self.buf.lea_mem(Reg::Rsp, Reg::Rsp, aligned_stack_bytes as i32);
        }

        // Capture return value!
        let Some(dv) = dst else { return; };
        match ret_cls {
            RetClass::Void => {}
            RetClass::Float32 => self.store_float_val(dv, Xmm::Xmm0, false),
            RetClass::Float64 => self.store_float_val(dv, Xmm::Xmm0, true),
            RetClass::Reg1(sz) => self.store_scalar_val(dv, Reg::Rax, sz),
            RetClass::Reg2(sz) => {
                let doff = self.val_stack_off(dv);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                if sz > 8 {
                    self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
                }
            }
            RetClass::Sret(_) => {
                // Already written directly into `dst` via RVO `sret_ptr`!
            }
            RetClass::ErrorUnion { ok_ty, ok_size, ok_is_f32, ok_is_f64 } => {
                // EonCall error-return convention: `r15` = error code (0=Ok), `CF` = 1 on error!
                let doff = self.val_stack_off(dv);
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                let l_err = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.jcc_label(Cond::B, l_err); // `jc l_err` (1-instruction EonCall fast-path!)
                // Ok path: err = 0
                self.buf.store_mem(Reg::Rbp, doff, Reg::R15, 4);
                if ok_is_f32 {
                    self.buf.movss_store(Reg::Rbp, doff + ok_off, Xmm::Xmm0);
                } else if ok_is_f64 {
                    self.buf.movsd_store(Reg::Rbp, doff + ok_off, Xmm::Xmm0);
                } else if ok_size > 0 && ok_size <= 8 {
                    self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, ok_size);
                } else if ok_size > 8 && ok_size <= 16 {
                    self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                    self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rdx, (ok_size - 8).min(8));
                }
                // if ok_size > 16, callee already wrote into `[rbp + doff + ok_off]` via sret!
                self.buf.jmp_label(l_done);

                self.buf.bind_label(l_err);
                self.buf.store_mem(Reg::Rbp, doff, Reg::R15, 4);
                self.buf.bind_label(l_done);
            }
        }
    }

    // ========================================================================
    // Builtin Calls (Arena, AllocCap, ArenaPool, SlotPool, Buf, Atomic, Mmio, Slices)
    // ========================================================================

    fn load_ptr_arg(&mut self, dst: Reg, args: &[EirValue], arg_slots: &[Option<EirSlot>], idx: usize) {
        if let Some(Some(slot)) = arg_slots.get(idx) {
            let (base, off) = self.slot_mem(*slot);
            if matches!(self.env.get(self.f.slots[slot.0 as usize].ty), Ty::Borrow { .. } | Ty::RawPtr { .. }) {
                self.buf.load_mem(dst, base, off, 8, false);
            } else {
                self.buf.lea_mem(dst, base, off);
            }
            return;
        }
        let v = args[idx];
        let vty = self.f.value_tys[v.0 as usize];
        if matches!(self.env.get(vty), Ty::Borrow { .. } | Ty::RawPtr { .. }) {
            self.load_scalar_val(dst, v);
        } else {
            self.val_addr_into(dst, v);
        }
    }

    fn emit_builtin_call(
        &mut self,
        dst: Option<EirValue>,
        op: &CfgBuiltinOp,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) -> Result<(), String> {
        match op {
            CfgBuiltinOp::ArenaInit => {
                // args: (cap_ptr, size) -> ErrorUnion { err, Arena { buf, cap, used } }
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
                self.load_scalar_val(Reg::Rsi, args[1]);
                self.buf.mov_ri64(Reg::Rdx, 64);
                self.helper_call("eon_alloc_raw");
                // If !rax && size > 0 => err = 7
                self.load_scalar_val(Reg::Rcx, args[1]);
                let l_ok = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::Ne, l_ok);
                self.buf.test_rr(Reg::Rcx, Reg::Rcx);
                self.buf.jcc_label(Cond::E, l_ok);
                self.buf.mov_ri64(Reg::Rdx, 7);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
                self.buf.jmp_label(l_done);

                self.buf.bind_label(l_ok);
                self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8); // buf
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rcx, 8); // cap
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 16, Reg::Rdx, 8); // used = 0
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArenaAlloc => {
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
                self.load_scalar_val(Reg::Rsi, args[1]);
                self.buf.mov_ri64(Reg::Rdx, 16);
                self.helper_call("eon_arena_alloc_aligned");
                self.load_scalar_val(Reg::Rcx, args[1]);
                let l_ok = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::Ne, l_ok);
                self.buf.test_rr(Reg::Rcx, Reg::Rcx);
                self.buf.jcc_label(Cond::E, l_ok);
                self.buf.mov_ri64(Reg::Rdx, 7);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_ok);
                self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rcx, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArenaReset => {
                if self.sanitize {
                    self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
                    self.helper_call("eon_rt_arena_reset_san");
                } else {
                    self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                    self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                    self.buf.store_mem(Reg::R8, 16, Reg::Rax, 8); // used = 0
                }
            }
            CfgBuiltinOp::ArenaDestroy => {
                // args: (arena_val_or_slot, cap_ptr)
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                self.buf.load_mem(Reg::Rsi, Reg::R8, 0, 8, false); // buf
                self.buf.load_mem(Reg::Rdx, Reg::R8, 8, 8, false); // cap
                self.helper_call("eon_free_raw");
            }
            CfgBuiltinOp::ArenaUsedBytes => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 16, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::ArenaCapacity => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 8, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::AllocCapLiveAllocs => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::AllocCapUsedBytes => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 8, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::AllocCapPeakBytes => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 16, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::ArenaPoolInit(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                let ncols = cols.len();

                // Zero result and write cap at `ok_off + (ncols + 1) * 8`
                self.emit_memzero(Reg::Rbp, doff, x64_layout(self.env, ret_ty).size);
                self.load_scalar_val(Reg::Rax, args[1]);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + ((ncols + 1) * 8) as i32, Reg::Rax, 8);

                let l_fail = self.buf.new_label();
                let l_done = self.buf.new_label();
                for (i, c) in cols.iter().enumerate() {
                    let esz = x64_layout_impl(self.env, c.ty, false).size.max(1);
                    self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
                    self.load_scalar_val(Reg::Rsi, args[1]);
                    let l_nz = self.buf.new_label();
                    self.buf.test_rr(Reg::Rsi, Reg::Rsi);
                    self.buf.jcc_label(Cond::Ne, l_nz);
                    self.buf.mov_ri64(Reg::Rsi, 1);
                    self.buf.bind_label(l_nz);
                    if esz != 1 {
                        self.buf.imul_rri32(Reg::Rsi, Reg::Rsi, esz as i32);
                    }
                    self.buf.mov_ri64(Reg::Rdx, 64);
                    self.helper_call("eon_arena_alloc_aligned");
                    self.buf.test_rr(Reg::Rax, Reg::Rax);
                    self.buf.jcc_label(Cond::E, l_fail);
                    self.buf.store_mem(Reg::Rbp, doff + ok_off + (i * 8) as i32, Reg::Rax, 8);
                }
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_fail);
                self.buf.mov_ri64(Reg::Rax, 7);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArenaPoolPush(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let doff = self.val_stack_off(dst.unwrap());
                let ok_off = eu_ok_offset(self.env, TyId::U32) as i32;

                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::R9, Reg::R8, (ncols * 8) as i32, 8, false); // len
                self.buf.load_mem(Reg::R10, Reg::R8, ((ncols + 1) * 8) as i32, 8, false); // cap
                let l_oom = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oom);

                // len++
                self.buf.lea_mem(Reg::Rax, Reg::R9, 1);
                self.buf.store_mem(Reg::R8, (ncols * 8) as i32, Reg::Rax, 8);

                // Copy each column field from row `args[1]` (at stack off) into `pool->col_i[r9]`.
                let (rbase, row_off) = self.val_mem(args[1]);
                let row_ty = self.f.value_tys[args[1].0 as usize];
                for (i, c) in cols.iter().enumerate() {
                    let (foff, _) = struct_field_offset(self.env, row_ty, i);
                    let col_esz = x64_layout_impl(self.env, c.ty, false).size.max(1);
                    self.buf.load_mem(Reg::R10, Reg::R8, (i * 8) as i32, 8, false); // col_i ptr
                    self.buf.mov_rr(Reg::R11, Reg::R9);
                    if col_esz != 1 {
                        self.buf.imul_rri32(Reg::R11, Reg::R11, col_esz as i32);
                    }
                    self.buf.add_rr(Reg::R10, Reg::R11);
                    // Notice: for Link/?Link, row stores 8B GenId { index: u32, gen: u32 }, and col stores 4B Id = low 4 bytes (index)!
                    // So copying `col_esz` bytes from `row + foff` ALWAYS copies exact `.index` for Link and exact full value for everything else!
                    self.copy_mem_using_rax(Reg::R10, 0, rbase, row_off + foff as i32, col_esz);
                }
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R9, 4);
                self.buf.jmp_label(l_done);

                self.buf.bind_label(l_oom);
                self.buf.mov_ri64(Reg::Rax, 7);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.mov_ri64(Reg::Rax, u32::MAX as u64);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 4);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArenaPoolIdAt(_) => {
                let p_ty = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(p_ty) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => p_ty };
                let ncols = match self.env.get(deref) { Ty::Builtin(_, ga) => pool_col_count(self.env, ga), _ => 0 };
                let doff = self.val_stack_off(dst.unwrap());
                let ok_off = eu_ok_offset(self.env, TyId::U32) as i32;
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.load_mem(Reg::R10, Reg::R8, (ncols * 8) as i32, 8, false);
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R9, 4);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.mov_ri64(Reg::Rax, u32::MAX as u64);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 4);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArenaPoolReset(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::R8, (ncols * 8) as i32, Reg::Rax, 8);
                if let Some(dv) = dst {
                    self.load_val_from_mem(dv, Reg::R8, 0, ret_ty);
                }
            }
            CfgBuiltinOp::ArenaPoolLen => {
                let p_ty = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(p_ty) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => p_ty };
                let ncols = match self.env.get(deref) { Ty::Builtin(_, ga) => pool_col_count(self.env, ga), _ => 0 };
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, (ncols * 8) as i32, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::SlotPoolInit(elem_ty) => {
                self.emit_slot_pool_init(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::SlotPoolInsert(elem_ty) => {
                self.emit_slot_pool_insert(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty, true);
            }
            CfgBuiltinOp::SlotPoolTryInsert(elem_ty) => {
                self.emit_slot_pool_insert(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty, false);
            }
            CfgBuiltinOp::SlotPoolRemove(elem_ty) => {
                self.emit_slot_pool_remove(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::SlotPoolGet { elem_ty, .. } => {
                self.emit_slot_pool_get(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty, false);
            }
            CfgBuiltinOp::SlotPoolAt(elem_ty) => {
                self.emit_slot_pool_get(*elem_ty, dst.unwrap(), args, arg_slots, ret_ty, true);
            }
            CfgBuiltinOp::SlotPoolLen => {
                let ncols = self.slot_pool_ncols_from_arg(args[0]);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 2) * 8) as i32, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::SlotPoolCap => {
                let ncols = self.slot_pool_ncols_from_arg(args[0]);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 3) * 8) as i32, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::SlotPoolIsAlive(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.load_mem(Reg::R10, Reg::R8, ((ncols + 3) * 8) as i32, 8, false); // cap
                let l_false = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_false);
                self.buf.load_mem(Reg::R10, Reg::R8, (ncols * 8) as i32, 8, false); // gen*
                self.buf.shl_imm(Reg::R9, 2);
                self.buf.add_rr(Reg::R10, Reg::R9);
                self.buf.load_mem(Reg::Rax, Reg::R10, 0, 4, false);
                self.buf.and_ri32(Reg::Rax, 1);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_false);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.bind_label(l_done);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 1);
            }
            CfgBuiltinOp::SlotPoolIdAt(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let doff = self.val_stack_off(dst.unwrap());
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.load_mem(Reg::R10, Reg::R8, (ncols * 8) as i32, 8, false); // gen*
                self.buf.mov_rr(Reg::R11, Reg::R9);
                self.buf.shl_imm(Reg::R11, 2);
                self.buf.add_rr(Reg::R10, Reg::R11);
                self.buf.load_mem(Reg::Rax, Reg::R10, 0, 4, false);
                self.buf.store_mem(Reg::Rbp, doff, Reg::R9, 4); // index
                self.buf.store_mem(Reg::Rbp, doff + 4, Reg::Rax, 4); // gen
            }
            CfgBuiltinOp::SlotPoolDestroy(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                // Free gen, free_stack, and each column.
                for (i, c) in cols.iter().enumerate() {
                    let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                    self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                    self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                    self.buf.load_mem(Reg::Rsi, Reg::R8, (i * 8) as i32, 8, false);
                    self.buf.load_mem(Reg::Rdx, Reg::R8, ((ncols + 3) * 8) as i32, 8, false);
                    if esz != 1 { self.buf.imul_rri32(Reg::Rdx, Reg::Rdx, esz as i32); }
                    self.helper_call("eon_free_raw");
                }
                for extra in [ncols, ncols + 1] {
                    self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                    self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                    self.buf.load_mem(Reg::Rsi, Reg::R8, (extra * 8) as i32, 8, false);
                    self.buf.load_mem(Reg::Rdx, Reg::R8, ((ncols + 3) * 8) as i32, 8, false);
                    self.buf.shl_imm(Reg::Rdx, 2); // * 4 bytes
                    self.helper_call("eon_free_raw");
                }
            }
            CfgBuiltinOp::BufInit(elem_ty) => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ealign = x64_layout(self.env, *elem_ty).align.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
                self.load_scalar_val(Reg::Rsi, args[1]);
                if esz != 1 { self.buf.imul_rri32(Reg::Rsi, Reg::Rsi, esz as i32); }
                self.buf.mov_ri64(Reg::Rdx, ealign as u64);
                self.helper_call("eon_alloc_raw");
                self.load_scalar_val(Reg::Rcx, args[1]);
                let l_ok = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::Ne, l_ok);
                self.buf.test_rr(Reg::Rcx, Reg::Rcx);
                self.buf.jcc_label(Cond::E, l_ok);
        self.buf.mov_ri64(Reg::Rdx, 7);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.jmp_label(l_done);
        self.buf.bind_label(l_ok);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rdx, 8); // len = 0
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 16, Reg::Rcx, 8); // cap
        self.buf.bind_label(l_done);
    }
    CfgBuiltinOp::BufPush(elem_ty) => {
        let esz = x64_layout(self.env, *elem_ty).size.max(1);
        let doff = self.val_stack_off(dst.unwrap());
        let val_arg = if args.len() >= 3 { args[2] } else { args[1] };
        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, Reg::R8, 8, 8, false); // len
        self.buf.load_mem(Reg::R10, Reg::R8, 16, 8, false); // cap
        let l_full = self.buf.new_label();
        let l_done = self.buf.new_label();
        self.buf.cmp_rr(Reg::R9, Reg::R10);
        self.buf.jcc_label(Cond::Ae, l_full);
        self.buf.lea_mem(Reg::Rax, Reg::R9, 1);
        self.buf.store_mem(Reg::R8, 8, Reg::Rax, 8);
        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false); // ptr
        if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
        self.buf.add_rr(Reg::R8, Reg::R9);
        self.store_val_to_mem(Reg::R8, 0, val_arg, *elem_ty);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.jmp_label(l_done);
        self.buf.bind_label(l_full);
        self.buf.mov_ri64(Reg::Rax, 7);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }
    CfgBuiltinOp::BufPushSlice(elem_ty) => {
        let esz = x64_layout(self.env, *elem_ty).size.max(1);
        let doff = self.val_stack_off(dst.unwrap());
        let slice_arg = if args.len() >= 3 { args[2] } else { args[1] };
        let soff = self.val_stack_off(slice_arg);
        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, Reg::R8, 8, 8, false);   // buf.len
        self.buf.load_mem(Reg::R10, Reg::R8, 16, 8, false); // buf.cap
        self.buf.load_mem(Reg::R11, Reg::Rbp, soff + 8, 8, false); // slice.len
        // Check: cap - len < slice.len => OOM (7)
        self.buf.mov_rr(Reg::Rax, Reg::R10);
        self.buf.sub_rr(Reg::Rax, Reg::R9);
        let l_full = self.buf.new_label();
        let l_copy_loop = self.buf.new_label();
        let l_copy_done = self.buf.new_label();
        let l_done = self.buf.new_label();
        self.buf.cmp_rr(Reg::Rax, Reg::R11);
        self.buf.jcc_label(Cond::B, l_full);
        // Update buf.len = len + slice.len
        self.buf.mov_rr(Reg::Rax, Reg::R9);
        self.buf.add_rr(Reg::Rax, Reg::R11);
        self.buf.store_mem(Reg::R8, 8, Reg::Rax, 8);
        // Copy byte count = slice.len * esz
        self.buf.mov_rr(Reg::Rcx, Reg::R11);
        if esz != 1 { self.buf.imul_rri32(Reg::Rcx, Reg::Rcx, esz as i32); }
        self.buf.test_rr(Reg::Rcx, Reg::Rcx);
        self.buf.jcc_label(Cond::E, l_copy_done);
        // Dst ptr = buf.ptr + old_len * esz (in R8), Src ptr = slice.ptr (in R10)
        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
        if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
        self.buf.add_rr(Reg::R8, Reg::R9);
        self.buf.load_mem(Reg::R10, Reg::Rbp, soff, 8, false);
        self.buf.bind_label(l_copy_loop);
        self.buf.load_mem(Reg::Rax, Reg::R10, 0, 1, false);
        self.buf.store_mem(Reg::R8, 0, Reg::Rax, 1);
        self.buf.add_ri32(Reg::R10, 1);
        self.buf.add_ri32(Reg::R8, 1);
        self.buf.sub_ri32(Reg::Rcx, 1);
        self.buf.jcc_label(Cond::Ne, l_copy_loop);
        self.buf.bind_label(l_copy_done);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.jmp_label(l_done);
        self.buf.bind_label(l_full);
        self.buf.mov_ri64(Reg::Rax, 7);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }
    CfgBuiltinOp::BufPop(elem_ty) => {
        let esz = x64_layout(self.env, *elem_ty).size.max(1);
        let doff = self.val_stack_off(dst.unwrap());
        let voff = opt_val_offset(self.env, *elem_ty) as i32;
        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, Reg::R8, 8, 8, false); // len
        let l_empty = self.buf.new_label();
        let l_done = self.buf.new_label();
        self.buf.test_rr(Reg::R9, Reg::R9);
        self.buf.jcc_label(Cond::E, l_empty);
        self.buf.sub_ri32(Reg::R9, 1);
        self.buf.store_mem(Reg::R8, 8, Reg::R9, 8);
        self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
        if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
        self.buf.add_rr(Reg::R8, Reg::R9);
        if is_niche_id_opt(self.env, *elem_ty) {
            self.load_val_from_mem(dst.unwrap(), Reg::R8, 0, *elem_ty);
        } else {
            self.buf.mov_ri64(Reg::Rax, 1);
            self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 1);
            self.copy_mem_using_rax(Reg::Rbp, doff + voff, Reg::R8, 0, esz);
        }
        self.buf.jmp_label(l_done);
        self.buf.bind_label(l_empty);
        self.emit_opt_null(dst.unwrap(), ret_ty);
        self.buf.bind_label(l_done);
    }
            CfgBuiltinOp::BufLen => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 8, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::BufCap => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 16, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::BufClear => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::R8, 8, Reg::Rax, 8);
            }
            CfgBuiltinOp::BufAsSlice { .. } => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                let doff = self.val_stack_off(dst.unwrap());
                self.buf.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
                self.buf.load_mem(Reg::Rdx, Reg::R8, 8, 8, false);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rdx, 8);
            }
            CfgBuiltinOp::BufAt { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let ok_off = eu_ok_offset(self.env, TyId::USIZE) as i32;
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.load_mem(Reg::R10, Reg::R8, 8, 8, false); // len
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::BufGet { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.load_mem(Reg::R10, Reg::R8, 8, 8, false); // len
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false); // ptr
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.store_mem(Reg::Rbp, doff, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::BufSub { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::R11, Reg::R8, 8, 8, false); // len
                self.buf.load_mem(Reg::R8, Reg::R8, 0, 8, false);  // ptr
                self.load_scalar_val(Reg::R9, args[1]);            // lo
                self.load_scalar_val(Reg::R10, args[2]);           // hi
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.cmp_rr(Reg::R10, Reg::R11);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.sub_rr(Reg::R10, Reg::R9); // new_len = hi - lo
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::R10, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::BufDestroy(elem_ty) => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                self.buf.load_mem(Reg::Rsi, Reg::R8, 0, 8, false);
                self.buf.load_mem(Reg::Rdx, Reg::R8, 16, 8, false);
                if esz != 1 { self.buf.imul_rri32(Reg::Rdx, Reg::Rdx, esz as i32); }
                self.helper_call("eon_free_raw");
            }
            CfgBuiltinOp::AtomicNew(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_scalar_raw(Reg::Rax, args[0], sz);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicLoad(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 0, sz, is_signed_int(self.env, *el));
                self.buf.lfence();
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicStore(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                // `xchg` guarantees SeqCst/Release visibility on x86_64 TSO!
                self.buf.xchg_mem(Reg::R8, Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicFetchAdd(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                self.buf.lock_xadd(Reg::R8, Reg::Rax, sz);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicFetchSub(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                self.buf.neg_r(Reg::Rax);
                self.buf.lock_xadd(Reg::R8, Reg::Rax, sz);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicSwap(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                self.buf.xchg_mem(Reg::R8, Reg::Rax, sz);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::AtomicCas(el) => {
                // cas(expected, desired, ok_ord, fail_ord) -> bool
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                self.load_scalar_raw(Reg::Rcx, args[2], sz);
                self.buf.lock_cmpxchg(Reg::R8, Reg::Rcx, sz);
                self.buf.setcc(Cond::E, Reg::Rax);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 1);
            }
            CfgBuiltinOp::MmioRead(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 0, sz, is_signed_int(self.env, *el));
                self.buf.lfence();
                self.store_scalar_val(dst.unwrap(), Reg::Rax, sz);
            }
            CfgBuiltinOp::MmioWrite(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_raw(Reg::Rax, args[1], sz);
                self.buf.store_mem(Reg::R8, 0, Reg::Rax, sz);
                self.buf.sfence();
            }
            CfgBuiltinOp::RawPtrFromUsize(_) | CfgBuiltinOp::RawPtrAsUsize => {
                self.load_scalar_val(Reg::Rax, args[0]);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::ScheduleBuild => {}
            CfgBuiltinOp::ParReduce { elem_ty, op } => {
                // Deterministic reduction starting from `init = args[2]` matching C11/EIR-Interp.
                let op_name = self.interner.get(*op).to_owned();
                let is_float = matches!(self.env.get(*elem_ty), Ty::Float(_));
                let is_f16 = matches!(self.env.get(*elem_ty), Ty::Float(16));
                let is_f64 = matches!(self.env.get(*elem_ty), Ty::Float(64));
                let esz = x64_layout(self.env, *elem_ty).size.clamp(1, 8);
                let signed = is_signed_int(self.env, *elem_ty);
                let (sbase, soff) = self.val_mem(args[0]);
                let tmp = self.plan.temp_copy_off;

                if is_f16 {
                    // Store ptr, len, acc_f16 into temp_copy_off across `eon_f16_to_f32` calls.
                    self.buf.load_mem(Reg::R8, sbase, soff, 8, false);
                    self.buf.load_mem(Reg::R9, sbase, soff + 8, 8, false);
                    self.buf.store_mem(Reg::Rbp, tmp, Reg::R8, 8);
                    self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::R9, 8);
                    if let Some(&init_v) = args.get(2) {
                        self.load_scalar_raw(Reg::Rax, init_v, 2);
                    } else {
                        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                    }
                    self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 2);
                    let l_hd = self.buf.new_label();
                    let l_end = self.buf.new_label();
                    self.buf.bind_label(l_hd);
                    self.buf.load_mem(Reg::R9, Reg::Rbp, tmp + 8, 8, false);
                    self.buf.test_rr(Reg::R9, Reg::R9);
                    self.buf.jcc_label(Cond::E, l_end);
                    self.buf.sub_ri32(Reg::R9, 1);
                    self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::R9, 8);
                    self.buf.load_mem(Reg::R8, Reg::Rbp, tmp, 8, false);
                    self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 2, false);
                    self.buf.add_ri32(Reg::R8, 2);
                    self.buf.store_mem(Reg::Rbp, tmp, Reg::R8, 8);
                    self.helper_call("eon_f16_to_f32");
                    self.buf.movss_store(Reg::Rbp, tmp + 24, Xmm::Xmm0);
                    self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 2, false);
                    self.helper_call("eon_f16_to_f32");
                    self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, tmp + 24);
                    let sse_op = match op_name.as_str() {
                        "Mul" => 0x59,
                        "Min" => 0x5D,
                        "Max" => 0x5F,
                        _ => 0x58,
                    };
                    self.buf.sse_bin(false, sse_op, Xmm::Xmm0, Xmm::Xmm1);
                    self.helper_call("eon_f32_to_f16");
                    self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 2);
                    self.buf.jmp_label(l_hd);
                    self.buf.bind_label(l_end);
                    self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 2, false);
                    self.store_scalar_val(dst.unwrap(), Reg::Rax, 2);
                } else {
                    self.buf.load_mem(Reg::R8, sbase, soff, 8, false); // ptr
                    self.buf.load_mem(Reg::R9, sbase, soff + 8, 8, false); // len
                    self.buf.xor_rr32(Reg::R10, Reg::R10); // i = 0
                    if is_float {
                        if let Some(&init_v) = args.get(2) {
                            self.load_float_val(Xmm::Xmm0, init_v, is_f64);
                        } else {
                            self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                            self.buf.movq_xmm_r64(Xmm::Xmm0, Reg::Rax);
                        }
                    } else if let Some(&init_v) = args.get(2) {
                        self.load_scalar_val(Reg::Rax, init_v);
                    } else {
                        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                    }
                    let l_hd = self.buf.new_label();
                    let l_end = self.buf.new_label();
                    self.buf.bind_label(l_hd);
                    self.buf.cmp_rr(Reg::R10, Reg::R9);
                    self.buf.jcc_label(Cond::Ae, l_end);
                    if is_float {
                        if is_f64 {
                            self.buf.movsd_load(Xmm::Xmm1, Reg::R8, 0);
                        } else {
                            self.buf.movss_load(Xmm::Xmm1, Reg::R8, 0);
                        }
                        let sse_op = match op_name.as_str() {
                            "Mul" => 0x59,
                            "Min" => 0x5D,
                            "Max" => 0x5F,
                            _ => 0x58,
                        };
                        self.buf.sse_bin(is_f64, sse_op, Xmm::Xmm0, Xmm::Xmm1);
                    } else {
                        self.buf.load_mem(Reg::Rcx, Reg::R8, 0, esz, signed);
                        match op_name.as_str() {
                            "Min" => {
                                self.buf.cmp_rr(Reg::Rcx, Reg::Rax);
                                self.buf.cmovcc(if signed { Cond::L } else { Cond::B }, Reg::Rax, Reg::Rcx);
                            }
                            "Max" => {
                                self.buf.cmp_rr(Reg::Rcx, Reg::Rax);
                                self.buf.cmovcc(if signed { Cond::G } else { Cond::A }, Reg::Rax, Reg::Rcx);
                            }
                            "Mul" => {
                                self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                                self.emit_norm_subword_int(Reg::Rax, *elem_ty);
                            }
                            "And" | "BitAnd" => self.buf.and_rr(Reg::Rax, Reg::Rcx),
                            "Or" | "BitOr" => self.buf.or_rr(Reg::Rax, Reg::Rcx),
                            "Xor" | "BitXor" => self.buf.xor_rr(Reg::Rax, Reg::Rcx),
                            _ => {
                                self.buf.add_rr(Reg::Rax, Reg::Rcx);
                                self.emit_norm_subword_int(Reg::Rax, *elem_ty);
                            }
                        }
                    }
                    self.buf.add_ri32(Reg::R8, esz as i32);
                    self.buf.add_ri32(Reg::R10, 1);
                    self.buf.jmp_label(l_hd);
                    self.buf.bind_label(l_end);
                    if is_float {
                        self.store_float_val(dst.unwrap(), Xmm::Xmm0, is_f64);
                    } else {
                        self.store_scalar_val(dst.unwrap(), Reg::Rax, esz);
                    }
                }
            }
            CfgBuiltinOp::SliceLen | CfgBuiltinOp::StrLen => {
                let (sbase, soff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::Rax, sbase, soff + 8, 8, false);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::ArrayLen => {
                let ty0 = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len, _ => 0 };
                self.buf.mov_ri64(Reg::Rax, len);
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 8);
            }
            CfgBuiltinOp::StrAsBytes => {
                self.copy_val_to_val(dst.unwrap(), args[0]);
            }
            CfgBuiltinOp::BytesAsStr => {
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) {
                    Ty::ErrorUnion { ok, .. } => *ok,
                    _ => ret_ty,
                };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                let (sbase, soff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::Rdi, sbase, soff, 8, false);
                self.buf.load_mem(Reg::Rsi, sbase, soff + 8, 8, false);
                self.helper_call("eon_rt_validate_utf8");
                let l_err = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::E, l_err);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                let (sbase2, soff2) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::R8, sbase2, soff2, 8, false);
                self.buf.load_mem(Reg::R9, sbase2, soff2 + 8, 8, false);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::R9, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_err);
                self.buf.mov_ri64(Reg::Rax, 5); // Utf8Error.InvalidUtf8
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::SliceAt { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let ok_off = eu_ok_offset(self.env, TyId::USIZE) as i32;
                let (sbase, soff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::R8, sbase, soff, 8, false); // ptr
                self.buf.load_mem(Reg::R10, sbase, soff + 8, 8, false); // len
                self.load_scalar_val(Reg::R9, args[1]);
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArrayAt { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len, _ => 0 };
                let doff = self.val_stack_off(dst.unwrap());
                let ok_off = eu_ok_offset(self.env, TyId::USIZE) as i32;
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.mov_ri64(Reg::R10, len);
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::SliceGet { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let (sbase, soff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::R8, sbase, soff, 8, false);
                self.buf.load_mem(Reg::R10, sbase, soff + 8, 8, false);
                self.load_scalar_val(Reg::R9, args[1]);
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.store_mem(Reg::Rbp, doff, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArrayGet { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len, _ => 0 };
                let doff = self.val_stack_off(dst.unwrap());
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_scalar_val(Reg::R9, args[1]);
                self.buf.mov_ri64(Reg::R10, len);
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::Ae, l_oob);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.store_mem(Reg::Rbp, doff, Reg::R8, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::SliceSub { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                let (sbase, soff) = self.val_mem(args[0]);
                self.buf.load_mem(Reg::R8, sbase, soff, 8, false);      // ptr
                self.buf.load_mem(Reg::R11, sbase, soff + 8, 8, false); // len
                self.load_scalar_val(Reg::R9, args[1]);                    // lo
                self.load_scalar_val(Reg::R10, args[2]);                   // hi
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.cmp_rr(Reg::R10, Reg::R11);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.sub_rr(Reg::R10, Reg::R9);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::R10, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::ArraySub { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = self.f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len, _ => 0 };
                let doff = self.val_stack_off(dst.unwrap());
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.mov_ri64(Reg::R11, len);
                self.load_scalar_val(Reg::R9, args[1]);  // lo
                self.load_scalar_val(Reg::R10, args[2]); // hi
                let l_oob = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R9, Reg::R10);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.cmp_rr(Reg::R10, Reg::R11);
                self.buf.jcc_label(Cond::A, l_oob);
                self.buf.sub_rr(Reg::R10, Reg::R9);
                if esz != 1 { self.buf.imul_rri32(Reg::R9, Reg::R9, esz as i32); }
                self.buf.add_rr(Reg::R8, Reg::R9);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::R10, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.mov_ri64(Reg::Rax, 4);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::VecSplat { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let doff = self.val_stack_off(d);
                    if let Ty::Float(bits) = self.env.get(elem) {
                        if *bits == 16 {
                            self.load_scalar_val(Reg::Rax, args[0]);
                            for lane in 0..(len as usize) {
                                self.buf.store_mem(Reg::Rbp, doff + (lane * esz) as i32, Reg::Rax, 2);
                            }
                        } else if *bits > 32 {
                            self.load_float_val(Xmm::Xmm0, args[0], true);
                            for lane in 0..(len as usize) {
                                self.buf.movsd_store(Reg::Rbp, doff + (lane * esz) as i32, Xmm::Xmm0);
                            }
                        } else {
                            self.load_float_val(Xmm::Xmm0, args[0], false);
                            for lane in 0..(len as usize) {
                                self.buf.movss_store(Reg::Rbp, doff + (lane * esz) as i32, Xmm::Xmm0);
                            }
                        }
                    } else {
                        self.load_scalar_val(Reg::Rax, args[0]);
                        for lane in 0..(len as usize) {
                            self.buf.store_mem(Reg::Rbp, doff + (lane * esz) as i32, Reg::Rax, esz.clamp(1, 8));
                        }
                    }
                }
            }
            CfgBuiltinOp::VecLen(len) => {
                if let Some(d) = dst {
                    self.buf.mov_ri64(Reg::Rax, *len as u64);
                    self.store_scalar_val(d, Reg::Rax, 8);
                }
            }
            CfgBuiltinOp::VecCmp { op: cmp_op, vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let (abase, aoff) = self.val_mem(args[0]);
                    let (bbase, boff) = self.val_mem(args[1]);
                    let doff = self.val_stack_off(d);
                    // SSE2 packed fast path: `u8x16.cmpeq` (`pcmpeqb` + `pand 0x01..`)
                    if matches!(elem, TyId::U8 | TyId::I8) && len == 16 && *cmp_op == BinaryOp::Eq {
                        self.buf.movups_load(Xmm::Xmm0, abase, aoff);
                        self.buf.movups_load(Xmm::Xmm1, bbase, boff);
                        self.buf.sse_66_bin(0x74, Xmm::Xmm0, Xmm::Xmm1); // pcmpeqb
                        self.buf.mov_ri64(Reg::Rax, 0x0101_0101_0101_0101);
                        self.buf.movq_xmm_r64(Xmm::Xmm1, Reg::Rax);
                        self.buf.sse_66_bin(0x6C, Xmm::Xmm1, Xmm::Xmm1); // punpcklqdq
                        self.buf.sse_66_bin(0xDB, Xmm::Xmm0, Xmm::Xmm1); // pand
                        self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                        return Ok(());
                    }
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let sgn = is_signed_int(self.env, elem);
                    for lane in 0..(len as usize) {
                        let ao = aoff + (lane * esz) as i32;
                        let bo = boff + (lane * esz) as i32;
                        if let Ty::Float(bits) = self.env.get(elem) {
                            let is_f16 = *bits == 16;
                            let is_f64 = *bits > 32;
                            if is_f16 {
                                self.buf.load_mem(Reg::Rdi, bbase, bo, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, abase, ao, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                            } else if is_f64 {
                                self.buf.movsd_load(Xmm::Xmm0, abase, ao);
                                self.buf.movsd_load(Xmm::Xmm1, bbase, bo);
                            } else {
                                self.buf.movss_load(Xmm::Xmm0, abase, ao);
                                self.buf.movss_load(Xmm::Xmm1, bbase, bo);
                            }
                            let use_f64 = !is_f16 && is_f64;
                            match cmp_op {
                                BinaryOp::Eq => {
                                    self.buf.ucomi(use_f64, Xmm::Xmm0, Xmm::Xmm1);
                                    self.buf.setcc(Cond::E, Reg::Rax);
                                    self.buf.setcc(Cond::Np, Reg::Rcx);
                                    self.buf.and_rr(Reg::Rax, Reg::Rcx);
                                }
                                BinaryOp::NotEq => {
                                    self.buf.ucomi(use_f64, Xmm::Xmm0, Xmm::Xmm1);
                                    self.buf.setcc(Cond::Ne, Reg::Rax);
                                    self.buf.setcc(Cond::P, Reg::Rcx);
                                    self.buf.or_rr(Reg::Rax, Reg::Rcx);
                                }
                                BinaryOp::Lt | BinaryOp::LtEq => {
                                    self.buf.ucomi(use_f64, Xmm::Xmm1, Xmm::Xmm0);
                                    self.buf.setcc(if *cmp_op == BinaryOp::Lt { Cond::A } else { Cond::Ae }, Reg::Rax);
                                }
                                _ => {
                                    self.buf.ucomi(use_f64, Xmm::Xmm0, Xmm::Xmm1);
                                    self.buf.setcc(if *cmp_op == BinaryOp::Gt { Cond::A } else { Cond::Ae }, Reg::Rax);
                                }
                            }
                        } else {
                            let lsz = esz.clamp(1, 8);
                            self.buf.load_mem(Reg::Rax, abase, ao, lsz, sgn);
                            self.buf.load_mem(Reg::Rcx, bbase, bo, lsz, sgn);
                            self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                            let cc = match (cmp_op, sgn) {
                                (BinaryOp::Eq, _) => Cond::E,
                                (BinaryOp::NotEq, _) => Cond::Ne,
                                (BinaryOp::Lt, true) => Cond::L,
                                (BinaryOp::Lt, false) => Cond::B,
                                (BinaryOp::LtEq, true) => Cond::Le,
                                (BinaryOp::LtEq, false) => Cond::Be,
                                (BinaryOp::Gt, true) => Cond::G,
                                (BinaryOp::Gt, false) => Cond::A,
                                (BinaryOp::GtEq, true) => Cond::Ge,
                                _ => Cond::Ae,
                            };
                            self.buf.setcc(cc, Reg::Rax);
                        }
                        self.buf.store_mem(Reg::Rbp, doff + lane as i32, Reg::Rax, 1);
                    }
                }
            }
            CfgBuiltinOp::VecMaskAny { len } => {
                let Some(d) = dst else { return Ok(()); };
                let (mbase, moff) = self.val_mem(args[0]);
                let l_hit = self.buf.new_label();
                let l_done = self.buf.new_label();
                for lane in 0..(*len as usize) {
                    self.buf.load_mem(Reg::Rax, mbase, moff + lane as i32, 1, false);
                    self.buf.test_rr(Reg::Rax, Reg::Rax);
                    self.buf.jcc_label(Cond::Ne, l_hit);
                }
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_hit);
                self.buf.mov_ri64(Reg::Rax, 1);
                self.buf.bind_label(l_done);
                self.store_scalar_val(d, Reg::Rax, 1);
            }
            CfgBuiltinOp::VecMaskAll { len } => {
                let Some(d) = dst else { return Ok(()); };
                let (mbase, moff) = self.val_mem(args[0]);
                let l_miss = self.buf.new_label();
                let l_done = self.buf.new_label();
                for lane in 0..(*len as usize) {
                    self.buf.load_mem(Reg::Rax, mbase, moff + lane as i32, 1, false);
                    self.buf.test_rr(Reg::Rax, Reg::Rax);
                    self.buf.jcc_label(Cond::E, l_miss);
                }
                self.buf.mov_ri64(Reg::Rax, 1);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_miss);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.bind_label(l_done);
                self.store_scalar_val(d, Reg::Rax, 1);
            }
            CfgBuiltinOp::VecBitmask { len } => {
                let Some(d) = dst else { return Ok(()); };
                let (mbase, moff) = self.val_mem(args[0]);
                if *len == 16 {
                    // 2-instruction SSE2 fast path: shift bool bit 0 -> bit 7, extract 16-bit mask!
                    self.buf.movups_load(Xmm::Xmm0, mbase, moff);
                    self.buf.psllw_imm(Xmm::Xmm0, 7);
                    self.buf.pmovmskb(Reg::Rax, Xmm::Xmm0);
                    self.store_scalar_val(d, Reg::Rax, 8);
                    return Ok(());
                }
                self.buf.xor_rr32(Reg::R8, Reg::R8);
                for lane in 0..(*len as usize).min(64) {
                    self.buf.load_mem(Reg::Rax, mbase, moff + lane as i32, 1, false);
                    self.buf.and_ri32(Reg::Rax, 1);
                    if lane > 0 {
                        self.buf.mov_ri64(Reg::Rcx, lane as u64);
                        self.buf.shl_cl(Reg::Rax);
                    }
                    self.buf.or_rr(Reg::R8, Reg::Rax);
                }
                self.store_scalar_val(d, Reg::R8, 8);
            }
            CfgBuiltinOp::VecSelect { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let esz = x64_layout(self.env, elem).size.max(1).clamp(1, 8);
                    let (mbase, moff) = self.val_mem(args[0]);
                    let (abase, aoff) = self.val_mem(args[1]);
                    let (bbase, boff) = self.val_mem(args[2]);
                    let doff = self.val_stack_off(d);
                    for lane in 0..(len as usize) {
                        let l_false = self.buf.new_label();
                        let l_done = self.buf.new_label();
                        self.buf.load_mem(Reg::Rax, mbase, moff + lane as i32, 1, false);
                        self.buf.test_rr(Reg::Rax, Reg::Rax);
                        self.buf.jcc_label(Cond::E, l_false);
                        self.buf.load_mem(Reg::Rax, abase, aoff + (lane * esz) as i32, esz, false);
                        self.buf.jmp_label(l_done);
                        self.buf.bind_label(l_false);
                        self.buf.load_mem(Reg::Rax, bbase, boff + (lane * esz) as i32, esz, false);
                        self.buf.bind_label(l_done);
                        self.buf.store_mem(Reg::Rbp, doff + (lane * esz) as i32, Reg::Rax, esz);
                    }
                }
            }
            CfgBuiltinOp::VecMin { vec_ty } | CfgBuiltinOp::VecMax { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                let is_min = matches!(op, CfgBuiltinOp::VecMin { .. });
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let (abase, aoff) = self.val_mem(args[0]);
                    let (bbase, boff) = self.val_mem(args[1]);
                    let doff = self.val_stack_off(d);
                    // SSE2 packed `minps`/`maxps` (`f32x4`) and `minpd`/`maxpd` (`f64x2`)
                    if elem == TyId::F32 && len == 4 {
                        self.buf.movups_load(Xmm::Xmm0, abase, aoff);
                        self.buf.movups_load(Xmm::Xmm1, bbase, boff);
                        self.buf.sse_ps_bin(if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                        self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                        return Ok(());
                    }
                    if elem == TyId::F64 && len == 2 {
                        self.buf.movups_load(Xmm::Xmm0, abase, aoff);
                        self.buf.movups_load(Xmm::Xmm1, bbase, boff);
                        self.buf.sse_66_bin(if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                        self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                        return Ok(());
                    }
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let sgn = is_signed_int(self.env, elem);
                    for lane in 0..(len as usize) {
                        let ao = aoff + (lane * esz) as i32;
                        let bo = boff + (lane * esz) as i32;
                        let dd = doff + (lane * esz) as i32;
                        if let Ty::Float(bits) = self.env.get(elem) {
                            let is_f16 = *bits == 16;
                            let is_f64 = *bits > 32;
                            if is_f16 {
                                self.buf.load_mem(Reg::Rdi, bbase, bo, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, abase, ao, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                                self.buf.sse_bin(false, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                                self.helper_call("eon_f32_to_f16");
                                self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, 2);
                            } else if is_f64 {
                                self.buf.movsd_load(Xmm::Xmm0, abase, ao);
                                self.buf.movsd_load(Xmm::Xmm1, bbase, bo);
                                self.buf.sse_bin(true, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movsd_store(Reg::Rbp, dd, Xmm::Xmm0);
                            } else {
                                self.buf.movss_load(Xmm::Xmm0, abase, ao);
                                self.buf.movss_load(Xmm::Xmm1, bbase, bo);
                                self.buf.sse_bin(false, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movss_store(Reg::Rbp, dd, Xmm::Xmm0);
                            }
                        } else {
                            let lsz = esz.clamp(1, 8);
                            let l_take_a = self.buf.new_label();
                            self.buf.load_mem(Reg::Rax, abase, ao, lsz, sgn);
                            self.buf.load_mem(Reg::Rcx, bbase, bo, lsz, sgn);
                            self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                            let cc = match (is_min, sgn) {
                                (true, true) => Cond::L,
                                (true, false) => Cond::B,
                                (false, true) => Cond::G,
                                (false, false) => Cond::A,
                            };
                            self.buf.jcc_label(cc, l_take_a);
                            self.buf.mov_rr(Reg::Rax, Reg::Rcx);
                            self.buf.bind_label(l_take_a);
                            self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, lsz);
                        }
                    }
                }
            }
            CfgBuiltinOp::VecFma { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let (abase, aoff) = self.val_mem(args[0]);
                    let (bbase, boff) = self.val_mem(args[1]);
                    let (cbase, coff) = self.val_mem(args[2]);
                    let doff = self.val_stack_off(d);
                    if elem == TyId::F32 && len == 4 {
                        self.buf.movups_load(Xmm::Xmm0, abase, aoff);
                        self.buf.movups_load(Xmm::Xmm1, bbase, boff);
                        self.buf.sse_ps_bin(0x59, Xmm::Xmm0, Xmm::Xmm1); // mulps
                        self.buf.movups_load(Xmm::Xmm1, cbase, coff);
                        self.buf.sse_ps_bin(0x58, Xmm::Xmm0, Xmm::Xmm1); // addps
                        self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                        return Ok(());
                    }
                    if elem == TyId::F64 && len == 2 {
                        self.buf.movups_load(Xmm::Xmm0, abase, aoff);
                        self.buf.movups_load(Xmm::Xmm1, bbase, boff);
                        self.buf.sse_66_bin(0x59, Xmm::Xmm0, Xmm::Xmm1); // mulpd
                        self.buf.movups_load(Xmm::Xmm1, cbase, coff);
                        self.buf.sse_66_bin(0x58, Xmm::Xmm0, Xmm::Xmm1); // addpd
                        self.buf.movups_store(Reg::Rbp, doff, Xmm::Xmm0);
                        return Ok(());
                    }
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let sgn = is_signed_int(self.env, elem);
                    for lane in 0..(len as usize) {
                        let ao = aoff + (lane * esz) as i32;
                        let bo = boff + (lane * esz) as i32;
                        let co = coff + (lane * esz) as i32;
                        let dd = doff + (lane * esz) as i32;
                        if let Ty::Float(bits) = self.env.get(elem) {
                            let is_f16 = *bits == 16;
                            let is_f64 = *bits > 32;
                            if is_f16 {
                                self.buf.load_mem(Reg::Rdi, bbase, bo, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, cbase, co, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off + 4, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, abase, ao, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                                self.buf.sse_bin(false, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off + 4);
                                self.buf.sse_bin(false, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                                self.helper_call("eon_f32_to_f16");
                                self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, 2);
                            } else if is_f64 {
                                self.buf.movsd_load(Xmm::Xmm0, abase, ao);
                                self.buf.movsd_load(Xmm::Xmm1, bbase, bo);
                                self.buf.sse_bin(true, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movsd_load(Xmm::Xmm1, cbase, co);
                                self.buf.sse_bin(true, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movsd_store(Reg::Rbp, dd, Xmm::Xmm0);
                            } else {
                                self.buf.movss_load(Xmm::Xmm0, abase, ao);
                                self.buf.movss_load(Xmm::Xmm1, bbase, bo);
                                self.buf.sse_bin(false, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movss_load(Xmm::Xmm1, cbase, co);
                                self.buf.sse_bin(false, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                                self.buf.movss_store(Reg::Rbp, dd, Xmm::Xmm0);
                            }
                        } else {
                            let lsz = esz.clamp(1, 8);
                            self.buf.load_mem(Reg::Rax, abase, ao, lsz, sgn);
                            self.buf.load_mem(Reg::Rcx, bbase, bo, lsz, sgn);
                            self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                            self.buf.load_mem(Reg::Rcx, cbase, co, lsz, sgn);
                            self.buf.add_rr(Reg::Rax, Reg::Rcx);
                            self.emit_norm_subword_int(Reg::Rax, elem);
                            self.buf.store_mem(Reg::Rbp, dd, Reg::Rax, lsz);
                        }
                    }
                }
            }
            CfgBuiltinOp::VecReduceAdd { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let (abase, aoff) = self.val_mem(args[0]);
                    if let Ty::Float(bits) = self.env.get(elem) {
                        let is_f16 = *bits == 16;
                        let is_f64 = *bits > 32;
                        if is_f16 {
                            self.buf.load_mem(Reg::Rdi, abase, aoff, 2, false);
                            self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rdi, 2);
                            for lane in 1..(len as usize) {
                                self.buf.load_mem(Reg::Rdi, abase, aoff + (lane * 2) as i32, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off + 4, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off + 4);
                                self.buf.sse_bin(false, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                                self.helper_call("eon_f32_to_f16");
                                self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rax, 2);
                            }
                            self.buf.load_mem(Reg::Rax, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                            self.store_scalar_val(d, Reg::Rax, 2);
                        } else if is_f64 {
                            self.buf.movsd_load(Xmm::Xmm0, abase, aoff);
                            for lane in 1..(len as usize) {
                                self.buf.movsd_load(Xmm::Xmm1, abase, aoff + (lane * 8) as i32);
                                self.buf.sse_bin(true, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                            }
                            self.store_float_val(d, Xmm::Xmm0, true);
                        } else {
                            self.buf.movss_load(Xmm::Xmm0, abase, aoff);
                            for lane in 1..(len as usize) {
                                self.buf.movss_load(Xmm::Xmm1, abase, aoff + (lane * 4) as i32);
                                self.buf.sse_bin(false, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                            }
                            self.store_float_val(d, Xmm::Xmm0, false);
                        }
                    } else {
                        let sgn = is_signed_int(self.env, elem);
                        let lsz = esz.clamp(1, 8);
                        self.buf.load_mem(Reg::Rax, abase, aoff, lsz, sgn);
                        for lane in 1..(len as usize) {
                            self.buf.load_mem(Reg::Rcx, abase, aoff + (lane * esz) as i32, lsz, sgn);
                            self.buf.add_rr(Reg::Rax, Reg::Rcx);
                            self.emit_norm_subword_int(Reg::Rax, elem);
                        }
                        self.store_scalar_val(d, Reg::Rax, lsz);
                    }
                }
            }
            CfgBuiltinOp::VecReduceMin { vec_ty } | CfgBuiltinOp::VecReduceMax { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                let is_min = matches!(op, CfgBuiltinOp::VecReduceMin { .. });
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let (abase, aoff) = self.val_mem(args[0]);
                    if let Ty::Float(bits) = self.env.get(elem) {
                        let is_f16 = *bits == 16;
                        let is_f64 = *bits > 32;
                        if is_f16 {
                            self.buf.load_mem(Reg::Rdi, abase, aoff, 2, false);
                            self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rdi, 2);
                            for lane in 1..(len as usize) {
                                self.buf.load_mem(Reg::Rdi, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off + 4, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, abase, aoff + (lane * 2) as i32, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off + 4);
                                self.buf.sse_bin(false, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm0, Xmm::Xmm1);
                                self.helper_call("eon_f32_to_f16");
                                self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rax, 2);
                            }
                            self.buf.load_mem(Reg::Rax, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                            self.store_scalar_val(d, Reg::Rax, 2);
                        } else if is_f64 {
                            self.buf.movsd_load(Xmm::Xmm0, abase, aoff);
                            for lane in 1..(len as usize) {
                                self.buf.movsd_load(Xmm::Xmm1, abase, aoff + (lane * 8) as i32);
                                // x86 minsd(lane, acc): (lane < acc) ? lane : acc
                                self.buf.sse_bin(true, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm1, Xmm::Xmm0);
                                self.buf.movsd_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm1);
                                self.buf.movsd_load(Xmm::Xmm0, Reg::Rbp, self.plan.temp_copy_off);
                            }
                            self.store_float_val(d, Xmm::Xmm0, true);
                        } else {
                            self.buf.movss_load(Xmm::Xmm0, abase, aoff);
                            for lane in 1..(len as usize) {
                                self.buf.movss_load(Xmm::Xmm1, abase, aoff + (lane * 4) as i32);
                                self.buf.sse_bin(false, if is_min { 0x5D } else { 0x5F }, Xmm::Xmm1, Xmm::Xmm0);
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm1);
                                self.buf.movss_load(Xmm::Xmm0, Reg::Rbp, self.plan.temp_copy_off);
                            }
                            self.store_float_val(d, Xmm::Xmm0, false);
                        }
                    } else {
                        let sgn = is_signed_int(self.env, elem);
                        let lsz = esz.clamp(1, 8);
                        self.buf.load_mem(Reg::Rax, abase, aoff, lsz, sgn);
                        for lane in 1..(len as usize) {
                            let l_keep = self.buf.new_label();
                            self.buf.load_mem(Reg::Rcx, abase, aoff + (lane * esz) as i32, lsz, sgn);
                            self.buf.cmp_rr(Reg::Rcx, Reg::Rax);
                            let keep_cc = match (is_min, sgn) {
                                (true, true) => Cond::Ge,
                                (true, false) => Cond::Ae,
                                (false, true) => Cond::Le,
                                (false, false) => Cond::Be,
                            };
                            self.buf.jcc_label(keep_cc, l_keep);
                            self.buf.mov_rr(Reg::Rax, Reg::Rcx);
                            self.buf.bind_label(l_keep);
                        }
                        self.store_scalar_val(d, Reg::Rax, lsz);
                    }
                }
            }
            CfgBuiltinOp::VecDot { vec_ty } => {
                let Some(d) = dst else { return Ok(()); };
                if let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() {
                    let esz = x64_layout(self.env, elem).size.max(1);
                    let (abase, aoff) = self.val_mem(args[0]);
                    let (bbase, boff) = self.val_mem(args[1]);
                    if let Ty::Float(bits) = self.env.get(elem) {
                        let is_f16 = *bits == 16;
                        let is_f64 = *bits > 32;
                        if is_f16 {
                            for lane in 0..(len as usize) {
                                self.buf.load_mem(Reg::Rdi, bbase, boff + (lane * 2) as i32, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off + 4, Xmm::Xmm0);
                                self.buf.load_mem(Reg::Rdi, abase, aoff + (lane * 2) as i32, 2, false);
                                self.helper_call("eon_f16_to_f32");
                                self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off + 4);
                                self.buf.sse_bin(false, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                self.helper_call("eon_f32_to_f16");
                                if lane == 0 {
                                    self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rax, 2);
                                } else {
                                    self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off + 8, Reg::Rax, 2);
                                    self.buf.load_mem(Reg::Rdi, Reg::Rbp, self.plan.temp_copy_off + 8, 2, false);
                                    self.helper_call("eon_f16_to_f32");
                                    self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off + 4, Xmm::Xmm0);
                                    self.buf.load_mem(Reg::Rdi, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                                    self.helper_call("eon_f16_to_f32");
                                    self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off + 4);
                                    self.buf.sse_bin(false, 0x58, Xmm::Xmm0, Xmm::Xmm1);
                                    self.helper_call("eon_f32_to_f16");
                                    self.buf.store_mem(Reg::Rbp, self.plan.temp_copy_off, Reg::Rax, 2);
                                }
                            }
                            self.buf.load_mem(Reg::Rax, Reg::Rbp, self.plan.temp_copy_off, 2, false);
                            self.store_scalar_val(d, Reg::Rax, 2);
                        } else if is_f64 {
                            for lane in 0..(len as usize) {
                                self.buf.movsd_load(Xmm::Xmm0, abase, aoff + (lane * 8) as i32);
                                self.buf.movsd_load(Xmm::Xmm1, bbase, boff + (lane * 8) as i32);
                                self.buf.sse_bin(true, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                if lane > 0 {
                                    self.buf.movsd_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                                    self.buf.sse_bin(true, 0x58, Xmm::Xmm1, Xmm::Xmm0);
                                    self.buf.movsd_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm1);
                                } else {
                                    self.buf.movsd_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                                }
                            }
                            self.buf.movsd_load(Xmm::Xmm0, Reg::Rbp, self.plan.temp_copy_off);
                            self.store_float_val(d, Xmm::Xmm0, true);
                        } else {
                            for lane in 0..(len as usize) {
                                self.buf.movss_load(Xmm::Xmm0, abase, aoff + (lane * 4) as i32);
                                self.buf.movss_load(Xmm::Xmm1, bbase, boff + (lane * 4) as i32);
                                self.buf.sse_bin(false, 0x59, Xmm::Xmm0, Xmm::Xmm1);
                                if lane > 0 {
                                    self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, self.plan.temp_copy_off);
                                    self.buf.sse_bin(false, 0x58, Xmm::Xmm1, Xmm::Xmm0);
                                    self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm1);
                                } else {
                                    self.buf.movss_store(Reg::Rbp, self.plan.temp_copy_off, Xmm::Xmm0);
                                }
                            }
                            self.buf.movss_load(Xmm::Xmm0, Reg::Rbp, self.plan.temp_copy_off);
                            self.store_float_val(d, Xmm::Xmm0, false);
                        }
                    } else {
                        let sgn = is_signed_int(self.env, elem);
                        let lsz = esz.clamp(1, 8);
                        for lane in 0..(len as usize) {
                            self.buf.load_mem(Reg::Rax, abase, aoff + (lane * esz) as i32, lsz, sgn);
                            self.buf.load_mem(Reg::Rcx, bbase, boff + (lane * esz) as i32, lsz, sgn);
                            self.buf.imul_rr(Reg::Rax, Reg::Rcx);
                            self.emit_norm_subword_int(Reg::Rax, elem);
                            if lane == 0 {
                                self.buf.mov_rr(Reg::R8, Reg::Rax);
                            } else {
                                self.buf.add_rr(Reg::R8, Reg::Rax);
                                self.emit_norm_subword_int(Reg::R8, elem);
                            }
                        }
                        self.store_scalar_val(d, Reg::R8, lsz);
                    }
                }
            }
            CfgBuiltinOp::VecShuffle { src_ty, idx_ty, two_src } => {
                let Some(d) = dst else { return Ok(()); };
                if let (Ty::Vec { elem, len: src_len }, Ty::Vec { elem: ielem, len: dst_len }) =
                    (self.env.get(*src_ty).clone(), self.env.get(*idx_ty).clone())
                {
                    let esz = x64_layout(self.env, elem).size.max(1).clamp(1, 8);
                    let isz = x64_layout(self.env, ielem).size.max(1).clamp(1, 8);
                    let (v0base, v0off) = self.val_mem(args[0]);
                    let (v1base, v1off) = if *two_src { self.val_mem(args[1]) } else { (v0base, v0off) };
                    let (ibase, idxoff) = self.val_mem(args[if *two_src { 2 } else { 1 }]);
                    let doff = self.val_stack_off(d);
                    let total = if *two_src { (src_len as u64) * 2 } else { src_len as u64 };
                    self.buf.lea_mem(Reg::R8, v0base, v0off);
                    if *two_src {
                        self.buf.lea_mem(Reg::R9, v1base, v1off);
                    }
                    for lane in 0..(dst_len as usize) {
                        self.buf.load_mem(Reg::Rax, ibase, idxoff + (lane * isz) as i32, isz, false);
                        if total.is_power_of_two() {
                            self.buf.and_ri32(Reg::Rax, (total - 1) as i32);
                            self.buf.mov_rr(Reg::Rdx, Reg::Rax);
                        } else {
                            self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
                            self.buf.mov_ri64(Reg::Rcx, total);
                            self.buf.div_r(Reg::Rcx);
                        }
                        if *two_src {
                            let l_v0 = self.buf.new_label();
                            let l_go = self.buf.new_label();
                            self.buf.mov_ri64(Reg::Rcx, src_len as u64);
                            self.buf.cmp_rr(Reg::Rdx, Reg::Rcx);
                            self.buf.jcc_label(Cond::B, l_v0);
                            self.buf.sub_rr(Reg::Rdx, Reg::Rcx);
                            self.buf.mov_rr(Reg::R10, Reg::R9);
                            self.buf.jmp_label(l_go);
                            self.buf.bind_label(l_v0);
                            self.buf.mov_rr(Reg::R10, Reg::R8);
                            self.buf.bind_label(l_go);
                        } else {
                            self.buf.mov_rr(Reg::R10, Reg::R8);
                        }
                        if esz != 1 {
                            self.buf.imul_rri32(Reg::Rdx, Reg::Rdx, esz as i32);
                        }
                        self.buf.add_rr(Reg::R10, Reg::Rdx);
                        self.buf.load_mem(Reg::Rax, Reg::R10, 0, esz, false);
                        self.buf.store_mem(Reg::Rbp, doff + (lane * esz) as i32, Reg::Rax, esz);
                    }
                }
            }
            CfgBuiltinOp::MathSqrt(ty) => {
                let Some(d) = dst else { return Ok(()); };
                match self.env.get(*ty) {
                    Ty::Float(16) => {
                        self.load_scalar_raw(Reg::Rdi, args[0], 2);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.sqrt_f(false, Xmm::Xmm0, Xmm::Xmm0);
                        self.helper_call("eon_f32_to_f16");
                        self.store_scalar_val(d, Reg::Rax, 2);
                    }
                    Ty::Float(32) => {
                        self.load_float_val(Xmm::Xmm0, args[0], false);
                        self.buf.sqrt_f(false, Xmm::Xmm0, Xmm::Xmm0);
                        self.store_float_val(d, Xmm::Xmm0, false);
                    }
                    _ => {
                        self.load_float_val(Xmm::Xmm0, args[0], true);
                        self.buf.sqrt_f(true, Xmm::Xmm0, Xmm::Xmm0);
                        self.store_float_val(d, Xmm::Xmm0, true);
                    }
                }
            }
            CfgBuiltinOp::MathFma(ty) => {
                let Some(d) = dst else { return Ok(()); };
                match self.env.get(*ty) {
                    Ty::Float(16) => {
                        let tmp = self.plan.temp_copy_off;
                        self.load_scalar_raw(Reg::Rdi, args[1], 2);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.movss_store(Reg::Rbp, tmp, Xmm::Xmm0);
                        self.load_scalar_raw(Reg::Rdi, args[2], 2);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.movss_store(Reg::Rbp, tmp + 4, Xmm::Xmm0);
                        self.load_scalar_raw(Reg::Rdi, args[0], 2);
                        self.helper_call("eon_f16_to_f32");
                        self.buf.movss_load(Xmm::Xmm1, Reg::Rbp, tmp);
                        self.buf.movss_load(Xmm::Xmm2, Reg::Rbp, tmp + 4);
                        self.buf.vfmadd213(false, Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
                        self.helper_call("eon_f32_to_f16");
                        self.store_scalar_val(d, Reg::Rax, 2);
                    }
                    Ty::Float(32) => {
                        self.load_float_val(Xmm::Xmm0, args[0], false);
                        self.load_float_val(Xmm::Xmm1, args[1], false);
                        self.load_float_val(Xmm::Xmm2, args[2], false);
                        self.buf.vfmadd213(false, Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
                        self.store_float_val(d, Xmm::Xmm0, false);
                    }
                    _ => {
                        self.load_float_val(Xmm::Xmm0, args[0], true);
                        self.load_float_val(Xmm::Xmm1, args[1], true);
                        self.load_float_val(Xmm::Xmm2, args[2], true);
                        self.buf.vfmadd213(true, Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
                        self.store_float_val(d, Xmm::Xmm0, true);
                    }
                }
            }
            CfgBuiltinOp::MathLzcnt(ty) => {
                let Some(d) = dst else { return Ok(()); };
                let bits = int_bits(self.env, *ty);
                let sz = (bits / 8).clamp(1, 8) as usize;
                self.load_scalar_raw(Reg::Rax, args[0], sz);
                if bits < 64 {
                    self.buf.extend_reg(Reg::Rax, Reg::Rax, bits, false);
                }
                match bits {
                    8 => {
                        self.buf.lzcnt(false, Reg::Rax, Reg::Rax);
                        self.buf.sub_ri32(Reg::Rax, 24);
                    }
                    16 => {
                        self.buf.lzcnt(false, Reg::Rax, Reg::Rax);
                        self.buf.sub_ri32(Reg::Rax, 16);
                    }
                    32 => self.buf.lzcnt(false, Reg::Rax, Reg::Rax),
                    _ => self.buf.lzcnt(true, Reg::Rax, Reg::Rax),
                }
                self.store_scalar_val(d, Reg::Rax, sz);
            }
            CfgBuiltinOp::MathTzcnt(ty) => {
                let Some(d) = dst else { return Ok(()); };
                let bits = int_bits(self.env, *ty);
                let sz = (bits / 8).clamp(1, 8) as usize;
                self.load_scalar_raw(Reg::Rax, args[0], sz);
                if bits < 64 {
                    self.buf.extend_reg(Reg::Rax, Reg::Rax, bits, false);
                }
                match bits {
                    8 => {
                        self.buf.or_ri32(Reg::Rax, 0x100);
                        self.buf.tzcnt(false, Reg::Rax, Reg::Rax);
                    }
                    16 => {
                        self.buf.or_ri32(Reg::Rax, 0x1_0000);
                        self.buf.tzcnt(false, Reg::Rax, Reg::Rax);
                    }
                    32 => self.buf.tzcnt(false, Reg::Rax, Reg::Rax),
                    _ => self.buf.tzcnt(true, Reg::Rax, Reg::Rax),
                }
                self.store_scalar_val(d, Reg::Rax, sz);
            }
            CfgBuiltinOp::MathPopcnt(ty) => {
                let Some(d) = dst else { return Ok(()); };
                let bits = int_bits(self.env, *ty);
                let sz = (bits / 8).clamp(1, 8) as usize;
                self.load_scalar_raw(Reg::Rax, args[0], sz);
                if bits < 64 {
                    self.buf.extend_reg(Reg::Rax, Reg::Rax, bits, false);
                }
                self.buf.popcnt(bits >= 64, Reg::Rax, Reg::Rax);
                self.store_scalar_val(d, Reg::Rax, sz);
            }
            CfgBuiltinOp::MathBswap(ty) => {
                let Some(d) = dst else { return Ok(()); };
                let bits = int_bits(self.env, *ty);
                let sz = (bits / 8).clamp(1, 8) as usize;
                self.load_scalar_raw(Reg::Rax, args[0], sz);
                if sz >= 2 {
                    self.buf.bswap(sz, Reg::Rax);
                }
                self.emit_norm_subword_int(Reg::Rax, *ty);
                self.store_scalar_val(d, Reg::Rax, sz);
            }
            CfgBuiltinOp::AllocCapSubBudget => {
                let Some(d) = dst else { return Ok(()); };
                let doff = self.val_stack_off(d);
                self.emit_memzero(Reg::Rbp, doff, 40);
                self.load_scalar_val(Reg::Rax, args[1]); // limit
                self.buf.store_mem(Reg::Rbp, doff + 24, Reg::Rax, 8);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0); // parent_ptr
                self.buf.store_mem(Reg::Rbp, doff + 32, Reg::R8, 8);
            }
            CfgBuiltinOp::ClockNowNs => {
                self.helper_call("eon_rt_now_ns");
                if let Some(d) = dst {
                    self.store_scalar_val(d, Reg::Rax, 8);
                }
            }
            CfgBuiltinOp::ArgsLen => {
                let Some(d) = dst else { return Ok(()); };
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
                self.store_scalar_val(d, Reg::Rax, 8);
            }
            CfgBuiltinOp::ArgsGet => {
                let Some(d) = dst else { return Ok(()); };
                let doff = self.val_stack_off(d);
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::R9, Reg::R8, 0, 8, false); // argc
                self.load_scalar_val(Reg::R10, args[1]); // idx
                let l_oob = self.buf.new_label();
                let l_strlen = self.buf.new_label();
                let l_strlen_done = self.buf.new_label();
                let l_done = self.buf.new_label();
                self.buf.cmp_rr(Reg::R10, Reg::R9);
                self.buf.jcc_label(Cond::Ae, l_oob);
                self.buf.load_mem(Reg::R11, Reg::R8, 8, 8, false); // argv
                self.buf.shl_imm(Reg::R10, 3);
                self.buf.add_rr(Reg::R11, Reg::R10);
                self.buf.load_mem(Reg::Rax, Reg::R11, 0, 8, false); // cstr
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::E, l_oob);
                self.buf.xor_rr32(Reg::Rcx, Reg::Rcx);
                self.buf.bind_label(l_strlen);
                self.buf.mov_rr(Reg::Rdi, Reg::Rax);
                self.buf.add_rr(Reg::Rdi, Reg::Rcx);
                self.buf.load_mem(Reg::Rdx, Reg::Rdi, 0, 1, false);
                self.buf.test_rr32(Reg::Rdx, Reg::Rdx);
                self.buf.jcc_label(Cond::E, l_strlen_done);
                self.buf.add_ri32(Reg::Rcx, 1);
                self.buf.jmp_label(l_strlen);
                self.buf.bind_label(l_strlen_done);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rcx, 8);
                self.buf.jmp_label(l_done);
                self.buf.bind_label(l_oob);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 8);
                self.buf.mov_ri64(Reg::Rcx, u64::MAX);
                self.buf.store_mem(Reg::Rbp, doff + 8, Reg::Rcx, 8);
                self.buf.bind_label(l_done);
            }
            CfgBuiltinOp::FsOpenDir => {
                self.emit_fs_open_dir(false, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::DirSubDir => {
                self.emit_fs_open_dir(true, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::FsReadFile => {
                self.emit_fs_read_file(false, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::DirReadFile => {
                self.emit_fs_read_file(true, dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::FsWriteFile => {
                self.emit_fs_write_file(false, false, dst.unwrap(), args, arg_slots);
            }
            CfgBuiltinOp::DirWriteFile => {
                self.emit_fs_write_file(true, false, dst.unwrap(), args, arg_slots);
            }
            CfgBuiltinOp::FsWriteExecutable => {
                self.emit_fs_write_file(false, true, dst.unwrap(), args, arg_slots);
            }
            CfgBuiltinOp::FsIsDir => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
                let (pbase, poff) = self.val_mem(args[1]);
                self.buf.load_mem(Reg::Rsi, pbase, poff, 8, false);
                self.buf.load_mem(Reg::Rdx, pbase, poff + 8, 8, false);
                self.helper_call("eon_rt_is_dir");
                self.store_scalar_val(dst.unwrap(), Reg::Rax, 1);
            }
            CfgBuiltinOp::FsMkdirAll => {
                self.emit_fs_mkdir_all(dst.unwrap(), args, arg_slots);
            }
            CfgBuiltinOp::FsListDir => {
                self.emit_fs_list_dir(dst.unwrap(), args, arg_slots, ret_ty);
            }
            CfgBuiltinOp::FsSpawnWait => {
                self.emit_fs_spawn_wait(dst.unwrap(), args, ret_ty);
            }
            CfgBuiltinOp::DirClose => {
                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
                self.buf.mov_ri64(Reg::Rax, u64::MAX);
                self.buf.store_mem(Reg::R8, 0, Reg::Rax, 8);
                let l_skip = self.buf.new_label();
                self.buf.cmp_ri32(Reg::Rdi, 0);
                self.buf.jcc_label(Cond::L, l_skip);
                self.helper_call("eon_rt_close_fd");
                self.buf.bind_label(l_skip);
            }
            CfgBuiltinOp::NetRestrictHost => {
                if let Some(d) = dst {
                    self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                    self.buf.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
                    self.buf.add_ri32(Reg::Rax, 1);
                    self.store_scalar_val(d, Reg::Rax, 8);
                }
            }
            CfgBuiltinOp::IoNopRoundtrip => {
                self.emit_io_nop_roundtrip(dst.unwrap(), args[1], ret_ty);
            }
            CfgBuiltinOp::IoRunFibers => {
                self.emit_io_run_fibers(dst.unwrap(), args[1], args[2], ret_ty);
            }
            CfgBuiltinOp::BuildAddModule
            | CfgBuiltinOp::BuildAddExe
            | CfgBuiltinOp::BuildAddSharedLib
            | CfgBuiltinOp::BuildSetOutDir
            | CfgBuiltinOp::BuildSetSanitize
            | CfgBuiltinOp::BuildSetAuditTcb
            | CfgBuiltinOp::BuildSetOptLevel
            | CfgBuiltinOp::BuildLinkLib
            | CfgBuiltinOp::BuildAllowHal => {}
            CfgBuiltinOp::BuildOutPath => {
                if let Some(d) = dst {
                    let doff = self.val_stack_off(d);
                    self.emit_memzero(Reg::Rbp, doff, 16);
                }
            }
            CfgBuiltinOp::ScheduleRun => {
                self.emit_schedule_run(args, arg_slots)?;
            }
            CfgBuiltinOp::GpuGlobalId
            | CfgBuiltinOp::GpuLocalId
            | CfgBuiltinOp::GpuWorkgroupId
            | CfgBuiltinOp::GpuWorkgroupBarrier => {
                return Err(format!(
                    "ICE [codegen_x64]: GPU intrinsic `{op:?}` cannot be emitted in CPU x86_64 function `{}`",
                    self.interner.get(self.f.name)
                ));
            }
        }
        Ok(())
    }

    /// Copy `path_val: *str` to a NUL-terminated 16B-aligned stack buffer, checking `len > 0 && len < 4096`
    /// and rejecting embedded NUL bytes -> `l_inval`. On return, `Rax` holds `fd` or `-errno`.
    fn emit_open_str(
        &mut self,
        confined: bool,
        dirfd_reg: Reg,
        path_val: EirValue,
        flags: u64,
        mode: u64,
        l_inval: Label,
    ) {
        let tmp = self.plan.temp_copy_off;
        self.buf.store_mem(Reg::Rbp, tmp, dirfd_reg, 8);
        let (pbase, poff) = self.val_mem(path_val);
        self.buf.load_mem(Reg::Rsi, pbase, poff, 8, false);
        self.buf.load_mem(Reg::Rcx, pbase, poff + 8, 8, false);
        self.buf.test_rr(Reg::Rsi, Reg::Rsi);
        self.buf.jcc_label(Cond::E, l_inval);
        self.buf.test_rr(Reg::Rcx, Reg::Rcx);
        self.buf.jcc_label(Cond::E, l_inval);
        self.buf.mov_ri64(Reg::Rax, 4096);
        self.buf.cmp_rr(Reg::Rcx, Reg::Rax);
        self.buf.jcc_label(Cond::Ae, l_inval);

        // Scan for embedded '\0' before subtracting RSP.
        self.buf.xor_rr32(Reg::R10, Reg::R10);
        let l_scan = self.buf.new_label();
        let l_scan_ok = self.buf.new_label();
        self.buf.bind_label(l_scan);
        self.buf.cmp_rr(Reg::R10, Reg::Rcx);
        self.buf.jcc_label(Cond::Ae, l_scan_ok);
        self.buf.mov_rr(Reg::Rdi, Reg::Rsi);
        self.buf.add_rr(Reg::Rdi, Reg::R10);
        self.buf.load_mem(Reg::Rax, Reg::Rdi, 0, 1, false);
        self.buf.test_rr32(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_inval);
        self.buf.add_ri32(Reg::R10, 1);
        self.buf.jmp_label(l_scan);
        self.buf.bind_label(l_scan_ok);

        // Allocate align_up(len + 1, 16) = (rcx + 16) & -16 on stack.
        self.buf.lea_mem(Reg::Rax, Reg::Rcx, 16);
        self.buf.and_ri32(Reg::Rax, !15);
        self.buf.sub_rr(Reg::Rsp, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::Rax, 8);

        self.buf.mov_rr(Reg::Rdi, Reg::Rsp);
        self.buf.rep_movsb();
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rdi, 0, Reg::Rax, 1); // NUL terminator

        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp, 8, false); // dirfd
        self.buf.mov_rr(Reg::Rsi, Reg::Rsp);
        self.buf.mov_ri64(Reg::Rdx, flags);
        self.buf.mov_ri64(Reg::Rcx, mode);
        if confined {
            self.helper_call("eon_rt_openat2_beneath");
        } else {
            self.buf.mov_rr(Reg::R10, Reg::Rcx);
            self.buf.mov_ri64(Reg::Rax, 257); // SYS_openat
            self.buf.syscall();
        }
        self.buf.mov_rr(Reg::R8, Reg::Rax);
        self.buf.load_mem(Reg::R9, Reg::Rbp, tmp + 8, 8, false);
        self.buf.add_rr(Reg::Rsp, Reg::R9);
        self.buf.mov_rr(Reg::Rax, Reg::R8);
    }

    fn emit_map_errno_to_io_error(&mut self, errno_reg: Reg, dst_reg: Reg) {
        let l_done = self.buf.new_label();
        let cases: &[(&[i32], u64)] = &[
            (&[-2, -20], 12),      // ENOENT | ENOTDIR -> NotFound
            (&[-13, -1], 13),      // EACCES | EPERM -> PermissionDenied
            (&[-18, -40], 14),     // EXDEV | ELOOP -> PathEscape
            (&[-17], 15),          // EEXIST -> AlreadyExists
            (&[-22, -36, -21], 16), // EINVAL | ENAMETOOLONG | EISDIR -> InvalidInput
        ];
        for &(errnos, tag) in cases {
            let l_hit = self.buf.new_label();
            let l_next = self.buf.new_label();
            for &e in errnos {
                self.buf.cmp_ri32(errno_reg, e);
                self.buf.jcc_label(Cond::E, l_hit);
            }
            self.buf.jmp_label(l_next);
            self.buf.bind_label(l_hit);
            self.buf.mov_ri64(dst_reg, tag);
            self.buf.jmp_label(l_done);
            self.buf.bind_label(l_next);
        }
        self.buf.mov_ri64(dst_reg, 17); // Other
        self.buf.bind_label(l_done);
    }

    fn emit_fs_open_dir(
        &mut self,
        confined: bool,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => ret_ty,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let l_inval = self.buf.new_label();
        let l_err = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        self.emit_open_str(confined, Reg::Rdi, args[1], 0o220_0000, 0, l_inval);
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_err);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_inval);
        self.buf.mov_ri64(Reg::Rax, (-22i64) as u64);
        self.buf.bind_label(l_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.mov_ri64(Reg::Rax, u64::MAX);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.bind_label(l_done);
    }

    fn emit_fs_read_file(
        &mut self,
        confined: bool,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => ret_ty,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let tmp = self.plan.temp_copy_off;
        // tmp+16=fd, tmp+24=alloc_cap, tmp+32=buf_ptr, tmp+40=len, tmp+48=cap, tmp+56=scratch
        let l_inval = self.buf.new_label();
        let l_open_err = self.buf.new_label();
        let l_oom_close = self.buf.new_label();
        let l_oom_free = self.buf.new_label();
        let l_read_err = self.buf.new_label();
        let l_read_lp = self.buf.new_label();
        let l_have_room = self.buf.new_label();
        let l_eof = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        self.emit_open_str(confined, Reg::Rdi, args[2], 0o200_0000, 0, l_inval);
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_open_err);
        self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8); // fd

        self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1); // AllocCap*
        self.buf.store_mem(Reg::Rbp, tmp + 24, Reg::Rdi, 8);
        self.buf.mov_ri64(Reg::Rsi, 4096);
        self.buf.mov_ri64(Reg::Rdx, 16);
        self.helper_call("eon_alloc_raw");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_oom_close);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8); // buf_ptr
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 40, Reg::Rax, 8); // len = 0
        self.buf.mov_ri64(Reg::Rax, 4096);
        self.buf.store_mem(Reg::Rbp, tmp + 48, Reg::Rax, 8); // cap = 4096

        self.buf.bind_label(l_read_lp);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 40, 8, false); // len
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 48, 8, false); // cap
        self.buf.cmp_rr(Reg::Rcx, Reg::R8);
        self.buf.jcc_label(Cond::B, l_have_room);

        // Grow capacity * 2
        self.buf.shl_imm(Reg::R8, 1);
        self.buf.store_mem(Reg::Rbp, tmp + 56, Reg::R8, 8); // new_cap
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.mov_rr(Reg::Rsi, Reg::R8);
        self.buf.mov_ri64(Reg::Rdx, 16);
        self.helper_call("eon_alloc_raw");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_oom_free);
        // Copy old bytes into new buffer
        self.buf.mov_rr(Reg::R10, Reg::Rax); // new_buf
        self.buf.mov_rr(Reg::Rdi, Reg::R10);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 40, 8, false);
        self.buf.rep_movsb();
        // Save new_buf over tmp+64 temporarily (we bumped max_temp to 96!)
        self.buf.store_mem(Reg::Rbp, tmp + 64, Reg::R10, 8);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 64, 8, false);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 56, 8, false);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8);
        self.buf.store_mem(Reg::Rbp, tmp + 48, Reg::R8, 8);

        self.buf.bind_label(l_have_room);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false); // fd
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false);
        self.buf.add_rr(Reg::Rsi, Reg::Rax);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.buf.sub_rr(Reg::Rdx, Reg::Rax);
        self.helper_call("eon_rt_read_fd");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_eof);
        self.buf.cmp_ri32(Reg::Rax, 0);
        let l_read_pos = self.buf.new_label();
        self.buf.jcc_label(Cond::G, l_read_pos);
        self.buf.cmp_ri32(Reg::Rax, -4); // EINTR
        self.buf.jcc_label(Cond::E, l_read_lp);
        self.buf.jmp_label(l_read_err);

        self.buf.bind_label(l_read_pos);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 40, 8, false);
        self.buf.add_rr(Reg::Rcx, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 40, Reg::Rcx, 8);
        self.buf.jmp_label(l_read_lp);

        self.buf.bind_label(l_eof);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 32, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 48, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 16, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_read_err);
        self.buf.store_mem(Reg::Rbp, tmp + 56, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 56, 8, false);
        self.buf.jmp_label(l_open_err);

        self.buf.bind_label(l_oom_free);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.bind_label(l_oom_close);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.mov_ri64(Reg::Rax, 7); // OutOfMemory
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.emit_memzero(Reg::Rbp, doff + ok_off, 24);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_inval);
        self.buf.mov_ri64(Reg::Rax, (-22i64) as u64);
        self.buf.bind_label(l_open_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.emit_memzero(Reg::Rbp, doff + ok_off, 24);
        self.buf.bind_label(l_done);
    }

    fn emit_fs_write_file(
        &mut self,
        confined: bool,
        executable: bool,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
    ) {
        let doff = self.val_stack_off(dst);
        let tmp = self.plan.temp_copy_off;
        // O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC = 0x80241, mode = 0o755 (exec) or 0o644
        let mode = if executable { 0o755 } else { 0o644 };
        let l_inval = self.buf.new_label();
        let l_open_err = self.buf.new_label();
        let l_wr_lp = self.buf.new_label();
        let l_wr_ok = self.buf.new_label();
        let l_wr_err = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        self.emit_open_str(confined, Reg::Rdi, args[1], (1 | 0o100 | 0o1000 | 0o200_0000) as u64, mode, l_inval);
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_open_err);
        self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8); // fd

        let (dbase, data_off) = self.val_mem(args[2]);
        self.buf.load_mem(Reg::Rax, dbase, data_off, 8, false);
        self.buf.store_mem(Reg::Rbp, tmp + 24, Reg::Rax, 8); // ptr
        self.buf.load_mem(Reg::Rax, dbase, data_off + 8, 8, false);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8); // remaining

        self.buf.bind_label(l_wr_lp);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 32, 8, false);
        self.buf.test_rr(Reg::Rdx, Reg::Rdx);
        self.buf.jcc_label(Cond::E, l_wr_ok);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 24, 8, false);
        self.helper_call("eon_rt_write_fd");
        self.buf.cmp_ri32(Reg::Rax, 0);
        let l_wr_pos = self.buf.new_label();
        self.buf.jcc_label(Cond::G, l_wr_pos);
        self.buf.cmp_ri32(Reg::Rax, -4); // EINTR
        self.buf.jcc_label(Cond::E, l_wr_lp);
        self.buf.jmp_label(l_wr_err);

        self.buf.bind_label(l_wr_pos);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 24, 8, false);
        self.buf.add_rr(Reg::Rcx, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 24, Reg::Rcx, 8);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 32, 8, false);
        self.buf.sub_rr(Reg::Rcx, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rcx, 8);
        self.buf.jmp_label(l_wr_lp);

        self.buf.bind_label(l_wr_ok);
        if executable {
            // Apply SYS_fchmod(91)(fd, 0o755) so umask or pre-existing mode never strips +x.
            self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
            self.buf.mov_ri64(Reg::Rsi, 0o755);
            self.buf.mov_ri64(Reg::Rax, 91); // SYS_fchmod
            self.buf.syscall();
            self.buf.cmp_ri32(Reg::Rax, 0);
            self.buf.jcc_label(Cond::L, l_wr_err);
        }
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_wr_err);
        self.buf.store_mem(Reg::Rbp, tmp + 40, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false);
        self.buf.jmp_label(l_open_err);

        self.buf.bind_label(l_inval);
        self.buf.mov_ri64(Reg::Rax, (-22i64) as u64);
        self.buf.bind_label(l_open_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }

    fn emit_fs_mkdir_all(
        &mut self,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
    ) {
        let doff = self.val_stack_off(dst);
        let l_err = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        let (pbase, poff) = self.val_mem(args[1]);
        self.buf.load_mem(Reg::Rsi, pbase, poff, 8, false);
        self.buf.load_mem(Reg::Rdx, pbase, poff + 8, 8, false);
        self.helper_call("eon_rt_mkdir_all");
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_err);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }

    fn emit_fs_list_dir(
        &mut self,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => ret_ty,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let tmp = self.plan.temp_copy_off;
        // Scratch layout (starting at tmp+16 so tmp+0..16 stays free for emit_open_str/load_ptr_arg):
        //   tmp+16=fd, tmp+24=alloc_cap, tmp+32=buf_ptr, tmp+40=len, tmp+48=cap,
        //   tmp+56=nread, tmp+64=pos, tmp+72=entry_name_ptr, tmp+80=entry_name_len,
        //   tmp+88=d_reclen, tmp+96=new_cap/saved_errno, tmp+104=new_buf
        let l_inval = self.buf.new_label();
        let l_open_err = self.buf.new_label();
        let l_oom_close = self.buf.new_label();
        let l_oom_free = self.buf.new_label();
        let l_read_err = self.buf.new_label();
        let l_getdents_lp = self.buf.new_label();
        let l_have_ents = self.buf.new_label();
        let l_ent_lp = self.buf.new_label();
        let l_strlen_lp = self.buf.new_label();
        let l_strlen_done = self.buf.new_label();
        let l_chk_dotdot = self.buf.new_label();
        let l_append = self.buf.new_label();
        let l_cap_chk = self.buf.new_label();
        let l_cap_ok = self.buf.new_label();
        let l_ent_next = self.buf.new_label();
        let l_eof = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        // O_RDONLY | O_DIRECTORY | O_CLOEXEC = 0x90000
        self.emit_open_str(false, Reg::Rdi, args[2], 0x90000, 0, l_inval);
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_open_err);
        self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8); // fd

        self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1); // AllocCap*
        self.buf.store_mem(Reg::Rbp, tmp + 24, Reg::Rdi, 8);
        self.buf.mov_ri64(Reg::Rsi, 4096);
        self.buf.mov_ri64(Reg::Rdx, 16);
        self.helper_call("eon_alloc_raw");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_oom_close);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8); // buf_ptr
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 40, Reg::Rax, 8); // len = 0
        self.buf.mov_ri64(Reg::Rax, 4096);
        self.buf.store_mem(Reg::Rbp, tmp + 48, Reg::Rax, 8); // cap = 4096

        // Allocate 4096B stack scratch for getdents64 (preserves 16-byte RSP alignment).
        self.buf.sub_ri32(Reg::Rsp, 4096);

        self.buf.bind_label(l_getdents_lp);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.buf.mov_rr(Reg::Rsi, Reg::Rsp);
        self.buf.mov_ri64(Reg::Rdx, 4096);
        self.buf.mov_ri64(Reg::Rax, 217); // SYS_getdents64
        self.buf.syscall();
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_eof);
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::G, l_have_ents);
        self.buf.cmp_ri32(Reg::Rax, -4); // EINTR
        self.buf.jcc_label(Cond::E, l_getdents_lp);
        self.buf.jmp_label(l_read_err);

        self.buf.bind_label(l_have_ents);
        self.buf.store_mem(Reg::Rbp, tmp + 56, Reg::Rax, 8); // nread
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, tmp + 64, Reg::Rax, 8); // pos = 0

        self.buf.bind_label(l_ent_lp);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 64, 8, false); // pos
        self.buf.load_mem(Reg::R9, Reg::Rbp, tmp + 56, 8, false); // nread
        self.buf.cmp_rr(Reg::R8, Reg::R9);
        self.buf.jcc_label(Cond::Ae, l_getdents_lp);

        self.buf.mov_rr(Reg::R10, Reg::Rsp);
        self.buf.add_rr(Reg::R10, Reg::R8); // ent = scratch + pos
        self.buf.load_mem(Reg::Rax, Reg::R10, 16, 2, false); // d_reclen: u16
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_getdents_lp);
        self.buf.store_mem(Reg::Rbp, tmp + 88, Reg::Rax, 8);

        // name_ptr = ent + 19; compute strlen in rcx.
        self.buf.lea_mem(Reg::R11, Reg::R10, 19);
        self.buf.xor_rr32(Reg::Rcx, Reg::Rcx);
        self.buf.bind_label(l_strlen_lp);
        self.buf.mov_rr(Reg::Rdi, Reg::R11);
        self.buf.add_rr(Reg::Rdi, Reg::Rcx);
        self.buf.load_mem(Reg::Rax, Reg::Rdi, 0, 1, false);
        self.buf.test_rr32(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_strlen_done);
        self.buf.add_ri32(Reg::Rcx, 1);
        self.buf.jmp_label(l_strlen_lp);

        self.buf.bind_label(l_strlen_done);
        self.buf.test_rr(Reg::Rcx, Reg::Rcx);
        self.buf.jcc_label(Cond::E, l_ent_next);
        // Skip "." and ".."
        self.buf.cmp_ri32(Reg::Rcx, 1);
        self.buf.jcc_label(Cond::Ne, l_chk_dotdot);
        self.buf.load_mem(Reg::Rax, Reg::R11, 0, 1, false);
        self.buf.cmp_ri32(Reg::Rax, b'.' as i32);
        self.buf.jcc_label(Cond::E, l_ent_next);
        self.buf.jmp_label(l_append);

        self.buf.bind_label(l_chk_dotdot);
        self.buf.cmp_ri32(Reg::Rcx, 2);
        self.buf.jcc_label(Cond::Ne, l_append);
        self.buf.load_mem(Reg::Rax, Reg::R11, 0, 1, false);
        self.buf.cmp_ri32(Reg::Rax, b'.' as i32);
        self.buf.jcc_label(Cond::Ne, l_append);
        self.buf.load_mem(Reg::Rax, Reg::R11, 1, 1, false);
        self.buf.cmp_ri32(Reg::Rax, b'.' as i32);
        self.buf.jcc_label(Cond::E, l_ent_next);

        self.buf.bind_label(l_append);
        self.buf.store_mem(Reg::Rbp, tmp + 72, Reg::R11, 8); // entry_name_ptr
        self.buf.store_mem(Reg::Rbp, tmp + 80, Reg::Rcx, 8); // entry_name_len

        // Ensure len + entry_name_len + 1 <= cap (grow * 2 via eon_alloc_raw / eon_free_raw).
        self.buf.bind_label(l_cap_chk);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false); // len
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 80, 8, false); // name_len
        self.buf.add_rr(Reg::Rax, Reg::Rdx);
        self.buf.add_ri32(Reg::Rax, 1);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 48, 8, false); // cap
        self.buf.cmp_rr(Reg::Rax, Reg::R8);
        self.buf.jcc_label(Cond::Be, l_cap_ok);

        self.buf.shl_imm(Reg::R8, 1);
        self.buf.store_mem(Reg::Rbp, tmp + 96, Reg::R8, 8); // new_cap
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.mov_rr(Reg::Rsi, Reg::R8);
        self.buf.mov_ri64(Reg::Rdx, 16);
        self.helper_call("eon_alloc_raw");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_oom_free);
        self.buf.store_mem(Reg::Rbp, tmp + 104, Reg::Rax, 8); // new_buf
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 40, 8, false);
        self.buf.rep_movsb();
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 104, 8, false);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 96, 8, false);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8);
        self.buf.store_mem(Reg::Rbp, tmp + 48, Reg::R8, 8);
        self.buf.jmp_label(l_cap_chk);

        self.buf.bind_label(l_cap_ok);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 32, 8, false); // buf_ptr
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false); // len
        self.buf.add_rr(Reg::Rdi, Reg::Rax);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 72, 8, false); // entry_name_ptr
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 80, 8, false); // entry_name_len
        self.buf.rep_movsb();
        self.buf.mov_ri64(Reg::Rax, 10); // '\n'
        self.buf.store_mem(Reg::Rdi, 0, Reg::Rax, 1);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 80, 8, false);
        self.buf.add_rr(Reg::Rax, Reg::Rdx);
        self.buf.add_ri32(Reg::Rax, 1);
        self.buf.store_mem(Reg::Rbp, tmp + 40, Reg::Rax, 8);

        self.buf.bind_label(l_ent_next);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp + 64, 8, false);
        self.buf.load_mem(Reg::R9, Reg::Rbp, tmp + 88, 8, false);
        self.buf.add_rr(Reg::R8, Reg::R9);
        self.buf.store_mem(Reg::Rbp, tmp + 64, Reg::R8, 8);
        self.buf.jmp_label(l_ent_lp);

        self.buf.bind_label(l_eof);
        self.buf.add_ri32(Reg::Rsp, 4096);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 32, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 40, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 48, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 16, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_read_err);
        self.buf.add_ri32(Reg::Rsp, 4096);
        self.buf.store_mem(Reg::Rbp, tmp + 96, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 96, 8, false);
        self.buf.jmp_label(l_open_err);

        self.buf.bind_label(l_oom_free);
        self.buf.add_ri32(Reg::Rsp, 4096);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 24, 8, false);
        self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 32, 8, false);
        self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp + 48, 8, false);
        self.helper_call("eon_free_raw");
        self.buf.bind_label(l_oom_close);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 16, 8, false);
        self.helper_call("eon_rt_close_fd");
        self.buf.mov_ri64(Reg::Rax, 7); // OutOfMemory
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.emit_memzero(Reg::Rbp, doff + ok_off, 24);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_inval);
        self.buf.mov_ri64(Reg::Rax, (-22i64) as u64);
        self.buf.bind_label(l_open_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.emit_memzero(Reg::Rbp, doff + ok_off, 24);
        self.buf.bind_label(l_done);
    }

    fn emit_fs_spawn_wait(
        &mut self,
        dst: EirValue,
        args: &[EirValue],
        ret_ty: TyId,
    ) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => TyId::I32,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let l_err = self.buf.new_label();
        let l_done = self.buf.new_label();

        let (pbase, poff) = self.val_mem(args[1]);
        self.buf.load_mem(Reg::Rdi, pbase, poff, 8, false);
        self.buf.load_mem(Reg::Rsi, pbase, poff + 8, 8, false);
        let (abase, aoff) = self.val_mem(args[2]);
        self.buf.load_mem(Reg::Rdx, abase, aoff, 8, false);
        self.buf.load_mem(Reg::Rcx, abase, aoff + 8, 8, false);
        self.helper_call("eon_rt_spawn_wait");
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_err);

        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 4);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_err);
        self.buf.mov_rr(Reg::Rdi, Reg::Rax);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }

    fn emit_io_nop_roundtrip(&mut self, dst: EirValue, count_v: EirValue, ret_ty: TyId) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => TyId::USIZE,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let tmp = self.plan.temp_copy_off;
        let l_zero = self.buf.new_label();
        let l_fail_close = self.buf.new_label();
        let l_fail = self.buf.new_label();
        let l_batch = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_scalar_val(Reg::Rax, count_v);
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_zero);
        self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::Rax, 8); // rem
        self.buf.xor_rr32(Reg::Rcx, Reg::Rcx);
        self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rcx, 8); // completed

        self.buf.mov_ri64(Reg::Rdi, 64);
        self.helper_call("eon_iouring_create");
        self.buf.test_rr(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_fail);
        self.buf.store_mem(Reg::Rbp, tmp, Reg::Rax, 8); // ring

        self.buf.bind_label(l_batch);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 8, 8, false); // rem
        self.buf.test_rr(Reg::Rcx, Reg::Rcx);
        let l_ring_ok = self.buf.new_label();
        self.buf.jcc_label(Cond::E, l_ring_ok);
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp, 8, false); // ring
        self.buf.load_mem(Reg::Rdx, Reg::R8, 120, 8, false); // sq_entries
        self.buf.cmp_rr(Reg::Rcx, Reg::Rdx);
        self.buf.cmovcc(Cond::A, Reg::Rcx, Reg::Rdx); // batch = min(rem, sq_entries)
        self.buf.store_mem(Reg::Rbp, tmp + 24, Reg::Rcx, 8);

        self.buf.load_mem(Reg::R9, Reg::R8, 64, 8, false); // sq_tail_ptr
        self.buf.load_mem(Reg::R9, Reg::R9, 0, 4, false); // sq_tail
        self.buf.load_mem(Reg::R10, Reg::R8, 72, 8, false); // sq_mask
        self.buf.load_mem(Reg::R11, Reg::R8, 40, 8, false); // sqes_ptr
        // Reuse Rdi in fill loop for k = 0..batch
        self.buf.xor_rr32(Reg::Rdi, Reg::Rdi);
        let l_fill = self.buf.new_label();
        let l_fill_done = self.buf.new_label();
        self.buf.bind_label(l_fill);
        self.buf.cmp_rr(Reg::Rdi, Reg::Rcx);
        self.buf.jcc_label(Cond::Ae, l_fill_done);
        self.buf.mov_rr(Reg::Rax, Reg::R9);
        self.buf.add_rr(Reg::Rax, Reg::Rdi);
        self.buf.and_rr(Reg::Rax, Reg::R10); // slot = (tail + k) & mask
        // Zero SQE at sqes_ptr + slot * 64 (opcode = 0 => IORING_OP_NOP)
        self.buf.mov_rr(Reg::Rsi, Reg::Rax);
        self.buf.shl_imm(Reg::Rsi, 6);
        self.buf.add_rr(Reg::Rsi, Reg::R11);
        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        for qoff in (0..64).step_by(8) {
            self.buf.store_mem(Reg::Rsi, qoff, Reg::Rdx, 8);
        }
        // sq_array_ptr[slot] = slot
        self.buf.load_mem(Reg::Rsi, Reg::R8, 80, 8, false);
        self.buf.mov_rr(Reg::Rdx, Reg::Rax);
        self.buf.shl_imm(Reg::Rdx, 2);
        self.buf.add_rr(Reg::Rsi, Reg::Rdx);
        self.buf.store_mem(Reg::Rsi, 0, Reg::Rax, 4);
        self.buf.add_ri32(Reg::Rdi, 1);
        self.buf.jmp_label(l_fill);
        self.buf.bind_label(l_fill_done);

        self.buf.sfence();
        self.buf.add_rr(Reg::R9, Reg::Rcx);
        self.buf.load_mem(Reg::Rdx, Reg::R8, 64, 8, false);
        self.buf.store_mem(Reg::Rdx, 0, Reg::R9, 4); // *sq_tail_ptr = tail + batch

        // SYS_io_uring_enter(fd, to_submit=batch, min_complete=batch, flags=1)
        self.buf.load_mem(Reg::Rdi, Reg::R8, 0, 8, false);
        self.buf.mov_rr(Reg::Rsi, Reg::Rcx);
        self.buf.mov_rr(Reg::Rdx, Reg::Rcx);
        self.buf.mov_ri64(Reg::R10, 1);
        self.buf.xor_rr32(Reg::R8, Reg::R8);
        self.buf.xor_rr32(Reg::R9, Reg::R9);
        self.buf.mov_ri64(Reg::Rax, 426);
        self.buf.syscall();
        self.buf.cmp_ri32(Reg::Rax, 0);
        self.buf.jcc_label(Cond::L, l_fail_close);

        // Advance cq_head by batch
        self.buf.load_mem(Reg::R8, Reg::Rbp, tmp, 8, false);
        self.buf.load_mem(Reg::R9, Reg::R8, 88, 8, false); // cq_head_ptr
        self.buf.load_mem(Reg::R10, Reg::R9, 0, 4, false);
        self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp + 24, 8, false);
        self.buf.add_rr(Reg::R10, Reg::Rcx);
        self.buf.store_mem(Reg::R9, 0, Reg::R10, 4);

        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 8, false);
        self.buf.add_rr(Reg::Rax, Reg::Rcx);
        self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 8, 8, false);
        self.buf.sub_rr(Reg::Rax, Reg::Rcx);
        self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::Rax, 8);
        self.buf.jmp_label(l_batch);

        self.buf.bind_label(l_ring_ok);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp, 8, false);
        self.helper_call("eon_iouring_destroy");
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 8, false);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_zero);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_fail_close);
        self.buf.store_mem(Reg::Rbp, tmp + 32, Reg::Rax, 8);
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp, 8, false);
        self.helper_call("eon_iouring_destroy");
        self.buf.load_mem(Reg::Rdi, Reg::Rbp, tmp + 32, 8, false);
        self.emit_map_errno_to_io_error(Reg::Rdi, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_fail);
        self.buf.mov_ri64(Reg::Rax, 17);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.bind_label(l_done);
    }

    fn emit_io_run_fibers(
        &mut self,
        dst: EirValue,
        count_v: EirValue,
        stack_bytes_v: EirValue,
        ret_ty: TyId,
    ) {
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) {
            Ty::ErrorUnion { ok, .. } => *ok,
            _ => TyId::USIZE,
        };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let l_inval = self.buf.new_label();
        let l_zero = self.buf.new_label();
        let l_err = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_scalar_val(Reg::Rsi, stack_bytes_v);
        self.buf.test_rr(Reg::Rsi, Reg::Rsi);
        self.buf.jcc_label(Cond::E, l_inval);
        self.load_scalar_val(Reg::Rdi, count_v);
        self.buf.test_rr(Reg::Rdi, Reg::Rdi);
        self.buf.jcc_label(Cond::E, l_zero);

        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx); // entry_fn = NULL
        self.buf.xor_rr32(Reg::Rcx, Reg::Rcx); // arg_ptr = NULL
        self.buf.mov_ri64(Reg::R8, u64::from(self.sanitize));
        self.helper_call("eon_run_fibers_iouring");
        self.load_scalar_val(Reg::Rcx, count_v);
        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
        self.buf.jcc_label(Cond::B, l_err);

        self.buf.xor_rr32(Reg::Rdx, Reg::Rdx);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rdx, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_zero);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_inval);
        self.buf.mov_ri64(Reg::Rax, 16);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_err);
        self.buf.mov_ri64(Reg::Rax, 17);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.bind_label(l_done);
    }

    fn resolve_const_schedule_idx(&self, mut v: EirValue) -> Option<usize> {
        for _ in 0..8 {
            let mut next: Option<EirValue> = None;
            let mut target_slot: Option<EirSlot> = None;
            for blk in &self.f.blocks {
                for inst in &blk.insts {
                    match &inst.kind {
                        EirInstKind::Const { dst, konst: CfgConst::Int(iv, _), .. } if *dst == v => {
                            return usize::try_from(*iv).ok();
                        }
                        EirInstKind::CopyVal { dst, src } if *dst == v => next = Some(*src),
                        EirInstKind::LoadSlot { dst, slot } if *dst == v => target_slot = Some(*slot),
                        _ => {}
                    }
                }
            }
            if let Some(nv) = next {
                v = nv;
                continue;
            }
            if let Some(sl) = target_slot {
                let mut found: Option<EirValue> = None;
                let mut multiple = false;
                for blk in &self.f.blocks {
                    for inst in &blk.insts {
                        if let EirInstKind::StoreSlot { slot, val } = &inst.kind
                            && *slot == sl
                        {
                            if found.is_some() {
                                multiple = true;
                            }
                            found = Some(*val);
                        }
                    }
                }
                if !multiple && let Some(sv) = found {
                    v = sv;
                    continue;
                }
            }
            return None;
        }
        None
    }

    fn match_schedule_arg(&self, caller_ty: TyId, param_ty: TyId) -> Option<u8> {
        // Return priority: 0 = Exact, 1 = MutToConst, 2 = Deref, 3 = Borrow
        if caller_ty == param_ty {
            return Some(0);
        }
        match (self.env.get(caller_ty), self.env.get(param_ty)) {
            (
                Ty::Borrow { is_mut: true, inner: ci },
                Ty::Borrow { is_mut: false, inner: pi },
            ) if ci == pi => Some(1),
            (Ty::Borrow { inner: BorrowInner::Single(ci), .. }, _) if *ci == param_ty => Some(2),
            (Ty::RawPtr { inner: ci, .. }, _) if *ci == param_ty => Some(2),
            (_, Ty::Borrow { inner: BorrowInner::Single(pi), .. }) if caller_ty == *pi => Some(3),
            (_, Ty::RawPtr { inner: pi, .. }) if caller_ty == *pi => Some(3),
            _ => None,
        }
    }

    fn is_pool_or_cap_ty(&self, ty: TyId) -> bool {
        let deref = match self.env.get(ty) {
            Ty::Borrow { inner: BorrowInner::Single(i), .. } | Ty::RawPtr { inner: i, .. } => *i,
            _ => ty,
        };
        matches!(
            self.env.get(deref),
            Ty::Builtin(
                BuiltinKind::ArenaPool
                    | BuiltinKind::SlotPool
                    | BuiltinKind::AllocCap
                    | BuiltinKind::FsCap
                    | BuiltinKind::DirCap
                    | BuiltinKind::NetCap
                    | BuiltinKind::ClockCap
                    | BuiltinKind::IoCap
                    | BuiltinKind::ArgsCap
                    | BuiltinKind::Env,
                _
            )
        )
    }

    fn resolve_sys_param_bindings(
        &self,
        sys_fn: &EirFunction,
        run_args: &[EirValue],
    ) -> Result<Vec<(usize, u8)>, String> {
        let mut used = vec![false; run_args.len()];
        let mut out = Vec::with_capacity(sys_fn.params.len());
        for (p_idx, &pv) in sys_fn.params.iter().enumerate() {
            let p_ty = sys_fn.value_tys[pv.0 as usize];
            let mut best: Option<(usize, u8)> = None;
            for (k, &av) in run_args.iter().enumerate() {
                if used[k] {
                    continue;
                }
                if let Some(kind) = self.match_schedule_arg(self.f.value_tys[av.0 as usize], p_ty)
                    && best.is_none_or(|(_, bk)| kind < bk)
                {
                    best = Some((k, kind));
                }
            }
            if best.is_none() && self.is_pool_or_cap_ty(p_ty) {
                for (k, &av) in run_args.iter().enumerate() {
                    if let Some(kind) = self.match_schedule_arg(self.f.value_tys[av.0 as usize], p_ty)
                        && best.is_none_or(|(_, bk)| kind < bk)
                    {
                        best = Some((k, kind));
                    }
                }
            }
            if let Some((k, kind)) = best {
                used[k] = true;
                out.push((k, kind));
            } else if p_idx < run_args.len() {
                out.push((p_idx, 0));
            } else {
                return Err(format!(
                    "ICE [codegen_x64]: missing argument {} for scheduled system `{}`",
                    p_idx,
                    self.interner.get(sys_fn.name)
                ));
            }
        }
        Ok(out)
    }

    fn emit_schedule_single_sys(
        &mut self,
        sys_def: DefId,
        run_args: &[EirValue],
        run_arg_slots: &[Option<EirSlot>],
    ) -> Result<(), String> {
        let Some(&fn_idx) = self.eir.by_key.get(&CfgFnKey::TopDef(sys_def)) else {
            return Err(format!("ICE [codegen_x64]: missing EIR for scheduled system `{sys_def:?}`"));
        };
        let sys_fn = &self.eir.functions[fn_idx];
        let bindings = self.resolve_sys_param_bindings(sys_fn, run_args)?;
        let sys_sym = fn_sym_name(CfgFnKey::TopDef(sys_def), self.eir, self.interner);

        enum SArg {
            Gp1 { reg: Reg, val: EirValue, kind: u8, sz: usize, arg_idx: usize },
            Gp1Ptr { reg: Reg, stack_off: i32 },
            Gp2 { r0: Reg, r1: Reg, val: EirValue, kind: u8, sz: usize },
            Float { xmm: Xmm, val: EirValue, kind: u8, is_f64: bool },
            Stack1 { val: EirValue, kind: u8, sz: usize, arg_idx: usize, stack_off: i32 },
            Stack2 { val: EirValue, kind: u8 },
        }
        let mut gp_i = 0usize;
        let mut xmm_i = 0usize;
        let mut cur_tmp_off = self.plan.temp_copy_off;
        let mut prepared = Vec::with_capacity(bindings.len());

        for (pi, &(k, kind)) in bindings.iter().enumerate() {
            let av = run_args[k];
            let p_ty = sys_fn.value_tys[sys_fn.params[pi].0 as usize];
            match classify_arg(self.env, p_ty) {
                ArgClass::Void => {}
                ArgClass::Float32 => {
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        prepared.push(SArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], val: av, kind, is_f64: false });
                        xmm_i += 1;
                    } else {
                        prepared.push(SArg::Stack1 { val: av, kind, sz: 4, arg_idx: k, stack_off: 0 });
                    }
                }
                ArgClass::Float64 => {
                    if xmm_i < FLOAT_ARG_REGS.len() {
                        prepared.push(SArg::Float { xmm: FLOAT_ARG_REGS[xmm_i], val: av, kind, is_f64: true });
                        xmm_i += 1;
                    } else {
                        prepared.push(SArg::Stack1 { val: av, kind, sz: 8, arg_idx: k, stack_off: 0 });
                    }
                }
                ArgClass::Reg1(sz) => {
                    let act_sz = if kind == 2 { sz } else { x64_layout(self.env, self.f.value_tys[av.0 as usize]).size.min(sz).max(1) };
                    if gp_i < EON_GP_ARG_REGS.len() {
                        prepared.push(SArg::Gp1 { reg: EON_GP_ARG_REGS[gp_i], val: av, kind, sz: act_sz, arg_idx: k });
                        gp_i += 1;
                    } else {
                        prepared.push(SArg::Stack1 { val: av, kind, sz: act_sz, arg_idx: k, stack_off: 0 });
                    }
                }
                ArgClass::Reg2(sz) => {
                    if gp_i + 1 < EON_GP_ARG_REGS.len() {
                        prepared.push(SArg::Gp2 { r0: EON_GP_ARG_REGS[gp_i], r1: EON_GP_ARG_REGS[gp_i + 1], val: av, kind, sz });
                        gp_i += 2;
                    } else {
                        prepared.push(SArg::Stack2 { val: av, kind });
                    }
                }
                ArgClass::Indirect(sz) => {
                    let copy_off = cur_tmp_off;
                    cur_tmp_off += align_up(sz, 16) as i32;
                    if kind == 2 {
                        self.load_scalar_raw(Reg::R9, av, 8);
                        if sz <= 64 {
                            self.copy_mem_using_rax(Reg::Rbp, copy_off, Reg::R9, 0, sz);
                        } else {
                            self.buf.lea_mem(Reg::Rdi, Reg::Rbp, copy_off);
                            self.buf.mov_rr(Reg::Rsi, Reg::R9);
                            self.buf.mov_ri64(Reg::Rcx, sz as u64);
                            self.buf.rep_movsb();
                        }
                    } else {
                        let (sbase, soff) = self.val_mem(av);
                        self.copy_mem_using_rax(Reg::Rbp, copy_off, sbase, soff, sz);
                    }
                    if gp_i < EON_GP_ARG_REGS.len() {
                        prepared.push(SArg::Gp1Ptr { reg: EON_GP_ARG_REGS[gp_i], stack_off: copy_off });
                        gp_i += 1;
                    } else {
                        prepared.push(SArg::Stack1 { val: av, kind: 4, sz: 8, arg_idx: k, stack_off: copy_off });
                    }
                }
            }
        }

        let stack_bytes: usize = prepared
            .iter()
            .filter_map(|p| match p {
                SArg::Stack1 { .. } => Some(8usize),
                SArg::Stack2 { .. } => Some(16usize),
                _ => None,
            })
            .sum();
        let aligned_stack = align_up(stack_bytes, 16);
        if aligned_stack > 0 {
            self.buf.sub_ri32(Reg::Rsp, aligned_stack as i32);
            let mut sp = 0i32;
            for p in &prepared {
                match p {
                    SArg::Stack1 { val, kind, sz, arg_idx, stack_off } => {
                        match *kind {
                            2 => {
                                self.load_scalar_raw(Reg::Rax, *val, 8);
                                self.buf.load_mem(Reg::Rax, Reg::Rax, 0, (*sz).clamp(1, 8), false);
                            }
                            3 => self.load_ptr_arg(Reg::Rax, run_args, run_arg_slots, *arg_idx),
                            4 => self.buf.lea_mem(Reg::Rax, Reg::Rbp, *stack_off),
                            _ => self.load_scalar_raw(Reg::Rax, *val, *sz),
                        }
                        self.buf.store_mem(Reg::Rsp, sp, Reg::Rax, 8);
                        sp += 8;
                    }
                    SArg::Stack2 { val, kind } => {
                        if *kind == 2 {
                            self.load_scalar_raw(Reg::R9, *val, 8);
                            self.buf.load_mem(Reg::Rax, Reg::R9, 0, 8, false);
                            self.buf.store_mem(Reg::Rsp, sp, Reg::Rax, 8);
                            self.buf.load_mem(Reg::Rax, Reg::R9, 8, 8, false);
                            self.buf.store_mem(Reg::Rsp, sp + 8, Reg::Rax, 8);
                        } else {
                            let (vbase, voff) = self.val_mem(*val);
                            self.buf.load_mem(Reg::Rax, vbase, voff, 8, false);
                            self.buf.store_mem(Reg::Rsp, sp, Reg::Rax, 8);
                            self.buf.load_mem(Reg::Rax, vbase, voff + 8, 8, false);
                            self.buf.store_mem(Reg::Rsp, sp + 8, Reg::Rax, 8);
                        }
                        sp += 16;
                    }
                    _ => {}
                }
            }
        }
        for p in &prepared {
            match p {
                SArg::Gp1 { reg, val, kind, sz, arg_idx } => match *kind {
                    2 => {
                        self.load_scalar_raw(Reg::Rax, *val, 8);
                        self.buf.load_mem(*reg, Reg::Rax, 0, (*sz).clamp(1, 8), false);
                    }
                    3 => self.load_ptr_arg(*reg, run_args, run_arg_slots, *arg_idx),
                    _ => self.load_scalar_raw(*reg, *val, *sz),
                },
                SArg::Gp1Ptr { reg, stack_off } => {
                    self.buf.lea_mem(*reg, Reg::Rbp, *stack_off);
                }
                SArg::Gp2 { r0, r1, val, kind, sz } => {
                    if *kind == 2 {
                        self.load_scalar_raw(Reg::Rax, *val, 8);
                        self.buf.load_mem(*r0, Reg::Rax, 0, 8, false);
                        if *sz > 8 {
                            self.buf.load_mem(*r1, Reg::Rax, 8, 8, false);
                        }
                    } else {
                        let (vbase, voff) = self.val_mem(*val);
                        self.buf.load_mem(*r0, vbase, voff, 8, false);
                        if *sz > 8 {
                            self.buf.load_mem(*r1, vbase, voff + 8, 8, false);
                        }
                    }
                }
                SArg::Float { xmm, val, kind, is_f64 } => {
                    if *kind == 2 {
                        self.load_scalar_raw(Reg::Rax, *val, 8);
                        if *is_f64 {
                            self.buf.movsd_load(*xmm, Reg::Rax, 0);
                        } else {
                            self.buf.movss_load(*xmm, Reg::Rax, 0);
                        }
                    } else {
                        self.load_float_val(*xmm, *val, *is_f64);
                    }
                }
                _ => {}
            }
        }
        self.buf.call_sym(sys_sym);
        if aligned_stack > 0 {
            self.buf.lea_mem(Reg::Rsp, Reg::Rsp, aligned_stack as i32);
        }
        Ok(())
    }

    fn emit_schedule_dag(
        &mut self,
        sched_idx: usize,
        run_args: &[EirValue],
        run_arg_slots: &[Option<EirSlot>],
    ) -> Result<(), String> {
        let Some(dag) = self.eir.schedules.get(sched_idx).cloned() else {
            return Ok(());
        };
        for wave in &dag.waves {
            if wave.systems.is_empty() {
                continue;
            }
            if wave.systems.len() == 1 {
                self.emit_schedule_single_sys(wave.systems[0], run_args, run_arg_slots)?;
                continue;
            }
            // Build stack bundle layout for all `run_args`: each arg gets value slot + 8B ptr slot.
            let mut val_offs = Vec::with_capacity(run_args.len());
            let mut ptr_offs = Vec::with_capacity(run_args.len());
            let mut bundle_sz = 0usize;
            for &av in run_args {
                let lay = x64_layout(self.env, self.f.value_tys[av.0 as usize]);
                bundle_sz = align_up(bundle_sz, lay.align.max(8));
                val_offs.push(bundle_sz);
                bundle_sz += align_up(lay.size.max(8), 8);
                ptr_offs.push(bundle_sz);
                bundle_sz += 8;
            }
            let nodes_bytes = align_up(wave.systems.len() * 32, 16);
            let bundle_bytes = align_up(bundle_sz, 16);
            let total_alloc = nodes_bytes + bundle_bytes;
            if total_alloc > 0 {
                self.buf.sub_ri32(Reg::Rsp, total_alloc as i32);
            }
            let bundle_disp = nodes_bytes as i32;
            // Populate arg_bundle freshly before each wave so mutations from earlier waves are visible!
            for (k, &av) in run_args.iter().enumerate() {
                let aty = self.f.value_tys[av.0 as usize];
                self.store_val_to_mem(Reg::Rsp, bundle_disp + val_offs[k] as i32, av, aty);
                self.load_ptr_arg(Reg::Rax, run_args, run_arg_slots, k);
                self.buf.store_mem(Reg::Rsp, bundle_disp + ptr_offs[k] as i32, Reg::Rax, 8);
            }
            self.buf.lea_mem(Reg::Rdx, Reg::Rsp, bundle_disp); // bundle_ptr in Rdx

            // Populate EonDagNode[num_sys] at [rsp + 0] and queue deferred wrappers.
            for (si, &sys_def) in wave.systems.iter().enumerate() {
                let Some(&fn_idx) = self.eir.by_key.get(&CfgFnKey::TopDef(sys_def)) else {
                    return Err(format!("ICE [codegen_x64]: missing EIR for scheduled system `{sys_def:?}`"));
                };
                let sys_fn = &self.eir.functions[fn_idx];
                let bindings = self.resolve_sys_param_bindings(sys_fn, run_args)?;
                let mut params = Vec::with_capacity(bindings.len());
                for (pi, &(k, kind)) in bindings.iter().enumerate() {
                    let p_ty = sys_fn.value_tys[sys_fn.params[pi].0 as usize];
                    let ac = classify_arg(self.env, p_ty);
                    let pass = match ac {
                        ArgClass::Void => SchedArgPass::BundleScalar { off: 0, bytes: 0 },
                        ArgClass::Reg2(_) => SchedArgPass::BundlePair { off: val_offs[k] },
                        ArgClass::Indirect(sz) => {
                            if kind == 2 {
                                SchedArgPass::DerefIndirect { off: val_offs[k], bytes: sz }
                            } else {
                                SchedArgPass::BundlePtr { off: val_offs[k] }
                            }
                        }
                        ArgClass::Float32 => SchedArgPass::BundleScalar { off: val_offs[k], bytes: 4 },
                        ArgClass::Float64 => SchedArgPass::BundleScalar { off: val_offs[k], bytes: 8 },
                        ArgClass::Reg1(sz) => match kind {
                            2 => SchedArgPass::DerefScalar { off: val_offs[k], bytes: sz },
                            3 => SchedArgPass::BundleScalar { off: ptr_offs[k], bytes: 8 },
                            _ => SchedArgPass::BundleScalar { off: val_offs[k], bytes: sz },
                        },
                    };
                    params.push((ac, pass));
                }
                let lbl = self.buf.new_label();
                let sys_sym = fn_sym_name(CfgFnKey::TopDef(sys_def), self.eir, self.interner);
                self.deferred_sched_wrappers.push(DeferredSchedWrapper {
                    lbl,
                    sys_sym,
                    params,
                });
                let node_off = (si * 32) as i32;
                self.buf.lea_rip_label(Reg::Rax, lbl);
                self.buf.store_mem(Reg::Rsp, node_off, Reg::Rax, 8);
                self.buf.store_mem(Reg::Rsp, node_off + 8, Reg::Rdx, 8);
                self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                self.buf.store_mem(Reg::Rsp, node_off + 16, Reg::Rax, 8); // init_deps=0, num_succs=0
                self.buf.store_mem(Reg::Rsp, node_off + 24, Reg::Rax, 8); // succs_ptr=0
            }

            self.buf.mov_rr(Reg::Rdi, Reg::Rsp);
            self.buf.mov_ri64(Reg::Rsi, wave.systems.len() as u64);
            // rdx already holds bundle_ptr
            self.helper_call("eon_sched_run_dag");
            if total_alloc > 0 {
                self.buf.add_ri32(Reg::Rsp, total_alloc as i32);
            }
        }
        Ok(())
    }

    fn emit_schedule_run(
        &mut self,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
    ) -> Result<(), String> {
        let sched_v = args[0];
        let run_args = &args[1..];
        let run_slots = if arg_slots.len() > 1 { &arg_slots[1..] } else { &[] };
        if let Some(idx) = self.resolve_const_schedule_idx(sched_v) {
            return self.emit_schedule_dag(idx, run_args, run_slots);
        }
        if self.eir.schedules.is_empty() {
            return Ok(());
        }
        let sched_ty = self.f.value_tys[sched_v.0 as usize];
        if matches!(self.env.get(sched_ty), Ty::Borrow { .. } | Ty::RawPtr { .. }) {
            self.load_scalar_raw(Reg::Rax, sched_v, 8);
            self.buf.load_mem(Reg::Rax, Reg::Rax, 0, 8, false);
        } else {
            self.load_scalar_raw(Reg::Rax, sched_v, 8);
        }
        let l_done = self.buf.new_label();
        for idx in 0..self.eir.schedules.len() {
            let l_next = self.buf.new_label();
            self.buf.cmp_ri32(Reg::Rax, idx as i32);
            self.buf.jcc_label(Cond::Ne, l_next);
            self.emit_schedule_dag(idx, run_args, run_slots)?;
            self.buf.jmp_label(l_done);
            self.buf.bind_label(l_next);
        }
        self.buf.bind_label(l_done);
        Ok(())
    }

    fn slot_pool_ncols_from_arg(&self, v: EirValue) -> usize {
        let ty = self.f.value_tys[v.0 as usize];
        let deref = match self.env.get(ty) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty };
        match self.env.get(deref) { Ty::Builtin(_, ga) => pool_col_count(self.env, ga), _ => 0 }
    }

    fn emit_slot_pool_init(
        &mut self,
        elem_ty: TyId,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) {
        let pdef = match self.env.get(elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
        let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
        let ncols = cols.len();
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;

        self.emit_memzero(Reg::Rbp, doff, x64_layout(self.env, ret_ty).size);
        // cap = max(args[1], 4)
        self.load_scalar_val(Reg::Rax, args[1]);
        self.buf.mov_ri64(Reg::Rcx, 4);
        self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
        self.buf.cmovcc(Cond::B, Reg::Rax, Reg::Rcx);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + ((ncols + 3) * 8) as i32, Reg::Rax, 8); // cap
        self.buf.store_mem(Reg::Rbp, doff + ok_off + ((ncols + 4) * 8) as i32, Reg::Rax, 8); // free_top = cap

        let l_fail = self.buf.new_label();
        let l_done = self.buf.new_label();

        // Allocate gen [ncols] and free_stack [ncols + 1]
        for extra in [ncols, ncols + 1] {
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
            self.buf.load_mem(Reg::Rsi, Reg::Rbp, doff + ok_off + ((ncols + 3) * 8) as i32, 8, false);
            self.buf.shl_imm(Reg::Rsi, 2); // * 4
            self.buf.mov_ri64(Reg::Rdx, 4);
            self.helper_call("eon_alloc_raw");
            self.buf.test_rr(Reg::Rax, Reg::Rax);
            self.buf.jcc_label(Cond::E, l_fail);
            self.buf.store_mem(Reg::Rbp, doff + ok_off + (extra * 8) as i32, Reg::Rax, 8);
        }
        // Allocate each column [0..ncols]
        for (i, c) in cols.iter().enumerate() {
            let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 0);
            self.buf.load_mem(Reg::Rsi, Reg::Rbp, doff + ok_off + ((ncols + 3) * 8) as i32, 8, false);
            if esz != 1 { self.buf.imul_rri32(Reg::Rsi, Reg::Rsi, esz as i32); }
            self.buf.mov_ri64(Reg::Rdx, 64);
            self.helper_call("eon_alloc_raw");
            self.buf.test_rr(Reg::Rax, Reg::Rax);
            self.buf.jcc_label(Cond::E, l_fail);
            self.buf.store_mem(Reg::Rbp, doff + ok_off + (i * 8) as i32, Reg::Rax, 8);
        }

        // Fill free_stack[i] = cap - 1 - i (gen is already 0 from mmap!)
        self.buf.load_mem(Reg::R8, Reg::Rbp, doff + ok_off + ((ncols + 1) * 8) as i32, 8, false); // free_stack
        self.buf.load_mem(Reg::R9, Reg::Rbp, doff + ok_off + ((ncols + 3) * 8) as i32, 8, false); // cap
        self.buf.xor_rr32(Reg::R10, Reg::R10); // i = 0
        let l_lp = self.buf.new_label();
        let l_lp_end = self.buf.new_label();
        self.buf.bind_label(l_lp);
        self.buf.cmp_rr(Reg::R10, Reg::R9);
        self.buf.jcc_label(Cond::Ae, l_lp_end);
        self.buf.mov_rr(Reg::Rax, Reg::R9);
        self.buf.sub_ri32(Reg::Rax, 1);
        self.buf.sub_rr(Reg::Rax, Reg::R10);
        self.buf.store_mem(Reg::R8, 0, Reg::Rax, 4);
        self.buf.add_ri32(Reg::R8, 4);
        self.buf.add_ri32(Reg::R10, 1);
        self.buf.jmp_label(l_lp);
        self.buf.bind_label(l_lp_end);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_fail);
        self.buf.mov_ri64(Reg::Rax, 7);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }

    fn emit_slot_pool_insert(
        &mut self,
        elem_ty: TyId,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
        allow_grow: bool,
    ) {
        let pdef = match self.env.get(elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
        let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
        let ncols = cols.len();
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let row_arg_idx = if allow_grow { 2 } else { 1 };

        let l_oom = self.buf.new_label();
        let l_has_slot = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, Reg::R8, ((ncols + 4) * 8) as i32, 8, false); // free_top
        self.buf.test_rr(Reg::R9, Reg::R9);
        self.buf.jcc_label(Cond::Ne, l_has_slot);

        if allow_grow {
            // Grow cap -> nc = oc == 0 ? 4 : oc * 2
            // Save oc at temp_copy_off, nc at temp_copy_off+8
            let tmp = self.plan.temp_copy_off;
            self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 3) * 8) as i32, 8, false); // oc
            self.buf.store_mem(Reg::Rbp, tmp, Reg::Rax, 8);
            self.buf.add_rr(Reg::Rax, Reg::Rax);
            self.buf.mov_ri64(Reg::Rcx, 4);
            self.buf.test_rr(Reg::Rax, Reg::Rax);
            self.buf.cmovcc(Cond::E, Reg::Rax, Reg::Rcx);
            self.buf.store_mem(Reg::Rbp, tmp + 8, Reg::Rax, 8); // nc

            // Grow gen
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
            self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 8, 8, false);
            self.buf.shl_imm(Reg::Rsi, 2);
            self.buf.mov_ri64(Reg::Rdx, 4);
            self.helper_call("eon_alloc_raw");
            self.buf.test_rr(Reg::Rax, Reg::Rax);
            self.buf.jcc_label(Cond::E, l_oom);
            self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8); // ng

            // Copy old gen -> ng, free old gen
            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.buf.mov_rr(Reg::Rdi, Reg::Rax);
            self.buf.load_mem(Reg::Rsi, Reg::R8, (ncols * 8) as i32, 8, false);
            self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp, 8, false);
            self.buf.shl_imm(Reg::Rcx, 2);
            self.buf.rep_movsb();
            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
            self.buf.load_mem(Reg::Rsi, Reg::R8, (ncols * 8) as i32, 8, false);
            self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp, 8, false);
            self.buf.shl_imm(Reg::Rdx, 2);
            self.helper_call("eon_free_raw");
            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 8, false);
            self.buf.store_mem(Reg::R8, (ncols * 8) as i32, Reg::Rax, 8);

            // Grow free_stack
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
            self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 8, 8, false);
            self.buf.shl_imm(Reg::Rsi, 2);
            self.buf.mov_ri64(Reg::Rdx, 4);
            self.helper_call("eon_alloc_raw");
            self.buf.test_rr(Reg::Rax, Reg::Rax);
            self.buf.jcc_label(Cond::E, l_oom);
            self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8); // nf
            // Fill nf[nc - 1 - k] = k for k = oc..nc
            self.buf.load_mem(Reg::R9, Reg::Rbp, tmp, 8, false); // k = oc
            self.buf.load_mem(Reg::R10, Reg::Rbp, tmp + 8, 8, false); // nc
            let l_fk = self.buf.new_label();
            let l_fk_end = self.buf.new_label();
            self.buf.bind_label(l_fk);
            self.buf.cmp_rr(Reg::R9, Reg::R10);
            self.buf.jcc_label(Cond::Ae, l_fk_end);
            self.buf.mov_rr(Reg::R11, Reg::R10);
            self.buf.sub_ri32(Reg::R11, 1);
            self.buf.sub_rr(Reg::R11, Reg::R9); // nc - 1 - k
            self.buf.shl_imm(Reg::R11, 2);
            self.buf.add_rr(Reg::R11, Reg::Rax);
            self.buf.store_mem(Reg::R11, 0, Reg::R9, 4);
            self.buf.add_ri32(Reg::R9, 1);
            self.buf.jmp_label(l_fk);
            self.buf.bind_label(l_fk_end);

            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
            self.buf.load_mem(Reg::Rsi, Reg::R8, ((ncols + 1) * 8) as i32, 8, false);
            self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp, 8, false);
            self.buf.shl_imm(Reg::Rdx, 2);
            self.helper_call("eon_free_raw");
            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 8, false);
            self.buf.store_mem(Reg::R8, ((ncols + 1) * 8) as i32, Reg::Rax, 8);

            // Grow each column
            for (i, c) in cols.iter().enumerate() {
                let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                self.buf.load_mem(Reg::Rsi, Reg::Rbp, tmp + 8, 8, false);
                if esz != 1 { self.buf.imul_rri32(Reg::Rsi, Reg::Rsi, esz as i32); }
                self.buf.mov_ri64(Reg::Rdx, 64);
                self.helper_call("eon_alloc_raw");
                self.buf.test_rr(Reg::Rax, Reg::Rax);
                self.buf.jcc_label(Cond::E, l_oom);
                self.buf.store_mem(Reg::Rbp, tmp + 16, Reg::Rax, 8);

                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.mov_rr(Reg::Rdi, Reg::Rax);
                self.buf.load_mem(Reg::Rsi, Reg::R8, (i * 8) as i32, 8, false);
                self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp, 8, false);
                if esz != 1 { self.buf.imul_rri32(Reg::Rcx, Reg::Rcx, esz as i32); }
                self.buf.rep_movsb();

                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.load_ptr_arg(Reg::Rdi, args, arg_slots, 1);
                self.buf.load_mem(Reg::Rsi, Reg::R8, (i * 8) as i32, 8, false);
                self.buf.load_mem(Reg::Rdx, Reg::Rbp, tmp, 8, false);
                if esz != 1 { self.buf.imul_rri32(Reg::Rdx, Reg::Rdx, esz as i32); }
                self.helper_call("eon_free_raw");

                self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
                self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 16, 8, false);
                self.buf.store_mem(Reg::R8, (i * 8) as i32, Reg::Rax, 8);
            }

            // cap = nc; free_top = nc - oc
            self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
            self.buf.load_mem(Reg::Rax, Reg::Rbp, tmp + 8, 8, false); // nc
            self.buf.store_mem(Reg::R8, ((ncols + 3) * 8) as i32, Reg::Rax, 8);
            self.buf.load_mem(Reg::Rcx, Reg::Rbp, tmp, 8, false); // oc
            self.buf.sub_rr(Reg::Rax, Reg::Rcx);
            self.buf.store_mem(Reg::R8, ((ncols + 4) * 8) as i32, Reg::Rax, 8);
            self.buf.mov_rr(Reg::R9, Reg::Rax);
        } else {
            self.buf.jmp_label(l_oom);
        }

        self.buf.bind_label(l_has_slot);
        // Pop slot: free_top -= 1; idx = free_stack[free_top]; g = gen[idx]; g = g==0 ? 1 : g+1; gen[idx] = g; len += 1
        self.buf.sub_ri32(Reg::R9, 1);
        self.buf.store_mem(Reg::R8, ((ncols + 4) * 8) as i32, Reg::R9, 8);
        self.buf.load_mem(Reg::R10, Reg::R8, ((ncols + 1) * 8) as i32, 8, false); // free_stack
        self.buf.shl_imm(Reg::R9, 2);
        self.buf.add_rr(Reg::R10, Reg::R9);
        self.buf.load_mem(Reg::R9, Reg::R10, 0, 4, false); // r9 = idx (u32)

        self.buf.load_mem(Reg::R10, Reg::R8, (ncols * 8) as i32, 8, false); // gen*
        self.buf.mov_rr(Reg::R11, Reg::R9);
        self.buf.shl_imm(Reg::R11, 2);
        self.buf.add_rr(Reg::R10, Reg::R11);
        self.buf.load_mem(Reg::Rax, Reg::R10, 0, 4, false); // g
        self.buf.add_ri32(Reg::Rax, 1); // 0 -> 1, 2 -> 3, ... (always odd!)
        self.buf.store_mem(Reg::R10, 0, Reg::Rax, 4);
        // Save GenId { idx (r9), g (rax) } into result ok!
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R9, 4);
        self.buf.store_mem(Reg::Rbp, doff + ok_off + 4, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4); // err = 0

        // len += 1
        self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 2) * 8) as i32, 8, false);
        self.buf.add_ri32(Reg::Rax, 1);
        self.buf.store_mem(Reg::R8, ((ncols + 2) * 8) as i32, Reg::Rax, 8);

        // Copy each row field into col_i[idx]
        let (rbase, row_off) = self.val_mem(args[row_arg_idx]);
        let row_ty = self.f.value_tys[args[row_arg_idx].0 as usize];
        for (i, c) in cols.iter().enumerate() {
            let (foff, _) = struct_field_offset(self.env, row_ty, i);
            let col_esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
            self.buf.load_mem(Reg::R10, Reg::R8, (i * 8) as i32, 8, false);
            self.buf.mov_rr(Reg::R11, Reg::R9);
            if col_esz != 1 { self.buf.imul_rri32(Reg::R11, Reg::R11, col_esz as i32); }
            self.buf.add_rr(Reg::R10, Reg::R11);
            self.copy_mem_using_rax(Reg::R10, 0, rbase, row_off + foff as i32, col_esz);
        }
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_oom);
        self.buf.mov_ri64(Reg::Rax, 7);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.xor_rr32(Reg::Rax, Reg::Rax);
        self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
        self.buf.bind_label(l_done);
    }

    fn emit_slot_pool_remove(
        &mut self,
        elem_ty: TyId,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
    ) {
        let pdef = match self.env.get(elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
        let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
        let ncols = cols.len();
        let doff = self.val_stack_off(dst);
        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
        let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
        let (gbase, gid_off) = self.val_mem(args[1]);

        let l_stale = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, gbase, gid_off, 4, false); // idx
        self.buf.load_mem(Reg::R10, gbase, gid_off + 4, 4, false); // gen
        // Validate: gen != 0 && (gen & 1) == 1 && idx < cap && pool->gen[idx] == gen
        self.buf.test_rr32(Reg::R10, Reg::R10);
        self.buf.jcc_label(Cond::E, l_stale);
        self.buf.mov_rr(Reg::Rax, Reg::R10);
        self.buf.and_ri32(Reg::Rax, 1);
        self.buf.test_rr32(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_stale);
        self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 3) * 8) as i32, 8, false); // cap
        self.buf.cmp_rr(Reg::R9, Reg::Rax);
        self.buf.jcc_label(Cond::Ae, l_stale);
        self.buf.load_mem(Reg::R11, Reg::R8, (ncols * 8) as i32, 8, false); // gen*
        self.buf.mov_rr(Reg::Rax, Reg::R9);
        self.buf.shl_imm(Reg::Rax, 2);
        self.buf.add_rr(Reg::R11, Reg::Rax);
        self.buf.load_mem(Reg::Rax, Reg::R11, 0, 4, false);
        self.buf.cmp_rr(Reg::Rax, Reg::R10);
        self.buf.jcc_label(Cond::Ne, l_stale);

        // Extract row fields into `dst.ok`
        self.emit_memzero(Reg::Rbp, doff, x64_layout(self.env, ret_ty).size);
        for (i, c) in cols.iter().enumerate() {
            let (foff, _) = struct_field_offset(self.env, ok_ty, i);
            let col_esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
            self.buf.load_mem(Reg::R10, Reg::R8, (i * 8) as i32, 8, false);
            self.buf.mov_rr(Reg::Rax, Reg::R9);
            if col_esz != 1 { self.buf.imul_rri32(Reg::Rax, Reg::Rax, col_esz as i32); }
            self.buf.add_rr(Reg::R10, Reg::Rax);
            self.copy_mem_using_rax(Reg::Rbp, doff + ok_off + foff as i32, Reg::R10, 0, col_esz);
        }

        // Bump gen[idx] += 1 (now even -> dead!), push idx onto free_stack, len -= 1
        self.buf.load_mem(Reg::Rax, Reg::R11, 0, 4, false);
        self.buf.add_ri32(Reg::Rax, 1);
        self.buf.store_mem(Reg::R11, 0, Reg::Rax, 4);
        self.buf.load_mem(Reg::R10, Reg::R8, ((ncols + 4) * 8) as i32, 8, false); // free_top
        self.buf.load_mem(Reg::R11, Reg::R8, ((ncols + 1) * 8) as i32, 8, false); // free_stack
        self.buf.mov_rr(Reg::Rax, Reg::R10);
        self.buf.shl_imm(Reg::Rax, 2);
        self.buf.add_rr(Reg::R11, Reg::Rax);
        self.buf.store_mem(Reg::R11, 0, Reg::R9, 4);
        self.buf.add_ri32(Reg::R10, 1);
        self.buf.store_mem(Reg::R8, ((ncols + 4) * 8) as i32, Reg::R10, 8);
        self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 2) * 8) as i32, 8, false);
        self.buf.sub_ri32(Reg::Rax, 1);
        self.buf.store_mem(Reg::R8, ((ncols + 2) * 8) as i32, Reg::Rax, 8);
        self.buf.jmp_label(l_done);

        self.buf.bind_label(l_stale);
        self.buf.mov_ri64(Reg::Rax, 11);
        self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
        self.buf.bind_label(l_done);
    }

    fn emit_slot_pool_get(
        &mut self,
        elem_ty: TyId,
        dst: EirValue,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
        returns_error_union: bool,
    ) {
        let pdef = match self.env.get(elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
        let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
        let doff = self.val_stack_off(dst);
        let (gbase, gid_off) = self.val_mem(args[1]);
        let l_stale = self.buf.new_label();
        let l_done = self.buf.new_label();

        self.load_ptr_arg(Reg::R8, args, arg_slots, 0);
        self.buf.load_mem(Reg::R9, gbase, gid_off, 4, false); // idx
        self.buf.load_mem(Reg::R10, gbase, gid_off + 4, 4, false); // gen
        self.buf.test_rr32(Reg::R10, Reg::R10);
        self.buf.jcc_label(Cond::E, l_stale);
        self.buf.mov_rr(Reg::Rax, Reg::R10);
        self.buf.and_ri32(Reg::Rax, 1);
        self.buf.test_rr32(Reg::Rax, Reg::Rax);
        self.buf.jcc_label(Cond::E, l_stale);
        self.buf.load_mem(Reg::Rax, Reg::R8, ((ncols + 3) * 8) as i32, 8, false); // cap
        self.buf.cmp_rr(Reg::R9, Reg::Rax);
        self.buf.jcc_label(Cond::Ae, l_stale);
        self.buf.load_mem(Reg::R11, Reg::R8, (ncols * 8) as i32, 8, false); // gen*
        self.buf.mov_rr(Reg::Rax, Reg::R9);
        self.buf.shl_imm(Reg::Rax, 2);
        self.buf.add_rr(Reg::R11, Reg::Rax);
        self.buf.load_mem(Reg::Rax, Reg::R11, 0, 4, false);
        self.buf.cmp_rr(Reg::Rax, Reg::R10);
        self.buf.jcc_label(Cond::Ne, l_stale);

        if returns_error_union {
            let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
            let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
            self.buf.xor_rr32(Reg::Rax, Reg::Rax);
            self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
            self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::R8, 8);
            self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::R9, 4);
            self.buf.jmp_label(l_done);

            self.buf.bind_label(l_stale);
            self.buf.mov_ri64(Reg::Rax, 11);
            self.buf.store_mem(Reg::Rbp, doff, Reg::Rax, 4);
            self.buf.xor_rr32(Reg::Rax, Reg::Rax);
            self.buf.store_mem(Reg::Rbp, doff + ok_off, Reg::Rax, 8);
            self.buf.store_mem(Reg::Rbp, doff + ok_off + 8, Reg::Rax, 4);
        } else {
            // `?*PoolRow` uses `OptNiche::FatPtrNullFirst` (`{ sp_ptr (0), idx (8) }`, null when `sp_ptr == 0`).
            self.buf.store_mem(Reg::Rbp, doff, Reg::R8, 8);
            self.buf.store_mem(Reg::Rbp, doff + 8, Reg::R9, 4);
            self.buf.jmp_label(l_done);

            self.buf.bind_label(l_stale);
            self.emit_opt_null(dst, ret_ty);
        }
        self.buf.bind_label(l_done);
    }

    fn emit_opt_null(&mut self, dst: EirValue, ty: TyId) {
        let lay = x64_layout(self.env, ty);
        match self.env.get(ty) {
            Ty::Optional(inner) => match opt_niche(self.env, *inner) {
                OptNiche::IdBrand => {
                    self.buf.mov_ri64(Reg::Rax, u32::MAX as u64);
                    self.store_scalar_val(dst, Reg::Rax, 4);
                }
                OptNiche::BoolTag => {
                    self.buf.mov_ri64(Reg::Rax, 2);
                    self.store_scalar_val(dst, Reg::Rax, 1);
                }
                OptNiche::ScalarSentinel { size, sentinel } => {
                    self.buf.mov_ri64(Reg::Rax, sentinel);
                    self.store_scalar_val(dst, Reg::Rax, size);
                }
                OptNiche::SliceMaxLen => {
                    let off = self.val_stack_off(dst);
                    self.buf.xor_rr32(Reg::Rax, Reg::Rax);
                    self.buf.store_mem(Reg::Rbp, off, Reg::Rax, 8);
                    self.buf.mov_ri64(Reg::Rax, u64::MAX);
                    self.buf.store_mem(Reg::Rbp, off + 8, Reg::Rax, 8);
                }
                OptNiche::GenId | OptNiche::Link | OptNiche::NullPtr | OptNiche::FatPtrNullFirst | OptNiche::None => {
                    let zsz = if matches!(opt_niche(self.env, *inner), OptNiche::Link) {
                        lay.size.max(8)
                    } else {
                        lay.size
                    };
                    match self.plan.val_locs[dst.0 as usize] {
                        ValLoc::Reg(r) => self.buf.xor_rr32(r, r),
                        ValLoc::Stack(off) => self.emit_memzero(Reg::Rbp, off, zsz),
                    }
                }
            },
            Ty::Builtin(BuiltinKind::IdBrand, _) => {
                self.buf.mov_ri64(Reg::Rax, u32::MAX as u64);
                self.store_scalar_val(dst, Reg::Rax, 4);
            }
            _ => match self.plan.val_locs[dst.0 as usize] {
                ValLoc::Reg(r) => self.buf.xor_rr32(r, r),
                ValLoc::Stack(off) => self.emit_memzero(Reg::Rbp, off, lay.size),
            },
        }
    }

    // ========================================================================
    // Terminator & Parallel Phi Edge Lowering
    // ========================================================================

    fn emit_phi_edge(&mut self, from_bb: BlockId, to_bb: BlockId) {
        let target = self.f.block(to_bb);
        if target.phis.is_empty() {
            return;
        }
        // Two-phase parallel copy via `temp_copy_off` so swapped phis never clobber each other!
        let mut staged: Vec<(EirValue, i32, usize)> = Vec::new();
        let mut cur_off = self.plan.temp_copy_off;
        for phi in &target.phis {
            let sz = x64_layout(self.env, phi.ty).size;
            if sz == 0 {
                continue;
            }
            if let Some(&(_, inc_v)) = phi.incoming.iter().find(|&&(p, _)| p == from_bb) {
                self.store_val_to_mem(Reg::Rbp, cur_off, inc_v, phi.ty);
                staged.push((phi.dst, cur_off, sz));
                cur_off += align_up(sz, 8) as i32;
            }
        }
        for (dst_v, toff, _sz) in staged {
            let dty = self.f.value_tys[dst_v.0 as usize];
            self.load_val_from_mem(dst_v, Reg::Rbp, toff, dty);
        }
    }

    fn resolve_branch_label(&self, target: BlockId) -> Label {
        if let Some(w) = &self.active_worker {
            if target == w.step_bb || target == w.header_bb {
                return w.worker_step_lbl;
            }
            if target == w.exit_bb {
                return w.worker_done_lbl;
            }
        }
        self.bb_labels[&target]
    }

    fn emit_term(&mut self, b: &EirBlock) {
        let in_worker = self.active_worker.is_some();
        match &b.term {
            EirTerm::Goto(t) => {
                if !in_worker {
                    self.emit_phi_edge(b.id, *t);
                }
                let l = self.resolve_branch_label(*t);
                self.buf.jmp_label(l);
            }
            EirTerm::Branch { cond, then_bb, else_bb, .. } => {
                self.load_scalar_raw(Reg::Rax, *cond, 1);
                self.buf.test_rr32(Reg::Rax, Reg::Rax);
                let then_phis = !in_worker && !self.f.block(*then_bb).phis.is_empty();
                let else_phis = !in_worker && !self.f.block(*else_bb).phis.is_empty();
                let lt = self.resolve_branch_label(*then_bb);
                let le = self.resolve_branch_label(*else_bb);
                if !then_phis && !else_phis {
                    self.buf.jcc_label(Cond::Ne, lt);
                    self.buf.jmp_label(le);
                } else {
                    let l_else_edge = self.buf.new_label();
                    self.buf.jcc_label(Cond::E, l_else_edge);
                    self.emit_phi_edge(b.id, *then_bb);
                    self.buf.jmp_label(lt);
                    self.buf.bind_label(l_else_edge);
                    self.emit_phi_edge(b.id, *else_bb);
                    self.buf.jmp_label(le);
                }
            }
            EirTerm::SwitchInt { discr, targets, otherwise } => {
                let dty = self.f.value_tys[discr.0 as usize];
                let dsz = x64_layout(self.env, dty).size.clamp(1, 8);
                self.load_scalar_raw(Reg::Rax, *discr, dsz);
                for &(val, tgt) in targets {
                    let l_next = self.buf.new_label();
                    self.buf.mov_ri64(Reg::Rcx, (val as i64) as u64);
                    self.buf.cmp_rr(Reg::Rax, Reg::Rcx);
                    self.buf.jcc_label(Cond::Ne, l_next);
                    if !in_worker {
                        self.emit_phi_edge(b.id, tgt);
                    }
                    let lt = self.resolve_branch_label(tgt);
                    self.buf.jmp_label(lt);
                    self.buf.bind_label(l_next);
                }
                if !in_worker {
                    self.emit_phi_edge(b.id, *otherwise);
                }
                let lo = self.resolve_branch_label(*otherwise);
                self.buf.jmp_label(lo);
            }
            EirTerm::Return(val_opt) => {
                if self.f.sig.effects.is_naked {
                    self.buf.ret();
                    return;
                }
                let ret_cls = classify_ret(self.env, self.f.sig.ret, self.f.sig.effects.is_export_c);
                let mut err_return_flag: Option<bool> = None; // Some(true)=clc/stc already set
                if let Some(rv) = val_opt {
                    match ret_cls {
                        RetClass::Void => {}
                        RetClass::Float32 => self.load_float_val(Xmm::Xmm0, *rv, false),
                        RetClass::Float64 => self.load_float_val(Xmm::Xmm0, *rv, true),
                        RetClass::Reg1(sz) => self.load_scalar_raw(Reg::Rax, *rv, sz),
                        RetClass::Reg2(sz) => {
                            let roff = self.val_stack_off(*rv);
                            self.buf.load_mem(Reg::Rax, Reg::Rbp, roff, 8, false);
                            if sz > 8 {
                                self.buf.load_mem(Reg::Rdx, Reg::Rbp, roff + 8, 8, false);
                            }
                        }
                        RetClass::Sret(sz) => {
                            let sret_off = self.plan.sret_off.unwrap();
                            self.buf.load_mem(Reg::R8, Reg::Rbp, sret_off, 8, false);
                            let roff = self.val_stack_off(*rv);
                            self.buf.lea_mem(Reg::R9, Reg::Rbp, roff);
                            self.emit_memcpy_ptrs(Reg::R8, Reg::R9, sz);
                            self.buf.load_mem(Reg::Rax, Reg::Rbp, sret_off, 8, false);
                        }
                        RetClass::ErrorUnion { ok_ty, ok_size, ok_is_f32, ok_is_f64 } => {
                            let roff = self.val_stack_off(*rv);
                            let ok_off = eu_ok_offset(self.env, ok_ty) as i32;
                            self.buf.load_mem(Reg::R15, Reg::Rbp, roff, 4, false);
                            let l_err = self.buf.new_label();
                            let l_ret = self.buf.new_label();
                            self.buf.test_rr32(Reg::R15, Reg::R15);
                            self.buf.jcc_label(Cond::Ne, l_err);
                            // Ok path: load ok value, set CF = 0 (clc) right before ret!
                            if ok_is_f32 {
                                self.buf.movss_load(Xmm::Xmm0, Reg::Rbp, roff + ok_off);
                            } else if ok_is_f64 {
                                self.buf.movsd_load(Xmm::Xmm0, Reg::Rbp, roff + ok_off);
                            } else if ok_size > 0 && ok_size <= 8 {
                                self.buf.load_mem(Reg::Rax, Reg::Rbp, roff + ok_off, ok_size, false);
                            } else if ok_size > 8 && ok_size <= 16 {
                                self.buf.load_mem(Reg::Rax, Reg::Rbp, roff + ok_off, 8, false);
                                self.buf.load_mem(Reg::Rdx, Reg::Rbp, roff + ok_off + 8, (ok_size - 8).min(8), false);
                            } else if ok_size > 16 {
                                let sret_off = self.plan.sret_off.unwrap();
                                self.buf.load_mem(Reg::R8, Reg::Rbp, sret_off, 8, false);
                                self.buf.lea_mem(Reg::R9, Reg::Rbp, roff + ok_off);
                                self.emit_memcpy_ptrs(Reg::R8, Reg::R9, ok_size);
                            }
                            self.emit_epilogue_preserving_flags(false);
                            self.buf.clc();
                            self.buf.ret();

                            self.buf.bind_label(l_err);
                            self.emit_epilogue_preserving_flags(false);
                            self.buf.stc();
                            self.buf.ret();
                            self.buf.bind_label(l_ret);
                            err_return_flag = Some(true);
                        }
                    }
                } else if matches!(ret_cls, RetClass::ErrorUnion { .. }) {
                    self.buf.xor_rr32(Reg::R15, Reg::R15);
                    self.emit_epilogue_preserving_flags(false);
                    self.buf.clc();
                    self.buf.ret();
                    err_return_flag = Some(true);
                }
                if err_return_flag.is_none() {
                    self.emit_epilogue_preserving_flags(self.f.sig.effects.is_interrupt);
                    if self.f.sig.effects.is_interrupt {
                        self.buf.iretq();
                    } else {
                        self.buf.ret();
                    }
                }
            }
            EirTerm::Unreachable { .. } => {
                self.buf.ud2();
            }
        }
    }

    fn emit_epilogue_preserving_flags(&mut self, is_interrupt: bool) {
        let saved_bytes = (self.plan.used_callee_regs.len() * 8) as i32;
        self.buf.lea_mem(Reg::Rsp, Reg::Rbp, -saved_bytes);
        for &r in self.plan.used_callee_regs.iter().rev() {
            self.buf.pop(r);
        }
        self.buf.pop(Reg::Rbp);
        if is_interrupt {
            for r in [
                Reg::R15, Reg::R11, Reg::R10, Reg::R9, Reg::R8,
                Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::Rax,
            ] {
                self.buf.pop(r);
            }
        }
    }
}

fn int_type_bounds(bits: u8, signed: bool) -> (u64, u64) {
    if bits >= 64 {
        if signed {
            (i64::MIN as u64, i64::MAX as u64)
        } else {
            (0, u64::MAX)
        }
    } else if bits == 0 {
        (0, 0)
    } else if signed {
        let half = 1u64 << (bits - 1);
        ((-(half as i64)) as u64, half - 1)
    } else {
        (0, (1u64 << bits) - 1)
    }
}

// ============================================================================
// Whole-Program Assembly & Built-in ELF64 Linking
// ============================================================================

pub fn compile_and_link_x64(
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    sm: &SourceMap,
    interner: &Interner,
    opts: &X64Options,
) -> Result<LinkedElf, String> {
    let (source_files, file_index_map) = collect_source_files(sm);
    let is_pie_or_shared = matches!(opts.mode, ElfMode::LinuxPie | ElfMode::LinuxShared { .. });
    let mut blobs = Vec::with_capacity(eir.functions.len());
    for f in &eir.functions {
        if f.sig.effects.is_gpu {
            continue;
        }
        blobs.push(compile_function_x64(
            f,
            eir,
            prog,
            env,
            sm,
            interner,
            &file_index_map,
            opts.sanitize,
            is_pie_or_shared,
        )?);
    }
    link_x64_blobs(&blobs, eir, prog, env, interner, source_files, opts)
}

pub fn collect_source_files(sm: &SourceMap) -> (Vec<String>, BTreeMap<FileId, u32>) {
    let mut names = Vec::new();
    let mut map = BTreeMap::new();
    for i in 0..sm.len() {
        let fid = FileId(i as u32);
        map.insert(fid, i as u32);
        names.push(sm.file(fid).name.clone());
    }
    (names, map)
}

pub fn link_x64_blobs(
    blobs: &[CompiledFnBlob],
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
    source_files: Vec<String>,
    opts: &X64Options,
) -> Result<LinkedElf, String> {
    let mut text = Vec::new();
    let mut rodata = Vec::new();
    let mut data_rel_ro = Vec::new();
    let mut symbols = Vec::new();
    let mut relocs = Vec::new();
    let mut line_entries = Vec::new();
    let mut all_helpers: BTreeSet<String> = BTreeSet::new();
    for blob in blobs {
        for h in &blob.used_helpers {
            all_helpers.insert(h.clone());
        }
    }

    let is_freestanding = matches!(opts.mode, ElfMode::Freestanding { .. });
    let is_pie_or_shared = !is_freestanding;
    let has_sched = all_helpers.contains("eon_sched_init")
        || all_helpers.contains("eon_par_for")
        || all_helpers.contains("eon_sched_run_dag");

    // Find entry function (`main` or `kmain` or `_start` or first pub fn in freestanding).
    let entry_fn = find_entry_function(eir, prog, interner, &opts.mode);
    let entry_symbol = match &opts.mode {
        ElfMode::LinuxPie => "_start".to_owned(),
        ElfMode::LinuxShared { .. } => String::new(),
        ElfMode::Freestanding { .. } => entry_fn
            .map(|f| fn_sym_name(f.key, eir, interner))
            .unwrap_or_else(|| "_start".to_owned()),
    };

    // 1. Emit Linux `_start` wrapper (only for LinuxPie!).
    if matches!(opts.mode, ElfMode::LinuxPie)
        && let Some(mf) = entry_fn
    {
        let start_blob = emit_linux_start(mf, eir, env, interner, opts, has_sched)?;
        for h in start_blob.helpers {
            all_helpers.insert(h);
        }
        let off = text.len();
        symbols.push(ElfSymbolDef {
            name: "_start".into(),
            section: RelocSection::Text,
            offset: off,
            size: start_blob.code.len() as u64,
            is_func: true,
        });
        for r in start_blob.relocs {
            relocs.push(ElfReloc {
                section: RelocSection::Text,
                offset: off + r.offset,
                kind: r.kind,
                target_sym: r.target_sym,
                addend: r.addend,
            });
        }
        text.extend_from_slice(&start_blob.code);
    }

    // 2. Append all compiled function blobs (16-byte aligned).
    for blob in blobs {
        pad_vec(&mut text, 16);
        let fn_off = text.len();
        symbols.push(ElfSymbolDef {
            name: blob.sym_name.clone(),
            section: RelocSection::Text,
            offset: fn_off,
            size: blob.code.len() as u64,
            is_func: true,
        });
        for r in &blob.relocs {
            relocs.push(ElfReloc {
                section: RelocSection::Text,
                offset: fn_off + r.offset,
                kind: r.kind.clone(),
                target_sym: r.target_sym.clone(),
                addend: r.addend,
            });
        }
        for le in &blob.line_entries {
            line_entries.push(LineEntry {
                code_offset: (fn_off as u32) + le.code_offset,
                file_idx: le.file_idx,
                line: le.line,
                col: le.col,
            });
        }
        text.extend_from_slice(&blob.code);

        for ro in &blob.rodata {
            pad_vec(&mut rodata, ro.align.max(1));
            let ro_off = rodata.len();
            symbols.push(ElfSymbolDef {
                name: ro.sym.clone(),
                section: RelocSection::Rodata,
                offset: ro_off,
                size: ro.bytes.len() as u64,
                is_func: false,
            });
            rodata.extend_from_slice(&ro.bytes);
        }
    }

    // 3. Emit `dyn` vtables into `.data.rel.ro` (preserving W^X `.rodata`!) & thunks in `.text`.
    emit_dyn_vtables_x64(
        eir,
        env,
        interner,
        &mut text,
        &mut data_rel_ro,
        &mut symbols,
        &mut relocs,
    )?;
    emit_fnref_thunks_x64(
        eir,
        env,
        interner,
        is_pie_or_shared,
        &mut text,
        &mut symbols,
        &mut relocs,
    )?;

    // 4. Register `.bss` symbols for LinuxPie / LinuxShared and resolve transitive helper deps.
    let mut bss_size = 0usize;
    if !is_freestanding {
        for &(name, size, align) in rt_bss_layout() {
            bss_size = align_up(bss_size, align.max(1));
            symbols.push(ElfSymbolDef {
                name: name.to_owned(),
                section: RelocSection::Bss,
                offset: bss_size,
                size: size as u64,
                is_func: false,
            });
            bss_size += size;
        }
        let mut queue: Vec<String> = all_helpers.iter().cloned().collect();
        while let Some(h) = queue.pop() {
            for &dep in rt_helper_deps(&h) {
                if all_helpers.insert(dep.to_owned()) {
                    queue.push(dep.to_owned());
                }
            }
        }
    }

    // 5. Emit ONLY referenced standalone runtime helpers (Tree-Shaking!).
    for h in &all_helpers {
        if is_freestanding
            && !matches!(
                h.as_str(),
                "eon_f16_to_f32"
                    | "eon_f32_to_f16"
                    | "eon_arena_alloc_aligned"
                    | "eon_rt_validate_utf8"
            )
        {
            continue;
        }
        let Some(res) = emit_rt_helper_by_name(h) else {
            continue;
        };
        let (hcode, hrelocs) = res?;
        pad_vec(&mut text, 16);
        let off = text.len();
        symbols.push(ElfSymbolDef {
            name: h.clone(),
            section: RelocSection::Text,
            offset: off,
            size: hcode.len() as u64,
            is_func: true,
        });
        for r in hrelocs {
            relocs.push(ElfReloc {
                section: RelocSection::Text,
                offset: off + r.offset,
                kind: r.kind,
                target_sym: r.target_sym,
                addend: r.addend,
            });
        }
        text.extend_from_slice(&hcode);
    }

    // 6. Export `pub fn` symbols for `LinuxShared` (both mangled and clean unmangled aliases).
    let mut exported_symbols: BTreeSet<String> = BTreeSet::new();
    if matches!(opts.mode, ElfMode::LinuxShared { .. }) {
        let mut text_defs: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for s in &symbols {
            if s.section == RelocSection::Text {
                text_defs.insert(s.name.clone(), (s.offset, s.size));
            }
        }
        for f in &eir.functions {
            if f.sig.effects.is_gpu || !f.sig.is_pub || !matches!(f.key, CfgFnKey::TopDef(_)) {
                continue;
            }
            let sym = fn_sym_name(f.key, eir, interner);
            exported_symbols.insert(sym.clone());
            let plain = sanitize_sym(interner.get(f.name));
            if !f.sig.effects.is_export_c
                && plain != sym
                && !symbols.iter().any(|s| s.name == plain)
                && let Some(&(off, sz)) = text_defs.get(&sym)
            {
                symbols.push(ElfSymbolDef {
                    name: plain.clone(),
                    section: RelocSection::Text,
                    offset: off,
                    size: sz,
                    is_func: true,
                });
                exported_symbols.insert(plain);
            }
        }
    }

    // 7. Collect unresolved external dynamic symbols referenced via PLT/GOT/Abs64.
    let external_symbols: Vec<String> = {
        let defined: BTreeSet<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        let mut ext: BTreeSet<String> = BTreeSet::new();
        for r in &relocs {
            if matches!(r.kind, RelocKind::Plt32 | RelocKind::GotPcRel32 | RelocKind::Abs64)
                && !defined.contains(r.target_sym.as_str())
                && !matches!(
                    r.target_sym.as_str(),
                    "__eon_ehdr_start" | "__eon_rela_dyn_start" | "__eon_rela_dyn_end"
                )
            {
                ext.insert(r.target_sym.clone());
            }
        }
        ext.into_iter().collect()
    };

    let input = ElfLinkInput {
        mode: opts.mode.clone(),
        entry_symbol,
        text,
        rodata,
        data_rel_ro,
        data: Vec::new(),
        bss_size,
        tdata: Vec::new(),
        tbss_size: 0,
        symbols,
        relocs,
        source_files,
        line_entries,
        emit_debug_line: true,
        needed_libs: opts.needed_libs.clone(),
        exported_symbols,
        external_symbols,
    };
    link_elf64(&input)
}

fn find_entry_function<'a>(
    eir: &'a EirProgram,
    prog: &ResolvedProgram,
    interner: &Interner,
    mode: &ElfMode,
) -> Option<&'a EirFunction> {
    if matches!(mode, ElfMode::LinuxShared { .. }) {
        return None;
    }
    for target_name in ["main", "_start", "kmain"] {
        for f in &eir.functions {
            if let CfgFnKey::TopDef(d) = f.key
                && matches!(prog.defs[d.0 as usize].kind, TopDefKind::Fn(_))
                && interner.get(f.name) == target_name
            {
                return Some(f);
            }
        }
    }
    if matches!(mode, ElfMode::Freestanding { .. }) {
        return eir.functions.iter().find(|f| !f.sig.effects.is_gpu);
    }
    None
}

struct LinuxStartBlob {
    code: Vec<u8>,
    relocs: Vec<FnReloc>,
    helpers: Vec<String>,
}

fn emit_linux_start(
    mf: &EirFunction,
    eir: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    opts: &X64Options,
    has_sched: bool,
) -> Result<LinuxStartBlob, String> {
    let mut b = X64Buf::new();
    let mut needed_helpers = Vec::new();

    // Kernel enters `_start` with `rsp` 16-byte aligned pointing at:
    // [argc, argv[0..], NULL, envp[0..], NULL, auxv[(a_type, a_val)..], (0, 0)]
    // Step A: Save initial stack in R12 and extract AT_BASE (7) -> R13, AT_SYSINFO_EHDR (33) -> R14.
    b.mov_rr(Reg::R12, Reg::Rsp);
    b.load_mem(Reg::Rdi, Reg::R12, 0, 8, false); // argc
    b.lea_mem(Reg::Rax, Reg::Rdi, 2); // argc + 1 (argv NULL) + 1 (skip argc slot = +16B/8)
    b.shl_imm(Reg::Rax, 3);
    b.add_rr(Reg::Rax, Reg::R12); // envp
    b.mov_rr(Reg::R15, Reg::Rax); // preserve envp for __eon_envp

    let l_env = b.new_label();
    b.bind_label(l_env);
    b.load_mem(Reg::Rcx, Reg::Rax, 0, 8, false);
    b.add_ri32(Reg::Rax, 8);
    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::Ne, l_env);

    b.xor_rr32(Reg::R13, Reg::R13); // AT_BASE
    b.xor_rr32(Reg::R14, Reg::R14); // AT_SYSINFO_EHDR
    let l_aux = b.new_label();
    let l_aux_done = b.new_label();
    let l_chk_vdso = b.new_label();
    let l_aux_next = b.new_label();
    b.bind_label(l_aux);
    b.load_mem(Reg::Rcx, Reg::Rax, 0, 8, false);
    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::E, l_aux_done);
    b.load_mem(Reg::Rdx, Reg::Rax, 8, 8, false);
    b.cmp_ri32(Reg::Rcx, 7);
    b.jcc_label(Cond::Ne, l_chk_vdso);
    b.mov_rr(Reg::R13, Reg::Rdx);
    b.jmp_label(l_aux_next);
    b.bind_label(l_chk_vdso);
    b.cmp_ri32(Reg::Rcx, 33);
    b.jcc_label(Cond::Ne, l_aux_next);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.bind_label(l_aux_next);
    b.add_ri32(Reg::Rax, 16);
    b.jmp_label(l_aux);
    b.bind_label(l_aux_done);

    // Step B: Self-relocate Static PIE if no PT_INTERP / needed_libs and kernel AT_BASE == 0.
    if opts.needed_libs.is_empty() {
        let l_skip_reloc = b.new_label();
        let l_rel_lp = b.new_label();
        let l_rel_next = b.new_label();
        b.test_rr(Reg::R13, Reg::R13);
        b.jcc_label(Cond::Ne, l_skip_reloc);
        b.lea_rip_sym(Reg::R8, "__eon_ehdr_start", 0);
        b.lea_rip_sym(Reg::R9, "__eon_rela_dyn_start", 0);
        b.lea_rip_sym(Reg::R10, "__eon_rela_dyn_end", 0);
        b.bind_label(l_rel_lp);
        b.cmp_rr(Reg::R9, Reg::R10);
        b.jcc_label(Cond::Ae, l_skip_reloc);
        b.load_mem(Reg::Rcx, Reg::R9, 8, 4, false); // ELF64_R_TYPE(r_info)
        b.cmp_ri32(Reg::Rcx, 8); // R_X86_64_RELATIVE
        b.jcc_label(Cond::Ne, l_rel_next);
        b.load_mem(Reg::Rax, Reg::R9, 0, 8, false); // r_offset
        b.load_mem(Reg::Rdx, Reg::R9, 16, 8, false); // r_addend
        b.add_rr(Reg::Rax, Reg::R8);
        b.add_rr(Reg::Rdx, Reg::R8);
        b.store_mem(Reg::Rax, 0, Reg::Rdx, 8);
        b.bind_label(l_rel_next);
        b.add_ri32(Reg::R9, 24);
        b.jmp_label(l_rel_lp);
        b.bind_label(l_skip_reloc);
    }

    // Step C: Runtime initialization (TLS TCB, vDSO clock_gettime, optional sanitizer handler).
    b.mov_rip_store("__eon_envp", 0, Reg::R15);
    b.call_sym("eon_rt_tls_init");
    needed_helpers.push("eon_rt_tls_init".to_owned());

    let l_no_vdso = b.new_label();
    b.test_rr(Reg::R14, Reg::R14);
    b.jcc_label(Cond::E, l_no_vdso);
    b.mov_rr(Reg::Rdi, Reg::R14);
    b.call_sym("eon_rt_parse_vdso");
    needed_helpers.push("eon_rt_parse_vdso".to_owned());
    b.bind_label(l_no_vdso);

    if opts.sanitize {
        b.call_sym("eon_rt_install_san_handler");
        needed_helpers.push("eon_rt_install_san_handler".to_owned());
    }

    // Step D: Prepare main argument (`Env`, `AllocCap`, or 0-arg) and invoke `main`.
    let p0_deref = mf.sig.params.first().map(|p| match env.get(p.ty) {
        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
        _ => p.ty,
    });
    let takes_env = mf.params.len() == 1
        && p0_deref.is_some_and(|t| matches!(env.get(t), Ty::Builtin(BuiltinKind::Env, _)));
    let takes_alloc = mf.params.len() == 1
        && p0_deref.is_some_and(|t| matches!(env.get(t), Ty::Builtin(BuiltinKind::AllocCap, _)));

    if takes_env {
        // Reserve 96B (16B aligned) for 88B Env struct.
        b.sub_ri32(Reg::Rsp, 96);
        b.xor_rr32(Reg::Rax, Reg::Rax);
        for off in (0..88).step_by(8) {
            b.store_mem(Reg::Rsp, off, Reg::Rax, 8);
        }
        b.mov_ri64(Reg::Rax, (-100i64) as u64);
        b.store_mem(Reg::Rsp, 40, Reg::Rax, 8); // FsCap = AT_FDCWD
        b.mov_ri64(Reg::Rax, 1);
        b.store_mem(Reg::Rsp, 56, Reg::Rax, 8); // ClockCap
        b.store_mem(Reg::Rsp, 64, Reg::Rax, 8); // IoCap
        b.load_mem(Reg::Rax, Reg::R12, 0, 8, false);
        b.store_mem(Reg::Rsp, 72, Reg::Rax, 8); // ArgsCap.argc
        b.lea_mem(Reg::Rax, Reg::R12, 8);
        b.store_mem(Reg::Rsp, 80, Reg::Rax, 8); // ArgsCap.argv
        b.mov_rr(Reg::Rdi, Reg::Rsp);
    } else if takes_alloc {
        // Reserve 48B (16B aligned) for zeroed 40B AllocCap.
        b.sub_ri32(Reg::Rsp, 48);
        b.xor_rr32(Reg::Rax, Reg::Rax);
        for off in (0..40).step_by(8) {
            b.store_mem(Reg::Rsp, off, Reg::Rax, 8);
        }
        b.mov_rr(Reg::Rdi, Reg::Rsp);
    }
    let main_sym = fn_sym_name(mf.key, eir, interner);
    b.call_sym(main_sym);

    // Compute exit code in callee-saved Rbx before any scheduler shutdown call!
    match env.get(mf.sig.ret) {
        Ty::Void => b.xor_rr32(Reg::Rbx, Reg::Rbx),
        Ty::Int { .. } => b.mov_rr(Reg::Rbx, Reg::Rax),
        Ty::ErrorUnion { ok, .. } => {
            let l_err = b.new_label();
            let l_exit = b.new_label();
            b.jcc_label(Cond::B, l_err); // CF=1 => error in r15!
            if *ok == TyId::VOID || *ok == TyId::NEVER {
                b.xor_rr32(Reg::Rbx, Reg::Rbx);
            } else {
                b.mov_rr(Reg::Rbx, Reg::Rax);
            }
            b.jmp_label(l_exit);
            b.bind_label(l_err);
            b.mov_rr(Reg::Rbx, Reg::R15);
            b.bind_label(l_exit);
        }
        _ => b.xor_rr32(Reg::Rbx, Reg::Rbx),
    }

    // Step E: Clean shutdown & SYS_exit_group(231).
    if has_sched {
        b.call_sym("eon_sched_shutdown");
        needed_helpers.push("eon_sched_shutdown".to_owned());
    }
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.mov_ri64(Reg::Rax, 231);
    b.syscall();
    b.ud2();
    let (code, relocs) = b.finish()?;
    Ok(LinuxStartBlob {
        code,
        relocs,
        helpers: needed_helpers,
    })
}

fn emit_dyn_vtables_x64(
    eir: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    text: &mut Vec<u8>,
    data_rel_ro: &mut Vec<u8>,
    symbols: &mut Vec<ElfSymbolDef>,
    relocs: &mut Vec<ElfReloc>,
) -> Result<(), String> {
    for &(iface_def, concrete_ty) in &eir.dyn_impl_pairs {
        let Some(imeta) = env.ifaces.get(&iface_def) else { continue; };
        let target = match env.get(concrete_ty) {
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def) => Some(crate::types::MethodTarget::Def(*def)),
            Ty::Builtin(bk, _) => Some(crate::types::MethodTarget::Builtin(*bk)),
            _ => None,
        };
        let Some(tk) = target else { continue; };
        let Some((impl_idx, im_ref)) = env
            .impls
            .iter()
            .enumerate()
            .find(|(_, im)| im.target == tk && im.iface == Some(iface_def))
        else {
            continue;
        };

        pad_vec(data_rel_ro, 8);
        let vt_off = data_rel_ro.len();
        let vt_sym = format!("eon_vtable_{}_{}", iface_def.0, concrete_ty.0);
        let vt_size = (imeta.methods.len().max(1) * 8) as u64;
        symbols.push(ElfSymbolDef {
            name: vt_sym,
            section: RelocSection::DataRelRo,
            offset: vt_off,
            size: vt_size,
            is_func: false,
        });
        if imeta.methods.is_empty() {
            data_rel_ro.extend_from_slice(&0u64.to_le_bytes());
            continue;
        }
        for (mi, imethod) in imeta.methods.iter().enumerate() {
            let Some(&(_, sub_idx, _)) = im_ref.methods.iter().find(|(_, _, m)| m.name == imethod.name) else {
                data_rel_ro.extend_from_slice(&0u64.to_le_bytes());
                continue;
            };
            let target_sym = fn_sym_name(CfgFnKey::ImplMethod { impl_idx, sub_idx }, eir, interner);
            // Emit 5-byte direct-jump thunk in `.text`
            let mut tb = X64Buf::new();
            tb.jmp_sym(target_sym);
            let (tcode, trelocs) = tb.finish()?;
            pad_vec(text, 8);
            let toff = text.len();
            let tsym = format!("eon_thunk_{}_{}_{}", iface_def.0, concrete_ty.0, mi);
            symbols.push(ElfSymbolDef {
                name: tsym.clone(),
                section: RelocSection::Text,
                offset: toff,
                size: tcode.len() as u64,
                is_func: true,
            });
            for r in trelocs {
                relocs.push(ElfReloc {
                    section: RelocSection::Text,
                    offset: toff + r.offset,
                    kind: r.kind,
                    target_sym: r.target_sym,
                    addend: r.addend,
                });
            }
            text.extend_from_slice(&tcode);

            let slot_off = data_rel_ro.len();
            data_rel_ro.extend_from_slice(&0u64.to_le_bytes());
            relocs.push(ElfReloc {
                section: RelocSection::DataRelRo,
                offset: slot_off,
                kind: RelocKind::Abs64,
                target_sym: tsym,
                addend: 0,
            });
        }
    }
    Ok(())
}

fn emit_fnref_thunks_x64(
    eir: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    is_pie_or_shared: bool,
    text: &mut Vec<u8>,
    symbols: &mut Vec<ElfSymbolDef>,
    relocs: &mut Vec<ElfReloc>,
) -> Result<(), String> {
    let mut fnrefs: BTreeSet<DefId> = BTreeSet::new();
    for f in &eir.functions {
        for b in &f.blocks {
            for inst in &b.insts {
                if let EirInstKind::Const { konst: CfgConst::FnRef(d, _), .. } = &inst.kind {
                    fnrefs.insert(*d);
                }
            }
        }
    }
    for d in fnrefs {
        let target_sym = fn_sym_name(CfgFnKey::TopDef(d), eir, interner);
        let sig_opt = eir
            .by_key
            .get(&CfgFnKey::TopDef(d))
            .map(|&fi| &eir.functions[fi].sig)
            .or_else(|| env.fns.get(&d));
        let is_export_c = sig_opt.is_some_and(|s| s.effects.is_export_c);
        let ret_cls = sig_opt.map_or(RetClass::Void, |s| classify_ret(env, s.ret, is_export_c));
        let start = usize::from(ret_uses_sret(ret_cls));
        // Count how many GP arg slots the target expects.
        let mut user_gp_slots = 0usize;
        if let Some(sig) = sig_opt {
            for p in &sig.params {
                match classify_arg(env, p.ty) {
                    ArgClass::Reg1(_) | ArgClass::Indirect(_) => user_gp_slots += 1,
                    ArgClass::Reg2(_) => user_gp_slots += 2,
                    ArgClass::Void | ArgClass::Float32 | ArgClass::Float64 => {}
                }
            }
        }
        let mut tb = X64Buf::new();
        // Shift GP arg registers left by 1 (dropping hidden `_env` at `EON_GP_ARG_REGS[start]`),
        // preserving `sret` in `Rdi` when `start == 1` and preserving flags/R15 on return.
        for i in start..(EON_GP_ARG_REGS.len() - 1) {
            tb.mov_rr(EON_GP_ARG_REGS[i], EON_GP_ARG_REGS[i + 1]);
        }
        if start + user_gp_slots == EON_GP_ARG_REGS.len() {
            // The 8th GP slot spilled to `[rsp + 8]` because `_env` occupied one reg!
            tb.load_mem(*EON_GP_ARG_REGS.last().unwrap(), Reg::Rsp, 8, 8, false);
        }
        let is_extern_only = eir.extern_fns.contains(&d)
            && !eir.by_key.contains_key(&CfgFnKey::TopDef(d));
        if is_extern_only && is_pie_or_shared && !is_builtin_rt_helper(&target_sym) {
            tb.jmp_plt_sym(target_sym);
        } else {
            tb.jmp_sym(target_sym);
        }
        let (tcode, trelocs) = tb.finish()?;
        pad_vec(text, 8);
        let toff = text.len();
        let tsym = format!("eon_fnref_thunk_{}", d.0);
        symbols.push(ElfSymbolDef {
            name: tsym,
            section: RelocSection::Text,
            offset: toff,
            size: tcode.len() as u64,
            is_func: true,
        });
        for r in trelocs {
            relocs.push(ElfReloc {
                section: RelocSection::Text,
                offset: toff + r.offset,
                kind: r.kind,
                target_sym: r.target_sym,
                addend: r.addend,
            });
        }
        text.extend_from_slice(&tcode);
    }
    Ok(())
}

fn pad_vec(v: &mut Vec<u8>, align: usize) {
    let target = align_up(v.len(), align);
    v.resize(target, 0);
}

// ============================================================================
// EIR Helpers for Live Interval Analysis
// ============================================================================

fn term_succs(t: &EirTerm) -> Vec<BlockId> {
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

fn eir_term_uses(t: &EirTerm) -> Vec<EirValue> {
    match t {
        EirTerm::Branch { cond, .. } => vec![*cond],
        EirTerm::SwitchInt { discr, .. } => vec![*discr],
        EirTerm::Return(Some(v)) => vec![*v],
        _ => Vec::new(),
    }
}

fn eir_inst_dst(k: &EirInstKind) -> Option<EirValue> {
    match k {
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
        | EirInstKind::GetVar { dst, .. } => Some(*dst),
        EirInstKind::Call { dst, .. } | EirInstKind::BuiltinCall { dst, .. } => *dst,
        EirInstKind::StoreSlot { .. }
        | EirInstKind::StorePlace { .. }
        | EirInstKind::PoolColumnStore { .. }
        | EirInstKind::InlineAsm { .. }
        | EirInstKind::SetVar { .. } => None,
    }
}

fn eir_inst_uses(k: &EirInstKind) -> Vec<EirValue> {
    let mut out = Vec::new();
    match k {
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
            if let EirPlaceBase::Val(v) = place.base { out.push(v); }
            for pr in &place.projs {
                if let EirProj::Index { index, .. } = pr { out.push(*index); }
            }
        }
        EirInstKind::StorePlace { place, val } => {
            if let EirPlaceBase::Val(v) = place.base { out.push(v); }
            for pr in &place.projs {
                if let EirProj::Index { index, .. } = pr { out.push(*index); }
            }
            out.push(*val);
        }
        EirInstKind::PoolColumnLoad { pool, index, .. } => {
            if let EirPlaceBase::Val(v) = pool { out.push(*v); }
            out.push(*index);
        }
        EirInstKind::PoolColumnStore { pool, index, val, .. } => {
            if let EirPlaceBase::Val(v) = pool { out.push(*v); }
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
            for (_, v) in inputs { out.push(*v); }
        }
        EirInstKind::Const { .. }
        | EirInstKind::LoadSlot { .. }
        | EirInstKind::SlotAddr { .. }
        | EirInstKind::GetVar { .. } => {}
    }
    out
}

fn eir_inst_slots(k: &EirInstKind, out: &mut Vec<EirSlot>) {
    match k {
        EirInstKind::LoadSlot { slot, .. }
        | EirInstKind::StoreSlot { slot, .. }
        | EirInstKind::SlotAddr { slot, .. } => out.push(*slot),
        EirInstKind::LoadPlace { place, .. }
        | EirInstKind::StorePlace { place, .. }
        | EirInstKind::PlaceAddr { place, .. } => {
            if let EirPlaceBase::Slot(s) = place.base {
                out.push(s);
            }
        }
        EirInstKind::PoolColumnLoad { pool, .. }
        | EirInstKind::PoolColumnStore { pool, .. } => {
            if let EirPlaceBase::Slot(s) = pool {
                out.push(*s);
            }
        }
        EirInstKind::BuiltinCall { arg_slots, .. } => {
            for s in arg_slots.iter().flatten() {
                out.push(*s);
            }
        }
        _ => {}
    }
}
