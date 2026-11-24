#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::ast::{self, AccessMode, Attr, DeclKind, ExprKind, GenericArg, GenericArgKind, GenericParam, Param, ParamModifier, TypeExpr, TypeKind, TypePath};
use crate::diag::{DiagBag, ErrorCode};
use crate::resolve::{BuiltinKind, DefId, ModuleId, PrimKind, ResolvedProgram, TopDefKind, TopScopeTarget};
use crate::span::{FileProfile, Interner, Span, Symbol};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TyId(pub u32);

impl TyId {
    pub const VOID: Self = Self(0);
    pub const NEVER: Self = Self(1);
    pub const BOOL: Self = Self(2);
    pub const CHAR: Self = Self(3);
    pub const STR: Self = Self(4);
    pub const TYPE: Self = Self(5);
    pub const COMPTIME_INT: Self = Self(6);
    pub const COMPTIME_FLOAT: Self = Self(7);
    pub const U8: Self = Self(8);
    pub const U16: Self = Self(9);
    pub const U32: Self = Self(10);
    pub const U64: Self = Self(11);
    pub const U128: Self = Self(12);
    pub const USIZE: Self = Self(13);
    pub const I8: Self = Self(14);
    pub const I16: Self = Self(15);
    pub const I32: Self = Self(16);
    pub const I64: Self = Self(17);
    pub const I128: Self = Self(18);
    pub const ISIZE: Self = Self(19);
    pub const F16: Self = Self(20);
    pub const F32: Self = Self(21);
    pub const F64: Self = Self(22);
    pub const PHYS_ADDR: Self = Self(23);
    pub const VIRT_ADDR: Self = Self(24);
    pub const DMA_ADDR: Self = Self(25);
    pub const ORDERING: Self = Self(26);
    pub const ARITH_ERROR: Self = Self(27);
    pub const BOUNDS_ERROR: Self = Self(28);
    pub const UTF8_ERROR: Self = Self(29);
    pub const OUT_OF_MEMORY: Self = Self(30);
    pub const ERROR: Self = Self(31);
    pub const MEM_ERROR: Self = Self(32);
    pub const STALE_ID: Self = Self(33);
    pub const IO_ERROR: Self = Self(34);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinErrorKind {
    ArithError,
    BoundsError,
    Utf8Error,
    OutOfMemory,
    MemError,
    StaleId,
    IoError,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSource {
    User(DefId),
    Builtin(BuiltinErrorKind),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BorrowInner {
    Single(TyId),
    Slice(TyId),
    Str,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GenericVal {
    Ty(TyId),
    Int(i128),
    Variant(Symbol),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PoolHandleKind {
    ArenaPool(Symbol),
    SlotPool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    Void,
    Never,
    Bool,
    Char,
    Str,
    TypeType,
    ComptimeInt,
    ComptimeFloat,
    Int {
        signed: bool,
        bits: u8,
        is_ptr_sized: bool,
    },
    Float(u8),
    PhysAddr,
    VirtAddr,
    DmaAddr,
    Vec {
        elem: TyId,
        len: u8,
    },
    Optional(TyId),
    ErrorUnion {
        ok: TyId,
        errors: Vec<ErrorSource>,
    },
    Borrow {
        is_mut: bool,
        inner: BorrowInner,
    },
    RawPtr {
        is_mut: bool,
        inner: TyId,
    },
    Array {
        len: u64,
        elem: TyId,
    },
    Tuple(Vec<TyId>),
    FnPtr {
        params: Vec<TyId>,
        ret: TyId,
    },
    Struct {
        def: DefId,
        args: Vec<GenericVal>,
    },
    AnonStruct(Vec<(Symbol, TyId)>),
    Pool {
        def: DefId,
        args: Vec<GenericVal>,
    },
    Enum(DefId),
    Union {
        def: DefId,
        args: Vec<GenericVal>,
    },
    Bitfield(DefId),
    ErrorSet(ErrorSource),
    Dyn(DefId),
    GenericParam {
        idx: u32,
        name: Symbol,
        is_comptime: bool,
    },
    Builtin(BuiltinKind, Vec<GenericVal>),
    Error,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StructLayoutMode {
    Eon,
    C,
    Std430,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
    pub has_niche: bool,
}

#[derive(Clone, Debug)]
pub struct FieldMeta {
    pub name: Symbol,
    pub span: Span,
    pub is_pub: bool,
    pub ty: TyId,
    pub has_default: bool,
    pub offset: u64,
    pub misaligned_in_packed: bool,
}

#[derive(Clone, Debug)]
pub struct StructMeta {
    pub def: DefId,
    pub module: ModuleId,
    pub is_linear: bool,
    pub is_view: bool,
    pub layout_mode: StructLayoutMode,
    pub is_packed: bool,
    pub explicit_align: Option<u64>,
    pub generics: Vec<GenericParamMeta>,
    pub fields: Vec<FieldMeta>,
    pub layout: Layout,
}

#[derive(Clone, Debug)]
pub struct PoolColumnMeta {
    pub name: Symbol,
    pub span: Span,
    pub is_pub: bool,
    pub ty: TyId,
    pub elem_layout: Layout,
    pub col_align: u64,
}

#[derive(Clone, Debug)]
pub struct PoolMeta {
    pub def: DefId,
    pub module: ModuleId,
    pub generics: Vec<GenericParamMeta>,
    pub columns: Vec<PoolColumnMeta>,
}

#[derive(Clone, Debug)]
pub struct EnumMeta {
    pub def: DefId,
    pub repr_ty: TyId,
    pub variants: Vec<(Symbol, i128)>,
    pub has_niche: bool,
}

#[derive(Clone, Debug)]
pub struct UnionVariantMeta {
    pub name: Symbol,
    pub span: Span,
    pub payload: Option<TyId>,
}

#[derive(Clone, Debug)]
pub struct UnionMeta {
    pub def: DefId,
    pub is_linear: bool,
    pub generics: Vec<GenericParamMeta>,
    pub variants: Vec<UnionVariantMeta>,
    pub layout: Layout,
}

#[derive(Clone, Debug)]
pub struct BitfieldMemberMeta {
    pub name: Symbol,
    pub span: Span,
    pub ty: TyId,
    pub lo: u8,
    pub hi: u8,
}

#[derive(Clone, Debug)]
pub struct BitfieldMeta {
    pub def: DefId,
    pub base_ty: TyId,
    pub base_bits: u8,
    pub members: Vec<BitfieldMemberMeta>,
}

#[derive(Clone, Debug)]
pub struct ErrorMeta {
    pub def: DefId,
    pub variants: Vec<Symbol>,
}

#[derive(Clone, Debug)]
pub struct GenericParamMeta {
    pub name: Symbol,
    pub is_comptime: bool,
    pub comptime_ty: Option<TyId>,
    pub bounds: Vec<DefId>,
    pub builtin_ord: bool,
    pub builtin_eq: bool,
}

#[derive(Clone, Debug)]
pub struct ParamSig {
    pub name: Symbol,
    pub span: Span,
    pub is_self: bool,
    pub is_comptime: bool,
    pub modifier: Option<ParamModifier>,
    pub access: Option<AccessMode>,
    pub ty: TyId,
    pub gpu_binding: Option<(u32, u32)>,
    pub is_push_const: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectSet {
    pub is_pure: bool,
    pub is_noalloc: bool,
    pub is_noblock: bool,
    pub is_interrupt: bool,
    pub is_naked: bool,
    pub is_export_c: bool,
    pub is_gpu: bool,
    pub gpu_local_size: Option<(u32, u32, u32)>,
    pub is_recursive: bool,
    pub is_hot: bool,
    pub is_inline: bool,
    pub save_simd: bool,
    pub stack_limit: Option<u64>,
    pub stack_span: Option<Span>,
    pub effect_span: Option<Span>,
}

#[derive(Clone, Debug)]
pub struct SysColAccess {
    pub param_idx: usize,
    pub mode: AccessMode,
    pub pool_def: DefId,
    pub columns: Vec<Symbol>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FnSig {
    pub def: Option<DefId>,
    pub module: ModuleId,
    pub name: Symbol,
    pub is_pub: bool,
    pub is_view: bool,
    pub is_comptime: bool,
    pub is_extern: bool,
    pub generics: Vec<GenericParamMeta>,
    pub params: Vec<ParamSig>,
    pub ret: TyId,
    pub has_gpu_attr: bool,
    pub effects: EffectSet,
}

#[derive(Clone, Debug)]
pub struct IfaceMeta {
    pub def: DefId,
    pub module: ModuleId,
    pub generics: Vec<GenericParamMeta>,
    pub methods: Vec<FnSig>,
}

#[derive(Clone, Debug)]
pub struct ConstMeta {
    pub def: DefId,
    pub ty: TyId,
    pub int_val: Option<i128>,
    pub ct_val: Option<crate::comptime::CtVal>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MethodTarget {
    Def(DefId),
    Builtin(BuiltinKind),
}

#[derive(Clone, Debug)]
pub struct MethodEntry {
    pub def: DefId,
    pub module: ModuleId,
    pub iface: Option<DefId>,
    pub sig: FnSig,
    pub decl_idx_in_impl: (usize, usize), // (unit_idx, top_decl_idx) -> wait, keep impl_idx & sub_idx
}

#[derive(Clone, Debug)]
pub struct ImplRef {
    pub module: ModuleId,
    pub impl_decl_idx: usize,
    pub iface: Option<DefId>,
    pub target: MethodTarget,
    pub methods: Vec<(DefId, usize, FnSig)>, // (synthetic/def id or idx in impl.decls)
}

#[derive(Clone, Debug)]
pub struct TypeAliasMeta {
    pub def: DefId,
    pub generics: Vec<GenericParamMeta>,
    pub ty: TyId,
}

#[derive(Clone, Debug)]
pub struct GenericCtx {
    pub params: Vec<(Symbol, u32, bool)>, // (name, idx, is_comptime)
    pub self_ty: Option<TyId>,
    pub comptime_vals: BTreeMap<Symbol, i128>,
}

impl GenericCtx {
    pub fn empty() -> Self {
        Self {
            params: Vec::new(),
            self_ty: None,
            comptime_vals: BTreeMap::new(),
        }
    }
}

pub struct TypeEnv {
    types: Vec<Ty>,
    intern: BTreeMap<Ty, TyId>,

    pub structs: BTreeMap<DefId, StructMeta>,
    pub pools: BTreeMap<DefId, PoolMeta>,
    pub enums: BTreeMap<DefId, EnumMeta>,
    pub unions: BTreeMap<DefId, UnionMeta>,
    pub bitfields: BTreeMap<DefId, BitfieldMeta>,
    pub errors: BTreeMap<DefId, ErrorMeta>,
    pub ifaces: BTreeMap<DefId, IfaceMeta>,
    pub type_aliases: BTreeMap<DefId, TypeAliasMeta>,
    pub fns: BTreeMap<DefId, FnSig>,
    pub consts: BTreeMap<DefId, ConstMeta>,
    pub sys_access: BTreeMap<DefId, Vec<SysColAccess>>,
    pub impls: Vec<ImplRef>,

    visiting_type_defs: BTreeSet<DefId>,
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        let mut env = Self {
            types: Vec::new(),
            intern: BTreeMap::new(),
            structs: BTreeMap::new(),
            pools: BTreeMap::new(),
            enums: BTreeMap::new(),
            unions: BTreeMap::new(),
            bitfields: BTreeMap::new(),
            errors: BTreeMap::new(),
            ifaces: BTreeMap::new(),
            type_aliases: BTreeMap::new(),
            fns: BTreeMap::new(),
            consts: BTreeMap::new(),
            sys_access: BTreeMap::new(),
            impls: Vec::new(),
            visiting_type_defs: BTreeSet::new(),
        };
        env.seed_primitives();
        env
    }

    fn seed_primitives(&mut self) {
        let fixed = [
            (TyId::VOID, Ty::Void),
            (TyId::NEVER, Ty::Never),
            (TyId::BOOL, Ty::Bool),
            (TyId::CHAR, Ty::Char),
            (TyId::STR, Ty::Str),
            (TyId::TYPE, Ty::TypeType),
            (TyId::COMPTIME_INT, Ty::ComptimeInt),
            (TyId::COMPTIME_FLOAT, Ty::ComptimeFloat),
            (TyId::U8, Ty::Int { signed: false, bits: 8, is_ptr_sized: false }),
            (TyId::U16, Ty::Int { signed: false, bits: 16, is_ptr_sized: false }),
            (TyId::U32, Ty::Int { signed: false, bits: 32, is_ptr_sized: false }),
            (TyId::U64, Ty::Int { signed: false, bits: 64, is_ptr_sized: false }),
            (TyId::U128, Ty::Int { signed: false, bits: 128, is_ptr_sized: false }),
            (TyId::USIZE, Ty::Int { signed: false, bits: 64, is_ptr_sized: true }),
            (TyId::I8, Ty::Int { signed: true, bits: 8, is_ptr_sized: false }),
            (TyId::I16, Ty::Int { signed: true, bits: 16, is_ptr_sized: false }),
            (TyId::I32, Ty::Int { signed: true, bits: 32, is_ptr_sized: false }),
            (TyId::I64, Ty::Int { signed: true, bits: 64, is_ptr_sized: false }),
            (TyId::I128, Ty::Int { signed: true, bits: 128, is_ptr_sized: false }),
            (TyId::ISIZE, Ty::Int { signed: true, bits: 64, is_ptr_sized: true }),
            (TyId::F16, Ty::Float(16)),
            (TyId::F32, Ty::Float(32)),
            (TyId::F64, Ty::Float(64)),
            (TyId::PHYS_ADDR, Ty::PhysAddr),
            (TyId::VIRT_ADDR, Ty::VirtAddr),
            (TyId::DMA_ADDR, Ty::DmaAddr),
            (TyId::ORDERING, Ty::Builtin(BuiltinKind::Ordering, Vec::new())),
            (TyId::ARITH_ERROR, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::ArithError))),
            (TyId::BOUNDS_ERROR, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::BoundsError))),
            (TyId::UTF8_ERROR, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::Utf8Error))),
            (TyId::OUT_OF_MEMORY, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::OutOfMemory))),
            (TyId::ERROR, Ty::Error),
            (TyId::MEM_ERROR, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::MemError))),
            (TyId::STALE_ID, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::StaleId))),
            (TyId::IO_ERROR, Ty::ErrorSet(ErrorSource::Builtin(BuiltinErrorKind::IoError))),
        ];
        for (expected_id, ty) in fixed {
            let id = self.intern(ty);
            debug_assert_eq!(id, expected_id);
        }
    }

    pub fn intern(&mut self, ty: Ty) -> TyId {
        if let Some(&id) = self.intern.get(&ty) {
            return id;
        }
        let id = TyId(self.types.len() as u32);
        self.intern.insert(ty.clone(), id);
        self.types.push(ty);
        id
    }

    pub fn find_interned(&self, ty: &Ty) -> Option<TyId> {
        self.intern.get(ty).copied()
    }

    #[inline]
    pub fn get(&self, id: TyId) -> &Ty {
        &self.types[id.0 as usize]
    }

    pub fn format_ty(&self, id: TyId, interner: &Interner) -> String {
        match self.get(id) {
            Ty::Void => "void".into(),
            Ty::Never => "never".into(),
            Ty::Bool => "bool".into(),
            Ty::Char => "char".into(),
            Ty::Str => "str".into(),
            Ty::TypeType => "type".into(),
            Ty::ComptimeInt => "comptime_int".into(),
            Ty::ComptimeFloat => "comptime_float".into(),
            Ty::Int { signed, bits, is_ptr_sized } => {
                if *is_ptr_sized {
                    if *signed { "isize".into() } else { "usize".into() }
                } else if *signed {
                    format!("i{}", bits)
                } else {
                    format!("u{}", bits)
                }
            }
            Ty::Float(b) => format!("f{}", b),
            Ty::PhysAddr => "PhysAddr".into(),
            Ty::VirtAddr => "VirtAddr".into(),
            Ty::DmaAddr => "DmaAddr".into(),
            Ty::Vec { elem, len } => format!("vec[{}, {}]", self.format_ty(*elem, interner), len),
            Ty::Optional(inner) => format!("?{}", self.format_ty(*inner, interner)),
            Ty::ErrorUnion { ok, errors } => {
                let err_s: Vec<String> = errors
                    .iter()
                    .map(|e| self.format_error_source(*e, interner))
                    .collect();
                if *ok == TyId::VOID {
                    format!("!{}", err_s.join(" | "))
                } else {
                    format!("{} ! {}", self.format_ty(*ok, interner), err_s.join(" | "))
                }
            }
            Ty::Borrow { is_mut, inner } => {
                let mut_s = if *is_mut { "mut " } else { "" };
                match inner {
                    BorrowInner::Single(t) => format!("*{}{}", mut_s, self.format_ty(*t, interner)),
                    BorrowInner::Slice(t) => format!("*{}[{}]", mut_s, self.format_ty(*t, interner)),
                    BorrowInner::Str => format!("*{}str", mut_s),
                }
            }
            Ty::RawPtr { is_mut, inner } => {
                let n = if *is_mut { "rawptr_mut" } else { "rawptr" };
                format!("{}[{}]", n, self.format_ty(*inner, interner))
            }
            Ty::Array { len, elem } => format!("[{}]{}", len, self.format_ty(*elem, interner)),
            Ty::Tuple(elems) => {
                let parts: Vec<String> = elems.iter().map(|&t| self.format_ty(t, interner)).collect();
                format!("({})", parts.join(", "))
            }
            Ty::FnPtr { params, ret } => {
                let parts: Vec<String> = params.iter().map(|&t| self.format_ty(t, interner)).collect();
                format!("fn({}) -> {}", parts.join(", "), self.format_ty(*ret, interner))
            }
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def)
            | Ty::Dyn(def) => {
                let _ = def;
                format!("type#{}", def.0)
            }
            Ty::AnonStruct(_) => "struct { ... }".into(),
            Ty::ErrorSet(src) => self.format_error_source(*src, interner),
            Ty::GenericParam { name, .. } => interner.get(*name).to_owned(),
            Ty::Builtin(b, args) => {
                let first_arg = args.first().map(|a| match a {
                    GenericVal::Ty(t) => self.format_ty(*t, interner),
                    GenericVal::Int(n) => n.to_string(),
                    GenericVal::Variant(sym) => interner.get(*sym).to_owned(),
                });
                match b {
                    BuiltinKind::Arena => "Arena".into(),
                    BuiltinKind::ArenaStr => "ArenaStr".into(),
                    BuiltinKind::AllocCap => "AllocCap".into(),
                    BuiltinKind::Env => "Env".into(),
                    BuiltinKind::FsCap => "FsCap".into(),
                    BuiltinKind::DirCap => "DirCap".into(),
                    BuiltinKind::NetCap => "NetCap".into(),
                    BuiltinKind::ClockCap => "ClockCap".into(),
                    BuiltinKind::IoCap => "IoCap".into(),
                    BuiltinKind::ArgsCap => "ArgsCap".into(),
                    BuiltinKind::BuildCap => "BuildCap".into(),
                    BuiltinKind::IoError => "IoError".into(),
                    BuiltinKind::Schedule => "Schedule".into(),
                    BuiltinKind::Atomic => format!("Atomic[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GpuBuf => format!("GpuBuf[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GpuImage2D => format!("GpuImage2D[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::MmioReg => format!("MmioReg[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::ArenaPool => format!("ArenaPool[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::SlotPool => format!("SlotPool[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::Buf => format!("Buf[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::LinearBuf => format!("LinearBuf[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::IdBrand => format!("Id[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GenId => format!("GenId[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::Link => format!("Link[{}]", first_arg.unwrap_or_default()),
                    _ => format!("{:?}", b),
                }
            }
            Ty::Error => "<error>".into(),
        }
    }

    pub fn format_ty_named(
        &self,
        id: TyId,
        prog: &ResolvedProgram,
        interner: &Interner,
    ) -> String {
        match self.get(id) {
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def) => interner.get(prog.defs[def.0 as usize].name).to_owned(),
            Ty::Dyn(def) => format!("dyn {}", interner.get(prog.defs[def.0 as usize].name)),
            _ => self.format_ty(id, interner),
        }
    }

    pub fn format_ty_resolved(
        &self,
        id: TyId,
        prog: &ResolvedProgram,
        interner: &Interner,
    ) -> String {
        match self.get(id) {
            Ty::Vec { elem, len } => {
                format!("vec[{}, {}]", self.format_ty_resolved(*elem, prog, interner), len)
            }
            Ty::Optional(inner) => {
                format!("?{}", self.format_ty_resolved(*inner, prog, interner))
            }
            Ty::ErrorUnion { ok, errors } => {
                let err_s: Vec<String> = errors
                    .iter()
                    .map(|e| self.format_error_source_resolved(*e, prog, interner))
                    .collect();
                if *ok == TyId::VOID {
                    format!("!{}", err_s.join(" | "))
                } else {
                    format!(
                        "{} ! {}",
                        self.format_ty_resolved(*ok, prog, interner),
                        err_s.join(" | ")
                    )
                }
            }
            Ty::Borrow { is_mut, inner } => {
                let mut_s = if *is_mut { "mut " } else { "" };
                match inner {
                    BorrowInner::Single(t) => {
                        format!("*{}{}", mut_s, self.format_ty_resolved(*t, prog, interner))
                    }
                    BorrowInner::Slice(t) => {
                        format!("*{}[{}]", mut_s, self.format_ty_resolved(*t, prog, interner))
                    }
                    BorrowInner::Str => format!("*{}str", mut_s),
                }
            }
            Ty::RawPtr { is_mut, inner } => {
                let n = if *is_mut { "rawptr_mut" } else { "rawptr" };
                format!("{}[{}]", n, self.format_ty_resolved(*inner, prog, interner))
            }
            Ty::Array { len, elem } => {
                format!("[{}]{}", len, self.format_ty_resolved(*elem, prog, interner))
            }
            Ty::Tuple(elems) => {
                let parts: Vec<String> = elems
                    .iter()
                    .map(|&t| self.format_ty_resolved(t, prog, interner))
                    .collect();
                format!("({})", parts.join(", "))
            }
            Ty::FnPtr { params, ret } => {
                let parts: Vec<String> = params
                    .iter()
                    .map(|&t| self.format_ty_resolved(t, prog, interner))
                    .collect();
                format!(
                    "fn({}) -> {}",
                    parts.join(", "),
                    self.format_ty_resolved(*ret, prog, interner)
                )
            }
            Ty::Struct { def, args } | Ty::Pool { def, args } | Ty::Union { def, args } => {
                let base = def_qualified_name(prog, *def, interner);
                if args.is_empty() {
                    base
                } else {
                    let parts: Vec<String> = args
                        .iter()
                        .map(|a| self.format_generic_val_resolved(*a, prog, interner))
                        .collect();
                    format!("{base}[{}]", parts.join(", "))
                }
            }
            Ty::Enum(def) | Ty::Bitfield(def) => def_qualified_name(prog, *def, interner),
            Ty::Dyn(def) => format!("dyn {}", def_qualified_name(prog, *def, interner)),
            Ty::AnonStruct(fields) => {
                let parts: Vec<String> = fields
                    .iter()
                    .map(|&(name, fty)| {
                        format!(
                            "{}: {}",
                            interner.get(name),
                            self.format_ty_resolved(fty, prog, interner)
                        )
                    })
                    .collect();
                format!("struct {{ {} }}", parts.join(", "))
            }
            Ty::ErrorSet(src) => self.format_error_source_resolved(*src, prog, interner),
            Ty::Builtin(b, args) => {
                let first_arg = args
                    .first()
                    .map(|a| self.format_generic_val_resolved(*a, prog, interner));
                match b {
                    BuiltinKind::Arena => "Arena".into(),
                    BuiltinKind::ArenaStr => "ArenaStr".into(),
                    BuiltinKind::AllocCap => "AllocCap".into(),
                    BuiltinKind::Env => "Env".into(),
                    BuiltinKind::FsCap => "FsCap".into(),
                    BuiltinKind::DirCap => "DirCap".into(),
                    BuiltinKind::NetCap => "NetCap".into(),
                    BuiltinKind::ClockCap => "ClockCap".into(),
                    BuiltinKind::IoCap => "IoCap".into(),
                    BuiltinKind::ArgsCap => "ArgsCap".into(),
                    BuiltinKind::BuildCap => "BuildCap".into(),
                    BuiltinKind::IoError => "IoError".into(),
                    BuiltinKind::Schedule => "Schedule".into(),
                    BuiltinKind::Atomic => format!("Atomic[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GpuBuf => format!("GpuBuf[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GpuImage2D => {
                        format!("GpuImage2D[{}]", first_arg.unwrap_or_default())
                    }
                    BuiltinKind::MmioReg => format!("MmioReg[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::ArenaPool => {
                        format!("ArenaPool[{}]", first_arg.unwrap_or_default())
                    }
                    BuiltinKind::SlotPool => {
                        format!("SlotPool[{}]", first_arg.unwrap_or_default())
                    }
                    BuiltinKind::Buf => format!("Buf[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::LinearBuf => {
                        format!("LinearBuf[{}]", first_arg.unwrap_or_default())
                    }
                    BuiltinKind::IdBrand => format!("Id[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::GenId => format!("GenId[{}]", first_arg.unwrap_or_default()),
                    BuiltinKind::Link => format!("Link[{}]", first_arg.unwrap_or_default()),
                    _ => format!("{:?}", b),
                }
            }
            _ => self.format_ty(id, interner),
        }
    }

    fn format_generic_val_resolved(
        &self,
        gv: GenericVal,
        prog: &ResolvedProgram,
        interner: &Interner,
    ) -> String {
        match gv {
            GenericVal::Ty(t) => self.format_ty_resolved(t, prog, interner),
            GenericVal::Int(n) => n.to_string(),
            GenericVal::Variant(sym) => interner.get(sym).to_owned(),
        }
    }

    fn format_error_source_resolved(
        &self,
        src: ErrorSource,
        prog: &ResolvedProgram,
        interner: &Interner,
    ) -> String {
        match src {
            ErrorSource::User(d) => def_qualified_name(prog, d, interner),
            ErrorSource::Builtin(b) => format!("{:?}", b),
        }
    }

    fn format_error_source(&self, src: ErrorSource, interner: &Interner) -> String {
        let _ = interner;
        match src {
            ErrorSource::User(d) => format!("err#{}", d.0),
            ErrorSource::Builtin(b) => format!("{:?}", b),
        }
    }

    pub fn is_integer(&self, id: TyId) -> bool {
        matches!(self.get(id), Ty::Int { .. } | Ty::ComptimeInt)
    }

    pub fn is_concrete_integer(&self, id: TyId) -> bool {
        matches!(self.get(id), Ty::Int { .. })
    }

    pub fn is_signed_integer(&self, id: TyId) -> bool {
        matches!(self.get(id), Ty::Int { signed: true, .. } | Ty::ComptimeInt)
    }

    pub fn is_float(&self, id: TyId) -> bool {
        matches!(self.get(id), Ty::Float(_) | Ty::ComptimeFloat)
    }

    pub fn is_numeric(&self, id: TyId) -> bool {
        self.is_integer(id) || self.is_float(id)
    }

    pub fn is_address(&self, id: TyId) -> bool {
        matches!(self.get(id), Ty::PhysAddr | Ty::VirtAddr | Ty::DmaAddr)
    }

    pub fn contains_type_type(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::TypeType => true,
            Ty::Optional(inner) | Ty::Array { elem: inner, .. } => self.contains_type_type(*inner),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.contains_type_type(t)),
            Ty::AnonStruct(fields) => fields.iter().any(|(_, t)| self.contains_type_type(*t)),
            _ => false,
        }
    }

    pub fn contains_second_class_ref(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Borrow { .. } | Ty::FnPtr { .. } => true,
            Ty::Optional(inner) | Ty::Array { elem: inner, .. } => {
                self.contains_second_class_ref(*inner)
            }
            Ty::ErrorUnion { ok, .. } => self.contains_second_class_ref(*ok),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.contains_second_class_ref(t)),
            Ty::AnonStruct(fields) => fields
                .iter()
                .any(|(_, t)| self.contains_second_class_ref(*t)),
            Ty::Struct { def, .. } => self
                .structs
                .get(def)
                .is_some_and(|s| s.is_view || s.fields.iter().any(|f| self.contains_second_class_ref(f.ty))),
            Ty::Union { def, .. } => self.unions.get(def).is_some_and(|u| {
                u.variants
                    .iter()
                    .any(|v| v.payload.is_some_and(|p| self.contains_second_class_ref(p)))
            }),
            _ => false,
        }
    }

    pub fn is_borrow_or_view(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Borrow { .. } => true,
            Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::ArenaStr, _) => true,
            Ty::Struct { def, .. } => self
                .structs
                .get(def)
                .is_some_and(|s| s.is_view || s.fields.iter().any(|f| self.is_borrow_or_view(f.ty))),
            Ty::Optional(inner) | Ty::Array { elem: inner, .. } => self.is_borrow_or_view(*inner),
            Ty::ErrorUnion { ok, .. } => self.is_borrow_or_view(*ok),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.is_borrow_or_view(t)),
            Ty::AnonStruct(fields) => fields.iter().any(|(_, t)| self.is_borrow_or_view(*t)),
            _ => false,
        }
    }

    pub fn error_sources_have_variant(
        &self,
        errors: &[ErrorSource],
        variant: Symbol,
        interner: &Interner,
    ) -> bool {
        let vname = interner.get(variant);
        for src in errors {
            match src {
                ErrorSource::User(d) => {
                    if let Some(meta) = self.errors.get(d)
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

    pub fn contains_borrowing_value(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::ArenaStr, _) => true,
            Ty::Optional(inner) | Ty::Array { elem: inner, .. } => {
                self.contains_borrowing_value(*inner)
            }
            Ty::ErrorUnion { ok, .. } => self.contains_borrowing_value(*ok),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.contains_borrowing_value(t)),
            Ty::AnonStruct(fields) => fields
                .iter()
                .any(|(_, t)| self.contains_borrowing_value(*t)),
            _ => false,
        }
    }

    pub fn contains_id_brand(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Builtin(BuiltinKind::IdBrand, _) => true,
            Ty::Optional(inner) | Ty::Array { elem: inner, .. } => self.contains_id_brand(*inner),
            Ty::ErrorUnion { ok, .. } => self.contains_id_brand(*ok),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.contains_id_brand(t)),
            Ty::AnonStruct(fields) => fields.iter().any(|(_, t)| self.contains_id_brand(*t)),
            _ => false,
        }
    }

    pub fn gen_id_schema(&self, id: TyId) -> Option<TyId> {
        match self.get(id) {
            Ty::Builtin(BuiltinKind::GenId, args) => match args.first() {
                Some(GenericVal::Ty(schema)) => Some(*schema),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_gen_id_of(&self, id: TyId, schema: TyId) -> bool {
        self.gen_id_schema(id) == Some(schema)
    }

    pub fn resolve_pool_link_ty(
        &mut self,
        ty: TyId,
        expected_pool_ty: TyId,
        kind: PoolHandleKind,
    ) -> TyId {
        match self.get(ty).clone() {
            Ty::Builtin(BuiltinKind::Link, args) => {
                let target_ty = match args.first() {
                    Some(GenericVal::Ty(t)) => *t,
                    _ => expected_pool_ty,
                };
                if target_ty != expected_pool_ty {
                    return ty;
                }
                match kind {
                    PoolHandleKind::ArenaPool(brand) => {
                        self.intern(Ty::Builtin(BuiltinKind::IdBrand, vec![GenericVal::Variant(brand)]))
                    }
                    PoolHandleKind::SlotPool => {
                        self.intern(Ty::Builtin(BuiltinKind::GenId, vec![GenericVal::Ty(target_ty)]))
                    }
                }
            }
            Ty::Optional(inner) => {
                let r = self.resolve_pool_link_ty(inner, expected_pool_ty, kind);
                if r != inner { self.intern(Ty::Optional(r)) } else { ty }
            }
            Ty::Array { elem, len } => {
                let r = self.resolve_pool_link_ty(elem, expected_pool_ty, kind);
                if r != elem { self.intern(Ty::Array { elem: r, len }) } else { ty }
            }
            Ty::ErrorUnion { ok, errors } => {
                let r = self.resolve_pool_link_ty(ok, expected_pool_ty, kind);
                if r != ok { self.intern(Ty::ErrorUnion { ok: r, errors }) } else { ty }
            }
            Ty::Tuple(elems) => {
                let mut changed = false;
                let mapped: Vec<TyId> = elems
                    .iter()
                    .map(|&e| {
                        let r = self.resolve_pool_link_ty(e, expected_pool_ty, kind);
                        if r != e { changed = true; }
                        r
                    })
                    .collect();
                if changed { self.intern(Ty::Tuple(mapped)) } else { ty }
            }
            Ty::AnonStruct(fields) => {
                let mut changed = false;
                let mapped: Vec<(Symbol, TyId)> = fields
                    .iter()
                    .map(|&(n, t)| {
                        let r = self.resolve_pool_link_ty(t, expected_pool_ty, kind);
                        if r != t { changed = true; }
                        (n, r)
                    })
                    .collect();
                if changed { self.intern(Ty::AnonStruct(mapped)) } else { ty }
            }
            Ty::Borrow { is_mut, inner } => match inner {
                BorrowInner::Single(t) => {
                    let r = self.resolve_pool_link_ty(t, expected_pool_ty, kind);
                    if r != t {
                        self.intern(Ty::Borrow { is_mut, inner: BorrowInner::Single(r) })
                    } else {
                        ty
                    }
                }
                BorrowInner::Slice(t) => {
                    let r = self.resolve_pool_link_ty(t, expected_pool_ty, kind);
                    if r != t {
                        self.intern(Ty::Borrow { is_mut, inner: BorrowInner::Slice(r) })
                    } else {
                        ty
                    }
                }
                BorrowInner::Str => ty,
            },
            _ => ty,
        }
    }

    pub fn is_linear(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Struct { def, .. } => self
                .structs
                .get(def)
                .map(|m| m.is_linear)
                .unwrap_or(false),
            Ty::Union { def, .. } => self
                .unions
                .get(def)
                .map(|m| m.is_linear)
                .unwrap_or(false),
            Ty::Array { elem, .. } | Ty::Optional(elem) => self.is_linear(*elem),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.is_linear(t)),
            Ty::AnonStruct(fields) => fields.iter().any(|(_, t)| self.is_linear(*t)),
            Ty::ErrorUnion { ok, .. } => self.is_linear(*ok),
            Ty::Builtin(
                BuiltinKind::Arena | BuiltinKind::SlotPool | BuiltinKind::LinearBuf | BuiltinKind::DirCap,
                _,
            ) => true,
            _ => false,
        }
    }

    pub fn is_trivial_copy(&self, id: TyId) -> bool {
        if self.is_linear(id) {
            return false;
        }
        self.layout(id).size <= 16
    }

    pub fn has_niche_forbidden_for_bitcast(&self, id: TyId) -> bool {
        match self.get(id) {
            Ty::Bool
            | Ty::Char
            | Ty::Optional(_)
            | Ty::Borrow { .. }
            | Ty::Enum(_)
            | Ty::ErrorSet(_)
            | Ty::ErrorUnion { .. }
            | Ty::Union { .. }
            | Ty::Dyn(_)
            | Ty::FnPtr { .. } => true,
            Ty::Int { bits, is_ptr_sized, .. } if !is_ptr_sized => {
                !matches!(*bits, 8 | 16 | 32 | 64 | 128)
            }
            Ty::Array { elem, .. } => self.has_niche_forbidden_for_bitcast(*elem),
            Ty::Tuple(elems) => elems.iter().any(|&t| self.has_niche_forbidden_for_bitcast(t)),
            Ty::AnonStruct(fields) => fields
                .iter()
                .any(|(_, t)| self.has_niche_forbidden_for_bitcast(*t)),
            Ty::Struct { def, .. } => self
                .structs
                .get(def)
                .map(|s| s.fields.iter().any(|f| self.has_niche_forbidden_for_bitcast(f.ty)))
                .unwrap_or(false),
            _ => false,
        }
    }

    pub fn layout(&self, id: TyId) -> Layout {
        match self.get(id) {
            Ty::Void | Ty::Never | Ty::TypeType | Ty::Error => Layout {
                size: 0,
                align: 1,
                has_niche: false,
            },
            Ty::ComptimeInt | Ty::ComptimeFloat => Layout {
                size: 0,
                align: 1,
                has_niche: false,
            },
            Ty::Bool => Layout {
                size: 1,
                align: 1,
                has_niche: true,
            },
            Ty::Char => Layout {
                size: 4,
                align: 4,
                has_niche: true,
            },
            Ty::Str => Layout {
                size: 0,
                align: 1,
                has_niche: false,
            },
            Ty::Int { bits, is_ptr_sized, .. } => {
                if *is_ptr_sized {
                    Layout {
                        size: 8,
                        align: 8,
                        has_niche: false,
                    }
                } else {
                    let bytes = match *bits {
                        1..=8 => 1,
                        9..=16 => 2,
                        17..=32 => 4,
                        33..=64 => 8,
                        _ => 16,
                    };
                    let has_niche = !matches!(*bits, 8 | 16 | 32 | 64 | 128);
                    Layout {
                        size: bytes,
                        align: bytes,
                        has_niche,
                    }
                }
            }
            Ty::Float(bits) => {
                let bytes = (*bits as u64) / 8;
                Layout {
                    size: bytes,
                    align: bytes,
                    has_niche: false,
                }
            }
            Ty::PhysAddr | Ty::VirtAddr | Ty::DmaAddr => Layout {
                size: 8,
                align: 8,
                has_niche: false,
            },
            Ty::Vec { elem, len } => {
                let el = self.layout(*elem);
                let size = el.size.saturating_mul(*len as u64);
                let align = size.clamp(1, 32).next_power_of_two().min(32);
                Layout {
                    size,
                    align,
                    has_niche: false,
                }
            }
            Ty::Borrow { inner, .. } => match inner {
                BorrowInner::Single(target) => {
                    let is_fat = matches!(self.get(*target), Ty::Dyn(_));
                    Layout {
                        size: if is_fat { 16 } else { 8 },
                        align: 8,
                        has_niche: true,
                    }
                }
                BorrowInner::Slice(_) | BorrowInner::Str => Layout {
                    size: 16,
                    align: 8,
                    has_niche: true,
                },
            },
            Ty::RawPtr { .. } => Layout {
                size: 8,
                align: 8,
                has_niche: true,
            },
            Ty::FnPtr { .. } => Layout {
                size: 16,
                align: 8,
                has_niche: true,
            },
            Ty::Dyn(_) => Layout {
                size: 0,
                align: 1,
                has_niche: false,
            },
            Ty::Array { len, elem } => {
                let el = self.layout(*elem);
                Layout {
                    size: el.size.saturating_mul(*len),
                    align: el.align,
                    has_niche: *len > 0 && el.has_niche,
                }
            }
            Ty::Tuple(elems) => {
                let mut pairs: Vec<(u64, u64, bool)> = elems
                    .iter()
                    .map(|&e| {
                        let l = self.layout(e);
                        (l.size, l.align, l.has_niche)
                    })
                    .collect();
                // Tuples use Eon optimal field ordering too
                pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
                let mut offset = 0u64;
                let mut max_align = 1u64;
                let mut niche = false;
                for (sz, al, hn) in pairs {
                    offset = align_up(offset, al);
                    offset += sz;
                    max_align = max_align.max(al);
                    niche |= hn;
                }
                Layout {
                    size: align_up(offset, max_align),
                    align: max_align,
                    has_niche: niche,
                }
            }
            Ty::Optional(inner) => {
                let l = self.layout(*inner);
                if l.has_niche {
                    Layout {
                        size: l.size,
                        align: l.align,
                        has_niche: false,
                    }
                } else {
                    Layout {
                        size: align_up(l.size + 1, l.align),
                        align: l.align,
                        has_niche: false,
                    }
                }
            }
            Ty::ErrorSet(_) => Layout {
                size: 2,
                align: 2,
                has_niche: true, // error code 0 is reserved for success
            },
            Ty::ErrorUnion { ok, .. } => {
                let l = self.layout(*ok);
                if l.size == 0 {
                    Layout {
                        size: 2,
                        align: 2,
                        has_niche: false,
                    }
                } else {
                    let al = l.align.max(2);
                    Layout {
                        size: align_up(l.size + 2, al),
                        align: al,
                        has_niche: false,
                    }
                }
            }
            Ty::Struct { def, .. } => self
                .structs
                .get(def)
                .map(|m| m.layout)
                .unwrap_or(Layout {
                    size: 0,
                    align: 1,
                    has_niche: false,
                }),
            Ty::AnonStruct(fields) => {
                let mut items: Vec<(u64, u64, bool)> = fields
                    .iter()
                    .map(|&(_, ty)| {
                        let l = self.layout(ty);
                        (l.size, l.align, l.has_niche)
                    })
                    .collect();
                items.sort_by_key(|a| std::cmp::Reverse(a.1));
                let mut offset = 0u64;
                let mut max_align = 1u64;
                let mut niche = false;
                for (sz, al, hn) in items {
                    offset = align_up(offset, al);
                    offset += sz;
                    max_align = max_align.max(al);
                    niche |= hn;
                }
                Layout {
                    size: align_up(offset, max_align),
                    align: max_align,
                    has_niche: niche,
                }
            }
            Ty::Enum(def) => self
                .enums
                .get(def)
                .map(|e| {
                    let mut l = self.layout(e.repr_ty);
                    l.has_niche = e.has_niche;
                    l
                })
                .unwrap_or(Layout {
                    size: 1,
                    align: 1,
                    has_niche: true,
                }),
            Ty::Union { def, .. } => self
                .unions
                .get(def)
                .map(|u| u.layout)
                .unwrap_or(Layout {
                    size: 1,
                    align: 1,
                    has_niche: false,
                }),
            Ty::Bitfield(def) => self
                .bitfields
                .get(def)
                .map(|b| self.layout(b.base_ty))
                .unwrap_or(Layout {
                    size: 8,
                    align: 8,
                    has_niche: false,
                }),
            Ty::Pool { .. } => Layout {
                size: 0,
                align: 1,
                has_niche: false,
            },
            Ty::GenericParam { .. } => Layout {
                size: 8,
                align: 8,
                has_niche: false,
            },
            Ty::Builtin(kind, args) => match kind {
                BuiltinKind::Ordering => Layout {
                    size: 1,
                    align: 1,
                    has_niche: true,
                },
                BuiltinKind::IdBrand => Layout {
                    size: 4,
                    align: 4,
                    has_niche: true,
                },
                BuiltinKind::GenId => Layout {
                    size: 8,
                    align: 4,
                    has_niche: true,
                },
                BuiltinKind::Link => Layout {
                    size: 8,
                    align: 4,
                    has_niche: true,
                },
                BuiltinKind::MmioReg => {
                    if let Some(GenericVal::Ty(inner)) = args.first() {
                        self.layout(*inner)
                    } else {
                        Layout {
                            size: 4,
                            align: 4,
                            has_niche: false,
                        }
                    }
                }
                BuiltinKind::Arena
                | BuiltinKind::ArenaPool
                | BuiltinKind::SlotPool
                | BuiltinKind::Buf
                | BuiltinKind::LinearBuf => Layout {
                    size: 32, // handle struct > 16B
                    align: 8,
                    has_niche: false,
                },
                BuiltinKind::ArenaStr => Layout {
                    size: 16,
                    align: 8,
                    has_niche: true,
                },
                BuiltinKind::AllocCap => Layout {
                    size: 40,
                    align: 8,
                    has_niche: false,
                },
                BuiltinKind::Env => Layout {
                    size: 88,
                    align: 8,
                    has_niche: false,
                },
                BuiltinKind::ArgsCap | BuiltinKind::BuildCap => Layout {
                    size: 16,
                    align: 8,
                    has_niche: false,
                },
                BuiltinKind::FsCap
                | BuiltinKind::DirCap
                | BuiltinKind::NetCap
                | BuiltinKind::ClockCap
                | BuiltinKind::IoCap
                | BuiltinKind::Schedule => Layout {
                    size: 8,
                    align: 8,
                    has_niche: false,
                },
                BuiltinKind::Atomic => {
                    if let Some(GenericVal::Ty(inner)) = args.first().copied() {
                        let l = self.layout(inner);
                        Layout {
                            size: l.size.max(1),
                            align: l.align.max(1),
                            has_niche: false,
                        }
                    } else {
                        Layout {
                            size: 8,
                            align: 8,
                            has_niche: false,
                        }
                    }
                }
                _ => Layout {
                    size: 8,
                    align: 8,
                    has_niche: false,
                },
            },
        }
    }
}

#[inline]
pub fn align_up(offset: u64, align: u64) -> u64 {
    if align <= 1 {
        offset
    } else {
        (offset + align - 1) & !(align - 1)
    }
}

pub fn build_type_env(
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> TypeEnv {
    let mut env = TypeEnv::new();

    // Pass 1: Pre-register enums, bitfields, errors, structs, pools, unions, type aliases, ifaces (shells).
    // Resolve type expressions & layouts lazily/ordered via `ensure_def_resolved`.


    for i in 0..prog.defs.len() {
        let def_id = DefId(i as u32);
        ensure_def_resolved(def_id, units, prog, &mut env, interner, diags);
    }

    // Pass 2: Signatures of functions, systems, extern functions, constants, and impl blocks.
    for i in 0..prog.defs.len() {
        let def_id = DefId(i as u32);
        let def_mod = prog.defs[i].module;
        let def_decl_idx = prog.defs[i].kind_decl_idx();
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else { continue };
        let decl = &units[ui].decls[def_decl_idx];
        match &decl.kind {
            DeclKind::Fn(f) => {
                let sig = resolve_fn_sig(
                    Some(def_id),
                    def_mod,
                    decl.is_pub,
                    &decl.attrs,
                    f,
                    None,
                    units,
                    prog,
                    &mut env,
                    interner,
                    diags,
                );
                env.fns.insert(def_id, sig);
            }
            DeclKind::Sys(s) => {
                let sig = resolve_sys_sig(
                    def_id,
                    def_mod,
                    decl.is_pub,
                    &decl.attrs,
                    s,
                    units,
                    prog,
                    &mut env,
                    interner,
                    diags,
                );
                env.fns.insert(def_id, sig);
            }
            DeclKind::ExternFn(ef) => {
                let ctx = GenericCtx::empty();
                let mut params = Vec::new();
                for p in &ef.params {
                    let ty = resolve_type_expr(
                        &p.ty,
                        def_mod,
                        &ctx, units, prog,
                        &mut env,
                        interner,
                        diags,
                    );
                    params.push(ParamSig {
                        name: p.name.sym,
                        span: p.span,
                        is_self: p.is_self,
                        is_comptime: p.is_comptime,
                        modifier: p.modifier,
                        access: p.access,
                        ty,
                        gpu_binding: None,
                        is_push_const: false,
                    });
                }
                let ret = ef
                    .ret_type
                    .as_ref()
                    .map(|t| {
                        resolve_type_expr(t, def_mod, &ctx, units, prog, &mut env, interner, diags)
                    })
                    .unwrap_or(TyId::VOID);
                let effects = parse_effect_set(&decl.attrs, &[], def_mod, &ctx, units, prog, &mut env, interner);
                env.fns.insert(
                    def_id,
                    FnSig {
                        def: Some(def_id),
                        module: def_mod,
                        name: ef.name.sym,
                        is_pub: decl.is_pub,
                        is_view: false,
                        is_comptime: false,
                        is_extern: true,
                        generics: Vec::new(),
                        params,
                        ret,
                        has_gpu_attr: effects.is_gpu,
                        effects,
                    },
                );
            }
            DeclKind::Const(c) => {
                let ctx = GenericCtx::empty();
                let ty = resolve_type_expr(
                    &c.ty,
                    def_mod,
                    &ctx,
                    units,
                    prog,
                    &mut env,
                    interner,
                    diags,
                );
                let int_val = eval_const_int(&c.value, def_mod, &ctx, units, prog, &mut env, interner);
                env.consts.insert(
                    def_id,
                    ConstMeta {
                        def: def_id,
                        ty,
                        int_val,
                        ct_val: None,
                    },
                );
            }
            _ => {}
        }
    }

    // Pass 2.5: Evaluate rich comptime `const` initializers (tables, comptime blocks/calls).
    for i in 0..prog.defs.len() {
        let def_id = DefId(i as u32);
        let def_mod = prog.defs[i].module;
        let TopDefKind::Const(decl_idx) = prog.defs[i].kind else { continue };
        let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else { continue };
        let DeclKind::Const(c) = &units[ui].decls[decl_idx].kind else { continue };
        let Some(cm) = env.consts.get(&def_id).cloned() else { continue };
        let needs_vm = cm.int_val.is_none()
            || matches!(
                &c.value.kind,
                ExprKind::ComptimeBlock(_)
                    | ExprKind::ComptimeExpr(_)
                    | ExprKind::ArrayLit(_)
                    | ExprKind::ArrayRepeat { .. }
                    | ExprKind::AnonStructLit(_)
                    | ExprKind::NamedStructLit { .. }
                    | ExprKind::Builtin { .. }
                    | ExprKind::Call { .. }
            );
        if needs_vm {
            let ctx = GenericCtx::empty();
            if let Some(cv) = crate::comptime::eval_const_decl(
                &c.value,
                cm.ty,
                def_mod,
                &ctx,
                units,
                prog,
                &mut env,
                interner,
                diags,
            ) {
                let iv = cv.as_i128();
                if let Some(entry) = env.consts.get_mut(&def_id) {
                    if entry.int_val.is_none() {
                        entry.int_val = iv;
                    }
                    entry.ct_val = Some(cv);
                }
            }
        }
    }

    // Pass 3: Resolve `impl` blocks & orphan rule.
    for impl_idx in 0..prog.impls.len() {
        let impl_info = prog.impls[impl_idx].clone();
        let ui = prog.modules[impl_info.module.0 as usize]
            .unit_idx
            .unwrap();
        let decl = &units[ui].decls[impl_info.decl_idx];
        let DeclKind::Impl(im) = &decl.kind else { continue };

        let mut ctx = GenericCtx::empty();
        let impl_generics = resolve_generic_params(
            &im.generics,
            impl_info.module,
            &mut ctx,
            units,
            prog,
            &mut env,
            interner,
            diags,
        );
        let _ = impl_generics;

        let target_ty = resolve_type_expr(
            &im.target_type,
            impl_info.module,
            &ctx,
            units,
            prog,
            &mut env,
            interner,
            diags,
        );
        ctx.self_ty = Some(target_ty);

        let iface_target = im
            .iface_type
            .as_ref()
            .and_then(|iface_te| resolve_iface_target_only(iface_te, impl_info.module, prog, interner, diags));
        let iface_def = match iface_target {
            Some(IfaceTarget::User(d)) => Some(d),
            _ => None,
        };

        let target_method_key = match env.get(target_ty) {
            Ty::Struct { def, .. }
            | Ty::Pool { def, .. }
            | Ty::Enum(def)
            | Ty::Union { def, .. }
            | Ty::Bitfield(def) => Some(MethodTarget::Def(*def)),
            Ty::Builtin(bk, _) => Some(MethodTarget::Builtin(*bk)),
            _ => None,
        };

        // Check forbidden `Ord`/`Eq` impls on floating-point types (`f16`/`f32`/`f64`).
        if matches!(target_ty, TyId::F32 | TyId::F64 | TyId::F16) {
            match iface_target {
                Some(IfaceTarget::BuiltinOrd) => {
                    diags.error(
                        ErrorCode::E0501,
                        impl_info.span,
                        "floating-point types do not implement `Ord` due to `NaN` semantics",
                    );
                }
                Some(IfaceTarget::BuiltinEq) => {
                    diags.error(
                        ErrorCode::E0501,
                        impl_info.span,
                        "floating-point types do not implement `Eq` due to `NaN` semantics",
                    );
                }
                Some(IfaceTarget::User(iface_id)) => {
                    let iname = interner.get(prog.defs[iface_id.0 as usize].name);
                    if matches!(iname, "Ord" | "Eq") {
                        diags.error(
                            ErrorCode::E0501,
                            impl_info.span,
                            format!("floating-point types do not implement `{}` due to `NaN` semantics", iname),
                        );
                    }
                }
                None => {}
            }
        }

        // Orphan rule check: `impl Iface for Type` allowed only in module of Iface or module of Type.
        if let (Some(iface_id), Some(tkey)) = (iface_def, target_method_key) {
            let iface_mod = prog.defs[iface_id.0 as usize].module;
            let type_mod = match tkey {
                MethodTarget::Def(d) => Some(prog.defs[d.0 as usize].module),
                MethodTarget::Builtin(_) => None,
            };
            if impl_info.module != iface_mod && type_mod != Some(impl_info.module) {
                diags.error(
                    ErrorCode::E0501,
                    impl_info.span,
                    "orphan rule violation: `impl Interface for Type` must be in the module defining either the interface or the type",
                );
            }
        } else if matches!(iface_target, Some(IfaceTarget::BuiltinOrd | IfaceTarget::BuiltinEq)) {
            let type_owned_here = matches!(
                target_method_key,
                Some(MethodTarget::Def(d)) if prog.defs[d.0 as usize].module == impl_info.module
            );
            if !type_owned_here && !matches!(target_ty, TyId::F16 | TyId::F32 | TyId::F64) {
                diags.error(
                    ErrorCode::E0501,
                    impl_info.span,
                    "orphan rule violation: `impl` of built-in interface for foreign type",
                );
            }
        }

        let mut methods = Vec::new();
        for (sub_idx, sub_decl) in im.decls.iter().enumerate() {
            if let DeclKind::Fn(f) = &sub_decl.kind {
                let sig = resolve_fn_sig_with_parent(
                    None,
                    impl_info.module,
                    sub_decl.is_pub,
                    &sub_decl.attrs,
                    f,
                    Some(target_ty),
                    Some(&ctx),
                    &impl_generics,
                    units,
                    prog,
                    &mut env,
                    interner,
                    diags,
                );
                methods.push((DefId(u32::MAX), sub_idx, sig));
            }
        }

        if let Some(target) = target_method_key {
            env.impls.push(ImplRef {
                module: impl_info.module,
                impl_decl_idx: impl_info.decl_idx,
                iface: iface_def,
                target,
                methods,
            });
        }
    }

    env
}

trait DefKindExt {
    fn kind_decl_idx(&self) -> usize;
}

impl DefKindExt for crate::resolve::TopDef {
    fn kind_decl_idx(&self) -> usize {
        match self.kind {
            TopDefKind::Fn(i)
            | TopDefKind::Sys(i)
            | TopDefKind::Struct(i)
            | TopDefKind::Pool(i)
            | TopDefKind::Enum(i)
            | TopDefKind::Union(i)
            | TopDefKind::Bitfield(i)
            | TopDefKind::Iface(i)
            | TopDefKind::Error(i)
            | TopDefKind::TypeAlias(i)
            | TopDefKind::Const(i)
            | TopDefKind::ExternFn(i)
            | TopDefKind::ExternStruct(i)
            | TopDefKind::ExternUnion(i) => i,
        }
    }
}

fn ensure_def_resolved(
    def_id: DefId,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let def_mod = prog.defs[def_id.0 as usize].module;
    let def_decl_idx = prog.defs[def_id.0 as usize].kind_decl_idx();
    let Some(ui) = prog.modules[def_mod.0 as usize].unit_idx else { return };
    let decl = &units[ui].decls[def_decl_idx];

    match &decl.kind {
        DeclKind::Struct(s) => {
            if env.structs.contains_key(&def_id) {
                return;
            }
            if !env.visiting_type_defs.insert(def_id) {
                diags.error(
                    ErrorCode::E0230,
                    decl.span,
                    format!("recursive struct `{}` has infinite size", interner.get(s.name.sym)),
                );
                return;
            }
            resolve_struct_def(def_id, def_mod, decl, s, units, prog, env, interner, diags);
            env.visiting_type_defs.remove(&def_id);
        }
        DeclKind::ExternStruct(es) => {
            if env.structs.contains_key(&def_id) {
                return;
            }
            if !env.visiting_type_defs.insert(def_id) {
                return;
            }
            resolve_extern_agg_def(def_id, def_mod, decl.span, &es.fields, units, prog, env, interner, diags);
            env.visiting_type_defs.remove(&def_id);
        }
        DeclKind::Pool(p) => {
            if env.pools.contains_key(&def_id) {
                return;
            }
            if !env.visiting_type_defs.insert(def_id) {
                return;
            }
            resolve_pool_def(def_id, def_mod, p, units, prog, env, interner, diags);
            env.visiting_type_defs.remove(&def_id);
        }
        DeclKind::Enum(e) => {
            if env.enums.contains_key(&def_id) {
                return;
            }
            resolve_enum_def(def_id, def_mod, e, units, prog, env, interner, diags);
        }
        DeclKind::Union(u) => {
            if env.unions.contains_key(&def_id) {
                return;
            }
            if !env.visiting_type_defs.insert(def_id) {
                diags.error(
                    ErrorCode::E0230,
                    decl.span,
                    format!("recursive union `{}` has infinite size", interner.get(u.name.sym)),
                );
                return;
            }
            resolve_union_def(def_id, def_mod, decl.span, u, units, prog, env, interner, diags);
            env.visiting_type_defs.remove(&def_id);
        }
        DeclKind::Bitfield(b) => {
            if env.bitfields.contains_key(&def_id) {
                return;
            }
            resolve_bitfield_def(def_id, def_mod, b, units, prog, env, interner, diags);
        }
        DeclKind::Error(er) => {
            if env.errors.contains_key(&def_id) {
                return;
            }
            let mut seen = BTreeSet::new();
            let mut variants = Vec::new();
            for v in &er.variants {
                if !seen.insert(v.sym) {
                    diags.error(
                        ErrorCode::E0230,
                        v.span,
                        format!("duplicate error variant `{}`", interner.get(v.sym)),
                    );
                } else {
                    variants.push(v.sym);
                }
            }
            env.errors.insert(def_id, ErrorMeta { def: def_id, variants });
        }
        DeclKind::Iface(iface) => {
            if env.ifaces.contains_key(&def_id) {
                return;
            }
            let mut ctx = GenericCtx::empty();
            let generics = resolve_generic_params(
                &iface.generics,
                def_mod,
                &mut ctx,
                units,
                prog,
                env,
                interner,
                diags,
            );
            let self_ty = env.intern(Ty::Dyn(def_id));
            ctx.self_ty = Some(self_ty);
            let mut methods = Vec::new();
            for m in &iface.methods {
                let mut m_ctx = ctx.clone();
                let m_gens = resolve_generic_params(
                    &m.generics,
                    def_mod,
                    &mut m_ctx,
                    units,
                    prog,
                    env,
                    interner,
                    diags,
                );
                let mut params = Vec::new();
                for p in &m.params {
                    let ty = resolve_type_expr(
                        &p.ty,
                        def_mod,
                        &m_ctx,
                        units,
                        prog,
                        env,
                        interner,
                        diags,
                    );
                    params.push(ParamSig {
                        name: p.name.sym,
                        span: p.span,
                        is_self: p.is_self,
                        is_comptime: p.is_comptime,
                        modifier: p.modifier,
                        access: p.access,
                        ty,
                        gpu_binding: None,
                        is_push_const: false,
                    });
                }
                let ret = m
                    .ret_type
                    .as_ref()
                    .map(|t| resolve_type_expr(t, def_mod, &m_ctx, units, prog, env, interner, diags))
                    .unwrap_or(TyId::VOID);
                let effects = parse_effect_set(&m.attrs, &m.effects, def_mod, &m_ctx, units, prog, env, interner);
                methods.push(FnSig {
                    def: None,
                    module: def_mod,
                    name: m.name.sym,
                    is_pub: true,
                    is_view: m.is_view,
                    is_comptime: false,
                    is_extern: false,
                    generics: m_gens,
                    params,
                    ret,
                    has_gpu_attr: effects.is_gpu,
                    effects,
                });
            }
            env.ifaces.insert(
                def_id,
                IfaceMeta {
                    def: def_id,
                    module: def_mod,
                    generics,
                    methods,
                },
            );
        }
        DeclKind::TypeAlias(ta) => {
            if env.type_aliases.contains_key(&def_id) {
                return;
            }
            if !env.visiting_type_defs.insert(def_id) {
                diags.error(
                    ErrorCode::E0230,
                    decl.span,
                    format!("cyclic type alias `{}`", interner.get(ta.name.sym)),
                );
                return;
            }
            let mut ctx = GenericCtx::empty();
            let generics = resolve_generic_params(
                &ta.generics,
                def_mod,
                &mut ctx,
                units,
                prog,
                env,
                interner,
                diags,
            );
            let ty = resolve_type_expr(&ta.ty, def_mod, &ctx, units, prog, env, interner, diags);
            env.visiting_type_defs.remove(&def_id);
            env.type_aliases.insert(
                def_id,
                TypeAliasMeta {
                    def: def_id,
                    generics,
                    ty,
                },
            );
        }
        _ => {}
    }
}

fn resolve_struct_def(
    def_id: DefId,
    mod_id: ModuleId,
    decl: &ast::Decl,
    s: &ast::StructDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let mut layout_mode = StructLayoutMode::Eon;
    let mut is_packed = false;
    let mut explicit_align = None;

    for attr in &decl.attrs {
        let aname = interner.get(attr.name.sym);
        match aname {
            "layout" => {
                if let Some(arg) = attr.args.first()
                    && let ExprKind::Ident(sym) = &arg.value.kind
                {
                    match interner.get(*sym) {
                        "eon" => layout_mode = StructLayoutMode::Eon,
                        "c" => layout_mode = StructLayoutMode::C,
                        "std430" => layout_mode = StructLayoutMode::Std430,
                        other => {
                            diags.error(
                                ErrorCode::E0230,
                                arg.span,
                                format!("unknown layout mode `{}` (expected `eon`, `c`, or `std430`)", other),
                            );
                        }
                    }
                }
            }
            "packed" => {
                is_packed = true;
                layout_mode = StructLayoutMode::C;
            }
            "align" => {
                if let Some(arg) = attr.args.first()
                    && let Some(val) = eval_const_int(
                        &arg.value,
                        mod_id,
                        &GenericCtx::empty(),
                        units,
                        prog,
                        env,
                        interner,
                    )
                {
                    if val >= 1 && (val as u64).is_power_of_two() {
                        explicit_align = Some(val as u64);
                    } else {
                        diags.error(
                            ErrorCode::E0230,
                            arg.span,
                            "`@align(N)` requires a power-of-two alignment >= 1",
                        );
                    }
                }
            }
            _ => {}
        }
    }

    let mut ctx = GenericCtx::empty();
    let generics = resolve_generic_params(
        &s.generics,
        mod_id,
        &mut ctx,
        units,
        prog,
        env,
        interner,
        diags,
    );

    let mut fields = Vec::new();
    let mut seen_names = BTreeSet::new();
    let mut any_linear_field = false;

    for f in &s.fields {
        if !seen_names.insert(f.name.sym) {
            diags.error(
                ErrorCode::E0132,
                f.name.span,
                format!("duplicate field `{}` in struct", interner.get(f.name.sym)),
            );
        }
        let ty = resolve_type_expr(&f.ty, mod_id, &ctx, units, prog, env, interner, diags);
        if s.is_view {
            validate_view_struct_field_ty(env, ty, f.ty.span, diags);
        } else {
            validate_aggregate_field_ty(env, ty, f.ty.span, "struct field", diags);
        }
        if env.is_linear(ty) {
            any_linear_field = true;
        }
        fields.push(FieldMeta {
            name: f.name.sym,
            span: f.span,
            is_pub: f.is_pub,
            ty,
            has_default: f.default.is_some(),
            offset: 0,
            misaligned_in_packed: false,
        });
    }

    if any_linear_field && !s.is_linear {
        diags.error(
            ErrorCode::E0322,
            s.name.span,
            format!(
                "struct `{}` contains a `linear` field and must be declared as `linear struct`",
                interner.get(s.name.sym)
            ),
        );
    }

    let layout = compute_struct_layout(&mut fields, layout_mode, is_packed, explicit_align, env);

    env.structs.insert(
        def_id,
        StructMeta {
            def: def_id,
            module: mod_id,
            is_linear: s.is_linear || any_linear_field,
            is_view: s.is_view,
            layout_mode,
            is_packed,
            explicit_align,
            generics,
            fields,
            layout,
        },
    );
}

fn resolve_extern_agg_def(
    def_id: DefId,
    mod_id: ModuleId,
    _span: Span,
    ast_fields: &[ast::Field],
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let ctx = GenericCtx::empty();
    let mut fields = Vec::new();
    for f in ast_fields {
        let ty = resolve_type_expr(&f.ty, mod_id, &ctx, units, prog, env, interner, diags);
        validate_aggregate_field_ty(env, ty, f.ty.span, "extern struct field", diags);
        fields.push(FieldMeta {
            name: f.name.sym,
            span: f.span,
            is_pub: true,
            ty,
            has_default: f.default.is_some(),
            offset: 0,
            misaligned_in_packed: false,
        });
    }
    let layout = compute_struct_layout(&mut fields, StructLayoutMode::C, false, None, env);
    env.structs.insert(
        def_id,
        StructMeta {
            def: def_id,
            module: mod_id,
            is_linear: false,
            is_view: false,
            layout_mode: StructLayoutMode::C,
            is_packed: false,
            explicit_align: None,
            generics: Vec::new(),
            fields,
            layout,
        },
    );
}

fn validate_view_struct_field_ty(
    env: &TypeEnv,
    ty: TyId,
    span: Span,
    diags: &mut DiagBag,
) {
    if env.contains_type_type(ty) {
        diags.error(
            ErrorCode::E0210,
            span,
            "comptime `type` cannot be stored in a view struct field",
        );
    } else if env.contains_id_brand(ty) {
        diags.error(
            ErrorCode::E0332,
            span,
            "branded `Id[p]` cannot be stored in a view struct field; use `GenId[T]`",
        );
    }
}

fn validate_aggregate_field_ty(
    env: &TypeEnv,
    ty: TyId,
    span: Span,
    where_label: &str,
    diags: &mut DiagBag,
) {
    if env.contains_type_type(ty) {
        diags.error(
            ErrorCode::E0210,
            span,
            format!("comptime `type` cannot be stored in a {}", where_label),
        );
    } else if env.contains_second_class_ref(ty) {
        diags.error(
            ErrorCode::E0310,
            span,
            format!("second-class reference or closure cannot be stored in a {}", where_label),
        );
    } else if env.contains_borrowing_value(ty) {
        diags.error(
            ErrorCode::E0311,
            span,
            format!("borrowing value (`ArenaPool` / `ArenaStr`) cannot be stored in a {}", where_label),
        );
    } else if env.contains_id_brand(ty) {
        diags.error(
            ErrorCode::E0332,
            span,
            format!("branded `Id[p]` cannot be stored in a {}; use `Link[Self]` inside own pool or `GenId[T]`", where_label),
        );
    }
}

fn compute_struct_layout(
    fields: &mut [FieldMeta],
    mode: StructLayoutMode,
    is_packed: bool,
    explicit_align: Option<u64>,
    env: &TypeEnv,
) -> Layout {
    let mut order: Vec<usize> = (0..fields.len()).collect();
    if mode == StructLayoutMode::Eon && !is_packed {
        order.sort_by(|&a, &b| {
            let la = env.layout(fields[a].ty);
            let lb = env.layout(fields[b].ty);
            lb.align.cmp(&la.align).then_with(|| a.cmp(&b))
        });
    }

    let mut offset = 0u64;
    let mut max_align = 1u64;
    let mut has_niche = false;

    for idx in order {
        let fl = env.layout(fields[idx].ty);
        let field_align = if is_packed {
            1
        } else if mode == StructLayoutMode::Std430 {
            match env.get(fields[idx].ty) {
                Ty::Vec { elem, len } if *len == 3 => env.layout(*elem).size * 4,
                _ => fl.align,
            }
        } else {
            fl.align
        };

        offset = align_up(offset, field_align);
        fields[idx].offset = offset;
        fields[idx].misaligned_in_packed = is_packed && fl.align > 1;
        offset += fl.size;
        max_align = max_align.max(field_align);
        if fl.has_niche {
            has_niche = true;
        }
    }

    let final_align = if is_packed {
        explicit_align.unwrap_or(1)
    } else {
        explicit_align.map_or(max_align, |a| a.max(max_align))
    };
    let final_size = align_up(offset, final_align);

    Layout {
        size: final_size,
        align: final_align,
        has_niche,
    }
}

fn resolve_pool_def(
    def_id: DefId,
    mod_id: ModuleId,
    p: &ast::PoolDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let mut ctx = GenericCtx::empty();
    let generics = resolve_generic_params(
        &p.generics,
        mod_id,
        &mut ctx,
        units,
        prog,
        env,
        interner,
        diags,
    );
    ctx.self_ty = Some(env.intern(Ty::Pool {
        def: def_id,
        args: Vec::new(),
    }));

    let mut columns = Vec::new();
    let mut seen = BTreeSet::new();
    for f in &p.fields {
        if !seen.insert(f.name.sym) {
            diags.error(
                ErrorCode::E0132,
                f.name.span,
                format!("duplicate column `{}` in pool", interner.get(f.name.sym)),
            );
        }
        let ty = resolve_type_expr(&f.ty, mod_id, &ctx, units, prog, env, interner, diags);
        validate_aggregate_field_ty(env, ty, f.ty.span, "pool column", diags);
        let elem_layout = env.layout(ty);
        columns.push(PoolColumnMeta {
            name: f.name.sym,
            span: f.span,
            is_pub: f.is_pub,
            ty,
            elem_layout,
            col_align: 64, // SoA cache-line alignment (spec 02 §5.1)
        });
    }

    env.pools.insert(
        def_id,
        PoolMeta {
            def: def_id,
            module: mod_id,
            generics,
            columns,
        },
    );
}

fn resolve_enum_def(
    def_id: DefId,
    mod_id: ModuleId,
    e: &ast::EnumDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let repr_ty = if let Some(rep) = &e.repr_int {
        let name = interner.get(rep.sym);
        match prog.prelude.get(&rep.sym).copied() {
            Some(BuiltinKind::Prim(p @ (PrimKind::UInt(_) | PrimKind::IInt(_) | PrimKind::USize | PrimKind::ISize))) => {
                prim_to_ty_id_in_env(p, env)
            }
            _ => {
                diags.error(
                    ErrorCode::E0230,
                    rep.span,
                    format!("invalid enum representation type `{}`", name),
                );
                TyId::U8
            }
        }
    } else if e.variants.len() <= 256 {
        TyId::U8
    } else if e.variants.len() <= 65536 {
        TyId::U16
    } else {
        TyId::U32
    };

    let (min_v, max_v) = int_bounds(env.get(repr_ty)).unwrap_or((0, 255));
    let mut next_val: i128 = 0;
    let mut used_vals = BTreeSet::new();
    let mut seen_names = BTreeSet::new();
    let mut variants = Vec::new();
    let ctx = GenericCtx::empty();

    for v in &e.variants {
        if !seen_names.insert(v.name.sym) {
            diags.error(
                ErrorCode::E0132,
                v.name.span,
                format!("duplicate enum variant `{}`", interner.get(v.name.sym)),
            );
        }
        let val = if let Some(val_expr) = &v.value {
            match eval_const_int(val_expr, mod_id, &ctx, units, prog, env, interner) {
                Some(n) => {
                    next_val = n + 1;
                    n
                }
                None => {
                    diags.error(
                        ErrorCode::E0230,
                        val_expr.span,
                        "enum discriminant must be a constant integer",
                    );
                    let n = next_val;
                    next_val += 1;
                    n
                }
            }
        } else {
            let n = next_val;
            next_val += 1;
            n
        };

        if val < min_v || val > max_v {
            diags.error(
                ErrorCode::E0201,
                v.span,
                format!("enum discriminant `{}` out of range for `{}`", val, env.format_ty(repr_ty, interner)),
            );
        }
        if !used_vals.insert(val) {
            diags.error(
                ErrorCode::E0230,
                v.span,
                format!("duplicate enum discriminant value `{}`", val),
            );
        }
        variants.push((v.name.sym, val));
    }

    let capacity = (max_v - min_v).saturating_add(1);
    let has_niche = (variants.len() as i128) < capacity;

    env.enums.insert(
        def_id,
        EnumMeta {
            def: def_id,
            repr_ty,
            variants,
            has_niche,
        },
    );
}

fn resolve_union_def(
    def_id: DefId,
    mod_id: ModuleId,
    _span: Span,
    u: &ast::UnionDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let mut ctx = GenericCtx::empty();
    let generics = resolve_generic_params(
        &u.generics,
        mod_id,
        &mut ctx,
        units,
        prog,
        env,
        interner,
        diags,
    );

    let mut variants = Vec::new();
    let mut seen = BTreeSet::new();
    let mut max_payload_size = 0u64;
    let mut max_payload_align = 1u64;
    let mut any_linear = false;

    for v in &u.variants {
        if !seen.insert(v.name.sym) {
            diags.error(
                ErrorCode::E0132,
                v.name.span,
                format!("duplicate union variant `{}`", interner.get(v.name.sym)),
            );
        }
        let payload = v.ty.as_ref().map(|t| {
            let ty = resolve_type_expr(t, mod_id, &ctx, units, prog, env, interner, diags);
            validate_aggregate_field_ty(env, ty, t.span, "union variant", diags);
            if env.is_linear(ty) {
                any_linear = true;
            }
            let l = env.layout(ty);
            max_payload_size = max_payload_size.max(l.size);
            max_payload_align = max_payload_align.max(l.align);
            ty
        });
        variants.push(UnionVariantMeta {
            name: v.name.sym,
            span: v.span,
            payload,
        });
    }

    if any_linear && !u.is_linear {
        diags.error(
            ErrorCode::E0322,
            u.name.span,
            format!(
                "union `{}` contains a `linear` variant and must be declared as `linear union`",
                interner.get(u.name.sym)
            ),
        );
    }

    let tag_bytes = if variants.len() <= 256 { 1 } else { 2 };
    let align = max_payload_align.max(tag_bytes);
    let size = align_up(max_payload_size + tag_bytes, align);
    let has_niche = variants.len() < 255;

    env.unions.insert(
        def_id,
        UnionMeta {
            def: def_id,
            is_linear: u.is_linear || any_linear,
            generics,
            variants,
            layout: Layout {
                size,
                align,
                has_niche,
            },
        },
    );
}

fn resolve_bitfield_def(
    def_id: DefId,
    mod_id: ModuleId,
    b: &ast::BitfieldDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) {
    let base_ty = match prog.prelude.get(&b.base_int.sym).copied() {
        Some(BuiltinKind::Prim(p @ PrimKind::UInt(8 | 16 | 32 | 64 | 128))) => {
            prim_to_ty_id_in_env(p, env)
        }
        _ => {
            diags.error(
                ErrorCode::E0220,
                b.base_int.span,
                format!(
                    "bitfield backing type `{}` must be an unsigned integer (`u8`, `u16`, `u32`, `u64`, `u128`)",
                    interner.get(b.base_int.sym)
                ),
            );
            TyId::U64
        }
    };
    let base_bits = match env.get(base_ty) {
        Ty::Int { bits, .. } => *bits,
        _ => 64,
    };

    let ctx = GenericCtx::empty();
    let mut occupied_mask: u128 = 0;
    let mut next_bit: u8 = 0;
    let mut members = Vec::new();
    let mut seen = BTreeSet::new();

    for m in &b.members {
        if !seen.insert(m.name.sym) {
            diags.error(
                ErrorCode::E0132,
                m.name.span,
                format!("duplicate bitfield member `{}`", interner.get(m.name.sym)),
            );
        }
        let (m_ty, m_bits) = match prog.prelude.get(&m.ty.sym).copied() {
            Some(BuiltinKind::Prim(p @ PrimKind::UInt(bits))) => (prim_to_ty_id_in_env(p, env), bits),
            _ => {
                diags.error(
                    ErrorCode::E0220,
                    m.ty.span,
                    format!(
                        "bitfield member type `{}` must be an unsigned integer `u1..u128`",
                        interner.get(m.ty.sym)
                    ),
                );
                (TyId::U8, 8)
            }
        };

        let (lo, hi) = if let Some((lo_e, hi_e)) = &m.bits {
            let lo_v = eval_const_int(lo_e, mod_id, &ctx, units, prog, env, interner);
            let hi_v = eval_const_int(hi_e, mod_id, &ctx, units, prog, env, interner);
            match (lo_v, hi_v) {
                (Some(l), Some(h)) => {
                    if l < 0 || h <= l || h > base_bits as i128 {
                        diags.error(
                            ErrorCode::E0220,
                            m.span,
                            format!(
                                "bit range `{}..{}` is out of bounds for {}-bit backing type",
                                l, h, base_bits
                            ),
                        );
                        (0, 0)
                    } else if (h - l) != m_bits as i128 {
                        diags.error(
                            ErrorCode::E0220,
                            m.span,
                            format!(
                                "bit range `{}..{}` has width {} bits, expected {} bits for `{}`",
                                l,
                                h,
                                h - l,
                                m_bits,
                                interner.get(m.ty.sym)
                            ),
                        );
                        (l as u8, h as u8)
                    } else {
                        (l as u8, h as u8)
                    }
                }
                _ => {
                    diags.error(
                        ErrorCode::E0220,
                        m.span,
                        "`@bits(lo..hi)` bounds must be constant integers",
                    );
                    (0, 0)
                }
            }
        } else {
            let lo = next_bit;
            let hi = lo.saturating_add(m_bits);
            if hi > base_bits {
                diags.error(
                    ErrorCode::E0220,
                    m.span,
                    format!("bitfield member overflows {}-bit backing type", base_bits),
                );
            }
            next_bit = hi;
            (lo, hi)
        };

        if hi > lo && hi <= 128 {
            let width = hi - lo;
            let mask = if width == 128 {
                u128::MAX
            } else {
                ((1u128 << width) - 1) << lo
            };
            if (occupied_mask & mask) != 0 {
                diags.error(
                    ErrorCode::E0220,
                    m.span,
                    format!(
                        "bit range `@bits({}..{})` overlaps with another bitfield member",
                        lo, hi
                    ),
                );
            }
            occupied_mask |= mask;
        }

        members.push(BitfieldMemberMeta {
            name: m.name.sym,
            span: m.span,
            ty: m_ty,
            lo,
            hi,
        });
    }

    env.bitfields.insert(
        def_id,
        BitfieldMeta {
            def: def_id,
            base_ty,
            base_bits,
            members,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn resolve_fn_sig(
    def_id: Option<DefId>,
    mod_id: ModuleId,
    is_pub: bool,
    decl_attrs: &[Attr],
    f: &ast::FnDecl,
    self_ty: Option<TyId>,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> FnSig {
    resolve_fn_sig_with_parent(
        def_id,
        mod_id,
        is_pub,
        decl_attrs,
        f,
        self_ty,
        None,
        &[],
        units,
        prog,
        env,
        interner,
        diags,
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_fn_sig_with_parent(
    def_id: Option<DefId>,
    mod_id: ModuleId,
    is_pub: bool,
    decl_attrs: &[Attr],
    f: &ast::FnDecl,
    self_ty: Option<TyId>,
    parent_ctx: Option<&GenericCtx>,
    parent_generics: &[GenericParamMeta],
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> FnSig {
    let mut ctx = parent_ctx.cloned().unwrap_or_else(GenericCtx::empty);
    ctx.self_ty = self_ty;
    let mut generics = parent_generics.to_vec();
    let fn_generics = resolve_generic_params(
        &f.generics,
        mod_id,
        &mut ctx,
        units,
        prog,
        env,
        interner,
        diags,
    );
    generics.extend(fn_generics);
    let params = resolve_params(&f.params, mod_id, &ctx, units, prog, env, interner, diags);
    let ret = f
        .ret_type
        .as_ref()
        .map(|t| resolve_type_expr(t, mod_id, &ctx, units, prog, env, interner, diags))
        .unwrap_or(TyId::VOID);
    if !f.is_view {
        let ret_span = f.ret_type.as_ref().map_or(f.name.span, |t| t.span);
        if env.contains_second_class_ref(ret) {
            diags.error(
                ErrorCode::E0310,
                ret_span,
                "cannot return second-class reference or closure from a non-`view` function; mark with `view fn`",
            );
        } else if env.contains_borrowing_value(ret) {
            diags.error(
                ErrorCode::E0311,
                ret_span,
                "cannot return borrowing value (`ArenaPool` / `ArenaStr`) from a non-`view` function",
            );
        }
    }
    let effects = parse_effect_set(decl_attrs, &f.effects, mod_id, &ctx, units, prog, env, interner);
    if effects.is_export_c {
        let ret_span = f.ret_type.as_ref().map_or(f.name.span, |t| t.span);
        if matches!(
            env.get(ret),
            Ty::ErrorUnion { .. }
                | Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. }
                | Ty::Dyn(_)
        ) || env.is_linear(ret)
        {
            diags.error(
                ErrorCode::E0810,
                ret_span,
                format!(
                    "`@extern(\"C\")` function cannot return Eon-specific type `{}`",
                    env.format_ty(ret, interner)
                ),
            );
        }
    }
    FnSig {
        def: def_id,
        module: mod_id,
        name: f.name.sym,
        is_pub,
        is_view: f.is_view,
        is_comptime: f.is_comptime,
        is_extern: false,
        generics,
        params,
        ret,
        has_gpu_attr: effects.is_gpu,
        effects,
    }
}

fn parse_effect_set(
    decl_attrs: &[Attr],
    fn_effects: &[Attr],
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
) -> EffectSet {
    let mut es = EffectSet::default();
    for a in decl_attrs.iter().chain(fn_effects.iter()) {
        let name = interner.get(a.name.sym).to_owned();
        match name.as_str() {
            "pure" => {
                es.is_pure = true;
                es.is_noalloc = true;
                es.is_noblock = true;
                if es.effect_span.is_none() {
                    es.effect_span = Some(a.span);
                }
            }
            "noalloc" => {
                es.is_noalloc = true;
                if es.effect_span.is_none() {
                    es.effect_span = Some(a.span);
                }
            }
            "noblock" => {
                es.is_noblock = true;
                if es.effect_span.is_none() {
                    es.effect_span = Some(a.span);
                }
            }
            "interrupt" => {
                es.is_interrupt = true;
                es.is_noalloc = true;
                es.is_noblock = true;
                if es.effect_span.is_none() {
                    es.effect_span = Some(a.span);
                }
            }
            "naked" => {
                es.is_naked = true;
            }
            "extern" => {
                es.is_export_c = true;
            }
            "gpu" => {
                es.is_gpu = true;
                es.is_noalloc = true;
                es.is_noblock = true;
                if es.effect_span.is_none() {
                    es.effect_span = Some(a.span);
                }
                let mut ls = (1u32, 1u32, 1u32);
                for arg in &a.args {
                    if let Some(an) = arg.name
                        && interner.get(an.sym) == "local_size"
                        && let ast::ExprKind::Tuple(ts) = &arg.value.kind
                    {
                        let x = ts.first().and_then(|e| eval_const_int(e, mod_id, ctx, units, prog, env, interner)).unwrap_or(1).max(1) as u32;
                        let y = ts.get(1).and_then(|e| eval_const_int(e, mod_id, ctx, units, prog, env, interner)).unwrap_or(1).max(1) as u32;
                        let z = ts.get(2).and_then(|e| eval_const_int(e, mod_id, ctx, units, prog, env, interner)).unwrap_or(1).max(1) as u32;
                        ls = (x, y, z);
                    }
                }
                es.gpu_local_size = Some(ls);
            }
            "recursive" => {
                es.is_recursive = true;
            }
            "hot" | "specialize" => {
                es.is_hot = true;
            }
            "inline" => {
                es.is_inline = true;
            }
            "save_simd" => {
                es.save_simd = true;
            }
            "stack" => {
                es.stack_span = Some(a.span);
                if let Some(arg) = a.args.first()
                    && let Some(n) = eval_const_int(&arg.value, mod_id, ctx, units, prog, env, interner)
                    && n >= 0
                {
                    es.stack_limit = Some(n as u64);
                }
            }
            _ => {}
        }
    }
    es
}

fn resolve_sys_sig(
    def_id: DefId,
    mod_id: ModuleId,
    is_pub: bool,
    decl_attrs: &[Attr],
    s: &ast::SysDecl,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> FnSig {
    let mut ctx = GenericCtx::empty();
    let generics = resolve_generic_params(
        &s.generics,
        mod_id,
        &mut ctx,
        units,
        prog,
        env,
        interner,
        diags,
    );
    let mut params = Vec::new();
    let mut accesses = Vec::new();
    for (pidx, p) in s.params.iter().enumerate() {
        let ty = resolve_type_expr(&p.ty, mod_id, &ctx, units, prog, env, interner, diags);
        if let Some(mode) = p.access {
            let pool_def = match env.get(ty) {
                Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::SlotPool, gargs) => {
                    gargs.first().and_then(|g| match g {
                        GenericVal::Ty(pt) => match env.get(*pt) {
                            Ty::Pool { def, .. } => Some(*def),
                            _ => None,
                        },
                        _ => None,
                    })
                }
                Ty::Pool { def, .. } => Some(*def),
                _ => None,
            };
            if let Some(pdef) = pool_def {
                let cols: Vec<Symbol> = if p.columns.is_empty() {
                    env.pools
                        .get(&pdef)
                        .map(|m| m.columns.iter().map(|c| c.name).collect())
                        .unwrap_or_default()
                } else {
                    p.columns.iter().map(|c| c.sym).collect()
                };
                accesses.push(SysColAccess {
                    param_idx: pidx,
                    mode,
                    pool_def: pdef,
                    columns: cols,
                    span: p.span,
                });
            }
        }
        params.push(ParamSig {
            name: p.name.sym,
            span: p.span,
            is_self: p.is_self,
            is_comptime: p.is_comptime,
            modifier: p.modifier,
            access: p.access,
            ty,
            gpu_binding: None,
            is_push_const: false,
        });
    }
    env.sys_access.insert(def_id, accesses);
    let ret = s
        .ret_type
        .as_ref()
        .map(|t| resolve_type_expr(t, mod_id, &ctx, units, prog, env, interner, diags))
        .unwrap_or(TyId::VOID);
    let effects = parse_effect_set(decl_attrs, &s.effects, mod_id, &ctx, units, prog, env, interner);
    FnSig {
        def: Some(def_id),
        module: mod_id,
        name: s.name.sym,
        is_pub,
        is_view: false,
        is_comptime: false,
        is_extern: false,
        generics,
        params,
        ret,
        has_gpu_attr: effects.is_gpu,
        effects,
    }
}

fn resolve_params(
    ast_params: &[Param],
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Vec<ParamSig> {
    let mut params = Vec::new();
    for p in ast_params {
        let ty = resolve_type_expr(&p.ty, mod_id, ctx, units, prog, env, interner, diags);
        if !p.is_comptime && env.contains_type_type(ty) {
            diags.error(
                ErrorCode::E0210,
                p.ty.span,
                "comptime `type` parameter must be marked `comptime`",
            );
        }
        let mut gpu_binding = None;
        let mut is_push_const = false;
        for a in &p.attrs {
            let an = interner.get(a.name.sym).to_owned();
            if an == "binding" {
                let s = a.args.first().and_then(|x| eval_const_int(&x.value, mod_id, ctx, units, prog, env, interner)).unwrap_or(0) as u32;
                let b = a.args.get(1).and_then(|x| eval_const_int(&x.value, mod_id, ctx, units, prog, env, interner)).unwrap_or(0) as u32;
                gpu_binding = Some((s, b));
            } else if an == "push_const" {
                is_push_const = true;
            }
        }
        params.push(ParamSig {
            name: p.name.sym,
            span: p.span,
            is_self: p.is_self,
            is_comptime: p.is_comptime,
            modifier: p.modifier,
            access: p.access,
            ty,
            gpu_binding,
            is_push_const,
        });
    }
    params
}

pub fn resolve_generic_params(
    ast_gens: &[GenericParam],
    mod_id: ModuleId,
    ctx: &mut GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Vec<GenericParamMeta> {
    let mut out = Vec::new();
    for g in ast_gens {
        let idx = ctx.params.len() as u32;
        ctx.params.push((g.name.sym, idx, g.is_comptime));
        let comptime_ty = g
            .comptime_ty
            .as_ref()
            .map(|t| resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags));
        let mut bounds = Vec::new();
        let mut builtin_ord = false;
        let mut builtin_eq = false;
        for b in &g.bounds {
            match resolve_iface_target_only(b, mod_id, prog, interner, diags) {
                Some(IfaceTarget::User(d)) => bounds.push(d),
                Some(IfaceTarget::BuiltinOrd) => builtin_ord = true,
                Some(IfaceTarget::BuiltinEq) => builtin_eq = true,
                None => {}
            }
        }
        out.push(GenericParamMeta {
            name: g.name.sym,
            is_comptime: g.is_comptime,
            comptime_ty,
            bounds,
            builtin_ord,
            builtin_eq,
        });
    }
    out
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IfaceTarget {
    User(DefId),
    BuiltinOrd,
    BuiltinEq,
}

fn resolve_iface_target_only(
    te: &TypeExpr,
    mod_id: ModuleId,
    prog: &mut ResolvedProgram,
    interner: &Interner,
    diags: &mut DiagBag,
) -> Option<IfaceTarget> {
    let TypeKind::Path(tp) = &te.kind else {
        return None;
    };
    let target = resolve_path_segments(&tp.segments, mod_id, prog, interner, diags)?;
    match target {
        TopScopeTarget::Def(d) if matches!(prog.defs[d.0 as usize].kind, TopDefKind::Iface(_)) => {
            Some(IfaceTarget::User(d))
        }
        TopScopeTarget::Builtin(BuiltinKind::OrdIface) => Some(IfaceTarget::BuiltinOrd),
        TopScopeTarget::Builtin(BuiltinKind::EqIface) => Some(IfaceTarget::BuiltinEq),
        _ => {
            diags.error(ErrorCode::E0501, te.span, "expected an interface name");
            None
        }
    }
}

pub fn resolve_type_expr(
    te: &TypeExpr,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> TyId {
    match &te.kind {
        TypeKind::Void => TyId::VOID,
        TypeKind::Never => TyId::NEVER,
        TypeKind::TypeType => TyId::TYPE,
        TypeKind::SelfType => ctx.self_ty.unwrap_or_else(|| {
            diags.error(ErrorCode::E0230, te.span, "`Self` is not valid outside an `impl`, `iface`, or `pool`");
            TyId::ERROR
        }),
        TypeKind::Inferred | TypeKind::Error => TyId::ERROR,
        TypeKind::Optional(inner) => {
            let inner_ty = resolve_type_expr(inner, mod_id, ctx, units, prog, env, interner, diags);
            env.intern(Ty::Optional(inner_ty))
        }
        TypeKind::Borrow { is_mut, target } => {
            let inner = match target {
                ast::BorrowTarget::Str => BorrowInner::Str,
                ast::BorrowTarget::Slice(t) => BorrowInner::Slice(resolve_type_expr(
                    t, mod_id, ctx, units, prog, env, interner, diags,
                )),
                ast::BorrowTarget::Single(t) => BorrowInner::Single(resolve_type_expr(
                    t, mod_id, ctx, units, prog, env, interner, diags,
                )),
            };
            env.intern(Ty::Borrow {
                is_mut: *is_mut,
                inner,
            })
        }
        TypeKind::Array { len, elem } => {
            let elem_ty = resolve_type_expr(elem, mod_id, ctx, units, prog, env, interner, diags);
            if env.contains_second_class_ref(elem_ty) {
                diags.error(
                    ErrorCode::E0310,
                    elem.span,
                    "second-class reference or closure cannot be stored inside an array `[N]T`",
                );
            } else if env.contains_borrowing_value(elem_ty) {
                diags.error(
                    ErrorCode::E0311,
                    elem.span,
                    "borrowing value cannot be stored inside an array `[N]T`",
                );
            } else if env.contains_id_brand(elem_ty) {
                diags.error(
                    ErrorCode::E0332,
                    elem.span,
                    "branded `Id[p]` cannot be stored inside an array `[N]T`",
                );
            }
            let n = eval_const_int(len, mod_id, ctx, units, prog, env, interner);
            let len_u64 = match n {
                Some(v) if v >= 0 => v as u64,
                _ => {
                    diags.error(
                        ErrorCode::E0230,
                        len.span,
                        "fixed array length must be a non-negative constant integer",
                    );
                    0
                }
            };
            env.intern(Ty::Array {
                len: len_u64,
                elem: elem_ty,
            })
        }
        TypeKind::Tuple(elems) => {
            let ts: Vec<TyId> = elems
                .iter()
                .map(|t| resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags))
                .collect();
            env.intern(Ty::Tuple(ts))
        }
        TypeKind::FnPtr { params, ret, .. } => {
            let ps: Vec<TyId> = params
                .iter()
                .map(|t| resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags))
                .collect();
            let r = ret
                .as_ref()
                .map(|t| resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags))
                .unwrap_or(TyId::VOID);
            env.intern(Ty::FnPtr { params: ps, ret: r })
        }
        TypeKind::InlineStruct(fields) => {
            let mut fs = Vec::new();
            for f in fields {
                let ft = resolve_type_expr(&f.ty, mod_id, ctx, units, prog, env, interner, diags);
                validate_aggregate_field_ty(env, ft, f.ty.span, "inline struct field", diags);
                fs.push((f.name.sym, ft));
            }
            env.intern(Ty::AnonStruct(fs))
        }
        TypeKind::Dyn(path) => {
            if let Some(target) = resolve_path_segments(&path.segments, mod_id, prog, interner, diags) {
                match target {
                    TopScopeTarget::Def(d) if matches!(prog.defs[d.0 as usize].kind, TopDefKind::Iface(_)) => {
                        ensure_def_resolved(d, units, prog, env, interner, diags);
                        let dyn_ty = env.intern(Ty::Dyn(d));
                        if let Some(im) = env.ifaces.get(&d) {
                            for m in &im.methods {
                                if !m.generics.is_empty() {
                                    diags.error(
                                        ErrorCode::E0512,
                                        te.span,
                                        format!(
                                            "interface `{}` is not dyn-compatible: method `{}` has generic parameters",
                                            interner.get(prog.defs[d.0 as usize].name),
                                            interner.get(m.name)
                                        ),
                                    );
                                } else if m.ret == dyn_ty {
                                    diags.error(
                                        ErrorCode::E0512,
                                        te.span,
                                        format!(
                                            "interface `{}` is not dyn-compatible: method `{}` returns `Self` by value",
                                            interner.get(prog.defs[d.0 as usize].name),
                                            interner.get(m.name)
                                        ),
                                    );
                                } else if !m.params.first().is_some_and(|p| p.is_self && matches!(env.get(p.ty), Ty::Borrow { .. })) {
                                    diags.error(
                                        ErrorCode::E0512,
                                        te.span,
                                        format!(
                                            "interface `{}` is not dyn-compatible: method `{}` must take `*Self` or `*mut Self`",
                                            interner.get(prog.defs[d.0 as usize].name),
                                            interner.get(m.name)
                                        ),
                                    );
                                }
                            }
                        }
                        dyn_ty
                    }
                    _ => {
                        diags.error(ErrorCode::E0501, te.span, "`dyn` requires an interface name");
                        TyId::ERROR
                    }
                }
            } else {
                TyId::ERROR
            }
        }
        TypeKind::ErrorUnion { ok, errors } => {
            let ok_ty = ok
                .as_ref()
                .map(|t| resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags))
                .unwrap_or(TyId::VOID);
            let mut err_srcs = Vec::new();
            for e in errors {
                let ety = resolve_type_expr(e, mod_id, ctx, units, prog, env, interner, diags);
                match env.get(ety).clone() {
                    Ty::ErrorSet(src) => err_srcs.push(src),
                    Ty::Error => {}
                    _ => diags.error(
                        ErrorCode::E0230,
                        e.span,
                        "right side of `!` must be an `error` set",
                    ),
                }
            }
            env.intern(Ty::ErrorUnion {
                ok: ok_ty,
                errors: err_srcs,
            })
        }
        TypeKind::Path(tp) => resolve_type_path(tp, te.span, mod_id, ctx, units, prog, env, interner, diags),
    }
}

fn resolve_type_path(
    tp: &TypePath,
    span: Span,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> TyId {
    if tp.segments.len() == 1 {
        let seg = tp.segments[0];
        if let Some((_, idx, is_ct)) = ctx.params.iter().find(|&&(n, _, _)| n == seg.sym) {
            return env.intern(Ty::GenericParam {
                idx: *idx,
                name: seg.sym,
                is_comptime: *is_ct,
            });
        }
    }

    let Some(target) = resolve_path_segments(&tp.segments, mod_id, prog, interner, diags) else {
        return TyId::ERROR;
    };

    match target {
        TopScopeTarget::Builtin(bk) => resolve_builtin_type(bk, &tp.generics, span, mod_id, ctx, units, prog, env, interner, diags),
        TopScopeTarget::Def(d) => {
            ensure_def_resolved(d, units, prog, env, interner, diags);
            let args = resolve_generic_args(&tp.generics, mod_id, ctx, units, prog, env, interner, diags);
            let def_kind = prog.defs[d.0 as usize].kind.clone();
            let def_name = prog.defs[d.0 as usize].name;
            match def_kind {
                TopDefKind::Struct(_) | TopDefKind::ExternStruct(_) => env.intern(Ty::Struct { def: d, args }),
                TopDefKind::Pool(_) => env.intern(Ty::Pool { def: d, args }),
                TopDefKind::Enum(_) => env.intern(Ty::Enum(d)),
                TopDefKind::Union(_) | TopDefKind::ExternUnion(_) => env.intern(Ty::Union { def: d, args }),
                TopDefKind::Bitfield(_) => env.intern(Ty::Bitfield(d)),
                TopDefKind::Error(_) => env.intern(Ty::ErrorSet(ErrorSource::User(d))),
                TopDefKind::TypeAlias(_) => env
                    .type_aliases
                    .get(&d)
                    .map(|ta| ta.ty)
                    .unwrap_or(TyId::ERROR),
                TopDefKind::Iface(_) => {
                    diags.error(
                        ErrorCode::E0230,
                        span,
                        "interface cannot be used directly as a value type; use `dyn Interface` or a generic bound",
                    );
                    TyId::ERROR
                }
                _ => {
                    diags.error(
                        ErrorCode::E0230,
                        span,
                        format!("`{}` is not a type", interner.get(def_name)),
                    );
                    TyId::ERROR
                }
            }
        }
        TopScopeTarget::Module(_) => {
            diags.error(ErrorCode::E0230, span, "expected type, found module");
            TyId::ERROR
        }
    }
}

fn resolve_builtin_type(
    bk: BuiltinKind,
    generics: &[GenericArg],
    span: Span,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> TyId {
    match bk {
        BuiltinKind::Prim(p) => prim_to_ty_id_in_env(p, env),
        BuiltinKind::SimdAlias(p, n) => {
            let elem = prim_to_ty_id_in_env(p, env);
            env.intern(Ty::Vec { elem, len: n })
        }
        BuiltinKind::VecType => {
            if generics.len() != 2 {
                diags.error(
                    ErrorCode::E0230,
                    span,
                    "`vec[T, N]` expects 2 arguments: scalar element type and lane count",
                );
                return TyId::ERROR;
            }
            let elem = match &generics[0].kind {
                GenericArgKind::Type(t) => resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, diags),
                GenericArgKind::TypeOrConst(segs) => {
                    let tp = TypePath {
                        segments: segs.clone(),
                        generics: Vec::new(),
                    };
                    resolve_type_path(&tp, generics[0].span, mod_id, ctx, units, prog, env, interner, diags)
                }
                GenericArgKind::Expr(_) => {
                    diags.error(ErrorCode::E0230, generics[0].span, "expected element type for `vec`");
                    TyId::ERROR
                }
            };
            if elem != TyId::ERROR && !env.is_numeric(elem) && elem != TyId::BOOL {
                diags.error(
                    ErrorCode::E0230,
                    generics[0].span,
                    "`vec` element type must be a scalar number or `bool`",
                );
            }
            let len_v = match &generics[1].kind {
                GenericArgKind::Expr(e) => eval_const_int(e, mod_id, ctx, units, prog, env, interner),
                GenericArgKind::TypeOrConst(segs) if segs.len() == 1 => {
                    ctx.comptime_vals.get(&segs[0].sym).copied()
                }
                _ => None,
            };
            let len = match len_v {
                Some(n) if (2..=64).contains(&n) && (n as u64).is_power_of_two() => n as u8,
                _ => {
                    diags.error(
                        ErrorCode::E0230,
                        generics[1].span,
                        "`vec[T, N]` lane count `N` must be a power of 2 in `2..=64`",
                    );
                    4
                }
            };
            env.intern(Ty::Vec { elem, len })
        }
        BuiltinKind::RawPtr { is_mut } => {
            if prog.modules[mod_id.0 as usize].profile == FileProfile::Safe {
                diags.error(
                    ErrorCode::E0111,
                    span,
                    "ZM-11: raw pointers (`rawptr`/`rawptr_mut`) are forbidden outside `.hal.eon` modules",
                );
            }
            let args = resolve_generic_args(generics, mod_id, ctx, units, prog, env, interner, diags);
            let inner = match args.first() {
                Some(GenericVal::Ty(t)) => *t,
                _ => {
                    diags.error(ErrorCode::E0230, span, "expected element type in `rawptr[T]`");
                    TyId::ERROR
                }
            };
            env.intern(Ty::RawPtr { is_mut, inner })
        }
        BuiltinKind::ArithError => TyId::ARITH_ERROR,
        BuiltinKind::BoundsError => TyId::BOUNDS_ERROR,
        BuiltinKind::Utf8Error => TyId::UTF8_ERROR,
        BuiltinKind::OutOfMemory => TyId::OUT_OF_MEMORY,
        BuiltinKind::MemError => TyId::MEM_ERROR,
        BuiltinKind::StaleId => TyId::STALE_ID,
        BuiltinKind::IoError => TyId::IO_ERROR,
        BuiltinKind::Ordering => TyId::ORDERING,
        BuiltinKind::IdBrand => {
            if generics.len() != 1 {
                diags.error(ErrorCode::E0230, span, "`Id[pool]` expects 1 pool brand argument");
                return TyId::ERROR;
            }
            let brand_sym = match &generics[0].kind {
                GenericArgKind::TypeOrConst(segs) if segs.len() == 1 => segs[0].sym,
                GenericArgKind::Expr(e) if matches!(e.kind, ExprKind::Ident(_)) => {
                    let ExprKind::Ident(s) = e.kind else { unreachable!() };
                    s
                }
                GenericArgKind::Type(t) if matches!(&t.kind, TypeKind::Path(tp) if tp.segments.len() == 1) => {
                    let TypeKind::Path(tp) = &t.kind else { unreachable!() };
                    tp.segments[0].sym
                }
                _ => {
                    diags.error(
                        ErrorCode::E0330,
                        generics[0].span,
                        "`Id[p]` brand must name a pool variable or parameter",
                    );
                    return TyId::ERROR;
                }
            };
            env.intern(Ty::Builtin(BuiltinKind::IdBrand, vec![GenericVal::Variant(brand_sym)]))
        }
        BuiltinKind::Buf => {
            let args = resolve_generic_args(generics, mod_id, ctx, units, prog, env, interner, diags);
            if let Some(GenericVal::Ty(elem)) = args.first().copied() {
                if env.is_linear(elem) {
                    diags.error(
                        ErrorCode::E0326,
                        span,
                        "linear element type cannot be stored in `Buf[T]`; use `LinearBuf[T]` with `.drain()`",
                    );
                } else if env.contains_second_class_ref(elem) {
                    diags.error(
                        ErrorCode::E0310,
                        span,
                        "second-class reference or closure cannot be stored in `Buf[T]`",
                    );
                } else if env.contains_borrowing_value(elem) {
                    diags.error(
                        ErrorCode::E0311,
                        span,
                        "borrowing value cannot be stored in `Buf[T]`",
                    );
                }
            }
            env.intern(Ty::Builtin(BuiltinKind::Buf, args))
        }
        BuiltinKind::Atomic => {
            let args = resolve_generic_args(generics, mod_id, ctx, units, prog, env, interner, diags);
            if let Some(GenericVal::Ty(elem)) = args.first().copied() {
                let lock_free = matches!(
                    env.get(elem),
                    Ty::Bool
                        | Ty::PhysAddr
                        | Ty::VirtAddr
                        | Ty::GenericParam { .. }
                        | Ty::Error
                        | Ty::Int { bits: 8 | 16 | 32 | 64, .. }
                );
                if !lock_free {
                    diags.error(
                        ErrorCode::E0610,
                        span,
                        format!(
                            "`Atomic[{}]` is not hardware lock-free (allowed: bool, u8..u64, usize, i8..i64, isize, PhysAddr, VirtAddr)",
                            env.format_ty(elem, interner)
                        ),
                    );
                }
            }
            env.intern(Ty::Builtin(BuiltinKind::Atomic, args))
        }
        _ => {
            let args = resolve_generic_args(generics, mod_id, ctx, units, prog, env, interner, diags);
            env.intern(Ty::Builtin(bk, args))
        }
    }
}

fn resolve_generic_args(
    args: &[GenericArg],
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Vec<GenericVal> {
    let mut out = Vec::new();
    for a in args {
        match &a.kind {
            GenericArgKind::Type(t) => {
                out.push(GenericVal::Ty(resolve_type_expr(
                    t, mod_id, ctx, units, prog, env, interner, diags,
                )));
            }
            GenericArgKind::TypeOrConst(segs) => {
                if segs.len() == 1
                    && let Some(&val) = ctx.comptime_vals.get(&segs[0].sym)
                {
                    out.push(GenericVal::Int(val));
                    continue;
                }
                let tp = TypePath {
                    segments: segs.clone(),
                    generics: Vec::new(),
                };
                out.push(GenericVal::Ty(resolve_type_path(
                    &tp, a.span, mod_id, ctx, units, prog, env, interner, diags,
                )));
            }
            GenericArgKind::Expr(e) => {
                if let ExprKind::EnumLit(v) = &e.kind {
                    out.push(GenericVal::Variant(v.sym));
                } else if let Some(n) = eval_const_int(e, mod_id, ctx, units, prog, env, interner) {
                    out.push(GenericVal::Int(n));
                } else {
                    out.push(GenericVal::Int(0));
                }
            }
        }
    }
    out
}

pub fn resolve_path_segments(
    segments: &[ast::IdentSpan],
    mod_id: ModuleId,
    prog: &mut ResolvedProgram,
    interner: &Interner,
    diags: &mut DiagBag,
) -> Option<TopScopeTarget> {
    if segments.is_empty() {
        return None;
    }
    let first = segments[0];
    let Some(mut cur) = prog.lookup_in_module(mod_id, first.sym) else {
        diags.error(
            ErrorCode::E0132,
            first.span,
            format!("unresolved name `{}`", interner.get(first.sym)),
        );
        return None;
    };
    prog.mark_module_binding_used(mod_id, first.sym);

    for &seg in &segments[1..] {
        match cur {
            TopScopeTarget::Module(target_mod) => {
                let exp = prog.modules[target_mod.0 as usize].exports.get(&seg.sym).cloned();
                let in_scope = prog.modules[target_mod.0 as usize].scope.contains_key(&seg.sym);
                let m_name = prog.modules[target_mod.0 as usize].full_name.clone();
                if let Some(exp) = exp {
                    let next = exp.target;
                    if let TopScopeTarget::Def(d) = next {
                        prog.mark_def_used(d);
                    }
                    cur = next;
                } else if in_scope {
                    diags.error(
                        ErrorCode::E0133,
                        seg.span,
                        format!(
                            "symbol `{}` in module `{}` is private",
                            interner.get(seg.sym),
                            m_name
                        ),
                    );
                    return None;
                } else {
                    diags.error(
                        ErrorCode::E0132,
                        seg.span,
                        format!(
                            "symbol `{}` not found in module `{}`",
                            interner.get(seg.sym),
                            m_name
                        ),
                    );
                    return None;
                }
            }
            _ => {
                diags.error(
                    ErrorCode::E0132,
                    seg.span,
                    format!("`{}` is not a module", interner.get(seg.sym)),
                );
                return None;
            }
        }
    }
    Some(cur)
}

pub fn prim_to_ty_id(p: PrimKind) -> TyId {
    match p {
        PrimKind::UInt(8) => TyId::U8,
        PrimKind::UInt(16) => TyId::U16,
        PrimKind::UInt(32) => TyId::U32,
        PrimKind::UInt(64) => TyId::U64,
        PrimKind::UInt(128) => TyId::U128,
        PrimKind::UInt(_) => TyId::U64,
        PrimKind::USize => TyId::USIZE,
        PrimKind::IInt(8) => TyId::I8,
        PrimKind::IInt(16) => TyId::I16,
        PrimKind::IInt(32) => TyId::I32,
        PrimKind::IInt(64) => TyId::I64,
        PrimKind::IInt(128) => TyId::I128,
        PrimKind::IInt(_) => TyId::I64,
        PrimKind::ISize => TyId::ISIZE,
        PrimKind::F16 => TyId::F16,
        PrimKind::F32 => TyId::F32,
        PrimKind::F64 => TyId::F64,
        PrimKind::Bool => TyId::BOOL,
        PrimKind::Char => TyId::CHAR,
        PrimKind::Void => TyId::VOID,
        PrimKind::Never => TyId::NEVER,
        PrimKind::TypeType => TyId::TYPE,
        PrimKind::Str => TyId::STR,
        PrimKind::PhysAddr => TyId::PHYS_ADDR,
        PrimKind::VirtAddr => TyId::VIRT_ADDR,
        PrimKind::DmaAddr => TyId::DMA_ADDR,
    }
}

pub fn prim_to_ty_id_in_env(p: PrimKind, env: &mut TypeEnv) -> TyId {
    match p {
        PrimKind::UInt(bits) if !matches!(bits, 8 | 16 | 32 | 64 | 128) => env.intern(Ty::Int {
            signed: false,
            bits,
            is_ptr_sized: false,
        }),
        PrimKind::IInt(bits) if !matches!(bits, 8 | 16 | 32 | 64 | 128) => env.intern(Ty::Int {
            signed: true,
            bits,
            is_ptr_sized: false,
        }),
        other => prim_to_ty_id(other),
    }
}

pub fn int_bounds(ty: &Ty) -> Option<(i128, i128)> {
    match ty {
        Ty::Int { signed, bits, .. } => {
            if *signed {
                if *bits >= 128 {
                    Some((i128::MIN, i128::MAX))
                } else {
                    let max = (1i128 << (*bits - 1)) - 1;
                    let min = -(1i128 << (*bits - 1));
                    Some((min, max))
                }
            } else if *bits >= 127 {
                // `u128` max exceeds `i128::MAX`, handled specially where needed
                Some((0, i128::MAX))
            } else {
                Some((0, (1i128 << *bits) - 1))
            }
        }
        _ => None,
    }
}

pub fn parse_int_literal(raw: &str) -> Option<u128> {
    let (radix, body) = if let Some(rest) = raw.strip_prefix("0x") {
        (16, rest)
    } else if let Some(rest) = raw.strip_prefix("0b") {
        (2, rest)
    } else if let Some(rest) = raw.strip_prefix("0o") {
        (8, rest)
    } else {
        (10, raw)
    };
    let mut acc: u128 = 0;
    for b in body.bytes() {
        if b == b'_' {
            continue;
        }
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u128,
            b'a'..=b'f' => (b - b'a' + 10) as u128,
            b'A'..=b'F' => (b - b'A' + 10) as u128,
            _ => return None,
        };
        if d >= radix {
            return None;
        }
        acc = acc.checked_mul(radix)?.checked_add(d)?;
    }
    Some(acc)
}

pub fn eval_const_int(
    expr: &ast::Expr,
    mod_id: ModuleId,
    ctx: &GenericCtx,
    units: &[ast::CompilationUnit],
    prog: &mut ResolvedProgram,
    env: &mut TypeEnv,
    interner: &mut Interner,
) -> Option<i128> {
    match &expr.kind {
        ExprKind::IntLit { raw, .. } => {
            let v = parse_int_literal(interner.get(*raw))?;
            i128::try_from(v).ok()
        }
        ExprKind::Paren(inner) => eval_const_int(inner, mod_id, ctx, units, prog, env, interner),
        ExprKind::Unary {
            op: ast::UnaryOp::Neg,
            expr: inner,
        } => {
            if let ExprKind::IntLit { raw, .. } = &inner.kind {
                let v = parse_int_literal(interner.get(*raw))?;
                if v == 1u128 << 127 {
                    return Some(i128::MIN);
                }
                return i128::try_from(v).ok().and_then(|x| x.checked_neg());
            }
            eval_const_int(inner, mod_id, ctx, units, prog, env, interner).and_then(|x| x.checked_neg())
        }
        ExprKind::Ident(sym) => {
            if let Some(&v) = ctx.comptime_vals.get(sym) {
                return Some(v);
            }
            if let Some(TopScopeTarget::Def(d)) = prog.lookup_in_module(mod_id, *sym)
                && let Some(cm) = env.consts.get(&d)
            {
                return cm.int_val;
            }
            None
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let l = eval_const_int(lhs, mod_id, ctx, units, prog, env, interner)?;
            let r = eval_const_int(rhs, mod_id, ctx, units, prog, env, interner)?;
            match op {
                ast::BinaryOp::Add | ast::BinaryOp::AddWrap => l.checked_add(r),
                ast::BinaryOp::Sub | ast::BinaryOp::SubWrap => l.checked_sub(r),
                ast::BinaryOp::Mul | ast::BinaryOp::MulWrap => l.checked_mul(r),
                ast::BinaryOp::Div if r != 0 => l.checked_div(r),
                ast::BinaryOp::Mod if r != 0 => l.checked_rem(r),
                ast::BinaryOp::Shl if (0..128).contains(&r) => l.checked_shl(r as u32),
                ast::BinaryOp::Shr if (0..128).contains(&r) => l.checked_shr(r as u32),
                ast::BinaryOp::BitAnd => Some(l & r),
                ast::BinaryOp::BitOr => Some(l | r),
                ast::BinaryOp::BitXor => Some(l ^ r),
                _ => None,
            }
        }
        ExprKind::Builtin { name, args } => {
            let bname = interner.get(name.sym);
            let is_size_of = bname == "size_of";
            let is_align_of = bname == "align_of";
            if (is_size_of || is_align_of) && args.len() == 1 {
                let ty = match &args[0].kind {
                    GenericArgKind::Type(t) => {
                        let mut dummy_diags = DiagBag::new();
                        resolve_type_expr(t, mod_id, ctx, units, prog, env, interner, &mut dummy_diags)
                    }
                    GenericArgKind::TypeOrConst(segs) => {
                        let tp = TypePath {
                            segments: segs.clone(),
                            generics: Vec::new(),
                        };
                        let mut dummy_diags = DiagBag::new();
                        resolve_type_path(&tp, args[0].span, mod_id, ctx, units, prog, env, interner, &mut dummy_diags)
                    }
                    GenericArgKind::Expr(_) => return None,
                };
                if ty == TyId::ERROR {
                    return None;
                }
                let layout = env.layout(ty);
                return Some(if is_size_of {
                    layout.size as i128
                } else {
                    layout.align as i128
                });
            }
            None
        }
        _ => None,
    }
}

fn def_qualified_name(prog: &ResolvedProgram, def: DefId, interner: &Interner) -> String {
    let d = &prog.defs[def.0 as usize];
    let m = &prog.modules[d.module.0 as usize];
    let name = interner.get(d.name);
    if m.full_name.is_empty() {
        name.to_owned()
    } else {
        format!("{}.{name}", m.full_name)
    }
}

fn align_up_usize(v: usize, a: usize) -> usize {
    if a <= 1 {
        v
    } else {
        (v + a - 1) & !(a - 1)
    }
}

fn fmt_generic_param_names(params: &[GenericParamMeta], interner: &Interner) -> String {
    if params.is_empty() {
        return String::new();
    }
    let parts: Vec<&str> = params.iter().map(|p| interner.get(p.name)).collect();
    format!("[{}]", parts.join(", "))
}

pub fn dump_types_canonical(
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> String {
    use crate::codegen_x64::{struct_field_offset, x64_layout, x64_layout_impl, X64Layout};

    let mut type_entries: Vec<(String, String)> = Vec::new();

    // Structs
    for (&def_id, meta) in &env.structs {
        let qname = def_qualified_name(prog, def_id, interner);
        let gstr = fmt_generic_param_names(&meta.generics, interner);
        let mut buf = String::new();

        if let Some(tid) = env.find_interned(&Ty::Struct {
            def: def_id,
            args: Vec::new(),
        }) {
            let lay = x64_layout(env, tid);
            let _ = writeln!(
                buf,
                "struct {qname}{gstr} size={} align={}",
                lay.size, lay.align
            );
            for (idx, f) in meta.fields.iter().enumerate() {
                let (off, fty) = struct_field_offset(env, tid, idx);
                let fsz = x64_layout(env, fty).size;
                let fty_str = env.format_ty_resolved(fty, prog, interner);
                let _ = writeln!(
                    buf,
                    "  field {}: {} offset={} size={}",
                    interner.get(f.name),
                    fty_str,
                    off,
                    fsz
                );
            }
        } else {
            let mut field_rows = Vec::with_capacity(meta.fields.len());
            let lay = if meta.fields.is_empty() {
                X64Layout { size: 1, align: 1 }
            } else {
                let mut off = 0usize;
                let mut max_a = 1usize;
                for f in &meta.fields {
                    let l = x64_layout(env, f.ty);
                    let a = if meta.is_packed { 1 } else { l.align.max(1) };
                    max_a = max_a.max(a);
                    off = align_up_usize(off, a);
                    field_rows.push((f.name, f.ty, off, l.size));
                    off += l.size.max(1);
                }
                X64Layout {
                    size: align_up_usize(off, max_a),
                    align: max_a,
                }
            };
            let _ = writeln!(
                buf,
                "struct {qname}{gstr} size={} align={}",
                lay.size, lay.align
            );
            for (fname, fty, off, fsz) in field_rows {
                let fty_str = env.format_ty_resolved(fty, prog, interner);
                let _ = writeln!(
                    buf,
                    "  field {}: {} offset={} size={}",
                    interner.get(fname),
                    fty_str,
                    off,
                    fsz
                );
            }
        }
        type_entries.push((format!("0:{qname}"), buf));
    }

    // Pools
    for (&def_id, meta) in &env.pools {
        let qname = def_qualified_name(prog, def_id, interner);
        let gstr = fmt_generic_param_names(&meta.generics, interner);
        let mut buf = String::new();

        if let Some(tid) = env.find_interned(&Ty::Pool {
            def: def_id,
            args: Vec::new(),
        }) {
            let lay = x64_layout(env, tid);
            let _ = writeln!(
                buf,
                "pool {qname}{gstr} size={} align={}",
                lay.size, lay.align
            );
            for (idx, c) in meta.columns.iter().enumerate() {
                let (off, cty) = struct_field_offset(env, tid, idx);
                let csz = x64_layout_impl(env, cty, true).size;
                let cty_str = env.format_ty_resolved(cty, prog, interner);
                let _ = writeln!(
                    buf,
                    "  field {}: {} offset={} size={}",
                    interner.get(c.name),
                    cty_str,
                    off,
                    csz
                );
            }
        } else {
            let mut col_rows = Vec::with_capacity(meta.columns.len());
            let lay = if meta.columns.is_empty() {
                X64Layout { size: 1, align: 1 }
            } else {
                let mut off = 0usize;
                let mut max_a = 1usize;
                for c in &meta.columns {
                    let l = x64_layout_impl(env, c.ty, true);
                    let a = l.align.max(1);
                    max_a = max_a.max(a);
                    off = align_up_usize(off, a);
                    col_rows.push((c.name, c.ty, off, l.size));
                    off += l.size.max(1);
                }
                X64Layout {
                    size: align_up_usize(off, max_a),
                    align: max_a,
                }
            };
            let _ = writeln!(
                buf,
                "pool {qname}{gstr} size={} align={}",
                lay.size, lay.align
            );
            for (cname, cty, off, csz) in col_rows {
                let cty_str = env.format_ty_resolved(cty, prog, interner);
                let _ = writeln!(
                    buf,
                    "  field {}: {} offset={} size={}",
                    interner.get(cname),
                    cty_str,
                    off,
                    csz
                );
            }
        }
        type_entries.push((format!("1:{qname}"), buf));
    }

    // Unions
    for (&def_id, meta) in &env.unions {
        let qname = def_qualified_name(prog, def_id, interner);
        let gstr = fmt_generic_param_names(&meta.generics, interner);
        let mut max_psz = 1usize;
        let mut max_pa = 1usize;
        for v in &meta.variants {
            if let Some(pt) = v.payload {
                let l = x64_layout(env, pt);
                max_psz = max_psz.max(l.size.max(1));
                max_pa = max_pa.max(l.align.max(1));
            }
        }
        let p_off = align_up_usize(4, max_pa);
        let lay = env
            .find_interned(&Ty::Union {
                def: def_id,
                args: Vec::new(),
            })
            .map(|tid| x64_layout(env, tid))
            .unwrap_or_else(|| {
                let align = max_pa.max(4);
                X64Layout {
                    size: align_up_usize(p_off + max_psz, align),
                    align,
                }
            });
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "union {qname}{gstr} size={} align={}",
            lay.size, lay.align
        );
        for v in &meta.variants {
            if let Some(pt) = v.payload {
                let psz = x64_layout(env, pt).size;
                let pty_str = env.format_ty_resolved(pt, prog, interner);
                let _ = writeln!(
                    buf,
                    "  field {}: {} offset={} size={}",
                    interner.get(v.name),
                    pty_str,
                    p_off,
                    psz
                );
            } else {
                let _ = writeln!(
                    buf,
                    "  field {}: void offset={} size=0",
                    interner.get(v.name),
                    p_off
                );
            }
        }
        type_entries.push((format!("2:{qname}"), buf));
    }

    // Enums
    for (&def_id, meta) in &env.enums {
        let qname = def_qualified_name(prog, def_id, interner);
        let lay = env
            .find_interned(&Ty::Enum(def_id))
            .map(|tid| x64_layout(env, tid))
            .unwrap_or(X64Layout { size: 4, align: 4 });
        let repr_str = env.format_ty_resolved(meta.repr_ty, prog, interner);
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "enum {qname} size={} align={} repr={repr_str}",
            lay.size, lay.align
        );
        for &(vname, val) in &meta.variants {
            let _ = writeln!(buf, "  variant {} = {val}", interner.get(vname));
        }
        type_entries.push((format!("3:{qname}"), buf));
    }

    // Bitfields
    for (&def_id, meta) in &env.bitfields {
        let qname = def_qualified_name(prog, def_id, interner);
        let lay = env
            .find_interned(&Ty::Bitfield(def_id))
            .map(|tid| x64_layout(env, tid))
            .unwrap_or_else(|| x64_layout(env, meta.base_ty));
        let base_str = env.format_ty_resolved(meta.base_ty, prog, interner);
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "bitfield {qname} size={} align={} base={base_str}",
            lay.size, lay.align
        );
        for m in &meta.members {
            let mty_str = env.format_ty_resolved(m.ty, prog, interner);
            let _ = writeln!(
                buf,
                "  field {}: {} bits={}..{}",
                interner.get(m.name),
                mty_str,
                m.lo,
                m.hi
            );
        }
        type_entries.push((format!("4:{qname}"), buf));
    }

    type_entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Functions & methods
    let mut fn_entries: Vec<(String, String)> = Vec::new();

    for (&def_id, sig) in &env.fns {
        let qname = def_qualified_name(prog, def_id, interner);
        let line = format_fn_sig_canonical(&qname, sig, prog, env, interner);
        fn_entries.push((qname, line));
    }

    for im in &env.impls {
        let target_name = match im.target {
            MethodTarget::Def(d) => def_qualified_name(prog, d, interner),
            MethodTarget::Builtin(b) => format!("{b:?}"),
        };
        let prefix = if let Some(ifc) = im.iface {
            format!("{}({})", target_name, def_qualified_name(prog, ifc, interner))
        } else {
            target_name
        };
        for (_, _, msig) in &im.methods {
            let qname = format!("{}.{}", prefix, interner.get(msig.name));
            let line = format_fn_sig_canonical(&qname, msig, prog, env, interner);
            fn_entries.push((qname, line));
        }
    }

    fn_entries.sort();

    let mut out = String::new();
    for (_, text) in type_entries {
        out.push_str(&text);
    }
    for (_, line) in fn_entries {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn format_fn_sig_canonical(
    qname: &str,
    sig: &FnSig,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> String {
    let mut s = String::new();
    if sig.is_pub {
        s.push_str("pub ");
    }
    if sig.is_comptime {
        s.push_str("comptime ");
    }
    if sig.is_extern {
        s.push_str("extern ");
    }
    let gstr = fmt_generic_param_names(&sig.generics, interner);
    let _ = write!(s, "fn {qname}{gstr}(");
    for (i, p) in sig.params.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let pname = if p.is_self {
            "self"
        } else {
            interner.get(p.name)
        };
        let pty = env.format_ty_resolved(p.ty, prog, interner);
        let _ = write!(s, "{pname}: {pty}");
    }
    let ret = env.format_ty_resolved(sig.ret, prog, interner);
    let _ = write!(s, ") -> {ret}");
    s
}
