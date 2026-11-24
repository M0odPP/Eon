use std::collections::BTreeMap;
use std::fmt::Write;

use crate::ast::*;
use crate::diag::{DiagBag, ErrorCode};
use crate::span::{FileId, FileProfile, Interner, Span, Symbol};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimKind {
    UInt(u8), // 1..=64, 128
    USize,
    IInt(u8), // 8, 16, 32, 64, 128
    ISize,
    F16,
    F32,
    F64,
    Bool,
    Char,
    Void,
    Never,
    TypeType,
    Str,
    PhysAddr,
    VirtAddr,
    DmaAddr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinKind {
    Prim(PrimKind),
    VecType,
    SimdAlias(PrimKind, u8),
    ArithError,
    BoundsError,
    Utf8Error,
    OutOfMemory,
    Ordering,
    OrdIface,
    EqIface,
    RawPtr { is_mut: bool },
    MmioReg,
    Arena,
    ArenaPool,
    ArenaStr,
    SlotPool,
    Buf,
    LinearBuf,
    Link,
    IdBrand,
    GenId,
    AllocCap,
    Env,
    FsCap,
    DirCap,
    NetCap,
    ClockCap,
    IoCap,
    ArgsCap,
    BuildCap,
    IoError,
    MemError,
    StaleId,
    Schedule,
    Atomic,
    GpuBuf,
    GpuImage2D,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TopScopeTarget {
    Def(DefId),
    Module(ModuleId),
    Builtin(BuiltinKind),
}

#[derive(Clone, Debug)]
pub struct ScopeBinding {
    pub target: TopScopeTarget,
    pub span: Span,
    pub is_import: bool,
    pub used: bool,
}

#[derive(Clone, Debug)]
pub enum TopDefKind {
    Fn(usize),           // index into unit.decls
    Sys(usize),          // index into unit.decls
    Struct(usize),       // index into unit.decls
    Pool(usize),         // index into unit.decls
    Enum(usize),         // index into unit.decls
    Union(usize),        // index into unit.decls
    Bitfield(usize),     // index into unit.decls
    Iface(usize),        // index into unit.decls
    Error(usize),        // index into unit.decls
    TypeAlias(usize),    // index into unit.decls
    Const(usize),        // index into unit.decls
    ExternFn(usize),     // index into unit.decls
    ExternStruct(usize), // index into unit.decls
    ExternUnion(usize),  // index into unit.decls
}

#[derive(Clone, Debug)]
pub struct TopDef {
    pub id: DefId,
    pub module: ModuleId,
    pub name: Symbol,
    pub span: Span,
    pub is_pub: bool,
    pub kind: TopDefKind,
    pub used: bool,
}

#[derive(Clone, Debug)]
pub struct ImplInfo {
    pub module: ModuleId,
    pub decl_idx: usize,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ModuleInfo {
    pub id: ModuleId,
    pub unit_idx: Option<usize>,
    pub file: FileId,
    pub profile: FileProfile,
    pub full_name: String,
    pub scope: BTreeMap<Symbol, ScopeBinding>,
    pub exports: BTreeMap<Symbol, ScopeBinding>,
    pub deps: Vec<(ModuleId, Span)>,
}

pub struct ResolvedProgram {
    pub modules: Vec<ModuleInfo>,
    pub defs: Vec<TopDef>,
    pub impls: Vec<ImplInfo>,
    pub prelude: BTreeMap<Symbol, BuiltinKind>,
    pub module_by_name: BTreeMap<String, ModuleId>,
}

impl ResolvedProgram {
    pub fn mark_module_binding_used(&mut self, mod_id: ModuleId, name: Symbol) {
        if let Some(b) = self.modules[mod_id.0 as usize].scope.get_mut(&name) {
            b.used = true;
            if let TopScopeTarget::Def(def_id) = b.target {
                self.defs[def_id.0 as usize].used = true;
            }
        }
    }

    pub fn mark_def_used(&mut self, def_id: DefId) {
        self.defs[def_id.0 as usize].used = true;
    }

    pub fn lookup_in_module(&self, mod_id: ModuleId, name: Symbol) -> Option<TopScopeTarget> {
        if let Some(b) = self.modules[mod_id.0 as usize].scope.get(&name) {
            return Some(b.target);
        }
        self.prelude.get(&name).copied().map(TopScopeTarget::Builtin)
    }

    pub fn check_unused_module_items(&self, diags: &mut DiagBag) {
        for m in &self.modules {
            if m.unit_idx.is_none() {
                continue;
            }
            for binding in m.scope.values() {
                if binding.is_import && !binding.used {
                    diags.error(ErrorCode::E0101, binding.span, "ZM-01: unused `import`");
                }
            }
        }

        for def in &self.defs {
            if def.is_pub || def.used {
                continue;
            }
            if matches!(def.kind, TopDefKind::Fn(_)) {
                diags.error(ErrorCode::E0101, def.span, "ZM-01: unused private function");
            }
        }
    }

    pub fn dump_canonical(&self, interner: &Interner) -> String {
        let mut out = String::new();
        let mut mods: Vec<&ModuleInfo> = self
            .modules
            .iter()
            .filter(|m| m.unit_idx.is_some())
            .collect();
        mods.sort_by(|a, b| a.full_name.cmp(&b.full_name).then_with(|| a.id.cmp(&b.id)));

        for m in mods {
            let mname = if m.full_name.is_empty() {
                "<unnamed>"
            } else {
                m.full_name.as_str()
            };
            let hal = if m.profile == FileProfile::Hal {
                " @hal"
            } else {
                ""
            };
            let _ = writeln!(out, "module {mname}{hal}");

            let mut imports: Vec<(String, String)> = Vec::new();
            for (&sym, binding) in &m.scope {
                if !binding.is_import {
                    continue;
                }
                let local = interner.get(sym).to_owned();
                let target = match binding.target {
                    TopScopeTarget::Module(tid) => {
                        let tm = &self.modules[tid.0 as usize];
                        format!("module {}", tm.full_name)
                    }
                    TopScopeTarget::Def(did) => {
                        let def = &self.defs[did.0 as usize];
                        let tm = &self.modules[def.module.0 as usize];
                        format!("item {}.{}", tm.full_name, interner.get(def.name))
                    }
                    TopScopeTarget::Builtin(bk) => format!("builtin {bk:?}"),
                };
                imports.push((local, target));
            }
            imports.sort();
            for (local, target) in imports {
                let _ = writeln!(out, "  import {local} -> {target}");
            }

            let mut items: Vec<(&str, &str, bool)> = Vec::new();
            for def in &self.defs {
                if def.module != m.id {
                    continue;
                }
                let kind_str = match def.kind {
                    TopDefKind::Fn(_) => "fn",
                    TopDefKind::Sys(_) => "sys",
                    TopDefKind::Struct(_) => "struct",
                    TopDefKind::Pool(_) => "pool",
                    TopDefKind::Enum(_) => "enum",
                    TopDefKind::Union(_) => "union",
                    TopDefKind::Bitfield(_) => "bitfield",
                    TopDefKind::Iface(_) => "iface",
                    TopDefKind::Error(_) => "error",
                    TopDefKind::TypeAlias(_) => "type",
                    TopDefKind::Const(_) => "const",
                    TopDefKind::ExternFn(_) => "extern_fn",
                    TopDefKind::ExternStruct(_) => "extern_struct",
                    TopDefKind::ExternUnion(_) => "extern_union",
                };
                items.push((interner.get(def.name), kind_str, def.is_pub));
            }
            items.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));
            for (name, kind_str, is_pub) in items {
                let vis = if is_pub { "pub " } else { "" };
                let _ = writeln!(out, "  {vis}{kind_str} {name}");
            }
        }
        out
    }
}

pub fn resolve_program(
    units: &[CompilationUnit],
    profiles: &[FileProfile],
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> ResolvedProgram {
    let prelude = build_prelude(interner);
    let mut modules = Vec::new();
    let mut defs = Vec::new();
    let mut impls = Vec::new();
    let mut module_by_name = BTreeMap::new();

    // 1. Register user modules.
    for (ui, unit) in units.iter().enumerate() {
        let full_name = join_module_path(&unit.module_path, interner);
        let mod_id = ModuleId(modules.len() as u32);
        if !full_name.is_empty() && module_by_name.contains_key(&full_name) {
            let sp = unit
                .module_path
                .first()
                .map(|s| s.span)
                .unwrap_or(Span::DUMMY);
            diags.error(
                ErrorCode::E0132,
                sp,
                format!("duplicate module `{}`", full_name),
            );
        } else if !full_name.is_empty() {
            module_by_name.insert(full_name.clone(), mod_id);
        }

        let mut scope = BTreeMap::new();
        let mut exports = BTreeMap::new();

        for (di, decl) in unit.decls.iter().enumerate() {
            if let DeclKind::Impl(_) = &decl.kind {
                impls.push(ImplInfo {
                    module: mod_id,
                    decl_idx: di,
                    span: decl.span,
                });
                continue;
            }
            let (name, kind, is_auto_used) = match &decl.kind {
                DeclKind::Fn(f) => {
                    let name_str = interner.get(f.name.sym);
                    let exported = decl.attrs.iter().any(|a| {
                        matches!(interner.get(a.name.sym), "extern" | "naked" | "interrupt")
                    });
                    (f.name, TopDefKind::Fn(di), name_str == "main" || exported)
                }
                DeclKind::Sys(s) => (s.name, TopDefKind::Sys(di), false),
                DeclKind::Struct(s) => (s.name, TopDefKind::Struct(di), false),
                DeclKind::Pool(p) => (p.name, TopDefKind::Pool(di), false),
                DeclKind::Enum(e) => (e.name, TopDefKind::Enum(di), false),
                DeclKind::Union(u) => (u.name, TopDefKind::Union(di), false),
                DeclKind::Bitfield(b) => (b.name, TopDefKind::Bitfield(di), false),
                DeclKind::Iface(i) => (i.name, TopDefKind::Iface(di), false),
                DeclKind::Error(e) => (e.name, TopDefKind::Error(di), false),
                DeclKind::TypeAlias(t) => (t.name, TopDefKind::TypeAlias(di), false),
                DeclKind::Const(c) => (c.name, TopDefKind::Const(di), false),
                DeclKind::ExternFn(e) => (e.name, TopDefKind::ExternFn(di), false),
                DeclKind::ExternStruct(e) => (e.name, TopDefKind::ExternStruct(di), false),
                DeclKind::ExternUnion(e) => (e.name, TopDefKind::ExternUnion(di), false),
                DeclKind::Impl(_) => unreachable!(),
            };

            if scope.contains_key(&name.sym) {
                diags.error(
                    ErrorCode::E0132,
                    name.span,
                    format!(
                        "duplicate definition of `{}` in module `{}`",
                        interner.get(name.sym),
                        full_name
                    ),
                );
                continue;
            }

            let def_id = DefId(defs.len() as u32);
            defs.push(TopDef {
                id: def_id,
                module: mod_id,
                name: name.sym,
                span: name.span,
                is_pub: decl.is_pub,
                kind,
                used: is_auto_used,
            });
            let binding = ScopeBinding {
                target: TopScopeTarget::Def(def_id),
                span: name.span,
                is_import: false,
                used: is_auto_used,
            };
            scope.insert(name.sym, binding.clone());
            if decl.is_pub {
                exports.insert(name.sym, binding);
            }
        }

        modules.push(ModuleInfo {
            id: mod_id,
            unit_idx: Some(ui),
            file: unit.file,
            profile: profiles[ui],
            full_name,
            scope,
            exports,
            deps: Vec::new(),
        });
    }

    // 2. Register virtual built-in `core.*` modules if not overridden by user files.
    register_virtual_core_modules(&mut modules, &mut module_by_name, interner);

    // 3. Resolve imports across modules.
    for ui in 0..units.len() {
        let unit = &units[ui];
        let mod_id = ModuleId(ui as u32);
        for imp in &unit.imports {
            let target_name = join_module_path(&imp.path, interner);
            let Some(&target_id) = module_by_name.get(&target_name) else {
                diags.error(
                    ErrorCode::E0132,
                    imp.span,
                    format!("unresolved module `{}`", target_name),
                );
                continue;
            };
            if target_id == mod_id {
                diags.error(
                    ErrorCode::E0131,
                    imp.span,
                    format!("ZM: module `{}` cannot import itself (cycle)", target_name),
                );
                continue;
            }
            modules[ui].deps.push((target_id, imp.span));

            match &imp.tail {
                ImportTail::None => {
                    if let Some(last) = imp.path.last() {
                        insert_import_binding(
                            &mut modules[ui].scope,
                            last.sym,
                            last.span,
                            TopScopeTarget::Module(target_id),
                            interner,
                            diags,
                        );
                    }
                }
                ImportTail::Alias(alias) => {
                    insert_import_binding(
                        &mut modules[ui].scope,
                        alias.sym,
                        alias.span,
                        TopScopeTarget::Module(target_id),
                        interner,
                        diags,
                    );
                }
                ImportTail::Items(items) => {
                    for item in items {
                        let exported = modules[target_id.0 as usize].exports.get(&item.name.sym).cloned();
                        let exists_private = modules[target_id.0 as usize].scope.contains_key(&item.name.sym);
                        if let Some(b) = exported {
                            let bind_sym = item.alias.unwrap_or(item.name).sym;
                            let bind_span = item.alias.unwrap_or(item.name).span;
                            insert_import_binding(
                                &mut modules[ui].scope,
                                bind_sym,
                                bind_span,
                                b.target,
                                interner,
                                diags,
                            );
                        } else if exists_private {
                            diags.error(
                                ErrorCode::E0133,
                                item.name.span,
                                format!(
                                    "symbol `{}` in module `{}` is private (missing `pub`)",
                                    interner.get(item.name.sym),
                                    target_name
                                ),
                            );
                        } else {
                            diags.error(
                                ErrorCode::E0132,
                                item.name.span,
                                format!(
                                    "symbol `{}` not found in module `{}`",
                                    interner.get(item.name.sym),
                                    target_name
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    // 4. Detect cyclic module dependencies (DAG invariant).
    check_module_cycles(&modules, diags);

    ResolvedProgram {
        modules,
        defs,
        impls,
        prelude,
        module_by_name,
    }
}

fn insert_import_binding(
    scope: &mut BTreeMap<Symbol, ScopeBinding>,
    sym: Symbol,
    span: Span,
    target: TopScopeTarget,
    interner: &Interner,
    diags: &mut DiagBag,
) {
    if scope.contains_key(&sym) {
        diags.error(
            ErrorCode::E0132,
            span,
            format!(
                "import `{}` conflicts with an existing definition or import in this module",
                interner.get(sym)
            ),
        );
        return;
    }
    scope.insert(
        sym,
        ScopeBinding {
            target,
            span,
            is_import: true,
            used: false,
        },
    );
}

fn check_module_cycles(modules: &[ModuleInfo], diags: &mut DiagBag) {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    fn dfs(
        m: usize,
        modules: &[ModuleInfo],
        colors: &mut [Color],
        stack: &mut Vec<usize>,
        diags: &mut DiagBag,
    ) {
        colors[m] = Color::Gray;
        stack.push(m);
        for &(dep, span) in &modules[m].deps {
            let d = dep.0 as usize;
            match colors[d] {
                Color::White => dfs(d, modules, colors, stack, diags),
                Color::Gray => {
                    let pos = stack.iter().position(|&x| x == d).unwrap_or(0);
                    let mut cycle_names: Vec<&str> =
                        stack[pos..].iter().map(|&idx| modules[idx].full_name.as_str()).collect();
                    cycle_names.push(modules[d].full_name.as_str());
                    diags.error(
                        ErrorCode::E0131,
                        span,
                        format!("cyclic module dependency detected: {}", cycle_names.join(" -> ")),
                    );
                }
                Color::Black => {}
            }
        }
        stack.pop();
        colors[m] = Color::Black;
    }

    let mut colors = vec![Color::White; modules.len()];
    let mut stack = Vec::new();
    for i in 0..modules.len() {
        if colors[i] == Color::White && modules[i].unit_idx.is_some() {
            dfs(i, modules, &mut colors, &mut stack, diags);
        }
    }
}

fn join_module_path(path: &[IdentSpan], interner: &Interner) -> String {
    let mut s = String::new();
    for (i, seg) in path.iter().enumerate() {
        if i > 0 {
            s.push('.');
        }
        s.push_str(interner.get(seg.sym));
    }
    s
}

fn register_virtual_core_modules(
    modules: &mut Vec<ModuleInfo>,
    module_by_name: &mut BTreeMap<String, ModuleId>,
    interner: &mut Interner,
) {
    let entries: &[(&str, &[(&str, BuiltinKind)])] = &[
        (
            "core.addr",
            &[
                ("PhysAddr", BuiltinKind::Prim(PrimKind::PhysAddr)),
                ("VirtAddr", BuiltinKind::Prim(PrimKind::VirtAddr)),
                ("DmaAddr", BuiltinKind::Prim(PrimKind::DmaAddr)),
            ],
        ),
        (
            "core.err",
            &[
                ("ArithError", BuiltinKind::ArithError),
                ("BoundsError", BuiltinKind::BoundsError),
                ("Utf8Error", BuiltinKind::Utf8Error),
                ("OutOfMemory", BuiltinKind::OutOfMemory),
                ("MemError", BuiltinKind::MemError),
                ("StaleId", BuiltinKind::StaleId),
                ("IoError", BuiltinKind::IoError),
            ],
        ),
        (
            "core.mem",
            &[
                ("Arena", BuiltinKind::Arena),
                ("ArenaPool", BuiltinKind::ArenaPool),
                ("ArenaStr", BuiltinKind::ArenaStr),
                ("SlotPool", BuiltinKind::SlotPool),
                ("Buf", BuiltinKind::Buf),
                ("LinearBuf", BuiltinKind::LinearBuf),
                ("AllocCap", BuiltinKind::AllocCap),
                ("Env", BuiltinKind::Env),
                ("FsCap", BuiltinKind::FsCap),
                ("DirCap", BuiltinKind::DirCap),
                ("NetCap", BuiltinKind::NetCap),
                ("ClockCap", BuiltinKind::ClockCap),
                ("IoCap", BuiltinKind::IoCap),
                ("ArgsCap", BuiltinKind::ArgsCap),
                ("BuildCap", BuiltinKind::BuildCap),
            ],
        ),
    ];

    for &(mod_name, items) in entries {
        if module_by_name.contains_key(mod_name) {
            continue;
        }
        let id = ModuleId(modules.len() as u32);
        module_by_name.insert(mod_name.to_owned(), id);
        let mut exports = BTreeMap::new();
        for &(item_name, kind) in items {
            let sym = interner.intern(item_name);
            exports.insert(
                sym,
                ScopeBinding {
                    target: TopScopeTarget::Builtin(kind),
                    span: Span::DUMMY,
                    is_import: false,
                    used: true,
                },
            );
        }
        modules.push(ModuleInfo {
            id,
            unit_idx: None,
            file: FileId::DUMMY,
            profile: FileProfile::Safe,
            full_name: mod_name.to_owned(),
            scope: exports.clone(),
            exports,
            deps: Vec::new(),
        });
    }
}

fn build_prelude(interner: &mut Interner) -> BTreeMap<Symbol, BuiltinKind> {
    let mut m = BTreeMap::new();
    let mut add = |name: &str, k: BuiltinKind| {
        m.insert(interner.intern(name), k);
    };

    for bits in 1..=64u8 {
        add(&format!("u{}", bits), BuiltinKind::Prim(PrimKind::UInt(bits)));
    }
    add("u128", BuiltinKind::Prim(PrimKind::UInt(128)));
    add("usize", BuiltinKind::Prim(PrimKind::USize));

    for bits in 1..=64u8 {
        add(&format!("i{}", bits), BuiltinKind::Prim(PrimKind::IInt(bits)));
    }
    add("i128", BuiltinKind::Prim(PrimKind::IInt(128)));
    add("isize", BuiltinKind::Prim(PrimKind::ISize));

    add("f16", BuiltinKind::Prim(PrimKind::F16));
    add("f32", BuiltinKind::Prim(PrimKind::F32));
    add("f64", BuiltinKind::Prim(PrimKind::F64));
    add("bool", BuiltinKind::Prim(PrimKind::Bool));
    add("char", BuiltinKind::Prim(PrimKind::Char));
    add("str", BuiltinKind::Prim(PrimKind::Str));

    add("vec", BuiltinKind::VecType);
    add("f32x2", BuiltinKind::SimdAlias(PrimKind::F32, 2));
    add("f32x4", BuiltinKind::SimdAlias(PrimKind::F32, 4));
    add("f32x8", BuiltinKind::SimdAlias(PrimKind::F32, 8));
    add("f64x2", BuiltinKind::SimdAlias(PrimKind::F64, 2));
    add("f64x4", BuiltinKind::SimdAlias(PrimKind::F64, 4));
    add("i32x4", BuiltinKind::SimdAlias(PrimKind::IInt(32), 4));
    add("i32x8", BuiltinKind::SimdAlias(PrimKind::IInt(32), 8));
    add("u8x4", BuiltinKind::SimdAlias(PrimKind::UInt(8), 4));
    add("u8x16", BuiltinKind::SimdAlias(PrimKind::UInt(8), 16));
    add("u8x32", BuiltinKind::SimdAlias(PrimKind::UInt(8), 32));
    add("u32x2", BuiltinKind::SimdAlias(PrimKind::UInt(32), 2));
    add("u32x4", BuiltinKind::SimdAlias(PrimKind::UInt(32), 4));

    add("PhysAddr", BuiltinKind::Prim(PrimKind::PhysAddr));
    add("VirtAddr", BuiltinKind::Prim(PrimKind::VirtAddr));
    add("DmaAddr", BuiltinKind::Prim(PrimKind::DmaAddr));

    add("ArithError", BuiltinKind::ArithError);
    add("BoundsError", BuiltinKind::BoundsError);
    add("Utf8Error", BuiltinKind::Utf8Error);
    add("OutOfMemory", BuiltinKind::OutOfMemory);
    add("MemError", BuiltinKind::MemError);
    add("StaleId", BuiltinKind::StaleId);
    add("IoError", BuiltinKind::IoError);
    add("Ordering", BuiltinKind::Ordering);
    add("Ord", BuiltinKind::OrdIface);
    add("Eq", BuiltinKind::EqIface);

    add("rawptr", BuiltinKind::RawPtr { is_mut: false });
    add("rawptr_mut", BuiltinKind::RawPtr { is_mut: true });
    add("MmioReg", BuiltinKind::MmioReg);
    add("Arena", BuiltinKind::Arena);
    add("ArenaPool", BuiltinKind::ArenaPool);
    add("ArenaStr", BuiltinKind::ArenaStr);
    add("SlotPool", BuiltinKind::SlotPool);
    add("Buf", BuiltinKind::Buf);
    add("LinearBuf", BuiltinKind::LinearBuf);
    add("Link", BuiltinKind::Link);
    add("Id", BuiltinKind::IdBrand);
    add("GenId", BuiltinKind::GenId);
    add("AllocCap", BuiltinKind::AllocCap);
    add("Env", BuiltinKind::Env);
    add("FsCap", BuiltinKind::FsCap);
    add("DirCap", BuiltinKind::DirCap);
    add("NetCap", BuiltinKind::NetCap);
    add("ClockCap", BuiltinKind::ClockCap);
    add("IoCap", BuiltinKind::IoCap);
    add("ArgsCap", BuiltinKind::ArgsCap);
    add("BuildCap", BuiltinKind::BuildCap);
    add("Schedule", BuiltinKind::Schedule);
    add("Atomic", BuiltinKind::Atomic);
    add("GpuBuf", BuiltinKind::GpuBuf);
    add("GpuImage2D", BuiltinKind::GpuImage2D);

    m
}
