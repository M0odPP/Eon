//! Binary Module Interface (`.emi`) cache & per-function incremental compilation (Early Cutoff).
//! Header format (64 bytes, per `spec/09-abi-modules-and-hotreload.md` §3):
//!   [0..4]   magic = "EOMI"
//!   [4..8]   compiler_ver: u32
//!   [8..24]  interface_hash: u128
//!   [24..28] types_offset: u32
//!   [28..32] types_count: u32
//!   [32..36] exports_offset: u32
//!   [36..40] exports_count: u32
//!   [40..44] fn_hashes_offset: u32
//!   [44..48] fn_hashes_count: u32
//!   [48..52] blobs_offset: u32
//!   [52..56] blobs_count: u32
//!   [56..60] flags: u32
//!   [60..64] reserved: u32

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::cfg::{CfgBuiltinOp, CfgConst, CfgFnKey};
use crate::codegen_x64::{
    CompiledFnBlob, RodataItem, X64Options, collect_source_files, compile_function_x64,
    fn_sym_name, link_x64_blobs,
};
use crate::eir::{EirCallee, EirFunction, EirInstKind, EirProgram, EirTerm};
use crate::elf64::{ElfMode, LineEntry, LinkedElf, RelocKind};
use crate::resolve::{ModuleId, ResolvedProgram, TopDefKind, TopScopeTarget};
use crate::span::{Interner, SourceMap};
use crate::types::{Ty, TypeEnv};
use crate::x64_asm::FnReloc;

pub const EMI_MAGIC: [u8; 4] = *b"EOMI";
pub const EMI_COMPILER_VER: u32 = 0x0002_0000;
pub const EMI_HEADER_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct EmiTypeEntry {
    pub name: String,
    pub size: u64,
    pub align: u64,
    pub has_niche: bool,
}

#[derive(Clone, Debug)]
pub struct EmiExportEntry {
    pub name: String,
    pub kind_tag: u8,
    pub sig_hash: u128,
}

#[derive(Clone, Debug)]
pub struct EmiFnHashEntry {
    pub sym_name: String,
    pub body_hash: u128,
}

#[derive(Clone, Debug)]
pub struct EmiFile {
    pub module_name: String,
    pub interface_hash: u128,
    pub types: Vec<EmiTypeEntry>,
    pub exports: Vec<EmiExportEntry>,
    pub fn_hashes: Vec<EmiFnHashEntry>,
    pub blobs: Vec<CompiledFnBlob>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IncrementalStats {
    pub modules_total: usize,
    pub modules_early_cutoff: usize,
    pub fns_total: usize,
    pub fns_cached: usize,
    pub fns_recompiled: usize,
}

// ============================================================================
// Deterministic 128-bit Dual-Lane Hasher (zero external deps!)
// ============================================================================

struct Hasher128 {
    h1: u64,
    h2: u64,
}

impl Hasher128 {
    fn new() -> Self {
        Self {
            h1: 0xcbf2_9ce4_8422_2325,
            h2: 0x8422_2325_cbf2_9ce4,
        }
    }

    fn write_bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.h1 ^= b as u64;
            self.h1 = self.h1.wrapping_mul(0x0000_0100_0000_01b3);
            self.h2 ^= (b as u64).wrapping_add(0x9e);
            self.h2 = self.h2.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            self.h2 = self.h2.rotate_left(13);
        }
    }

    fn write_u8(&mut self, v: u8) { self.write_bytes(&[v]); }
    fn write_u32(&mut self, v: u32) { self.write_bytes(&v.to_le_bytes()); }
    fn write_u64(&mut self, v: u64) { self.write_bytes(&v.to_le_bytes()); }
    fn write_str(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.write_bytes(s.as_bytes());
    }

    fn finish(self) -> u128 {
        ((self.h1 as u128) << 64) | (self.h2 as u128)
    }
}

/// Compute `InterfaceHash: u128` for a module: depends ONLY on `pub` items (signatures,
/// layouts, effects, `pub const`) and bodies of `pub` generics / `@inline` / `comptime fn`.
pub fn compute_module_interface(
    mod_id: ModuleId,
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> (u128, Vec<EmiTypeEntry>, Vec<EmiExportEntry>) {
    let minfo = &prog.modules[mod_id.0 as usize];
    let mut h = Hasher128::new();
    h.write_str(&minfo.full_name);

    let mut types = Vec::new();
    let mut exports = Vec::new();

    for (&sym, binding) in &minfo.exports {
        let name = interner.get(sym).to_owned();
        if let TopScopeTarget::Def(d) = binding.target {
            let def = &prog.defs[d.0 as usize];
            if !def.is_pub {
                continue;
            }
            let mut sh = Hasher128::new();
            sh.write_str(&name);
            let kind_tag: u8 = match def.kind {
                TopDefKind::Fn(_) | TopDefKind::Sys(_) | TopDefKind::ExternFn(_) => {
                    if let Some(sig) = env.fns.get(&d) {
                        hash_fn_sig(&mut sh, sig, env, interner);
                        // Spec §3.1: body of `pub` generics, `@inline`, and `comptime fn` IS part of interface!
                        if (!sig.generics.is_empty() || sig.effects.is_inline || sig.is_comptime)
                            && let Some(&fi) = eir.by_key.get(&CfgFnKey::TopDef(d))
                        {
                            let bh = compute_fn_body_hash(&eir.functions[fi], eir, env, interner);
                            sh.write_u64((bh >> 64) as u64);
                            sh.write_u64(bh as u64);
                        }
                    }
                    1
                }
                TopDefKind::Struct(_) => {
                    if let Some(sm) = env.structs.get(&d) {
                        sh.write_u8(u8::from(sm.is_linear));
                        sh.write_u8(u8::from(sm.is_view));
                        sh.write_u64(sm.layout.size);
                        sh.write_u64(sm.layout.align);
                        for fld in &sm.fields {
                            sh.write_str(interner.get(fld.name));
                            sh.write_str(&env.format_ty(fld.ty, interner));
                            sh.write_u64(fld.offset);
                        }
                        types.push(EmiTypeEntry {
                            name: name.clone(),
                            size: sm.layout.size,
                            align: sm.layout.align,
                            has_niche: sm.layout.has_niche,
                        });
                    }
                    2
                }
                TopDefKind::Pool(_) => {
                    if let Some(pm) = env.pools.get(&d) {
                        for col in &pm.columns {
                            sh.write_str(interner.get(col.name));
                            sh.write_str(&env.format_ty(col.ty, interner));
                            sh.write_u64(col.col_align);
                        }
                    }
                    3
                }
                TopDefKind::Enum(_) => {
                    if let Some(em) = env.enums.get(&d) {
                        sh.write_str(&env.format_ty(em.repr_ty, interner));
                        for (vn, vv) in &em.variants {
                            sh.write_str(interner.get(*vn));
                            sh.write_u64(*vv as u64);
                        }
                    }
                    4
                }
                TopDefKind::Union(_) => {
                    if let Some(um) = env.unions.get(&d) {
                        sh.write_u64(um.layout.size);
                        sh.write_u64(um.layout.align);
                        for v in &um.variants {
                            sh.write_str(interner.get(v.name));
                            if let Some(pt) = v.payload {
                                sh.write_str(&env.format_ty(pt, interner));
                            }
                        }
                        types.push(EmiTypeEntry {
                            name: name.clone(),
                            size: um.layout.size,
                            align: um.layout.align,
                            has_niche: um.layout.has_niche,
                        });
                    }
                    5
                }
                TopDefKind::Const(_) => {
                    if let Some(cm) = env.consts.get(&d) {
                        sh.write_str(&env.format_ty(cm.ty, interner));
                        if let Some(v) = cm.int_val {
                            sh.write_u64(v as u64);
                        }
                    }
                    6
                }
                _ => 7,
            };
            let sig_hash = sh.finish();
            h.write_str(&name);
            h.write_u8(kind_tag);
            h.write_u64((sig_hash >> 64) as u64);
            h.write_u64(sig_hash as u64);
            exports.push(EmiExportEntry {
                name,
                kind_tag,
                sig_hash,
            });
        }
    }

    (h.finish(), types, exports)
}

fn hash_fn_sig(h: &mut Hasher128, sig: &crate::types::FnSig, env: &TypeEnv, interner: &Interner) {
    h.write_str(interner.get(sig.name));
    h.write_u8(u8::from(sig.is_pub));
    h.write_u8(u8::from(sig.is_view));
    h.write_u8(u8::from(sig.is_extern));
    h.write_u8(u8::from(sig.effects.is_pure));
    h.write_u8(u8::from(sig.effects.is_noalloc));
    h.write_u8(u8::from(sig.effects.is_noblock));
    h.write_u8(u8::from(sig.effects.is_interrupt));
    h.write_u8(u8::from(sig.effects.is_naked));
    h.write_u8(u8::from(sig.effects.is_export_c));
    h.write_u8(u8::from(sig.effects.is_gpu));
    if let Some(lim) = sig.effects.stack_limit {
        h.write_u64(lim);
    }
    h.write_u32(sig.params.len() as u32);
    for p in &sig.params {
        h.write_str(&env.format_ty(p.ty, interner));
    }
    h.write_str(&env.format_ty(sig.ret, interner));
}

/// Compute canonical per-function `FnBodyHash: u128` over EIR structure (immune to whitespace,
/// comments, and edits to other functions!).
pub fn compute_fn_body_hash(
    f: &EirFunction,
    eir: &EirProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> u128 {
    let mut h = Hasher128::new();
    h.write_str(&fn_sym_name(f.key, eir, interner));
    hash_fn_sig(&mut h, &f.sig, env, interner);
    if let Some(tag) = &f.layout_shared_tag {
        h.write_str(tag);
    }
    h.write_u32(f.slots.len() as u32);
    for sl in &f.slots {
        h.write_str(&env.format_ty(sl.ty, interner));
    }
    h.write_u32(f.value_tys.len() as u32);
    for &vt in &f.value_tys {
        h.write_str(&env.format_ty(vt, interner));
    }
    h.write_u32(f.par_for_regions.len() as u32);
    for pf in &f.par_for_regions {
        h.write_u32(pf.header_bb.0);
        h.write_u32(pf.body_bb.0);
        h.write_u32(pf.step_bb.0);
        h.write_u32(pf.exit_bb.0);
        h.write_u32(pf.idx_slot.0);
        h.write_u32(pf.end_slot.0);
        h.write_u64(pf.chunk_size);
    }
    h.write_u32(f.blocks.len() as u32);
    for b in &f.blocks {
        h.write_u32(b.id.0);
        h.write_u32(b.phis.len() as u32);
        for phi in &b.phis {
            h.write_u32(phi.dst.0);
            for &(pb, pv) in &phi.incoming {
                h.write_u32(pb.0);
                h.write_u32(pv.0);
            }
        }
        h.write_u32(b.insts.len() as u32);
        for inst in &b.insts {
            hash_inst(&mut h, &inst.kind, eir, env, interner);
        }
        hash_term(&mut h, &b.term);
    }
    h.finish()
}

fn hash_inst(h: &mut Hasher128, k: &EirInstKind, eir: &EirProgram, env: &TypeEnv, interner: &Interner) {
    match k {
        EirInstKind::Const { dst, ty, konst } => {
            h.write_u8(1);
            h.write_u32(dst.0);
            h.write_str(&env.format_ty(*ty, interner));
            match konst {
                CfgConst::Void => h.write_u8(0),
                CfgConst::Bool(b) => { h.write_u8(1); h.write_u8(u8::from(*b)); }
                CfgConst::Char(c) => { h.write_u8(2); h.write_u32(*c as u32); }
                CfgConst::Int(n, _) => { h.write_u8(3); h.write_u64(*n as u64); h.write_u64((*n >> 64) as u64); }
                CfgConst::Float(s, _) => { h.write_u8(4); h.write_str(s); }
                CfgConst::Str(s) => { h.write_u8(5); h.write_str(s); }
                CfgConst::Bytes(bs) => { h.write_u8(6); h.write_bytes(bs); }
                CfgConst::Null(_) => h.write_u8(7),
                CfgConst::EnumTag { tag_val, .. } => { h.write_u8(8); h.write_u64(*tag_val as u64); }
                CfgConst::ErrorTag { tag_val, .. } => { h.write_u8(9); h.write_u32(*tag_val); }
                CfgConst::FnRef(d, _) => { h.write_u8(10); h.write_str(&fn_sym_name(CfgFnKey::TopDef(*d), eir, interner)); }
                CfgConst::ConstDef(d, _) => { h.write_u8(11); h.write_u32(d.0); }
            }
        }
        EirInstKind::CopyVal { dst, src } => { h.write_u8(2); h.write_u32(dst.0); h.write_u32(src.0); }
        EirInstKind::LoadSlot { dst, slot } => { h.write_u8(3); h.write_u32(dst.0); h.write_u32(slot.0); }
        EirInstKind::StoreSlot { slot, val } => { h.write_u8(4); h.write_u32(slot.0); h.write_u32(val.0); }
        EirInstKind::SlotAddr { dst, slot, is_mut, .. } => {
            h.write_u8(5); h.write_u32(dst.0); h.write_u32(slot.0); h.write_u8(u8::from(*is_mut));
        }
        EirInstKind::Unary { dst, op, operand, .. } => {
            h.write_u8(6); h.write_u32(dst.0); h.write_u8(*op as u8); h.write_u32(operand.0);
        }
        EirInstKind::Binary { dst, op, lhs, rhs, .. } => {
            h.write_u8(7); h.write_u32(dst.0); h.write_u8(*op as u8); h.write_u32(lhs.0); h.write_u32(rhs.0);
        }
        EirInstKind::Call { dst, callee, args, .. } => {
            h.write_u8(8);
            h.write_u32(dst.map_or(u32::MAX, |v| v.0));
            match callee {
                EirCallee::DirectFn(d) => h.write_str(&fn_sym_name(CfgFnKey::TopDef(*d), eir, interner)),
                EirCallee::Method { impl_idx, sub_idx } => {
                    h.write_str(&fn_sym_name(CfgFnKey::ImplMethod { impl_idx: *impl_idx, sub_idx: *sub_idx }, eir, interner));
                }
                EirCallee::GenericInst { inst_id } => {
                    h.write_str(&fn_sym_name(CfgFnKey::GenericInst(*inst_id), eir, interner));
                }
                EirCallee::DynCall { iface_def, method_idx } => {
                    h.write_u32(iface_def.0); h.write_u32(*method_idx as u32);
                }
                EirCallee::ExternFn { name, .. } => h.write_str(interner.get(*name)),
                EirCallee::FnPtr(v) => h.write_u32(v.0),
            }
            for &a in args { h.write_u32(a.0); }
        }
        EirInstKind::InlineAsm { template, inputs } => {
            h.write_u8(9);
            h.write_str(template);
            for (c, v) in inputs { h.write_str(c); h.write_u32(v.0); }
        }
        EirInstKind::BuiltinCall { dst, op, args, arg_slots, ret_ty } => {
            h.write_u8(11);
            h.write_str(&format!("{dst:?}:{op:?}:{args:?}:{arg_slots:?}:{ret_ty:?}"));
            if matches!(op, CfgBuiltinOp::ScheduleRun) {
                h.write_str(&format!("{:?}", eir.schedules));
            }
        }
        other => {
            // Deterministic structural debug digest for remaining EIR variants.
            h.write_u8(10);
            h.write_str(&format!("{other:?}"));
        }
    }
}

fn hash_term(h: &mut Hasher128, t: &EirTerm) {
    match t {
        EirTerm::Goto(b) => { h.write_u8(1); h.write_u32(b.0); }
        EirTerm::Branch { cond, then_bb, else_bb, .. } => {
            h.write_u8(2); h.write_u32(cond.0); h.write_u32(then_bb.0); h.write_u32(else_bb.0);
        }
        EirTerm::SwitchInt { discr, targets, otherwise } => {
            h.write_u8(3); h.write_u32(discr.0); h.write_u32(otherwise.0);
            for &(v, b) in targets { h.write_u64(v as u64); h.write_u32(b.0); }
        }
        EirTerm::Return(v) => { h.write_u8(4); h.write_u32(v.map_or(u32::MAX, |x| x.0)); }
        EirTerm::Unreachable { explicit, .. } => { h.write_u8(5); h.write_u8(u8::from(*explicit)); }
    }
}

// ============================================================================
// Binary `.emi` Serialization / Deserialization (64-byte fixed header)
// ============================================================================

pub fn write_emi(emi: &EmiFile) -> Vec<u8> {
    let mut body = Vec::new();
    // Module name precedes section payloads right after the 64B header.
    write_str(&mut body, &emi.module_name);

    let types_off = (EMI_HEADER_SIZE + body.len()) as u32;
    for t in &emi.types {
        write_str(&mut body, &t.name);
        body.extend_from_slice(&t.size.to_le_bytes());
        body.extend_from_slice(&t.align.to_le_bytes());
        body.push(u8::from(t.has_niche));
    }

    let exports_off = (EMI_HEADER_SIZE + body.len()) as u32;
    for e in &emi.exports {
        write_str(&mut body, &e.name);
        body.push(e.kind_tag);
        body.extend_from_slice(&e.sig_hash.to_le_bytes());
    }

    let fn_hashes_off = (EMI_HEADER_SIZE + body.len()) as u32;
    for fh in &emi.fn_hashes {
        write_str(&mut body, &fh.sym_name);
        body.extend_from_slice(&fh.body_hash.to_le_bytes());
    }

    let blobs_off = (EMI_HEADER_SIZE + body.len()) as u32;
    for b in &emi.blobs {
        write_str(&mut body, &b.sym_name);
        write_bytes(&mut body, &b.code);
        body.extend_from_slice(&(b.relocs.len() as u32).to_le_bytes());
        for r in &b.relocs {
            body.extend_from_slice(&(r.offset as u32).to_le_bytes());
            body.push(match r.kind {
                RelocKind::Rel32 => 0,
                RelocKind::Abs64 => 1,
                RelocKind::GotPcRel32 => 2,
                RelocKind::Plt32 => 3,
                RelocKind::TpOff32 => 4,
            });
            write_str(&mut body, &r.target_sym);
            body.extend_from_slice(&r.addend.to_le_bytes());
        }
        body.extend_from_slice(&(b.rodata.len() as u32).to_le_bytes());
        for ro in &b.rodata {
            write_str(&mut body, &ro.sym);
            body.extend_from_slice(&(ro.align as u32).to_le_bytes());
            write_bytes(&mut body, &ro.bytes);
        }
        body.extend_from_slice(&(b.line_entries.len() as u32).to_le_bytes());
        for le in &b.line_entries {
            body.extend_from_slice(&le.code_offset.to_le_bytes());
            body.extend_from_slice(&le.file_idx.to_le_bytes());
            body.extend_from_slice(&le.line.to_le_bytes());
            body.extend_from_slice(&le.col.to_le_bytes());
        }
        body.extend_from_slice(&(b.used_helpers.len() as u32).to_le_bytes());
        for h in &b.used_helpers {
            write_str(&mut body, h);
        }
    }

    let mut out = Vec::with_capacity(EMI_HEADER_SIZE + body.len());
    out.extend_from_slice(&EMI_MAGIC); // 0..4
    out.extend_from_slice(&EMI_COMPILER_VER.to_le_bytes()); // 4..8
    out.extend_from_slice(&emi.interface_hash.to_le_bytes()); // 8..24
    out.extend_from_slice(&types_off.to_le_bytes()); // 24..28
    out.extend_from_slice(&(emi.types.len() as u32).to_le_bytes()); // 28..32
    out.extend_from_slice(&exports_off.to_le_bytes()); // 32..36
    out.extend_from_slice(&(emi.exports.len() as u32).to_le_bytes()); // 36..40
    out.extend_from_slice(&fn_hashes_off.to_le_bytes()); // 40..44
    out.extend_from_slice(&(emi.fn_hashes.len() as u32).to_le_bytes()); // 44..48
    out.extend_from_slice(&blobs_off.to_le_bytes()); // 48..52
    out.extend_from_slice(&(emi.blobs.len() as u32).to_le_bytes()); // 52..56
    out.extend_from_slice(&0u32.to_le_bytes()); // 56..60 flags
    out.extend_from_slice(&0u32.to_le_bytes()); // 60..64 reserved
    debug_assert_eq!(out.len(), EMI_HEADER_SIZE);
    out.extend_from_slice(&body);
    out
}

pub fn read_emi(data: &[u8]) -> Result<EmiFile, String> {
    if data.len() < EMI_HEADER_SIZE {
        return Err("E0901: truncated .emi header (< 64 bytes)".into());
    }
    if data[0..4] != EMI_MAGIC {
        return Err("E0901: invalid .emi magic (expected `EOMI`)".into());
    }
    let ver = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if ver != EMI_COMPILER_VER {
        return Err(format!("E0901: incompatible .emi version 0x{ver:08x}"));
    }
    let interface_hash = u128::from_le_bytes(data[8..24].try_into().unwrap());
    let types_off = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
    let types_cnt = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
    let exports_off = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
    let exports_cnt = u32::from_le_bytes(data[36..40].try_into().unwrap()) as usize;
    let fn_hashes_off = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
    let fn_hashes_cnt = u32::from_le_bytes(data[44..48].try_into().unwrap()) as usize;
    let blobs_off = u32::from_le_bytes(data[48..52].try_into().unwrap()) as usize;
    let blobs_cnt = u32::from_le_bytes(data[52..56].try_into().unwrap()) as usize;

    let mut cur = EMI_HEADER_SIZE;
    let module_name = read_str(data, &mut cur)?;

    let mut cur = types_off;
    let mut types = Vec::with_capacity(types_cnt);
    for _ in 0..types_cnt {
        let name = read_str(data, &mut cur)?;
        let size = read_u64(data, &mut cur)?;
        let align = read_u64(data, &mut cur)?;
        let has_niche = read_u8(data, &mut cur)? != 0;
        types.push(EmiTypeEntry { name, size, align, has_niche });
    }

    let mut cur = exports_off;
    let mut exports = Vec::with_capacity(exports_cnt);
    for _ in 0..exports_cnt {
        let name = read_str(data, &mut cur)?;
        let kind_tag = read_u8(data, &mut cur)?;
        let sig_hash = read_u128(data, &mut cur)?;
        exports.push(EmiExportEntry { name, kind_tag, sig_hash });
    }

    let mut cur = fn_hashes_off;
    let mut fn_hashes = Vec::with_capacity(fn_hashes_cnt);
    for _ in 0..fn_hashes_cnt {
        let sym_name = read_str(data, &mut cur)?;
        let body_hash = read_u128(data, &mut cur)?;
        fn_hashes.push(EmiFnHashEntry { sym_name, body_hash });
    }

    let mut cur = blobs_off;
    let mut blobs = Vec::with_capacity(blobs_cnt);
    for _ in 0..blobs_cnt {
        let sym_name = read_str(data, &mut cur)?;
        let code = read_bytes(data, &mut cur)?;
        let rc = read_u32(data, &mut cur)? as usize;
        let mut relocs = Vec::with_capacity(rc);
        for _ in 0..rc {
            let offset = read_u32(data, &mut cur)? as usize;
            let k = read_u8(data, &mut cur)?;
            let kind = match k {
                0 => RelocKind::Rel32,
                1 => RelocKind::Abs64,
                2 => RelocKind::GotPcRel32,
                3 => RelocKind::Plt32,
                4 => RelocKind::TpOff32,
                other => return Err(format!("E0901: unknown .emi relocation kind tag {other}")),
            };
            let target_sym = read_str(data, &mut cur)?;
            let addend = read_i64(data, &mut cur)?;
            relocs.push(FnReloc { offset, kind, target_sym, addend });
        }
        let roc = read_u32(data, &mut cur)? as usize;
        let mut rodata = Vec::with_capacity(roc);
        for _ in 0..roc {
            let sym = read_str(data, &mut cur)?;
            let align = read_u32(data, &mut cur)? as usize;
            let bytes = read_bytes(data, &mut cur)?;
            rodata.push(RodataItem { sym, align, bytes });
        }
        let lec = read_u32(data, &mut cur)? as usize;
        let mut line_entries = Vec::with_capacity(lec);
        for _ in 0..lec {
            let code_offset = read_u32(data, &mut cur)?;
            let file_idx = read_u32(data, &mut cur)?;
            let line = read_u32(data, &mut cur)?;
            let col = read_u32(data, &mut cur)?;
            line_entries.push(LineEntry { code_offset, file_idx, line, col });
        }
        let hc = read_u32(data, &mut cur)? as usize;
        let mut used_helpers = BTreeSet::new();
        for _ in 0..hc {
            used_helpers.insert(read_str(data, &mut cur)?);
        }
        blobs.push(CompiledFnBlob {
            sym_name,
            code,
            relocs,
            rodata,
            line_entries,
            used_helpers,
        });
    }

    Ok(EmiFile {
        module_name,
        interface_hash,
        types,
        exports,
        fn_hashes,
        blobs,
    })
}

// ============================================================================
// Per-Function Incremental Native x86_64 Build Using `.emi` Cache
// ============================================================================

fn hash_x64_options(h: &mut Hasher128, opts: &X64Options) {
    h.write_u8(u8::from(opts.sanitize));
    match &opts.mode {
        ElfMode::LinuxPie => h.write_u8(0),
        ElfMode::LinuxShared { soname } => {
            h.write_u8(1);
            h.write_u8(u8::from(soname.is_some()));
            if let Some(s) = soname {
                h.write_str(s);
            }
        }
        ElfMode::Freestanding { load_addr, multiboot2 } => {
            h.write_u8(2);
            h.write_u64(*load_addr);
            h.write_u8(u8::from(*multiboot2));
        }
    }
    h.write_u32(opts.needed_libs.len() as u32);
    for lib in &opts.needed_libs {
        h.write_str(lib);
    }
}

fn mix_u128(a: u128, b: u128) -> u128 {
    let mut h = Hasher128::new();
    h.write_bytes(&a.to_le_bytes());
    h.write_bytes(&b.to_le_bytes());
    h.finish()
}

pub fn compile_x64_incremental(
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    sm: &SourceMap,
    interner: &Interner,
    opts: &X64Options,
    cache_dir: &Path,
) -> Result<(LinkedElf, IncrementalStats), String> {
    fs::create_dir_all(cache_dir)
        .map_err(|e| format!("failed to create .emi cache dir `{}`: {e}", cache_dir.display()))?;

    let mut oh = Hasher128::new();
    hash_x64_options(&mut oh, opts);
    let opts_hash = oh.finish();
    let is_pie_or_shared = matches!(opts.mode, ElfMode::LinuxPie | ElfMode::LinuxShared { .. });

    let (source_files, file_index_map) = collect_source_files(sm);
    let mut stats = IncrementalStats::default();

    // Load prior `.emi` files per module and check Early Cutoff on `InterfaceHash`!
    let mut cached_by_mod: BTreeMap<ModuleId, BTreeMap<String, (u128, CompiledFnBlob)>> = BTreeMap::new();
    let mut new_mod_meta: BTreeMap<ModuleId, (String, u128, Vec<EmiTypeEntry>, Vec<EmiExportEntry>)> = BTreeMap::new();

    for m in &prog.modules {
        if m.unit_idx.is_none() {
            continue;
        }
        stats.modules_total += 1;
        let (raw_iface_hash, types, exports) = compute_module_interface(m.id, eir, prog, env, interner);
        let iface_hash = mix_u128(raw_iface_hash, opts_hash);
        new_mod_meta.insert(m.id, (m.full_name.clone(), iface_hash, types, exports));

        let emi_path = cache_dir.join(format!("{}.emi", sanitize_file_name(&m.full_name)));
        let mut fn_cache = BTreeMap::new();
        if let Ok(raw) = fs::read(&emi_path)
            && let Ok(old_emi) = read_emi(&raw)
        {
            if old_emi.interface_hash == iface_hash {
                stats.modules_early_cutoff += 1;
            }
            let hash_map: BTreeMap<String, u128> = old_emi
                .fn_hashes
                .into_iter()
                .map(|fh| (fh.sym_name, fh.body_hash))
                .collect();
            for blob in old_emi.blobs {
                if let Some(&bh) = hash_map.get(&blob.sym_name) {
                    fn_cache.insert(blob.sym_name.clone(), (bh, blob));
                }
            }
        }
        cached_by_mod.insert(m.id, fn_cache);
    }

    // Compile ONLY functions whose `FnBodyHash` changed; reuse cached machine-code blobs for the rest!
    let mut all_blobs: Vec<CompiledFnBlob> = Vec::new();
    let mut per_mod_hashes: BTreeMap<ModuleId, Vec<EmiFnHashEntry>> = BTreeMap::new();
    let mut per_mod_blobs: BTreeMap<ModuleId, Vec<CompiledFnBlob>> = BTreeMap::new();
    let mut dirty_mods: BTreeSet<ModuleId> = BTreeSet::new();

    for f in &eir.functions {
        if f.sig.effects.is_gpu {
            continue;
        }
        stats.fns_total += 1;
        let sym = fn_sym_name(f.key, eir, interner);
        let body_hash = mix_u128(compute_fn_body_hash(f, eir, env, interner), opts_hash);

        let reused = cached_by_mod
            .get(&f.module)
            .and_then(|mc| mc.get(&sym))
            .filter(|(old_h, _)| *old_h == body_hash)
            .map(|(_, blob)| blob.clone());

        let blob = if let Some(b) = reused {
            stats.fns_cached += 1;
            b
        } else {
            stats.fns_recompiled += 1;
            dirty_mods.insert(f.module);
            compile_function_x64(
                f,
                eir,
                prog,
                env,
                sm,
                interner,
                &file_index_map,
                opts.sanitize,
                is_pie_or_shared,
            )?
        };

        per_mod_hashes.entry(f.module).or_default().push(EmiFnHashEntry {
            sym_name: sym,
            body_hash,
        });
        per_mod_blobs.entry(f.module).or_default().push(blob.clone());
        all_blobs.push(blob);
    }

    // Persist updated `.emi` files to `cache_dir`!
    for (mid, (mname, iface_hash, types, exports)) in new_mod_meta {
        let emi_path = cache_dir.join(format!("{}.emi", sanitize_file_name(&mname)));
        if !emi_path.exists() || dirty_mods.contains(&mid) {
            let emi = EmiFile {
                module_name: mname,
                interface_hash: iface_hash,
                types,
                exports,
                fn_hashes: per_mod_hashes.remove(&mid).unwrap_or_default(),
                blobs: per_mod_blobs.remove(&mid).unwrap_or_default(),
            };
            let bytes = write_emi(&emi);
            fs::write(&emi_path, bytes)
                .map_err(|e| format!("failed to write `{}`: {e}", emi_path.display()))?;
        }
    }

    let linked = link_x64_blobs(&all_blobs, eir, prog, env, interner, source_files, opts)?;
    Ok((linked, stats))
}

fn sanitize_file_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_bytes(out: &mut Vec<u8>, bs: &[u8]) {
    out.extend_from_slice(&(bs.len() as u32).to_le_bytes());
    out.extend_from_slice(bs);
}

fn read_u8(data: &[u8], cur: &mut usize) -> Result<u8, String> {
    if *cur >= data.len() { return Err("E0901: unexpected EOF in .emi".into()); }
    let v = data[*cur];
    *cur += 1;
    Ok(v)
}

fn read_u32(data: &[u8], cur: &mut usize) -> Result<u32, String> {
    if *cur + 4 > data.len() { return Err("E0901: unexpected EOF in .emi".into()); }
    let v = u32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap());
    *cur += 4;
    Ok(v)
}

fn read_u64(data: &[u8], cur: &mut usize) -> Result<u64, String> {
    if *cur + 8 > data.len() { return Err("E0901: unexpected EOF in .emi".into()); }
    let v = u64::from_le_bytes(data[*cur..*cur + 8].try_into().unwrap());
    *cur += 8;
    Ok(v)
}

fn read_i64(data: &[u8], cur: &mut usize) -> Result<i64, String> {
    Ok(read_u64(data, cur)? as i64)
}

fn read_u128(data: &[u8], cur: &mut usize) -> Result<u128, String> {
    if *cur + 16 > data.len() { return Err("E0901: unexpected EOF in .emi".into()); }
    let v = u128::from_le_bytes(data[*cur..*cur + 16].try_into().unwrap());
    *cur += 16;
    Ok(v)
}

fn read_bytes(data: &[u8], cur: &mut usize) -> Result<Vec<u8>, String> {
    let n = read_u32(data, cur)? as usize;
    if *cur + n > data.len() { return Err("E0901: unexpected EOF in .emi".into()); }
    let v = data[*cur..*cur + n].to_vec();
    *cur += n;
    Ok(v)
}

fn read_str(data: &[u8], cur: &mut usize) -> Result<String, String> {
    let bs = read_bytes(data, cur)?;
    String::from_utf8(bs).map_err(|_| "E0901: invalid UTF-8 in .emi".into())
}

#[allow(dead_code)]
fn _ty_ref(_t: &Ty) {}
