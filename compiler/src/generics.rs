use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::CfgFnKey;
use crate::codegen_x64::{OptNiche, opt_niche, x64_layout};
use crate::eir::{EirCallee, EirFunction, EirInstKind, EirProgram, LayoutSharedGroup};
use crate::types::{GenericVal, Ty, TyId, TypeEnv};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LayoutKey {
    base_key: CfgFnKey,
    arg_layouts: Vec<(u64, u64, bool, OptNiche, u8)>,
}

/// Phase 4 Layout Sharing pass:
/// Groups instantiated generic functions by physical layout `(size_bytes, align_bytes, is_linear)`
/// unless the function requires hot specialization (`@hot`, `@inline`, `@gpu`, const generics, or
/// type-specific arithmetic/method dispatch on the generic parameter).
///
/// Deduplicated instances are removed from `eir.functions` and all call sites are rewritten to the
/// single canonical shared function (`__ls_<size>_<align>`).
pub fn apply_layout_sharing(eir: &mut EirProgram, env: &TypeEnv) {
    let mut groups: BTreeMap<LayoutKey, Vec<usize>> = BTreeMap::new();

    for (fi, f) in eir.functions.iter().enumerate() {
        let Some((base_key, gargs)) = &f.base_generic else {
            continue;
        };
        if !is_layout_shareable(f, gargs, env) {
            continue;
        }
        let mut arg_layouts = Vec::with_capacity(gargs.len());
        let mut all_ty = true;
        for ga in gargs {
            match ga {
                GenericVal::Ty(t) => {
                    let lay = x64_layout(env, *t);
                    let is_lin = env.is_linear(*t);
                    let niche = opt_niche(env, *t);
                    // Only distinguish float ABI (XMM vs GPR) when the signature passes/returns a bare `f32`/`f64`
                    // (not pointers like `swap(a: *mut f64, b: *mut f64)` which share `ls_8_8` with `u64`/`i64`).
                    let float_abi = if sig_uses_bare_ty(&f.sig, *t, env) {
                        match env.get(*t) {
                            Ty::Float(32) => 32,
                            Ty::Float(64) => 64,
                            _ => 0,
                        }
                    } else {
                        0
                    };
                    arg_layouts.push((lay.size as u64, lay.align as u64, is_lin, niche, float_abi));
                }
                _ => {
                    all_ty = false;
                    break;
                }
            }
        }
        if !all_ty || arg_layouts.is_empty() {
            continue;
        }
        let lk = LayoutKey {
            base_key: *base_key,
            arg_layouts,
        };
        groups.entry(lk).or_default().push(fi);
    }

    // Build remap from duplicate GenericInst(dup_id) -> canonical GenericInst(canon_id).
    let mut inst_remap: BTreeMap<usize, usize> = BTreeMap::new();
    let mut drop_indices: BTreeSet<usize> = BTreeSet::new();
    let mut recorded_groups: Vec<LayoutSharedGroup> = Vec::new();

    for (lk, members) in &groups {
        let canon_idx = members[0];
        let CfgFnKey::GenericInst(canon_inst_id) = eir.functions[canon_idx].key else {
            continue;
        };
        let tag = format_layout_tag(&lk.arg_layouts);
        // Tag every shareable group so codegen emits `__ls_<size>_<align>` even for 1-member groups
        // when shared across multiple types (or always for layout-shared generics).
        if members.len() >= 2 {
            eir.functions[canon_idx].layout_shared_tag = Some(tag.clone());
            let mut member_insts = Vec::with_capacity(members.len());
            for &mi in members {
                if let Some((_, gargs)) = &eir.functions[mi].base_generic {
                    member_insts.push((eir.functions[mi].key, gargs.clone()));
                }
                if mi != canon_idx {
                    if let CfgFnKey::GenericInst(dup_id) = eir.functions[mi].key {
                        inst_remap.insert(dup_id, canon_inst_id);
                    }
                    drop_indices.insert(mi);
                }
            }
            recorded_groups.push(LayoutSharedGroup {
                canonical_key: CfgFnKey::GenericInst(canon_inst_id),
                layout_tag: tag,
                member_insts,
            });
        }
    }

    if drop_indices.is_empty() {
        return;
    }

    // Rewrite all call sites pointing to deduplicated GenericInst IDs to point to canonical ID.
    for f in &mut eir.functions {
        for blk in &mut f.blocks {
            for inst in &mut blk.insts {
                if let EirInstKind::Call {
                    callee: EirCallee::GenericInst { inst_id },
                    ..
                } = &mut inst.kind
                    && let Some(&canon_id) = inst_remap.get(inst_id)
                {
                    *inst_id = canon_id;
                }
            }
        }
    }

    // Compact `eir.functions` removing duplicate instances, and rebuild `eir.by_key`.
    let old_fns = std::mem::take(&mut eir.functions);
    let mut new_by_key: BTreeMap<CfgFnKey, usize> = BTreeMap::new();
    for (old_idx, f) in old_fns.into_iter().enumerate() {
        if drop_indices.contains(&old_idx) {
            continue;
        }
        let new_idx = eir.functions.len();
        new_by_key.insert(f.key, new_idx);
        eir.functions.push(f);
    }
    // Map remapped duplicate keys to their canonical function index too.
    for (&dup_id, &canon_id) in &inst_remap {
        if let Some(&cidx) = new_by_key.get(&CfgFnKey::GenericInst(canon_id)) {
            new_by_key.insert(CfgFnKey::GenericInst(dup_id), cidx);
        }
    }
    eir.by_key = new_by_key;
    eir.layout_shared_groups = recorded_groups;
}

fn sig_uses_bare_ty(sig: &crate::types::FnSig, t: TyId, env: &TypeEnv) -> bool {
    let check_ty = |ty: TyId| -> bool {
        if ty == t {
            return true;
        }
        match env.get(ty) {
            Ty::ErrorUnion { ok, .. } => *ok == t,
            _ => false,
        }
    };
    check_ty(sig.ret) || sig.params.iter().any(|p| check_ty(p.ty))
}

fn format_layout_tag(layouts: &[(u64, u64, bool, OptNiche, u8)]) -> String {
    let parts: Vec<String> = layouts
        .iter()
        .map(|&(sz, al, lin, niche, fabi)| {
            let mut s = format!("{sz}_{al}");
            if lin {
                s.push_str("_lin");
            }
            if niche != OptNiche::None {
                let ncode = match niche {
                    OptNiche::None => 0u8,
                    OptNiche::IdBrand => 1,
                    OptNiche::GenId => 2,
                    OptNiche::Link => 3,
                    OptNiche::NullPtr => 4,
                    OptNiche::FatPtrNullFirst => 5,
                    OptNiche::SliceMaxLen => 6,
                    OptNiche::BoolTag => 7,
                    OptNiche::ScalarSentinel { .. } => 8,
                };
                s.push_str(&format!("_n{ncode}"));
                if let OptNiche::ScalarSentinel { size, sentinel } = niche {
                    s.push_str(&format!("_{size}_{sentinel:x}"));
                }
            }
            if fabi != 0 {
                s.push_str(&format!("_f{fabi}"));
            }
            s
        })
        .collect();
    format!("ls_{}", parts.join("__"))
}

fn is_layout_shareable(f: &EirFunction, gargs: &[GenericVal], env: &TypeEnv) -> bool {
    // Hot specialization for `@hot`, `@inline`, `@gpu`.
    if f.sig.effects.is_hot || f.sig.effects.is_inline || f.sig.effects.is_gpu {
        return false;
    }
    // Const generics specialize per constant value.
    if gargs.iter().any(|g| !matches!(g, GenericVal::Ty(_))) {
        return false;
    }
    let concrete_tys: BTreeSet<TyId> = gargs
        .iter()
        .filter_map(|g| match g {
            GenericVal::Ty(t) => Some(*t),
            _ => None,
        })
        .collect();
    if concrete_tys.is_empty() {
        return false;
    }

    // Check if the body performs type-specific numeric arithmetic, casts, or direct method calls
    // on values of the concrete generic type (if so, monomorphize for full speed / semantics!).
    for blk in &f.blocks {
        for inst in &blk.insts {
            match &inst.kind {
                EirInstKind::Binary { op_ty, dst_ty, .. } => {
                    if concrete_tys.contains(op_ty) || concrete_tys.contains(dst_ty) {
                        return false;
                    }
                }
                EirInstKind::Unary { ty, .. } => {
                    if concrete_tys.contains(ty) {
                        return false;
                    }
                }
                EirInstKind::Cast { from_ty, to_ty, .. } => {
                    if concrete_tys.contains(from_ty) || concrete_tys.contains(to_ty) {
                        return false;
                    }
                }
                EirInstKind::Call { callee, args, .. }
                    if !matches!(callee, EirCallee::DynCall { .. })
                        && args.iter().any(|&v| {
                            let vt = f.value_tys[v.0 as usize];
                            ty_mentions_any(vt, &concrete_tys, env)
                        }) =>
                {
                    return false;
                }
                _ => {}
            }
        }
    }
    true
}

fn ty_mentions_any(ty: TyId, set: &BTreeSet<TyId>, env: &TypeEnv) -> bool {
    if set.contains(&ty) {
        return true;
    }
    match env.get(ty) {
        Ty::Borrow { inner: crate::types::BorrowInner::Single(t) | crate::types::BorrowInner::Slice(t), .. }
        | Ty::Optional(t)
        | Ty::RawPtr { inner: t, .. }
        | Ty::Array { elem: t, .. } => ty_mentions_any(*t, set, env),
        _ => false,
    }
}
