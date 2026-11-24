//! Reference `EIR` Interpreter (`eonc run --interp`) — zero-C semantic oracle.
//!
//! Executes verified SSA `EIR` directly over a strict byte-addressed virtual memory model:
//! - Every allocation (stack slot, `Arena` buffer, `Buf` backing store, `ArenaPool`/`SlotPool`
//!   columns, string literals, `AllocCap`) lives in its own `MemBlock` with a unique `AllocId`.
//! - Every load/store verifies non-null provenance, liveness (use-after-free detection),
//!   exact byte bounds (1-byte OOB detection), and read-only protection.
//! - Proven arithmetic (`+`, `-`, `*`, `/`, `%`, `<<`, `>>`) checks at runtime that no overflow,
//!   division by zero, or shift >= bitwidth occurs — acting as a runtime soundness oracle for the
//!   static DBM verifier.
//! - Uses the exact same type layouts as `codegen_x64` so `bitcast`, `union`, `Optional`,
//!   `ErrorUnion`, `ArenaPool`, `SlotPool`, and `dyn` behave identically across all backends.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use crate::ast::{BinaryOp, UnaryOp};
use crate::build_sys::{BuildPlan, BuildTarget, BuildTargetKind};
use crate::cfg::{AggregateKind, BlockId, CastKind, CfgBuiltinOp, CfgConst, CfgFnKey};
use crate::codegen_x64::{
    eu_ok_offset, int_bits, is_niche_id_opt, is_signed_int, opt_niche, opt_val_offset,
    pool_col_count, struct_field_offset, union_payload_offset, x64_layout, x64_layout_impl,
    OptNiche,
};
use crate::eir::{
    EirCallee, EirFunction, EirInst, EirInstKind, EirPlace, EirPlaceBase, EirProgram, EirProj,
    EirSlot, EirTerm, EirValue,
};
use crate::resolve::{BuiltinKind, DefId, ResolvedProgram};
use crate::span::Interner;
use crate::types::{BorrowInner, MethodTarget, Ty, TyId, TypeEnv};

pub const DEFAULT_STEP_LIMIT: u64 = 50_000_000;

#[derive(Clone, Debug)]
pub struct InterpResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub steps: u64,
}

#[derive(Clone, Debug)]
struct MemBlock {
    bytes: Vec<u8>,
    alive: bool,
    readonly: bool,
}

struct VmMem {
    blocks: Vec<MemBlock>, // index 0 unused; valid AllocId = 1..
}

impl VmMem {
    fn new() -> Self {
        Self {
            blocks: vec![MemBlock {
                bytes: Vec::new(),
                alive: false,
                readonly: true,
            }],
        }
    }

    #[inline]
    fn encode_ptr(alloc_id: u32, offset: u32) -> u64 {
        ((alloc_id as u64) << 32) | (offset as u64)
    }

    #[inline]
    fn decode_ptr(ptr: u64) -> (u32, usize) {
        ((ptr >> 32) as u32, (ptr & 0xFFFF_FFFF) as usize)
    }

    fn alloc(&mut self, size: usize, readonly: bool) -> Result<u64, String> {
        if size > (1 << 30) {
            return Err(format!("EIR-Interp: allocation too large ({size} bytes)"));
        }
        let id = u32::try_from(self.blocks.len())
            .map_err(|_| "EIR-Interp: allocation table exhausted".to_owned())?;
        self.blocks.push(MemBlock {
            bytes: vec![0u8; size],
            alive: true,
            readonly,
        });
        Ok(Self::encode_ptr(id, 0))
    }

    fn alloc_bytes(&mut self, data: &[u8], readonly: bool) -> Result<u64, String> {
        let id = u32::try_from(self.blocks.len())
            .map_err(|_| "EIR-Interp: allocation table exhausted".to_owned())?;
        self.blocks.push(MemBlock {
            bytes: data.to_vec(),
            alive: true,
            readonly,
        });
        Ok(Self::encode_ptr(id, 0))
    }

    fn free(&mut self, ptr: u64) -> Result<(), String> {
        if ptr == 0 {
            return Ok(());
        }
        let (id, off) = Self::decode_ptr(ptr);
        if off != 0 {
            return Err(format!("EIR-Interp: freeing interior pointer 0x{ptr:x}"));
        }
        let Some(blk) = self.blocks.get_mut(id as usize) else {
            return Err(format!("EIR-Interp: freeing invalid allocation id {id}"));
        };
        if !blk.alive {
            return Err(format!("EIR-Interp: double-free / use-after-free of allocation {id}"));
        }
        blk.alive = false;
        // Poison freed memory to catch any stale reads immediately.
        blk.bytes.fill(0xDE);
        Ok(())
    }

    fn block_len(&self, ptr: u64) -> Result<usize, String> {
        let (id, off) = Self::decode_ptr(ptr);
        let Some(blk) = self.blocks.get(id as usize) else {
            return Err(format!("EIR-Interp: invalid pointer 0x{ptr:x}"));
        };
        if !blk.alive {
            return Err(format!("EIR-Interp: use-after-free on allocation {id}"));
        }
        Ok(blk.bytes.len().saturating_sub(off))
    }

    fn read(&self, ptr: u64, size: usize) -> Result<&[u8], String> {
        if size == 0 {
            return Ok(&[]);
        }
        if ptr == 0 {
            return Err("EIR-Interp: null pointer dereference on read".to_owned());
        }
        let (id, off) = Self::decode_ptr(ptr);
        let Some(blk) = self.blocks.get(id as usize) else {
            return Err(format!("EIR-Interp: read from invalid allocation {id}"));
        };
        if !blk.alive {
            return Err(format!("EIR-Interp: use-after-free read on allocation {id}"));
        }
        let end = off
            .checked_add(size)
            .ok_or_else(|| "EIR-Interp: pointer offset overflow on read".to_owned())?;
        if end > blk.bytes.len() {
            return Err(format!(
                "EIR-Interp: out-of-bounds read [{}..{}] on allocation {} (len {})",
                off,
                end,
                id,
                blk.bytes.len()
            ));
        }
        Ok(&blk.bytes[off..end])
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        if ptr == 0 {
            return Err("EIR-Interp: null pointer dereference on write".to_owned());
        }
        let (id, off) = Self::decode_ptr(ptr);
        let Some(blk) = self.blocks.get_mut(id as usize) else {
            return Err(format!("EIR-Interp: write to invalid allocation {id}"));
        };
        if !blk.alive {
            return Err(format!("EIR-Interp: use-after-free write on allocation {id}"));
        }
        if blk.readonly {
            return Err(format!("EIR-Interp: write to read-only allocation {id}"));
        }
        let end = off
            .checked_add(data.len())
            .ok_or_else(|| "EIR-Interp: pointer offset overflow on write".to_owned())?;
        if end > blk.bytes.len() {
            return Err(format!(
                "EIR-Interp: out-of-bounds write [{}..{}] on allocation {} (len {})",
                off,
                end,
                id,
                blk.bytes.len()
            ));
        }
        blk.bytes[off..end].copy_from_slice(data);
        Ok(())
    }

    fn ptr_add(ptr: u64, delta: usize) -> Result<u64, String> {
        if delta == 0 {
            return Ok(ptr);
        }
        if ptr == 0 {
            return Err("EIR-Interp: pointer arithmetic on null pointer".to_owned());
        }
        let (id, off) = Self::decode_ptr(ptr);
        let new_off = off
            .checked_add(delta)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| "EIR-Interp: pointer offset overflow".to_owned())?;
        Ok(Self::encode_ptr(id, new_off))
    }

    fn read_u64(&self, ptr: u64) -> Result<u64, String> {
        let b = self.read(ptr, 8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u32(&self, ptr: u64) -> Result<u32, String> {
        let b = self.read(ptr, 4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn write_u64(&mut self, ptr: u64, v: u64) -> Result<(), String> {
        self.write(ptr, &v.to_le_bytes())
    }

    fn write_u32(&mut self, ptr: u64, v: u32) -> Result<(), String> {
        self.write(ptr, &v.to_le_bytes())
    }

    fn collect_cap_chain(&self, mut cur: u64) -> Result<Vec<u64>, String> {
        let mut chain = Vec::new();
        while cur != 0 {
            if chain.len() >= 64 || chain.contains(&cur) {
                return Err("EIR-Interp: AllocCap parent chain cycle or depth exceeded".to_owned());
            }
            chain.push(cur);
            cur = self.read_u64(Self::ptr_add(cur, 32)?)?;
        }
        Ok(chain)
    }

    fn alloc_raw_tracked(&mut self, cap_ptr: u64, req_size: usize) -> Result<Option<u64>, String> {
        let delta = req_size as u64;
        let chain = self.collect_cap_chain(cap_ptr)?;
        for &node in &chain {
            let used = self.read_u64(Self::ptr_add(node, 8)?)?;
            let limit = self.read_u64(Self::ptr_add(node, 24)?)?;
            let Some(new_used) = used.checked_add(delta) else {
                return Ok(None);
            };
            if limit != 0 && new_used > limit {
                return Ok(None);
            }
        }
        let actual = req_size.max(1);
        let ptr = self.alloc(actual, false)?;
        for node in chain {
            let live = self.read_u64(node)?;
            let used = self.read_u64(Self::ptr_add(node, 8)?)?;
            let peak = self.read_u64(Self::ptr_add(node, 16)?)?;
            let new_used = used + delta;
            self.write_u64(node, live + 1)?;
            self.write_u64(Self::ptr_add(node, 8)?, new_used)?;
            if new_used > peak {
                self.write_u64(Self::ptr_add(node, 16)?, new_used)?;
            }
        }
        Ok(Some(ptr))
    }

    fn free_raw_tracked(&mut self, cap_ptr: u64, ptr: u64, req_size: usize) -> Result<(), String> {
        if ptr == 0 {
            return Ok(());
        }
        self.free(ptr)?;
        let chain = self.collect_cap_chain(cap_ptr)?;
        for node in chain {
            let live = self.read_u64(node)?;
            let used = self.read_u64(Self::ptr_add(node, 8)?)?;
            self.write_u64(node, live.saturating_sub(1))?;
            self.write_u64(Self::ptr_add(node, 8)?, used.saturating_sub(req_size as u64))?;
        }
        Ok(())
    }
}

/// Compact runtime value representation for SSA registers.
#[derive(Clone, Debug)]
enum IVal {
    Void,
    Scalar(u64),
    Bytes(Vec<u8>),
}

impl IVal {
    fn from_bytes(b: &[u8]) -> Self {
        match b.len() {
            0 => IVal::Void,
            1..=8 => {
                let mut buf = [0u8; 8];
                buf[..b.len()].copy_from_slice(b);
                IVal::Scalar(u64::from_le_bytes(buf))
            }
            _ => IVal::Bytes(b.to_vec()),
        }
    }

    fn to_bytes(&self, size: usize) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        match self {
            IVal::Void => vec![0u8; size],
            IVal::Scalar(v) => {
                let raw = v.to_le_bytes();
                let mut out = vec![0u8; size];
                let n = size.min(8);
                out[..n].copy_from_slice(&raw[..n]);
                out
            }
            IVal::Bytes(b) => {
                let mut out = vec![0u8; size];
                let n = size.min(b.len());
                out[..n].copy_from_slice(&b[..n]);
                out
            }
        }
    }

    fn as_u64(&self) -> u64 {
        match self {
            IVal::Void => 0,
            IVal::Scalar(v) => *v,
            IVal::Bytes(b) => {
                let mut buf = [0u8; 8];
                let n = b.len().min(8);
                buf[..n].copy_from_slice(&b[..n]);
                u64::from_le_bytes(buf)
            }
        }
    }

    fn as_pair(&self) -> (u64, u64) {
        let b = self.to_bytes(16);
        let lo = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let hi = u64::from_le_bytes(b[8..16].try_into().unwrap());
        (lo, hi)
    }

    fn from_pair(a: u64, b: u64) -> Self {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
        IVal::Bytes(out)
    }
}

fn mask_bits(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else if bits == 0 {
        0
    } else {
        (1u64 << bits) - 1
    }
}

fn sign_extend_i128(raw: u64, bits: u8, signed: bool) -> i128 {
    let masked = raw & mask_bits(bits);
    if !signed || bits == 0 {
        masked as i128
    } else if bits >= 64 {
        (masked as i64) as i128
    } else {
        let shift = 64 - bits;
        (((masked << shift) as i64) >> shift) as i128
    }
}

fn int_bounds_for(bits: u8, signed: bool) -> (i128, i128) {
    if signed {
        let hi = (1i128 << (bits.saturating_sub(1))) - 1;
        let lo = -(1i128 << (bits.saturating_sub(1)));
        (lo, hi)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

/// IEEE-754 binary16 -> f32 conversion (software, exact).
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal binary16 -> normal f32
            let mut f = frac;
            let mut e: i32 = 127 - 14;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03FF;
            sign | ((e as u32) << 23) | (f << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (frac << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// IEEE-754 f32 -> binary16 conversion (round-to-nearest-even).
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 0xFF {
        // Inf or NaN
        return sign | 0x7C00 | if frac != 0 { ((frac >> 13) as u16).max(1) } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow -> Inf
    }
    if unbiased < -24 {
        return sign; // underflow -> signed zero
    }
    if unbiased < -14 {
        // Subnormal binary16
        let full = frac | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let mant = full >> shift;
        let rem = full & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round = if rem > half || (rem == half && (mant & 1) != 0) { 1 } else { 0 };
        return sign | ((mant + round) as u16);
    }
    let h_exp = ((unbiased + 15) as u32) << 10;
    let mant = frac >> 13;
    let rem = frac & 0x1FFF;
    let round = if rem > 0x1000 || (rem == 0x1000 && (mant & 1) != 0) { 1 } else { 0 };
    sign | ((h_exp + mant + round) as u16)
}

fn interp_ty_size(env: &TypeEnv, ty: TyId) -> usize {
    match env.get(ty) {
        Ty::Builtin(BuiltinKind::AllocCap, _) => 40,
        Ty::Builtin(BuiltinKind::Env, _) => 88,
        Ty::Builtin(BuiltinKind::ArgsCap | BuiltinKind::BuildCap, _) => 16,
        Ty::Builtin(
            BuiltinKind::FsCap
            | BuiltinKind::DirCap
            | BuiltinKind::NetCap
            | BuiltinKind::ClockCap
            | BuiltinKind::IoCap,
            _,
        ) => 8,
        _ => x64_layout(env, ty).size,
    }
}

fn io_err_code(err: &std::io::Error) -> u32 {
    match err.kind() {
        std::io::ErrorKind::NotFound => 12,
        std::io::ErrorKind::PermissionDenied => 13,
        std::io::ErrorKind::AlreadyExists => 15,
        std::io::ErrorKind::InvalidInput => 16,
        _ => 17,
    }
}

fn validate_build_rel_path(rel: &str, label: &str) -> Result<(), String> {
    if rel.is_empty() {
        return Err(format!("build.eon sandbox error: {label} cannot be empty"));
    }
    if rel.contains('\0') {
        return Err(format!("build.eon sandbox error: {label} contains NUL byte"));
    }
    if rel.starts_with('/') || rel.starts_with('\\') || Path::new(rel).is_absolute() {
        return Err(format!("build.eon sandbox error: absolute path `{rel}` is forbidden"));
    }
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!(
                    "build.eon sandbox error: path `{rel}` escapes package root via `..`"
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("build.eon sandbox error: non-relative path `{rel}` is forbidden"));
            }
        }
    }
    Ok(())
}

fn resolve_confined_path(root: &Path, rel: &str, must_exist: bool) -> Result<PathBuf, u32> {
    if rel.is_empty() {
        return Err(16);
    }
    if rel.contains('\0') || rel.starts_with('/') || rel.starts_with('\\') || Path::new(rel).is_absolute() {
        return Err(14);
    }
    let mut depth: isize = 0;
    for comp in Path::new(rel).components() {
        match comp {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(14);
                }
            }
            Component::RootDir | Component::Prefix(_) => return Err(14),
        }
    }

    let canon_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let joined = canon_root.join(Path::new(rel));
    match fs::canonicalize(&joined) {
        Ok(canon_target) => {
            if canon_target.starts_with(&canon_root) {
                Ok(canon_target)
            } else {
                Err(14)
            }
        }
        Err(err) => {
            let mut probe = joined.as_path();
            let mut tail: Vec<std::ffi::OsString> = Vec::new();
            let canon_anc = loop {
                match fs::canonicalize(probe) {
                    Ok(c) => break c,
                    Err(_) => {
                        if let Ok(meta) = fs::symlink_metadata(probe)
                            && meta.file_type().is_symlink()
                        {
                            if !must_exist {
                                return Err(14);
                            }
                            if let Ok(link_target) = fs::read_link(probe) {
                                if link_target.is_absolute() {
                                    return Err(14);
                                }
                                let parent_dir = probe.parent().unwrap_or(&canon_root);
                                let resolved_parent = fs::canonicalize(parent_dir)
                                    .unwrap_or_else(|_| parent_dir.to_path_buf());
                                let Ok(rel_from_root) = resolved_parent.strip_prefix(&canon_root) else {
                                    return Err(14);
                                };
                                let mut depth: isize = rel_from_root
                                    .components()
                                    .filter(|c| matches!(c, Component::Normal(_)))
                                    .count() as isize;
                                for c in link_target.components() {
                                    match c {
                                        Component::Normal(_) => depth += 1,
                                        Component::ParentDir => {
                                            depth -= 1;
                                            if depth < 0 {
                                                return Err(14);
                                            }
                                        }
                                        Component::CurDir => {}
                                        Component::RootDir | Component::Prefix(_) => return Err(14),
                                    }
                                }
                            }
                            return Err(io_err_code(&err));
                        }
                        let Some(name) = probe.file_name() else {
                            return Err(14);
                        };
                        tail.push(name.to_os_string());
                        probe = probe.parent().unwrap_or_else(|| Path::new(""));
                    }
                }
            };
            if !canon_anc.starts_with(&canon_root) {
                return Err(14);
            }
            if must_exist {
                return Err(io_err_code(&err));
            }
            let mut resolved = canon_anc;
            for seg in tail.into_iter().rev() {
                if seg == ".." {
                    return Err(14);
                }
                resolved.push(seg);
            }
            Ok(resolved)
        }
    }
}

struct Interp<'a> {
    eir: &'a EirProgram,
    _prog: &'a ResolvedProgram,
    env: &'a TypeEnv,
    interner: &'a Interner,
    mem: VmMem,
    vtables: BTreeMap<(DefId, TyId), u64>,
    vtable_methods: Vec<Vec<CfgFnKey>>,
    fn_handles: BTreeMap<CfgFnKey, u64>,
    handle_to_fn: Vec<CfgFnKey>,
    str_cache: BTreeMap<Vec<u8>, u64>,
    stdout: String,
    stderr: String,
    steps: u64,
    step_limit: u64,
    clock_tick: u64,
    start_instant: Instant,
    last_clock_ns: u64,
    dir_roots: Vec<Option<PathBuf>>,
    net_hosts: Vec<String>,
    build_plan: BuildPlan,
    depth: usize,
}

impl<'a> Interp<'a> {
    fn new(
        eir: &'a EirProgram,
        prog: &'a ResolvedProgram,
        env: &'a TypeEnv,
        interner: &'a Interner,
        step_limit: u64,
    ) -> Self {
        let cwd_root = fs::canonicalize(".")
            .or_else(|_| std::env::current_dir())
            .unwrap_or_else(|_| PathBuf::from("."));
        let mut interp = Self {
            eir,
            _prog: prog,
            env,
            interner,
            mem: VmMem::new(),
            vtables: BTreeMap::new(),
            vtable_methods: vec![Vec::new()], // index 0 unused
            fn_handles: BTreeMap::new(),
            handle_to_fn: vec![CfgFnKey::TopDef(DefId(0))], // index 0 unused
            str_cache: BTreeMap::new(),
            stdout: String::new(),
            stderr: String::new(),
            steps: 0,
            step_limit,
            clock_tick: 1_000_000,
            start_instant: Instant::now(),
            last_clock_ns: 0,
            dir_roots: vec![None, Some(cwd_root)], // 0 = invalid, 1 = FsCap root
            net_hosts: vec![String::new()],       // 0 = unrestricted
            build_plan: BuildPlan::default(),
            depth: 0,
        };
        interp.init_vtables();
        interp
    }

    fn init_vtables(&mut self) {
        for &(iface_def, concrete_ty) in &self.eir.dyn_impl_pairs {
            let Some(imeta) = self.env.ifaces.get(&iface_def) else { continue; };
            let target = match self.env.get(concrete_ty) {
                Ty::Struct { def, .. }
                | Ty::Pool { def, .. }
                | Ty::Enum(def)
                | Ty::Union { def, .. }
                | Ty::Bitfield(def) => Some(MethodTarget::Def(*def)),
                Ty::Builtin(bk, _) => Some(MethodTarget::Builtin(*bk)),
                _ => None,
            };
            let Some(tk) = target else { continue; };
            let Some((impl_idx, im_ref)) = self
                .env
                .impls
                .iter()
                .enumerate()
                .find(|(_, im)| im.target == tk && im.iface == Some(iface_def))
            else {
                continue;
            };
            let mut methods = Vec::with_capacity(imeta.methods.len());
            for m in &imeta.methods {
                let sub_idx = im_ref
                    .methods
                    .iter()
                    .position(|im_m| im_m.2.name == m.name)
                    .unwrap_or(0);
                methods.push(CfgFnKey::ImplMethod { impl_idx, sub_idx });
            }
            let vt_id = self.vtable_methods.len() as u64;
            self.vtable_methods.push(methods);
            self.vtables.insert((iface_def, concrete_ty), vt_id);
        }
    }

    fn intern_ro_bytes(&mut self, bytes: &[u8]) -> Result<u64, String> {
        if let Some(&p) = self.str_cache.get(bytes) {
            return Ok(p);
        }
        let p = self.mem.alloc_bytes(bytes, true)?;
        self.str_cache.insert(bytes.to_vec(), p);
        Ok(p)
    }

    fn intern_fn_handle(&mut self, key: CfgFnKey) -> u64 {
        if let Some(&h) = self.fn_handles.get(&key) {
            return h;
        }
        let h = self.handle_to_fn.len() as u64;
        self.handle_to_fn.push(key);
        self.fn_handles.insert(key, h);
        h
    }

    fn call_key(&mut self, key: CfgFnKey, args: Vec<IVal>) -> Result<IVal, String> {
        let Some(&idx) = self.eir.by_key.get(&key) else {
            return Err(format!("EIR-Interp: unresolved function `{key:?}`"));
        };
        let f = &self.eir.functions[idx];
        self.exec_fn(f, args)
    }

    fn exec_fn(&mut self, f: &'a EirFunction, args: Vec<IVal>) -> Result<IVal, String> {
        if self.depth >= 512 {
            return Err(format!(
                "EIR-Interp: call stack overflow (depth >= 512) entering `{}`",
                self.interner.get(f.name)
            ));
        }
        self.depth += 1;
        let res = self.exec_fn_inner(f, args);
        self.depth -= 1;
        res
    }

    fn exec_fn_inner(&mut self, f: &'a EirFunction, args: Vec<IVal>) -> Result<IVal, String> {
        let mut vals: Vec<IVal> = vec![IVal::Void; f.value_tys.len()];
        for (&pv, av) in f.params.iter().zip(args) {
            vals[pv.0 as usize] = av;
        }

        // Allocate a private virtual memory block per stack slot; freed on function return!
        let mut slot_ptrs: Vec<u64> = Vec::with_capacity(f.slots.len());
        for s in &f.slots {
            let sz = interp_ty_size(self.env, s.ty).max(1);
            let p = self.mem.alloc(sz, false)?;
            slot_ptrs.push(p);
        }

        let exec_res = (|| -> Result<IVal, String> {
            let mut cur_bb = f.entry;
            let mut prev_bb: Option<BlockId> = None;

            loop {
                let b = f.block(cur_bb);

                // Resolve parallel phi nodes using `prev_bb`!
                if let Some(pbb) = prev_bb {
                    let mut phi_updates: Vec<(EirValue, IVal)> = Vec::with_capacity(b.phis.len());
                    for phi in &b.phis {
                        let (_, src_v) = phi
                            .incoming
                            .iter()
                            .find(|&&(bb, _)| bb == pbb)
                            .ok_or_else(|| {
                                format!(
                                    "EIR-Interp: missing phi incoming edge {:?} -> {:?} in `{}`",
                                    pbb,
                                    cur_bb,
                                    self.interner.get(f.name)
                                )
                            })?;
                        phi_updates.push((phi.dst, vals[src_v.0 as usize].clone()));
                    }
                    for (dst, v) in phi_updates {
                        vals[dst.0 as usize] = v;
                    }
                }

                for inst in &b.insts {
                    self.steps += 1;
                    if self.steps > self.step_limit {
                        return Err(format!(
                            "EIR-Interp: step limit ({}) exceeded in `{}`",
                            self.step_limit,
                            self.interner.get(f.name)
                        ));
                    }
                    self.exec_inst(f, inst, &mut vals, &slot_ptrs)?;
                }

                self.steps += 1;
                prev_bb = Some(cur_bb);
                match &b.term {
                    EirTerm::Goto(t) => cur_bb = *t,
                    EirTerm::Branch { cond, then_bb, else_bb, .. } => {
                        let c = vals[cond.0 as usize].as_u64() != 0;
                        cur_bb = if c { *then_bb } else { *else_bb };
                    }
                    EirTerm::SwitchInt { discr, targets, otherwise } => {
                        let dty = f.value_tys[discr.0 as usize];
                        let signed = is_signed_int(self.env, dty);
                        let bits = int_bits(self.env, dty);
                        let dv = sign_extend_i128(vals[discr.0 as usize].as_u64(), bits, signed);
                        cur_bb = targets
                            .iter()
                            .find(|&&(c, _)| c == dv)
                            .map_or(*otherwise, |&(_, bb)| bb);
                    }
                    EirTerm::Return(rv) => {
                        return Ok(match rv {
                            Some(v) => vals[v.0 as usize].clone(),
                            None => IVal::Void,
                        });
                    }
                    EirTerm::Unreachable { .. } => {
                        return Err(format!(
                            "EIR-Interp: reached `unreachable` in `{}`",
                            self.interner.get(f.name)
                        ));
                    }
                }
            }
        })();

        // Always free stack slots on return to detect dangling pointers!
        for p in slot_ptrs {
            let _ = self.mem.free(p);
        }
        exec_res
    }

    /// Evaluate an `EirPlace` to a virtual memory address, or return a materialized PoolCol slice.
    fn eval_place_addr(
        &mut self,
        f: &EirFunction,
        place: &EirPlace,
        vals: &[IVal],
        slot_ptrs: &[u64],
    ) -> Result<PlaceTarget, String> {
        // Temporary allocation if base is an rvalue `EirValue` rather than a slot.
        let (mut addr, mut cur_ty, temp_alloc) = match place.base {
            EirPlaceBase::Slot(s) => (slot_ptrs[s.0 as usize], f.slots[s.0 as usize].ty, None),
            EirPlaceBase::Val(v) => {
                let vty = f.value_tys[v.0 as usize];
                let sz = interp_ty_size(self.env, vty).max(1);
                let p = self.mem.alloc_bytes(&vals[v.0 as usize].to_bytes(sz), false)?;
                (p, vty, Some(p))
            }
        };

        let mut i = 0usize;
        while i < place.projs.len() {
            match &place.projs[i] {
                EirProj::Deref => {
                    let inner = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if !matches!(self.env.get(inner), Ty::Pool { .. }) {
                        addr = self.mem.read_u64(addr)?;
                    }
                    cur_ty = inner;
                }
                EirProj::Field { field_idx, ty, .. } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if matches!(self.env.get(deref_t), Ty::Pool { .. }) && is_ptr {
                        // RowView { pool_ptr (0), index (8) }
                        let pool_ptr = self.mem.read_u64(addr)?;
                        let row_idx = self.mem.read_u32(VmMem::ptr_add(addr, 8)?)? as usize;
                        let col_ptr = self.mem.read_u64(VmMem::ptr_add(pool_ptr, field_idx * 8)?)?;
                        let elem_sz = x64_layout_impl(self.env, *ty, true).size.max(1);
                        addr = VmMem::ptr_add(col_ptr, row_idx * elem_sz)?;
                    } else if let Ty::Bitfield(bf_def) = self.env.get(deref_t) {
                        if is_ptr {
                            addr = self.mem.read_u64(addr)?;
                        }
                        let meta = self.env.bitfields.get(bf_def).ok_or_else(|| {
                            format!("EIR-Interp: missing bitfield metadata for {bf_def:?}")
                        })?;
                        let member = meta.members.get(*field_idx).ok_or_else(|| {
                            format!("EIR-Interp: bitfield field_idx {field_idx} out of range")
                        })?;
                        return Ok(PlaceTarget::Bitfield {
                            base_addr: addr,
                            base_ty: meta.base_ty,
                            lo_bit: member.lo,
                            hi_bit: member.hi,
                            field_ty: *ty,
                            temp_alloc,
                        });
                    } else {
                        if is_ptr {
                            addr = self.mem.read_u64(addr)?;
                        }
                        let foff = if matches!(self.env.get(deref_t), Ty::Builtin(BuiltinKind::Env, _)) {
                            match *field_idx {
                                0 => 0,  // alloc (40B)
                                1 => 40, // fs (8B)
                                2 => 48, // net (8B)
                                3 => 56, // clock (8B)
                                4 => 64, // io (8B)
                                5 => 72, // args (16B)
                                _ => 0,
                            }
                        } else {
                            struct_field_offset(self.env, deref_t, *field_idx).0
                        };
                        addr = VmMem::ptr_add(addr, foff)?;
                    }
                    cur_ty = *ty;
                }
                EirProj::TupleField { index, ty } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if is_ptr {
                        addr = self.mem.read_u64(addr)?;
                    }
                    let (foff, _) = struct_field_offset(self.env, deref_t, *index);
                    addr = VmMem::ptr_add(addr, foff)?;
                    cur_ty = *ty;
                }
                EirProj::PoolCol { col_idx, is_slot_pool, elem_ty, .. } => {
                    let is_ptr = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    if is_ptr {
                        addr = self.mem.read_u64(addr)?;
                    }
                    if let Some(EirProj::Index { index, ty }) = place.projs.get(i + 1) {
                        let col_ptr = self.mem.read_u64(VmMem::ptr_add(addr, col_idx * 8)?)?;
                        let idx = if *is_slot_pool {
                            (vals[index.0 as usize].as_u64() & 0xFFFF_FFFF) as usize
                        } else {
                            vals[index.0 as usize].as_u64() as usize
                        };
                        let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                        addr = VmMem::ptr_add(col_ptr, idx * esz)?;
                        cur_ty = *ty;
                        i += 2;
                        continue;
                    } else {
                        let deref = match self.env.get(cur_ty) {
                            Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t,
                            _ => cur_ty,
                        };
                        let ncols = match self.env.get(deref) {
                            Ty::Builtin(_, args) => pool_col_count(self.env, args),
                            _ => 0,
                        };
                        let len_off = if *is_slot_pool { (ncols + 2) * 8 } else { ncols * 8 };
                        let col_ptr = self.mem.read_u64(VmMem::ptr_add(addr, col_idx * 8)?)?;
                        let len = self.mem.read_u64(VmMem::ptr_add(addr, len_off)?)?;
                        if let Some(tp) = temp_alloc {
                            let _ = self.mem.free(tp);
                        }
                        return Ok(PlaceTarget::SliceVal(IVal::from_pair(col_ptr, len)));
                    }
                }
                EirProj::Index { index, ty } => {
                    let is_ptr_single = matches!(
                        self.env.get(cur_ty),
                        Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }
                    );
                    let deref_t = match self.env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. }
                        | Ty::RawPtr { inner: t, .. } => *t,
                        _ => cur_ty,
                    };
                    if is_ptr_single {
                        addr = self.mem.read_u64(addr)?;
                    }
                    let idx = vals[index.0 as usize].as_u64() as usize;
                    match self.env.get(deref_t) {
                        Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str => {
                            let data_ptr = self.mem.read_u64(addr)?;
                            let len = self.mem.read_u64(VmMem::ptr_add(addr, 8)?)? as usize;
                            if idx >= len {
                                if let Some(tp) = temp_alloc {
                                    let _ = self.mem.free(tp);
                                }
                                return Err(format!(
                                    "EIR-Interp: slice index out of bounds (index {idx}, len {len})"
                                ));
                            }
                            addr = data_ptr;
                        }
                        Ty::Array { len, .. } => {
                            if (idx as u64) >= *len {
                                if let Some(tp) = temp_alloc {
                                    let _ = self.mem.free(tp);
                                }
                                return Err(format!(
                                    "EIR-Interp: array index out of bounds (index {idx}, len {len})"
                                ));
                            }
                        }
                        Ty::Vec { len, .. } if idx >= (*len as usize) => {
                            if let Some(tp) = temp_alloc {
                                let _ = self.mem.free(tp);
                            }
                            return Err(format!(
                                "EIR-Interp: vec index out of bounds (index {idx}, len {len})"
                            ));
                        }
                        _ => {}
                    }
                    let esz = x64_layout(self.env, *ty).size.max(1);
                    addr = VmMem::ptr_add(addr, idx * esz)?;
                    cur_ty = *ty;
                }
            }
            i += 1;
        }
        Ok(PlaceTarget::Addr { addr, temp_alloc })
    }

    fn eval_pool_base_ptr(
        &mut self,
        pool: &EirPlaceBase,
        pool_is_ptr: bool,
        vals: &[IVal],
        slot_ptrs: &[u64],
    ) -> Result<(u64, Option<u64>), String> {
        match pool {
            EirPlaceBase::Slot(s) => {
                let sp = slot_ptrs[s.0 as usize];
                if pool_is_ptr {
                    Ok((self.mem.read_u64(sp)?, None))
                } else {
                    Ok((sp, None))
                }
            }
            EirPlaceBase::Val(v) => {
                if pool_is_ptr {
                    Ok((vals[v.0 as usize].as_u64(), None))
                } else {
                    let b = match &vals[v.0 as usize] {
                        IVal::Bytes(b) => b.clone(),
                        other => other.to_bytes(8),
                    };
                    let tp = self.mem.alloc_bytes(&b, false)?;
                    Ok((tp, Some(tp)))
                }
            }
        }
    }

    fn exec_inst(
        &mut self,
        f: &EirFunction,
        inst: &EirInst,
        vals: &mut [IVal],
        slot_ptrs: &[u64],
    ) -> Result<(), String> {
        match &inst.kind {
            EirInstKind::Const { dst, ty, konst } => {
                vals[dst.0 as usize] = self.eval_const(*ty, konst)?;
            }
            EirInstKind::CopyVal { dst, src } => {
                vals[dst.0 as usize] = vals[src.0 as usize].clone();
            }
            EirInstKind::LoadSlot { dst, slot } => {
                let ty = f.value_tys[dst.0 as usize];
                let sz = interp_ty_size(self.env, ty);
                let b = self.mem.read(slot_ptrs[slot.0 as usize], sz)?;
                vals[dst.0 as usize] = IVal::from_bytes(b);
            }
            EirInstKind::StoreSlot { slot, val } => {
                let ty = f.slots[slot.0 as usize].ty;
                let sz = interp_ty_size(self.env, ty);
                let bytes = vals[val.0 as usize].to_bytes(sz);
                self.mem.write(slot_ptrs[slot.0 as usize], &bytes)?;
            }
            EirInstKind::SlotAddr { dst, slot, .. } => {
                vals[dst.0 as usize] = IVal::Scalar(slot_ptrs[slot.0 as usize]);
            }
            EirInstKind::LoadPlace { dst, place } => {
                match self.eval_place_addr(f, place, vals, slot_ptrs)? {
                    PlaceTarget::SliceVal(v) => vals[dst.0 as usize] = v,
                    PlaceTarget::Bitfield {
                        base_addr,
                        base_ty,
                        lo_bit,
                        hi_bit,
                        field_ty,
                        temp_alloc,
                    } => {
                        let bsz = x64_layout(self.env, base_ty).size.clamp(1, 8);
                        let raw = IVal::from_bytes(self.mem.read(base_addr, bsz)?).as_u64();
                        let width = hi_bit.saturating_sub(lo_bit).max(1);
                        let extracted = (raw >> lo_bit) & mask_bits(width);
                        let signed = is_signed_int(self.env, field_ty);
                        let final_v = if signed {
                            sign_extend_i128(extracted, width, true) as i64 as u64
                        } else {
                            extracted
                        };
                        vals[dst.0 as usize] = IVal::Scalar(final_v);
                        if let Some(tp) = temp_alloc {
                            let _ = self.mem.free(tp);
                        }
                    }
                    PlaceTarget::Addr { addr, temp_alloc } => {
                        let ty = f.value_tys[dst.0 as usize];
                        let sz = interp_ty_size(self.env, ty);
                        let b = self.mem.read(addr, sz)?;
                        vals[dst.0 as usize] = IVal::from_bytes(b);
                        if let Some(tp) = temp_alloc {
                            let _ = self.mem.free(tp);
                        }
                    }
                }
            }
            EirInstKind::StorePlace { place, val } => {
                match self.eval_place_addr(f, place, vals, slot_ptrs)? {
                    PlaceTarget::SliceVal(_) => {
                        return Err("EIR-Interp: cannot store directly into bare pool column slice".to_owned());
                    }
                    PlaceTarget::Bitfield {
                        base_addr,
                        base_ty,
                        lo_bit,
                        hi_bit,
                        temp_alloc,
                        ..
                    } => {
                        let bsz = x64_layout(self.env, base_ty).size.clamp(1, 8);
                        let old = IVal::from_bytes(self.mem.read(base_addr, bsz)?).as_u64();
                        let width = hi_bit.saturating_sub(lo_bit).max(1);
                        let m = mask_bits(width);
                        let ins = vals[val.0 as usize].as_u64() & m;
                        let updated = (old & !(m << lo_bit)) | (ins << lo_bit);
                        self.mem.write(base_addr, &updated.to_le_bytes()[..bsz])?;
                        if let Some(tp) = temp_alloc {
                            let _ = self.mem.free(tp);
                        }
                    }
                    PlaceTarget::Addr { addr, temp_alloc } => {
                        let sz = interp_ty_size(self.env, place.ty);
                        let bytes = vals[val.0 as usize].to_bytes(sz);
                        self.mem.write(addr, &bytes)?;
                        if let Some(tp) = temp_alloc {
                            let _ = self.mem.free(tp);
                        }
                    }
                }
            }
            EirInstKind::PlaceAddr { dst, place, .. } => {
                match self.eval_place_addr(f, place, vals, slot_ptrs)? {
                    PlaceTarget::Addr { addr, temp_alloc: None } => {
                        vals[dst.0 as usize] = IVal::Scalar(addr);
                    }
                    PlaceTarget::Addr { addr, temp_alloc: Some(_) } => {
                        // Keep temporary alive for the frame duration (rvalue borrow).
                        vals[dst.0 as usize] = IVal::Scalar(addr);
                    }
                    PlaceTarget::SliceVal(_) | PlaceTarget::Bitfield { .. } => {
                        return Err("EIR-Interp: cannot take address of non-addressable place".to_owned());
                    }
                }
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
                let (pool_ptr, temp) = self.eval_pool_base_ptr(pool, *pool_is_ptr, vals, slot_ptrs)?;
                let col_ptr = self.mem.read_u64(VmMem::ptr_add(pool_ptr, col_idx * 8)?)?;
                let idx = if *is_slot_pool {
                    (vals[index.0 as usize].as_u64() & 0xFFFF_FFFF) as usize
                } else {
                    vals[index.0 as usize].as_u64() as usize
                };
                let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                let elem_addr = VmMem::ptr_add(col_ptr, idx * esz)?;
                let dst_ty = f.value_tys[dst.0 as usize];
                let dsz = x64_layout_impl(self.env, dst_ty, *is_slot_pool).size;
                let b = self.mem.read(elem_addr, dsz)?;
                vals[dst.0 as usize] = IVal::from_bytes(b);
                if let Some(tp) = temp {
                    let _ = self.mem.free(tp);
                }
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
                let (pool_ptr, temp) = self.eval_pool_base_ptr(pool, *pool_is_ptr, vals, slot_ptrs)?;
                let col_ptr = self.mem.read_u64(VmMem::ptr_add(pool_ptr, col_idx * 8)?)?;
                let idx = if *is_slot_pool {
                    (vals[index.0 as usize].as_u64() & 0xFFFF_FFFF) as usize
                } else {
                    vals[index.0 as usize].as_u64() as usize
                };
                let esz = x64_layout_impl(self.env, *elem_ty, *is_slot_pool).size.max(1);
                let elem_addr = VmMem::ptr_add(col_ptr, idx * esz)?;
                let bytes = vals[val.0 as usize].to_bytes(esz);
                self.mem.write(elem_addr, &bytes)?;
                if let Some(tp) = temp {
                    let _ = self.mem.free(tp);
                }
            }
            EirInstKind::Unary { dst, op, operand, ty } => {
                vals[dst.0 as usize] = self.eval_unary(*op, &vals[operand.0 as usize], *ty)?;
            }
            EirInstKind::Binary { dst, op, lhs, rhs, op_ty, dst_ty } => {
                vals[dst.0 as usize] = self.eval_binary(
                    *op,
                    &vals[lhs.0 as usize],
                    &vals[rhs.0 as usize],
                    *op_ty,
                    *dst_ty,
                )?;
            }
            EirInstKind::Cast { dst, kind, operand, from_ty, to_ty } => {
                vals[dst.0 as usize] = self.eval_cast(
                    *kind,
                    &vals[operand.0 as usize],
                    *from_ty,
                    *to_ty,
                )?;
            }
            EirInstKind::Slice { dst, base, base_ty, lo, hi, dst_ty } => {
                let elem_sz = match self.env.get(*dst_ty) {
                    Ty::Borrow { inner: BorrowInner::Slice(el), .. } => {
                        x64_layout(self.env, *el).size.max(1)
                    }
                    _ => 1,
                };
                let (base_ptr, base_len) = match self.env.get(*base_ty) {
                    Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str => {
                        vals[base.0 as usize].as_pair()
                    }
                    Ty::Borrow { inner: BorrowInner::Single(at), .. } => {
                        let ptr = vals[base.0 as usize].as_u64();
                        match self.env.get(*at) {
                            Ty::Array { len, .. } => (ptr, *len),
                            _ => {
                                let p = self.mem.read_u64(ptr)?;
                                let l = self.mem.read_u64(VmMem::ptr_add(ptr, 8)?)?;
                                (p, l)
                            }
                        }
                    }
                    Ty::Array { len, .. } => {
                        let sz = x64_layout(self.env, *base_ty).size.max(1);
                        let p = self.mem.alloc_bytes(&vals[base.0 as usize].to_bytes(sz), false)?;
                        (p, *len)
                    }
                    _ => vals[base.0 as usize].as_pair(),
                };
                let lo_v = lo.map_or(0, |v| vals[v.0 as usize].as_u64());
                let hi_v = hi.map_or(base_len, |v| vals[v.0 as usize].as_u64());
                if lo_v > hi_v || hi_v > base_len {
                    return Err(format!(
                        "EIR-Interp: proven slice bounds violated: {}..{} on len {}",
                        lo_v, hi_v, base_len
                    ));
                }
                let new_ptr = if lo_v == 0 {
                    base_ptr
                } else {
                    VmMem::ptr_add(base_ptr, (lo_v as usize) * elem_sz)?
                };
                vals[dst.0 as usize] = IVal::from_pair(new_ptr, hi_v - lo_v);
            }
            EirInstKind::Aggregate { dst, kind, elems, dst_ty } => {
                vals[dst.0 as usize] = self.eval_aggregate(kind, elems, *dst_ty, vals)?;
            }
            EirInstKind::UnwrapOptional { dst, src, src_ty, dst_ty } => {
                if *dst_ty == TyId::VOID {
                    vals[dst.0 as usize] = IVal::Void;
                    return Ok(());
                }
                let inner = match self.env.get(*src_ty) { Ty::Optional(i) => *i, _ => *dst_ty };
                if !self.eval_is_non_null(&vals[src.0 as usize], inner) {
                    return Err("EIR-Interp: attempted to unwrap null Optional".to_owned());
                }
                if is_niche_id_opt(self.env, inner) {
                    vals[dst.0 as usize] = vals[src.0 as usize].clone();
                } else {
                    let sz = x64_layout(self.env, *src_ty).size;
                    let raw = vals[src.0 as usize].to_bytes(sz);
                    let voff = opt_val_offset(self.env, inner);
                    let dsz = x64_layout(self.env, *dst_ty).size;
                    vals[dst.0 as usize] = IVal::from_bytes(&raw[voff..voff + dsz]);
                }
            }
            EirInstKind::UnwrapErrorOk { dst, src, src_ty, dst_ty } => {
                let sz = x64_layout(self.env, *src_ty).size;
                let raw = vals[src.0 as usize].to_bytes(sz);
                let err_tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
                if err_tag != 0 {
                    return Err(format!("EIR-Interp: UnwrapErrorOk on ErrorUnion with err={err_tag}"));
                }
                if *dst_ty == TyId::VOID {
                    vals[dst.0 as usize] = IVal::Void;
                } else {
                    let ok_off = eu_ok_offset(self.env, *dst_ty);
                    let dsz = x64_layout(self.env, *dst_ty).size;
                    vals[dst.0 as usize] = IVal::from_bytes(&raw[ok_off..ok_off + dsz]);
                }
            }
            EirInstKind::UnwrapErrorErr { dst, src, src_ty, .. } => {
                let sz = x64_layout(self.env, *src_ty).size.max(4);
                let raw = vals[src.0 as usize].to_bytes(sz);
                let err_tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
                vals[dst.0 as usize] = IVal::Scalar(err_tag as u64);
            }
            EirInstKind::IsNonNull { dst, src, src_ty } => {
                let inner = match self.env.get(*src_ty) { Ty::Optional(i) => *i, _ => *src_ty };
                let ok = self.eval_is_non_null(&vals[src.0 as usize], inner);
                vals[dst.0 as usize] = IVal::Scalar(u64::from(ok));
            }
            EirInstKind::IsError { dst, src, src_ty } => {
                let sz = x64_layout(self.env, *src_ty).size.max(4);
                let raw = vals[src.0 as usize].to_bytes(sz);
                let err_tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
                vals[dst.0 as usize] = IVal::Scalar(u64::from(err_tag != 0));
            }
            EirInstKind::MatchTag { dst, src, src_ty } => match self.env.get(*src_ty) {
                Ty::ErrorUnion { .. } | Ty::Union { .. } => {
                    let sz = x64_layout(self.env, *src_ty).size.max(4);
                    let raw = vals[src.0 as usize].to_bytes(sz);
                    let tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
                    vals[dst.0 as usize] = IVal::Scalar(tag as u64);
                }
                _ => {
                    vals[dst.0 as usize] = IVal::Scalar(vals[src.0 as usize].as_u64() & 0xFFFF_FFFF);
                }
            },
            EirInstKind::FieldExtract { dst, src, src_ty, dst_ty, .. } => {
                if *dst_ty == TyId::VOID {
                    vals[dst.0 as usize] = IVal::Void;
                    return Ok(());
                }
                let p_off = match self.env.get(*src_ty) {
                    Ty::Union { def, .. } => union_payload_offset(self.env, *def),
                    _ => 4,
                };
                let sz = x64_layout(self.env, *src_ty).size;
                let raw = vals[src.0 as usize].to_bytes(sz);
                let dsz = x64_layout(self.env, *dst_ty).size;
                vals[dst.0 as usize] = IVal::from_bytes(&raw[p_off..p_off + dsz]);
            }
            EirInstKind::MakeDyn { dst, ptr, iface_def, concrete_ty, .. } => {
                let data_ptr = vals[ptr.0 as usize].as_u64();
                let vt_id = *self
                    .vtables
                    .get(&(*iface_def, *concrete_ty))
                    .ok_or_else(|| {
                        format!(
                            "EIR-Interp: missing vtable for iface {:?} / concrete {:?}",
                            iface_def, concrete_ty
                        )
                    })?;
                vals[dst.0 as usize] = IVal::from_pair(data_ptr, vt_id);
            }
            EirInstKind::MakeClosure { dst, closure_id, env_ptr, .. } => {
                let h = self.intern_fn_handle(CfgFnKey::Closure(*closure_id));
                let ep = env_ptr.map_or(0, |ev| vals[ev.0 as usize].as_u64());
                vals[dst.0 as usize] = IVal::from_pair(h, ep);
            }
            EirInstKind::InlineAsm { template, .. } => {
                if template.trim() != "pause" {
                    return Err(format!(
                        "EIR-Interp: inline asm `{template}` cannot be executed in portable interpreter"
                    ));
                }
            }
            EirInstKind::Call { dst, callee, args, .. } => {
                let mut call_args: Vec<IVal> = args.iter().map(|a| vals[a.0 as usize].clone()).collect();
                let res = match callee {
                    EirCallee::DirectFn(d) => self.call_key(CfgFnKey::TopDef(*d), call_args)?,
                    EirCallee::Method { impl_idx, sub_idx } => {
                        self.call_key(CfgFnKey::ImplMethod { impl_idx: *impl_idx, sub_idx: *sub_idx }, call_args)?
                    }
                    EirCallee::GenericInst { inst_id } => {
                        self.call_key(CfgFnKey::GenericInst(*inst_id), call_args)?
                    }
                    EirCallee::DynCall { method_idx, .. } => {
                        let (data_ptr, vt_id) = call_args[0].as_pair();
                        call_args[0] = IVal::Scalar(data_ptr);
                        let key = self
                            .vtable_methods
                            .get(vt_id as usize)
                            .and_then(|m| m.get(*method_idx))
                            .copied()
                            .ok_or_else(|| {
                                format!("EIR-Interp: invalid dyn vtable {vt_id} method {method_idx}")
                            })?;
                        self.call_key(key, call_args)?
                    }
                    EirCallee::ExternFn { name, .. } => self.exec_extern_fn(*name, &call_args)?,
                    EirCallee::FnPtr(fpv) => {
                        let (h, env_ptr) = vals[fpv.0 as usize].as_pair();
                        let key = *self
                            .handle_to_fn
                            .get(h as usize)
                            .filter(|_| h != 0)
                            .ok_or_else(|| format!("EIR-Interp: invalid function pointer handle {h}"))?;
                        if matches!(key, CfgFnKey::Closure(_)) {
                            call_args.insert(0, IVal::Scalar(env_ptr));
                        }
                        self.call_key(key, call_args)?
                    }
                };
                if let Some(d) = dst {
                    vals[d.0 as usize] = res;
                }
            }
            EirInstKind::BuiltinCall { dst, op, args, arg_slots, ret_ty } => {
                let res = self.exec_builtin(f, op, args, arg_slots, *ret_ty, vals, slot_ptrs)?;
                if let Some(d) = dst {
                    vals[d.0 as usize] = res;
                }
            }
            EirInstKind::GetVar { .. } | EirInstKind::SetVar { .. } => {
                return Err("ICE [eir_interp]: unlowered GetVar/SetVar reached interpreter".to_owned());
            }
        }
        Ok(())
    }

    fn eval_const(&mut self, ty: TyId, konst: &CfgConst) -> Result<IVal, String> {
        match konst {
            CfgConst::Void => Ok(IVal::Void),
            CfgConst::Bool(b) => Ok(IVal::Scalar(u64::from(*b))),
            CfgConst::Char(c) => Ok(IVal::Scalar(*c as u64)),
            CfgConst::Int(v, _) => Ok(IVal::Scalar(*v as u64)),
            CfgConst::Float(s, _) => match self.env.get(ty) {
                Ty::Float(16) => {
                    let fv: f32 = s.parse().unwrap_or(0.0);
                    Ok(IVal::Scalar(f32_to_f16_bits(fv) as u64))
                }
                Ty::Float(32) => {
                    let fv: f32 = s.parse().unwrap_or(0.0);
                    Ok(IVal::Scalar(fv.to_bits() as u64))
                }
                _ => {
                    let fv: f64 = s.parse().unwrap_or(0.0);
                    Ok(IVal::Scalar(fv.to_bits()))
                }
            },
            CfgConst::Str(s) => {
                let ptr = self.intern_ro_bytes(s.as_bytes())?;
                Ok(IVal::from_pair(ptr, s.len() as u64))
            }
            CfgConst::Bytes(bs) => {
                let ptr = self.intern_ro_bytes(bs)?;
                Ok(IVal::from_pair(ptr, bs.len() as u64))
            }
            CfgConst::Null(_) => Ok(self.opt_null_val(ty)),
            CfgConst::EnumTag { tag_val, .. } => Ok(IVal::Scalar(*tag_val as u64)),
            CfgConst::ErrorTag { tag_val, .. } => Ok(IVal::Scalar(*tag_val as u64)),
            CfgConst::FnRef(def, _) => {
                let h = self.intern_fn_handle(CfgFnKey::TopDef(*def));
                Ok(IVal::from_pair(h, 0))
            }
            CfgConst::ConstDef(d, ty) => {
                if let Some(cm) = self.env.consts.get(d) {
                    if let Some(ref ct) = cm.ct_val
                        && let Some(bytes) = crate::codegen_x64::encode_ct_val_bytes(ct, *ty, self.env)
                    {
                        return Ok(IVal::from_bytes(&bytes));
                    }
                    if let Some(iv) = cm.int_val {
                        return Ok(IVal::Scalar((iv as i64) as u64));
                    }
                }
                let sz = x64_layout(self.env, *ty).size;
                if sz <= 8 {
                    Ok(IVal::Scalar(0))
                } else {
                    Ok(IVal::from_bytes(&vec![0u8; sz]))
                }
            }
        }
    }

    fn opt_null_val(&self, ty: TyId) -> IVal {
        match self.env.get(ty) {
            Ty::Optional(inner) => match opt_niche(self.env, *inner) {
                OptNiche::IdBrand => IVal::Scalar(u32::MAX as u64),
                OptNiche::BoolTag => IVal::Scalar(2),
                OptNiche::ScalarSentinel { sentinel, .. } => IVal::Scalar(sentinel),
                OptNiche::SliceMaxLen => IVal::from_pair(0, u64::MAX),
                OptNiche::GenId | OptNiche::Link | OptNiche::NullPtr | OptNiche::FatPtrNullFirst | OptNiche::None => {
                    IVal::from_bytes(&vec![0u8; x64_layout(self.env, ty).size])
                }
            },
            Ty::Builtin(BuiltinKind::IdBrand, _) => IVal::Scalar(u32::MAX as u64),
            _ => IVal::from_bytes(&vec![0u8; x64_layout(self.env, ty).size]),
        }
    }

    fn eval_is_non_null(&self, v: &IVal, inner: TyId) -> bool {
        match opt_niche(self.env, inner) {
            OptNiche::IdBrand => (v.as_u64() as u32) != u32::MAX,
            OptNiche::GenId | OptNiche::Link => {
                let raw = v.to_bytes(8);
                let slot_gen = u32::from_le_bytes(raw[4..8].try_into().unwrap());
                slot_gen != 0
            }
            OptNiche::NullPtr => v.as_u64() != 0,
            OptNiche::FatPtrNullFirst => v.as_pair().0 != 0,
            OptNiche::SliceMaxLen => v.as_pair().1 != u64::MAX,
            OptNiche::BoolTag => (v.as_u64() as u8) != 2,
            OptNiche::ScalarSentinel { size, sentinel } => {
                let mask = if size >= 8 { u64::MAX } else { (1u64 << (size * 8)) - 1 };
                (v.as_u64() & mask) != (sentinel & mask)
            }
            OptNiche::None => {
                let raw = v.to_bytes(1);
                raw[0] != 0
            }
        }
    }

    fn eval_unary(&self, op: UnaryOp, v: &IVal, ty: TyId) -> Result<IVal, String> {
        if let Ty::Float(bits) = self.env.get(ty) {
            if op != UnaryOp::Neg {
                return Err(format!("EIR-Interp: unsupported float unary op {op:?}"));
            }
            return Ok(match *bits {
                16 => IVal::Scalar((v.as_u64() as u16 ^ 0x8000) as u64),
                32 => IVal::Scalar((-f32::from_bits(v.as_u64() as u32)).to_bits() as u64),
                _ => IVal::Scalar((-f64::from_bits(v.as_u64())).to_bits()),
            });
        }
        if let Ty::Vec { elem, len } = self.env.get(ty) {
            let esz = x64_layout(self.env, *elem).size.max(1);
            let raw = v.to_bytes(esz * (*len as usize));
            let mut out = vec![0u8; raw.len()];
            for i in 0..(*len as usize) {
                let ev = IVal::from_bytes(&raw[i * esz..(i + 1) * esz]);
                let rv = self.eval_unary(op, &ev, *elem)?;
                out[i * esz..(i + 1) * esz].copy_from_slice(&rv.to_bytes(esz));
            }
            return Ok(IVal::from_bytes(&out));
        }
        let bits = int_bits(self.env, ty);
        let signed = is_signed_int(self.env, ty);
        match op {
            UnaryOp::Neg => {
                let x = sign_extend_i128(v.as_u64(), bits, signed);
                let r = (-x) as i64 as u64 & mask_bits(bits);
                Ok(IVal::Scalar(r))
            }
            UnaryOp::Not => Ok(IVal::Scalar(u64::from(v.as_u64() == 0))),
            UnaryOp::BitNot => Ok(IVal::Scalar((!v.as_u64()) & mask_bits(bits))),
            other => Err(format!("EIR-Interp: unsupported unary op {other:?}")),
        }
    }

    fn eval_binary(
        &self,
        op: BinaryOp,
        lhs: &IVal,
        rhs: &IVal,
        op_ty: TyId,
        dst_ty: TyId,
    ) -> Result<IVal, String> {
        // Vector elementwise operations
        if let Ty::Vec { elem, len } = self.env.get(op_ty) {
            let esz = x64_layout(self.env, *elem).size.max(1);
            let lb = lhs.to_bytes(esz * (*len as usize));
            let rb = rhs.to_bytes(esz * (*len as usize));
            if matches!(op, BinaryOp::Eq | BinaryOp::NotEq) && dst_ty == TyId::BOOL {
                let mut all_eq = true;
                for i in 0..(*len as usize) {
                    let le = IVal::from_bytes(&lb[i * esz..(i + 1) * esz]);
                    let re = IVal::from_bytes(&rb[i * esz..(i + 1) * esz]);
                    let rv = self.eval_binary(BinaryOp::Eq, &le, &re, *elem, TyId::BOOL)?;
                    if rv.as_u64() == 0 {
                        all_eq = false;
                        break;
                    }
                }
                return Ok(IVal::Scalar(u64::from(if op == BinaryOp::Eq { all_eq } else { !all_eq })));
            }
            let out_elem = match self.env.get(dst_ty) {
                Ty::Vec { elem: oe, .. } => *oe,
                _ => *elem,
            };
            let oesz = x64_layout(self.env, out_elem).size.max(1);
            let mut out = vec![0u8; oesz * (*len as usize)];
            for i in 0..(*len as usize) {
                let le = IVal::from_bytes(&lb[i * esz..(i + 1) * esz]);
                let re = IVal::from_bytes(&rb[i * esz..(i + 1) * esz]);
                let rv = self.eval_binary(op, &le, &re, *elem, out_elem)?;
                out[i * oesz..(i + 1) * oesz].copy_from_slice(&rv.to_bytes(oesz));
            }
            return Ok(IVal::from_bytes(&out));
        }

        // Float operations (exact IEEE-754)
        if let Ty::Float(bits) = self.env.get(op_ty) {
            return match *bits {
                16 => {
                    let a = f16_bits_to_f32(lhs.as_u64() as u16);
                    let b = f16_bits_to_f32(rhs.as_u64() as u16);
                    match op {
                        BinaryOp::Add => Ok(IVal::Scalar(f32_to_f16_bits(a + b) as u64)),
                        BinaryOp::Sub => Ok(IVal::Scalar(f32_to_f16_bits(a - b) as u64)),
                        BinaryOp::Mul => Ok(IVal::Scalar(f32_to_f16_bits(a * b) as u64)),
                        BinaryOp::Div => Ok(IVal::Scalar(f32_to_f16_bits(a / b) as u64)),
                        BinaryOp::Eq => Ok(IVal::Scalar(u64::from(a == b))),
                        BinaryOp::NotEq => Ok(IVal::Scalar(u64::from(a != b))),
                        BinaryOp::Lt => Ok(IVal::Scalar(u64::from(a < b))),
                        BinaryOp::LtEq => Ok(IVal::Scalar(u64::from(a <= b))),
                        BinaryOp::Gt => Ok(IVal::Scalar(u64::from(a > b))),
                        BinaryOp::GtEq => Ok(IVal::Scalar(u64::from(a >= b))),
                        other => Err(format!("EIR-Interp: unsupported f16 op {other:?}")),
                    }
                }
                32 => {
                    let a = f32::from_bits(lhs.as_u64() as u32);
                    let b = f32::from_bits(rhs.as_u64() as u32);
                    match op {
                        BinaryOp::Add => Ok(IVal::Scalar((a + b).to_bits() as u64)),
                        BinaryOp::Sub => Ok(IVal::Scalar((a - b).to_bits() as u64)),
                        BinaryOp::Mul => Ok(IVal::Scalar((a * b).to_bits() as u64)),
                        BinaryOp::Div => Ok(IVal::Scalar((a / b).to_bits() as u64)),
                        BinaryOp::Eq => Ok(IVal::Scalar(u64::from(a == b))),
                        BinaryOp::NotEq => Ok(IVal::Scalar(u64::from(a != b))),
                        BinaryOp::Lt => Ok(IVal::Scalar(u64::from(a < b))),
                        BinaryOp::LtEq => Ok(IVal::Scalar(u64::from(a <= b))),
                        BinaryOp::Gt => Ok(IVal::Scalar(u64::from(a > b))),
                        BinaryOp::GtEq => Ok(IVal::Scalar(u64::from(a >= b))),
                        other => Err(format!("EIR-Interp: unsupported f32 op {other:?}")),
                    }
                }
                _ => {
                    let a = f64::from_bits(lhs.as_u64());
                    let b = f64::from_bits(rhs.as_u64());
                    match op {
                        BinaryOp::Add => Ok(IVal::Scalar((a + b).to_bits())),
                        BinaryOp::Sub => Ok(IVal::Scalar((a - b).to_bits())),
                        BinaryOp::Mul => Ok(IVal::Scalar((a * b).to_bits())),
                        BinaryOp::Div => Ok(IVal::Scalar((a / b).to_bits())),
                        BinaryOp::Eq => Ok(IVal::Scalar(u64::from(a == b))),
                        BinaryOp::NotEq => Ok(IVal::Scalar(u64::from(a != b))),
                        BinaryOp::Lt => Ok(IVal::Scalar(u64::from(a < b))),
                        BinaryOp::LtEq => Ok(IVal::Scalar(u64::from(a <= b))),
                        BinaryOp::Gt => Ok(IVal::Scalar(u64::from(a > b))),
                        BinaryOp::GtEq => Ok(IVal::Scalar(u64::from(a >= b))),
                        other => Err(format!("EIR-Interp: unsupported f64 op {other:?}")),
                    }
                }
            };
        }

        // GenId / Link equality comparison (8 bytes)
        if matches!(op, BinaryOp::Eq | BinaryOp::NotEq)
            && matches!(self.env.get(op_ty), Ty::Builtin(BuiltinKind::GenId | BuiltinKind::Link, _))
        {
            let sz = x64_layout(self.env, op_ty).size.clamp(4, 8);
            let eq = lhs.to_bytes(sz) == rhs.to_bytes(sz);
            return Ok(IVal::Scalar(u64::from(if op == BinaryOp::Eq { eq } else { !eq })));
        }

        let signed = is_signed_int(self.env, op_ty);
        let bits = int_bits(self.env, op_ty);
        let mask = mask_bits(bits);
        let (lo, hi) = int_bounds_for(bits, signed);
        let a = sign_extend_i128(lhs.as_u64(), bits, signed);
        let b = sign_extend_i128(rhs.as_u64(), bits, signed);

        // Checked ops returning ErrorUnion
        if matches!(
            op,
            BinaryOp::AddCheck | BinaryOp::SubCheck | BinaryOp::MulCheck | BinaryOp::DivCheck | BinaryOp::ModCheck
        ) {
            let (err_code, ok_bits): (u32, u64) = match op {
                BinaryOp::AddCheck => {
                    let s = a + b;
                    if s < lo || s > hi { (1, 0) } else { (0, (s as i64 as u64) & mask) }
                }
                BinaryOp::SubCheck => {
                    let s = a - b;
                    if s < lo || s > hi { (1, 0) } else { (0, (s as i64 as u64) & mask) }
                }
                BinaryOp::MulCheck => {
                    if signed {
                        match a.checked_mul(b) {
                            Some(s) if s >= lo && s <= hi => (0, (s as i64 as u64) & mask),
                            _ => (1, 0),
                        }
                    } else {
                        let ua = a as u128;
                        let ub = b as u128;
                        match ua.checked_mul(ub) {
                            Some(p) if p <= (hi as u128) => (0, (p as u64) & mask),
                            _ => (1, 0),
                        }
                    }
                }
                BinaryOp::DivCheck => {
                    if b == 0 {
                        (2, 0)
                    } else if signed && a == lo && b == -1 {
                        (1, 0)
                    } else {
                        (0, ((a / b) as i64 as u64) & mask)
                    }
                }
                BinaryOp::ModCheck => {
                    if b == 0 {
                        (2, 0)
                    } else if signed && a == lo && b == -1 {
                        (1, 0)
                    } else {
                        (0, ((a % b) as i64 as u64) & mask)
                    }
                }
                _ => unreachable!(),
            };
            return Ok(self.pack_error_union(dst_ty, op_ty, err_code, &IVal::Scalar(ok_bits)));
        }

        // Saturating ops
        if matches!(op, BinaryOp::AddSat | BinaryOp::SubSat | BinaryOp::MulSat) {
            if op == BinaryOp::MulSat && !signed {
                let ua = a as u128;
                let ub = b as u128;
                let p = ua.saturating_mul(ub).min(hi as u128);
                return Ok(IVal::Scalar((p as u64) & mask));
            }
            let clamped = match op {
                BinaryOp::AddSat => (a + b).clamp(lo, hi),
                BinaryOp::SubSat => (a - b).clamp(lo, hi),
                BinaryOp::MulSat => match a.checked_mul(b) {
                    Some(p) => p.clamp(lo, hi),
                    None => {
                        if (a < 0) ^ (b < 0) { lo } else { hi }
                    }
                },
                _ => unreachable!(),
            };
            return Ok(IVal::Scalar((clamped as i64 as u64) & mask));
        }

        match op {
            // Proven arithmetic: verify at runtime that static proof holds!
            BinaryOp::Add => {
                let s = a + b;
                if s < lo || s > hi {
                    return Err(format!("EIR-Interp fault: proven `+` overflowed ({a} + {b} outside [{lo}..={hi}])"));
                }
                Ok(IVal::Scalar((s as i64 as u64) & mask))
            }
            BinaryOp::Sub => {
                let s = a - b;
                if s < lo || s > hi {
                    return Err(format!("EIR-Interp fault: proven `-` overflowed ({a} - {b} outside [{lo}..={hi}])"));
                }
                Ok(IVal::Scalar((s as i64 as u64) & mask))
            }
            BinaryOp::Mul => {
                if signed {
                    match a.checked_mul(b) {
                        Some(s) if s >= lo && s <= hi => Ok(IVal::Scalar((s as i64 as u64) & mask)),
                        _ => Err(format!("EIR-Interp fault: proven `*` overflowed ({a} * {b} outside [{lo}..={hi}])")),
                    }
                } else {
                    let ua = a as u128;
                    let ub = b as u128;
                    match ua.checked_mul(ub) {
                        Some(p) if p <= (hi as u128) => Ok(IVal::Scalar((p as u64) & mask)),
                        _ => Err(format!("EIR-Interp fault: proven `*` overflowed ({ua} * {ub} outside [0..={hi}])")),
                    }
                }
            }
            BinaryOp::Div => {
                if b == 0 {
                    return Err("EIR-Interp fault: proven `/` divided by zero".to_owned());
                }
                if signed && a == lo && b == -1 {
                    return Err("EIR-Interp fault: proven `/` overflowed MIN / -1".to_owned());
                }
                Ok(IVal::Scalar(((a / b) as i64 as u64) & mask))
            }
            BinaryOp::Mod => {
                if b == 0 {
                    return Err("EIR-Interp fault: proven `%` divided by zero".to_owned());
                }
                if signed && a == lo && b == -1 {
                    return Err("EIR-Interp fault: proven `%` overflowed MIN % -1".to_owned());
                }
                Ok(IVal::Scalar(((a % b) as i64 as u64) & mask))
            }
            BinaryOp::AddWrap => Ok(IVal::Scalar(((a.wrapping_add(b)) as i64 as u64) & mask)),
            BinaryOp::SubWrap => Ok(IVal::Scalar(((a.wrapping_sub(b)) as i64 as u64) & mask)),
            BinaryOp::MulWrap => Ok(IVal::Scalar(((a.wrapping_mul(b)) as i64 as u64) & mask)),
            BinaryOp::BitAnd | BinaryOp::And => Ok(IVal::Scalar((lhs.as_u64() & rhs.as_u64()) & mask)),
            BinaryOp::BitOr | BinaryOp::Or => Ok(IVal::Scalar((lhs.as_u64() | rhs.as_u64()) & mask)),
            BinaryOp::BitXor => Ok(IVal::Scalar((lhs.as_u64() ^ rhs.as_u64()) & mask)),
            BinaryOp::Shl => {
                let sh = rhs.as_u64();
                if sh >= (bits as u64) {
                    return Err(format!("EIR-Interp fault: proven `<<` shift {sh} >= bitwidth {bits}"));
                }
                Ok(IVal::Scalar(((lhs.as_u64() & mask).wrapping_shl(sh as u32)) & mask))
            }
            BinaryOp::ShlWrap => {
                let sh = (rhs.as_u64() as u32) & ((bits as u32).saturating_sub(1));
                Ok(IVal::Scalar(((lhs.as_u64() & mask).wrapping_shl(sh)) & mask))
            }
            BinaryOp::Shr => {
                let sh = rhs.as_u64();
                if sh >= (bits as u64) {
                    return Err(format!("EIR-Interp fault: proven `>>` shift {sh} >= bitwidth {bits}"));
                }
                let res = if signed {
                    ((a >> (sh as u32)) as i64 as u64) & mask
                } else {
                    ((lhs.as_u64() & mask) >> (sh as u32)) & mask
                };
                Ok(IVal::Scalar(res))
            }
            BinaryOp::Eq => Ok(IVal::Scalar(u64::from(a == b))),
            BinaryOp::NotEq => Ok(IVal::Scalar(u64::from(a != b))),
            BinaryOp::Lt => Ok(IVal::Scalar(u64::from(a < b))),
            BinaryOp::LtEq => Ok(IVal::Scalar(u64::from(a <= b))),
            BinaryOp::Gt => Ok(IVal::Scalar(u64::from(a > b))),
            BinaryOp::GtEq => Ok(IVal::Scalar(u64::from(a >= b))),
            other => Err(format!("EIR-Interp: unsupported binary op {other:?}")),
        }
    }

    fn pack_error_union(&self, eu_ty: TyId, ok_ty: TyId, err_code: u32, ok_val: &IVal) -> IVal {
        let has_ok = err_code == 0 && ok_ty != TyId::VOID && ok_ty != TyId::NEVER;
        let ok_off = if has_ok { eu_ok_offset(self.env, ok_ty) } else { 0 };
        let ok_sz = if has_ok { interp_ty_size(self.env, ok_ty) } else { 0 };
        let sz = x64_layout(self.env, eu_ty)
            .size
            .max(4)
            .max(if has_ok { ok_off + ok_sz } else { 0 });
        let mut buf = vec![0u8; sz];
        buf[0..4].copy_from_slice(&err_code.to_le_bytes());
        if has_ok {
            buf[ok_off..ok_off + ok_sz].copy_from_slice(&ok_val.to_bytes(ok_sz));
        }
        IVal::from_bytes(&buf)
    }

    fn eval_cast(&self, kind: CastKind, op: &IVal, from_ty: TyId, to_ty: TyId) -> Result<IVal, String> {
        match kind {
            CastKind::Widen | CastKind::Trunc | CastKind::AsU64 | CastKind::MutToConst => {
                if let (Ty::Float(fb), Ty::Float(tb)) = (self.env.get(from_ty), self.env.get(to_ty)) {
                    let as_f64 = match *fb {
                        16 => f16_bits_to_f32(op.as_u64() as u16) as f64,
                        32 => f32::from_bits(op.as_u64() as u32) as f64,
                        _ => f64::from_bits(op.as_u64()),
                    };
                    return Ok(match *tb {
                        16 => IVal::Scalar(f32_to_f16_bits(as_f64 as f32) as u64),
                        32 => IVal::Scalar((as_f64 as f32).to_bits() as u64),
                        _ => IVal::Scalar(as_f64.to_bits()),
                    });
                }
                let fsz = x64_layout(self.env, from_ty).size;
                let tsz = x64_layout(self.env, to_ty).size;
                if fsz == tsz && tsz > 8 {
                    return Ok(op.clone());
                }
                let fbits = int_bits(self.env, from_ty);
                let fsigned = is_signed_int(self.env, from_ty);
                let tbits = int_bits(self.env, to_ty);
                let ext = sign_extend_i128(op.as_u64(), fbits, fsigned);
                Ok(IVal::Scalar((ext as i64 as u64) & mask_bits(tbits)))
            }
            CastKind::Clamp => {
                let fbits = int_bits(self.env, from_ty);
                let fsigned = is_signed_int(self.env, from_ty);
                let tbits = int_bits(self.env, to_ty);
                let tsigned = is_signed_int(self.env, to_ty);
                let (tlo, thi) = int_bounds_for(tbits, tsigned);
                let val = sign_extend_i128(op.as_u64(), fbits, fsigned).clamp(tlo, thi);
                Ok(IVal::Scalar((val as i64 as u64) & mask_bits(tbits)))
            }
            CastKind::ToFloat => {
                let fbits = int_bits(self.env, from_ty);
                let fsigned = is_signed_int(self.env, from_ty);
                let iv = sign_extend_i128(op.as_u64(), fbits, fsigned);
                Ok(match self.env.get(to_ty) {
                    Ty::Float(16) => IVal::Scalar(f32_to_f16_bits(iv as f32) as u64),
                    Ty::Float(32) => IVal::Scalar((iv as f32).to_bits() as u64),
                    _ => IVal::Scalar((iv as f64).to_bits()),
                })
            }
            CastKind::TruncToInt => {
                let fv = match self.env.get(from_ty) {
                    Ty::Float(16) => f16_bits_to_f32(op.as_u64() as u16) as f64,
                    Ty::Float(32) => f32::from_bits(op.as_u64() as u32) as f64,
                    _ => f64::from_bits(op.as_u64()),
                };
                let tbits = int_bits(self.env, to_ty);
                let tsigned = is_signed_int(self.env, to_ty);
                let iv: u64 = if tsigned { (fv as i64) as u64 } else { fv as u64 };
                Ok(IVal::Scalar(iv & mask_bits(tbits)))
            }
            CastKind::Bitcast => {
                let sz = x64_layout(self.env, to_ty).size;
                Ok(IVal::from_bytes(&op.to_bytes(sz)))
            }
            CastKind::WrapOptional => {
                let inner = match self.env.get(to_ty) { Ty::Optional(i) => *i, _ => from_ty };
                if is_niche_id_opt(self.env, inner) {
                    Ok(op.clone())
                } else {
                    let sz = x64_layout(self.env, to_ty).size;
                    let mut buf = vec![0u8; sz];
                    buf[0] = 1;
                    let voff = opt_val_offset(self.env, inner);
                    let isz = x64_layout(self.env, inner).size;
                    buf[voff..voff + isz].copy_from_slice(&op.to_bytes(isz));
                    Ok(IVal::from_bytes(&buf))
                }
            }
            CastKind::WrapOk => {
                let ok_ty = match self.env.get(to_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => from_ty };
                Ok(self.pack_error_union(to_ty, ok_ty, 0, op))
            }
            CastKind::WrapErr => {
                let err_code = op.as_u64() as u32;
                Ok(self.pack_error_union(to_ty, TyId::VOID, err_code, &IVal::Void))
            }
            CastKind::ArrayBorrowToSlice => {
                let arr_len = match self.env.get(from_ty) {
                    Ty::Borrow { inner: BorrowInner::Single(at), .. } => match self.env.get(*at) {
                        Ty::Array { len, .. } => *len,
                        _ => 0,
                    },
                    _ => 0,
                };
                Ok(IVal::from_pair(op.as_u64(), arr_len))
            }
        }
    }

    fn eval_aggregate(
        &self,
        kind: &AggregateKind,
        elems: &[EirValue],
        dst_ty: TyId,
        vals: &[IVal],
    ) -> Result<IVal, String> {
        let sz = x64_layout(self.env, dst_ty).size;
        let mut buf = vec![0u8; sz];
        match kind {
            AggregateKind::Struct(_)
            | AggregateKind::AnonStruct(_)
            | AggregateKind::Tuple
            | AggregateKind::PoolRow { .. } => {
                let link_is_slot = matches!(kind, AggregateKind::PoolRow { .. });
                for (i, &ev) in elems.iter().enumerate() {
                    let (foff, fty) = struct_field_offset(self.env, dst_ty, i);
                    let fsz = x64_layout_impl(self.env, fty, link_is_slot).size;
                    buf[foff..foff + fsz].copy_from_slice(&vals[ev.0 as usize].to_bytes(fsz));
                }
            }
            AggregateKind::Array(elem_ty) | AggregateKind::Vec { elem: elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                for (i, &ev) in elems.iter().enumerate() {
                    let start = i * esz;
                    if start + esz <= buf.len() {
                        buf[start..start + esz].copy_from_slice(&vals[ev.0 as usize].to_bytes(esz));
                    }
                }
            }
            AggregateKind::Bitfield(def) => {
                let mut acc = 0u64;
                if let Some(m) = self.env.bitfields.get(def) {
                    for (i, &ev) in elems.iter().enumerate() {
                        if let Some(mb) = m.members.get(i) {
                            let width = mb.hi.saturating_sub(mb.lo).max(1);
                            let mask = mask_bits(width);
                            let v = vals[ev.0 as usize].as_u64() & mask;
                            acc |= v.wrapping_shl(mb.lo as u32);
                        }
                    }
                }
                return Ok(IVal::Scalar(acc));
            }
            AggregateKind::ArrayRepeat { elem, count } => {
                if let Some(&ev) = elems.first() {
                    let esz = x64_layout(self.env, *elem).size.max(1);
                    let ebytes = vals[ev.0 as usize].to_bytes(esz);
                    for i in 0..(*count as usize) {
                        let start = i * esz;
                        if start + esz <= buf.len() {
                            buf[start..start + esz].copy_from_slice(&ebytes);
                        }
                    }
                }
            }
            AggregateKind::UnionVariant { def, variant_idx } => {
                buf[0..4].copy_from_slice(&(*variant_idx as u32).to_le_bytes());
                if let Some(&ev) = elems.first()
                    && let Some(m) = self.env.unions.get(def)
                    && let Some(Some(pty)) = m.variants.get(*variant_idx).map(|v| v.payload)
                {
                    let p_off = union_payload_offset(self.env, *def);
                    let psz = x64_layout(self.env, pty).size;
                    buf[p_off..p_off + psz].copy_from_slice(&vals[ev.0 as usize].to_bytes(psz));
                }
            }
        }
        Ok(IVal::from_bytes(&buf))
    }

    fn exec_extern_fn(&mut self, name: crate::span::Symbol, args: &[IVal]) -> Result<IVal, String> {
        let s = self.interner.get(name);
        match s {
            "eon_rt_print_str" | "eon_rt_print_bytes" => {
                let (ptr, len) = args[0].as_pair();
                let bytes = self.mem.read(ptr, len as usize)?;
                self.stdout.push_str(&String::from_utf8_lossy(bytes));
                Ok(IVal::Void)
            }
            "eon_rt_eprint_str" | "eon_rt_eprint_bytes" => {
                let (ptr, len) = args[0].as_pair();
                let bytes = self.mem.read(ptr, len as usize)?;
                self.stderr.push_str(&String::from_utf8_lossy(bytes));
                Ok(IVal::Void)
            }
            "eon_rt_print_u64" => {
                let _ = write!(self.stdout, "{}", args[0].as_u64());
                Ok(IVal::Void)
            }
            "eon_rt_print_i64" => {
                let _ = write!(self.stdout, "{}", args[0].as_u64() as i64);
                Ok(IVal::Void)
            }
            "eon_rt_print_f32" => {
                let fv = f32::from_bits(args[0].as_u64() as u32);
                let _ = write!(self.stdout, "{fv:.2}");
                Ok(IVal::Void)
            }
            "eon_rt_print_nl" => {
                self.stdout.push('\n');
                Ok(IVal::Void)
            }
            "eon_rt_eprint_nl" => {
                self.stderr.push('\n');
                Ok(IVal::Void)
            }
            "eon_rt_now_ns" => {
                self.clock_tick += 100;
                Ok(IVal::Scalar(self.clock_tick))
            }
            other => Err(format!("EIR-Interp: unknown extern \"C\" function `{other}`")),
        }
    }

    fn resolve_ptr_arg(
        &mut self,
        f: &EirFunction,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        idx: usize,
        vals: &[IVal],
        slot_ptrs: &[u64],
    ) -> Result<(u64, Option<u64>), String> {
        if let Some(Some(slot)) = arg_slots.get(idx) {
            let sp = slot_ptrs[slot.0 as usize];
            if matches!(self.env.get(f.slots[slot.0 as usize].ty), Ty::Borrow { .. } | Ty::RawPtr { .. }) {
                return Ok((self.mem.read_u64(sp)?, None));
            }
            return Ok((sp, None));
        }
        let v = args[idx];
        let vty = f.value_tys[v.0 as usize];
        if matches!(self.env.get(vty), Ty::Borrow { .. } | Ty::RawPtr { .. }) {
            Ok((vals[v.0 as usize].as_u64(), None))
        } else {
            let sz = interp_ty_size(self.env, vty).max(1);
            let tp = self.mem.alloc_bytes(&vals[v.0 as usize].to_bytes(sz), false)?;
            Ok((tp, Some(tp)))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_builtin(
        &mut self,
        f: &EirFunction,
        op: &CfgBuiltinOp,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        ret_ty: TyId,
        vals: &[IVal],
        slot_ptrs: &[u64],
    ) -> Result<IVal, String> {
        match op {
            CfgBuiltinOp::ArenaInit => {
                let (cap_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let cap_bytes = vals[args[1].0 as usize].as_u64() as usize;
                let alloc_res = self.mem.alloc_raw_tracked(cap_ptr, cap_bytes)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                match alloc_res {
                    Some(base_ptr) => {
                        let mut arena_bytes = vec![0u8; 24];
                        arena_bytes[0..8].copy_from_slice(&base_ptr.to_le_bytes());
                        arena_bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
                        arena_bytes[16..24].copy_from_slice(&(cap_bytes as u64).to_le_bytes());
                        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                        Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(arena_bytes)))
                    }
                    None => Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)),
                }
            }
            CfgBuiltinOp::ArenaAlloc => {
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let req = vals[args[1].0 as usize].as_u64() as usize;
                let base = self.mem.read_u64(arena_ptr)?;
                let used = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 8)?)? as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 16)?)? as usize;
                let aligned_used = (used + 7) & !7;
                let res = if aligned_used.checked_add(req).is_some_and(|end| end <= cap) {
                    self.mem.write_u64(VmMem::ptr_add(arena_ptr, 8)?, (aligned_used + req) as u64)?;
                    let slice_ptr = VmMem::ptr_add(base, aligned_used)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_pair(slice_ptr, req as u64))
                } else {
                    self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::ArenaReset => {
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                self.mem.write_u64(VmMem::ptr_add(arena_ptr, 8)?, 0)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::ArenaDestroy => {
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let (cap_ptr, t1) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                let base = self.mem.read_u64(arena_ptr)?;
                let cap = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 16)?)? as usize;
                self.mem.free_raw_tracked(cap_ptr, base, cap)?;
                self.mem.write_u64(arena_ptr, 0)?;
                self.mem.write_u64(VmMem::ptr_add(arena_ptr, 8)?, 0)?;
                self.mem.write_u64(VmMem::ptr_add(arena_ptr, 16)?, 0)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::ArenaUsedBytes => {
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 8)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::ArenaCapacity => {
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 16)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::AllocCapLiveAllocs => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(p)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::AllocCapUsedBytes => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(p, 8)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::AllocCapPeakBytes => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(p, 16)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::AllocCapSubBudget => {
                let (parent_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let limit = vals[args[1].0 as usize].as_u64();
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let mut cap_bytes = vec![0u8; 40];
                cap_bytes[24..32].copy_from_slice(&limit.to_le_bytes());
                cap_bytes[32..40].copy_from_slice(&parent_ptr.to_le_bytes());
                Ok(IVal::Bytes(cap_bytes))
            }
            CfgBuiltinOp::ClockNowNs => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(p)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let now = (self.start_instant.elapsed().as_nanos() as u64).max(self.last_clock_ns);
                self.last_clock_ns = now;
                Ok(IVal::Scalar(now))
            }
            CfgBuiltinOp::ArgsLen => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let argc = self.mem.read_u64(p)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(argc))
            }
            CfgBuiltinOp::ArgsGet => {
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let idx = vals[args[1].0 as usize].as_u64();
                let argc = self.mem.read_u64(p)?;
                let table_ptr = self.mem.read_u64(VmMem::ptr_add(p, 8)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if idx < argc {
                    let entry = VmMem::ptr_add(table_ptr, (idx as usize) * 16)?;
                    let sptr = self.mem.read_u64(entry)?;
                    let slen = self.mem.read_u64(VmMem::ptr_add(entry, 8)?)?;
                    Ok(IVal::from_pair(sptr, slen))
                } else {
                    Ok(IVal::from_pair(0, u64::MAX))
                }
            }
            CfgBuiltinOp::FsOpenDir => {
                let (fs_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let fs_handle = self.mem.read_u64(fs_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let base = match self.resolve_dir_root(fs_handle) {
                    Ok(b) => b,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let path_str = match self.read_str_ival(&vals[args[1].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                if path_str.is_empty() || path_str.contains('\0') {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let target = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else {
                    base.join(&path_str)
                };
                let canon = match fs::canonicalize(&target) {
                    Ok(c) => c,
                    Err(e) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
                };
                if !canon.is_dir() {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let id = self.dir_roots.len() as u64;
                self.dir_roots.push(Some(canon));
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(id)))
            }
            CfgBuiltinOp::DirSubDir => {
                let (dir_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let dir_handle = self.mem.read_u64(dir_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(dir_handle) {
                    Ok(r) => r,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let rel = match self.read_str_ival(&vals[args[1].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let canon = match resolve_confined_path(&root, &rel, true) {
                    Ok(p) => p,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                if !canon.is_dir() {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let id = self.dir_roots.len() as u64;
                self.dir_roots.push(Some(canon));
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(id)))
            }
            CfgBuiltinOp::FsReadFile | CfgBuiltinOp::DirReadFile => {
                let confined = matches!(op, CfgBuiltinOp::DirReadFile);
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let (alloc_ptr, t1) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(handle) {
                    Ok(r) => r,
                    Err(code) => {
                        if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void));
                    }
                };
                let path_str = match self.read_str_ival(&vals[args[2].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => {
                        if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void));
                    }
                };
                let target = if confined {
                    match resolve_confined_path(&root, &path_str, true) {
                        Ok(p) => p,
                        Err(code) => {
                            if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                            return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void));
                        }
                    }
                } else {
                    if path_str.is_empty() || path_str.contains('\0') {
                        if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                    }
                    if Path::new(&path_str).is_absolute() {
                        PathBuf::from(&path_str)
                    } else {
                        root.join(&path_str)
                    }
                };
                let res = self.read_into_buf_u8(alloc_ptr, &target, ret_ty);
                if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                res
            }
            CfgBuiltinOp::FsWriteFile | CfgBuiltinOp::DirWriteFile | CfgBuiltinOp::FsWriteExecutable => {
                let confined = matches!(op, CfgBuiltinOp::DirWriteFile);
                let executable = matches!(op, CfgBuiltinOp::FsWriteExecutable);
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(handle) {
                    Ok(r) => r,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let path_str = match self.read_str_ival(&vals[args[1].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let target = if confined {
                    match resolve_confined_path(&root, &path_str, false) {
                        Ok(p) => p,
                        Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                    }
                } else {
                    if path_str.is_empty() || path_str.contains('\0') {
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                    }
                    if Path::new(&path_str).is_absolute() {
                        PathBuf::from(&path_str)
                    } else {
                        root.join(&path_str)
                    }
                };
                self.write_from_slice(&target, &vals[args[2].0 as usize], ret_ty, executable)
            }
            CfgBuiltinOp::FsIsDir => {
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let Ok(root) = self.resolve_dir_root(handle) else {
                    return Ok(IVal::Scalar(0));
                };
                let Ok(path_str) = self.read_str_ival(&vals[args[1].0 as usize])? else {
                    return Ok(IVal::Scalar(0));
                };
                if path_str.is_empty() || path_str.contains('\0') {
                    return Ok(IVal::Scalar(0));
                }
                let target = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else {
                    root.join(&path_str)
                };
                Ok(IVal::Scalar(u64::from(target.is_dir())))
            }
            CfgBuiltinOp::FsMkdirAll => {
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(handle) {
                    Ok(r) => r,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let path_str = match self.read_str_ival(&vals[args[1].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                if path_str.is_empty() || path_str.contains('\0') {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let target = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else {
                    root.join(&path_str)
                };
                match fs::create_dir_all(&target) {
                    Ok(()) => Ok(self.pack_error_union(ret_ty, TyId::VOID, 0, &IVal::Void)),
                    Err(e) => Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
                }
            }
            CfgBuiltinOp::FsListDir => {
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let (alloc_ptr, t1) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(handle) {
                    Ok(r) => r,
                    Err(code) => {
                        if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void));
                    }
                };
                let path_str = match self.read_str_ival(&vals[args[2].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => {
                        if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void));
                    }
                };
                if path_str.is_empty() || path_str.contains('\0') {
                    if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let target = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else {
                    root.join(&path_str)
                };
                let res = self.list_dir_into_buf_u8(alloc_ptr, &target, ret_ty);
                if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                res
            }
            CfgBuiltinOp::FsSpawnWait => {
                let (recv_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(recv_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let root = match self.resolve_dir_root(handle) {
                    Ok(r) => r,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                let path_str = match self.read_str_ival(&vals[args[1].0 as usize])? {
                    Ok(s) => s,
                    Err(code) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, code, &IVal::Void)),
                };
                if path_str.is_empty() || path_str.contains('\0') {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let target = if Path::new(&path_str).is_absolute() {
                    PathBuf::from(&path_str)
                } else {
                    root.join(&path_str)
                };
                let (aptr, alen) = vals[args[2].0 as usize].as_pair();
                let args_bytes = self.mem.read(aptr, alen as usize)?;
                let mut cmd = std::process::Command::new(&target);
                if !args_bytes.is_empty() {
                    let trimmed = args_bytes.strip_suffix(&[0]).unwrap_or(args_bytes);
                    let parts: Vec<&[u8]> = trimmed.split(|&b| b == 0).collect();
                    if let Some((first, rest)) = parts.split_first() {
                        #[cfg(unix)]
                        cmd.arg0(std::ffi::OsStr::from_bytes(first));
                        #[cfg(not(unix))]
                        let _ = first;
                        for a in rest {
                            #[cfg(unix)]
                            cmd.arg(std::ffi::OsStr::from_bytes(a));
                            #[cfg(not(unix))]
                            cmd.arg(String::from_utf8_lossy(a).into_owned());
                        }
                    }
                }
                match cmd.status() {
                    Ok(status) => {
                        let code: i32 = if let Some(c) = status.code() {
                            c
                        } else {
                            #[cfg(unix)]
                            {
                                status.signal().map_or(1, |sig| 128 + sig)
                            }
                            #[cfg(not(unix))]
                            {
                                1
                            }
                        };
                        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => TyId::I32 };
                        Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(code as u32 as u64)))
                    }
                    Err(e) => Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
                }
            }
            CfgBuiltinOp::DirClose => {
                let (dir_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let handle = self.mem.read_u64(dir_ptr)? as usize;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if handle > 1 && handle < self.dir_roots.len() {
                    self.dir_roots[handle] = None;
                }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::NetRestrictHost => {
                let (net_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(net_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let (hptr, hlen) = vals[args[1].0 as usize].as_pair();
                let host = String::from_utf8_lossy(self.mem.read(hptr, hlen as usize)?).into_owned();
                let handle = self.net_hosts.len() as u64;
                self.net_hosts.push(host);
                Ok(IVal::Scalar(handle))
            }
            CfgBuiltinOp::IoNopRoundtrip => {
                let (io_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(io_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let count = vals[args[1].0 as usize].as_u64();
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => TyId::USIZE };
                Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(count)))
            }
            CfgBuiltinOp::IoRunFibers => {
                let (io_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(io_ptr)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let count = vals[args[1].0 as usize].as_u64();
                let stack_bytes = vals[args[2].0 as usize].as_u64();
                if stack_bytes == 0 {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 16, &IVal::Void));
                }
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => TyId::USIZE };
                Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(count)))
            }
            CfgBuiltinOp::BuildAddModule => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let _name = self.read_build_str(&vals[args[1].0 as usize])?;
                let _root = self.read_build_str(&vals[args[2].0 as usize])?;
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildAddExe => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let name = self.read_build_str(&vals[args[1].0 as usize])?;
                let root = self.read_build_str(&vals[args[2].0 as usize])?;
                self.build_plan.targets.push(BuildTarget {
                    name,
                    root,
                    kind: BuildTargetKind::Exe,
                });
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildAddSharedLib => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let name = self.read_build_str(&vals[args[1].0 as usize])?;
                let root = self.read_build_str(&vals[args[2].0 as usize])?;
                self.build_plan.targets.push(BuildTarget {
                    name,
                    root,
                    kind: BuildTargetKind::Shared,
                });
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildSetOutDir => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                self.build_plan.out_dir = self.read_build_str(&vals[args[1].0 as usize])?;
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildSetSanitize => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                self.build_plan.sanitize = vals[args[1].0 as usize].as_u64() != 0;
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildSetAuditTcb => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                self.build_plan.audit_tcb = vals[args[1].0 as usize].as_u64() != 0;
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildSetOptLevel => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let _level = vals[args[1].0 as usize].as_u64();
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildLinkLib => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let lib = self.read_build_str(&vals[args[1].0 as usize])?;
                self.build_plan.needed_libs.push(lib);
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildAllowHal => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let mod_name = self.read_build_str(&vals[args[1].0 as usize])?;
                self.build_plan.allow_hal.insert(mod_name);
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BuildOutPath => {
                let (bp, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let _ = self.mem.read_u64(bp)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let rel = self.read_build_str(&vals[args[1].0 as usize])?;
                validate_build_rel_path(&rel, "relative out_path")?;
                let out_dir = if self.build_plan.out_dir.is_empty() {
                    "out"
                } else {
                    validate_build_rel_path(&self.build_plan.out_dir, "out_dir")?;
                    self.build_plan.out_dir.as_str()
                };
                let joined = format!("{}/{}", out_dir.trim_end_matches('/'), rel);
                let ptr = self.intern_ro_bytes(joined.as_bytes())?;
                Ok(IVal::from_pair(ptr, joined.len() as u64))
            }
            CfgBuiltinOp::ScheduleRun => {
                self.exec_schedule_run(f, args, arg_slots, vals, slot_ptrs)?;
                Ok(IVal::Void)
            }
            CfgBuiltinOp::MathSqrt(ty) => {
                let x = vals[args[0].0 as usize].as_u64();
                let res = match self.env.get(*ty) {
                    Ty::Float(16) => f32_to_f16_bits(f16_bits_to_f32(x as u16).sqrt()) as u64,
                    Ty::Float(32) => f32::from_bits(x as u32).sqrt().to_bits() as u64,
                    _ => f64::from_bits(x).sqrt().to_bits(),
                };
                Ok(IVal::Scalar(res))
            }
            CfgBuiltinOp::MathFma(ty) => {
                let xa = vals[args[0].0 as usize].as_u64();
                let xb = vals[args[1].0 as usize].as_u64();
                let xc = vals[args[2].0 as usize].as_u64();
                let res = match self.env.get(*ty) {
                    Ty::Float(16) => {
                        let a = f16_bits_to_f32(xa as u16);
                        let b = f16_bits_to_f32(xb as u16);
                        let c = f16_bits_to_f32(xc as u16);
                        f32_to_f16_bits(a.mul_add(b, c)) as u64
                    }
                    Ty::Float(32) => {
                        let a = f32::from_bits(xa as u32);
                        let b = f32::from_bits(xb as u32);
                        let c = f32::from_bits(xc as u32);
                        a.mul_add(b, c).to_bits() as u64
                    }
                    _ => {
                        let a = f64::from_bits(xa);
                        let b = f64::from_bits(xb);
                        let c = f64::from_bits(xc);
                        a.mul_add(b, c).to_bits()
                    }
                };
                Ok(IVal::Scalar(res))
            }
            CfgBuiltinOp::MathLzcnt(ty) => {
                let v = vals[args[0].0 as usize].as_u64();
                let res = match int_bits(self.env, *ty) {
                    8 => (v as u8).leading_zeros(),
                    16 => (v as u16).leading_zeros(),
                    32 => (v as u32).leading_zeros(),
                    _ => v.leading_zeros(),
                };
                Ok(IVal::Scalar(res as u64))
            }
            CfgBuiltinOp::MathTzcnt(ty) => {
                let v = vals[args[0].0 as usize].as_u64();
                let res = match int_bits(self.env, *ty) {
                    8 => (v as u8).trailing_zeros(),
                    16 => (v as u16).trailing_zeros(),
                    32 => (v as u32).trailing_zeros(),
                    _ => v.trailing_zeros(),
                };
                Ok(IVal::Scalar(res as u64))
            }
            CfgBuiltinOp::MathPopcnt(ty) => {
                let v = vals[args[0].0 as usize].as_u64();
                let res = match int_bits(self.env, *ty) {
                    8 => (v as u8).count_ones(),
                    16 => (v as u16).count_ones(),
                    32 => (v as u32).count_ones(),
                    _ => v.count_ones(),
                };
                Ok(IVal::Scalar(res as u64))
            }
            CfgBuiltinOp::MathBswap(ty) => {
                let v = vals[args[0].0 as usize].as_u64();
                let res = match int_bits(self.env, *ty) {
                    8 => (v as u8) as u64,
                    16 => (v as u16).swap_bytes() as u64,
                    32 => (v as u32).swap_bytes() as u64,
                    _ => v.swap_bytes(),
                };
                Ok(IVal::Scalar(res))
            }
            CfgBuiltinOp::ArenaPoolInit(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (arena_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let cap = vals[args[1].0 as usize].as_u64() as usize;
                let base = self.mem.read_u64(arena_ptr)?;
                let mut used = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 8)?)? as usize;
                let arena_cap = self.mem.read_u64(VmMem::ptr_add(arena_ptr, 16)?)? as usize;
                let mut pool_bytes = vec![0u8; (ncols + 2) * 8];
                let mut oom = false;
                for (i, c) in cols.iter().enumerate() {
                    let esz = x64_layout_impl(self.env, c.ty, false).size.max(1);
                    let bytes_needed = cap.saturating_mul(esz);
                    let aligned = (used + 63) & !63;
                    if aligned.checked_add(bytes_needed).is_some_and(|e| e <= arena_cap) {
                        let cptr = VmMem::ptr_add(base, aligned)?;
                        pool_bytes[i * 8..(i + 1) * 8].copy_from_slice(&cptr.to_le_bytes());
                        used = aligned + bytes_needed;
                    } else {
                        oom = true;
                        break;
                    }
                }
                if !oom {
                    self.mem.write_u64(VmMem::ptr_add(arena_ptr, 8)?, used as u64)?;
                }
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if oom {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void))
                } else {
                    pool_bytes[ncols * 8..(ncols + 1) * 8].copy_from_slice(&0u64.to_le_bytes());
                    pool_bytes[(ncols + 1) * 8..(ncols + 2) * 8].copy_from_slice(&(cap as u64).to_le_bytes());
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(pool_bytes)))
                }
            }
            CfgBuiltinOp::ArenaPoolPush(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (pool_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let len = self.mem.read_u64(VmMem::ptr_add(pool_ptr, ncols * 8)?)? as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(pool_ptr, (ncols + 1) * 8)?)? as usize;
                let res = if len >= cap {
                    self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)
                } else {
                    let row_sz = x64_layout(self.env, *elem_ty).size;
                    let row_bytes = vals[args[1].0 as usize].to_bytes(row_sz);
                    for (i, c) in cols.iter().enumerate() {
                        let (foff, _) = struct_field_offset(self.env, *elem_ty, i);
                        let col_esz = x64_layout_impl(self.env, c.ty, false).size.max(1);
                        let col_ptr = self.mem.read_u64(VmMem::ptr_add(pool_ptr, i * 8)?)?;
                        let dst_addr = VmMem::ptr_add(col_ptr, len * col_esz)?;
                        self.mem.write(dst_addr, &row_bytes[foff..foff + col_esz])?;
                    }
                    self.mem.write_u64(VmMem::ptr_add(pool_ptr, ncols * 8)?, (len + 1) as u64)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(len as u64))
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::ArenaPoolIdAt(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let (pool_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let len = self.mem.read_u64(VmMem::ptr_add(pool_ptr, ncols * 8)?)?;
                let idx = vals[args[1].0 as usize].as_u64();
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if idx < len {
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(idx)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::ArenaPoolReset(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let sz = (ncols + 2) * 8;
                let mut pool_bytes = vals[args[0].0 as usize].to_bytes(sz);
                pool_bytes[ncols * 8..(ncols + 1) * 8].copy_from_slice(&0u64.to_le_bytes());
                Ok(IVal::Bytes(pool_bytes))
            }
            CfgBuiltinOp::ArenaPoolLen => {
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let ncols = match self.env.get(deref) { Ty::Builtin(_, ga) => pool_col_count(self.env, ga), _ => 0 };
                let (pool_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let len = self.mem.read_u64(VmMem::ptr_add(pool_ptr, ncols * 8)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(len))
            }
            CfgBuiltinOp::SlotPoolInit(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (cap_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let cap = (vals[args[1].0 as usize].as_u64() as usize).max(4);
                let mut sp_bytes = vec![0u8; (ncols + 5) * 8];
                let mut allocs: Vec<(u64, usize)> = Vec::with_capacity(ncols + 2);
                let rollback = |mem: &mut VmMem, cap_ptr: u64, allocs: Vec<(u64, usize)>| -> Result<(), String> {
                    for (p, sz) in allocs {
                        mem.free_raw_tracked(cap_ptr, p, sz)?;
                    }
                    Ok(())
                };
                let Some(gen_ptr) = self.mem.alloc_raw_tracked(cap_ptr, cap * 4)? else {
                    if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                };
                allocs.push((gen_ptr, cap * 4));
                let Some(free_ptr) = self.mem.alloc_raw_tracked(cap_ptr, cap * 4)? else {
                    rollback(&mut self.mem, cap_ptr, allocs)?;
                    if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                };
                allocs.push((free_ptr, cap * 4));
                for (i, c) in cols.iter().enumerate() {
                    let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                    let Some(cptr) = self.mem.alloc_raw_tracked(cap_ptr, cap * esz)? else {
                        rollback(&mut self.mem, cap_ptr, allocs)?;
                        if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                    };
                    allocs.push((cptr, cap * esz));
                    sp_bytes[i * 8..(i + 1) * 8].copy_from_slice(&cptr.to_le_bytes());
                }
                for i in 0..cap {
                    let slot_idx = (cap - 1 - i) as u32;
                    self.mem.write_u32(VmMem::ptr_add(free_ptr, i * 4)?, slot_idx)?;
                }
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                sp_bytes[ncols * 8..(ncols + 1) * 8].copy_from_slice(&gen_ptr.to_le_bytes());
                sp_bytes[(ncols + 1) * 8..(ncols + 2) * 8].copy_from_slice(&free_ptr.to_le_bytes());
                sp_bytes[(ncols + 2) * 8..(ncols + 3) * 8].copy_from_slice(&0u64.to_le_bytes()); // len
                sp_bytes[(ncols + 3) * 8..(ncols + 4) * 8].copy_from_slice(&(cap as u64).to_le_bytes());
                sp_bytes[(ncols + 4) * 8..(ncols + 5) * 8].copy_from_slice(&(cap as u64).to_le_bytes()); // free_top
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(sp_bytes)))
            }
            CfgBuiltinOp::SlotPoolInsert(elem_ty) | CfgBuiltinOp::SlotPoolTryInsert(elem_ty) => {
                let is_try = matches!(op, CfgBuiltinOp::SlotPoolTryInsert(_));
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let mut free_top = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 4) * 8)?)? as usize;
                let mut cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;

                // Grow if full and not `try_insert`.
                if free_top == 0 && !is_try {
                    let (cap_ptr, t_cap) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                    let new_cap = (cap * 2).max(4);
                    let old_gen = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                    let old_free = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 1) * 8)?)?;
                    let mut new_allocs: Vec<(u64, usize)> = Vec::with_capacity(ncols + 2);
                    let rollback = |mem: &mut VmMem, cap_ptr: u64, allocs: Vec<(u64, usize)>| -> Result<(), String> {
                        for (p, sz) in allocs {
                            mem.free_raw_tracked(cap_ptr, p, sz)?;
                        }
                        Ok(())
                    };
                    let Some(new_gen) = self.mem.alloc_raw_tracked(cap_ptr, new_cap * 4)? else {
                        if let Some(tc) = t_cap { let _ = self.mem.free(tc); }
                        if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                    };
                    new_allocs.push((new_gen, new_cap * 4));
                    let Some(new_free) = self.mem.alloc_raw_tracked(cap_ptr, new_cap * 4)? else {
                        rollback(&mut self.mem, cap_ptr, new_allocs)?;
                        if let Some(tc) = t_cap { let _ = self.mem.free(tc); }
                        if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                        return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                    };
                    new_allocs.push((new_free, new_cap * 4));
                    let mut new_cols: Vec<(usize, u64, u64, usize)> = Vec::with_capacity(ncols);
                    for (ci, c) in cols.iter().enumerate() {
                        let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                        let old_c = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ci * 8)?)?;
                        let Some(new_c) = self.mem.alloc_raw_tracked(cap_ptr, new_cap * esz)? else {
                            rollback(&mut self.mem, cap_ptr, new_allocs)?;
                            if let Some(tc) = t_cap { let _ = self.mem.free(tc); }
                            if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                            return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
                        };
                        new_allocs.push((new_c, new_cap * esz));
                        new_cols.push((ci, old_c, new_c, esz));
                    }

                    let old_gen_bytes = self.mem.read(old_gen, cap * 4)?.to_vec();
                    self.mem.write(new_gen, &old_gen_bytes)?;
                    self.mem.free_raw_tracked(cap_ptr, old_gen, cap * 4)?;
                    self.mem.free_raw_tracked(cap_ptr, old_free, cap * 4)?;
                    for i in 0..(new_cap - cap) {
                        let sidx = (new_cap - 1 - i) as u32;
                        self.mem.write_u32(VmMem::ptr_add(new_free, i * 4)?, sidx)?;
                    }
                    for (ci, old_c, new_c, esz) in new_cols {
                        let old_cb = self.mem.read(old_c, cap * esz)?.to_vec();
                        self.mem.write(new_c, &old_cb)?;
                        self.mem.free_raw_tracked(cap_ptr, old_c, cap * esz)?;
                        self.mem.write_u64(VmMem::ptr_add(sp_ptr, ci * 8)?, new_c)?;
                    }
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?, new_gen)?;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 1) * 8)?, new_free)?;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?, new_cap as u64)?;
                    free_top = new_cap - cap;
                    cap = new_cap;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 4) * 8)?, free_top as u64)?;
                    if let Some(tc) = t_cap { let _ = self.mem.free(tc); }
                }
                let _ = cap;
                let res = if free_top == 0 {
                    self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)
                } else {
                    let row_arg_idx = if is_try { 1 } else { 2 };
                    let free_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 1) * 8)?)?;
                    let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                    free_top -= 1;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 4) * 8)?, free_top as u64)?;
                    let idx = self.mem.read_u32(VmMem::ptr_add(free_ptr, free_top * 4)?)? as usize;
                    let old_g = self.mem.read_u32(VmMem::ptr_add(gen_ptr, idx * 4)?)?;
                    let new_g = old_g.wrapping_add(1) | 1;
                    self.mem.write_u32(VmMem::ptr_add(gen_ptr, idx * 4)?, new_g)?;

                    let row_sz = x64_layout(self.env, *elem_ty).size;
                    let row_bytes = vals[args[row_arg_idx].0 as usize].to_bytes(row_sz);
                    for (ci, c) in cols.iter().enumerate() {
                        let (foff, _) = struct_field_offset(self.env, *elem_ty, ci);
                        let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                        let cptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ci * 8)?)?;
                        self.mem.write(VmMem::ptr_add(cptr, idx * esz)?, &row_bytes[foff..foff + esz])?;
                    }
                    let len = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 2) * 8)?)?;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 2) * 8)?, len + 1)?;
                    let mut gid = [0u8; 8];
                    gid[0..4].copy_from_slice(&(idx as u32).to_le_bytes());
                    gid[4..8].copy_from_slice(&new_g.to_le_bytes());
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_bytes(&gid))
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::SlotPoolRemove(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let gid = vals[args[1].0 as usize].to_bytes(8);
                let idx = u32::from_le_bytes(gid[0..4].try_into().unwrap()) as usize;
                let slot_gen = u32::from_le_bytes(gid[4..8].try_into().unwrap());
                let cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;
                let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                let alive = slot_gen != 0
                    && (slot_gen & 1) != 0
                    && idx < cap
                    && self.mem.read_u32(VmMem::ptr_add(gen_ptr, idx * 4)?)? == slot_gen;
                let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => *elem_ty };
                let res = if !alive {
                    self.pack_error_union(ret_ty, TyId::VOID, 11, &IVal::Void)
                } else {
                    let row_sz = x64_layout(self.env, ok_ty).size.max(1);
                    let mut row_bytes = vec![0u8; row_sz];
                    for (ci, c) in cols.iter().enumerate() {
                        let (foff, _) = struct_field_offset(self.env, ok_ty, ci);
                        let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                        let cptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ci * 8)?)?;
                        let col_bytes = self.mem.read(VmMem::ptr_add(cptr, idx * esz)?, esz)?;
                        if foff + esz <= row_bytes.len() {
                            row_bytes[foff..foff + esz].copy_from_slice(col_bytes);
                        }
                    }
                    self.mem.write_u32(VmMem::ptr_add(gen_ptr, idx * 4)?, slot_gen.wrapping_add(1))?;
                    let free_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 1) * 8)?)?;
                    let free_top = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 4) * 8)?)? as usize;
                    self.mem.write_u32(VmMem::ptr_add(free_ptr, free_top * 4)?, idx as u32)?;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 4) * 8)?, (free_top + 1) as u64)?;
                    let len = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 2) * 8)?)?;
                    self.mem.write_u64(VmMem::ptr_add(sp_ptr, (ncols + 2) * 8)?, len.saturating_sub(1))?;
                    self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_bytes(&row_bytes))
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::SlotPoolGet { elem_ty, .. } | CfgBuiltinOp::SlotPoolAt(elem_ty) => {
                let returns_eu = matches!(op, CfgBuiltinOp::SlotPoolAt(_));
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let gid = vals[args[1].0 as usize].to_bytes(8);
                let idx = u32::from_le_bytes(gid[0..4].try_into().unwrap()) as usize;
                let slot_gen = u32::from_le_bytes(gid[4..8].try_into().unwrap());
                let cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;
                let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                let alive = slot_gen != 0
                    && (slot_gen & 1) != 0
                    && idx < cap
                    && self.mem.read_u32(VmMem::ptr_add(gen_ptr, idx * 4)?)? == slot_gen;
                let rv = IVal::from_pair(sp_ptr, idx as u64);
                let res = if returns_eu {
                    if alive {
                        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                        self.pack_error_union(ret_ty, ok_ty, 0, &rv)
                    } else {
                        self.pack_error_union(ret_ty, TyId::VOID, 11, &IVal::Void)
                    }
                } else if alive {
                    let inner = match self.env.get(ret_ty) { Ty::Optional(i) => *i, _ => ret_ty };
                    self.eval_cast(CastKind::WrapOptional, &rv, inner, ret_ty)?
                } else {
                    IVal::from_bytes(&vec![0u8; x64_layout(self.env, ret_ty).size])
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::SlotPoolLen | CfgBuiltinOp::SlotPoolCap => {
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let ncols = match self.env.get(deref) { Ty::Builtin(_, ga) => pool_col_count(self.env, ga), _ => 0 };
                let off = if matches!(op, CfgBuiltinOp::SlotPoolLen) { (ncols + 2) * 8 } else { (ncols + 3) * 8 };
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(sp_ptr, off)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::SlotPoolIsAlive(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;
                let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                let alive = idx < cap && (self.mem.read_u32(VmMem::ptr_add(gen_ptr, idx * 4)?)? & 1) != 0;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(u64::from(alive)))
            }
            CfgBuiltinOp::SlotPoolIdAt(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let ncols = self.env.pools.get(&pdef).map_or(0, |m| m.columns.len());
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;
                let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                let slot_gen = if idx < cap { self.mem.read_u32(VmMem::ptr_add(gen_ptr, idx * 4)?)? } else { 0 };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                let mut gid = [0u8; 8];
                gid[0..4].copy_from_slice(&(idx as u32).to_le_bytes());
                gid[4..8].copy_from_slice(&slot_gen.to_le_bytes());
                Ok(IVal::from_bytes(&gid))
            }
            CfgBuiltinOp::SlotPoolDestroy(elem_ty) => {
                let pdef = match self.env.get(*elem_ty) { Ty::Pool { def, .. } => *def, _ => DefId(0) };
                let cols = self.env.pools.get(&pdef).map(|m| m.columns.clone()).unwrap_or_default();
                let ncols = cols.len();
                let (sp_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let (cap_ptr, t1) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                let cap = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 3) * 8)?)? as usize;
                for (ci, c) in cols.iter().enumerate() {
                    let esz = x64_layout_impl(self.env, c.ty, true).size.max(1);
                    let cptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ci * 8)?)?;
                    self.mem.free_raw_tracked(cap_ptr, cptr, cap * esz)?;
                }
                let gen_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, ncols * 8)?)?;
                let free_ptr = self.mem.read_u64(VmMem::ptr_add(sp_ptr, (ncols + 1) * 8)?)?;
                self.mem.free_raw_tracked(cap_ptr, gen_ptr, cap * 4)?;
                self.mem.free_raw_tracked(cap_ptr, free_ptr, cap * 4)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BufInit(elem_ty) => {
                let (cap_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let cap = (vals[args[1].0 as usize].as_u64() as usize).max(1);
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let alloc_res = self.mem.alloc_raw_tracked(cap_ptr, cap * esz)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                match alloc_res {
                    Some(data_ptr) => {
                        let mut buf = vec![0u8; 24];
                        buf[0..8].copy_from_slice(&data_ptr.to_le_bytes());
                        buf[8..16].copy_from_slice(&0u64.to_le_bytes());
                        buf[16..24].copy_from_slice(&(cap as u64).to_le_bytes());
                        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                        Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(buf)))
                    }
                    None => Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)),
                }
            }
            CfgBuiltinOp::BufPush(elem_ty) => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let val_arg = if args.len() >= 3 { args[2] } else { args[1] };
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 16)?)? as usize;
                let res = if len >= cap {
                    self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)
                } else {
                    let elem_bytes = vals[val_arg.0 as usize].to_bytes(esz);
                    self.mem.write(VmMem::ptr_add(data_ptr, len * esz)?, &elem_bytes)?;
                    self.mem.write_u64(VmMem::ptr_add(buf_ptr, 8)?, (len + 1) as u64)?;
                    self.pack_error_union(ret_ty, TyId::VOID, 0, &IVal::Void)
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::BufPushSlice(elem_ty) => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let slice_arg = if args.len() >= 3 { args[2] } else { args[1] };
                let (sptr, slen_u64) = vals[slice_arg.0 as usize].as_pair();
                let slen = slen_u64 as usize;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let cap = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 16)?)? as usize;
                let res = if cap.saturating_sub(len) < slen {
                    self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void)
                } else {
                    if slen > 0 {
                        let nbytes = slen * esz;
                        let bytes = self.mem.read(sptr, nbytes)?.to_vec();
                        self.mem.write(VmMem::ptr_add(data_ptr, len * esz)?, &bytes)?;
                    }
                    self.mem.write_u64(VmMem::ptr_add(buf_ptr, 8)?, (len + slen) as u64)?;
                    self.pack_error_union(ret_ty, TyId::VOID, 0, &IVal::Void)
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::BufPop(elem_ty) => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let res = if len == 0 {
                    self.opt_null_val(ret_ty)
                } else {
                    let new_len = len - 1;
                    self.mem.write_u64(VmMem::ptr_add(buf_ptr, 8)?, new_len as u64)?;
                    let b = self.mem.read(VmMem::ptr_add(data_ptr, new_len * esz)?, esz)?.to_vec();
                    self.eval_cast(CastKind::WrapOptional, &IVal::from_bytes(&b), *elem_ty, ret_ty)?
                };
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(res)
            }
            CfgBuiltinOp::BufLen | CfgBuiltinOp::BufCap => {
                let off = if matches!(op, CfgBuiltinOp::BufLen) { 8 } else { 16 };
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = self.mem.read_u64(VmMem::ptr_add(buf_ptr, off)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(v))
            }
            CfgBuiltinOp::BufClear => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                self.mem.write_u64(VmMem::ptr_add(buf_ptr, 8)?, 0)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::BufAsSlice { .. } => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::from_pair(data_ptr, len))
            }
            CfgBuiltinOp::BufAt { elem_ty, .. } => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if idx < len {
                    let elem_ptr = VmMem::ptr_add(data_ptr, idx * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(elem_ptr)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::BufGet { elem_ty, .. } => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if idx < len {
                    Ok(IVal::Scalar(VmMem::ptr_add(data_ptr, idx * esz)?))
                } else {
                    Ok(IVal::Scalar(0))
                }
            }
            CfgBuiltinOp::BufSub { elem_ty, .. } => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let len = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 8)?)? as usize;
                let lo = vals[args[1].0 as usize].as_u64() as usize;
                let hi = vals[args[2].0 as usize].as_u64() as usize;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if lo <= hi && hi <= len {
                    let sub_ptr = VmMem::ptr_add(data_ptr, lo * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_pair(sub_ptr, (hi - lo) as u64)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::BufDestroy(elem_ty) => {
                let (buf_ptr, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let (cap_ptr, t1) = self.resolve_ptr_arg(f, args, arg_slots, 1, vals, slot_ptrs)?;
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let data_ptr = self.mem.read_u64(buf_ptr)?;
                let cap = self.mem.read_u64(VmMem::ptr_add(buf_ptr, 16)?)? as usize;
                self.mem.free_raw_tracked(cap_ptr, data_ptr, cap * esz)?;
                self.mem.write_u64(buf_ptr, 0)?;
                self.mem.write_u64(VmMem::ptr_add(buf_ptr, 8)?, 0)?;
                self.mem.write_u64(VmMem::ptr_add(buf_ptr, 16)?, 0)?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                if let Some(tp) = t1 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::AtomicNew(_) => Ok(vals[args[0].0 as usize].clone()),
            CfgBuiltinOp::AtomicLoad(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = IVal::from_bytes(self.mem.read(p, sz)?);
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(v)
            }
            CfgBuiltinOp::AtomicStore(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                self.mem.write(p, &vals[args[1].0 as usize].to_bytes(sz))?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::AtomicFetchAdd(el) | CfgBuiltinOp::AtomicFetchSub(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let bits = int_bits(self.env, *el);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let old = IVal::from_bytes(self.mem.read(p, sz)?).as_u64();
                let delta = vals[args[1].0 as usize].as_u64();
                let new_v = if matches!(op, CfgBuiltinOp::AtomicFetchAdd(_)) {
                    old.wrapping_add(delta)
                } else {
                    old.wrapping_sub(delta)
                } & mask_bits(bits);
                self.mem.write(p, &new_v.to_le_bytes()[..sz])?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(old & mask_bits(bits)))
            }
            CfgBuiltinOp::AtomicSwap(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let old = IVal::from_bytes(self.mem.read(p, sz)?);
                self.mem.write(p, &vals[args[1].0 as usize].to_bytes(sz))?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(old)
            }
            CfgBuiltinOp::AtomicCas(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let cur = self.mem.read(p, sz)?;
                let exp = vals[args[1].0 as usize].to_bytes(sz);
                let ok = cur == exp.as_slice();
                if ok {
                    self.mem.write(p, &vals[args[2].0 as usize].to_bytes(sz))?;
                }
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Scalar(u64::from(ok)))
            }
            CfgBuiltinOp::MmioRead(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let v = IVal::from_bytes(self.mem.read(p, sz)?);
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(v)
            }
            CfgBuiltinOp::MmioWrite(el) => {
                let sz = x64_layout(self.env, *el).size.clamp(1, 8);
                let (p, t0) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                self.mem.write(p, &vals[args[1].0 as usize].to_bytes(sz))?;
                if let Some(tp) = t0 { let _ = self.mem.free(tp); }
                Ok(IVal::Void)
            }
            CfgBuiltinOp::RawPtrFromUsize(_) | CfgBuiltinOp::RawPtrAsUsize => {
                Ok(IVal::Scalar(vals[args[0].0 as usize].as_u64()))
            }
            CfgBuiltinOp::ScheduleBuild => Ok(IVal::Scalar(0)),
            CfgBuiltinOp::ParReduce { elem_ty, op: op_sym } => {
                self.exec_par_reduce(*elem_ty, *op_sym, args, vals)
            }
            CfgBuiltinOp::SliceLen | CfgBuiltinOp::StrLen => {
                let (_, len) = vals[args[0].0 as usize].as_pair();
                Ok(IVal::Scalar(len))
            }
            CfgBuiltinOp::ArrayLen => {
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len, _ => 0 };
                Ok(IVal::Scalar(len))
            }
            CfgBuiltinOp::StrAsBytes => Ok(vals[args[0].0 as usize].clone()),
            CfgBuiltinOp::BytesAsStr => {
                let (ptr, len) = vals[args[0].0 as usize].as_pair();
                let bytes = self.mem.read(ptr, len as usize)?;
                if std::str::from_utf8(bytes).is_ok() {
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_pair(ptr, len)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 5, &IVal::Void))
                }
            }
            CfgBuiltinOp::SliceAt { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let (ptr, len) = vals[args[0].0 as usize].as_pair();
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if idx < (len as usize) {
                    let ep = VmMem::ptr_add(ptr, idx * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(ep)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::ArrayAt { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len as usize, _ => 0 };
                let (ptr, _) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if idx < len {
                    let ep = VmMem::ptr_add(ptr, idx * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Scalar(ep)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::SliceGet { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let (ptr, len) = vals[args[0].0 as usize].as_pair();
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if idx < (len as usize) {
                    Ok(IVal::Scalar(VmMem::ptr_add(ptr, idx * esz)?))
                } else {
                    Ok(IVal::Scalar(0))
                }
            }
            CfgBuiltinOp::ArrayGet { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len as usize, _ => 0 };
                let (ptr, _) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let idx = vals[args[1].0 as usize].as_u64() as usize;
                if idx < len {
                    Ok(IVal::Scalar(VmMem::ptr_add(ptr, idx * esz)?))
                } else {
                    Ok(IVal::Scalar(0))
                }
            }
            CfgBuiltinOp::SliceSub { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let (ptr, len) = vals[args[0].0 as usize].as_pair();
                let lo = vals[args[1].0 as usize].as_u64() as usize;
                let hi = vals[args[2].0 as usize].as_u64() as usize;
                if lo <= hi && hi <= (len as usize) {
                    let sub_ptr = VmMem::ptr_add(ptr, lo * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_pair(sub_ptr, (hi - lo) as u64)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::ArraySub { elem_ty, .. } => {
                let esz = x64_layout(self.env, *elem_ty).size.max(1);
                let ty0 = f.value_tys[args[0].0 as usize];
                let deref = match self.env.get(ty0) { Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t, _ => ty0 };
                let len = match self.env.get(deref) { Ty::Array { len, .. } => *len as usize, _ => 0 };
                let (ptr, _) = self.resolve_ptr_arg(f, args, arg_slots, 0, vals, slot_ptrs)?;
                let lo = vals[args[1].0 as usize].as_u64() as usize;
                let hi = vals[args[2].0 as usize].as_u64() as usize;
                if lo <= hi && hi <= len {
                    let sub_ptr = VmMem::ptr_add(ptr, lo * esz)?;
                    let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
                    Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::from_pair(sub_ptr, (hi - lo) as u64)))
                } else {
                    Ok(self.pack_error_union(ret_ty, TyId::VOID, 4, &IVal::Void))
                }
            }
            CfgBuiltinOp::VecSplat { vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecSplat on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let lane = vals[args[0].0 as usize].to_bytes(esz);
                let mut out = Vec::with_capacity(esz * (len as usize));
                for _ in 0..(len as usize) {
                    out.extend_from_slice(&lane);
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::VecLen(len) => Ok(IVal::Scalar(*len as u64)),
            CfgBuiltinOp::VecCmp { op: cmp_op, vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecCmp on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let bb = vals[args[1].0 as usize].to_bytes(esz * (len as usize));
                let mut out = vec![0u8; len as usize];
                for i in 0..(len as usize) {
                    let la = IVal::from_bytes(&ab[i * esz..(i + 1) * esz]);
                    let lb = IVal::from_bytes(&bb[i * esz..(i + 1) * esz]);
                    let rv = self.eval_binary(*cmp_op, &la, &lb, elem, TyId::BOOL)?;
                    out[i] = u8::from(rv.as_u64() != 0);
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::VecMaskAny { len } => {
                let mb = vals[args[0].0 as usize].to_bytes(*len as usize);
                Ok(IVal::Scalar(u64::from(mb.iter().any(|&b| b != 0))))
            }
            CfgBuiltinOp::VecMaskAll { len } => {
                let mb = vals[args[0].0 as usize].to_bytes(*len as usize);
                Ok(IVal::Scalar(u64::from(mb.iter().all(|&b| b != 0))))
            }
            CfgBuiltinOp::VecBitmask { len } => {
                let mb = vals[args[0].0 as usize].to_bytes(*len as usize);
                let mut bits = 0u64;
                for (i, &b) in mb.iter().enumerate() {
                    if i < 64 && b != 0 {
                        bits |= 1u64 << i;
                    }
                }
                Ok(IVal::Scalar(bits))
            }
            CfgBuiltinOp::VecSelect { vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecSelect on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let mb = vals[args[0].0 as usize].to_bytes(len as usize);
                let ab = vals[args[1].0 as usize].to_bytes(esz * (len as usize));
                let bb = vals[args[2].0 as usize].to_bytes(esz * (len as usize));
                let mut out = vec![0u8; esz * (len as usize)];
                for i in 0..(len as usize) {
                    let src = if mb[i] != 0 { &ab[i * esz..(i + 1) * esz] } else { &bb[i * esz..(i + 1) * esz] };
                    out[i * esz..(i + 1) * esz].copy_from_slice(src);
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::VecMin { vec_ty } | CfgBuiltinOp::VecMax { vec_ty } => {
                let is_min = matches!(op, CfgBuiltinOp::VecMin { .. });
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecMin/Max on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let bb = vals[args[1].0 as usize].to_bytes(esz * (len as usize));
                let cmp = if is_min { BinaryOp::Lt } else { BinaryOp::Gt };
                let mut out = vec![0u8; esz * (len as usize)];
                for i in 0..(len as usize) {
                    let la = IVal::from_bytes(&ab[i * esz..(i + 1) * esz]);
                    let lb = IVal::from_bytes(&bb[i * esz..(i + 1) * esz]);
                    let take_a = self.eval_binary(cmp, &la, &lb, elem, TyId::BOOL)?.as_u64() != 0;
                    let chosen = if take_a { &ab[i * esz..(i + 1) * esz] } else { &bb[i * esz..(i + 1) * esz] };
                    out[i * esz..(i + 1) * esz].copy_from_slice(chosen);
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::VecFma { vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecFma on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let bb = vals[args[1].0 as usize].to_bytes(esz * (len as usize));
                let cb = vals[args[2].0 as usize].to_bytes(esz * (len as usize));
                let is_flt = self.env.is_float(elem);
                let mul_op = if is_flt { BinaryOp::Mul } else { BinaryOp::MulWrap };
                let add_op = if is_flt { BinaryOp::Add } else { BinaryOp::AddWrap };
                let mut out = vec![0u8; esz * (len as usize)];
                for i in 0..(len as usize) {
                    let la = IVal::from_bytes(&ab[i * esz..(i + 1) * esz]);
                    let lb = IVal::from_bytes(&bb[i * esz..(i + 1) * esz]);
                    let lc = IVal::from_bytes(&cb[i * esz..(i + 1) * esz]);
                    let prod = self.eval_binary(mul_op, &la, &lb, elem, elem)?;
                    let sum = self.eval_binary(add_op, &prod, &lc, elem, elem)?;
                    out[i * esz..(i + 1) * esz].copy_from_slice(&sum.to_bytes(esz));
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::VecReduceAdd { vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecReduceAdd on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let add_op = if self.env.is_float(elem) { BinaryOp::Add } else { BinaryOp::AddWrap };
                let mut acc = IVal::from_bytes(&ab[0..esz]);
                for i in 1..(len as usize) {
                    let lane = IVal::from_bytes(&ab[i * esz..(i + 1) * esz]);
                    acc = self.eval_binary(add_op, &acc, &lane, elem, elem)?;
                }
                Ok(acc)
            }
            CfgBuiltinOp::VecReduceMin { vec_ty } | CfgBuiltinOp::VecReduceMax { vec_ty } => {
                let is_min = matches!(op, CfgBuiltinOp::VecReduceMin { .. });
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecReduceMin/Max on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let cmp = if is_min { BinaryOp::Lt } else { BinaryOp::Gt };
                let mut acc = IVal::from_bytes(&ab[0..esz]);
                for i in 1..(len as usize) {
                    let lane = IVal::from_bytes(&ab[i * esz..(i + 1) * esz]);
                    if self.eval_binary(cmp, &lane, &acc, elem, TyId::BOOL)?.as_u64() != 0 {
                        acc = lane;
                    }
                }
                Ok(acc)
            }
            CfgBuiltinOp::VecDot { vec_ty } => {
                let Ty::Vec { elem, len } = self.env.get(*vec_ty).clone() else {
                    return Err("EIR-Interp: VecDot on non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let ab = vals[args[0].0 as usize].to_bytes(esz * (len as usize));
                let bb = vals[args[1].0 as usize].to_bytes(esz * (len as usize));
                let is_flt = self.env.is_float(elem);
                let mul_op = if is_flt { BinaryOp::Mul } else { BinaryOp::MulWrap };
                let add_op = if is_flt { BinaryOp::Add } else { BinaryOp::AddWrap };
                let mut acc = self.eval_binary(
                    mul_op,
                    &IVal::from_bytes(&ab[0..esz]),
                    &IVal::from_bytes(&bb[0..esz]),
                    elem,
                    elem,
                )?;
                for i in 1..(len as usize) {
                    let prod = self.eval_binary(
                        mul_op,
                        &IVal::from_bytes(&ab[i * esz..(i + 1) * esz]),
                        &IVal::from_bytes(&bb[i * esz..(i + 1) * esz]),
                        elem,
                        elem,
                    )?;
                    acc = self.eval_binary(add_op, &acc, &prod, elem, elem)?;
                }
                Ok(acc)
            }
            CfgBuiltinOp::VecShuffle { src_ty, idx_ty, two_src } => {
                let Ty::Vec { elem, len: src_len } = self.env.get(*src_ty).clone() else {
                    return Err("EIR-Interp: VecShuffle on non-vec".to_owned());
                };
                let Ty::Vec { elem: ielem, len: dst_len } = self.env.get(*idx_ty).clone() else {
                    return Err("EIR-Interp: VecShuffle idx non-vec".to_owned());
                };
                let esz = x64_layout(self.env, elem).size.max(1);
                let isz = x64_layout(self.env, ielem).size.max(1);
                let v0 = vals[args[0].0 as usize].to_bytes(esz * (src_len as usize));
                let v1 = if *two_src {
                    vals[args[1].0 as usize].to_bytes(esz * (src_len as usize))
                } else {
                    Vec::new()
                };
                let idx_bytes = vals[args[if *two_src { 2 } else { 1 }].0 as usize].to_bytes(isz * (dst_len as usize));
                let total = if *two_src { (src_len as usize) * 2 } else { src_len as usize };
                let mut out = vec![0u8; esz * (dst_len as usize)];
                for i in 0..(dst_len as usize) {
                    let raw_k = IVal::from_bytes(&idx_bytes[i * isz..(i + 1) * isz]).as_u64() as usize;
                    let k = raw_k % total;
                    let src = if k < (src_len as usize) {
                        &v0[k * esz..(k + 1) * esz]
                    } else {
                        let k2 = k - (src_len as usize);
                        &v1[k2 * esz..(k2 + 1) * esz]
                    };
                    out[i * esz..(i + 1) * esz].copy_from_slice(src);
                }
                Ok(IVal::from_bytes(&out))
            }
            CfgBuiltinOp::GpuGlobalId
            | CfgBuiltinOp::GpuLocalId
            | CfgBuiltinOp::GpuWorkgroupId
            | CfgBuiltinOp::GpuWorkgroupBarrier => Err(format!(
                "ICE [eir_interp]: GPU intrinsic `{op:?}` cannot execute in CPU interpreter"
            )),
        }
    }

    fn read_str_ival(&self, v: &IVal) -> Result<Result<String, u32>, String> {
        let (ptr, len) = v.as_pair();
        let bytes = self.mem.read(ptr, len as usize)?;
        Ok(std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| 16))
    }

    fn read_build_str(&self, v: &IVal) -> Result<String, String> {
        let (ptr, len) = v.as_pair();
        let bytes = self.mem.read(ptr, len as usize)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| "build.eon error: string argument is not valid UTF-8".to_owned())
    }

    fn resolve_dir_root(&self, handle: u64) -> Result<PathBuf, u32> {
        self.dir_roots
            .get(handle as usize)
            .and_then(|o| o.clone())
            .ok_or(13)
    }

    fn read_into_buf_u8(&mut self, alloc_ptr: u64, target: &Path, ret_ty: TyId) -> Result<IVal, String> {
        let bytes = match fs::read(target) {
            Ok(b) => b,
            Err(e) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
        };
        let cap = bytes.len();
        let Some(data_ptr) = self.mem.alloc_raw_tracked(alloc_ptr, cap)? else {
            return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
        };
        if !bytes.is_empty() {
            self.mem.write(data_ptr, &bytes)?;
        }
        let mut buf = vec![0u8; 24];
        buf[0..8].copy_from_slice(&data_ptr.to_le_bytes());
        buf[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf[16..24].copy_from_slice(&(cap as u64).to_le_bytes());
        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
        Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(buf)))
    }

    fn list_dir_into_buf_u8(&mut self, alloc_ptr: u64, target: &Path, ret_ty: TyId) -> Result<IVal, String> {
        let rd = match fs::read_dir(target) {
            Ok(r) => r,
            Err(e) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
        };
        let mut names: Vec<Vec<u8>> = Vec::new();
        for entry in rd {
            match entry {
                Ok(e) => {
                    let name = e.file_name();
                    let raw = name.as_encoded_bytes();
                    if raw != b"." && raw != b".." {
                        names.push(raw.to_vec());
                    }
                }
                Err(e) => return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void)),
            }
        }
        names.sort();
        let mut bytes = Vec::new();
        for n in names {
            bytes.extend_from_slice(&n);
            bytes.push(b'\n');
        }
        let cap = bytes.len();
        let Some(data_ptr) = self.mem.alloc_raw_tracked(alloc_ptr, cap)? else {
            return Ok(self.pack_error_union(ret_ty, TyId::VOID, 7, &IVal::Void));
        };
        if !bytes.is_empty() {
            self.mem.write(data_ptr, &bytes)?;
        }
        let mut buf = vec![0u8; 24];
        buf[0..8].copy_from_slice(&data_ptr.to_le_bytes());
        buf[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf[16..24].copy_from_slice(&(cap as u64).to_le_bytes());
        let ok_ty = match self.env.get(ret_ty) { Ty::ErrorUnion { ok, .. } => *ok, _ => ret_ty };
        Ok(self.pack_error_union(ret_ty, ok_ty, 0, &IVal::Bytes(buf)))
    }

    fn write_from_slice(
        &self,
        target: &Path,
        slice_val: &IVal,
        ret_ty: TyId,
        executable: bool,
    ) -> Result<IVal, String> {
        let (dptr, dlen) = slice_val.as_pair();
        let data = self.mem.read(dptr, dlen as usize)?;
        if let Err(e) = fs::write(target, data) {
            return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void));
        }
        if executable {
            #[cfg(unix)]
            {
                let perms = fs::Permissions::from_mode(0o755);
                if let Err(e) = fs::set_permissions(target, perms) {
                    return Ok(self.pack_error_union(ret_ty, TyId::VOID, io_err_code(&e), &IVal::Void));
                }
            }
        }
        Ok(self.pack_error_union(ret_ty, TyId::VOID, 0, &IVal::Void))
    }

    fn exec_par_reduce(
        &self,
        elem_ty: TyId,
        op_sym: crate::span::Symbol,
        args: &[EirValue],
        vals: &[IVal],
    ) -> Result<IVal, String> {
        let (ptr, len) = vals[args[0].0 as usize].as_pair();
        let mut acc = if args.len() >= 3 {
            vals[args[2].0 as usize].clone()
        } else {
            IVal::Scalar(0)
        };
        let esz = x64_layout(self.env, elem_ty).size.max(1);
        let op_name = self.interner.get(op_sym);
        let float_bits = match self.env.get(elem_ty) {
            Ty::Float(b) => Some(*b),
            _ => None,
        };
        for i in 0..(len as usize) {
            let elem = IVal::from_bytes(self.mem.read(VmMem::ptr_add(ptr, i * esz)?, esz)?);
            match op_name {
                "Min" | "min" => {
                    let is_lt = match float_bits {
                        Some(16) => {
                            let e = f16_bits_to_f32(elem.as_u64() as u16);
                            let a = f16_bits_to_f32(acc.as_u64() as u16);
                            e.total_cmp(&a) == std::cmp::Ordering::Less
                        }
                        Some(32) => {
                            let e = f32::from_bits(elem.as_u64() as u32);
                            let a = f32::from_bits(acc.as_u64() as u32);
                            e.total_cmp(&a) == std::cmp::Ordering::Less
                        }
                        Some(_) => {
                            let e = f64::from_bits(elem.as_u64());
                            let a = f64::from_bits(acc.as_u64());
                            e.total_cmp(&a) == std::cmp::Ordering::Less
                        }
                        None => self.eval_binary(BinaryOp::Lt, &elem, &acc, elem_ty, TyId::BOOL)?.as_u64() != 0,
                    };
                    if is_lt {
                        acc = elem;
                    }
                }
                "Max" | "max" => {
                    let is_gt = match float_bits {
                        Some(16) => {
                            let e = f16_bits_to_f32(elem.as_u64() as u16);
                            let a = f16_bits_to_f32(acc.as_u64() as u16);
                            e.total_cmp(&a) == std::cmp::Ordering::Greater
                        }
                        Some(32) => {
                            let e = f32::from_bits(elem.as_u64() as u32);
                            let a = f32::from_bits(acc.as_u64() as u32);
                            e.total_cmp(&a) == std::cmp::Ordering::Greater
                        }
                        Some(_) => {
                            let e = f64::from_bits(elem.as_u64());
                            let a = f64::from_bits(acc.as_u64());
                            e.total_cmp(&a) == std::cmp::Ordering::Greater
                        }
                        None => self.eval_binary(BinaryOp::Gt, &elem, &acc, elem_ty, TyId::BOOL)?.as_u64() != 0,
                    };
                    if is_gt {
                        acc = elem;
                    }
                }
                "Mul" | "MulWrap" | "mul" => {
                    let bin = if float_bits.is_some() { BinaryOp::Mul } else { BinaryOp::MulWrap };
                    acc = self.eval_binary(bin, &acc, &elem, elem_ty, elem_ty)?;
                }
                "And" | "BitAnd" | "and" | "bitand" => {
                    acc = self.eval_binary(BinaryOp::BitAnd, &acc, &elem, elem_ty, elem_ty)?;
                }
                "Or" | "BitOr" | "or" | "bitor" => {
                    acc = self.eval_binary(BinaryOp::BitOr, &acc, &elem, elem_ty, elem_ty)?;
                }
                "Xor" | "BitXor" | "xor" | "bitxor" => {
                    acc = self.eval_binary(BinaryOp::BitXor, &acc, &elem, elem_ty, elem_ty)?;
                }
                _ => {
                    let bin = if float_bits.is_some() { BinaryOp::Add } else { BinaryOp::AddWrap };
                    acc = self.eval_binary(bin, &acc, &elem, elem_ty, elem_ty)?;
                }
            }
        }
        Ok(acc)
    }

    fn exec_schedule_run(
        &mut self,
        f: &EirFunction,
        args: &[EirValue],
        arg_slots: &[Option<EirSlot>],
        vals: &[IVal],
        slot_ptrs: &[u64],
    ) -> Result<(), String> {
        let sched_idx = vals[args[0].0 as usize].as_u64() as usize;
        if sched_idx >= self.eir.schedules.len() {
            return Err(format!("EIR-Interp: invalid schedule index {sched_idx}"));
        }
        let run_args = &args[1..];
        let num_waves = self.eir.schedules[sched_idx].waves.len();
        for wi in 0..num_waves {
            let num_sys = self.eir.schedules[sched_idx].waves[wi].systems.len();
            for si in 0..num_sys {
                let sys_def = self.eir.schedules[sched_idx].waves[wi].systems[si];
                let Some(&fn_idx) = self.eir.by_key.get(&CfgFnKey::TopDef(sys_def)) else {
                    return Err(format!("EIR-Interp: unresolved scheduled system `{sys_def:?}`"));
                };
                let sys_fn: &'a EirFunction = &self.eir.functions[fn_idx];
                let mut used = vec![false; run_args.len()];
                let mut sys_args = Vec::with_capacity(sys_fn.params.len());
                let mut temp_allocs: Vec<u64> = Vec::new();

                for (p_idx, &pv) in sys_fn.params.iter().enumerate() {
                    let p_ty = sys_fn.value_tys[pv.0 as usize];
                    let mut best: Option<(usize, ArgMatchKind)> = None;
                    for (k, &av) in run_args.iter().enumerate() {
                        if used[k] {
                            continue;
                        }
                        if let Some(kind) = match_arg_type(self.env, f.value_tys[av.0 as usize], p_ty)
                            && best.is_none_or(|(_, bk)| kind < bk)
                        {
                            best = Some((k, kind));
                        }
                    }
                    if best.is_none() && is_pool_or_cap_ty(self.env, p_ty) {
                        for (k, &av) in run_args.iter().enumerate() {
                            if let Some(kind) = match_arg_type(self.env, f.value_tys[av.0 as usize], p_ty)
                                && best.is_none_or(|(_, bk)| kind < bk)
                            {
                                best = Some((k, kind));
                            }
                        }
                    }
                    if let Some((k, kind)) = best {
                        used[k] = true;
                        let val = match kind {
                            ArgMatchKind::Exact | ArgMatchKind::MutToConst => {
                                vals[run_args[k].0 as usize].clone()
                            }
                            ArgMatchKind::Deref => {
                                let ptr = vals[run_args[k].0 as usize].as_u64();
                                let sz = interp_ty_size(self.env, p_ty);
                                IVal::from_bytes(self.mem.read(ptr, sz)?)
                            }
                            ArgMatchKind::Borrow => {
                                let (ptr, tp) =
                                    self.resolve_ptr_arg(f, args, arg_slots, 1 + k, vals, slot_ptrs)?;
                                if let Some(t) = tp {
                                    temp_allocs.push(t);
                                }
                                IVal::Scalar(ptr)
                            }
                        };
                        sys_args.push(val);
                    } else if p_idx < run_args.len() {
                        sys_args.push(vals[run_args[p_idx].0 as usize].clone());
                    } else {
                        for tp in temp_allocs {
                            let _ = self.mem.free(tp);
                        }
                        return Err(format!(
                            "EIR-Interp: missing argument {} for scheduled system `{}`",
                            p_idx,
                            self.interner.get(sys_fn.name)
                        ));
                    }
                }

                let res = self.exec_fn(sys_fn, sys_args);
                for tp in temp_allocs {
                    let _ = self.mem.free(tp);
                }
                let _ = res?;
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ArgMatchKind {
    Exact = 0,
    MutToConst = 1,
    Deref = 2,
    Borrow = 3,
}

fn match_arg_type(env: &TypeEnv, arg_ty: TyId, param_ty: TyId) -> Option<ArgMatchKind> {
    if arg_ty == param_ty {
        return Some(ArgMatchKind::Exact);
    }
    match (env.get(arg_ty), env.get(param_ty)) {
        (
            Ty::Borrow { is_mut: true, inner: ia },
            Ty::Borrow { is_mut: false, inner: ip },
        ) if ia == ip => Some(ArgMatchKind::MutToConst),
        (
            Ty::RawPtr { is_mut: true, inner: ia },
            Ty::RawPtr { is_mut: false, inner: ip },
        ) if ia == ip => Some(ArgMatchKind::MutToConst),
        (
            Ty::Borrow { inner: BorrowInner::Single(ia), .. } | Ty::RawPtr { inner: ia, .. },
            _,
        ) if *ia == param_ty => Some(ArgMatchKind::Deref),
        (
            _,
            Ty::Borrow { inner: BorrowInner::Single(ip), .. } | Ty::RawPtr { inner: ip, .. },
        ) if arg_ty == *ip => Some(ArgMatchKind::Borrow),
        _ => None,
    }
}

fn is_pool_or_cap_ty(env: &TypeEnv, ty: TyId) -> bool {
    let inner = match env.get(ty) {
        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
        _ => ty,
    };
    matches!(
        env.get(inner),
        Ty::Pool { .. }
            | Ty::Builtin(
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

enum PlaceTarget {
    Addr { addr: u64, temp_alloc: Option<u64> },
    SliceVal(IVal),
    Bitfield {
        base_addr: u64,
        base_ty: TyId,
        lo_bit: u8,
        hi_bit: u8,
        field_ty: TyId,
        temp_alloc: Option<u64>,
    },
}

/// Execute an `EirProgram` starting at `main` with command-line arguments, returning exit code,
/// captured stdout, and step count.
pub fn interpret_eir_with_args(
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
    step_limit: u64,
    prog_args: &[String],
) -> Result<InterpResult, String> {
    let mf = eir
        .functions
        .iter()
        .find(|f| !f.sig.effects.is_gpu && interner.get(f.name) == "main")
        .ok_or_else(|| "EIR-Interp: no `main` function found".to_owned())?;

    let mut interp = Interp::new(eir, prog, env, interner, step_limit);
    let takes_env = mf.params.len() == 1
        && matches!(
            env.get(mf.sig.params[0].ty),
            Ty::Borrow { inner: BorrowInner::Single(t), .. }
                if matches!(env.get(*t), Ty::Builtin(BuiltinKind::Env, _))
        );
    let takes_alloc = !takes_env
        && mf.params.len() == 1
        && matches!(
            env.get(mf.sig.params[0].ty),
            Ty::Borrow { inner: BorrowInner::Single(t), .. }
                if matches!(env.get(*t), Ty::Builtin(BuiltinKind::AllocCap, _))
        );

    let mut main_args = Vec::new();
    let root_cap_ptr = if takes_env {
        let env_ptr = interp.mem.alloc(88, false)?;
        let mut table_bytes = Vec::with_capacity(prog_args.len() * 16);
        for arg in prog_args {
            let sptr = interp.intern_ro_bytes(arg.as_bytes())?;
            table_bytes.extend_from_slice(&sptr.to_le_bytes());
            table_bytes.extend_from_slice(&(arg.len() as u64).to_le_bytes());
        }
        let args_table_ptr = if table_bytes.is_empty() {
            0u64
        } else {
            interp.mem.alloc_bytes(&table_bytes, true)?
        };
        // +0..+40: AllocCap (zeroed = unlimited root)
        interp.mem.write_u64(VmMem::ptr_add(env_ptr, 40)?, 1)?; // FsCap handle 1 (cwd)
        interp.mem.write_u64(VmMem::ptr_add(env_ptr, 48)?, 0)?; // NetCap handle 0 (unrestricted)
        interp.mem.write_u64(VmMem::ptr_add(env_ptr, 56)?, 1)?; // ClockCap handle 1
        interp.mem.write_u64(VmMem::ptr_add(env_ptr, 64)?, 1)?; // IoCap handle 1
        interp
            .mem
            .write_u64(VmMem::ptr_add(env_ptr, 72)?, prog_args.len() as u64)?; // ArgsCap.argc
        interp
            .mem
            .write_u64(VmMem::ptr_add(env_ptr, 80)?, args_table_ptr)?; // ArgsCap.args_table_ptr
        main_args.push(IVal::Scalar(env_ptr));
        Some(env_ptr)
    } else if takes_alloc {
        let p = interp.mem.alloc(40, false)?;
        main_args.push(IVal::Scalar(p));
        Some(p)
    } else {
        None
    };

    let ret_val = interp.exec_fn(mf, main_args)?;
    let _ = root_cap_ptr.map(|p| interp.mem.block_len(p));

    let exit_code = match env.get(mf.sig.ret) {
        Ty::Void => 0,
        Ty::Int { .. } => ret_val.as_u64() as i32,
        Ty::ErrorUnion { ok, .. } => {
            let sz = x64_layout(env, mf.sig.ret).size.max(4);
            let raw = ret_val.to_bytes(sz);
            let err_tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            if err_tag != 0 {
                err_tag as i32
            } else if *ok == TyId::VOID || *ok == TyId::NEVER {
                0
            } else {
                let ok_off = eu_ok_offset(env, *ok);
                let ok_sz = x64_layout(env, *ok).size.clamp(1, 8);
                IVal::from_bytes(&raw[ok_off..ok_off + ok_sz]).as_u64() as i32
            }
        }
        _ => 0,
    };

    Ok(InterpResult {
        exit_code,
        stdout: interp.stdout,
        stderr: interp.stderr,
        steps: interp.steps,
    })
}

/// Execute an `EirProgram` starting at `main` with no arguments, returning exit code,
/// captured stdout, and step count.
pub fn interpret_eir(
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
    step_limit: u64,
) -> Result<InterpResult, String> {
    interpret_eir_with_args(eir, prog, env, interner, step_limit, &[])
}

/// Execute a sandboxed `build.eon` script starting at `fn build` or `fn main`, returning the
/// populated [`BuildPlan`].
pub fn interpret_build_script(
    eir: &EirProgram,
    prog: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> Result<BuildPlan, String> {
    let entry = eir
        .functions
        .iter()
        .find(|f| !f.sig.effects.is_gpu && interner.get(f.name) == "build")
        .or_else(|| {
            eir.functions
                .iter()
                .find(|f| !f.sig.effects.is_gpu && interner.get(f.name) == "main")
        });

    let Some(entry_fn) = entry else {
        return Ok(BuildPlan::default());
    };

    let mut interp = Interp::new(eir, prog, env, interner, DEFAULT_STEP_LIMIT);
    let mut args = Vec::new();
    if entry_fn.params.len() == 1 {
        let cap_ptr = interp.mem.alloc(16, false)?;
        args.push(IVal::Scalar(cap_ptr));
    }

    let ret_val = interp.exec_fn(entry_fn, args)?;
    if let Ty::ErrorUnion { .. } = env.get(entry_fn.sig.ret) {
        let sz = x64_layout(env, entry_fn.sig.ret).size.max(4);
        let raw = ret_val.to_bytes(sz);
        let err_tag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        if err_tag != 0 {
            return Err(format!(
                "build.eon entry function `{}` failed with error code {err_tag}",
                interner.get(entry_fn.name)
            ));
        }
    }

    Ok(interp.build_plan)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::diag::DiagBag;
    use crate::span::{Interner, SourceMap};

    #[test]
    fn interp_captures_stderr_utf8_and_fs_primitives() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("repo root")
            .to_path_buf();
        let rt_bytes = fs::read(root.join("std/rt.hal.eon")).expect("read std/rt.hal.eon");
        let str_bytes = fs::read(root.join("core/str.eon")).expect("read core/str.eon");

        let tmp = std::env::temp_dir().join(format!(
            "eon_wave0b_interp_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create temp dir");

        let nested = tmp.join("sub/deep");
        let script_path = tmp.join("run_me.sh");
        let src = format!(
            r##"module wave0b_unit;
import std.rt;

view fn to_str_view(b: *[u8]) -> *str ! Utf8Error {{
    return try b.as_str();
}}

pub fn main(env: *mut Env) -> i32 ! IoError | OutOfMemory | Utf8Error {{
    let ok_s = try to_str_view("ok".as_bytes());
    rt.print_bytes(ok_s.as_bytes());
    rt.print_nl();
    rt.eprint_str("err1");
    rt.eprint_bytes(":err2".as_bytes());
    rt.eprint_nl();

    let bad: [3]u8 = [0xED_u8, 0xA0_u8, 0x80_u8];
    let bad_s: *[u8] = &bad;
    let is_bad: i32 = match bad_s.as_str() {{
        .InvalidUtf8 => 1_i32,
        _ => 0_i32,
    }};
    if is_bad != 1_i32 {{ return 10_i32; }}

    try env.fs.mkdir_all({nested:?});
    if !env.fs.is_dir({nested:?}) {{ return 11_i32; }}

    let entries = try env.fs.list_dir(&mut env.alloc, {tmp_str:?});
    let entries_str = try entries.as_slice().as_str();
    if !rt.str_eq(entries_str, "sub\n") {{
        entries.destroy(&mut env.alloc);
        return 12_i32;
    }}
    entries.destroy(&mut env.alloc);

    try env.fs.write_executable({script:?}, "#!/bin/sh\nexit 17\n".as_bytes());
    let rc = try env.fs.spawn_wait({script:?}, "".as_bytes());
    if rc != 17_i32 {{ return 13_i32; }}

    if env.alloc.live_allocs() != 0 {{ return 14_i32; }}
    return 0_i32;
}}
"##,
            nested = nested.to_string_lossy(),
            tmp_str = tmp.to_string_lossy(),
            script = script_path.to_string_lossy(),
        );

        let files = vec![
            ("core/str.eon".to_owned(), str_bytes),
            ("std/rt.hal.eon".to_owned(), rt_bytes),
            ("wave0b_unit.eon".to_owned(), src.into_bytes()),
        ];
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut diags = DiagBag::new();
        let res = crate::interpret_files(&files, &mut sm, &mut interner, &mut diags)
            .expect("interpret wave0b_unit");
        let _ = fs::remove_dir_all(&tmp);

        assert_eq!(res.exit_code, 0);
        assert_eq!(res.stdout, "ok\n");
        assert_eq!(res.stderr, "err1:err2\n");
    }
}
