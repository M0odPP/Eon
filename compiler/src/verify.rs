#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{BinaryOp, Expr, ExprKind, UnaryOp};
use crate::cfg::{BlockId, CfgBuiltinOp, CfgConst, CfgFnKey};
use crate::diag::{DiagBag, Diagnostic, ErrorCode};
use crate::eir::{
    EirCallee, EirFunction, EirInst, EirInstKind, EirPlace, EirPlaceBase, EirProgram, EirProj,
    EirSlot, EirTerm, EirValue,
};
use crate::resolve::BuiltinKind;
use crate::span::{Interner, Span, Symbol};
use crate::types::{int_bounds, parse_int_literal, BorrowInner, Ty, TyId, TypeEnv};

const MAX_DBM_VERTS: usize = 64;
const WIDEN_AFTER_ITERS: usize = 3;
const MAX_FIXPOINT_ITERS: usize = 32;
const NARROW_ITERS: usize = 2;

/// Exact bound in `Z ∪ {+∞}` (`None` represents `+∞`).
type Bound = Option<i128>;

fn add_bound(a: Bound, b: Bound) -> Bound {
    match (a, b) {
        (Some(x), Some(y)) => x.checked_add(y),
        _ => None,
    }
}

fn min_bound(a: Bound, b: Bound) -> Bound {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn max_bound(a: Bound, b: Bound) -> Bound {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        _ => None,
    }
}

fn bound_lt(a: Bound, b: Bound) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x < y,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

fn bound_le(a: Bound, b: Bound) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x <= y,
        (_, None) => true,
        (None, Some(_)) => false,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Interval {
    lo: Option<i128>, // None = -inf
    hi: Option<i128>, // None = +inf
}

impl Interval {
    fn top() -> Self {
        Self { lo: None, hi: None }
    }

    fn const_val(c: i128) -> Self {
        Self {
            lo: Some(c),
            hi: Some(c),
        }
    }

    fn from_range(lo: i128, hi: i128) -> Self {
        Self {
            lo: Some(lo),
            hi: Some(hi),
        }
    }

    fn is_empty(&self) -> bool {
        matches!((self.lo, self.hi), (Some(l), Some(h)) if l > h)
    }

    fn intersect(self, other: Self) -> Self {
        let lo = match (self.lo, other.lo) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let hi = match (self.hi, other.hi) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Self { lo, hi }
    }

    fn hull(self, other: Self) -> Self {
        let lo = match (self.lo, other.lo) {
            (Some(a), Some(b)) => Some(a.min(b)),
            _ => None,
        };
        let hi = match (self.hi, other.hi) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
        Self { lo, hi }
    }

    fn widen(self, next: Self, type_bounds: Option<(i128, i128)>) -> Self {
        let lo = match (self.lo, next.lo) {
            (Some(a), Some(b)) if b >= a => Some(a),
            _ => type_bounds.map(|(tl, _)| tl),
        };
        let hi = match (self.hi, next.hi) {
            (Some(a), Some(b)) if b <= a => Some(a),
            _ => type_bounds.map(|(_, th)| th),
        };
        Self { lo, hi }
    }

    fn narrow(self, next: Self) -> Self {
        let lo = match (self.lo, next.lo) {
            (None, Some(b)) => Some(b),
            (Some(a), Some(b)) if b > a => Some(b),
            _ => self.lo,
        };
        let hi = match (self.hi, next.hi) {
            (None, Some(b)) => Some(b),
            (Some(a), Some(b)) if b < a => Some(b),
            _ => self.hi,
        };
        Self { lo, hi }
    }

    fn fmt_str(&self) -> String {
        let l = self.lo.map_or("-inf".into(), |v| v.to_string());
        let h = self.hi.map_or("+inf".into(), |v| v.to_string());
        format!("[{l}..={h}]")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum VarKey {
    Val(EirValue),
    Slot(EirSlot),
    LenVal(EirValue),
    LenSlot(EirSlot),
    FieldVal(EirValue, u32),
    FieldSlot(EirSlot, u32),
    FieldLenVal(EirValue, u32),
    FieldLenSlot(EirSlot, u32),
}

impl VarKey {
    fn is_len(self) -> bool {
        matches!(
            self,
            VarKey::LenVal(_) | VarKey::LenSlot(_) | VarKey::FieldLenVal(_, _) | VarKey::FieldLenSlot(_, _)
        )
    }

    fn is_priority(self) -> bool {
        matches!(
            self,
            VarKey::LenVal(_)
                | VarKey::LenSlot(_)
                | VarKey::Slot(_)
                | VarKey::FieldVal(_, _)
                | VarKey::FieldSlot(_, _)
                | VarKey::FieldLenVal(_, _)
                | VarKey::FieldLenSlot(_, _)
        )
    }
}

/// Bit-level known-bits domain for non-negative integers (`zeros & ones == 0`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct KnownBits {
    zeros: u128,
    ones: u128,
}

impl KnownBits {
    fn from_const(c: u128) -> Self {
        Self { zeros: !c, ones: c }
    }

    fn from_nonneg_interval(iv: Interval, bw: u32) -> Self {
        let mask = if bw >= 128 { u128::MAX } else { (1u128 << bw) - 1 };
        let mut zeros = !mask;
        let mut ones = 0u128;
        if let (Some(lo), Some(hi)) = (iv.lo, iv.hi)
            && lo >= 0
            && hi >= lo
        {
            let ulo = lo as u128;
            let uhi = hi as u128;
            if ulo == uhi {
                return Self::from_const(ulo);
            }
            // All bits strictly above the highest differing bit are identical in [lo..=hi].
            let diff = ulo ^ uhi;
            let leading = diff.leading_zeros();
            if leading > 0 {
                let common_mask = u128::MAX << (128 - leading);
                zeros |= common_mask & !ulo;
                ones |= common_mask & ulo;
            }
        } else if let Some(hi) = iv.hi
            && hi >= 0
        {
            let uhi = hi as u128;
            let used_bits = 128 - uhi.leading_zeros();
            if used_bits < 128 {
                zeros |= u128::MAX << used_bits;
            }
        }
        Self { zeros: zeros & !ones, ones }
    }

    fn join(self, other: Self) -> Self {
        Self {
            zeros: self.zeros & other.zeros,
            ones: self.ones & other.ones,
        }
    }

    fn bit_and(self, other: Self) -> Self {
        Self {
            zeros: self.zeros | other.zeros,
            ones: self.ones & other.ones,
        }
    }

    fn bit_or(self, other: Self) -> Self {
        Self {
            zeros: self.zeros & other.zeros,
            ones: self.ones | other.ones,
        }
    }

    fn bit_xor(self, other: Self) -> Self {
        Self {
            zeros: (self.zeros & other.zeros) | (self.ones & other.ones),
            ones: (self.zeros & other.ones) | (self.ones & other.zeros),
        }
    }

    fn shl(self, k: u32, bw: u32) -> Self {
        if k >= 128 {
            return Self::from_const(0);
        }
        let mask = if bw >= 128 { u128::MAX } else { (1u128 << bw) - 1 };
        let low_zeros = if k == 0 { 0 } else { (1u128 << k) - 1 };
        let zeros = ((self.zeros << k) | low_zeros | !mask) & !((self.ones << k) & mask);
        let ones = (self.ones << k) & mask;
        Self { zeros, ones }
    }

    fn lshr(self, k: u32) -> Self {
        if k >= 128 {
            return Self::from_const(0);
        }
        let high_zeros = if k == 0 { 0 } else { u128::MAX << (128 - k) };
        Self {
            zeros: (self.zeros >> k) | high_zeros,
            ones: self.ones >> k,
        }
    }

    fn to_nonneg_interval(self, bw: u32) -> Option<Interval> {
        let mask = if bw >= 128 { u128::MAX } else { (1u128 << bw) - 1 };
        let lo = self.ones & mask;
        let hi = (!self.zeros) & mask;
        if lo <= hi && hi <= (i128::MAX as u128) {
            Some(Interval::from_range(lo as i128, hi as i128))
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CmpOp {
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eq,
    NotEq,
}

impl CmpOp {
    fn negate(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::GtEq,
            CmpOp::LtEq => CmpOp::Gt,
            CmpOp::Gt => CmpOp::LtEq,
            CmpOp::GtEq => CmpOp::Lt,
            CmpOp::Eq => CmpOp::NotEq,
            CmpOp::NotEq => CmpOp::Eq,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BoolPred {
    Cmp { op: CmpOp, lhs: VarKey, rhs: VarKey },
    Const(bool),
    Not(Box<BoolPred>),
    And(Box<BoolPred>, Box<BoolPred>),
    Or(Box<BoolPred>, Box<BoolPred>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DbmState {
    is_bottom: bool,
    intervals: BTreeMap<VarKey, Interval>,
    non_zero: BTreeSet<VarKey>,
    // DBM vertices: index 0 = V0 (zero), indices 1..len = active VarKeys (<= MAX_DBM_VERTS).
    verts: Vec<Option<VarKey>>,
    vmap: BTreeMap<VarKey, usize>,
    // mat[i][j] = upper bound on (v_i - v_j).
    mat: Vec<Vec<Bound>>,
    // Track which SSA values hold known comparison predicates (for path-sensitive branching).
    preds: BTreeMap<EirValue, BoolPred>,
    // Track known base-slot aliases for SSA values (so `LoadSlot`/`CopyVal` preserve symbolic Len).
    len_alias: BTreeMap<VarKey, VarKey>,
    // Known-bits domain for bitwise operations (`x & mask`, `x | mask`, `x ^ mask`, shifts).
    known_bits: BTreeMap<VarKey, KnownBits>,
    // Symbolic subtraction/addition origins for difference algebra (`len - start >= 3`, `tlen <= len - tstart`).
    diff_exprs: BTreeMap<VarKey, (VarKey, VarKey)>,
    sum_exprs: BTreeMap<VarKey, (VarKey, VarKey)>,
    // Checked arithmetic origins so `UnwrapErrorOk` propagates exact postconditions (`r == a + b`).
    checked_op: BTreeMap<EirValue, (BinaryOp, EirValue, EirValue, TyId)>,
    slot_checked_op: BTreeMap<EirSlot, (BinaryOp, EirValue, EirValue, TyId)>,
    // Track which slot a borrow/pointer SSA value points to (`SlotAddr`).
    val_slot_origin: BTreeMap<EirValue, EirSlot>,
    // Track literal integer constants (only true `Const` defs skip DBM vertex allocation in `add_diff`).
    const_lits: BTreeSet<VarKey>,
}

impl DbmState {
    fn bottom() -> Self {
        Self {
            is_bottom: true,
            intervals: BTreeMap::new(),
            non_zero: BTreeSet::new(),
            verts: vec![None],
            vmap: BTreeMap::new(),
            mat: vec![vec![Some(0)]],
            preds: BTreeMap::new(),
            len_alias: BTreeMap::new(),
            known_bits: BTreeMap::new(),
            diff_exprs: BTreeMap::new(),
            sum_exprs: BTreeMap::new(),
            checked_op: BTreeMap::new(),
            slot_checked_op: BTreeMap::new(),
            val_slot_origin: BTreeMap::new(),
            const_lits: BTreeSet::new(),
        }
    }

    fn top() -> Self {
        Self {
            is_bottom: false,
            intervals: BTreeMap::new(),
            non_zero: BTreeSet::new(),
            verts: vec![None],
            vmap: BTreeMap::new(),
            mat: vec![vec![Some(0)]],
            preds: BTreeMap::new(),
            len_alias: BTreeMap::new(),
            known_bits: BTreeMap::new(),
            diff_exprs: BTreeMap::new(),
            sum_exprs: BTreeMap::new(),
            checked_op: BTreeMap::new(),
            slot_checked_op: BTreeMap::new(),
            val_slot_origin: BTreeMap::new(),
            const_lits: BTreeSet::new(),
        }
    }

    fn ensure_vertex_protected(&mut self, k: VarKey, protected: Option<VarKey>) -> usize {
        if let Some(&idx) = self.vmap.get(&k) {
            return idx;
        }
        if self.verts.len() > MAX_DBM_VERTS {
            self.evict_one_vertex(protected);
        }
        let idx = self.verts.len();
        self.verts.push(Some(k));
        self.vmap.insert(k, idx);
        for row in &mut self.mat {
            row.push(None);
        }
        let mut new_row = vec![None; idx + 1];
        new_row[idx] = Some(0);
        self.mat.push(new_row);

        if let Some(iv) = self.intervals.get(&k).copied() {
            if let Some(h) = iv.hi {
                self.mat[idx][0] = min_bound(self.mat[idx][0], Some(h));
            }
            if let Some(l) = iv.lo
                && let Some(nl) = l.checked_neg()
            {
                self.mat[0][idx] = min_bound(self.mat[0][idx], Some(nl));
            }
            // Incremental closure for the newly seeded 0-edges.
            self.close_around(idx, 0);
            self.close_around(0, idx);
        }
        idx
    }

    fn evict_one_vertex(&mut self, protected: Option<VarKey>) {
        // Deterministically evict the oldest (lowest-id) non-priority, non-protected vertex,
        // preserving Len*, Slot, and Field* vertices ahead of ephemeral SSA temporaries.
        let victim = (1..self.verts.len())
            .filter(|&i| self.verts[i].is_some_and(|k| !k.is_priority() && Some(k) != protected))
            .min_by_key(|&i| self.verts[i])
            .or_else(|| {
                (1..self.verts.len())
                    .filter(|&i| self.verts[i].is_some_and(|k| !k.is_len() && Some(k) != protected))
                    .min_by_key(|&i| self.verts[i])
            })
            .or_else(|| (1..self.verts.len()).find(|&i| self.verts[i] != protected))
            .unwrap_or(1);
        if let Some(vk) = self.verts[victim] {
            let iv = self.get_interval(vk);
            self.intervals.insert(vk, iv);
            self.vmap.remove(&vk);
        }
        let last = self.verts.len() - 1;
        if victim != last {
            let moved = self.verts[last];
            self.verts[victim] = moved;
            if let Some(mk) = moved {
                self.vmap.insert(mk, victim);
            }
            self.mat.swap(victim, last);
            for row in &mut self.mat {
                row.swap(victim, last);
            }
        }
        self.verts.pop();
        self.mat.pop();
        for row in &mut self.mat {
            row.pop();
        }
    }

    fn close_around(&mut self, u: usize, v: usize) {
        if self.is_bottom {
            return;
        }
        let Some(c) = self.mat[u][v] else {
            return;
        };
        let n = self.verts.len();
        for i in 0..n {
            let Some(d_iu) = self.mat[i][u] else {
                continue;
            };
            let Some(sum_iu) = d_iu.checked_add(c) else {
                continue;
            };
            for j in 0..n {
                let Some(d_vj) = self.mat[v][j] else {
                    continue;
                };
                if let Some(cand) = sum_iu.checked_add(d_vj)
                    && bound_lt(Some(cand), self.mat[i][j])
                {
                    self.mat[i][j] = Some(cand);
                    if i == j && cand < 0 {
                        self.is_bottom = true;
                        return;
                    }
                }
            }
        }
    }

    fn sync_intervals_from_dbm(&mut self) {
        if self.is_bottom {
            return;
        }
        for i in 1..self.verts.len() {
            let Some(k) = self.verts[i] else { continue };
            let hi = self.mat[i][0];
            let lo = self.mat[0][i].and_then(|v| v.checked_neg());
            let cur = self.intervals.get(&k).copied().unwrap_or_else(Interval::top);
            let next = cur.intersect(Interval { lo, hi });
            if next.is_empty() {
                self.is_bottom = true;
                return;
            }
            if next.lo.is_some_and(|l| l > 0) || next.hi.is_some_and(|h| h < 0) {
                self.non_zero.insert(k);
            }
            self.intervals.insert(k, next);
        }
    }

    fn get_interval(&self, k: VarKey) -> Interval {
        let mut iv = self.intervals.get(&k).copied().unwrap_or_else(Interval::top);
        if let Some(&idx) = self.vmap.get(&k) {
            let hi = self.mat[idx][0];
            let lo = self.mat[0][idx].and_then(|v| v.checked_neg());
            iv = iv.intersect(Interval { lo, hi });
        }
        iv
    }

    fn diff_upper(&self, u: VarKey, v: VarKey) -> Bound {
        if u == v {
            return Some(0);
        }
        let mut best: Bound = None;
        if let (Some(&ui), Some(&vi)) = (self.vmap.get(&u), self.vmap.get(&v)) {
            best = min_bound(best, self.mat[ui][vi]);
        }
        let iu = self.get_interval(u);
        let iv = self.get_interval(v);
        if let (Some(hu), Some(lv)) = (iu.hi, iv.lo)
            && let Some(d) = hu.checked_sub(lv)
        {
            best = min_bound(best, Some(d));
        }
        best
    }

    fn is_non_zero(&self, k: VarKey) -> bool {
        if self.non_zero.contains(&k) {
            return true;
        }
        let iv = self.get_interval(k);
        iv.lo.is_some_and(|l| l > 0) || iv.hi.is_some_and(|h| h < 0)
    }

    /// Constrain `u - v <= c`.
    fn add_diff(&mut self, u: VarKey, v: VarKey, c: i128) {
        if self.is_bottom {
            return;
        }
        if u == v {
            if c < 0 {
                self.is_bottom = true;
            }
            return;
        }
        // Fast path: if either side is a true literal constant, fold into unary interval bound without allocating a DBM vertex!
        // Non-constant variables (e.g. loop phis that happen to be `[0..=0]` on iteration 0) MUST retain DBM edges across widening.
        let iu = self.get_interval(u);
        let iv = self.get_interval(v);
        if self.const_lits.contains(&v)
            && let (Some(vl), Some(vh)) = (iv.lo, iv.hi)
            && vl == vh
        {
            if let Some(hi) = vl.checked_add(c) {
                self.add_upper(u, hi);
            }
            return;
        }
        if self.const_lits.contains(&u)
            && let (Some(ul), Some(uh)) = (iu.lo, iu.hi)
            && ul == uh
        {
            if let Some(lo) = ul.checked_sub(c) {
                self.add_lower(v, lo);
            }
            return;
        }
        self.ensure_vertex_protected(u, Some(v));
        self.ensure_vertex_protected(v, Some(u));
        let ui = self.vmap[&u];
        let vi = self.vmap[&v];
        if bound_lt(Some(c), self.mat[ui][vi]) {
            self.mat[ui][vi] = Some(c);
            self.close_around(ui, vi);
            self.sync_intervals_from_dbm();
        }
    }

    /// Constrain `k <= hi`.
    fn add_upper(&mut self, k: VarKey, hi: i128) {
        if self.is_bottom {
            return;
        }
        let cur = self.get_interval(k).intersect(Interval { lo: None, hi: Some(hi) });
        if cur.is_empty() {
            self.is_bottom = true;
            return;
        }
        if cur.hi.is_some_and(|h| h < 0) {
            self.non_zero.insert(k);
        }
        self.intervals.insert(k, cur);
        if let Some(&idx) = self.vmap.get(&k)
            && bound_lt(Some(hi), self.mat[idx][0])
        {
            self.mat[idx][0] = Some(hi);
            self.close_around(idx, 0);
            self.sync_intervals_from_dbm();
        }
    }

    /// Constrain `k >= lo`.
    fn add_lower(&mut self, k: VarKey, lo: i128) {
        if self.is_bottom {
            return;
        }
        let cur = self.get_interval(k).intersect(Interval { lo: Some(lo), hi: None });
        if cur.is_empty() {
            self.is_bottom = true;
            return;
        }
        if cur.lo.is_some_and(|l| l > 0) {
            self.non_zero.insert(k);
        }
        self.intervals.insert(k, cur);
        if let Some(neg_lo) = lo.checked_neg()
            && let Some(&idx) = self.vmap.get(&k)
            && bound_lt(Some(neg_lo), self.mat[0][idx])
        {
            self.mat[0][idx] = Some(neg_lo);
            self.close_around(0, idx);
            self.sync_intervals_from_dbm();
        }
    }

    fn set_interval(&mut self, k: VarKey, iv: Interval) {
        if self.is_bottom {
            return;
        }
        if iv.is_empty() {
            self.is_bottom = true;
            return;
        }
        let cur = self.get_interval(k).intersect(iv);
        if cur.is_empty() {
            self.is_bottom = true;
            return;
        }
        if cur.lo.is_some_and(|l| l > 0) || cur.hi.is_some_and(|h| h < 0) {
            self.non_zero.insert(k);
        }
        self.intervals.insert(k, cur);
        if self.vmap.contains_key(&k) {
            if let Some(l) = cur.lo {
                self.add_lower(k, l);
            }
            if let Some(h) = cur.hi {
                self.add_upper(k, h);
            }
        }
    }

    fn add_eq(&mut self, u: VarKey, v: VarKey) {
        self.add_diff(u, v, 0);
        self.add_diff(v, u, 0);
        if self.is_non_zero(u) {
            self.non_zero.insert(v);
        } else if self.is_non_zero(v) {
            self.non_zero.insert(u);
        }
    }

    fn add_const(&mut self, k: VarKey, c: i128) {
        self.set_interval(k, Interval::const_val(c));
        if c != 0 {
            self.non_zero.insert(k);
        }
    }

    fn forget(&mut self, k: VarKey) {
        self.intervals.remove(&k);
        self.non_zero.remove(&k);
        self.len_alias.remove(&k);
        self.known_bits.remove(&k);
        self.diff_exprs.remove(&k);
        self.sum_exprs.remove(&k);
        self.diff_exprs.retain(|_, (a, b)| *a != k && *b != k);
        self.sum_exprs.retain(|_, (a, b)| *a != k && *b != k);
        if let Some(&idx) = self.vmap.get(&k) {
            let n = self.verts.len();
            for j in 0..n {
                self.mat[idx][j] = if j == idx { Some(0) } else { None };
                self.mat[j][idx] = if j == idx { Some(0) } else { None };
            }
        }
    }

    fn forget_fields_for_slot(&mut self, slot: EirSlot) {
        let keys: Vec<VarKey> = self
            .intervals
            .keys()
            .chain(self.vmap.keys())
            .copied()
            .filter(|k| matches!(k, VarKey::FieldSlot(s, _) | VarKey::FieldLenSlot(s, _) if *s == slot))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for k in keys {
            self.forget(k);
        }
        self.val_slot_origin.retain(|_, s| *s != slot);
    }

    fn forget_all_mutable_fields(&mut self, f: &EirFunction, env: &TypeEnv) {
        let keys: Vec<VarKey> = self
            .intervals
            .keys()
            .chain(self.vmap.keys())
            .copied()
            .filter(|k| match k {
                VarKey::FieldSlot(_, _) | VarKey::FieldLenSlot(_, _) => true,
                VarKey::FieldVal(v, _) | VarKey::FieldLenVal(v, _) => {
                    // Under Eon's Law of Exclusivity, shared `*T` fields CANNOT be mutated while active!
                    let ty = f.value_tys.get(v.0 as usize).copied().unwrap_or(TyId::ERROR);
                    !matches!(env.get(ty), Ty::Borrow { is_mut: false, .. })
                }
                _ => false,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for k in keys {
            self.forget(k);
        }
    }

    /// Assign `dst = src + c` cleanly even when `dst == src`!
    fn assign_offset(&mut self, dst: VarKey, src: VarKey, offset: i128, dst_bounds: Option<(i128, i128)>) {
        if self.is_bottom {
            return;
        }
        if dst == src {
            let old_iv = self.get_interval(src);
            if let Some(&idx) = self.vmap.get(&src) {
                let n = self.verts.len();
                for j in 0..n {
                    if j == idx {
                        continue;
                    }
                    // new_src - v_j = (old_src + offset) - v_j <= old + offset
                    self.mat[idx][j] = add_bound(self.mat[idx][j], Some(offset));
                    // v_j - new_src = v_j - old_src - offset <= old - offset
                    self.mat[j][idx] = offset.checked_neg().and_then(|no| add_bound(self.mat[j][idx], Some(no)));
                }
            }
            let new_iv = Interval {
                lo: old_iv.lo.and_then(|l| l.checked_add(offset)),
                hi: old_iv.hi.and_then(|h| h.checked_add(offset)),
            };
            self.intervals.insert(dst, new_iv);
            self.non_zero.remove(&dst);
            if let Some((tl, th)) = dst_bounds {
                self.set_interval(dst, Interval::from_range(tl, th));
            }
            self.sync_intervals_from_dbm();
            return;
        }
        let src_iv = self.get_interval(src);
        self.forget(dst);
        if let Some((tl, th)) = dst_bounds {
            self.set_interval(dst, Interval::from_range(tl, th));
        }
        let new_iv = Interval {
            lo: src_iv.lo.and_then(|l| l.checked_add(offset)),
            hi: src_iv.hi.and_then(|h| h.checked_add(offset)),
        };
        self.set_interval(dst, new_iv);
        self.add_diff(dst, src, offset);
        if let Some(neg) = offset.checked_neg() {
            self.add_diff(src, dst, neg);
        }
        if offset == 0 && self.is_non_zero(src) {
            self.non_zero.insert(dst);
        }
    }

    fn refine_cmp(&mut self, op: CmpOp, lhs: VarKey, rhs: VarKey) {
        if self.is_bottom {
            return;
        }
        match op {
            CmpOp::Lt => self.add_diff(lhs, rhs, -1),
            CmpOp::LtEq => self.add_diff(lhs, rhs, 0),
            CmpOp::Gt => self.add_diff(rhs, lhs, -1),
            CmpOp::GtEq => self.add_diff(rhs, lhs, 0),
            CmpOp::Eq => self.add_eq(lhs, rhs),
            CmpOp::NotEq => {
                if lhs == rhs {
                    self.is_bottom = true;
                    return;
                }
                let li = self.get_interval(lhs);
                let ri = self.get_interval(rhs);
                // If rhs is constant c: tighten lhs bounds if c == lo or c == hi, and track non_zero!
                if let (Some(rl), Some(rh)) = (ri.lo, ri.hi)
                    && rl == rh
                {
                    let c = rl;
                    if c == 0 {
                        self.non_zero.insert(lhs);
                    }
                    if li.lo == Some(c) && li.hi == Some(c) {
                        self.is_bottom = true;
                        return;
                    }
                    if li.lo == Some(c)
                        && let Some(c1) = c.checked_add(1)
                    {
                        self.add_lower(lhs, c1);
                    }
                    if li.hi == Some(c)
                        && let Some(cm1) = c.checked_sub(1)
                    {
                        self.add_upper(lhs, cm1);
                    }
                }
                if let (Some(ll), Some(lh)) = (li.lo, li.hi)
                    && ll == lh
                {
                    let c = ll;
                    if c == 0 {
                        self.non_zero.insert(rhs);
                    }
                    if ri.lo == Some(c) && ri.hi == Some(c) {
                        self.is_bottom = true;
                        return;
                    }
                    if ri.lo == Some(c)
                        && let Some(c1) = c.checked_add(1)
                    {
                        self.add_lower(rhs, c1);
                    }
                    if ri.hi == Some(c)
                        && let Some(cm1) = c.checked_sub(1)
                    {
                        self.add_upper(rhs, cm1);
                    }
                }
            }
        }
        // Difference & sum algebra:
        // 1) `(a - b) op const_c` => direct DBM edge between `a` and `b`!
        //    e.g., `src.len - start >= 3` => `start - src.len <= -3` (proves `start + 2 < src.len`).
        let apply_diff_const = |st: &mut DbmState, a: VarKey, b: VarKey, eff_op: CmpOp, c: i128| {
            match eff_op {
                CmpOp::GtEq => {
                    if let Some(nc) = c.checked_neg() {
                        st.add_diff(b, a, nc);
                    }
                }
                CmpOp::Gt => {
                    if let Some(c1) = c.checked_add(1).and_then(|v| v.checked_neg()) {
                        st.add_diff(b, a, c1);
                    }
                }
                CmpOp::LtEq => st.add_diff(a, b, c),
                CmpOp::Lt => {
                    if let Some(cm1) = c.checked_sub(1) {
                        st.add_diff(a, b, cm1);
                    }
                }
                CmpOp::Eq => {
                    st.add_diff(a, b, c);
                    if let Some(nc) = c.checked_neg() {
                        st.add_diff(b, a, nc);
                    }
                }
                CmpOp::NotEq => {}
            }
        };
        let li = self.get_interval(lhs);
        let ri = self.get_interval(rhs);
        if let Some(&(a, b)) = self.diff_exprs.get(&lhs)
            && let (Some(rl), Some(rh)) = (ri.lo, ri.hi)
            && rl == rh
        {
            apply_diff_const(self, a, b, op, rl);
        }
        if let Some(&(a, b)) = self.diff_exprs.get(&rhs)
            && let (Some(ll), Some(lh)) = (li.lo, li.hi)
            && ll == lh
        {
            let flipped = match op {
                CmpOp::Lt => CmpOp::Gt,
                CmpOp::LtEq => CmpOp::GtEq,
                CmpOp::Gt => CmpOp::Lt,
                CmpOp::GtEq => CmpOp::LtEq,
                CmpOp::Eq => CmpOp::Eq,
                CmpOp::NotEq => CmpOp::NotEq,
            };
            apply_diff_const(self, a, b, flipped, ll);
        }
        // 2) `(a + b) <= rhs` or `(a + b) < rhs` with lower bound on `b` (or `a`):
        //    e.g., `start + 2 < len` => `start - len <= -3`.
        let apply_sum_upper = |st: &mut DbmState, a: VarKey, b: VarKey, bound: VarKey, strict: bool| {
            let extra = if strict { 1i128 } else { 0i128 };
            let bi = st.get_interval(b);
            if let Some(blo) = bi.lo
                && let Some(edge) = blo.checked_add(extra).and_then(|v| v.checked_neg())
            {
                st.add_diff(a, bound, edge);
            }
            let ai = st.get_interval(a);
            if let Some(alo) = ai.lo
                && let Some(edge) = alo.checked_add(extra).and_then(|v| v.checked_neg())
            {
                st.add_diff(b, bound, edge);
            }
        };
        if let Some(&(a, b)) = self.sum_exprs.get(&lhs) {
            match op {
                CmpOp::Lt => apply_sum_upper(self, a, b, rhs, true),
                CmpOp::LtEq | CmpOp::Eq => apply_sum_upper(self, a, b, rhs, false),
                _ => {}
            }
        }
        if let Some(&(a, b)) = self.sum_exprs.get(&rhs) {
            match op {
                CmpOp::Gt => apply_sum_upper(self, a, b, lhs, true),
                CmpOp::GtEq | CmpOp::Eq => apply_sum_upper(self, a, b, lhs, false),
                _ => {}
            }
        }
    }

    fn refine_pred(&mut self, pred: &BoolPred, truth: bool) {
        if self.is_bottom {
            return;
        }
        match pred {
            BoolPred::Const(b) => {
                if *b != truth {
                    self.is_bottom = true;
                }
            }
            BoolPred::Cmp { op, lhs, rhs } => {
                let eff = if truth { *op } else { op.negate() };
                self.refine_cmp(eff, *lhs, *rhs);
            }
            BoolPred::Not(inner) => self.refine_pred(inner, !truth),
            BoolPred::And(a, b) => {
                if truth {
                    self.refine_pred(a, true);
                    self.refine_pred(b, true);
                } else {
                    let mut left = self.clone();
                    left.refine_pred(a, false);
                    let mut right = self.clone();
                    right.refine_pred(b, false);
                    *self = DbmState::join(&left, &right);
                }
            }
            BoolPred::Or(a, b) => {
                if !truth {
                    self.refine_pred(a, false);
                    self.refine_pred(b, false);
                } else {
                    let mut left = self.clone();
                    left.refine_pred(a, true);
                    let mut right = self.clone();
                    right.refine_pred(b, true);
                    *self = DbmState::join(&left, &right);
                }
            }
        }
    }

    fn join(a: &Self, b: &Self) -> Self {
        if a.is_bottom {
            return b.clone();
        }
        if b.is_bottom {
            return a.clone();
        }
        let mut out = DbmState::top();

        // Interval hull for keys present in both (or present in one -> top if missing).
        for (&k, &ia) in &a.intervals {
            if let Some(&ib) = b.intervals.get(&k) {
                out.intervals.insert(k, ia.hull(ib));
            }
        }
        for &k in &a.non_zero {
            if b.non_zero.contains(&k) {
                out.non_zero.insert(k);
            }
        }
        for (&k, &va) in &a.len_alias {
            if b.len_alias.get(&k) == Some(&va) {
                out.len_alias.insert(k, va);
            }
        }
        for (&v, pa) in &a.preds {
            if b.preds.get(&v) == Some(pa) {
                out.preds.insert(v, pa.clone());
            }
        }
        for (&k, &ka) in &a.known_bits {
            if let Some(&kb) = b.known_bits.get(&k) {
                out.known_bits.insert(k, ka.join(kb));
            }
        }
        for (&k, &va) in &a.diff_exprs {
            if b.diff_exprs.get(&k) == Some(&va) {
                out.diff_exprs.insert(k, va);
            }
        }
        for (&k, &va) in &a.sum_exprs {
            if b.sum_exprs.get(&k) == Some(&va) {
                out.sum_exprs.insert(k, va);
            }
        }
        for (&k, &va) in &a.checked_op {
            if b.checked_op.get(&k) == Some(&va) {
                out.checked_op.insert(k, va);
            }
        }
        for (&k, &va) in &a.slot_checked_op {
            if b.slot_checked_op.get(&k) == Some(&va) {
                out.slot_checked_op.insert(k, va);
            }
        }
        for (&k, &va) in &a.val_slot_origin {
            if b.val_slot_origin.get(&k) == Some(&va) {
                out.val_slot_origin.insert(k, va);
            }
        }
        for &k in &a.const_lits {
            if b.const_lits.contains(&k) {
                out.const_lits.insert(k);
            }
        }

        // Shared DBM vertices (prioritize priority vertices first, cap at MAX_DBM_VERTS).
        let mut shared: Vec<VarKey> = a
            .vmap
            .keys()
            .filter(|k| b.vmap.contains_key(k))
            .copied()
            .collect();
        shared.sort_by_key(|k| (!k.is_len(), !k.is_priority(), *k));
        shared.truncate(MAX_DBM_VERTS);

        let m = shared.len() + 1;
        out.verts = Vec::with_capacity(m);
        out.verts.push(None);
        for (idx, &k) in shared.iter().enumerate() {
            out.verts.push(Some(k));
            out.vmap.insert(k, idx + 1);
        }
        out.mat = vec![vec![None; m]; m];
        out.mat[0][0] = Some(0);
        for i in 1..m {
            let ki = shared[i - 1];
            let ia = a.vmap[&ki];
            let ib = b.vmap[&ki];
            out.mat[i][i] = Some(0);
            out.mat[i][0] = max_bound(a.mat[ia][0], b.mat[ib][0]);
            out.mat[0][i] = max_bound(a.mat[0][ia], b.mat[0][ib]);
            for j in 1..m {
                if i == j {
                    continue;
                }
                let kj = shared[j - 1];
                let ja = a.vmap[&kj];
                let jb = b.vmap[&kj];
                out.mat[i][j] = max_bound(a.mat[ia][ja], b.mat[ib][jb]);
            }
        }
        out.sync_intervals_from_dbm();
        out
    }

    fn widen(
        old: &Self,
        new: &Self,
        widen_keys: &BTreeSet<VarKey>,
        type_bounds_of: &impl Fn(VarKey) -> Option<(i128, i128)>,
    ) -> Self {
        if old.is_bottom {
            return new.clone();
        }
        if new.is_bottom {
            return old.clone();
        }
        let mut out = DbmState::top();

        for (&k, &inw) in &new.intervals {
            if widen_keys.contains(&k) {
                if let Some(&io) = old.intervals.get(&k) {
                    let tb = type_bounds_of(k);
                    out.intervals.insert(k, io.widen(inw, tb));
                } else {
                    out.intervals.insert(k, inw);
                }
            } else {
                // Loop-invariant SSA value or unmodified outer slot: preserve exact `new` bound.
                out.intervals.insert(k, inw);
            }
        }
        for &k in &old.non_zero {
            if new.non_zero.contains(&k) {
                out.non_zero.insert(k);
            }
        }
        for (&k, &va) in &old.len_alias {
            if new.len_alias.get(&k) == Some(&va) {
                out.len_alias.insert(k, va);
            }
        }
        for (&v, pa) in &old.preds {
            if new.preds.get(&v) == Some(pa) {
                out.preds.insert(v, pa.clone());
            }
        }
        for (&k, &kn) in &new.known_bits {
            if widen_keys.contains(&k) {
                if let Some(&ko) = old.known_bits.get(&k) {
                    out.known_bits.insert(k, ko.join(kn));
                }
            } else {
                out.known_bits.insert(k, kn);
            }
        }
        for (&k, &va) in &old.diff_exprs {
            if new.diff_exprs.get(&k) == Some(&va) && !widen_keys.contains(&k) {
                out.diff_exprs.insert(k, va);
            }
        }
        for (&k, &va) in &old.sum_exprs {
            if new.sum_exprs.get(&k) == Some(&va) && !widen_keys.contains(&k) {
                out.sum_exprs.insert(k, va);
            }
        }
        for (&k, &va) in &old.checked_op {
            if new.checked_op.get(&k) == Some(&va) {
                out.checked_op.insert(k, va);
            }
        }
        for (&k, &va) in &old.slot_checked_op {
            if new.slot_checked_op.get(&k) == Some(&va) {
                out.slot_checked_op.insert(k, va);
            }
        }
        for (&k, &va) in &old.val_slot_origin {
            if new.val_slot_origin.get(&k) == Some(&va) {
                out.val_slot_origin.insert(k, va);
            }
        }
        for &k in &old.const_lits {
            if new.const_lits.contains(&k) {
                out.const_lits.insert(k);
            }
        }

        let mut shared: Vec<VarKey> = old
            .vmap
            .keys()
            .filter(|k| new.vmap.contains_key(k))
            .copied()
            .collect();
        shared.sort_by_key(|k| (!k.is_len(), !k.is_priority(), *k));
        shared.truncate(MAX_DBM_VERTS);

        let m = shared.len() + 1;
        out.verts = Vec::with_capacity(m);
        out.verts.push(None);
        for (idx, &k) in shared.iter().enumerate() {
            out.verts.push(Some(k));
            out.vmap.insert(k, idx + 1);
        }
        out.mat = vec![vec![None; m]; m];
        out.mat[0][0] = Some(0);
        for i in 1..m {
            let ki = shared[i - 1];
            let io = old.vmap[&ki];
            let inw = new.vmap[&ki];
            let wi = widen_keys.contains(&ki);
            let tb = type_bounds_of(ki);
            out.mat[i][i] = Some(0);
            out.mat[i][0] = if !wi {
                new.mat[inw][0]
            } else if bound_le(new.mat[inw][0], old.mat[io][0]) {
                old.mat[io][0]
            } else {
                tb.map(|(_, h)| h)
            };
            out.mat[0][i] = if !wi {
                new.mat[0][inw]
            } else if bound_le(new.mat[0][inw], old.mat[0][io]) {
                old.mat[0][io]
            } else {
                tb.and_then(|(l, _)| l.checked_neg())
            };
            for j in 1..m {
                if i == j {
                    continue;
                }
                let kj = shared[j - 1];
                let jo = old.vmap[&kj];
                let jnw = new.vmap[&kj];
                let wj = widen_keys.contains(&kj);
                out.mat[i][j] = if !wi && !wj {
                    new.mat[inw][jnw]
                } else if bound_le(new.mat[inw][jnw], old.mat[io][jo]) {
                    old.mat[io][jo]
                } else if bound_le(new.mat[inw][jnw], Some(0)) {
                    // Standard DBM widening with threshold 0 (`p <= len - 1` -> `p <= len`).
                    Some(0)
                } else {
                    None
                };
            }
        }
        out.sync_intervals_from_dbm();
        out
    }

    fn narrow(old: &Self, new: &Self) -> Self {
        if old.is_bottom || new.is_bottom {
            return new.clone();
        }
        let mut out = old.clone();
        for (&k, &inw) in &new.intervals {
            let cur = out.intervals.get(&k).copied().unwrap_or_else(Interval::top);
            out.intervals.insert(k, cur.narrow(inw));
        }
        for i in 1..out.verts.len() {
            let Some(ki) = out.verts[i] else { continue };
            let Some(&inw) = new.vmap.get(&ki) else { continue };
            if bound_lt(new.mat[inw][0], out.mat[i][0]) {
                out.mat[i][0] = new.mat[inw][0];
            }
            if bound_lt(new.mat[0][inw], out.mat[0][i]) {
                out.mat[0][i] = new.mat[0][inw];
            }
            for j in 1..out.verts.len() {
                if i == j {
                    continue;
                }
                let Some(kj) = out.verts[j] else { continue };
                let Some(&jnw) = new.vmap.get(&kj) else { continue };
                if bound_lt(new.mat[inw][jnw], out.mat[i][j]) {
                    out.mat[i][j] = new.mat[inw][jnw];
                }
            }
        }
        out.sync_intervals_from_dbm();
        out
    }
}

struct FnContract {
    param_names: Vec<Symbol>,
    where_clauses: Vec<Expr>,
    ensures_clauses: Vec<Expr>,
}

type ContractMap = BTreeMap<CfgFnKey, FnContract>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FnVerifyInfo {
    pub proven_checks: usize,
    pub total_checks: usize,
    pub proven_safe_insts: usize,
    pub ranged_values: usize,
}

#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    pub functions: BTreeMap<CfgFnKey, FnVerifyInfo>,
}

/// Public entry point: verify all EIR functions with the Phase 3B `@analysis(1)` relational verifier.
pub fn verify_program(
    eir: &mut EirProgram,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) -> VerifyReport {
    // Collect interprocedural contracts (`where` and `ensures`) for every function.
    let mut contracts: ContractMap = BTreeMap::new();
    for f in &eir.functions {
        contracts.insert(
            f.key,
            FnContract {
                param_names: f.param_names.clone(),
                where_clauses: f.where_clauses.clone(),
                ensures_clauses: f.ensures_clauses.clone(),
            },
        );
    }

    let mut report = VerifyReport::default();
    for f in &mut eir.functions {
        let info = verify_function(f, &contracts, env, interner, diags);
        report.functions.insert(f.key, info);
    }
    report
}

fn verify_function(
    f: &mut EirFunction,
    contracts: &ContractMap,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) -> FnVerifyInfo {
    let n = f.blocks.len();
    if n == 0 {
        return FnVerifyInfo::default();
    }

    // Identify loop headers (targets of back-edges where `target` dominates `src`) and their natural loop widen-keys.
    let mut loop_headers = BTreeSet::new();
    let mut header_back_srcs: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        let src = BlockId(bi as u32);
        for succ in term_succs(&b.term) {
            if dominates_dense(&f.idom, succ, src) {
                loop_headers.insert(succ);
                header_back_srcs.entry(succ).or_default().push(src);
            }
        }
    }
    let mut header_widen_keys: BTreeMap<BlockId, BTreeSet<VarKey>> = BTreeMap::new();
    for (&hdr, srcs) in &header_back_srcs {
        let mut loop_blocks = BTreeSet::new();
        loop_blocks.insert(hdr);
        let mut work: Vec<BlockId> = Vec::new();
        for &s in srcs {
            if loop_blocks.insert(s) {
                work.push(s);
            }
        }
        while let Some(cur) = work.pop() {
            for &p in &f.preds[cur.0 as usize] {
                if loop_blocks.insert(p) {
                    work.push(p);
                }
            }
        }
        let mut wkeys = BTreeSet::new();
        for phi in &f.blocks[hdr.0 as usize].phis {
            wkeys.insert(VarKey::Val(phi.dst));
            wkeys.insert(VarKey::LenVal(phi.dst));
        }
        for &lb in &loop_blocks {
            for inst in &f.blocks[lb.0 as usize].insts {
                if let Some(dv) = inst_def_val(&inst.kind) {
                    wkeys.insert(VarKey::Val(dv));
                    wkeys.insert(VarKey::LenVal(dv));
                }
                match &inst.kind {
                    EirInstKind::StoreSlot { slot, .. }
                    | EirInstKind::SlotAddr { slot, is_mut: true, .. } => {
                        wkeys.insert(VarKey::Slot(*slot));
                        wkeys.insert(VarKey::LenSlot(*slot));
                    }
                    EirInstKind::StorePlace { place, .. } => {
                        if let Some(fk) = field_key_of_place_static(place) {
                            wkeys.insert(fk);
                            match fk {
                                VarKey::FieldSlot(s, i) => { wkeys.insert(VarKey::FieldLenSlot(s, i)); }
                                VarKey::FieldVal(v, i) => { wkeys.insert(VarKey::FieldLenVal(v, i)); }
                                _ => {}
                            }
                        }
                    }
                    EirInstKind::BuiltinCall { arg_slots, .. } => {
                        for s in arg_slots.iter().flatten() {
                            wkeys.insert(VarKey::Slot(*s));
                            wkeys.insert(VarKey::LenSlot(*s));
                        }
                    }
                    _ => {}
                }
            }
        }
        header_widen_keys.insert(hdr, wkeys);
    }

    let type_bounds_of = |k: VarKey| -> Option<(i128, i128)> {
        match k {
            VarKey::Val(v) => f.value_tys.get(v.0 as usize).and_then(|&t| ty_bounds(t, env)),
            VarKey::Slot(s) => f.slots.get(s.0 as usize).and_then(|si| ty_bounds(si.ty, env)),
            VarKey::LenVal(_) | VarKey::LenSlot(_) | VarKey::FieldLenVal(_, _) | VarKey::FieldLenSlot(_, _) => {
                Some((0, i64::MAX as i128))
            }
            VarKey::FieldVal(_, _) | VarKey::FieldSlot(_, _) => None,
        }
    };

    // Build initial entry state from parameter types, Axiom A1, and `where` clauses.
    let mut entry_state = DbmState::top();
    for (pi, &pv) in f.params.iter().enumerate() {
        let pty = f.sig.params[pi].ty;
        if let Some((lo, hi)) = ty_bounds(pty, env) {
            entry_state.set_interval(VarKey::Val(pv), Interval::from_range(lo, hi));
        }
        init_len_for_key(&mut entry_state, VarKey::LenVal(pv), pty, env);
    }
    for s in &f.slots {
        if let Some((lo, hi)) = ty_bounds(s.ty, env) {
            entry_state.set_interval(VarKey::Slot(s.id), Interval::from_range(lo, hi));
        }
        init_len_for_key(&mut entry_state, VarKey::LenSlot(s.id), s.ty, env);
    }

    // Assume function `where` preconditions on entry!
    let param_env = build_param_env(f);
    for w in &f.where_clauses {
        assume_contract_expr(&mut entry_state, w, &param_env, None, interner);
    }

    let mut in_states: Vec<DbmState> = vec![DbmState::bottom(); n];
    in_states[0] = entry_state;
    let mut header_iters: Vec<usize> = vec![0; n];

    // Phase 1: Forward abstract interpretation with widening on loop headers.
    let mut edge_states: BTreeMap<(BlockId, BlockId), DbmState> = BTreeMap::new();
    for _pass in 0..MAX_FIXPOINT_ITERS {
        let mut changed = false;
        for &bb in &f.rpo {
            let bi = bb.0 as usize;
            if bi != 0 {
                let mut joined = DbmState::bottom();
                for &p in &f.preds[bi] {
                    if let Some(es) = edge_states.get(&(p, bb)) {
                        joined = DbmState::join(&joined, es);
                    }
                }
                if loop_headers.contains(&bb) && !in_states[bi].is_bottom {
                    header_iters[bi] += 1;
                    if header_iters[bi] > WIDEN_AFTER_ITERS {
                        let empty = BTreeSet::new();
                        let wkeys = header_widen_keys.get(&bb).unwrap_or(&empty);
                        joined = DbmState::widen(&in_states[bi], &joined, wkeys, &type_bounds_of);
                    }
                }
                // Apply explicit `while ... where ...` loop invariants if present.
                if let Some(invs) = f.loop_invariants.get(&bb) {
                    let sym_env = build_block_sym_env(f, &joined);
                    for inv in invs {
                        assume_contract_expr(&mut joined, inv, &sym_env, None, interner);
                    }
                }
                if joined != in_states[bi] {
                    in_states[bi] = joined;
                    changed = true;
                }
            }
            if in_states[bi].is_bottom {
                continue;
            }
            let mut st = in_states[bi].clone();
            for inst in &f.blocks[bi].insts {
                transfer_inst(&mut st, inst, f, contracts, env, interner);
                if st.is_bottom {
                    break;
                }
            }
            compute_out_edges(bb, &f.blocks[bi], &st, f, env, &mut edge_states);
        }
        if !changed {
            break;
        }
    }

    // Phase 2: Narrowing passes (tighten loop headers without widening).
    for _ in 0..NARROW_ITERS {
        for &bb in &f.rpo {
            let bi = bb.0 as usize;
            if bi != 0 {
                let mut joined = DbmState::bottom();
                for &p in &f.preds[bi] {
                    if let Some(es) = edge_states.get(&(p, bb)) {
                        joined = DbmState::join(&joined, es);
                    }
                }
                if loop_headers.contains(&bb) {
                    in_states[bi] = DbmState::narrow(&in_states[bi], &joined);
                } else {
                    in_states[bi] = joined;
                }
            }
            if in_states[bi].is_bottom {
                continue;
            }
            let mut st = in_states[bi].clone();
            for inst in &f.blocks[bi].insts {
                transfer_inst(&mut st, inst, f, contracts, env, interner);
                if st.is_bottom {
                    break;
                }
            }
            compute_out_edges(bb, &f.blocks[bi], &st, f, env, &mut edge_states);
        }
    }

    // Phase 3: Single post-convergence verification pass (emits diagnostics once + records proven facts).
    let mut value_ranges: BTreeMap<EirValue, (i128, i128)> = BTreeMap::new();
    let mut proven_safe: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    let mut stats = FnVerifyInfo::default();

    for &bb in &f.rpo {
        let bi = bb.0 as usize;
        if in_states[bi].is_bottom {
            if matches!(f.blocks[bi].term, EirTerm::Unreachable { explicit: true, .. }) {
                stats.total_checks += 1;
                stats.proven_checks += 1;
            }
            continue;
        }
        let mut st = in_states[bi].clone();
        for (ii, inst) in f.blocks[bi].insts.iter().enumerate() {
            verify_inst(
                bb,
                ii,
                inst,
                &st,
                f,
                contracts,
                env,
                interner,
                diags,
                &mut proven_safe,
                &mut stats,
            );
            transfer_inst(&mut st, inst, f, contracts, env, interner);
            if let Some(dst) = inst_def_val(&inst.kind)
                && env.is_integer(f.value_tys[dst.0 as usize])
            {
                let iv = st.get_interval(VarKey::Val(dst));
                if let (Some(lo), Some(hi)) = (iv.lo, iv.hi) {
                    value_ranges.insert(dst, (lo, hi));
                }
            }
            if st.is_bottom {
                break;
            }
        }
        if st.is_bottom {
            if matches!(f.blocks[bi].term, EirTerm::Unreachable { explicit: true, .. }) {
                stats.total_checks += 1;
                stats.proven_checks += 1;
            }
            continue;
        }

        // Verify terminator checks (`unreachable;`, `ensures` on `Return`, dead user `if` `ZM-14`).
        match &f.blocks[bi].term {
            EirTerm::Unreachable { span, explicit: true } => {
                stats.total_checks += 1;
                let mut d = Diagnostic::error(
                    ErrorCode::E0115,
                    *span,
                    "ZM-15: `unreachable;` statement is statically reachable",
                );
                d.notes.push(format_state_summary(&st, f, interner));
                diags.emit(d);
            }
            EirTerm::Return(Some(val)) => {
                if !f.ensures_clauses.is_empty() {
                    // Only check ensures on Ok values (or non-error-union returns).
                    let is_err_ret = matches!(env.get(f.sig.ret), Ty::ErrorUnion { .. })
                        && is_definitely_err_val(*val, &f.blocks[bi].insts);
                    if !is_err_ret {
                        let sym_env = build_block_sym_env(f, &st);
                        for ens in &f.ensures_clauses {
                            stats.total_checks += 1;
                            if check_contract_expr(&st, ens, &sym_env, Some(VarKey::Val(*val)), interner) {
                                stats.proven_checks += 1;
                            } else {
                                let mut d = Diagnostic::error(
                                    ErrorCode::E0108,
                                    ens.span,
                                    "cannot prove postcondition `ensures` at return",
                                );
                                d.notes.push(format!(
                                    "known: return in {}",
                                    st.get_interval(VarKey::Val(*val)).fmt_str()
                                ));
                                diags.emit(d);
                            }
                        }
                    }
                }
            }
            EirTerm::Branch {
                cond,
                span,
                is_user_if: true,
                ..
            } => {
                if let Some(pred) = st.preds.get(cond).cloned() {
                    let mut then_st = st.clone();
                    then_st.refine_pred(&pred, true);
                    let mut else_st = st.clone();
                    else_st.refine_pred(&pred, false);
                    if then_st.is_bottom && !else_st.is_bottom {
                        diags.error(
                            ErrorCode::E0114,
                            *span,
                            "ZM-14: `if` condition is always false; dead `then` branch (use `comptime if` for conditional compilation)",
                        );
                    } else if else_st.is_bottom && !then_st.is_bottom {
                        diags.error(
                            ErrorCode::E0114,
                            *span,
                            "ZM-14: `if` condition is always true; dead `else` branch (use `comptime if` for conditional compilation)",
                        );
                    }
                }
            }
            _ => {}
        }
    }

    stats.proven_safe_insts = proven_safe.len();
    stats.ranged_values = value_ranges.len();
    f.value_ranges = value_ranges;
    f.proven_safe_insts = proven_safe;
    stats
}

fn is_definitely_err_val(val: EirValue, insts: &[EirInst]) -> bool {
    for inst in insts.iter().rev() {
        if let EirInstKind::Cast {
            dst,
            kind: crate::cfg::CastKind::WrapErr,
            ..
        } = &inst.kind
            && *dst == val
        {
            return true;
        }
        if let EirInstKind::LoadSlot { dst, slot } = &inst.kind
            && *dst == val
        {
            for prev in insts.iter().rev() {
                if let EirInstKind::StoreSlot { slot: s2, val: v2 } = &prev.kind
                    && s2 == slot
                {
                    return is_definitely_err_val(*v2, insts);
                }
            }
        }
    }
    false
}

fn compute_out_edges(
    bb: BlockId,
    blk: &crate::eir::EirBlock,
    out: &DbmState,
    f: &EirFunction,
    env: &TypeEnv,
    edge_states: &mut BTreeMap<(BlockId, BlockId), DbmState>,
) {
    match &blk.term {
        EirTerm::Goto(t) => {
            let mut st = out.clone();
            apply_phis_on_edge(&mut st, bb, *t, f, env);
            edge_states.insert((bb, *t), st);
        }
        EirTerm::Branch {
            cond,
            then_bb,
            else_bb,
            ..
        } => {
            let mut then_st = out.clone();
            let mut else_st = out.clone();
            if let Some(pred) = out.preds.get(cond).cloned() {
                then_st.refine_pred(&pred, true);
                else_st.refine_pred(&pred, false);
            }
            apply_phis_on_edge(&mut then_st, bb, *then_bb, f, env);
            apply_phis_on_edge(&mut else_st, bb, *else_bb, f, env);
            edge_states.insert((bb, *then_bb), then_st);
            edge_states.insert((bb, *else_bb), else_st);
        }
        EirTerm::SwitchInt {
            discr,
            targets,
            otherwise,
        } => {
            let mut oth_st = out.clone();
            let mut arm_states: BTreeMap<BlockId, DbmState> = BTreeMap::new();
            for &(cval, tgt) in targets {
                let mut ast = out.clone();
                ast.add_const(VarKey::Val(*discr), cval);
                let prev = arm_states.remove(&tgt).unwrap_or_else(DbmState::bottom);
                arm_states.insert(tgt, DbmState::join(&prev, &ast));
                // Tighten `otherwise` for `discr != cval`.
                let iv = oth_st.get_interval(VarKey::Val(*discr));
                if cval == 0 {
                    oth_st.non_zero.insert(VarKey::Val(*discr));
                }
                if iv.lo == Some(cval) && iv.hi == Some(cval) {
                    oth_st.is_bottom = true;
                } else if iv.lo == Some(cval)
                    && let Some(c1) = cval.checked_add(1)
                {
                    oth_st.add_lower(VarKey::Val(*discr), c1);
                } else if iv.hi == Some(cval)
                    && let Some(cm1) = cval.checked_sub(1)
                {
                    oth_st.add_upper(VarKey::Val(*discr), cm1);
                }
            }
            for (tgt, mut st) in arm_states {
                apply_phis_on_edge(&mut st, bb, tgt, f, env);
                edge_states.insert((bb, tgt), st);
            }
            apply_phis_on_edge(&mut oth_st, bb, *otherwise, f, env);
            edge_states.insert((bb, *otherwise), oth_st);
        }
        EirTerm::Return(_) | EirTerm::Unreachable { .. } => {}
    }
}

fn apply_phis_on_edge(
    st: &mut DbmState,
    from_bb: BlockId,
    to_bb: BlockId,
    f: &EirFunction,
    env: &TypeEnv,
) {
    if st.is_bottom {
        return;
    }
    let phis = &f.blocks[to_bb.0 as usize].phis;
    if phis.is_empty() {
        return;
    }
    // Evaluate incoming values in pre-phi state (parallel copy semantics).
    let snap = st.clone();
    for phi in phis {
        let Some(&(_, inc)) = phi.incoming.iter().find(|&&(p, _)| p == from_bb) else {
            continue;
        };
        if env.is_integer(phi.ty) {
            let dk = VarKey::Val(phi.dst);
            let sk = VarKey::Val(inc);
            let dst_tb = ty_bounds(phi.ty, env);
            st.forget(dk);
            if let Some((tl, th)) = dst_tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
            let siv = snap.get_interval(sk);
            st.set_interval(dk, siv);
            // If `sk` is a loop-invariant outer variable (not one of this block's own phis),
            // bind `dk == sk` directly so monotonic loop invariants like `start <= p` are seeded
            // on the loop-entry edge and survive widening!
            if sk != dk && !phis.iter().any(|p2| VarKey::Val(p2.dst) == sk) {
                st.add_eq(dk, sk);
            }
            if let Some(&kb) = snap.known_bits.get(&sk) {
                st.known_bits.insert(dk, kb);
            }
            // Copy relational bounds from `sk` in `snap` to `dk` for all stable vertices (like `Len*` and params).
            if let Some(&si) = snap.vmap.get(&sk) {
                for (oi, okey_opt) in snap.verts.iter().enumerate().skip(1) {
                    let Some(ok) = *okey_opt else { continue };
                    if ok == dk || ok == sk {
                        continue;
                    }
                    // Skip other phi destinations to avoid stale reads in parallel swaps.
                    if phis.iter().any(|p2| VarKey::Val(p2.dst) == ok) {
                        continue;
                    }
                    if let Some(c) = snap.mat[si][oi] {
                        st.add_diff(dk, ok, c);
                    }
                    if let Some(c) = snap.mat[oi][si] {
                        st.add_diff(ok, dk, c);
                    }
                }
            }
            if snap.is_non_zero(sk) {
                st.non_zero.insert(dk);
            }
        }
        // Propagate symbolic length across slice/pool phis.
        let lsrc = VarKey::LenVal(inc);
        let ldst = VarKey::LenVal(phi.dst);
        if snap.intervals.contains_key(&lsrc) || snap.vmap.contains_key(&lsrc) {
            st.add_eq(ldst, lsrc);
        }
    }
}

fn init_len_for_key(st: &mut DbmState, lk: VarKey, ty: TyId, env: &TypeEnv) {
    let deref = match env.get(ty) {
        Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t,
        _ => ty,
    };
    match env.get(ty) {
        Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str => {
            // Axiom A1: slice/str length is in [0, isize::MAX].
            st.set_interval(lk, Interval::from_range(0, i64::MAX as i128));
        }
        _ => match env.get(deref) {
            Ty::Array { len, .. } => {
                st.add_const(lk, *len as i128);
            }
            Ty::Builtin(BuiltinKind::ArenaPool | BuiltinKind::SlotPool, _) => {
                // Axiom A1: pool length is in [0, isize::MAX].
                st.set_interval(lk, Interval::from_range(0, i64::MAX as i128));
            }
            _ => {}
        },
    }
}

fn ty_bounds(ty: TyId, env: &TypeEnv) -> Option<(i128, i128)> {
    int_bounds(env.get(ty))
}

fn int_bitwidth(ty: &Ty) -> u32 {
    match ty {
        Ty::Int { bits, is_ptr_sized, .. } => if *is_ptr_sized { 64 } else { *bits as u32 },
        _ => 64,
    }
}

fn field_key_of_place_static(place: &EirPlace) -> Option<VarKey> {
    let fidx = match place.projs.as_slice() {
        [EirProj::Field { field_idx, .. }] => *field_idx as u32,
        [EirProj::Deref, EirProj::Field { field_idx, .. }] => *field_idx as u32,
        [EirProj::TupleField { index, .. }] => *index as u32,
        [EirProj::Deref, EirProj::TupleField { index, .. }] => *index as u32,
        _ => return None,
    };
    match place.base {
        EirPlaceBase::Slot(s) => Some(VarKey::FieldSlot(s, fidx)),
        EirPlaceBase::Val(v) => Some(VarKey::FieldVal(v, fidx)),
    }
}

fn field_key_of_place(st: &DbmState, place: &EirPlace, f: &EirFunction, env: &TypeEnv) -> Option<VarKey> {
    let fidx = match place.projs.as_slice() {
        [EirProj::Field { field_idx, .. }] => *field_idx as u32,
        [EirProj::Deref, EirProj::Field { field_idx, .. }] => *field_idx as u32,
        [EirProj::TupleField { index, .. }] => *index as u32,
        [EirProj::Deref, EirProj::TupleField { index, .. }] => *index as u32,
        _ => return None,
    };
    // Avoid treating RowView (`*Pool`) fields as single-struct fields, since row index is dynamic.
    let base_ty = match place.base {
        EirPlaceBase::Slot(s) => f.slots[s.0 as usize].ty,
        EirPlaceBase::Val(v) => f.value_tys[v.0 as usize],
    };
    let deref_t = match env.get(base_ty) {
        Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
        _ => base_ty,
    };
    if matches!(env.get(deref_t), Ty::Pool { .. }) {
        return None;
    }
    match place.base {
        EirPlaceBase::Slot(s) => Some(VarKey::FieldSlot(s, fidx)),
        EirPlaceBase::Val(v) => {
            if let Some(&orig_slot) = st.val_slot_origin.get(&v) {
                Some(VarKey::FieldSlot(orig_slot, fidx))
            } else {
                Some(VarKey::FieldVal(v, fidx))
            }
        }
    }
}

fn transfer_inst(
    st: &mut DbmState,
    inst: &EirInst,
    f: &EirFunction,
    contracts: &ContractMap,
    env: &TypeEnv,
    interner: &Interner,
) {
    match &inst.kind {
        EirInstKind::Const { dst, ty, konst } => match konst {
            CfgConst::Int(v, _) => {
                st.add_const(VarKey::Val(*dst), *v);
                st.const_lits.insert(VarKey::Val(*dst));
                if *v >= 0 {
                    st.known_bits.insert(VarKey::Val(*dst), KnownBits::from_const(*v as u128));
                }
            }
            CfgConst::Bool(b) => {
                st.preds.insert(*dst, BoolPred::Const(*b));
            }
            CfgConst::Str(s) => {
                st.add_const(VarKey::LenVal(*dst), s.len() as i128);
            }
            CfgConst::Bytes(bs) => {
                st.add_const(VarKey::LenVal(*dst), bs.len() as i128);
            }
            _ => {
                init_len_for_key(st, VarKey::LenVal(*dst), *ty, env);
            }
        },
        EirInstKind::CopyVal { dst, src } => {
            let dty = f.value_tys[dst.0 as usize];
            if env.is_integer(dty) {
                st.assign_offset(VarKey::Val(*dst), VarKey::Val(*src), 0, ty_bounds(dty, env));
                if let Some(&kb) = st.known_bits.get(&VarKey::Val(*src)) {
                    st.known_bits.insert(VarKey::Val(*dst), kb);
                }
            }
            if let Some(p) = st.preds.get(src).cloned() {
                st.preds.insert(*dst, p);
            }
            if let Some(&orig) = st.val_slot_origin.get(src) {
                st.val_slot_origin.insert(*dst, orig);
            }
            if let Some(&cop) = st.checked_op.get(src) {
                st.checked_op.insert(*dst, cop);
            }
            if has_len_ty(dty, env) {
                st.add_eq(VarKey::LenVal(*dst), VarKey::LenVal(*src));
            }
        }
        EirInstKind::LoadSlot { dst, slot } => {
            let dty = f.value_tys[dst.0 as usize];
            if env.is_integer(dty) {
                st.assign_offset(VarKey::Val(*dst), VarKey::Slot(*slot), 0, ty_bounds(dty, env));
                if let Some(&kb) = st.known_bits.get(&VarKey::Slot(*slot)) {
                    st.known_bits.insert(VarKey::Val(*dst), kb);
                }
            }
            if matches!(env.get(dty), Ty::Borrow { inner: BorrowInner::Single(_), .. } | Ty::RawPtr { .. }) {
                st.val_slot_origin.insert(*dst, *slot);
            }
            if let Some(&cop) = st.slot_checked_op.get(slot) {
                st.checked_op.insert(*dst, cop);
            }
            if has_len_ty(dty, env) {
                init_len_for_key(st, VarKey::LenSlot(*slot), dty, env);
                st.add_eq(VarKey::LenVal(*dst), VarKey::LenSlot(*slot));
            }
        }
        EirInstKind::StoreSlot { slot, val } => {
            let sty = f.slots[slot.0 as usize].ty;
            st.forget_fields_for_slot(*slot);
            if let Some(&cop) = st.checked_op.get(val) {
                st.slot_checked_op.insert(*slot, cop);
            } else {
                st.slot_checked_op.remove(slot);
            }
            if env.is_integer(sty) {
                st.assign_offset(VarKey::Slot(*slot), VarKey::Val(*val), 0, ty_bounds(sty, env));
                if let Some(&kb) = st.known_bits.get(&VarKey::Val(*val)) {
                    st.known_bits.insert(VarKey::Slot(*slot), kb);
                }
            }
            if has_len_ty(sty, env) {
                st.forget(VarKey::LenSlot(*slot));
                init_len_for_key(st, VarKey::LenSlot(*slot), sty, env);
                let lval = VarKey::LenVal(*val);
                if st.intervals.contains_key(&lval) || st.vmap.contains_key(&lval) {
                    st.add_eq(VarKey::LenSlot(*slot), lval);
                }
            }
        }
        EirInstKind::SlotAddr { dst, slot, ty, .. } => {
            st.val_slot_origin.insert(*dst, *slot);
            let sty = f.slots[slot.0 as usize].ty;
            if has_len_ty(sty, env) || has_len_ty(*ty, env) {
                init_len_for_key(st, VarKey::LenSlot(*slot), sty, env);
                st.add_eq(VarKey::LenVal(*dst), VarKey::LenSlot(*slot));
            }
        }
        EirInstKind::LoadPlace { dst, place } => {
            let dty = f.value_tys[dst.0 as usize];
            if let Some(fk) = field_key_of_place(st, place, f, env) {
                if env.is_integer(dty) {
                    if let Some((lo, hi)) = ty_bounds(dty, env) {
                        st.set_interval(fk, Interval::from_range(lo, hi));
                    }
                    st.assign_offset(VarKey::Val(*dst), fk, 0, ty_bounds(dty, env));
                    if let Some(&kb) = st.known_bits.get(&fk) {
                        st.known_bits.insert(VarKey::Val(*dst), kb);
                    }
                } else {
                    init_len_for_key(st, VarKey::LenVal(*dst), dty, env);
                    if has_len_ty(dty, env) {
                        let flk = match fk {
                            VarKey::FieldSlot(s, idx) => VarKey::FieldLenSlot(s, idx),
                            VarKey::FieldVal(v, idx) => VarKey::FieldLenVal(v, idx),
                            other => other,
                        };
                        init_len_for_key(st, flk, dty, env);
                        st.add_eq(VarKey::LenVal(*dst), flk);
                    }
                }
            } else {
                if let Some((lo, hi)) = ty_bounds(dty, env) {
                    st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
                }
                init_len_for_key(st, VarKey::LenVal(*dst), dty, env);
            }
        }
        EirInstKind::StorePlace { place, val } => {
            if let Some(fk) = field_key_of_place(st, place, f, env) {
                st.forget(fk);
                let flk = match fk {
                    VarKey::FieldSlot(s, idx) => Some(VarKey::FieldLenSlot(s, idx)),
                    VarKey::FieldVal(v, idx) => Some(VarKey::FieldLenVal(v, idx)),
                    _ => None,
                };
                if let Some(flk) = flk {
                    st.forget(flk);
                    if has_len_ty(place.ty, env) {
                        init_len_for_key(st, flk, place.ty, env);
                        st.add_eq(flk, VarKey::LenVal(*val));
                    }
                }
                if env.is_integer(place.ty) {
                    st.assign_offset(fk, VarKey::Val(*val), 0, ty_bounds(place.ty, env));
                    if let Some(&kb) = st.known_bits.get(&VarKey::Val(*val)) {
                        st.known_bits.insert(fk, kb);
                    }
                }
            } else {
                st.forget_all_mutable_fields(f, env);
            }
        }
        EirInstKind::PoolColumnLoad { dst, .. } => {
            let dty = f.value_tys[dst.0 as usize];
            if let Some((lo, hi)) = ty_bounds(dty, env) {
                st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
            }
            init_len_for_key(st, VarKey::LenVal(*dst), dty, env);
        }
        EirInstKind::Unary { dst, op, operand, ty } => match op {
            UnaryOp::Not => {
                if let Some(p) = st.preds.get(operand).cloned() {
                    st.preds.insert(*dst, BoolPred::Not(Box::new(p)));
                }
            }
            UnaryOp::Neg if env.is_integer(*ty) => {
                let iv = st.get_interval(VarKey::Val(*operand));
                let lo = iv.hi.and_then(|h| h.checked_neg());
                let hi = iv.lo.and_then(|l| l.checked_neg());
                let mut res = Interval { lo, hi };
                if let Some((tl, th)) = ty_bounds(*ty, env) {
                    res = res.intersect(Interval::from_range(tl, th));
                }
                st.set_interval(VarKey::Val(*dst), res);
            }
            _ => {
                if let Some((lo, hi)) = ty_bounds(*ty, env) {
                    st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
                }
            }
        },
        EirInstKind::Binary { dst, op, lhs, rhs, op_ty, dst_ty } => {
            transfer_binary(st, *dst, *op, *lhs, *rhs, *op_ty, *dst_ty, env);
        }
        EirInstKind::Cast { dst, kind, operand, from_ty, to_ty } => {
            match kind {
                crate::cfg::CastKind::Widen => {
                    if env.is_integer(*from_ty) && env.is_integer(*to_ty) {
                        st.assign_offset(VarKey::Val(*dst), VarKey::Val(*operand), 0, ty_bounds(*to_ty, env));
                    }
                }
                crate::cfg::CastKind::Trunc => {
                    if env.is_integer(*from_ty) && env.is_integer(*to_ty) {
                        let siv = st.get_interval(VarKey::Val(*operand));
                        if let Some((tl, th)) = ty_bounds(*to_ty, env) {
                            if siv.lo.is_some_and(|l| l >= tl) && siv.hi.is_some_and(|h| h <= th) {
                                st.assign_offset(VarKey::Val(*dst), VarKey::Val(*operand), 0, Some((tl, th)));
                            } else {
                                st.set_interval(VarKey::Val(*dst), Interval::from_range(tl, th));
                            }
                        }
                    }
                }
                crate::cfg::CastKind::Clamp => {
                    if let Some((tl, th)) = ty_bounds(*to_ty, env) {
                        let siv = st.get_interval(VarKey::Val(*operand));
                        let lo = siv.lo.map_or(tl, |l| l.clamp(tl, th));
                        let hi = siv.hi.map_or(th, |h| h.clamp(tl, th));
                        st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
                    }
                }
                crate::cfg::CastKind::ArrayBorrowToSlice => {
                    let arr_len = match env.get(*from_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(at), .. } => match env.get(*at) {
                            Ty::Array { len, .. } => Some(*len as i128),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(n) = arr_len {
                        st.add_const(VarKey::LenVal(*dst), n);
                    } else {
                        init_len_for_key(st, VarKey::LenVal(*dst), *to_ty, env);
                    }
                }
                crate::cfg::CastKind::MutToConst => {
                    if has_len_ty(*to_ty, env) {
                        st.add_eq(VarKey::LenVal(*dst), VarKey::LenVal(*operand));
                    }
                }
                _ => {
                    if let Some((lo, hi)) = ty_bounds(*to_ty, env) {
                        st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
                    }
                }
            }
        }
        EirInstKind::Slice { dst, base, base_ty, lo, hi, .. } => {
            let base_len_k = VarKey::LenVal(*base);
            init_len_for_key(st, base_len_k, *base_ty, env);
            let lo_iv = lo.map_or(Interval::const_val(0), |v| st.get_interval(VarKey::Val(v)));
            let hi_iv = hi.map_or(st.get_interval(base_len_k), |v| st.get_interval(VarKey::Val(v)));
            let len_lo = match (hi_iv.lo, lo_iv.hi) {
                (Some(h), Some(l)) => (h - l).max(0),
                _ => 0,
            };
            let len_hi = match (hi_iv.hi, lo_iv.lo) {
                (Some(h), Some(l)) => (h - l).clamp(0, i64::MAX as i128),
                _ => i64::MAX as i128,
            };
            st.set_interval(VarKey::LenVal(*dst), Interval::from_range(len_lo, len_hi.max(len_lo)));
            if let (None, Some(hv)) = (lo, hi) {
                st.add_eq(VarKey::LenVal(*dst), VarKey::Val(*hv));
            } else if let (Some(lv), Some(hv)) = (lo, hi) {
                let liv = st.get_interval(VarKey::Val(*lv));
                if liv.lo == Some(0) && liv.hi == Some(0) {
                    st.add_eq(VarKey::LenVal(*dst), VarKey::Val(*hv));
                }
            }
        }
        EirInstKind::Aggregate { dst, dst_ty, .. } => {
            init_len_for_key(st, VarKey::LenVal(*dst), *dst_ty, env);
        }
        EirInstKind::UnwrapOptional { dst, dst_ty, .. }
        | EirInstKind::FieldExtract { dst, dst_ty, .. } => {
            if let Some((lo, hi)) = ty_bounds(*dst_ty, env) {
                st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
            }
            init_len_for_key(st, VarKey::LenVal(*dst), *dst_ty, env);
        }
        EirInstKind::UnwrapErrorOk { dst, src, dst_ty, .. } => {
            if let Some((cop, a, b, op_ty)) = st.checked_op.get(src).copied() {
                // On the non-error branch of a checked op (`+?`, `-?`, `*?`, `/?`, `%?`), the
                // operation succeeded without overflow / div-by-zero, so propagate exact proven
                // arithmetic postconditions (e.g. `r == a + b`, `b >= 0 => r >= a`).
                let proven_op = match cop {
                    BinaryOp::AddCheck => BinaryOp::Add,
                    BinaryOp::SubCheck => BinaryOp::Sub,
                    BinaryOp::MulCheck => BinaryOp::Mul,
                    BinaryOp::DivCheck => BinaryOp::Div,
                    BinaryOp::ModCheck => BinaryOp::Mod,
                    other => other,
                };
                transfer_binary(st, *dst, proven_op, a, b, op_ty, *dst_ty, env);
            } else {
                if let Some((lo, hi)) = ty_bounds(*dst_ty, env) {
                    st.set_interval(VarKey::Val(*dst), Interval::from_range(lo, hi));
                }
                init_len_for_key(st, VarKey::LenVal(*dst), *dst_ty, env);
            }
        }
        EirInstKind::BuiltinCall { dst, op, args, arg_slots, ret_ty } => {
            if let Some(dv) = dst {
                match op {
                    CfgBuiltinOp::SliceLen | CfgBuiltinOp::StrLen | CfgBuiltinOp::ArrayLen
                    | CfgBuiltinOp::ArenaPoolLen | CfgBuiltinOp::SlotPoolLen | CfgBuiltinOp::BufLen => {
                        let lk = if let Some(Some(slot)) = arg_slots.first() {
                            let sty = f.slots[slot.0 as usize].ty;
                            init_len_for_key(st, VarKey::LenSlot(*slot), sty, env);
                            VarKey::LenSlot(*slot)
                        } else if let Some(&a0) = args.first() {
                            let aty = f.value_tys[a0.0 as usize];
                            init_len_for_key(st, VarKey::LenVal(a0), aty, env);
                            VarKey::LenVal(a0)
                        } else {
                            VarKey::Val(*dv)
                        };
                        st.set_interval(VarKey::Val(*dv), Interval::from_range(0, i64::MAX as i128));
                        if lk != VarKey::Val(*dv) {
                            st.add_eq(VarKey::Val(*dv), lk);
                        }
                    }
                    CfgBuiltinOp::BufAsSlice { .. } => {
                        init_len_for_key(st, VarKey::LenVal(*dv), *ret_ty, env);
                        if let Some(Some(slot)) = arg_slots.first() {
                            let sty = f.slots[slot.0 as usize].ty;
                            init_len_for_key(st, VarKey::LenSlot(*slot), sty, env);
                            st.add_eq(VarKey::LenVal(*dv), VarKey::LenSlot(*slot));
                        } else if let Some(&a0) = args.first() {
                            let aty = f.value_tys[a0.0 as usize];
                            init_len_for_key(st, VarKey::LenVal(a0), aty, env);
                            st.add_eq(VarKey::LenVal(*dv), VarKey::LenVal(a0));
                        }
                    }
                    _ => {
                        if let Some((lo, hi)) = ty_bounds(*ret_ty, env) {
                            st.set_interval(VarKey::Val(*dv), Interval::from_range(lo, hi));
                        }
                        init_len_for_key(st, VarKey::LenVal(*dv), *ret_ty, env);
                    }
                }
            }
            // Pool / Buf mutation invalidates cached length unless tracked.
            match op {
                CfgBuiltinOp::ArenaPoolPush(_)
                | CfgBuiltinOp::ArenaPoolReset(_)
                | CfgBuiltinOp::SlotPoolInsert(_)
                | CfgBuiltinOp::SlotPoolTryInsert(_)
                | CfgBuiltinOp::SlotPoolRemove(_)
                | CfgBuiltinOp::BufPush(_)
                | CfgBuiltinOp::BufPushSlice(_)
                | CfgBuiltinOp::BufPop(_)
                | CfgBuiltinOp::BufClear => {
                    if let Some(Some(slot)) = arg_slots.first() {
                        st.forget(VarKey::LenSlot(*slot));
                        st.set_interval(VarKey::LenSlot(*slot), Interval::from_range(0, i64::MAX as i128));
                    }
                }
                _ => {}
            }
        }
        EirInstKind::Call { dst, callee, args, ret_ty } => {
            // Sound invalidation: if any argument may carry mutable access (`*mut`, rawptr, or
            // a struct/view-struct that may hold `*mut`), forget mutable field facts while keeping
            // shared `*T` field facts (guaranteed immutable by the Law of Exclusivity).
            let any_mut_arg = args.iter().any(|a| {
                let aty = f.value_tys.get(a.0 as usize).copied().unwrap_or(TyId::ERROR);
                matches!(
                    env.get(aty),
                    Ty::Borrow { is_mut: true, .. } | Ty::RawPtr { is_mut: true, .. }
                ) || matches!(env.get(aty), Ty::Struct { def, .. } if env.structs.get(def).is_some_and(|m| m.is_view))
            });
            if any_mut_arg {
                st.forget_all_mutable_fields(f, env);
            }
            if let Some(dv) = dst {
                if let Some((lo, hi)) = ty_bounds(*ret_ty, env) {
                    st.set_interval(VarKey::Val(*dv), Interval::from_range(lo, hi));
                }
                init_len_for_key(st, VarKey::LenVal(*dv), *ret_ty, env);
            }
            // Assume callee's `ensures` postconditions!
            let callee_key = match callee {
                EirCallee::DirectFn(d) => Some(CfgFnKey::TopDef(*d)),
                EirCallee::Method { impl_idx, sub_idx } => Some(CfgFnKey::ImplMethod {
                    impl_idx: *impl_idx,
                    sub_idx: *sub_idx,
                }),
                EirCallee::GenericInst { inst_id } => Some(CfgFnKey::GenericInst(*inst_id)),
                _ => None,
            };
            if let Some(ck) = callee_key
                && let Some(fc) = contracts.get(&ck)
                && !fc.ensures_clauses.is_empty()
            {
                let mut call_env: BTreeMap<Symbol, (VarKey, Option<VarKey>)> = BTreeMap::new();
                for (pn, &av) in fc.param_names.iter().zip(args.iter()) {
                    call_env.insert(*pn, (VarKey::Val(av), Some(VarKey::LenVal(av))));
                }
                let ret_k = dst.map(VarKey::Val);
                for ens in &fc.ensures_clauses {
                    assume_contract_expr(st, ens, &call_env, ret_k, interner);
                }
            }
        }
        _ => {}
    }
}

fn has_len_ty(ty: TyId, env: &TypeEnv) -> bool {
    let deref = match env.get(ty) {
        Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t,
        _ => ty,
    };
    matches!(
        env.get(ty),
        Ty::Borrow { inner: BorrowInner::Slice(_) | BorrowInner::Str, .. } | Ty::Str
    ) || matches!(
        env.get(deref),
        Ty::Array { .. }
            | Ty::Builtin(
                BuiltinKind::ArenaPool | BuiltinKind::SlotPool | BuiltinKind::Buf | BuiltinKind::LinearBuf,
                _
            )
    )
}

fn transfer_binary(
    st: &mut DbmState,
    dst: EirValue,
    op: BinaryOp,
    lhs: EirValue,
    rhs: EirValue,
    op_ty: TyId,
    dst_ty: TyId,
    env: &TypeEnv,
) {
    let lk = VarKey::Val(lhs);
    let rk = VarKey::Val(rhs);
    let dk = VarKey::Val(dst);

    // Record comparison predicates for path-sensitive branching!
    match op {
        BinaryOp::Lt => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::Lt, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::LtEq => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::LtEq, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::Gt => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::Gt, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::GtEq => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::GtEq, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::Eq => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::Eq, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::NotEq => {
            st.preds.insert(dst, BoolPred::Cmp { op: CmpOp::NotEq, lhs: lk, rhs: rk });
            return;
        }
        BinaryOp::And => {
            if let (Some(lp), Some(rp)) = (st.preds.get(&lhs).cloned(), st.preds.get(&rhs).cloned()) {
                st.preds.insert(dst, BoolPred::And(Box::new(lp), Box::new(rp)));
            }
            return;
        }
        BinaryOp::Or => {
            if let (Some(lp), Some(rp)) = (st.preds.get(&lhs).cloned(), st.preds.get(&rhs).cloned()) {
                st.preds.insert(dst, BoolPred::Or(Box::new(lp), Box::new(rp)));
            }
            return;
        }
        _ => {}
    }

    // Record checked arithmetic ops so `UnwrapErrorOk` propagates exact non-overflowing facts!
    if matches!(
        op,
        BinaryOp::AddCheck
            | BinaryOp::SubCheck
            | BinaryOp::MulCheck
            | BinaryOp::DivCheck
            | BinaryOp::ModCheck
    ) {
        st.checked_op.insert(dst, (op, lhs, rhs, op_ty));
        return;
    }

    if !env.is_integer(dst_ty) {
        return;
    }

    let tb = ty_bounds(dst_ty, env);
    let li = st.get_interval(lk);
    let ri = st.get_interval(rk);

    match op {
        BinaryOp::Add | BinaryOp::AddWrap => {
            // Check if adding a constant to a variable (preserves exact DBM offset!).
            let is_const_r = ri.lo.is_some() && ri.lo == ri.hi;
            let is_const_l = li.lo.is_some() && li.lo == li.hi;
            let no_wrap = if op == BinaryOp::Add {
                true
            } else if let (Some(ll), Some(lh), Some(rl), Some(rh), Some((tl, th))) = (li.lo, li.hi, ri.lo, ri.hi, tb) {
                ll.checked_add(rl).is_some_and(|v| v >= tl) && lh.checked_add(rh).is_some_and(|v| v <= th)
            } else {
                false
            };
            if no_wrap && is_const_r {
                st.assign_offset(dk, lk, ri.lo.unwrap(), tb);
                st.sum_exprs.insert(dk, (lk, rk));
            } else if no_wrap && is_const_l {
                st.assign_offset(dk, rk, li.lo.unwrap(), tb);
                st.sum_exprs.insert(dk, (lk, rk));
            } else if no_wrap {
                let lo = match (li.lo, ri.lo) {
                    (Some(a), Some(b)) => a.checked_add(b),
                    _ => None,
                };
                let hi = match (li.hi, ri.hi) {
                    (Some(a), Some(b)) => a.checked_add(b),
                    _ => None,
                };
                let mut res = Interval { lo, hi };
                if let Some((tl, th)) = tb {
                    res = res.intersect(Interval::from_range(tl, th));
                }
                st.set_interval(dk, res);
                st.sum_exprs.insert(dk, (lk, rk));
                // Monotonicity: if rhs >= c (e.g. rhs >= 0), then lhs - (lhs + rhs) <= -c!
                if let Some(rl) = ri.lo
                    && let Some(neg_rl) = rl.checked_neg()
                {
                    st.add_diff(lk, dk, neg_rl);
                }
                if let Some(ll) = li.lo
                    && let Some(neg_ll) = ll.checked_neg()
                {
                    st.add_diff(rk, dk, neg_ll);
                }
                // Difference cancellation: if `rhs - diff <= c` where `diff == M - lhs`,
                // then `(lhs + rhs) - M <= c`! (Proves `tstart + tlen <= src.len` from `tlen <= src.len - tstart`).
                let diff_edges: Vec<(VarKey, i128)> = st
                    .diff_exprs
                    .iter()
                    .filter_map(|(&dkey, &(m, sub))| {
                        if sub == lk {
                            st.diff_upper(rk, dkey).map(|c| (m, c))
                        } else if sub == rk {
                            st.diff_upper(lk, dkey).map(|c| (m, c))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (m, c) in diff_edges {
                    st.add_diff(dk, m, c);
                }
            } else if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Sub | BinaryOp::SubWrap => {
            let is_const_r = ri.lo.is_some() && ri.lo == ri.hi;
            let no_wrap = if op == BinaryOp::Sub {
                true
            } else if let (Some(ll), Some(lh), Some(rl), Some(rh), Some((tl, th))) = (li.lo, li.hi, ri.lo, ri.hi, tb) {
                ll.checked_sub(rh).is_some_and(|v| v >= tl) && lh.checked_sub(rl).is_some_and(|v| v <= th)
            } else {
                false
            };
            if no_wrap && is_const_r
                && let Some(neg) = ri.lo.unwrap().checked_neg()
            {
                st.assign_offset(dk, lk, neg, tb);
                st.diff_exprs.insert(dk, (lk, rk));
            } else if no_wrap {
                // Use DBM relational difference `lhs - rhs` too!
                let hi_dbm = st.diff_upper(lk, rk);
                let lo_dbm = st.diff_upper(rk, lk).and_then(|v| v.checked_neg());
                let lo_iv = match (li.lo, ri.hi) {
                    (Some(a), Some(b)) => a.checked_sub(b),
                    _ => None,
                };
                let hi_iv = match (li.hi, ri.lo) {
                    (Some(a), Some(b)) => a.checked_sub(b),
                    _ => None,
                };
                let mut res = Interval { lo: lo_iv, hi: hi_iv }.intersect(Interval { lo: lo_dbm, hi: hi_dbm });
                if let Some((tl, th)) = tb {
                    res = res.intersect(Interval::from_range(tl, th));
                }
                st.set_interval(dk, res);
                st.diff_exprs.insert(dk, (lk, rk));
                if let Some(rl) = ri.lo
                    && let Some(neg_rl) = rl.checked_neg()
                {
                    // `rhs >= rl` => `(lhs - rhs) - lhs <= -rl`
                    st.add_diff(dk, lk, neg_rl);
                }
            } else if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Mul => {
            if let (Some(ll), Some(lh), Some(rl), Some(rh)) = (li.lo, li.hi, ri.lo, ri.hi) {
                let prods = [
                    ll.checked_mul(rl),
                    ll.checked_mul(rh),
                    lh.checked_mul(rl),
                    lh.checked_mul(rh),
                ];
                if prods.iter().all(|p| p.is_some()) {
                    let lo = prods.iter().filter_map(|p| *p).min().unwrap();
                    let hi = prods.iter().filter_map(|p| *p).max().unwrap();
                    let mut res = Interval::from_range(lo, hi);
                    if let Some((tl, th)) = tb {
                        res = res.intersect(Interval::from_range(tl, th));
                    }
                    st.set_interval(dk, res);
                    return;
                }
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Div => {
            if let (Some(ll), Some(lh), Some(rl), Some(rh)) = (li.lo, li.hi, ri.lo, ri.hi)
                && ll >= 0
                && rl >= 1
            {
                let lo = ll / rh;
                let hi = lh / rl;
                st.set_interval(dk, Interval::from_range(lo, hi));
                return;
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Mod => {
            // `x % k` with `x >= 0` and `k > 0` gives `[0, k - 1]` (and `res < rhs`!).
            if li.lo.is_some_and(|l| l >= 0) && ri.lo.is_some_and(|l| l >= 1) {
                st.add_lower(dk, 0);
                st.add_diff(dk, rk, -1);
                if let Some(rh) = ri.hi
                    && let Some(rm1) = rh.checked_sub(1)
                {
                    st.add_upper(dk, rm1);
                }
                return;
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
            let bw = int_bitwidth(env.get(op_ty));
            let nonneg = li.lo.is_some_and(|l| l >= 0) && ri.lo.is_some_and(|l| l >= 0);
            if nonneg {
                let lkb = st.known_bits.get(&lk).copied().unwrap_or_else(|| KnownBits::from_nonneg_interval(li, bw));
                let rkb = st.known_bits.get(&rk).copied().unwrap_or_else(|| KnownBits::from_nonneg_interval(ri, bw));
                let lkb_i = KnownBits::from_nonneg_interval(li, bw);
                let rkb_i = KnownBits::from_nonneg_interval(ri, bw);
                let l_eff = KnownBits { zeros: lkb.zeros | lkb_i.zeros, ones: lkb.ones | lkb_i.ones };
                let r_eff = KnownBits { zeros: rkb.zeros | rkb_i.zeros, ones: rkb.ones | rkb_i.ones };
                let res_kb = match op {
                    BinaryOp::BitAnd => l_eff.bit_and(r_eff),
                    BinaryOp::BitOr => l_eff.bit_or(r_eff),
                    _ => l_eff.bit_xor(r_eff),
                };
                st.known_bits.insert(dk, res_kb);
                let mut iv = res_kb.to_nonneg_interval(bw).unwrap_or_else(Interval::top);
                if op == BinaryOp::BitAnd {
                    if let Some(rh) = ri.hi { iv = iv.intersect(Interval::from_range(0, rh)); }
                    if let Some(lh) = li.hi { iv = iv.intersect(Interval::from_range(0, lh)); }
                }
                if let Some((tl, th)) = tb {
                    iv = iv.intersect(Interval::from_range(tl, th));
                }
                st.set_interval(dk, iv);
                return;
            }
            if op == BinaryOp::BitAnd {
                let mut hi_cand: Option<i128> = None;
                if ri.lo.is_some_and(|l| l >= 0) && let Some(rh) = ri.hi {
                    hi_cand = Some(rh);
                }
                if li.lo.is_some_and(|l| l >= 0) && let Some(lh) = li.hi {
                    hi_cand = Some(hi_cand.map_or(lh, |h| h.min(lh)));
                }
                if let Some(h) = hi_cand {
                    st.set_interval(dk, Interval::from_range(0, h));
                    return;
                }
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Shr => {
            let bw = int_bitwidth(env.get(op_ty));
            if let (Some(ll), Some(lh), Some(rl), Some(rh)) = (li.lo, li.hi, ri.lo, ri.hi)
                && rl >= 0
                && (rh as u32) < bw
            {
                let a = ll >> (rh as u32);
                let b = ll >> (rl as u32);
                let c = lh >> (rh as u32);
                let d = lh >> (rl as u32);
                let lo = a.min(b).min(c).min(d);
                let hi = a.max(b).max(c).max(d);
                if ll >= 0 && rl == rh {
                    let lkb = st.known_bits.get(&lk).copied().unwrap_or_else(|| KnownBits::from_nonneg_interval(li, bw));
                    let rkb = lkb.lshr(rl as u32);
                    st.known_bits.insert(dk, rkb);
                }
                st.set_interval(dk, Interval::from_range(lo, hi));
                return;
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        BinaryOp::Shl => {
            let bw = int_bitwidth(env.get(op_ty));
            if li.lo.is_some_and(|l| l >= 0)
                && let (Some(ll), Some(lh), Some(rl), Some(rh)) = (li.lo, li.hi, ri.lo, ri.hi)
                && rl >= 0
                && (rh as u32) < bw
                && let (Some(lo), Some(hi)) = (ll.checked_shl(rl as u32), lh.checked_shl(rh as u32))
            {
                if rl == rh {
                    let lkb = st.known_bits.get(&lk).copied().unwrap_or_else(|| KnownBits::from_nonneg_interval(li, bw));
                    st.known_bits.insert(dk, lkb.shl(rl as u32, bw));
                }
                let mut res = Interval::from_range(lo, hi);
                if let Some((tl, th)) = tb {
                    res = res.intersect(Interval::from_range(tl, th));
                }
                st.set_interval(dk, res);
                return;
            }
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
        _ => {
            if let Some((tl, th)) = tb {
                st.set_interval(dk, Interval::from_range(tl, th));
            }
        }
    }
}

fn verify_inst(
    bb: BlockId,
    ii: usize,
    inst: &EirInst,
    st: &DbmState,
    f: &EirFunction,
    contracts: &ContractMap,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
    proven_safe: &mut BTreeSet<(BlockId, usize)>,
    stats: &mut FnVerifyInfo,
) {
    match &inst.kind {
        EirInstKind::LoadPlace { place, .. }
        | EirInstKind::StorePlace { place, .. }
        | EirInstKind::PlaceAddr { place, .. } => {
            if verify_place_bounds(place, inst.span, st, f, env, interner, diags, stats) {
                proven_safe.insert((bb, ii));
            }
        }
        EirInstKind::Slice { base, base_ty, lo, hi, .. } => {
            stats.total_checks += 1;
            let len_k = VarKey::LenVal(*base);
            let mut tmp = st.clone();
            init_len_for_key(&mut tmp, len_k, *base_ty, env);
            let lo_ok = lo.is_none_or(|v| tmp.get_interval(VarKey::Val(v)).lo.is_some_and(|l| l >= 0));
            let hi_ok = hi.is_none_or(|v| bound_le(tmp.diff_upper(VarKey::Val(v), len_k), Some(0)));
            let order_ok = match (lo, hi) {
                (Some(lv), Some(hv)) => bound_le(tmp.diff_upper(VarKey::Val(*lv), VarKey::Val(*hv)), Some(0)),
                (Some(lv), None) => bound_le(tmp.diff_upper(VarKey::Val(*lv), len_k), Some(0)),
                (None, Some(hv)) => tmp.get_interval(VarKey::Val(*hv)).lo.is_some_and(|l| l >= 0),
                (None, None) => true,
            };
            if lo_ok && hi_ok && order_ok {
                stats.proven_checks += 1;
                proven_safe.insert((bb, ii));
            } else {
                let mut d = Diagnostic::error(
                    ErrorCode::E0107,
                    inst.span,
                    "ZM-07: cannot prove slice bounds `0 <= lo <= hi <= len`",
                );
                d.notes.push(format!(
                    "known: len in {}, lo in {}, hi in {}",
                    tmp.get_interval(len_k).fmt_str(),
                    lo.map_or("[0..=0]".into(), |v| tmp.get_interval(VarKey::Val(v)).fmt_str()),
                    hi.map_or("len".into(), |v| tmp.get_interval(VarKey::Val(v)).fmt_str()),
                ));
                d.helps.push("add a bounds check, strengthen `where`, or use `try buf.sub(lo, hi)`".into());
                diags.emit(d);
            }
        }
        EirInstKind::Unary { op: UnaryOp::Neg, operand, ty, .. } => {
            if env.is_signed_integer(*ty)
                && let Some((tl, _)) = ty_bounds(*ty, env)
            {
                stats.total_checks += 1;
                let iv = st.get_interval(VarKey::Val(*operand));
                if iv.lo.is_some_and(|l| l > tl) {
                    stats.proven_checks += 1;
                    proven_safe.insert((bb, ii));
                } else {
                    let mut d = Diagnostic::error(
                        ErrorCode::E0108,
                        inst.span,
                        format!(
                            "ZM-08: cannot prove unary `-` does not overflow `{}` (operand could be MIN)",
                            env.format_ty(*ty, interner)
                        ),
                    );
                    d.notes.push(format!(
                        "known: {} in {}",
                        fmt_val_name(*operand, f, interner),
                        iv.fmt_str()
                    ));
                    diags.emit(d);
                }
            }
        }
        EirInstKind::Binary { op, lhs, rhs, op_ty, .. } => {
            if !env.is_integer(*op_ty) {
                return;
            }
            if !matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod
                    | BinaryOp::Shl
                    | BinaryOp::Shr
            ) {
                return;
            }
            stats.total_checks += 1;
            if verify_binary_arith(*op, *lhs, *rhs, *op_ty, inst.span, st, f, env, interner, diags) {
                stats.proven_checks += 1;
                proven_safe.insert((bb, ii));
            }
        }
        EirInstKind::Call { callee, args, .. } => {
            let callee_key = match callee {
                EirCallee::DirectFn(d) => Some(CfgFnKey::TopDef(*d)),
                EirCallee::Method { impl_idx, sub_idx } => Some(CfgFnKey::ImplMethod {
                    impl_idx: *impl_idx,
                    sub_idx: *sub_idx,
                }),
                EirCallee::GenericInst { inst_id } => Some(CfgFnKey::GenericInst(*inst_id)),
                _ => None,
            };
            if let Some(ck) = callee_key
                && let Some(fc) = contracts.get(&ck)
                && !fc.where_clauses.is_empty()
            {
                let mut call_env: BTreeMap<Symbol, (VarKey, Option<VarKey>)> = BTreeMap::new();
                for (pn, &av) in fc.param_names.iter().zip(args.iter()) {
                    call_env.insert(*pn, (VarKey::Val(av), Some(VarKey::LenVal(av))));
                }
                for w in &fc.where_clauses {
                    stats.total_checks += 1;
                    if check_contract_expr(st, w, &call_env, None, interner) {
                        stats.proven_checks += 1;
                    } else {
                        let mut d = Diagnostic::error(
                            ErrorCode::E0107,
                            inst.span,
                            "cannot prove callee `where` precondition at call site",
                        );
                        d.notes.push(format_state_summary(st, f, interner));
                        d.helps.push("add a runtime check or propagate the precondition to the caller's `where` clause".into());
                        diags.emit(d);
                    }
                }
            }
        }
        _ => {}
    }
}

fn verify_place_bounds(
    place: &EirPlace,
    span: Span,
    st: &DbmState,
    f: &EirFunction,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
    stats: &mut FnVerifyInfo,
) -> bool {
    let mut all_ok = true;
    let mut cur_ty = match place.base {
        EirPlaceBase::Slot(s) => f.slots[s.0 as usize].ty,
        EirPlaceBase::Val(v) => f.value_tys[v.0 as usize],
    };
    let mut base_len_key = match place.base {
        EirPlaceBase::Slot(s) => Some(VarKey::LenSlot(s)),
        EirPlaceBase::Val(v) => Some(VarKey::LenVal(v)),
    };
    let mut prev_was_pool_col = false;

    for (pi, pr) in place.projs.iter().enumerate() {
        match pr {
            EirProj::Deref => {
                cur_ty = match env.get(cur_ty) {
                    Ty::Borrow { inner: BorrowInner::Single(t), .. }
                    | Ty::RawPtr { inner: t, .. } => *t,
                    _ => cur_ty,
                };
                prev_was_pool_col = false;
            }
            EirProj::Field { field_idx, ty, .. } => {
                let is_root_field = pi == 0 || (pi == 1 && matches!(place.projs[0], EirProj::Deref));
                let deref_cur = match env.get(cur_ty) {
                    Ty::Borrow { inner: BorrowInner::Single(t), .. } | Ty::RawPtr { inner: t, .. } => *t,
                    _ => cur_ty,
                };
                base_len_key = if is_root_field && !matches!(env.get(deref_cur), Ty::Pool { .. }) {
                    match place.base {
                        EirPlaceBase::Slot(s) => Some(VarKey::FieldLenSlot(s, *field_idx as u32)),
                        EirPlaceBase::Val(v) => {
                            if let Some(&os) = st.val_slot_origin.get(&v) {
                                Some(VarKey::FieldLenSlot(os, *field_idx as u32))
                            } else {
                                Some(VarKey::FieldLenVal(v, *field_idx as u32))
                            }
                        }
                    }
                } else {
                    None
                };
                cur_ty = *ty;
                prev_was_pool_col = false;
            }
            EirProj::TupleField { index, ty } => {
                let is_root_field = pi == 0 || (pi == 1 && matches!(place.projs[0], EirProj::Deref));
                base_len_key = if is_root_field {
                    match place.base {
                        EirPlaceBase::Slot(s) => Some(VarKey::FieldLenSlot(s, *index as u32)),
                        EirPlaceBase::Val(v) => {
                            if let Some(&os) = st.val_slot_origin.get(&v) {
                                Some(VarKey::FieldLenSlot(os, *index as u32))
                            } else {
                                Some(VarKey::FieldLenVal(v, *index as u32))
                            }
                        }
                    }
                } else {
                    None
                };
                cur_ty = *ty;
                prev_was_pool_col = false;
            }
            EirProj::PoolCol { elem_ty, .. } => {
                cur_ty = *elem_ty;
                prev_was_pool_col = true;
            }
            EirProj::Index { index, ty } => {
                let idx_ty = f.value_tys[index.0 as usize];
                let is_pool_id = matches!(
                    env.get(idx_ty),
                    Ty::Builtin(BuiltinKind::IdBrand | BuiltinKind::GenId | BuiltinKind::Link, _)
                );
                if !(prev_was_pool_col && is_pool_id) {
                    let idx_k = VarKey::Val(*index);
                    let idx_iv = st.get_interval(idx_k);
                    let non_neg = idx_iv.lo.is_some_and(|l| l >= 0);

                    let deref_cur = match env.get(cur_ty) {
                        Ty::Borrow { inner: BorrowInner::Single(t), .. } => *t,
                        _ => cur_ty,
                    };
                    let mut proven = false;
                    let mut fact_str = format!(
                        "{} in {}",
                        fmt_val_name(*index, f, interner),
                        idx_iv.fmt_str()
                    );
                    if matches!(env.get(deref_cur), Ty::Builtin(BuiltinKind::GpuBuf, _)) {
                        proven = true;
                    } else if let Ty::Array { len, .. } = env.get(deref_cur) {
                        if non_neg && idx_iv.hi.is_some_and(|h| h < *len as i128) {
                            proven = true;
                        } else {
                            fact_str.push_str(&format!(", array len = {len}"));
                        }
                    } else if let Ty::Vec { len, .. } = env.get(deref_cur) {
                        if non_neg && idx_iv.hi.is_some_and(|h| h < *len as i128) {
                            proven = true;
                        } else {
                            fact_str.push_str(&format!(", vec len = {len}"));
                        }
                    } else if let Some(lk) = base_len_key {
                        let diff = st.diff_upper(idx_k, lk);
                        if non_neg && bound_le(diff, Some(-1)) {
                            proven = true;
                        } else if let Some(d) = diff {
                            fact_str.push_str(&format!(", {} - len <= {d}", fmt_val_name(*index, f, interner)));
                        } else {
                            fact_str.push_str(&format!(", len in {}", st.get_interval(lk).fmt_str()));
                        }
                    }
                    stats.total_checks += 1;
                    if proven {
                        stats.proven_checks += 1;
                    } else {
                        all_ok = false;
                        let mut d = Diagnostic::error(
                            ErrorCode::E0107,
                            span,
                            format!(
                                "ZM-07: cannot prove index bounds `0 <= {} < len`",
                                fmt_val_name(*index, f, interner)
                            ),
                        );
                        d.notes.push(format!("known: {fact_str}"));
                        d.helps.push(
                            "add a check `if i < buf.len { ... }`, strengthen `where`, or use `try buf.at(i)`".into(),
                        );
                        diags.emit(d);
                    }
                }
                cur_ty = *ty;
                base_len_key = None;
                prev_was_pool_col = false;
            }
        }
    }
    all_ok
}

fn verify_binary_arith(
    op: BinaryOp,
    lhs: EirValue,
    rhs: EirValue,
    op_ty: TyId,
    span: Span,
    st: &DbmState,
    f: &EirFunction,
    env: &TypeEnv,
    interner: &Interner,
    diags: &mut DiagBag,
) -> bool {
    let Some((tl, th)) = ty_bounds(op_ty, env) else {
        return true;
    };
    let lk = VarKey::Val(lhs);
    let rk = VarKey::Val(rhs);
    let li = st.get_interval(lk);
    let ri = st.get_interval(rk);
    let ty_name = env.format_ty(op_ty, interner);

    match op {
        BinaryOp::Add => {
            let mut ok = match (li.lo, li.hi, ri.lo, ri.hi) {
                (Some(ll), Some(lh), Some(rl), Some(rh)) => {
                    ll.checked_add(rl).is_some_and(|v| v >= tl)
                        && lh.checked_add(rh).is_some_and(|v| v <= th)
                }
                _ => false,
            };
            // Difference cancellation: if `lhs + rhs <= M + c` because `rhs <= (M - lhs) + c`
            // (e.g. `tlen <= src.len - tstart` => `tstart + tlen <= src.len <= isize::MAX`),
            // and lower bound is safe, then `+` cannot overflow!
            if !ok && li.lo.zip(ri.lo).is_some_and(|(a, b)| a.checked_add(b).is_some_and(|s| s >= tl)) {
                for (&dkey, &(m, sub)) in &st.diff_exprs {
                    let other = if sub == lk {
                        Some(rk)
                    } else if sub == rk {
                        Some(lk)
                    } else {
                        None
                    };
                    if let Some(ok_var) = other
                        && let Some(c) = st.diff_upper(ok_var, dkey)
                        && let Some(mhi) = st.get_interval(m).hi
                        && mhi.checked_add(c).is_some_and(|hi| hi <= th)
                    {
                        ok = true;
                        break;
                    }
                }
            }
            if ok {
                return true;
            }
            let mut d = Diagnostic::error(
                ErrorCode::E0108,
                span,
                format!("ZM-08: cannot prove `+` does not overflow `{ty_name}`"),
            );
            d.notes.push(format!(
                "known: {} in {}, {} in {}",
                fmt_val_name(lhs, f, interner),
                li.fmt_str(),
                fmt_val_name(rhs, f, interner),
                ri.fmt_str()
            ));
            d.helps.push("use checked `+?`, wrapping `+%`, or saturating `+|`, or constrain operands via `if`/`where`".into());
            diags.emit(d);
            false
        }
        BinaryOp::Sub => {
            let dbm_hi = st.diff_upper(lk, rk);
            let dbm_lo = st.diff_upper(rk, lk).and_then(|v| v.checked_neg());
            let iv_lo = match (li.lo, ri.hi) {
                (Some(a), Some(b)) => a.checked_sub(b),
                _ => None,
            };
            let iv_hi = match (li.hi, ri.lo) {
                (Some(a), Some(b)) => a.checked_sub(b),
                _ => None,
            };
            let eff_lo = match (iv_lo, dbm_lo) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            let eff_hi = min_bound(iv_hi, dbm_hi);
            if eff_lo.is_some_and(|l| l >= tl) && eff_hi.is_some_and(|h| h <= th) {
                return true;
            }
            let mut d = Diagnostic::error(
                ErrorCode::E0108,
                span,
                format!("ZM-08: cannot prove `-` does not overflow/underflow `{ty_name}`"),
            );
            d.notes.push(format!(
                "known: {} in {}, {} in {}",
                fmt_val_name(lhs, f, interner),
                li.fmt_str(),
                fmt_val_name(rhs, f, interner),
                ri.fmt_str()
            ));
            d.helps.push("use checked `-?`, wrapping `-%`, or saturating `-|`, or prove `lhs >= rhs`".into());
            diags.emit(d);
            false
        }
        BinaryOp::Mul => {
            let ok = if let (Some(ll), Some(lh), Some(rl), Some(rh)) = (li.lo, li.hi, ri.lo, ri.hi) {
                [
                    ll.checked_mul(rl),
                    ll.checked_mul(rh),
                    lh.checked_mul(rl),
                    lh.checked_mul(rh),
                ]
                .iter()
                .all(|p| p.is_some_and(|v| v >= tl && v <= th))
            } else {
                false
            };
            if ok {
                return true;
            }
            let mut d = Diagnostic::error(
                ErrorCode::E0108,
                span,
                format!("ZM-08: cannot prove `*` does not overflow `{ty_name}`"),
            );
            d.notes.push(format!(
                "known: {} in {}, {} in {}",
                fmt_val_name(lhs, f, interner),
                li.fmt_str(),
                fmt_val_name(rhs, f, interner),
                ri.fmt_str()
            ));
            d.helps.push("use checked `*?`, wrapping `*%`, or saturating `*|`".into());
            diags.emit(d);
            false
        }
        BinaryOp::Div | BinaryOp::Mod => {
            let nonzero = st.is_non_zero(rk);
            let min_div_neg1_possible = env.is_signed_integer(op_ty)
                && li.lo.is_none_or(|l| l <= tl)
                && ri.lo.is_none_or(|l| l <= -1)
                && ri.hi.is_none_or(|h| h >= -1);
            if nonzero && !min_div_neg1_possible {
                return true;
            }
            let msg = if !nonzero {
                format!("ZM-08: cannot prove divisor is non-zero for `{ty_name}`")
            } else {
                format!("ZM-08: cannot prove signed division does not overflow `{ty_name}.MIN / -1`")
            };
            let mut d = Diagnostic::error(ErrorCode::E0108, span, msg);
            d.notes.push(format!(
                "known: {} in {}, {} in {}",
                fmt_val_name(lhs, f, interner),
                li.fmt_str(),
                fmt_val_name(rhs, f, interner),
                ri.fmt_str()
            ));
            d.helps.push("add a check `if d == 0 { return .DivByZero; }` or use `/?` / `%?`".into());
            diags.emit(d);
            false
        }
        BinaryOp::Shl | BinaryOp::Shr => {
            let bw = int_bitwidth(env.get(op_ty)) as i128;
            if ri.lo.is_some_and(|l| l >= 0) && ri.hi.is_some_and(|h| h < bw) {
                return true;
            }
            let mut d = Diagnostic::error(
                ErrorCode::E0108,
                span,
                format!("ZM-08: cannot prove shift count is in `0..{bw}` for `{ty_name}`"),
            );
            d.notes.push(format!(
                "known: shift {} in {}",
                fmt_val_name(rhs, f, interner),
                ri.fmt_str()
            ));
            d.helps.push("mask the shift count, constrain it with `if`/`where`, or use `<<%`".into());
            diags.emit(d);
            false
        }
        _ => true,
    }
}

fn fmt_val_name(v: EirValue, f: &EirFunction, interner: &Interner) -> String {
    if let Some(&sym) = f.val_names.get(&v) {
        interner.get(sym).to_owned()
    } else {
        format!("%v{}", v.0)
    }
}

fn format_state_summary(st: &DbmState, f: &EirFunction, interner: &Interner) -> String {
    let mut parts = Vec::new();
    for (pi, &pv) in f.params.iter().enumerate() {
        let iv = st.get_interval(VarKey::Val(pv));
        if iv.lo.is_some() || iv.hi.is_some() {
            let name = f
                .param_names
                .get(pi)
                .map_or_else(|| format!("p{pi}"), |&s| interner.get(s).to_owned());
            parts.push(format!("{name} in {}", iv.fmt_str()));
        }
    }
    if parts.is_empty() {
        "known: state is reachable (non-contradictory)".into()
    } else {
        format!("known: {}", parts.join(", "))
    }
}

fn build_param_env(f: &EirFunction) -> BTreeMap<Symbol, (VarKey, Option<VarKey>)> {
    let mut m = BTreeMap::new();
    for (pi, &pv) in f.params.iter().enumerate() {
        if let Some(&sym) = f.param_names.get(pi) {
            m.insert(sym, (VarKey::Val(pv), Some(VarKey::LenVal(pv))));
        }
    }
    m
}

fn build_block_sym_env(
    f: &EirFunction,
    _st: &DbmState,
) -> BTreeMap<Symbol, (VarKey, Option<VarKey>)> {
    let mut m = build_param_env(f);
    for s in &f.slots {
        if let Some(nm) = s.name {
            m.insert(nm, (VarKey::Slot(s.id), Some(VarKey::LenSlot(s.id))));
        }
    }
    for (&v, &nm) in &f.val_names {
        m.entry(nm).or_insert((VarKey::Val(v), Some(VarKey::LenVal(v))));
    }
    m
}

#[derive(Clone, Debug)]
enum ContractTerm {
    Const(i128),
    Var(VarKey),
    VarPlusConst(VarKey, i128),
}

fn resolve_contract_term(
    expr: &Expr,
    sym_env: &BTreeMap<Symbol, (VarKey, Option<VarKey>)>,
    ret_key: Option<VarKey>,
    interner: &Interner,
) -> Option<ContractTerm> {
    match &expr.kind {
        ExprKind::Paren(inner) => resolve_contract_term(inner, sym_env, ret_key, interner),
        ExprKind::IntLit { raw, .. } => {
            parse_int_literal(interner.get(*raw)).and_then(|u| i128::try_from(u).ok()).map(ContractTerm::Const)
        }
        ExprKind::Unary { op: UnaryOp::Neg, expr: inner } => {
            if let Some(ContractTerm::Const(c)) = resolve_contract_term(inner, sym_env, ret_key, interner) {
                return c.checked_neg().map(ContractTerm::Const);
            }
            None
        }
        ExprKind::ReturnVal => ret_key.map(ContractTerm::Var),
        ExprKind::Ident(sym) => sym_env.get(sym).map(|&(vk, _)| ContractTerm::Var(vk)),
        ExprKind::Field { base, field } if interner.get(field.sym) == "len" => {
            if let ExprKind::Ident(bsym) = &base.kind
                && let Some(&(_, Some(lk))) = sym_env.get(bsym)
            {
                return Some(ContractTerm::Var(lk));
            }
            None
        }
        ExprKind::Call { callee, args } if args.is_empty() => {
            if let ExprKind::Field { base, field } = &callee.kind
                && interner.get(field.sym) == "len"
                && let ExprKind::Ident(bsym) = &base.kind
                && let Some(&(_, Some(lk))) = sym_env.get(bsym)
            {
                return Some(ContractTerm::Var(lk));
            }
            None
        }
        ExprKind::Binary { op: BinaryOp::Add | BinaryOp::AddWrap, lhs, rhs } => {
            let lt = resolve_contract_term(lhs, sym_env, ret_key, interner)?;
            let rt = resolve_contract_term(rhs, sym_env, ret_key, interner)?;
            match (lt, rt) {
                (ContractTerm::Const(a), ContractTerm::Const(b)) => a.checked_add(b).map(ContractTerm::Const),
                (ContractTerm::Var(v), ContractTerm::Const(c))
                | (ContractTerm::Const(c), ContractTerm::Var(v)) => Some(ContractTerm::VarPlusConst(v, c)),
                (ContractTerm::VarPlusConst(v, c1), ContractTerm::Const(c2)) => {
                    c1.checked_add(c2).map(|c| ContractTerm::VarPlusConst(v, c))
                }
                _ => None,
            }
        }
        ExprKind::Binary { op: BinaryOp::Sub | BinaryOp::SubWrap, lhs, rhs } => {
            let lt = resolve_contract_term(lhs, sym_env, ret_key, interner)?;
            let rt = resolve_contract_term(rhs, sym_env, ret_key, interner)?;
            match (lt, rt) {
                (ContractTerm::Const(a), ContractTerm::Const(b)) => a.checked_sub(b).map(ContractTerm::Const),
                (ContractTerm::Var(v), ContractTerm::Const(c)) => c.checked_neg().map(|nc| ContractTerm::VarPlusConst(v, nc)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn assume_contract_expr(
    st: &mut DbmState,
    expr: &Expr,
    sym_env: &BTreeMap<Symbol, (VarKey, Option<VarKey>)>,
    ret_key: Option<VarKey>,
    interner: &Interner,
) {
    match &expr.kind {
        ExprKind::Paren(inner) => assume_contract_expr(st, inner, sym_env, ret_key, interner),
        ExprKind::Binary { op: BinaryOp::And, lhs, rhs } => {
            assume_contract_expr(st, lhs, sym_env, ret_key, interner);
            assume_contract_expr(st, rhs, sym_env, ret_key, interner);
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let cmp = match op {
                BinaryOp::Lt => CmpOp::Lt,
                BinaryOp::LtEq => CmpOp::LtEq,
                BinaryOp::Gt => CmpOp::Gt,
                BinaryOp::GtEq => CmpOp::GtEq,
                BinaryOp::Eq => CmpOp::Eq,
                BinaryOp::NotEq => CmpOp::NotEq,
                _ => return,
            };
            let Some(lt) = resolve_contract_term(lhs, sym_env, ret_key, interner) else { return };
            let Some(rt) = resolve_contract_term(rhs, sym_env, ret_key, interner) else { return };
            assume_term_cmp(st, cmp, lt, rt);
        }
        _ => {}
    }
}

fn assume_term_cmp(st: &mut DbmState, cmp: CmpOp, lt: ContractTerm, rt: ContractTerm) {
    match (lt, rt) {
        (ContractTerm::Var(u), ContractTerm::Var(v)) => st.refine_cmp(cmp, u, v),
        (ContractTerm::Var(u), ContractTerm::Const(c)) => match cmp {
            CmpOp::Lt => { if let Some(cm1) = c.checked_sub(1) { st.add_upper(u, cm1); } }
            CmpOp::LtEq => st.add_upper(u, c),
            CmpOp::Gt => { if let Some(c1) = c.checked_add(1) { st.add_lower(u, c1); } }
            CmpOp::GtEq => st.add_lower(u, c),
            CmpOp::Eq => st.add_const(u, c),
            CmpOp::NotEq => {
                if c == 0 {
                    st.non_zero.insert(u);
                }
            }
        },
        (ContractTerm::Const(c), ContractTerm::Var(v)) => {
            let flipped = match cmp {
                CmpOp::Lt => CmpOp::Gt,
                CmpOp::LtEq => CmpOp::GtEq,
                CmpOp::Gt => CmpOp::Lt,
                CmpOp::GtEq => CmpOp::LtEq,
                CmpOp::Eq => CmpOp::Eq,
                CmpOp::NotEq => CmpOp::NotEq,
            };
            assume_term_cmp(st, flipped, ContractTerm::Var(v), ContractTerm::Const(c));
        }
        (ContractTerm::VarPlusConst(u, c), ContractTerm::Var(v)) => {
            // u + c <= v  <=>  u - v <= -c
            match cmp {
                CmpOp::LtEq => { if let Some(nc) = c.checked_neg() { st.add_diff(u, v, nc); } }
                CmpOp::Lt => { if let Some(nc) = c.checked_neg().and_then(|x| x.checked_sub(1)) { st.add_diff(u, v, nc); } }
                CmpOp::GtEq => st.add_diff(v, u, c),
                CmpOp::Gt => { if let Some(cm1) = c.checked_sub(1) { st.add_diff(v, u, cm1); } }
                CmpOp::Eq => {
                    if let Some(nc) = c.checked_neg() { st.add_diff(u, v, nc); }
                    st.add_diff(v, u, c);
                }
                CmpOp::NotEq => {}
            }
        }
        (ContractTerm::Var(u), ContractTerm::VarPlusConst(v, c)) => {
            // u <= v + c  <=>  u - v <= c
            match cmp {
                CmpOp::LtEq => st.add_diff(u, v, c),
                CmpOp::Lt => { if let Some(cm1) = c.checked_sub(1) { st.add_diff(u, v, cm1); } }
                CmpOp::GtEq => { if let Some(nc) = c.checked_neg() { st.add_diff(v, u, nc); } }
                CmpOp::Gt => { if let Some(nc) = c.checked_neg().and_then(|x| x.checked_sub(1)) { st.add_diff(v, u, nc); } }
                CmpOp::Eq => {
                    st.add_diff(u, v, c);
                    if let Some(nc) = c.checked_neg() { st.add_diff(v, u, nc); }
                }
                CmpOp::NotEq => {}
            }
        }
        _ => {}
    }
}

fn check_contract_expr(
    st: &DbmState,
    expr: &Expr,
    sym_env: &BTreeMap<Symbol, (VarKey, Option<VarKey>)>,
    ret_key: Option<VarKey>,
    interner: &Interner,
) -> bool {
    match &expr.kind {
        ExprKind::Paren(inner) => check_contract_expr(st, inner, sym_env, ret_key, interner),
        ExprKind::Binary { op: BinaryOp::And, lhs, rhs } => {
            check_contract_expr(st, lhs, sym_env, ret_key, interner)
                && check_contract_expr(st, rhs, sym_env, ret_key, interner)
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let cmp = match op {
                BinaryOp::Lt => CmpOp::Lt,
                BinaryOp::LtEq => CmpOp::LtEq,
                BinaryOp::Gt => CmpOp::Gt,
                BinaryOp::GtEq => CmpOp::GtEq,
                BinaryOp::Eq => CmpOp::Eq,
                BinaryOp::NotEq => CmpOp::NotEq,
                _ => return true,
            };
            let Some(lt) = resolve_contract_term(lhs, sym_env, ret_key, interner) else {
                return true;
            };
            let Some(rt) = resolve_contract_term(rhs, sym_env, ret_key, interner) else {
                return true;
            };
            // Prove `lt <cmp> rt` by refutation: check whether `st ∧ ¬(lt <cmp> rt)` is bottom!
            let mut neg_st = st.clone();
            assume_term_cmp(&mut neg_st, cmp.negate(), lt, rt);
            neg_st.is_bottom
        }
        _ => true,
    }
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

fn inst_def_val(k: &EirInstKind) -> Option<EirValue> {
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
        | EirInstKind::FieldExtract { dst, .. } => *dst,
        EirInstKind::Call { dst, .. } | EirInstKind::BuiltinCall { dst, .. } => return *dst,
        _ => return None,
    })
}
