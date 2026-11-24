use eonc::check_files_in_mem;
use eonc::diag::DiagBag;
use eonc::resolve::DefId;
use eonc::span::{Interner, SourceMap};
use eonc::types::{Ty, TyId};

fn check_single(src: &str) -> (eonc::CheckResult, SourceMap, Interner, DiagBag) {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let files = vec![("test.eon".to_owned(), src.as_bytes().to_vec())];
    let res = check_files_in_mem(&files, &mut sm, &mut interner, &mut diags);
    (res, sm, interner, diags)
}

fn find_struct_def(res: &eonc::CheckResult, interner: &Interner, name: &str) -> DefId {
    res.resolved
        .defs
        .iter()
        .find(|d| interner.get(d.name) == name)
        .expect("struct def should exist")
        .id
}

#[test]
fn layout_eon_reorders_fields_optimally() {
    let src = r#"
module test.layout;

pub struct EonOptimal {
    pub a: u8;
    pub b: u64;
    pub c: u8;
}

@layout(c)
pub struct CAbi {
    pub a: u8;
    pub b: u64;
    pub c: u8;
}

@packed
pub struct PackedAbi {
    pub tag: u8;
    pub addr: u64;
}

@layout(std430)
pub struct GpuBuf {
    pub a: u8;
    pub b: u64;
}

@layout(std430)
pub struct GpuVec4 {
    pub tag: u8;
    pub pos: f32x4;
}
"#;
    let (mut res, sm, interner, diags) = check_single(src);
    assert!(!diags.has_errors(), "unexpected errors:\n{}", diags.render_all(&sm));

    let eon_id = find_struct_def(&res, &interner, "EonOptimal");
    let eon_meta = &res.env.structs[&eon_id];
    assert_eq!(eon_meta.layout.size, 16);
    assert_eq!(eon_meta.layout.align, 8);

    let c_id = find_struct_def(&res, &interner, "CAbi");
    let c_meta = &res.env.structs[&c_id];
    assert_eq!(c_meta.layout.size, 24);
    assert_eq!(c_meta.layout.align, 8);
    assert_eq!(c_meta.fields[0].offset, 0);
    assert_eq!(c_meta.fields[1].offset, 8);
    assert_eq!(c_meta.fields[2].offset, 16);

    let p_id = find_struct_def(&res, &interner, "PackedAbi");
    let p_meta = &res.env.structs[&p_id];
    assert_eq!(p_meta.layout.size, 9);
    assert_eq!(p_meta.layout.align, 1);
    assert!(!p_meta.fields[0].misaligned_in_packed);
    assert!(p_meta.fields[1].misaligned_in_packed);

    let g_id = find_struct_def(&res, &interner, "GpuBuf");
    let g_meta = &res.env.structs[&g_id];
    assert_eq!(g_meta.layout.align, 8);
    assert_eq!(g_meta.layout.size, 16);

    let gv_id = find_struct_def(&res, &interner, "GpuVec4");
    let gv_meta = &res.env.structs[&gv_id];
    assert_eq!(gv_meta.layout.align, 16);
    assert_eq!(gv_meta.fields[1].offset, 16);
    assert_eq!(gv_meta.layout.size, 32);

    let _ = &mut res.env;
}

#[test]
fn niche_optimizations_and_trivial_copy() {
    let src = r#"
module test.niches;

pub enum State: u8 {
    Idle = 0,
    Running = 1,
    Done = 2,
}

pub struct WithNiche {
    pub active: bool;
    pub count: u32;
}

pub struct Small16 {
    pub lo: u64;
    pub hi: u64;
}

pub struct Big24 {
    pub a: u64;
    pub b: u64;
    pub c: u8;
}

pub linear struct LinearSmall {
    pub fd: i32;
}

pub fn ping(n3: u3) -> usize {
    let _ = n3;
    return @size_of(bool);
}
"#;
    let (mut res, sm, interner, diags) = check_single(src);
    assert!(!diags.has_errors(), "unexpected errors:\n{}", diags.render_all(&sm));

    // `?bool` == 1 byte, `?char` == 4 bytes, `?u32` == 8 bytes
    let opt_bool = res.env.intern(Ty::Optional(TyId::BOOL));
    assert_eq!(res.env.layout(opt_bool).size, 1);

    let opt_char = res.env.intern(Ty::Optional(TyId::CHAR));
    assert_eq!(res.env.layout(opt_char).size, 4);

    let opt_u32 = res.env.intern(Ty::Optional(TyId::U32));
    assert_eq!(res.env.layout(opt_u32).size, 8);

    // `u3` has niche (only 8 of 256 bit patterns valid) -> `?u3` == 1 byte
    let u3_ty = res.env.intern(Ty::Int {
        signed: false,
        bits: 3,
        is_ptr_sized: false,
    });
    let opt_u3 = res.env.intern(Ty::Optional(u3_ty));
    assert_eq!(res.env.layout(opt_u3).size, 1);

    // `?*u64` == 8 bytes
    let ptr_u64 = res.env.intern(Ty::Borrow {
        is_mut: false,
        inner: eonc::types::BorrowInner::Single(TyId::U64),
    });
    let opt_ptr = res.env.intern(Ty::Optional(ptr_u64));
    assert_eq!(res.env.layout(opt_ptr).size, 8);

    // `?State` == 1 byte
    let state_id = find_struct_def(&res, &interner, "State");
    let state_ty = res.env.intern(Ty::Enum(state_id));
    let opt_state = res.env.intern(Ty::Optional(state_ty));
    assert_eq!(res.env.layout(opt_state).size, 1);

    // `?WithNiche` == 8 bytes (propagates niche from `bool` field)
    let wn_id = find_struct_def(&res, &interner, "WithNiche");
    let wn_ty = res.env.intern(Ty::Struct {
        def: wn_id,
        args: Vec::new(),
    });
    assert!(res.env.layout(wn_ty).has_niche);
    let opt_wn = res.env.intern(Ty::Optional(wn_ty));
    assert_eq!(res.env.layout(opt_wn).size, res.env.layout(wn_ty).size);

    // TrivialCopy threshold <= 16B & non-linear
    let s16 = res.env.intern(Ty::Struct {
        def: find_struct_def(&res, &interner, "Small16"),
        args: Vec::new(),
    });
    let b24 = res.env.intern(Ty::Struct {
        def: find_struct_def(&res, &interner, "Big24"),
        args: Vec::new(),
    });
    let lin = res.env.intern(Ty::Struct {
        def: find_struct_def(&res, &interner, "LinearSmall"),
        args: Vec::new(),
    });
    assert!(res.env.is_trivial_copy(s16));
    assert!(!res.env.is_trivial_copy(b24));
    assert!(!res.env.is_trivial_copy(lin));
}
