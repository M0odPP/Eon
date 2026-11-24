use eonc::check_files_in_mem;
use eonc::diag::{DiagBag, ErrorCode};
use eonc::span::{Interner, SourceMap};

fn check_single(src: &str) -> (SourceMap, DiagBag) {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let files = vec![("phase2_test.eon".to_owned(), src.as_bytes().to_vec())];
    let _ = check_files_in_mem(&files, &mut sm, &mut interner, &mut diags);
    (sm, diags)
}

#[test]
fn nll_releases_arena_borrow_after_last_use() {
    let src = r#"
module test.nll_ok;

pub pool Node {
    pub v: u32;
}

pub fn reset_after_last_use(arena: *mut Arena) -> usize ! MemError {
    let nodes = try ArenaPool.[Node].init(arena, 64_usize);
    let n = nodes.len();
    // Last use of `nodes` is above -> borrow is released, so `arena.reset()` succeeds!
    arena.reset();
    return n;
}
"#;
    let (sm, diags) = check_single(src);
    assert!(!diags.has_errors(), "unexpected errors:\n{}", diags.render_all(&sm));
}

#[test]
fn view_fn_requires_unambiguous_source() {
    let src = r#"
module test.view_ambig;

pub view fn ambiguous(a: *u32, b: *u32) -> *u32 {
    let _ = b;
    return a;
}
"#;
    let (_sm, diags) = check_single(src);
    assert!(diags.diagnostics().iter().any(|d| d.code == ErrorCode::E0310));
}

#[test]
fn linear_if_early_return_is_sound() {
    let src = r#"
module test.linear_branch_ret;

pub linear struct Page {
    pub addr: u64;
}

fn free(p: Page) {
    let Page.{ addr: _ } = p;
}

pub fn branch_ret(p: Page, early: bool) -> u64 {
    if early {
        let val = p.addr;
        free(p);
        return val;
    }
    let val2 = p.addr;
    free(p);
    return val2;
}
"#;
    let (sm, diags) = check_single(src);
    assert!(!diags.has_errors(), "unexpected errors:\n{}", diags.render_all(&sm));
}

#[test]
fn slot_pool_and_gen_id_roundtrip_passes() {
    let src = r#"
module test.slot_pool;

pub pool Entity {
    pub hp: u32;
}

pub fn spawn_and_remove(alloc: *mut AllocCap) -> u32 ! OutOfMemory | StaleId {
    var entities = try SlotPool.[Entity].init(alloc, 16_usize);
    errdefer entities.destroy(alloc);

    let id = try entities.insert(alloc, .{ hp: 100_u32 });
    if let r = entities.get_mut(id) {
        r.hp = 80_u32;
    }
    let removed = try entities.remove(id);
    entities.destroy(alloc);
    return removed.hp;
}
"#;
    let (sm, diags) = check_single(src);
    assert!(!diags.has_errors(), "unexpected errors:\n{}", diags.render_all(&sm));
}
