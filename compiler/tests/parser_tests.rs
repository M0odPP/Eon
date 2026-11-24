use std::path::Path;

use eonc::diag::{DiagBag, ErrorCode};
use eonc::lexer::{TokenKind, lex_file};
use eonc::parser::parse_file;
use eonc::span::{FileProfile, Interner, SourceMap};
use eonc::ui_test::run_ui_dir;

#[test]
fn ui_suite_all_pass() {
    let ui_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/ui");
    let summary = run_ui_dir(&ui_root).expect("failed to run ui tests");
    assert!(summary.total >= 60, "expected >= 60 ui tests, got {}", summary.total);
    if !summary.failures.is_empty() {
        for f in &summary.failures {
            eprintln!("FAIL {}: {}", f.path.display(), f.reason);
        }
        panic!("{} ui test(s) failed", summary.failures.len());
    }
}

#[test]
fn bom_and_invalid_utf8_emit_e0001() {
    let mut sm = SourceMap::new();
    let bom_bytes = b"\xEF\xBB\xBFmodule bom;\npub fn f() {}\n".to_vec();
    let f1 = sm.add_file_bytes("bom.eon", bom_bytes);
    let mut diags1 = DiagBag::new();
    let lex1 = lex_file(sm.file(f1), &mut diags1);
    let mut interner = Interner::new();
    parse_file(sm.file(f1), &lex1, &mut interner, &mut diags1);
    assert!(
        diags1.diagnostics().iter().any(|d| d.code == ErrorCode::E0001),
        "expected E0001 for BOM"
    );

    let bad_utf8 = b"module bad;\npub fn f() { // \xFF\xFE\n}\n".to_vec();
    let f2 = sm.add_file_bytes("bad_utf8.eon", bad_utf8);
    let mut diags2 = DiagBag::new();
    let lex2 = lex_file(sm.file(f2), &mut diags2);
    parse_file(sm.file(f2), &lex2, &mut interner, &mut diags2);
    assert!(
        diags2.diagnostics().iter().any(|d| d.code == ErrorCode::E0001),
        "expected E0001 for invalid UTF-8"
    );
}

#[test]
fn crlf_line_endings_and_raw_strings() {
    let src = "module crlf;\r\npub const S: *str =\r\n    \\\\line1\r\n    \\\\line2\r\n;\r\n";
    let mut sm = SourceMap::new();
    let fid = sm.add_file_str("crlf.eon", src);
    let file = sm.file(fid);
    assert_eq!(file.profile, FileProfile::Safe);
    let mut diags = DiagBag::new();
    let lexed = lex_file(file, &mut diags);
    assert!(!diags.has_errors(), "{}", diags.render_all(&sm));

    let raw_str_tok = lexed
        .tokens
        .iter()
        .find_map(|t| match t.kind {
            TokenKind::RawStringLit(id) => Some(lexed.string_lits[id.0 as usize].as_str()),
            _ => None,
        })
        .expect("expected RawStringLit");
    assert_eq!(raw_str_tok, "line1\nline2");

    let mut interner = Interner::new();
    parse_file(file, &lexed, &mut interner, &mut diags);
    assert!(!diags.has_errors(), "{}", diags.render_all(&sm));
}

#[test]
fn interner_deterministic_and_dedups() {
    let mut interner = Interner::new();
    let mut syms = Vec::new();
    for i in 0..2000 {
        let name = format!("symbol_name_{}", i);
        syms.push(interner.intern(&name));
    }
    for (i, &sym) in syms.iter().enumerate() {
        let name = format!("symbol_name_{}", i);
        assert_eq!(interner.intern(&name), sym);
        assert_eq!(interner.get(sym), name);
    }
}

#[test]
fn lexer_maximal_munch_and_contextual_keywords() {
    let src = "module m;\npub fn f(a: u32, b: u32) -> u32 {\n    let x = (a <<% 1_u32) +% (b *| 2_u32);\n    let y = read_write(a.read(), b.write());\n    return x +| y;\n}\n";
    let mut sm = SourceMap::new();
    let fid = sm.add_file_str("munch.eon", src);
    let mut diags = DiagBag::new();
    let lexed = lex_file(sm.file(fid), &mut diags);
    assert!(!diags.has_errors());
    assert!(lexed.tokens.iter().any(|t| t.kind == TokenKind::ShlWrap));
    assert!(lexed.tokens.iter().any(|t| t.kind == TokenKind::PlusWrap));
    assert!(lexed.tokens.iter().any(|t| t.kind == TokenKind::StarSat));
    assert!(lexed.tokens.iter().any(|t| t.kind == TokenKind::PlusSat));
}

#[test]
fn parser_never_panics_on_extreme_nesting() {
    let deep_parens = format!("module deep;\npub fn f() -> u32 {{ return {}1_u32{}; }}", "(".repeat(1000), ")".repeat(1000));
    let deep_types = format!("module deep_t;\npub fn f(x: {}u32) {{ _ = x; }}", "?".repeat(1000));
    let deep_blocks = format!("module deep_b;\npub fn f() {{ {} {} }}", "if true { ".repeat(600), "}".repeat(600));
    let deep_arrays = format!("module deep_a;\npub fn f(x: {}u8) {{ _ = x; }}", "[1]".repeat(600));
    let deep_else_if = format!(
        "module deep_ei;\npub fn f(c: bool) {{ if c {{}} {} else {{}} }}",
        " else if c {}".repeat(600)
    );

    for (name, src) in [
        ("parens.eon", deep_parens),
        ("types.eon", deep_types),
        ("blocks.eon", deep_blocks),
        ("arrays.eon", deep_arrays),
        ("else_if.eon", deep_else_if),
    ] {
        let mut sm = SourceMap::new();
        let fid = sm.add_file_str(name, &src);
        let mut diags = DiagBag::new();
        let lexed = lex_file(sm.file(fid), &mut diags);
        let mut interner = Interner::new();
        parse_file(sm.file(fid), &lexed, &mut interner, &mut diags);
        assert!(diags.has_errors(), "expected depth error on {}", name);
    }
}

#[test]
fn fuzz_garbage_and_mutations_zero_panics() {
    let corpus: &[&[u8]] = &[
        b"module a; pub fn f(x: *mut [16]u8) -> u32 { return 0_u32; }",
        b"module h @hal; extern \"C\" fn foo(); pub fn g() { raw { asm(\"nop\", out(reg) x); } }",
        b"module m; pub pool P { a: f32x2; parent: ?Link[Self]; } pub sys s(p: write P.a) { par for r in p.each_mut() { r.a = r.a; } }",
        b"module e; pub fn m() { let x = match y { .A(1..=2) => return 1, _ => 0 }; }",
        b"",
        b"\xEF\xBB\xBF\xFF\x00\x07\xE2\x80\xAE'\"\\/*",
    ];

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_u64 = || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for iter in 0..2500 {
        let seed = corpus[iter % corpus.len()];
        let mut buf = seed.to_vec();
        let ops = (next_u64() % 6) + 1;
        for _ in 0..ops {
            if buf.is_empty() {
                buf.push((next_u64() & 0xFF) as u8);
                continue;
            }
            match next_u64() % 4 {
                0 => {
                    let idx = (next_u64() as usize) % buf.len();
                    buf[idx] ^= 1u8 << (next_u64() % 8);
                }
                1 => {
                    let idx = (next_u64() as usize) % buf.len();
                    buf.truncate(idx);
                }
                2 => {
                    let idx = (next_u64() as usize) % (buf.len() + 1);
                    buf.insert(idx, (next_u64() & 0xFF) as u8);
                }
                _ => {
                    let idx = (next_u64() as usize) % buf.len();
                    buf.remove(idx);
                }
            }
        }
        let name = if iter % 3 == 0 { "fuzz.hal.eon" } else { "fuzz.eon" };
        let mut sm = SourceMap::new();
        let fid = sm.add_file_bytes(name, buf);
        let mut diags = DiagBag::new();
        let lexed = lex_file(sm.file(fid), &mut diags);
        let mut interner = Interner::new();
        parse_file(sm.file(fid), &lexed, &mut interner, &mut diags);
    }
}
