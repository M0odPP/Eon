#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Instant;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

const NUM_BLOCKS: usize = 4000; // 4000 * 25 = 100,000 lines exact
const ITERS: usize = 5;
const NONE_IDX: u32 = u32::MAX;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TokKind {
    Eof = 0,
    Ident = 1,
    Int = 2,
    KwPub = 3,
    KwStruct = 4,
    KwFn = 5,
    KwLet = 6,
    KwVar = 7,
    KwIf = 8,
    KwElse = 9,
    KwWhile = 10,
    KwReturn = 11,
    LParen = 12,
    RParen = 13,
    LBrace = 14,
    RBrace = 15,
    Comma = 16,
    Colon = 17,
    Semi = 18,
    Arrow = 19,
    Eq = 20,
    EqEq = 21,
    Bang = 22,
    BangEq = 23,
    Lt = 24,
    Le = 25,
    Shl = 26,
    Gt = 27,
    Ge = 28,
    Shr = 29,
    Amp = 30,
    AmpAmp = 31,
    Pipe = 32,
    PipePipe = 33,
    Caret = 34,
    Plus = 35,
    Minus = 36,
    Star = 37,
    Slash = 38,
    Percent = 39,
}

#[derive(Clone, Copy)]
struct Token {
    kind: TokKind,
    start: u32,
    len: u32,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum AstKind {
    StructDecl = 0,
    FieldDecl = 1,
    FnDecl = 2,
    Param = 3,
    Block = 4,
    LetStmt = 5,
    VarStmt = 6,
    AssignStmt = 7,
    IfStmt = 8,
    WhileStmt = 9,
    ReturnStmt = 10,
    IntExpr = 11,
    IdentExpr = 12,
    UnaryExpr = 13,
    BinaryExpr = 14,
}

#[derive(Clone, Copy)]
struct AstNode {
    kind: AstKind,
    op: TokKind,
    val: i64,
    tok_idx: u32,
    a: u32,
    b: u32,
    c: u32,
    next: u32,
}

#[derive(Debug)]
enum ParseError {
    UnexpectedChar,
    UnexpectedToken,
    UnexpectedEof,
    TooDeep,
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn push_u64_dec(out: &mut Vec<u8>, mut val: u64) {
    if val == 0 {
        out.push(b'0');
        return;
    }
    let mut digits = [0u8; 24];
    let mut count = 0usize;
    while val > 0 {
        digits[count] = b'0' + (val % 10) as u8;
        val /= 10;
        count += 1;
    }
    while count > 0 {
        count -= 1;
        out.push(digits[count]);
    }
}

fn push_hex_u16(out: &mut Vec<u8>, val: u16) {
    out.extend_from_slice(b"0x");
    let hex = b"0123456789ABCDEF";
    out.push(hex[((val >> 12) & 0xF) as usize]);
    out.push(hex[((val >> 8) & 0xF) as usize]);
    out.push(hex[((val >> 4) & 0xF) as usize]);
    out.push(hex[(val & 0xF) as usize]);
}

fn generate_eon_source(num_blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_blocks * 1150);
    let mut rng: u64 = 0x5EED_CAFE_BEEF_1234;

    for b in 0..num_blocks {
        let c1 = 2 + (next_u64(&mut rng) % 13);
        let c2 = 2 + (next_u64(&mut rng) % 7);
        let c3 = 3 + (next_u64(&mut rng) % 19);
        let c4 = 1 + (next_u64(&mut rng) % 97);
        let hex1 = (next_u64(&mut rng) & 0xFFFF) as u16;
        let c5 = 1 + (next_u64(&mut rng) % 15);
        let c6 = 2 + (next_u64(&mut rng) % 11);
        let c7 = 3 + (next_u64(&mut rng) % 13);
        let c8 = 1 + (next_u64(&mut rng) % 53);
        let op_sel = next_u64(&mut rng) % 4;
        let cmp_sel = next_u64(&mut rng) % 4;

        // Line 1
        out.extend_from_slice(b"// Generated module block ");
        push_u64_dec(&mut out, b as u64);
        out.extend_from_slice(b" deterministic lexer and Pratt parser workload\n");
        // Lines 2..7 (6 lines)
        out.extend_from_slice(b"pub struct RecordState_");
        push_u64_dec(&mut out, b as u64);
        out.extend_from_slice(b" {\n");
        out.extend_from_slice(b"    pub accumulator_val: i64;\n");
        out.extend_from_slice(b"    pub threshold_limit: u64;\n");
        out.extend_from_slice(b"    pub stride_offset: usize;\n");
        out.extend_from_slice(b"    pub flags_bitmask: u32;\n");
        out.extend_from_slice(b"}\n");
        // Line 8
        out.extend_from_slice(b"// Kernel computation block ");
        push_u64_dec(&mut out, b as u64);
        out.extend_from_slice(b" with nested expressions and control flow\n");
        // Line 9
        out.extend_from_slice(b"pub fn compute_kernel_block_");
        push_u64_dec(&mut out, b as u64);
        out.extend_from_slice(b"(alpha_input: i64, beta_scale: i64, max_limit: i64) -> i64 {\n");
        // Line 10
        out.extend_from_slice(b"    var accumulator_state: i64 = (alpha_input + beta_scale * ");
        push_u64_dec(&mut out, c1);
        out.extend_from_slice(b") - (max_limit / ");
        push_u64_dec(&mut out, c2);
        out.extend_from_slice(b");\n");
        // Line 11
        out.extend_from_slice(b"    let stride_factor: i64 = (alpha_input * ");
        push_u64_dec(&mut out, c3);
        out.extend_from_slice(b" + beta_scale - ");
        push_u64_dec(&mut out, c4);
        out.extend_from_slice(b") ^ ");
        push_hex_u16(&mut out, hex1);
        out.extend_from_slice(b";\n");
        // Line 12
        out.extend_from_slice(b"    var loop_index: i64 = 0;\n");
        // Line 13
        out.extend_from_slice(b"    while loop_index < max_limit && accumulator_state != 0 {\n");
        // Line 14
        out.extend_from_slice(b"        if (loop_index & ");
        push_u64_dec(&mut out, c5);
        out.extend_from_slice(b") == 0 || accumulator_state ");
        match cmp_sel {
            0 => out.extend_from_slice(b"<"),
            1 => out.extend_from_slice(b"<="),
            2 => out.extend_from_slice(b">"),
            _ => out.extend_from_slice(b">="),
        }
        out.extend_from_slice(b" stride_factor {\n");
        // Line 15
        out.extend_from_slice(b"            accumulator_state = accumulator_state ");
        match op_sel {
            0 => out.extend_from_slice(b"+"),
            1 => out.extend_from_slice(b"-"),
            2 => out.extend_from_slice(b"^"),
            _ => out.extend_from_slice(b"|"),
        }
        out.extend_from_slice(
            b" (loop_index * stride_factor - beta_scale) ^ (alpha_input >> 1);\n",
        );
        // Line 16
        out.extend_from_slice(b"        } else {\n");
        // Line 17
        out.extend_from_slice(b"            accumulator_state = accumulator_state - (loop_index + 1) * ");
        push_u64_dec(&mut out, c6);
        out.extend_from_slice(b" + (max_limit % ");
        push_u64_dec(&mut out, c7);
        out.extend_from_slice(b");\n");
        // Line 18
        out.extend_from_slice(b"        }\n");
        // Line 19
        out.extend_from_slice(b"        if !(accumulator_state <= 0) {\n");
        // Line 20
        out.extend_from_slice(b"            accumulator_state = (accumulator_state << 1) | (loop_index & 7);\n");
        // Line 21
        out.extend_from_slice(b"        }\n");
        // Line 22
        out.extend_from_slice(b"        loop_index = loop_index + 1;\n");
        // Line 23
        out.extend_from_slice(b"    }\n");
        // Line 24
        out.extend_from_slice(
            b"    return accumulator_state + stride_factor * (alpha_input - beta_scale) + (-",
        );
        push_u64_dec(&mut out, c8);
        out.extend_from_slice(b");\n");
        // Line 25
        out.extend_from_slice(b"}\n");
    }
    out
}

#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_ident_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

#[inline]
fn classify_ident(word: &[u8]) -> TokKind {
    match word.len() {
        2 => {
            if word == b"fn" {
                TokKind::KwFn
            } else if word == b"if" {
                TokKind::KwIf
            } else {
                TokKind::Ident
            }
        }
        3 => {
            if word == b"pub" {
                TokKind::KwPub
            } else if word == b"let" {
                TokKind::KwLet
            } else if word == b"var" {
                TokKind::KwVar
            } else {
                TokKind::Ident
            }
        }
        4 => {
            if word == b"else" {
                TokKind::KwElse
            } else {
                TokKind::Ident
            }
        }
        5 => {
            if word == b"while" {
                TokKind::KwWhile
            } else {
                TokKind::Ident
            }
        }
        6 => {
            if word == b"struct" {
                TokKind::KwStruct
            } else if word == b"return" {
                TokKind::KwReturn
            } else {
                TokKind::Ident
            }
        }
        _ => TokKind::Ident,
    }
}

fn lex_source(src: &[u8], toks: &mut Vec<Token>) -> Result<(), ParseError> {
    let n = src.len();
    let mut p: usize = 0;

    while p < n {
        let b = src[p];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            p += 1;
            while p < n {
                let wb = src[p];
                if wb != b' ' && wb != b'\t' && wb != b'\n' && wb != b'\r' {
                    break;
                }
                p += 1;
            }
            continue;
        }
        if b == b'/' && p + 1 < n && src[p + 1] == b'/' {
            p += 2;
            while p < n && src[p] != b'\n' {
                p += 1;
            }
            continue;
        }

        let start = p;
        if is_ident_start(b) {
            p += 1;
            while p < n && is_ident_cont(src[p]) {
                p += 1;
            }
            let kind = classify_ident(&src[start..p]);
            toks.push(Token {
                kind,
                start: start as u32,
                len: (p - start) as u32,
            });
            continue;
        }
        if b.is_ascii_digit() {
            if b == b'0' && p + 1 < n && (src[p + 1] == b'x' || src[p + 1] == b'X') {
                p += 2;
                while p < n && is_hex_digit(src[p]) {
                    p += 1;
                }
            } else {
                p += 1;
                while p < n && src[p].is_ascii_digit() {
                    p += 1;
                }
            }
            toks.push(Token {
                kind: TokKind::Int,
                start: start as u32,
                len: (p - start) as u32,
            });
            continue;
        }

        let next_b = if p + 1 < n { src[p + 1] } else { 0 };
        let (kind, adv) = match b {
            b'(' => (TokKind::LParen, 1),
            b')' => (TokKind::RParen, 1),
            b'{' => (TokKind::LBrace, 1),
            b'}' => (TokKind::RBrace, 1),
            b',' => (TokKind::Comma, 1),
            b':' => (TokKind::Colon, 1),
            b';' => (TokKind::Semi, 1),
            b'+' => (TokKind::Plus, 1),
            b'*' => (TokKind::Star, 1),
            b'/' => (TokKind::Slash, 1),
            b'%' => (TokKind::Percent, 1),
            b'^' => (TokKind::Caret, 1),
            b'-' => {
                if next_b == b'>' {
                    (TokKind::Arrow, 2)
                } else {
                    (TokKind::Minus, 1)
                }
            }
            b'=' => {
                if next_b == b'=' {
                    (TokKind::EqEq, 2)
                } else {
                    (TokKind::Eq, 1)
                }
            }
            b'!' => {
                if next_b == b'=' {
                    (TokKind::BangEq, 2)
                } else {
                    (TokKind::Bang, 1)
                }
            }
            b'<' => {
                if next_b == b'=' {
                    (TokKind::Le, 2)
                } else if next_b == b'<' {
                    (TokKind::Shl, 2)
                } else {
                    (TokKind::Lt, 1)
                }
            }
            b'>' => {
                if next_b == b'=' {
                    (TokKind::Ge, 2)
                } else if next_b == b'>' {
                    (TokKind::Shr, 2)
                } else {
                    (TokKind::Gt, 1)
                }
            }
            b'&' => {
                if next_b == b'&' {
                    (TokKind::AmpAmp, 2)
                } else {
                    (TokKind::Amp, 1)
                }
            }
            b'|' => {
                if next_b == b'|' {
                    (TokKind::PipePipe, 2)
                } else {
                    (TokKind::Pipe, 1)
                }
            }
            _ => return Err(ParseError::UnexpectedChar),
        };
        p += adv;
        toks.push(Token {
            kind,
            start: start as u32,
            len: adv as u32,
        });
    }

    toks.push(Token {
        kind: TokKind::Eof,
        start: n as u32,
        len: 0,
    });
    Ok(())
}

struct Parser {
    pos: usize,
    depth: u32,
}

#[inline]
fn peek_kind(toks: &[Token], pos: usize) -> TokKind {
    if pos < toks.len() {
        toks[pos].kind
    } else {
        TokKind::Eof
    }
}

#[inline]
fn expect_tok(toks: &[Token], st: &mut Parser, expected: TokKind) -> Result<u32, ParseError> {
    if st.pos >= toks.len() {
        return Err(ParseError::UnexpectedEof);
    }
    if toks[st.pos].kind != expected {
        return Err(ParseError::UnexpectedToken);
    }
    let idx = st.pos as u32;
    st.pos += 1;
    Ok(idx)
}

#[inline]
fn parse_int_token(src: &[u8], tok: Token) -> i64 {
    let s = tok.start as usize;
    let e = s + tok.len as usize;
    if e > src.len() || s >= e {
        return 0;
    }
    if e - s >= 2 && src[s] == b'0' && (src[s + 1] == b'x' || src[s + 1] == b'X') {
        let mut acc: i64 = 0;
        let mut i = s + 2;
        while i < e {
            let b = src[i];
            let d = if b >= b'0' && b <= b'9' {
                (b - b'0') as i64
            } else if b >= b'a' && b <= b'f' {
                (b - b'a' + 10) as i64
            } else if b >= b'A' && b <= b'F' {
                (b - b'A' + 10) as i64
            } else {
                0
            };
            acc = (acc << 4) | d;
            i += 1;
        }
        acc
    } else {
        let mut acc: i64 = 0;
        let mut i = s;
        while i < e {
            let b = src[i];
            if b >= b'0' && b <= b'9' {
                acc = acc.wrapping_mul(10).wrapping_add((b - b'0') as i64);
            }
            i += 1;
        }
        acc
    }
}

#[inline]
fn hash_ident_token(src: &[u8], tok: Token) -> i64 {
    let s = tok.start as usize;
    let e = s + tok.len as usize;
    let mut h: u64 = FNV_OFFSET;
    if e <= src.len() {
        let mut i = s;
        while i < e {
            h = (h ^ src[i] as u64).wrapping_mul(FNV_PRIME);
            i += 1;
        }
    }
    h as i64
}

#[inline]
fn infix_prec(k: TokKind) -> u8 {
    match k {
        TokKind::PipePipe => 1,
        TokKind::AmpAmp => 2,
        TokKind::Pipe => 3,
        TokKind::Caret => 4,
        TokKind::Amp => 5,
        TokKind::EqEq | TokKind::BangEq => 6,
        TokKind::Lt | TokKind::Le | TokKind::Gt | TokKind::Ge => 7,
        TokKind::Shl | TokKind::Shr => 8,
        TokKind::Plus | TokKind::Minus => 9,
        TokKind::Star | TokKind::Slash | TokKind::Percent => 10,
        _ => 0,
    }
}

#[inline]
fn push_ast(nodes: &mut Vec<AstNode>, node: AstNode) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(node);
    id
}

fn parse_expr(
    src: &[u8],
    toks: &[Token],
    st: &mut Parser,
    nodes: &mut Vec<AstNode>,
    min_prec: u8,
) -> Result<u32, ParseError> {
    if st.depth >= 64 {
        return Err(ParseError::TooDeep);
    }
    st.depth += 1;

    let k0 = peek_kind(toks, st.pos);
    let mut lhs = match k0 {
        TokKind::Int => {
            let tidx = st.pos as u32;
            let val = parse_int_token(src, toks[st.pos]);
            st.pos += 1;
            push_ast(
                nodes,
                AstNode {
                    kind: AstKind::IntExpr,
                    op: TokKind::Int,
                    val,
                    tok_idx: tidx,
                    a: NONE_IDX,
                    b: NONE_IDX,
                    c: NONE_IDX,
                    next: NONE_IDX,
                },
            )
        }
        TokKind::Ident => {
            let tidx = st.pos as u32;
            let h = hash_ident_token(src, toks[st.pos]);
            st.pos += 1;
            push_ast(
                nodes,
                AstNode {
                    kind: AstKind::IdentExpr,
                    op: TokKind::Ident,
                    val: h,
                    tok_idx: tidx,
                    a: NONE_IDX,
                    b: NONE_IDX,
                    c: NONE_IDX,
                    next: NONE_IDX,
                },
            )
        }
        TokKind::Minus | TokKind::Bang => {
            let tidx = st.pos as u32;
            st.pos += 1;
            let sub = parse_expr(src, toks, st, nodes, 11)?;
            push_ast(
                nodes,
                AstNode {
                    kind: AstKind::UnaryExpr,
                    op: k0,
                    val: 0,
                    tok_idx: tidx,
                    a: sub,
                    b: NONE_IDX,
                    c: NONE_IDX,
                    next: NONE_IDX,
                },
            )
        }
        TokKind::LParen => {
            st.pos += 1;
            let inner = parse_expr(src, toks, st, nodes, 1)?;
            let _ = expect_tok(toks, st, TokKind::RParen)?;
            inner
        }
        _ => return Err(ParseError::UnexpectedToken),
    };

    loop {
        let op = peek_kind(toks, st.pos);
        let prec = infix_prec(op);
        if prec == 0 || prec < min_prec {
            break;
        }
        let op_tidx = st.pos as u32;
        st.pos += 1;
        let rhs = parse_expr(src, toks, st, nodes, prec + 1)?;
        lhs = push_ast(
            nodes,
            AstNode {
                kind: AstKind::BinaryExpr,
                op,
                val: prec as i64,
                tok_idx: op_tidx,
                a: lhs,
                b: rhs,
                c: NONE_IDX,
                next: NONE_IDX,
            },
        );
    }

    st.depth -= 1;
    Ok(lhs)
}

fn parse_block(
    src: &[u8],
    toks: &[Token],
    st: &mut Parser,
    nodes: &mut Vec<AstNode>,
) -> Result<u32, ParseError> {
    if st.depth >= 64 {
        return Err(ParseError::TooDeep);
    }
    st.depth += 1;
    let lb_idx = expect_tok(toks, st, TokKind::LBrace)?;
    let blk_id = push_ast(
        nodes,
        AstNode {
            kind: AstKind::Block,
            op: TokKind::LBrace,
            val: 0,
            tok_idx: lb_idx,
            a: NONE_IDX,
            b: NONE_IDX,
            c: NONE_IDX,
            next: NONE_IDX,
        },
    );

    let mut prev_stmt: u32 = NONE_IDX;
    let mut stmt_count: i64 = 0;

    while peek_kind(toks, st.pos) != TokKind::RBrace && peek_kind(toks, st.pos) != TokKind::Eof {
        let sk = peek_kind(toks, st.pos);
        let stmt_id = match sk {
            TokKind::KwLet | TokKind::KwVar => {
                st.pos += 1;
                let name_idx = expect_tok(toks, st, TokKind::Ident)?;
                let _ = expect_tok(toks, st, TokKind::Colon)?;
                let ty_idx = expect_tok(toks, st, TokKind::Ident)?;
                let _ = expect_tok(toks, st, TokKind::Eq)?;
                let init_expr = parse_expr(src, toks, st, nodes, 1)?;
                let _ = expect_tok(toks, st, TokKind::Semi)?;
                let ast_k = if sk == TokKind::KwLet {
                    AstKind::LetStmt
                } else {
                    AstKind::VarStmt
                };
                push_ast(
                    nodes,
                    AstNode {
                        kind: ast_k,
                        op: sk,
                        val: ty_idx as i64,
                        tok_idx: name_idx,
                        a: init_expr,
                        b: NONE_IDX,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                )
            }
            TokKind::KwIf => {
                let if_idx = st.pos as u32;
                st.pos += 1;
                let cond = parse_expr(src, toks, st, nodes, 1)?;
                let then_blk = parse_block(src, toks, st, nodes)?;
                let else_blk = if peek_kind(toks, st.pos) == TokKind::KwElse {
                    st.pos += 1;
                    parse_block(src, toks, st, nodes)?
                } else {
                    NONE_IDX
                };
                push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::IfStmt,
                        op: TokKind::KwIf,
                        val: 0,
                        tok_idx: if_idx,
                        a: cond,
                        b: then_blk,
                        c: else_blk,
                        next: NONE_IDX,
                    },
                )
            }
            TokKind::KwWhile => {
                let w_idx = st.pos as u32;
                st.pos += 1;
                let cond = parse_expr(src, toks, st, nodes, 1)?;
                let body = parse_block(src, toks, st, nodes)?;
                push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::WhileStmt,
                        op: TokKind::KwWhile,
                        val: 0,
                        tok_idx: w_idx,
                        a: cond,
                        b: body,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                )
            }
            TokKind::KwReturn => {
                let r_idx = st.pos as u32;
                st.pos += 1;
                let val_expr = parse_expr(src, toks, st, nodes, 1)?;
                let _ = expect_tok(toks, st, TokKind::Semi)?;
                push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::ReturnStmt,
                        op: TokKind::KwReturn,
                        val: 0,
                        tok_idx: r_idx,
                        a: val_expr,
                        b: NONE_IDX,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                )
            }
            TokKind::Ident => {
                let lhs_idx = st.pos as u32;
                st.pos += 1;
                let _ = expect_tok(toks, st, TokKind::Eq)?;
                let rhs = parse_expr(src, toks, st, nodes, 1)?;
                let _ = expect_tok(toks, st, TokKind::Semi)?;
                push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::AssignStmt,
                        op: TokKind::Eq,
                        val: 0,
                        tok_idx: lhs_idx,
                        a: rhs,
                        b: NONE_IDX,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                )
            }
            _ => return Err(ParseError::UnexpectedToken),
        };

        if prev_stmt != NONE_IDX {
            nodes[prev_stmt as usize].next = stmt_id;
        } else {
            nodes[blk_id as usize].a = stmt_id;
        }
        prev_stmt = stmt_id;
        stmt_count += 1;
    }

    let _ = expect_tok(toks, st, TokKind::RBrace)?;
    nodes[blk_id as usize].val = stmt_count;
    st.depth -= 1;
    Ok(blk_id)
}

fn parse_program(
    src: &[u8],
    toks: &[Token],
    nodes: &mut Vec<AstNode>,
) -> Result<u32, ParseError> {
    let mut st = Parser { pos: 0, depth: 0 };
    let mut first_decl: u32 = NONE_IDX;
    let mut prev_decl: u32 = NONE_IDX;

    while peek_kind(toks, st.pos) != TokKind::Eof {
        let mut is_pub: i64 = 0;
        if peek_kind(toks, st.pos) == TokKind::KwPub {
            is_pub = 1;
            st.pos += 1;
        }
        let dk = peek_kind(toks, st.pos);
        let decl_id = if dk == TokKind::KwStruct {
            st.pos += 1;
            let name_idx = expect_tok(toks, &mut st, TokKind::Ident)?;
            let _ = expect_tok(toks, &mut st, TokKind::LBrace)?;
            let s_id = push_ast(
                nodes,
                AstNode {
                    kind: AstKind::StructDecl,
                    op: TokKind::KwStruct,
                    val: is_pub,
                    tok_idx: name_idx,
                    a: NONE_IDX,
                    b: NONE_IDX,
                    c: NONE_IDX,
                    next: NONE_IDX,
                },
            );
            let mut prev_f: u32 = NONE_IDX;
            while peek_kind(toks, st.pos) != TokKind::RBrace
                && peek_kind(toks, st.pos) != TokKind::Eof
            {
                if peek_kind(toks, st.pos) == TokKind::KwPub {
                    st.pos += 1;
                }
                let fname = expect_tok(toks, &mut st, TokKind::Ident)?;
                let _ = expect_tok(toks, &mut st, TokKind::Colon)?;
                let fty = expect_tok(toks, &mut st, TokKind::Ident)?;
                let _ = expect_tok(toks, &mut st, TokKind::Semi)?;
                let f_id = push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::FieldDecl,
                        op: TokKind::Colon,
                        val: fty as i64,
                        tok_idx: fname,
                        a: NONE_IDX,
                        b: NONE_IDX,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                );
                if prev_f != NONE_IDX {
                    nodes[prev_f as usize].next = f_id;
                } else {
                    nodes[s_id as usize].a = f_id;
                }
                prev_f = f_id;
            }
            let _ = expect_tok(toks, &mut st, TokKind::RBrace)?;
            s_id
        } else if dk == TokKind::KwFn {
            st.pos += 1;
            let name_idx = expect_tok(toks, &mut st, TokKind::Ident)?;
            let _ = expect_tok(toks, &mut st, TokKind::LParen)?;
            let fn_id = push_ast(
                nodes,
                AstNode {
                    kind: AstKind::FnDecl,
                    op: TokKind::KwFn,
                    val: is_pub,
                    tok_idx: name_idx,
                    a: NONE_IDX,
                    b: NONE_IDX,
                    c: NONE_IDX,
                    next: NONE_IDX,
                },
            );
            let mut prev_p: u32 = NONE_IDX;
            while peek_kind(toks, st.pos) != TokKind::RParen
                && peek_kind(toks, st.pos) != TokKind::Eof
            {
                let pname = expect_tok(toks, &mut st, TokKind::Ident)?;
                let _ = expect_tok(toks, &mut st, TokKind::Colon)?;
                let pty = expect_tok(toks, &mut st, TokKind::Ident)?;
                let p_id = push_ast(
                    nodes,
                    AstNode {
                        kind: AstKind::Param,
                        op: TokKind::Colon,
                        val: pty as i64,
                        tok_idx: pname,
                        a: NONE_IDX,
                        b: NONE_IDX,
                        c: NONE_IDX,
                        next: NONE_IDX,
                    },
                );
                if prev_p != NONE_IDX {
                    nodes[prev_p as usize].next = p_id;
                } else {
                    nodes[fn_id as usize].a = p_id;
                }
                prev_p = p_id;
                if peek_kind(toks, st.pos) == TokKind::Comma {
                    st.pos += 1;
                } else {
                    break;
                }
            }
            let _ = expect_tok(toks, &mut st, TokKind::RParen)?;
            let _ = expect_tok(toks, &mut st, TokKind::Arrow)?;
            let _ret_ty = expect_tok(toks, &mut st, TokKind::Ident)?;
            let body = parse_block(src, toks, &mut st, nodes)?;
            nodes[fn_id as usize].b = body;
            fn_id
        } else {
            return Err(ParseError::UnexpectedToken);
        };

        if prev_decl != NONE_IDX {
            nodes[prev_decl as usize].next = decl_id;
        } else {
            first_decl = decl_id;
        }
        prev_decl = decl_id;
    }

    Ok(first_decl)
}

#[inline]
fn fnv_mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(FNV_PRIME)
}

fn walk_ast(nodes: &[AstNode], mut id: u32, mut h: u64) -> u64 {
    while id != NONE_IDX {
        let n = &nodes[id as usize];
        let tag = ((n.kind as u64) << 8) | (n.op as u64);
        h = fnv_mix(h, tag);
        h = fnv_mix(h, n.val as u64);
        h = fnv_mix(h, n.tok_idx as u64);
        if n.a != NONE_IDX {
            h = walk_ast(nodes, n.a, fnv_mix(h, 0x1A));
        }
        if n.b != NONE_IDX {
            h = walk_ast(nodes, n.b, fnv_mix(h, 0x2B));
        }
        if n.c != NONE_IDX {
            h = walk_ast(nodes, n.c, fnv_mix(h, 0x3C));
        }
        id = n.next;
    }
    h
}

fn main() {
    let source = generate_eon_source(NUM_BLOCKS);
    let mut toks: Vec<Token> = Vec::with_capacity(1_048_576);
    let mut nodes: Vec<AstNode> = Vec::with_capacity(524_288);

    let t0 = Instant::now();
    let mut checksum: u64 = FNV_OFFSET;
    for iter in 0..ITERS {
        let src = black_box(source.as_slice());
        toks.clear();
        nodes.clear();
        lex_source(src, &mut toks).expect("valid generated Eon source");
        let first_decl = parse_program(src, &toks, &mut nodes).expect("valid generated AST");

        let mut h: u64 = FNV_OFFSET;
        for t in &toks {
            h = fnv_mix(h, ((t.kind as u64) << 16) | (t.len as u64));
        }
        h = fnv_mix(h, toks.len() as u64);
        h = fnv_mix(h, nodes.len() as u64);
        h = walk_ast(&nodes, first_decl, h);
        checksum = fnv_mix(checksum, h).wrapping_add(iter as u64);
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    black_box(checksum);

    println!(
        "RESULT eon_lexer_parser checksum={} elapsed_ns={}",
        checksum, elapsed_ns
    );
}
