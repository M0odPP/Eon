mod decl;
mod expr;
mod stmt;
mod ty;

use crate::ast::{CompilationUnit, IdentSpan};
use crate::diag::{DiagBag, ErrorCode};
use crate::lexer::{LexedFile, Token, TokenKind};
use crate::span::{FileId, FileProfile, Interner, SourceFile, Span};

const MAX_RECURSION_DEPTH: u32 = 128;

pub fn parse_file(
    file: &SourceFile,
    lexed: &LexedFile,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> CompilationUnit {
    let mut parser = Parser {
        file_id: file.id,
        profile: file.profile,
        src: file.text(),
        lexed,
        tokens: &lexed.tokens,
        pos: 0,
        interner,
        diags,
        depth: 0,
        allow_return_as_expr: false,
        in_ensures_or_where: false,
    };
    parser.parse_compilation_unit()
}

pub(crate) struct Parser<'a> {
    pub(crate) file_id: FileId,
    pub(crate) profile: FileProfile,
    pub(crate) src: &'a str,
    pub(crate) lexed: &'a LexedFile,
    pub(crate) tokens: &'a [Token],
    pub(crate) pos: usize,
    pub(crate) interner: &'a mut Interner,
    pub(crate) diags: &'a mut DiagBag,
    pub(crate) depth: u32,
    pub(crate) allow_return_as_expr: bool,
    pub(crate) in_ensures_or_where: bool,
}

impl<'a> Parser<'a> {
    #[inline]
    pub(crate) fn cur(&self) -> Token {
        self.tokens
            .get(self.pos)
            .copied()
            .unwrap_or_else(|| *self.tokens.last().unwrap())
    }

    #[inline]
    pub(crate) fn kind(&self) -> TokenKind {
        self.cur().kind
    }

    #[inline]
    pub(crate) fn peek_kind(&self, offset: usize) -> TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map(|t| t.kind)
            .unwrap_or(TokenKind::Eof)
    }

    #[inline]
    pub(crate) fn span(&self) -> Span {
        self.cur().span
    }

    #[inline]
    pub(crate) fn slice(&self, span: Span) -> &'a str {
        &self.src[span.start as usize..span.end as usize]
    }

    #[inline]
    pub(crate) fn at(&self, k: TokenKind) -> bool {
        self.kind() == k
    }

    #[inline]
    pub(crate) fn at_eof(&self) -> bool {
        matches!(self.kind(), TokenKind::Eof)
    }

    #[inline]
    pub(crate) fn bump(&mut self) -> Token {
        let t = self.cur();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    #[inline]
    pub(crate) fn eat(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    #[inline]
    pub(crate) fn prev_end_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            self.span()
        }
    }

    pub(crate) fn enter_depth(&mut self) -> bool {
        if self.depth >= MAX_RECURSION_DEPTH {
            self.diags.error(
                ErrorCode::E0005,
                self.span(),
                "maximum syntactic nesting depth exceeded",
            );
            return false;
        }
        self.depth += 1;
        true
    }

    pub(crate) fn leave_depth(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    pub(crate) fn expect(&mut self, expected: TokenKind) -> Option<Token> {
        if self.at(expected) {
            Some(self.bump())
        } else {
            self.diags.error(
                ErrorCode::E0005,
                self.span(),
                format!(
                    "expected {}, found {}",
                    expected.description(),
                    self.kind().description()
                ),
            );
            None
        }
    }

    pub(crate) fn expect_semi(&mut self) {
        if !self.eat(TokenKind::Semi) {
            let err_span = self.prev_end_span();
            self.diags.error(
                ErrorCode::E0005,
                err_span,
                format!("expected `;`, found {}", self.kind().description()),
            );
        }
    }

    pub(crate) fn expect_ident(&mut self, what: &str) -> Option<IdentSpan> {
        let tok = self.cur();
        match tok.kind {
            TokenKind::Ident => {
                self.bump();
                let sym = self.interner.intern(self.slice(tok.span));
                Some(IdentSpan {
                    sym,
                    span: tok.span,
                })
            }
            TokenKind::KwRaw | TokenKind::KwAsm | TokenKind::KwVolatile => {
                self.bump();
                let name = self.slice(tok.span);
                if self.profile == FileProfile::Safe {
                    self.diags.error(
                        ErrorCode::E0111,
                        tok.span,
                        format!("`{}` is reserved for `.hal.eon` modules", name),
                    );
                } else {
                    self.diags.error(
                        ErrorCode::E0005,
                        tok.span,
                        format!("expected {}, found reserved keyword `{}`", what, name),
                    );
                }
                None
            }
            TokenKind::Error => {
                self.bump();
                None
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    tok.span,
                    format!("expected {}, found {}", what, tok.kind.description()),
                );
                None
            }
        }
    }

    pub(crate) fn dummy_ident(&mut self, span: Span) -> IdentSpan {
        let sym = self.interner.intern("<error>");
        IdentSpan { sym, span }
    }

    pub(crate) fn expect_ident_or_dummy(&mut self, what: &str) -> IdentSpan {
        let sp = self.span();
        self.expect_ident(what)
            .unwrap_or_else(|| self.dummy_ident(sp))
    }

    pub(crate) fn sync_to_stmt(&mut self) {
        while !self.at_eof() {
            match self.kind() {
                TokenKind::Semi => {
                    self.bump();
                    return;
                }
                TokenKind::RBrace => return,
                TokenKind::KwFn
                | TokenKind::KwSys
                | TokenKind::KwStruct
                | TokenKind::KwPool
                | TokenKind::KwEnum
                | TokenKind::KwUnion
                | TokenKind::KwBitfield
                | TokenKind::KwIface
                | TokenKind::KwImpl
                | TokenKind::KwError
                | TokenKind::KwType
                | TokenKind::KwConst
                | TokenKind::KwPub
                | TokenKind::KwLet
                | TokenKind::KwVar
                | TokenKind::KwReturn
                | TokenKind::KwIf
                | TokenKind::KwMatch
                | TokenKind::KwWhile
                | TokenKind::KwFor
                | TokenKind::KwPar
                | TokenKind::KwLoop
                | TokenKind::KwDefer
                | TokenKind::KwErrdefer => return,
                _ => {
                    self.bump();
                }
            }
        }
    }

    pub(crate) fn sync_to_decl(&mut self) {
        while !self.at_eof() {
            match self.kind() {
                TokenKind::At
                | TokenKind::KwPub
                | TokenKind::KwFn
                | TokenKind::KwSys
                | TokenKind::KwStruct
                | TokenKind::KwLinear
                | TokenKind::KwPool
                | TokenKind::KwEnum
                | TokenKind::KwUnion
                | TokenKind::KwBitfield
                | TokenKind::KwIface
                | TokenKind::KwImpl
                | TokenKind::KwError
                | TokenKind::KwType
                | TokenKind::KwConst
                | TokenKind::KwExtern
                | TokenKind::KwComptime
                | TokenKind::KwView
                | TokenKind::RBrace => return,
                TokenKind::Semi => {
                    self.bump();
                    return;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }
}
