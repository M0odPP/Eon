use crate::ast::*;
use crate::diag::ErrorCode;
use crate::lexer::TokenKind;
use crate::parser::Parser;

impl<'a> Parser<'a> {
    pub(crate) fn is_base_type_start(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Question
                | TokenKind::Star
                | TokenKind::LBracket
                | TokenKind::KwDyn
                | TokenKind::KwFn
                | TokenKind::KwStruct
                | TokenKind::LParen
                | TokenKind::KwVoid
                | TokenKind::KwNever
                | TokenKind::KwType
                | TokenKind::KwSelfType
                | TokenKind::Ident
        )
    }

    pub(crate) fn parse_optional_generic_params(&mut self) -> Vec<GenericParam> {
        if !self.eat(TokenKind::LBracket) {
            return Vec::new();
        }
        let mut params = Vec::new();
        while !self.at(TokenKind::RBracket) && !self.at_eof() {
            let start = self.span();
            let is_comptime = self.eat(TokenKind::KwComptime);
            let name = self.expect_ident_or_dummy("generic parameter name");
            let mut end = name.span;
            let mut comptime_ty = None;
            let mut bounds = Vec::new();

            if is_comptime {
                self.expect(TokenKind::Colon);
                let ty = self.parse_type_expr();
                end = ty.span;
                comptime_ty = Some(ty);
            } else if self.eat(TokenKind::Colon) {
                loop {
                    let b = self.parse_type_expr();
                    end = b.span;
                    bounds.push(b);
                    if !self.eat(TokenKind::Plus) {
                        break;
                    }
                }
            }

            params.push(GenericParam {
                span: start.merge(end),
                is_comptime,
                name,
                comptime_ty,
                bounds,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket);
        params
    }

    pub(crate) fn parse_type_expr(&mut self) -> TypeExpr {
        if !self.enter_depth() {
            return TypeExpr {
                span: self.span(),
                kind: TypeKind::Error,
            };
        }

        let res = if self.at(TokenKind::Bang) {
            let bang = self.bump().span;
            let errors = self.parse_error_set_expr();
            TypeExpr {
                span: bang.merge(self.prev_end_span()),
                kind: TypeKind::ErrorUnion { ok: None, errors },
            }
        } else {
            let base = self.parse_base_type();
            if self.eat(TokenKind::Bang) {
                let start = base.span;
                let errors = self.parse_error_set_expr();
                TypeExpr {
                    span: start.merge(self.prev_end_span()),
                    kind: TypeKind::ErrorUnion {
                        ok: Some(Box::new(base)),
                        errors,
                    },
                }
            } else {
                base
            }
        };

        self.leave_depth();
        res
    }

    fn parse_error_set_expr(&mut self) -> Vec<TypeExpr> {
        let mut errs = Vec::new();
        loop {
            let start = self.span();
            let path = self.parse_type_path();
            errs.push(TypeExpr {
                span: start.merge(self.prev_end_span()),
                kind: TypeKind::Path(path),
            });
            if !self.eat(TokenKind::Pipe) {
                break;
            }
        }
        errs
    }

    pub(crate) fn parse_base_type(&mut self) -> TypeExpr {
        if !self.enter_depth() {
            return TypeExpr {
                span: self.span(),
                kind: TypeKind::Error,
            };
        }
        let start = self.span();
        let kind = match self.kind() {
            TokenKind::Question => {
                self.bump();
                TypeKind::Optional(Box::new(self.parse_base_type()))
            }
            TokenKind::Star => {
                self.bump();
                let is_mut = self.eat(TokenKind::KwMut);
                let target = self.parse_borrow_target();
                TypeKind::Borrow { is_mut, target }
            }
            TokenKind::LBracket => {
                let arr = self.parse_array_type();
                self.leave_depth();
                return arr;
            }
            TokenKind::KwDyn => {
                self.bump();
                TypeKind::Dyn(self.parse_type_path())
            }
            TokenKind::KwFn => {
                self.bump();
                self.expect(TokenKind::LParen);
                let mut params = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at_eof() {
                    params.push(self.parse_type_expr());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen);
                let ret = if self.eat(TokenKind::Arrow) {
                    Some(Box::new(self.parse_type_expr()))
                } else {
                    None
                };
                let effects = self.parse_attributes();
                TypeKind::FnPtr {
                    params,
                    ret,
                    effects,
                }
            }
            TokenKind::KwStruct => {
                self.bump();
                TypeKind::InlineStruct(self.parse_field_block())
            }
            TokenKind::LParen => {
                self.bump();
                let mut elems = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at_eof() {
                    elems.push(self.parse_type_expr());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen);
                TypeKind::Tuple(elems)
            }
            TokenKind::KwVoid => {
                self.bump();
                TypeKind::Void
            }
            TokenKind::KwNever => {
                self.bump();
                TypeKind::Never
            }
            TokenKind::KwType => {
                self.bump();
                TypeKind::TypeType
            }
            TokenKind::KwSelfType => {
                self.bump();
                TypeKind::SelfType
            }
            TokenKind::Ident => TypeKind::Path(self.parse_type_path()),
            TokenKind::Error => {
                self.bump();
                TypeKind::Error
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    format!("expected type, found {}", self.kind().description()),
                );
                TypeKind::Error
            }
        };

        self.leave_depth();
        TypeExpr {
            span: start.merge(self.prev_end_span()),
            kind,
        }
    }

    fn parse_array_type(&mut self) -> TypeExpr {
        let start = self.bump().span; // `[`
        let len = self.parse_expr();
        self.expect(TokenKind::RBracket);
        let elem = self.parse_base_type();
        let span = start.merge(elem.span);
        TypeExpr {
            span,
            kind: TypeKind::Array {
                len: Box::new(len),
                elem: Box::new(elem),
            },
        }
    }

    fn parse_borrow_target(&mut self) -> BorrowTarget {
        if self.at(TokenKind::Ident)
            && self.slice(self.span()) == "str"
            && !matches!(self.peek_kind(1), TokenKind::Dot | TokenKind::LBracket)
        {
            self.bump();
            return BorrowTarget::Str;
        }
        if self.at(TokenKind::LBracket) {
            if self.is_bracketed_followed_by_base_type() {
                BorrowTarget::Single(Box::new(self.parse_base_type()))
            } else {
                self.bump(); // `[`
                let elem = self.parse_type_expr();
                self.expect(TokenKind::RBracket);
                BorrowTarget::Slice(Box::new(elem))
            }
        } else {
            BorrowTarget::Single(Box::new(self.parse_base_type()))
        }
    }

    pub(crate) fn is_bracketed_followed_by_base_type(&self) -> bool {
        if !self.at(TokenKind::LBracket) {
            return false;
        }
        let mut i = self.pos + 1;
        let mut depth_sq = 1usize;
        while i < self.tokens.len() && depth_sq > 0 {
            match self.tokens[i].kind {
                TokenKind::LBracket => depth_sq += 1,
                TokenKind::RBracket => depth_sq -= 1,
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        if depth_sq != 0 {
            return false;
        }
        let after = self.tokens.get(i).map(|t| t.kind).unwrap_or(TokenKind::Eof);
        Self::is_base_type_start(after)
    }

    pub(crate) fn parse_type_path(&mut self) -> TypePath {
        let mut segments = vec![self.expect_ident_or_dummy("type name")];
        while self.at(TokenKind::Dot) && self.peek_kind(1) == TokenKind::Ident {
            self.bump(); // `.`
            segments.push(self.expect_ident_or_dummy("type segment"));
        }
        let generics = if self.at(TokenKind::LBracket) {
            self.parse_generic_args()
        } else {
            Vec::new()
        };
        TypePath { segments, generics }
    }

    pub(crate) fn parse_generic_args(&mut self) -> Vec<GenericArg> {
        self.expect(TokenKind::LBracket);
        let mut args = Vec::new();
        while !self.at(TokenKind::RBracket) && !self.at_eof() {
            args.push(self.parse_generic_arg());
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket);
        args
    }

    pub(crate) fn parse_generic_arg(&mut self) -> GenericArg {
        let start = self.span();
        match self.kind() {
            TokenKind::Question
            | TokenKind::Star
            | TokenKind::KwDyn
            | TokenKind::KwFn
            | TokenKind::KwStruct
            | TokenKind::KwVoid
            | TokenKind::KwNever
            | TokenKind::KwType
            | TokenKind::KwSelfType
            | TokenKind::Bang => {
                let ty = self.parse_type_expr();
                GenericArg {
                    span: ty.span,
                    kind: GenericArgKind::Type(ty),
                }
            }
            TokenKind::LBracket => {
                if self.is_bracketed_followed_by_base_type() {
                    let ty = self.parse_type_expr();
                    GenericArg {
                        span: ty.span,
                        kind: GenericArgKind::Type(ty),
                    }
                } else {
                    let e = self.parse_expr();
                    GenericArg {
                        span: e.span,
                        kind: GenericArgKind::Expr(e),
                    }
                }
            }
            TokenKind::LParen => {
                if self.looks_like_tuple_type_in_generic() {
                    let ty = self.parse_type_expr();
                    GenericArg {
                        span: ty.span,
                        kind: GenericArgKind::Type(ty),
                    }
                } else {
                    let e = self.parse_expr();
                    GenericArg {
                        span: e.span,
                        kind: GenericArgKind::Expr(e),
                    }
                }
            }
            TokenKind::Ident => {
                let mut look = self.pos + 1;
                while look + 1 < self.tokens.len()
                    && self.tokens[look].kind == TokenKind::Dot
                    && self.tokens[look + 1].kind == TokenKind::Ident
                {
                    look += 2;
                }
                let next_k = self
                    .tokens
                    .get(look)
                    .map(|t| t.kind)
                    .unwrap_or(TokenKind::Eof);
                match next_k {
                    TokenKind::LBracket | TokenKind::Bang => {
                        let ty = self.parse_type_expr();
                        GenericArg {
                            span: ty.span,
                            kind: GenericArgKind::Type(ty),
                        }
                    }
                    TokenKind::Comma | TokenKind::RBracket | TokenKind::RParen => {
                        let first = self.expect_ident_or_dummy("generic argument");
                        let mut end = first.span;
                        let mut segs = vec![first];
                        while self.at(TokenKind::Dot) && self.peek_kind(1) == TokenKind::Ident {
                            self.bump();
                            let seg = self.expect_ident_or_dummy("path segment");
                            end = seg.span;
                            segs.push(seg);
                        }
                        GenericArg {
                            span: start.merge(end),
                            kind: GenericArgKind::TypeOrConst(segs),
                        }
                    }
                    _ => {
                        let e = self.parse_expr();
                        GenericArg {
                            span: e.span,
                            kind: GenericArgKind::Expr(e),
                        }
                    }
                }
            }
            _ => {
                let e = self.parse_expr();
                GenericArg {
                    span: e.span,
                    kind: GenericArgKind::Expr(e),
                }
            }
        }
    }

    fn looks_like_tuple_type_in_generic(&self) -> bool {
        if self.peek_kind(1) == TokenKind::RParen {
            return true;
        }
        let mut i = self.pos + 1;
        let mut depth = 1usize;
        let mut has_comma_top = false;
        while i < self.tokens.len() && depth > 0 {
            match self.tokens[i].kind {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => depth -= 1,
                TokenKind::Comma if depth == 1 => has_comma_top = true,
                TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::EqEq
                | TokenKind::NotEq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::AndAnd
                | TokenKind::OrOr
                | TokenKind::Amp
                | TokenKind::Pipe
                | TokenKind::Caret
                | TokenKind::Shl
                | TokenKind::Shr
                    if depth == 1 =>
                {
                    return false;
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        has_comma_top
    }
}
