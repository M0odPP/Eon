use crate::ast::*;
use crate::diag::ErrorCode;
use crate::lexer::TokenKind;
use crate::parser::Parser;
use crate::span::{FileProfile, Span};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrecCat {
    Other,
    Bitwise,
    Arith,
    Cmp,
    LogAnd,
    LogOr,
}

impl<'a> Parser<'a> {
    pub(crate) fn parse_expr(&mut self) -> Expr {
        if !self.enter_depth() {
            return Expr {
                span: self.span(),
                kind: ExprKind::Error,
            };
        }
        let res = self.parse_catch_orelse_expr();
        self.leave_depth();
        res
    }

    fn parse_catch_orelse_expr(&mut self) -> Expr {
        let mut lhs = if self.at(TokenKind::KwTry) {
            let try_span = self.bump().span;
            let (inner, _) = self.parse_binary_expr_bp(0);
            let span = try_span.merge(inner.span);
            Expr {
                span,
                kind: ExprKind::Unary {
                    op: UnaryOp::Try,
                    expr: Box::new(inner),
                },
            }
        } else {
            self.parse_binary_expr_bp(0).0
        };

        loop {
            let lhs_span = lhs.span;
            if self.eat(TokenKind::KwOrelse) {
                let (rhs, end) = self.parse_catch_or_orelse_rhs(true);
                lhs = Expr {
                    span: lhs_span.merge(end),
                    kind: ExprKind::Orelse {
                        lhs: Box::new(lhs),
                        rhs,
                    },
                };
            } else if self.eat(TokenKind::KwCatch) {
                let err_bind = if self.eat(TokenKind::Pipe) {
                    let id = self.expect_ident_or_dummy("error binding");
                    self.expect(TokenKind::Pipe);
                    Some(id)
                } else {
                    None
                };
                let (rhs, end) = self.parse_catch_or_orelse_rhs(true);
                lhs = Expr {
                    span: lhs_span.merge(end),
                    kind: ExprKind::Catch {
                        lhs: Box::new(lhs),
                        err_bind,
                        rhs,
                    },
                };
            } else {
                break;
            }
        }

        lhs
    }

    fn parse_catch_or_orelse_rhs(&mut self, allow_block: bool) -> (CatchBody, Span) {
        if allow_block && self.at(TokenKind::LBrace) {
            let blk = self.parse_block(true);
            let sp = blk.span;
            return (CatchBody::Block(blk), sp);
        }
        let sp = self.span();
        match self.kind() {
            TokenKind::KwReturn => {
                let j = self.parse_return_action();
                (CatchBody::Jump(j), sp.merge(self.prev_end_span()))
            }
            TokenKind::KwBreak => {
                let j = self.parse_break_action();
                (CatchBody::Jump(j), sp.merge(self.prev_end_span()))
            }
            TokenKind::KwContinue => {
                let j = self.parse_continue_action();
                (CatchBody::Jump(j), sp.merge(self.prev_end_span()))
            }
            TokenKind::KwUnreachable => {
                self.bump();
                (CatchBody::Jump(JumpAction::Unreachable), sp)
            }
            _ => {
                let (e, _) = self.parse_binary_expr_bp(0);
                let esp = e.span;
                (CatchBody::Expr(Box::new(e)), esp)
            }
        }
    }

    pub(crate) fn parse_binary_expr_bp(&mut self, min_bp: u8) -> (Expr, PrecCat) {
        if !self.enter_depth() {
            return (
                Expr {
                    span: self.span(),
                    kind: ExprKind::Error,
                },
                PrecCat::Other,
            );
        }

        let (mut lhs, mut lhs_cat) = self.parse_unary_expr();

        loop {
            if self.at(TokenKind::KwIs) {
                let l_bp = 9u8;
                if l_bp < min_bp {
                    break;
                }
                let op_tok = self.bump();
                if lhs_cat == PrecCat::Bitwise {
                    self.diags.error(
                        ErrorCode::E0120,
                        op_tok.span,
                        "mixing bitwise operators with comparisons without parentheses is forbidden",
                    );
                } else if lhs_cat == PrecCat::Cmp {
                    self.diags.error(
                        ErrorCode::E0125,
                        op_tok.span,
                        "chained comparison operators are forbidden; use `&&` with explicit comparisons",
                    );
                }
                let pat = self.parse_pattern();
                let span = lhs.span.merge(pat.span);
                lhs = Expr {
                    span,
                    kind: ExprKind::Is {
                        expr: Box::new(lhs),
                        pat,
                    },
                };
                lhs_cat = PrecCat::Cmp;
                continue;
            }

            let Some((op, l_bp, r_bp, op_cat)) = infix_binding_power(self.kind()) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }

            let op_tok = self.bump();
            let (rhs, rhs_cat) = self.parse_binary_expr_bp(r_bp);

            self.check_operator_mixing(op_tok.span, op_cat, lhs_cat, rhs_cat);

            let span = lhs.span.merge(rhs.span);
            lhs = Expr {
                span,
                kind: ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
            lhs_cat = op_cat;
        }

        self.leave_depth();
        (lhs, lhs_cat)
    }

    fn check_operator_mixing(
        &mut self,
        op_span: Span,
        op_cat: PrecCat,
        lhs_cat: PrecCat,
        rhs_cat: PrecCat,
    ) {
        let mix_bitwise = (op_cat == PrecCat::Bitwise
            && (matches!(lhs_cat, PrecCat::Arith | PrecCat::Cmp)
                || matches!(rhs_cat, PrecCat::Arith | PrecCat::Cmp)))
            || (matches!(op_cat, PrecCat::Arith | PrecCat::Cmp)
                && (lhs_cat == PrecCat::Bitwise || rhs_cat == PrecCat::Bitwise));
        if mix_bitwise {
            self.diags.error(
                ErrorCode::E0120,
                op_span,
                "mixing bitwise operators (`&`, `|`, `^`, `<<`, `>>`) with arithmetic or comparisons without parentheses is forbidden",
            );
        }

        let mix_logic = (op_cat == PrecCat::LogAnd
            && (lhs_cat == PrecCat::LogOr || rhs_cat == PrecCat::LogOr))
            || (op_cat == PrecCat::LogOr
                && (lhs_cat == PrecCat::LogAnd || rhs_cat == PrecCat::LogAnd));
        if mix_logic {
            self.diags.error(
                ErrorCode::E0121,
                op_span,
                "mixing `&&` and `||` without parentheses is forbidden",
            );
        }

        if op_cat == PrecCat::Cmp && (lhs_cat == PrecCat::Cmp || rhs_cat == PrecCat::Cmp) {
            self.diags.error(
                ErrorCode::E0125,
                op_span,
                "chained comparison operators are forbidden; add explicit `&&`",
            );
        }
    }

    fn parse_unary_expr(&mut self) -> (Expr, PrecCat) {
        let start = self.span();
        let op = match self.kind() {
            TokenKind::Minus => {
                self.bump();
                Some(UnaryOp::Neg)
            }
            TokenKind::Bang => {
                self.bump();
                Some(UnaryOp::Not)
            }
            TokenKind::Tilde => {
                self.bump();
                Some(UnaryOp::BitNot)
            }
            TokenKind::Amp => {
                self.bump();
                Some(if self.eat(TokenKind::KwMut) {
                    UnaryOp::BorrowMut
                } else {
                    UnaryOp::Borrow
                })
            }
            TokenKind::KwMove => {
                self.bump();
                Some(UnaryOp::Move)
            }
            TokenKind::KwCopy => {
                self.bump();
                Some(UnaryOp::Copy)
            }
            TokenKind::KwTry => {
                let try_tok = self.bump();
                self.diags.error(
                    ErrorCode::E0005,
                    try_tok.span,
                    "`try` is only allowed at the beginning of an expression; bind it with `let` first",
                );
                Some(UnaryOp::Try)
            }
            _ => None,
        };

        if let Some(op) = op {
            let (inner, _) = self.parse_unary_expr();
            let span = start.merge(inner.span);
            (
                Expr {
                    span,
                    kind: ExprKind::Unary {
                        op,
                        expr: Box::new(inner),
                    },
                },
                PrecCat::Other,
            )
        } else {
            self.parse_postfix_expr()
        }
    }

    fn parse_postfix_expr(&mut self) -> (Expr, PrecCat) {
        let mut expr = self.parse_primary_expr();

        loop {
            let base_span = expr.span;
            match self.kind() {
                TokenKind::DotStar => {
                    let tok = self.bump();
                    expr = Expr {
                        span: base_span.merge(tok.span),
                        kind: ExprKind::Deref(Box::new(expr)),
                    };
                }
                TokenKind::Dot => {
                    self.bump();
                    match self.kind() {
                        TokenKind::LBracket => {
                            let args = self.parse_generic_args();
                            expr = Expr {
                                span: base_span.merge(self.prev_end_span()),
                                kind: ExprKind::GenericApply {
                                    base: Box::new(expr),
                                    args,
                                },
                            };
                        }
                        TokenKind::LBrace => {
                            let fields = self.parse_field_init_block();
                            expr = Expr {
                                span: base_span.merge(self.prev_end_span()),
                                kind: ExprKind::NamedStructLit {
                                    target: Box::new(expr),
                                    fields,
                                },
                            };
                        }
                        _ => {
                            let field = self.expect_ident_or_dummy("field or method name");
                            expr = Expr {
                                span: base_span.merge(field.span),
                                kind: ExprKind::Field {
                                    base: Box::new(expr),
                                    field,
                                },
                            };
                        }
                    }
                }
                TokenKind::LParen => {
                    self.bump();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) && !self.at_eof() {
                        args.push(self.parse_expr());
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    let end = self
                        .expect(TokenKind::RParen)
                        .map(|t| t.span)
                        .unwrap_or_else(|| self.prev_end_span());
                    expr = Expr {
                        span: base_span.merge(end),
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                    };
                }
                TokenKind::LBracket => {
                    self.bump();
                    if self.eat(TokenKind::DotDot) {
                        let hi = if !self.at(TokenKind::RBracket) {
                            Some(Box::new(self.parse_expr()))
                        } else {
                            None
                        };
                        let end = self
                            .expect(TokenKind::RBracket)
                            .map(|t| t.span)
                            .unwrap_or_else(|| self.prev_end_span());
                        expr = Expr {
                            span: base_span.merge(end),
                            kind: ExprKind::Slice {
                                base: Box::new(expr),
                                lo: None,
                                hi,
                            },
                        };
                    } else {
                        let first = self.parse_expr();
                        if self.eat(TokenKind::DotDot) {
                            let hi = if !self.at(TokenKind::RBracket) {
                                Some(Box::new(self.parse_expr()))
                            } else {
                                None
                            };
                            let end = self
                                .expect(TokenKind::RBracket)
                                .map(|t| t.span)
                                .unwrap_or_else(|| self.prev_end_span());
                            expr = Expr {
                                span: base_span.merge(end),
                                kind: ExprKind::Slice {
                                    base: Box::new(expr),
                                    lo: Some(Box::new(first)),
                                    hi,
                                },
                            };
                        } else {
                            let end = self
                                .expect(TokenKind::RBracket)
                                .map(|t| t.span)
                                .unwrap_or_else(|| self.prev_end_span());
                            expr = Expr {
                                span: base_span.merge(end),
                                kind: ExprKind::Index {
                                    base: Box::new(expr),
                                    index: Box::new(first),
                                },
                            };
                        }
                    }
                }
                _ => break,
            }
        }

        (expr, PrecCat::Other)
    }

    fn parse_primary_expr(&mut self) -> Expr {
        let tok = self.cur();
        let kind = match tok.kind {
            TokenKind::IntLit(id) => {
                self.bump();
                let raw = self.interner.intern(self.lexed.num_raw(self.src, id));
                let suffix = self
                    .lexed
                    .num_suffix(self.src, id)
                    .map(|s| self.interner.intern(s));
                ExprKind::IntLit { raw, suffix }
            }
            TokenKind::FloatLit(id) => {
                self.bump();
                let raw = self.interner.intern(self.lexed.num_raw(self.src, id));
                let suffix = self
                    .lexed
                    .num_suffix(self.src, id)
                    .map(|s| self.interner.intern(s));
                ExprKind::FloatLit { raw, suffix }
            }
            TokenKind::StringLit(id) => {
                self.bump();
                ExprKind::StringLit(self.lexed.string_lits[id.0 as usize].clone())
            }
            TokenKind::RawStringLit(id) => {
                self.bump();
                ExprKind::RawStringLit(self.lexed.string_lits[id.0 as usize].clone())
            }
            TokenKind::ByteStringLit(id) => {
                self.bump();
                ExprKind::ByteStringLit(self.lexed.byte_string_lits[id.0 as usize].clone())
            }
            TokenKind::CharLit(ch) => {
                self.bump();
                ExprKind::CharLit(ch)
            }
            TokenKind::ByteCharLit(b) => {
                self.bump();
                ExprKind::ByteCharLit(b)
            }
            TokenKind::KwTrue => {
                self.bump();
                ExprKind::BoolLit(true)
            }
            TokenKind::KwFalse => {
                self.bump();
                ExprKind::BoolLit(false)
            }
            TokenKind::KwNull => {
                self.bump();
                ExprKind::Null
            }
            TokenKind::KwSelf => {
                self.bump();
                ExprKind::SelfVal
            }
            TokenKind::KwReturn if self.in_ensures_or_where => {
                self.bump();
                if !self.allow_return_as_expr {
                    self.diags.error(
                        ErrorCode::E0126,
                        tok.span,
                        "`return` in a constraint is only allowed inside `ensures`, not `where`",
                    );
                }
                ExprKind::ReturnVal
            }
            TokenKind::Ident => {
                self.bump();
                ExprKind::Ident(self.interner.intern(self.slice(tok.span)))
            }
            TokenKind::Dot => {
                self.bump();
                if self.at(TokenKind::LBrace) {
                    ExprKind::AnonStructLit(self.parse_field_init_block())
                } else {
                    ExprKind::EnumLit(self.expect_ident_or_dummy("variant name or `{`"))
                }
            }
            TokenKind::LBracket => {
                self.bump();
                if self.eat(TokenKind::RBracket) {
                    ExprKind::ArrayLit(Vec::new())
                } else {
                    let first = self.parse_expr();
                    if self.eat(TokenKind::Semi) {
                        let count = self.parse_expr();
                        self.expect(TokenKind::RBracket);
                        ExprKind::ArrayRepeat {
                            val: Box::new(first),
                            count: Box::new(count),
                        }
                    } else {
                        let mut elems = vec![first];
                        while self.eat(TokenKind::Comma) {
                            if self.at(TokenKind::RBracket) {
                                break;
                            }
                            elems.push(self.parse_expr());
                        }
                        self.expect(TokenKind::RBracket);
                        ExprKind::ArrayLit(elems)
                    }
                }
            }
            TokenKind::LParen => {
                self.bump();
                if self.eat(TokenKind::RParen) {
                    ExprKind::Tuple(Vec::new())
                } else {
                    let first = self.parse_expr();
                    if self.eat(TokenKind::Comma) {
                        let mut elems = vec![first];
                        while !self.at(TokenKind::RParen) && !self.at_eof() {
                            elems.push(self.parse_expr());
                            if !self.eat(TokenKind::Comma) {
                                break;
                            }
                        }
                        self.expect(TokenKind::RParen);
                        ExprKind::Tuple(elems)
                    } else {
                        self.expect(TokenKind::RParen);
                        ExprKind::Paren(Box::new(first))
                    }
                }
            }
            TokenKind::Pipe => {
                self.bump();
                let mut params = Vec::new();
                while !self.at(TokenKind::Pipe) && !self.at_eof() {
                    let p_start = self.span();
                    let name = self.expect_ident_or_dummy("closure parameter");
                    let ty = if self.eat(TokenKind::Colon) {
                        self.parse_type_expr()
                    } else {
                        TypeExpr {
                            span: name.span,
                            kind: TypeKind::Inferred,
                        }
                    };
                    params.push(Param {
                        span: p_start.merge(self.prev_end_span()),
                        attrs: Vec::new(),
                        is_comptime: false,
                        modifier: None,
                        name,
                        is_self: false,
                        access: None,
                        ty,
                    });
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::Pipe);
                let body = self.parse_control_flow_block("closure", true);
                ExprKind::Closure { params, body }
            }
            TokenKind::OrOr => {
                self.bump();
                let body = self.parse_control_flow_block("closure", true);
                ExprKind::Closure {
                    params: Vec::new(),
                    body,
                }
            }
            TokenKind::KwIf => {
                let data = self.parse_if_data(true);
                if data.else_branch.is_none() {
                    self.diags.error(
                        ErrorCode::E0005,
                        data.span,
                        "`if` expression must have an `else` branch",
                    );
                }
                ExprKind::If {
                    is_comptime: false,
                    data: Box::new(data),
                }
            }
            TokenKind::KwMatch => {
                let (scrutinee, arms, _) = self.parse_match_parts();
                ExprKind::Match {
                    scrutinee: Box::new(scrutinee),
                    arms,
                }
            }
            TokenKind::KwLoop => {
                self.bump();
                let body = self.parse_control_flow_block("loop", false);
                ExprKind::Loop { label: None, body }
            }
            TokenKind::Label => {
                let label_tok = self.bump();
                let sym = self.interner.intern(self.slice(label_tok.span));
                let label = Some(IdentSpan {
                    sym,
                    span: label_tok.span,
                });
                self.expect(TokenKind::Colon);
                self.expect(TokenKind::KwLoop);
                let body = self.parse_control_flow_block("loop", false);
                ExprKind::Loop { label, body }
            }
            TokenKind::KwComptime => {
                self.bump();
                if self.at(TokenKind::KwIf) {
                    let data = self.parse_if_data(true);
                    if data.else_branch.is_none() {
                        self.diags.error(
                            ErrorCode::E0005,
                            data.span,
                            "`comptime if` expression must have an `else` branch",
                        );
                    }
                    ExprKind::If {
                        is_comptime: true,
                        data: Box::new(data),
                    }
                } else if self.at(TokenKind::LBrace) {
                    ExprKind::ComptimeBlock(self.parse_block(true))
                } else {
                    ExprKind::ComptimeExpr(Box::new(self.parse_expr()))
                }
            }
            TokenKind::At => {
                self.bump();
                let name = self.expect_ident_or_dummy("builtin name");
                let is_value_builtin = matches!(
                    self.interner.get(name.sym),
                    "sqrt" | "fma" | "lzcnt" | "tzcnt" | "popcnt" | "bswap"
                );
                let mut args = Vec::new();
                if self.eat(TokenKind::LParen) {
                    while !self.at(TokenKind::RParen) && !self.at_eof() {
                        if is_value_builtin {
                            let e = self.parse_expr();
                            args.push(GenericArg {
                                span: e.span,
                                kind: GenericArgKind::Expr(e),
                            });
                        } else {
                            args.push(self.parse_generic_arg());
                        }
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen);
                }
                ExprKind::Builtin { name, args }
            }
            TokenKind::KwAsm => {
                self.bump();
                if self.profile == FileProfile::Safe {
                    self.diags.error(
                        ErrorCode::E0111,
                        tok.span,
                        "ZM-11: `asm(...)` is forbidden outside `.hal.eon` modules",
                    );
                }
                self.expect(TokenKind::LParen);
                let template = match self.kind() {
                    TokenKind::StringLit(sid) => {
                        self.bump();
                        self.lexed.string_lits[sid.0 as usize].clone()
                    }
                    _ => {
                        self.diags.error(
                            ErrorCode::E0005,
                            self.span(),
                            "expected assembly template string literal after `asm(`",
                        );
                        String::new()
                    }
                };
                let mut operands = Vec::new();
                while self.eat(TokenKind::Comma) {
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    let op_start = self.span();
                    let kind = if self.at(TokenKind::Ident)
                        && self.peek_kind(1) == TokenKind::LParen
                    {
                        let dir = self.expect_ident_or_dummy("asm operand direction");
                        self.bump();
                        let constraint = self.parse_expr();
                        self.expect(TokenKind::RParen);
                        let arg = if !matches!(
                            self.kind(),
                            TokenKind::Comma | TokenKind::RParen | TokenKind::Eof
                        ) {
                            Some(self.parse_expr())
                        } else {
                            None
                        };
                        AsmOperandKind::Named {
                            dir,
                            constraint,
                            arg,
                        }
                    } else {
                        AsmOperandKind::Expr(self.parse_expr())
                    };
                    operands.push(AsmOperand {
                        span: op_start.merge(self.prev_end_span()),
                        kind,
                    });
                }
                self.expect(TokenKind::RParen);
                ExprKind::Asm { template, operands }
            }
            TokenKind::KwRaw | TokenKind::KwVolatile => {
                self.bump();
                let name = self.slice(tok.span);
                if self.profile == FileProfile::Safe {
                    self.diags.error(
                        ErrorCode::E0111,
                        tok.span,
                        format!("ZM-11: `{}` is reserved for `.hal.eon` modules", name),
                    );
                } else {
                    self.diags.error(
                        ErrorCode::E0005,
                        tok.span,
                        format!("unexpected `{}` in expression", name),
                    );
                }
                ExprKind::Error
            }
            TokenKind::Error => {
                self.bump();
                ExprKind::Error
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    tok.span,
                    format!("expected expression, found {}", tok.kind.description()),
                );
                if !self.at_eof()
                    && !matches!(
                        tok.kind,
                        TokenKind::Semi
                            | TokenKind::Comma
                            | TokenKind::RParen
                            | TokenKind::RBracket
                            | TokenKind::RBrace
                    )
                {
                    self.bump();
                }
                ExprKind::Error
            }
        };

        Expr {
            span: tok.span.merge(self.prev_end_span()),
            kind,
        }
    }

    fn parse_field_init_block(&mut self) -> Vec<FieldInit> {
        self.expect(TokenKind::LBrace);
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let f_start = self.span();
            let (name, value) =
                if self.at(TokenKind::Ident) && self.peek_kind(1) == TokenKind::Colon {
                    let n = self.expect_ident_or_dummy("field name");
                    self.bump();
                    let v = self.parse_expr();
                    (Some(n), v)
                } else {
                    (None, self.parse_expr())
                };
            let vspan = value.span;
            fields.push(FieldInit {
                span: f_start.merge(vspan),
                name,
                value,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace);
        fields
    }
}

fn infix_binding_power(kind: TokenKind) -> Option<(BinaryOp, u8, u8, PrecCat)> {
    Some(match kind {
        TokenKind::Star => (BinaryOp::Mul, 21, 22, PrecCat::Arith),
        TokenKind::Slash => (BinaryOp::Div, 21, 22, PrecCat::Arith),
        TokenKind::Percent => (BinaryOp::Mod, 21, 22, PrecCat::Arith),
        TokenKind::StarWrap => (BinaryOp::MulWrap, 21, 22, PrecCat::Arith),
        TokenKind::StarSat => (BinaryOp::MulSat, 21, 22, PrecCat::Arith),
        TokenKind::StarCheck => (BinaryOp::MulCheck, 21, 22, PrecCat::Arith),
        TokenKind::SlashCheck => (BinaryOp::DivCheck, 21, 22, PrecCat::Arith),
        TokenKind::PercentCheck => (BinaryOp::ModCheck, 21, 22, PrecCat::Arith),

        TokenKind::Plus => (BinaryOp::Add, 19, 20, PrecCat::Arith),
        TokenKind::Minus => (BinaryOp::Sub, 19, 20, PrecCat::Arith),
        TokenKind::PlusWrap => (BinaryOp::AddWrap, 19, 20, PrecCat::Arith),
        TokenKind::MinusWrap => (BinaryOp::SubWrap, 19, 20, PrecCat::Arith),
        TokenKind::PlusSat => (BinaryOp::AddSat, 19, 20, PrecCat::Arith),
        TokenKind::MinusSat => (BinaryOp::SubSat, 19, 20, PrecCat::Arith),
        TokenKind::PlusCheck => (BinaryOp::AddCheck, 19, 20, PrecCat::Arith),
        TokenKind::MinusCheck => (BinaryOp::SubCheck, 19, 20, PrecCat::Arith),

        TokenKind::Shl => (BinaryOp::Shl, 17, 18, PrecCat::Bitwise),
        TokenKind::Shr => (BinaryOp::Shr, 17, 18, PrecCat::Bitwise),
        TokenKind::ShlWrap => (BinaryOp::ShlWrap, 17, 18, PrecCat::Bitwise),

        TokenKind::Amp => (BinaryOp::BitAnd, 15, 16, PrecCat::Bitwise),
        TokenKind::Caret => (BinaryOp::BitXor, 13, 14, PrecCat::Bitwise),
        TokenKind::Pipe => (BinaryOp::BitOr, 11, 12, PrecCat::Bitwise),

        TokenKind::EqEq => (BinaryOp::Eq, 9, 10, PrecCat::Cmp),
        TokenKind::NotEq => (BinaryOp::NotEq, 9, 10, PrecCat::Cmp),
        TokenKind::Lt => (BinaryOp::Lt, 9, 10, PrecCat::Cmp),
        TokenKind::LtEq => (BinaryOp::LtEq, 9, 10, PrecCat::Cmp),
        TokenKind::Gt => (BinaryOp::Gt, 9, 10, PrecCat::Cmp),
        TokenKind::GtEq => (BinaryOp::GtEq, 9, 10, PrecCat::Cmp),

        TokenKind::AndAnd => (BinaryOp::And, 7, 8, PrecCat::LogAnd),
        TokenKind::OrOr => (BinaryOp::Or, 5, 6, PrecCat::LogOr),

        _ => return None,
    })
}
