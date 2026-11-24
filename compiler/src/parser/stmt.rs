use crate::ast::*;
use crate::diag::ErrorCode;
use crate::lexer::TokenKind;
use crate::parser::Parser;
use crate::span::{FileProfile, Span, Symbol};

impl<'a> Parser<'a> {
    pub(crate) fn parse_block(&mut self, allow_tail_expr: bool) -> Block {
        if !self.enter_depth() {
            return Block {
                span: self.span(),
                stmts: Vec::new(),
                tail_expr: None,
            };
        }
        let start = self.span();
        if !self.eat(TokenKind::LBrace) {
            self.diags.error(
                ErrorCode::E0005,
                start,
                format!("expected `{{`, found {}", self.kind().description()),
            );
            self.leave_depth();
            return Block {
                span: start,
                stmts: Vec::new(),
                tail_expr: None,
            };
        }
        let block = self.parse_block_body_after_lbrace(start, allow_tail_expr);
        self.leave_depth();
        block
    }

    pub(crate) fn parse_control_flow_block(
        &mut self,
        construct: &str,
        allow_tail_expr: bool,
    ) -> Block {
        if !self.enter_depth() {
            return Block {
                span: self.span(),
                stmts: Vec::new(),
                tail_expr: None,
            };
        }
        if !self.at(TokenKind::LBrace) {
            let err_span = self.span();
            self.diags.error(
                ErrorCode::E0117,
                err_span,
                format!("ZM-17: missing braces `{{}}` around `{}` body", construct),
            );
            let mut stmts = Vec::new();
            if !self.at(TokenKind::RBrace)
                && !self.at_eof()
                && let Some(s) = self.parse_stmt_in_recovery()
            {
                stmts.push(s);
            }
            self.leave_depth();
            return Block {
                span: err_span.merge(self.prev_end_span()),
                stmts,
                tail_expr: None,
            };
        }
        let start = self.bump().span;
        let block = self.parse_block_body_after_lbrace(start, allow_tail_expr);
        self.leave_depth();
        block
    }

    fn parse_block_body_after_lbrace(&mut self, start: Span, allow_tail_expr: bool) -> Block {
        let mut stmts = Vec::new();
        let mut tail_expr = None;

        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            match self.kind() {
                TokenKind::At if self.starts_stmt_attrs() => {
                    let attrs = self.parse_attributes();
                    match self.kind() {
                        TokenKind::KwLet => stmts.push(self.parse_let_stmt(attrs)),
                        TokenKind::KwVar => stmts.push(self.parse_var_stmt(attrs)),
                        _ => unreachable!(),
                    }
                }
                TokenKind::Underscore => {
                    let u_span = self.bump().span;
                    self.expect(TokenKind::Eq);
                    let rhs = self.parse_expr();
                    self.expect_semi();
                    stmts.push(Stmt {
                        span: u_span.merge(self.prev_end_span()),
                        kind: StmtKind::Discard(rhs),
                    });
                }
                TokenKind::KwLet => stmts.push(self.parse_let_stmt(Vec::new())),
                TokenKind::KwVar => stmts.push(self.parse_var_stmt(Vec::new())),
                TokenKind::KwConst => {
                    let c_span = self.span();
                    let c = self.parse_const_decl();
                    stmts.push(Stmt {
                        span: c_span.merge(self.prev_end_span()),
                        kind: StmtKind::Const(c),
                    });
                }
                TokenKind::KwDefer => stmts.push(self.parse_defer_stmt()),
                TokenKind::KwErrdefer => stmts.push(self.parse_errdefer_stmt()),
                TokenKind::KwReturn => {
                    let sp = self.span();
                    let j = self.parse_return_action();
                    self.expect_semi();
                    stmts.push(Stmt {
                        span: sp.merge(self.prev_end_span()),
                        kind: StmtKind::Jump(j),
                    });
                }
                TokenKind::KwBreak => {
                    let sp = self.span();
                    let j = self.parse_break_action();
                    self.expect_semi();
                    stmts.push(Stmt {
                        span: sp.merge(self.prev_end_span()),
                        kind: StmtKind::Jump(j),
                    });
                }
                TokenKind::KwContinue => {
                    let sp = self.span();
                    let j = self.parse_continue_action();
                    self.expect_semi();
                    stmts.push(Stmt {
                        span: sp.merge(self.prev_end_span()),
                        kind: StmtKind::Jump(j),
                    });
                }
                TokenKind::KwUnreachable => {
                    let sp = self.bump().span;
                    self.expect_semi();
                    stmts.push(Stmt {
                        span: sp.merge(self.prev_end_span()),
                        kind: StmtKind::Jump(JumpAction::Unreachable),
                    });
                }
                TokenKind::Label => {
                    let label_tok = self.bump();
                    let sym = self.interner.intern(self.slice(label_tok.span));
                    let label = Some(IdentSpan {
                        sym,
                        span: label_tok.span,
                    });
                    self.expect(TokenKind::Colon);
                    match self.kind() {
                        TokenKind::KwWhile => stmts.push(self.parse_while_stmt(label)),
                        TokenKind::KwFor => stmts.push(self.parse_for_stmt(label)),
                        TokenKind::KwLoop => stmts.push(self.parse_loop_stmt(label)),
                        _ => {
                            self.diags.error(
                                ErrorCode::E0005,
                                self.span(),
                                "loop label must precede `while`, `for`, or `loop`",
                            );
                            self.sync_to_stmt();
                        }
                    }
                }
                TokenKind::KwWhile => stmts.push(self.parse_while_stmt(None)),
                TokenKind::KwFor => stmts.push(self.parse_for_stmt(None)),
                TokenKind::KwPar => stmts.push(self.parse_par_for_stmt()),
                TokenKind::KwLoop => stmts.push(self.parse_loop_stmt(None)),
                TokenKind::KwRaw => stmts.push(self.parse_raw_stmt()),
                TokenKind::KwIf => {
                    let sp = self.span();
                    let data = self.parse_if_data(allow_tail_expr);
                    let end = data.span;
                    if allow_tail_expr
                        && self.at(TokenKind::RBrace)
                        && data.else_branch.is_some()
                    {
                        tail_expr = Some(Box::new(Expr {
                            span: sp.merge(end),
                            kind: ExprKind::If {
                                is_comptime: false,
                                data: Box::new(data),
                            },
                        }));
                        break;
                    }
                    stmts.push(Stmt {
                        span: sp.merge(end),
                        kind: StmtKind::If {
                            is_comptime: false,
                            data,
                        },
                    });
                }
                TokenKind::KwMatch => {
                    let sp = self.span();
                    let (scrutinee, arms, end) = self.parse_match_parts();
                    if allow_tail_expr && self.at(TokenKind::RBrace) {
                        tail_expr = Some(Box::new(Expr {
                            span: sp.merge(end),
                            kind: ExprKind::Match {
                                scrutinee: Box::new(scrutinee),
                                arms,
                            },
                        }));
                        break;
                    }
                    stmts.push(Stmt {
                        span: sp.merge(end),
                        kind: StmtKind::Match { scrutinee, arms },
                    });
                }
                TokenKind::KwComptime => {
                    let sp = self.bump().span;
                    if self.at(TokenKind::KwIf) {
                        let data = self.parse_if_data(allow_tail_expr);
                        let end = data.span;
                        if allow_tail_expr
                            && self.at(TokenKind::RBrace)
                            && data.else_branch.is_some()
                        {
                            tail_expr = Some(Box::new(Expr {
                                span: sp.merge(end),
                                kind: ExprKind::If {
                                    is_comptime: true,
                                    data: Box::new(data),
                                },
                            }));
                            break;
                        }
                        stmts.push(Stmt {
                            span: sp.merge(end),
                            kind: StmtKind::If {
                                is_comptime: true,
                                data,
                            },
                        });
                    } else if self.at(TokenKind::LBrace) {
                        let blk = self.parse_block(true);
                        let end = blk.span;
                        if allow_tail_expr && self.at(TokenKind::RBrace) {
                            tail_expr = Some(Box::new(Expr {
                                span: sp.merge(end),
                                kind: ExprKind::ComptimeBlock(blk),
                            }));
                            break;
                        }
                        stmts.push(Stmt {
                            span: sp.merge(end),
                            kind: StmtKind::ComptimeBlock(blk),
                        });
                    } else {
                        let inner = self.parse_expr();
                        let span = sp.merge(inner.span);
                        let ce = Expr {
                            span,
                            kind: ExprKind::ComptimeExpr(Box::new(inner)),
                        };
                        if self.finish_expr_stmt_or_tail(
                            ce,
                            allow_tail_expr,
                            &mut stmts,
                            &mut tail_expr,
                        ) {
                            break;
                        }
                    }
                }
                _ => {
                    let lhs = self.parse_expr();
                    if self.finish_expr_stmt_or_tail(
                        lhs,
                        allow_tail_expr,
                        &mut stmts,
                        &mut tail_expr,
                    ) {
                        break;
                    }
                }
            }

            if self.pos == prev {
                self.bump();
            }
        }

        let end = self
            .expect(TokenKind::RBrace)
            .map(|t| t.span)
            .unwrap_or_else(|| self.prev_end_span());
        Block {
            span: start.merge(end),
            stmts,
            tail_expr,
        }
    }

    fn starts_stmt_attrs(&self) -> bool {
        let mut i = self.pos;
        while i < self.tokens.len() && self.tokens[i].kind == TokenKind::At {
            i += 1;
            if i >= self.tokens.len() {
                return false;
            }
            i += 1;
            if i < self.tokens.len() && self.tokens[i].kind == TokenKind::LParen {
                i += 1;
                let mut depth = 1usize;
                while i < self.tokens.len() && depth > 0 {
                    match self.tokens[i].kind {
                        TokenKind::LParen => depth += 1,
                        TokenKind::RParen => depth -= 1,
                        TokenKind::Eof => return false,
                        _ => {}
                    }
                    i += 1;
                }
            }
        }
        matches!(
            self.tokens.get(i).map(|t| t.kind),
            Some(TokenKind::KwLet | TokenKind::KwVar)
        )
    }

    fn finish_expr_stmt_or_tail(
        &mut self,
        lhs: Expr,
        allow_tail_expr: bool,
        stmts: &mut Vec<Stmt>,
        tail_expr: &mut Option<Box<Expr>>,
    ) -> bool {
        let lhs_span = lhs.span;
        if let Some(op) = self.peek_assign_op() {
            self.bump();
            let rhs = self.parse_expr();
            let mut end = rhs.span;
            if self.peek_assign_op().is_some() {
                self.diags.error(
                    ErrorCode::E0125,
                    self.span(),
                    "assignment is a statement, not an expression; chained assignments are forbidden",
                );
                while self.peek_assign_op().is_some() {
                    self.bump();
                    let extra = self.parse_expr();
                    end = extra.span;
                }
            }
            self.expect_semi();
            stmts.push(Stmt {
                span: lhs_span.merge(end),
                kind: StmtKind::Assign { lhs, op, rhs },
            });
            return false;
        }

        if self.eat(TokenKind::Semi) {
            stmts.push(Stmt {
                span: lhs_span.merge(self.prev_end_span()),
                kind: StmtKind::Expr(lhs),
            });
            return false;
        }

        if self.at(TokenKind::RBrace) {
            if allow_tail_expr {
                *tail_expr = Some(Box::new(lhs));
                return true;
            } else {
                self.diags.error(
                    ErrorCode::E0124,
                    lhs_span,
                    "trailing expression without `;` is not allowed in function body; use explicit `return ...;` or add `;`",
                );
                stmts.push(Stmt {
                    span: lhs_span,
                    kind: StmtKind::Expr(lhs),
                });
                return false;
            }
        }

        self.expect_semi();
        stmts.push(Stmt {
            span: lhs_span,
            kind: StmtKind::Expr(lhs),
        });
        false
    }

    fn parse_stmt_in_recovery(&mut self) -> Option<Stmt> {
        let sp = self.span();
        match self.kind() {
            TokenKind::KwReturn => {
                let j = self.parse_return_action();
                self.expect_semi();
                Some(Stmt {
                    span: sp.merge(self.prev_end_span()),
                    kind: StmtKind::Jump(j),
                })
            }
            TokenKind::KwBreak => {
                let j = self.parse_break_action();
                self.expect_semi();
                Some(Stmt {
                    span: sp.merge(self.prev_end_span()),
                    kind: StmtKind::Jump(j),
                })
            }
            TokenKind::KwContinue => {
                let j = self.parse_continue_action();
                self.expect_semi();
                Some(Stmt {
                    span: sp.merge(self.prev_end_span()),
                    kind: StmtKind::Jump(j),
                })
            }
            TokenKind::Underscore => {
                self.bump();
                self.expect(TokenKind::Eq);
                let rhs = self.parse_expr();
                self.expect_semi();
                Some(Stmt {
                    span: sp.merge(self.prev_end_span()),
                    kind: StmtKind::Discard(rhs),
                })
            }
            _ => {
                let e = self.parse_expr();
                self.eat(TokenKind::Semi);
                Some(Stmt {
                    span: sp.merge(self.prev_end_span()),
                    kind: StmtKind::Expr(e),
                })
            }
        }
    }

    fn peek_assign_op(&self) -> Option<AssignOp> {
        Some(match self.kind() {
            TokenKind::Eq => AssignOp::Assign,
            TokenKind::PlusEq => AssignOp::AddAssign,
            TokenKind::MinusEq => AssignOp::SubAssign,
            TokenKind::StarEq => AssignOp::MulAssign,
            TokenKind::PlusWrapEq => AssignOp::AddWrapAssign,
            TokenKind::MinusWrapEq => AssignOp::SubWrapAssign,
            TokenKind::StarWrapEq => AssignOp::MulWrapAssign,
            TokenKind::PlusSatEq => AssignOp::AddSatAssign,
            TokenKind::MinusSatEq => AssignOp::SubSatAssign,
            TokenKind::StarSatEq => AssignOp::MulSatAssign,
            TokenKind::AmpEq => AssignOp::BitAndAssign,
            TokenKind::PipeEq => AssignOp::BitOrAssign,
            TokenKind::CaretEq => AssignOp::BitXorAssign,
            TokenKind::ShlEq => AssignOp::ShlAssign,
            TokenKind::ShrEq => AssignOp::ShrAssign,
            _ => return None,
        })
    }

    fn parse_let_stmt(&mut self, attrs: Vec<Attr>) -> Stmt {
        let start = self.bump().span;
        let pat = self.parse_pattern();
        let ty = if self.eat(TokenKind::Colon) {
            Some(self.parse_type_expr())
        } else {
            None
        };
        self.expect(TokenKind::Eq);
        let init = self.parse_expr();
        self.expect_semi();
        Stmt {
            span: start.merge(self.prev_end_span()),
            kind: StmtKind::Let {
                attrs,
                pat,
                ty,
                init,
            },
        }
    }

    fn parse_var_stmt(&mut self, attrs: Vec<Attr>) -> Stmt {
        let start = self.bump().span;
        let name = self.expect_ident_or_dummy("variable name");
        let ty = if self.eat(TokenKind::Colon) {
            Some(self.parse_type_expr())
        } else {
            None
        };
        self.expect(TokenKind::Eq);
        let init = self.parse_expr();
        self.expect_semi();
        Stmt {
            span: start.merge(self.prev_end_span()),
            kind: StmtKind::Var {
                attrs,
                name,
                ty,
                init,
            },
        }
    }

    fn parse_defer_stmt(&mut self) -> Stmt {
        let start = self.bump().span;
        let body = if self.at(TokenKind::LBrace) {
            DeferBody::Block(self.parse_block(false))
        } else {
            let e = self.parse_expr();
            self.expect_semi();
            DeferBody::Expr(e)
        };
        Stmt {
            span: start.merge(self.prev_end_span()),
            kind: StmtKind::Defer(body),
        }
    }

    fn parse_errdefer_stmt(&mut self) -> Stmt {
        let start = self.bump().span;
        let err_bind = if self.eat(TokenKind::Pipe) {
            let id = self.expect_ident_or_dummy("error binding");
            self.expect(TokenKind::Pipe);
            Some(id)
        } else {
            None
        };
        let body = if self.at(TokenKind::LBrace) {
            DeferBody::Block(self.parse_block(false))
        } else {
            let e = self.parse_expr();
            self.expect_semi();
            DeferBody::Expr(e)
        };
        Stmt {
            span: start.merge(self.prev_end_span()),
            kind: StmtKind::Errdefer { err_bind, body },
        }
    }

    pub(crate) fn parse_return_action(&mut self) -> JumpAction {
        self.bump();
        let val = if !matches!(
            self.kind(),
            TokenKind::Semi | TokenKind::Comma | TokenKind::RBrace | TokenKind::Eof
        ) {
            Some(Box::new(self.parse_expr()))
        } else {
            None
        };
        JumpAction::Return(val)
    }

    pub(crate) fn parse_break_action(&mut self) -> JumpAction {
        self.bump();
        let label = if self.at(TokenKind::Label) {
            let tok = self.bump();
            let sym = self.interner.intern(self.slice(tok.span));
            Some(IdentSpan {
                sym,
                span: tok.span,
            })
        } else {
            None
        };
        let value = if !matches!(
            self.kind(),
            TokenKind::Semi | TokenKind::Comma | TokenKind::RBrace | TokenKind::Eof
        ) {
            Some(Box::new(self.parse_expr()))
        } else {
            None
        };
        JumpAction::Break { label, value }
    }

    pub(crate) fn parse_continue_action(&mut self) -> JumpAction {
        self.bump();
        let label = if self.at(TokenKind::Label) {
            let tok = self.bump();
            let sym = self.interner.intern(self.slice(tok.span));
            Some(IdentSpan {
                sym,
                span: tok.span,
            })
        } else {
            None
        };
        JumpAction::Continue { label }
    }

    pub(crate) fn parse_if_data(&mut self, allow_tail_expr: bool) -> IfExpr {
        if !self.enter_depth() {
            let sp = self.span();
            return IfExpr {
                span: sp,
                cond: IfCond::Expr(Box::new(Expr {
                    span: sp,
                    kind: ExprKind::Error,
                })),
                then_block: Block {
                    span: sp,
                    stmts: Vec::new(),
                    tail_expr: None,
                },
                else_branch: None,
            };
        }

        let start = self.bump().span; // `if`
        let cond = if self.eat(TokenKind::KwLet) {
            let pat = self.parse_pattern();
            self.expect(TokenKind::Eq);
            let expr = self.parse_expr();
            IfCond::Let {
                pat,
                expr: Box::new(expr),
            }
        } else {
            IfCond::Expr(Box::new(self.parse_expr()))
        };
        let then_block = self.parse_control_flow_block("if", allow_tail_expr);
        let mut end = then_block.span;

        let else_branch = if self.eat(TokenKind::KwElse) {
            if self.at(TokenKind::KwIf) {
                let nested = self.parse_if_data(allow_tail_expr);
                end = nested.span;
                Some(ElseBranch::If {
                    is_comptime: false,
                    data: Box::new(nested),
                })
            } else if self.at(TokenKind::KwComptime) && self.peek_kind(1) == TokenKind::KwIf {
                self.bump();
                let nested = self.parse_if_data(allow_tail_expr);
                end = nested.span;
                Some(ElseBranch::If {
                    is_comptime: true,
                    data: Box::new(nested),
                })
            } else {
                let blk = self.parse_control_flow_block("else", allow_tail_expr);
                end = blk.span;
                Some(ElseBranch::Block(blk))
            }
        } else {
            None
        };

        self.leave_depth();
        IfExpr {
            span: start.merge(end),
            cond,
            then_block,
            else_branch,
        }
    }

    pub(crate) fn parse_match_parts(&mut self) -> (Expr, Vec<MatchArm>, Span) {
        self.bump(); // `match`
        let scrutinee = self.parse_expr();
        self.expect(TokenKind::LBrace);
        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            let a_start = self.span();
            let pat = self.parse_pattern();
            let guard = if self.eat(TokenKind::KwIf) {
                Some(self.parse_expr())
            } else {
                None
            };
            self.expect(TokenKind::FatArrow);
            let body = if self.at(TokenKind::LBrace) {
                let blk = self.parse_block(true);
                self.eat(TokenKind::Comma);
                MatchArmBody::Block(blk)
            } else {
                let b = match self.kind() {
                    TokenKind::KwReturn => MatchArmBody::Jump(self.parse_return_action()),
                    TokenKind::KwBreak => MatchArmBody::Jump(self.parse_break_action()),
                    TokenKind::KwContinue => MatchArmBody::Jump(self.parse_continue_action()),
                    TokenKind::KwUnreachable => {
                        self.bump();
                        MatchArmBody::Jump(JumpAction::Unreachable)
                    }
                    _ => MatchArmBody::Expr(self.parse_expr()),
                };
                if !self.eat(TokenKind::Comma) && !self.at(TokenKind::RBrace) {
                    self.diags.error(
                        ErrorCode::E0005,
                        self.span(),
                        format!(
                            "expected `,` after match arm, found {}",
                            self.kind().description()
                        ),
                    );
                }
                b
            };
            arms.push(MatchArm {
                span: a_start.merge(self.prev_end_span()),
                pat,
                guard,
                body,
            });
            if self.pos == prev {
                self.bump();
            }
        }
        let end = self
            .expect(TokenKind::RBrace)
            .map(|t| t.span)
            .unwrap_or_else(|| self.prev_end_span());
        (scrutinee, arms, end)
    }

    fn parse_while_stmt(&mut self, label: Option<IdentSpan>) -> Stmt {
        let start = label.map(|l| l.span).unwrap_or_else(|| self.span());
        self.bump(); // `while`
        let cond = self.parse_expr();
        let where_clauses = self.parse_optional_where_clause();
        let body = self.parse_control_flow_block("while", false);
        Stmt {
            span: start.merge(body.span),
            kind: StmtKind::While {
                label,
                cond,
                where_clauses,
                body,
            },
        }
    }

    fn parse_for_stmt(&mut self, label: Option<IdentSpan>) -> Stmt {
        let start = label.map(|l| l.span).unwrap_or_else(|| self.span());
        self.bump(); // `for`
        let pat = self.parse_pattern();
        self.expect(TokenKind::KwIn);
        let iter_start = self.parse_expr();
        let iter_end = if self.eat(TokenKind::DotDotEq) {
            Some((RangeKind::Inclusive, self.parse_expr()))
        } else if self.eat(TokenKind::DotDot) {
            Some((RangeKind::Exclusive, self.parse_expr()))
        } else {
            None
        };
        let body = self.parse_control_flow_block("for", false);
        Stmt {
            span: start.merge(body.span),
            kind: StmtKind::For {
                label,
                pat,
                iter_start,
                iter_end,
                body,
            },
        }
    }

    fn parse_par_for_stmt(&mut self) -> Stmt {
        let start = self.bump().span; // `par`
        self.expect(TokenKind::KwFor);
        let pat = self.parse_pattern();
        self.expect(TokenKind::KwIn);
        let iter_start = self.parse_expr();
        let iter_end = if self.eat(TokenKind::DotDotEq) {
            Some((RangeKind::Inclusive, self.parse_expr()))
        } else if self.eat(TokenKind::DotDot) {
            Some((RangeKind::Exclusive, self.parse_expr()))
        } else {
            None
        };
        let body = self.parse_control_flow_block("par for", false);
        Stmt {
            span: start.merge(body.span),
            kind: StmtKind::ParFor {
                pat,
                iter_start,
                iter_end,
                body,
            },
        }
    }

    fn parse_loop_stmt(&mut self, label: Option<IdentSpan>) -> Stmt {
        let start = label.map(|l| l.span).unwrap_or_else(|| self.span());
        self.bump(); // `loop`
        let body = self.parse_control_flow_block("loop", false);
        Stmt {
            span: start.merge(body.span),
            kind: StmtKind::Loop { label, body },
        }
    }

    fn parse_raw_stmt(&mut self) -> Stmt {
        let raw_tok = self.bump();
        if self.profile == FileProfile::Safe {
            self.diags.error(
                ErrorCode::E0111,
                raw_tok.span,
                "ZM-11: `raw { ... }` blocks are forbidden outside `.hal.eon` modules",
            );
        }
        let body = self.parse_control_flow_block("raw", false);
        Stmt {
            span: raw_tok.span.merge(body.span),
            kind: StmtKind::RawBlock(body),
        }
    }

    pub(crate) fn parse_pattern(&mut self) -> Pat {
        if !self.enter_depth() {
            return Pat {
                span: self.span(),
                kind: PatKind::Error,
            };
        }
        let start = self.span();
        let kind = match self.kind() {
            TokenKind::Underscore => {
                self.bump();
                PatKind::Wildcard
            }
            TokenKind::KwTrue => {
                self.bump();
                PatKind::BoolLit(true)
            }
            TokenKind::KwFalse => {
                self.bump();
                PatKind::BoolLit(false)
            }
            TokenKind::KwNull => {
                self.bump();
                PatKind::Null
            }
            TokenKind::Minus | TokenKind::IntLit(_) => self.parse_int_pattern(),
            TokenKind::CharLit(ch) => {
                self.bump();
                let range_end = if self.eat(TokenKind::DotDotEq) {
                    self.parse_pattern_char().map(|c| (RangeKind::Inclusive, c))
                } else if self.eat(TokenKind::DotDot) {
                    self.parse_pattern_char().map(|c| (RangeKind::Exclusive, c))
                } else {
                    None
                };
                PatKind::CharLit { ch, range_end }
            }
            TokenKind::Dot => {
                self.bump();
                if self.eat(TokenKind::LBrace) {
                    let fields = self.parse_field_pat_list();
                    self.expect(TokenKind::RBrace);
                    PatKind::Struct { path: None, fields }
                } else {
                    let name = self.expect_ident_or_dummy("variant name");
                    let args = if self.eat(TokenKind::LParen) {
                        let mut ps = Vec::new();
                        while !self.at(TokenKind::RParen) && !self.at_eof() {
                            ps.push(self.parse_pattern());
                            if !self.eat(TokenKind::Comma) {
                                break;
                            }
                        }
                        self.expect(TokenKind::RParen);
                        ps
                    } else {
                        Vec::new()
                    };
                    PatKind::Variant { name, args }
                }
            }
            TokenKind::LParen => {
                self.bump();
                let mut ps = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at_eof() {
                    ps.push(self.parse_pattern());
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen);
                PatKind::Tuple(ps)
            }
            TokenKind::Ident => {
                if matches!(self.peek_kind(1), TokenKind::Dot | TokenKind::LBracket) {
                    let path = self.parse_type_path();
                    self.expect(TokenKind::Dot);
                    self.expect(TokenKind::LBrace);
                    let fields = self.parse_field_pat_list();
                    self.expect(TokenKind::RBrace);
                    PatKind::Struct {
                        path: Some(path),
                        fields,
                    }
                } else {
                    let tok = self.bump();
                    PatKind::Ident(self.interner.intern(self.slice(tok.span)))
                }
            }
            TokenKind::KwRaw | TokenKind::KwAsm | TokenKind::KwVolatile => {
                let tok = self.bump();
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
                        format!("expected pattern, found `{}`", name),
                    );
                }
                PatKind::Error
            }
            TokenKind::Error => {
                self.bump();
                PatKind::Error
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    format!("expected pattern, found {}", self.kind().description()),
                );
                PatKind::Error
            }
        };

        self.leave_depth();
        Pat {
            span: start.merge(self.prev_end_span()),
            kind,
        }
    }

    fn parse_int_pattern(&mut self) -> PatKind {
        let neg = self.eat(TokenKind::Minus);
        let raw = self.expect_pattern_int_raw();
        let range_end = if self.eat(TokenKind::DotDotEq) {
            let neg2 = self.eat(TokenKind::Minus);
            let raw2 = self.expect_pattern_int_raw();
            Some((RangeKind::Inclusive, neg2, raw2))
        } else if self.eat(TokenKind::DotDot) {
            let neg2 = self.eat(TokenKind::Minus);
            let raw2 = self.expect_pattern_int_raw();
            Some((RangeKind::Exclusive, neg2, raw2))
        } else {
            None
        };
        PatKind::IntLit {
            neg,
            raw,
            range_end,
        }
    }

    fn expect_pattern_int_raw(&mut self) -> Symbol {
        match self.kind() {
            TokenKind::IntLit(id) => {
                self.bump();
                let raw = self.lexed.num_raw(self.src, id);
                self.interner.intern(raw)
            }
            TokenKind::Error => {
                self.bump();
                self.interner.intern("0")
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    format!(
                        "expected integer literal in pattern, found {}",
                        self.kind().description()
                    ),
                );
                self.interner.intern("0")
            }
        }
    }

    fn parse_pattern_char(&mut self) -> Option<char> {
        match self.kind() {
            TokenKind::CharLit(ch) => {
                self.bump();
                Some(ch)
            }
            TokenKind::Error => {
                self.bump();
                None
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    format!(
                        "expected character literal in range pattern, found {}",
                        self.kind().description()
                    ),
                );
                None
            }
        }
    }

    fn parse_field_pat_list(&mut self) -> Vec<FieldPat> {
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let name = self.expect_ident_or_dummy("field pattern name");
            let mut span = name.span;
            let pat = if self.eat(TokenKind::Colon) {
                let p = self.parse_pattern();
                span = span.merge(p.span);
                Some(p)
            } else {
                None
            };
            fields.push(FieldPat { span, name, pat });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        fields
    }
}
