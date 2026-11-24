use crate::ast::*;
use crate::diag::ErrorCode;
use crate::lexer::TokenKind;
use crate::parser::Parser;
use crate::span::{FileProfile, Span};

impl<'a> Parser<'a> {
    pub(crate) fn parse_compilation_unit(&mut self) -> CompilationUnit {
        let (module_path, is_hal, hal_span) = self.parse_module_header();

        let mut imports = Vec::new();
        while self.at(TokenKind::KwImport) {
            imports.push(self.parse_import_decl());
        }

        let mut decls = Vec::new();
        while !self.at_eof() {
            if self.at(TokenKind::KwImport) {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    "`import` declarations must appear immediately after the `module` header",
                );
                imports.push(self.parse_import_decl());
                continue;
            }
            let prev_pos = self.pos;
            if let Some(d) = self.parse_top_level_decl() {
                decls.push(d);
            }
            if self.pos == prev_pos {
                self.bump();
            }
        }

        CompilationUnit {
            file: self.file_id,
            module_path,
            is_hal,
            hal_span,
            imports,
            decls,
        }
    }

    fn parse_module_header(&mut self) -> (Vec<IdentSpan>, bool, Option<Span>) {
        if !self.eat(TokenKind::KwModule) {
            self.diags.error(
                ErrorCode::E0005,
                self.span(),
                "expected `module` header at start of file",
            );
            if self.profile == FileProfile::Hal {
                self.diags.error(
                    ErrorCode::E0111,
                    self.span(),
                    "`.hal.eon` files must declare `module ... @hal;`",
                );
            }
            return (Vec::new(), false, None);
        }

        let path = self.parse_module_path();
        let mut is_hal = false;
        let mut hal_span = None;
        if self.eat(TokenKind::At) {
            let at_span = self.prev_end_span();
            if let Some(id) = self.expect_ident("`hal`") {
                let s = at_span.merge(id.span);
                if self.interner.get(id.sym) == "hal" {
                    is_hal = true;
                    hal_span = Some(s);
                    if self.profile == FileProfile::Safe {
                        self.diags.error(
                            ErrorCode::E0111,
                            s,
                            "`@hal` module header is only allowed in `.hal.eon` files",
                        );
                    }
                } else {
                    self.diags.error(
                        ErrorCode::E0005,
                        s,
                        format!("expected `@hal` in module header, found `@{}`", self.interner.get(id.sym)),
                    );
                }
            }
        } else if self.profile == FileProfile::Hal {
            self.diags.error(
                ErrorCode::E0111,
                self.span(),
                "`.hal.eon` files must declare `module ... @hal;`",
            );
        }

        self.expect_semi();
        (path, is_hal, hal_span)
    }

    fn parse_module_path(&mut self) -> Vec<IdentSpan> {
        let mut segs = vec![self.expect_ident_or_dummy("module identifier")];
        while self.eat(TokenKind::Dot) {
            segs.push(self.expect_ident_or_dummy("module segment"));
        }
        segs
    }

    fn parse_import_decl(&mut self) -> ImportDecl {
        let start = self.bump().span; // `import`
        let mut path = vec![self.expect_ident_or_dummy("module path")];
        let mut tail = ImportTail::None;

        while self.eat(TokenKind::Dot) {
            if self.eat(TokenKind::LBrace) {
                let mut items = Vec::new();
                while !self.at(TokenKind::RBrace) && !self.at_eof() {
                    let name = self.expect_ident_or_dummy("imported symbol");
                    let mut span = name.span;
                    let alias = if self.eat(TokenKind::KwAs) {
                        let a = self.expect_ident_or_dummy("alias identifier");
                        span = span.merge(a.span);
                        Some(a)
                    } else {
                        None
                    };
                    items.push(ImportItem { span, name, alias });
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
                tail = ImportTail::Items(items);
                break;
            } else {
                path.push(self.expect_ident_or_dummy("module segment"));
            }
        }

        if matches!(tail, ImportTail::None) && self.eat(TokenKind::KwAs) {
            tail = ImportTail::Alias(self.expect_ident_or_dummy("alias identifier"));
        }

        self.expect_semi();
        ImportDecl {
            span: start.merge(self.prev_end_span()),
            path,
            tail,
        }
    }

    pub(crate) fn parse_attributes(&mut self) -> Vec<Attr> {
        let mut out = Vec::new();
        while self.at(TokenKind::At) {
            out.push(self.parse_one_attribute());
        }
        out
    }

    fn parse_attr_name(&mut self) -> IdentSpan {
        let tok = self.cur();
        if matches!(tok.kind, TokenKind::Ident | TokenKind::KwExtern) {
            self.bump();
            IdentSpan {
                sym: self.interner.intern(self.slice(tok.span)),
                span: tok.span,
            }
        } else {
            self.expect_ident_or_dummy("attribute name")
        }
    }

    fn parse_one_attribute(&mut self) -> Attr {
        let at_span = self.bump().span; // `@`
        let name = self.parse_attr_name();
        let mut span = at_span.merge(name.span);
        let mut has_parens = false;
        let mut args = Vec::new();
        if self.eat(TokenKind::LParen) {
            has_parens = true;
            while !self.at(TokenKind::RParen) && !self.at_eof() {
                let arg_start = self.span();
                let arg_name = if self.at(TokenKind::Ident) && self.peek_kind(1) == TokenKind::Eq {
                    let n = self.expect_ident_or_dummy("attribute argument name");
                    self.bump(); // `=`
                    Some(n)
                } else {
                    None
                };
                let value = self.parse_expr();
                let vspan = value.span;
                args.push(AttrArg {
                    span: arg_start.merge(vspan),
                    name: arg_name,
                    value,
                });
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            if let Some(rp) = self.expect(TokenKind::RParen) {
                span = span.merge(rp.span);
            }
        }
        Attr {
            span,
            name,
            args,
            has_parens,
        }
    }

    pub(crate) fn parse_top_level_decl(&mut self) -> Option<Decl> {
        let start_span = self.span();
        let attrs = self.parse_attributes();
        let is_pub = self.eat(TokenKind::KwPub);

        let kind = match self.kind() {
            TokenKind::KwFn | TokenKind::KwComptime => {
                DeclKind::Fn(self.parse_fn_decl(&attrs))
            }
            TokenKind::KwView => {
                if self.peek_kind(1) == TokenKind::KwStruct {
                    self.bump(); // `view`
                    DeclKind::Struct(self.parse_struct_decl(false, true))
                } else {
                    DeclKind::Fn(self.parse_fn_decl(&attrs))
                }
            }
            TokenKind::KwSys => DeclKind::Sys(self.parse_sys_decl()),
            TokenKind::KwStruct => DeclKind::Struct(self.parse_struct_decl(false, false)),
            TokenKind::KwLinear => {
                let lin_span = self.bump().span;
                match self.kind() {
                    TokenKind::KwStruct => DeclKind::Struct(self.parse_struct_decl(true, false)),
                    TokenKind::KwView if self.peek_kind(1) == TokenKind::KwStruct => {
                        self.bump(); // `view`
                        DeclKind::Struct(self.parse_struct_decl(true, true))
                    }
                    TokenKind::KwUnion => DeclKind::Union(self.parse_union_decl(true)),
                    _ => {
                        self.diags.error(
                            ErrorCode::E0005,
                            lin_span,
                            "expected `struct`, `view struct`, or `union` after `linear`",
                        );
                        self.sync_to_decl();
                        return None;
                    }
                }
            }
            TokenKind::KwPool => DeclKind::Pool(self.parse_pool_decl()),
            TokenKind::KwEnum => DeclKind::Enum(self.parse_enum_decl()),
            TokenKind::KwUnion => DeclKind::Union(self.parse_union_decl(false)),
            TokenKind::KwBitfield => DeclKind::Bitfield(self.parse_bitfield_decl()),
            TokenKind::KwIface => DeclKind::Iface(self.parse_iface_decl()),
            TokenKind::KwImpl => DeclKind::Impl(self.parse_impl_decl()),
            TokenKind::KwError => DeclKind::Error(self.parse_error_decl()),
            TokenKind::KwType => DeclKind::TypeAlias(self.parse_type_alias_decl()),
            TokenKind::KwConst => DeclKind::Const(self.parse_const_decl()),
            TokenKind::KwExtern => self.parse_extern_decl()?,
            TokenKind::KwVar => {
                let var_tok = self.bump();
                self.diags.error(
                    ErrorCode::E0110,
                    var_tok.span,
                    "ZM-10: module-level mutable `var` declarations are forbidden in Eon (even in `.hal.eon`)",
                );
                self.sync_to_stmt();
                return None;
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    format!("expected declaration, found {}", self.kind().description()),
                );
                self.sync_to_decl();
                return None;
            }
        };

        Some(Decl {
            span: start_span.merge(self.prev_end_span()),
            attrs,
            is_pub,
            kind,
        })
    }

    fn has_gpu_attr(&self, attrs: &[Attr], effects: &[Attr]) -> bool {
        attrs
            .iter()
            .chain(effects.iter())
            .any(|a| self.interner.get(a.name.sym) == "gpu")
    }

    fn validate_param_access_modes(&mut self, params: &[Param], has_gpu: bool) {
        if has_gpu {
            return;
        }
        for p in params {
            if p.access.is_some() {
                self.diags.error(
                    ErrorCode::E0823,
                    p.span,
                    "`read`/`write` parameter access qualifiers are only allowed on `sys` and `@gpu` functions",
                );
            }
        }
    }

    fn parse_fn_decl(&mut self, decl_attrs: &[Attr]) -> FnDecl {
        let mut is_comptime = false;
        let mut is_view = false;
        if self.eat(TokenKind::KwComptime) {
            is_comptime = true;
        } else if self.eat(TokenKind::KwView) {
            is_view = true;
        }
        self.expect(TokenKind::KwFn);
        let name = self.expect_ident_or_dummy("function name");
        let generics = self.parse_optional_generic_params();
        let params = self.parse_param_list();
        let ret_type = if self.eat(TokenKind::Arrow) || self.at(TokenKind::Bang) {
            Some(self.parse_type_expr())
        } else {
            None
        };
        let effects = self.parse_attributes();
        let has_gpu = self.has_gpu_attr(decl_attrs, &effects);
        self.validate_param_access_modes(&params, has_gpu);
        let where_clauses = self.parse_optional_where_clause();
        let ensures_clauses = self.parse_optional_ensures_clause();
        let body = self.parse_block(false);
        FnDecl {
            is_comptime,
            is_view,
            name,
            generics,
            params,
            ret_type,
            effects,
            where_clauses,
            ensures_clauses,
            body,
        }
    }

    fn parse_sys_decl(&mut self) -> SysDecl {
        self.bump(); // `sys`
        let name = self.expect_ident_or_dummy("system name");
        let generics = self.parse_optional_generic_params();
        let params = self.parse_sys_param_list();
        let ret_type = if self.eat(TokenKind::Arrow) || self.at(TokenKind::Bang) {
            Some(self.parse_type_expr())
        } else {
            None
        };
        let effects = self.parse_attributes();
        let where_clauses = self.parse_optional_where_clause();
        let ensures_clauses = self.parse_optional_ensures_clause();
        let body = self.parse_block(false);
        SysDecl {
            name,
            generics,
            params,
            ret_type,
            effects,
            where_clauses,
            ensures_clauses,
            body,
        }
    }

    pub(crate) fn parse_param_list(&mut self) -> Vec<Param> {
        self.expect(TokenKind::LParen);
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) && !self.at_eof() {
            params.push(self.parse_param());
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RParen);
        params
    }

    fn parse_param(&mut self) -> Param {
        let start = self.span();
        let attrs = self.parse_attributes();
        let is_comptime = self.eat(TokenKind::KwComptime);
        let modifier = if self.eat(TokenKind::KwMut) {
            Some(ParamModifier::Mut)
        } else if self.eat(TokenKind::KwFrom) {
            Some(ParamModifier::From)
        } else {
            None
        };
        let (name, is_self) = self.parse_param_name();
        self.expect(TokenKind::Colon);
        let access = self.parse_contextual_access_mode();
        let ty = self.parse_type_expr();
        let end = ty.span;
        Param {
            span: start.merge(end),
            attrs,
            is_comptime,
            modifier,
            name,
            is_self,
            access,
            ty,
        }
    }

    fn parse_param_name(&mut self) -> (IdentSpan, bool) {
        if self.at(TokenKind::KwSelf) {
            let tok = self.bump();
            let sym = self.interner.intern("self");
            (IdentSpan { sym, span: tok.span }, true)
        } else {
            (self.expect_ident_or_dummy("parameter name"), false)
        }
    }

    fn parse_sys_param_list(&mut self) -> Vec<SysParam> {
        self.expect(TokenKind::LParen);
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) && !self.at_eof() {
            params.push(self.parse_sys_param());
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RParen);
        params
    }

    fn parse_sys_param(&mut self) -> SysParam {
        let start = self.span();
        let attrs = self.parse_attributes();
        let is_comptime = self.eat(TokenKind::KwComptime);
        let modifier = if self.eat(TokenKind::KwMut) {
            Some(ParamModifier::Mut)
        } else if self.eat(TokenKind::KwFrom) {
            Some(ParamModifier::From)
        } else {
            None
        };
        let (name, is_self) = self.parse_param_name();
        self.expect(TokenKind::Colon);
        let access = self.parse_contextual_access_mode();
        let ty = self.parse_type_expr();
        let mut columns = Vec::new();
        if access.is_some() && self.eat(TokenKind::Dot) {
            if self.eat(TokenKind::LBrace) {
                while !self.at(TokenKind::RBrace) && !self.at_eof() {
                    columns.push(self.expect_ident_or_dummy("pool column name"));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace);
            } else {
                columns.push(self.expect_ident_or_dummy("pool column name"));
            }
        }
        SysParam {
            span: start.merge(self.prev_end_span()),
            attrs,
            is_comptime,
            modifier,
            name,
            is_self,
            access,
            ty,
            columns,
        }
    }

    fn parse_contextual_access_mode(&mut self) -> Option<AccessMode> {
        if self.at(TokenKind::Ident)
            && !matches!(self.peek_kind(1), TokenKind::Comma | TokenKind::RParen)
        {
            match self.slice(self.span()) {
                "read" => {
                    self.bump();
                    return Some(AccessMode::Read);
                }
                "write" => {
                    self.bump();
                    return Some(AccessMode::Write);
                }
                _ => {}
            }
        }
        None
    }

    pub(crate) fn parse_optional_where_clause(&mut self) -> Vec<Expr> {
        if !self.eat(TokenKind::KwWhere) {
            return Vec::new();
        }
        self.parse_constraint_conjunction(false)
    }

    fn parse_optional_ensures_clause(&mut self) -> Vec<Expr> {
        if !self.eat(TokenKind::KwEnsures) {
            return Vec::new();
        }
        self.parse_constraint_conjunction(true)
    }

    fn parse_constraint_conjunction(&mut self, allow_return: bool) -> Vec<Expr> {
        let old_ret = self.allow_return_as_expr;
        let old_in = self.in_ensures_or_where;
        self.allow_return_as_expr = allow_return;
        self.in_ensures_or_where = true;

        let mut exprs = Vec::new();
        loop {
            let (e, _) = self.parse_binary_expr_bp(8);
            exprs.push(e);
            if !self.eat(TokenKind::AndAnd) {
                break;
            }
        }

        self.allow_return_as_expr = old_ret;
        self.in_ensures_or_where = old_in;
        exprs
    }

    fn parse_struct_decl(&mut self, is_linear: bool, is_view: bool) -> StructDecl {
        self.bump(); // `struct`
        let name = self.expect_ident_or_dummy("struct name");
        let generics = self.parse_optional_generic_params();
        let where_clauses = self.parse_optional_where_clause();
        let fields = self.parse_field_block();
        StructDecl {
            is_linear,
            is_view,
            name,
            generics,
            where_clauses,
            fields,
        }
    }

    fn parse_pool_decl(&mut self) -> PoolDecl {
        self.bump(); // `pool`
        let name = self.expect_ident_or_dummy("pool name");
        let generics = self.parse_optional_generic_params();
        let where_clauses = self.parse_optional_where_clause();
        let fields = self.parse_field_block();
        PoolDecl {
            name,
            generics,
            where_clauses,
            fields,
        }
    }

    pub(crate) fn parse_field_block(&mut self) -> Vec<Field> {
        self.expect(TokenKind::LBrace);
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            fields.push(self.parse_field_decl());
            if self.pos == prev {
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace);
        fields
    }

    fn parse_field_decl(&mut self) -> Field {
        let start = self.span();
        let mut attrs = self.parse_attributes();
        let is_pub = self.eat(TokenKind::KwPub);
        let name = self.expect_ident_or_dummy("field name");
        self.expect(TokenKind::Colon);
        let ty = self.parse_type_expr();
        attrs.extend(self.parse_attributes());
        let default = if self.eat(TokenKind::Eq) {
            Some(self.parse_expr())
        } else {
            None
        };
        self.expect_semi();
        Field {
            span: start.merge(self.prev_end_span()),
            attrs,
            is_pub,
            name,
            ty,
            default,
        }
    }

    fn parse_enum_decl(&mut self) -> EnumDecl {
        self.bump(); // `enum`
        let name = self.expect_ident_or_dummy("enum name");
        let repr_int = if self.eat(TokenKind::Colon) {
            Some(self.expect_ident_or_dummy("integer representation type"))
        } else {
            None
        };
        self.expect(TokenKind::LBrace);
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let vname = self.expect_ident_or_dummy("enum variant");
            let mut vspan = vname.span;
            let value = if self.eat(TokenKind::Eq) {
                let e = self.parse_expr();
                vspan = vspan.merge(e.span);
                Some(e)
            } else {
                None
            };
            variants.push(EnumVariant {
                span: vspan,
                name: vname,
                value,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace);
        EnumDecl {
            name,
            repr_int,
            variants,
        }
    }

    fn parse_union_decl(&mut self, is_linear: bool) -> UnionDecl {
        self.bump(); // `union`
        let name = self.expect_ident_or_dummy("union name");
        let generics = self.parse_optional_generic_params();
        let where_clauses = self.parse_optional_where_clause();
        self.expect(TokenKind::LBrace);
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let vname = self.expect_ident_or_dummy("union variant");
            let mut vspan = vname.span;
            let ty = if self.eat(TokenKind::Colon) {
                let t = self.parse_type_expr();
                vspan = vspan.merge(t.span);
                Some(t)
            } else {
                None
            };
            variants.push(UnionVariant {
                span: vspan,
                name: vname,
                ty,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace);
        UnionDecl {
            is_linear,
            name,
            generics,
            where_clauses,
            variants,
        }
    }

    fn parse_bitfield_decl(&mut self) -> BitfieldDecl {
        self.bump(); // `bitfield`
        let name = self.expect_ident_or_dummy("bitfield name");
        self.expect(TokenKind::Colon);
        let base_int = self.expect_ident_or_dummy("unsigned integer backing type");
        self.expect(TokenKind::LBrace);
        let mut members = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            let mname = self.expect_ident_or_dummy("bitfield member name");
            self.expect(TokenKind::Colon);
            let mty = self.expect_ident_or_dummy("bitfield member type");
            let bits = if self.eat(TokenKind::At) {
                let attr_name = self.expect_ident_or_dummy("`bits`");
                if self.interner.get(attr_name.sym) != "bits" {
                    self.diags.error(
                        ErrorCode::E0005,
                        attr_name.span,
                        "expected `@bits(lo..hi)` on bitfield member",
                    );
                }
                self.expect(TokenKind::LParen);
                let lo = self.parse_expr();
                self.expect(TokenKind::DotDot);
                let hi = self.parse_expr();
                self.expect(TokenKind::RParen);
                Some((lo, hi))
            } else {
                None
            };
            self.expect_semi();
            members.push(BitfieldMember {
                span: mname.span.merge(self.prev_end_span()),
                name: mname,
                ty: mty,
                bits,
            });
            if self.pos == prev {
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace);
        BitfieldDecl {
            name,
            base_int,
            members,
        }
    }

    fn parse_iface_decl(&mut self) -> IfaceDecl {
        self.bump(); // `iface`
        let name = self.expect_ident_or_dummy("interface name");
        let generics = self.parse_optional_generic_params();
        self.expect(TokenKind::LBrace);
        let mut methods = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            let start = self.span();
            let attrs = self.parse_attributes();
            let is_view = self.eat(TokenKind::KwView);
            self.expect(TokenKind::KwFn);
            let mname = self.expect_ident_or_dummy("method name");
            let mgen = self.parse_optional_generic_params();
            let params = self.parse_param_list();
            let ret_type = if self.eat(TokenKind::Arrow) || self.at(TokenKind::Bang) {
                Some(self.parse_type_expr())
            } else {
                None
            };
            let effects = self.parse_attributes();
            let has_gpu = self.has_gpu_attr(&attrs, &effects);
            self.validate_param_access_modes(&params, has_gpu);
            let where_clauses = self.parse_optional_where_clause();
            let ensures_clauses = self.parse_optional_ensures_clause();
            self.expect_semi();
            methods.push(IfaceMethod {
                span: start.merge(self.prev_end_span()),
                attrs,
                is_view,
                name: mname,
                generics: mgen,
                params,
                ret_type,
                effects,
                where_clauses,
                ensures_clauses,
            });
            if self.pos == prev {
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace);
        IfaceDecl {
            name,
            generics,
            methods,
        }
    }

    fn parse_impl_decl(&mut self) -> ImplDecl {
        self.bump(); // `impl`
        let generics = self.parse_optional_generic_params();
        let first_ty = self.parse_type_expr();
        let (iface_type, target_type) = if self.eat(TokenKind::KwFor) {
            (Some(first_ty), self.parse_type_expr())
        } else {
            (None, first_ty)
        };
        let where_clauses = self.parse_optional_where_clause();
        self.expect(TokenKind::LBrace);
        let mut decls = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let prev = self.pos;
            if let Some(d) = self.parse_top_level_decl() {
                decls.push(d);
            }
            if self.pos == prev {
                self.bump();
            }
        }
        self.expect(TokenKind::RBrace);
        ImplDecl {
            generics,
            iface_type,
            target_type,
            where_clauses,
            decls,
        }
    }

    fn parse_error_decl(&mut self) -> ErrorDecl {
        self.bump(); // `error`
        let name = self.expect_ident_or_dummy("error set name");
        self.expect(TokenKind::LBrace);
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            variants.push(self.expect_ident_or_dummy("error variant"));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace);
        ErrorDecl { name, variants }
    }

    fn parse_type_alias_decl(&mut self) -> TypeAliasDecl {
        self.bump(); // `type`
        let name = self.expect_ident_or_dummy("type alias name");
        let generics = self.parse_optional_generic_params();
        let where_clauses = self.parse_optional_where_clause();
        self.expect(TokenKind::Eq);
        let ty = self.parse_type_expr();
        self.expect_semi();
        TypeAliasDecl {
            name,
            generics,
            where_clauses,
            ty,
        }
    }

    pub(crate) fn parse_const_decl(&mut self) -> ConstDecl {
        self.bump(); // `const`
        let name = self.expect_ident_or_dummy("constant name");
        self.expect(TokenKind::Colon);
        let ty = self.parse_type_expr();
        self.expect(TokenKind::Eq);
        let value = self.parse_expr();
        self.expect_semi();
        ConstDecl { name, ty, value }
    }

    fn parse_extern_decl(&mut self) -> Option<DeclKind> {
        let ext_tok = self.bump(); // `extern`
        if self.profile == FileProfile::Safe {
            self.diags.error(
                ErrorCode::E0111,
                ext_tok.span,
                "ZM-11: `extern` declarations are only allowed inside `.hal.eon` modules",
            );
        }
        let abi = match self.kind() {
            TokenKind::StringLit(sid) => {
                self.bump();
                let s = self.lexed.string_lits[sid.0 as usize].clone();
                self.interner.intern(&s)
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    "expected ABI string literal (e.g. `\"C\"`) after `extern`",
                );
                self.interner.intern("C")
            }
        };

        match self.kind() {
            TokenKind::KwFn => {
                self.bump();
                let name = self.expect_ident_or_dummy("extern function name");
                let params = self.parse_param_list();
                let ret_type = if self.eat(TokenKind::Arrow) {
                    Some(self.parse_type_expr())
                } else {
                    None
                };
                self.expect_semi();
                self.validate_param_access_modes(&params, false);
                Some(DeclKind::ExternFn(ExternFnDecl {
                    abi,
                    name,
                    params,
                    ret_type,
                }))
            }
            TokenKind::KwStruct => {
                self.bump();
                let name = self.expect_ident_or_dummy("extern struct name");
                let fields = self.parse_field_block();
                Some(DeclKind::ExternStruct(ExternAggDecl { abi, name, fields }))
            }
            TokenKind::KwUnion => {
                self.bump();
                let name = self.expect_ident_or_dummy("extern union name");
                let fields = self.parse_field_block();
                Some(DeclKind::ExternUnion(ExternAggDecl { abi, name, fields }))
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0005,
                    self.span(),
                    "expected `fn`, `struct`, or `union` after `extern \"...\"`",
                );
                self.sync_to_decl();
                None
            }
        }
    }
}
