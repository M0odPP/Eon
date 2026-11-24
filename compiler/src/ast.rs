use std::fmt::Write;

use crate::span::{FileId, Interner, Span, Symbol};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IdentSpan {
    pub sym: Symbol,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct CompilationUnit {
    pub file: FileId,
    pub module_path: Vec<IdentSpan>,
    pub is_hal: bool,
    pub hal_span: Option<Span>,
    pub imports: Vec<ImportDecl>,
    pub decls: Vec<Decl>,
}

impl CompilationUnit {
    pub fn dump(&self, interner: &Interner) -> String {
        let mut out = String::new();
        out.push_str("module ");
        for (i, seg) in self.module_path.iter().enumerate() {
            if i > 0 {
                out.push('.');
            }
            out.push_str(interner.get(seg.sym));
        }
        if self.is_hal {
            out.push_str(" @hal");
        }
        out.push_str(";\n");

        for imp in &self.imports {
            out.push_str("import ");
            for (i, seg) in imp.path.iter().enumerate() {
                if i > 0 {
                    out.push('.');
                }
                out.push_str(interner.get(seg.sym));
            }
            match &imp.tail {
                ImportTail::None => {}
                ImportTail::Alias(a) => {
                    let _ = write!(out, " as {}", interner.get(a.sym));
                }
                ImportTail::Items(items) => {
                    out.push_str(".{");
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(interner.get(item.name.sym));
                        if let Some(a) = item.alias {
                            let _ = write!(out, " as {}", interner.get(a.sym));
                        }
                    }
                    out.push('}');
                }
            }
            out.push_str(";\n");
        }

        for decl in &self.decls {
            dump_decl(decl, interner, 0, &mut out);
        }
        out
    }

    pub fn dump_full(&self, interner: &Interner) -> String {
        let mut out = String::new();
        out.push_str("module ");
        write_ident_path(&self.module_path, interner, &mut out);
        if self.is_hal {
            out.push_str(" @hal");
        }
        out.push('\n');

        for imp in &self.imports {
            out.push_str("import ");
            write_ident_path(&imp.path, interner, &mut out);
            match &imp.tail {
                ImportTail::None => {}
                ImportTail::Alias(a) => {
                    let _ = write!(out, " as {}", interner.get(a.sym));
                }
                ImportTail::Items(items) => {
                    out.push_str(".{");
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(interner.get(item.name.sym));
                        if let Some(a) = item.alias {
                            let _ = write!(out, " as {}", interner.get(a.sym));
                        }
                    }
                    out.push('}');
                }
            }
            out.push('\n');
        }

        for decl in &self.decls {
            dump_decl_full(decl, interner, 0, &mut out);
        }
        out
    }
}

fn dump_decl(d: &Decl, interner: &Interner, depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    for attr in &d.attrs {
        let _ = write!(out, "@{} ", interner.get(attr.name.sym));
    }
    if d.is_pub {
        out.push_str("pub ");
    }
    match &d.kind {
        DeclKind::Fn(f) => {
            if f.is_comptime {
                out.push_str("comptime ");
            }
            if f.is_view {
                out.push_str("view ");
            }
            let _ = writeln!(out, "fn {} {{ ... }}", interner.get(f.name.sym));
        }
        DeclKind::Sys(s) => {
            let _ = writeln!(out, "sys {} {{ ... }}", interner.get(s.name.sym));
        }
        DeclKind::Struct(s) => {
            if s.is_linear {
                out.push_str("linear ");
            }
            if s.is_view {
                out.push_str("view ");
            }
            let _ = writeln!(out, "struct {} ({} fields)", interner.get(s.name.sym), s.fields.len());
        }
        DeclKind::Pool(p) => {
            let _ = writeln!(out, "pool {} ({} fields)", interner.get(p.name.sym), p.fields.len());
        }
        DeclKind::Enum(e) => {
            let _ = writeln!(out, "enum {} ({} variants)", interner.get(e.name.sym), e.variants.len());
        }
        DeclKind::Union(u) => {
            if u.is_linear {
                out.push_str("linear ");
            }
            let _ = writeln!(out, "union {} ({} variants)", interner.get(u.name.sym), u.variants.len());
        }
        DeclKind::Bitfield(b) => {
            let _ = writeln!(
                out,
                "bitfield {} : {} ({} members)",
                interner.get(b.name.sym),
                interner.get(b.base_int.sym),
                b.members.len()
            );
        }
        DeclKind::Iface(i) => {
            let _ = writeln!(out, "iface {} ({} methods)", interner.get(i.name.sym), i.methods.len());
        }
        DeclKind::Impl(i) => {
            let _ = writeln!(out, "impl ({} decls)", i.decls.len());
            for sd in &i.decls {
                dump_decl(sd, interner, depth + 1, out);
            }
        }
        DeclKind::Error(e) => {
            let _ = writeln!(out, "error {} ({} variants)", interner.get(e.name.sym), e.variants.len());
        }
        DeclKind::TypeAlias(t) => {
            let _ = writeln!(out, "type {}", interner.get(t.name.sym));
        }
        DeclKind::Const(c) => {
            let _ = writeln!(out, "const {}", interner.get(c.name.sym));
        }
        DeclKind::ExternFn(e) => {
            let _ = writeln!(out, "extern \"{}\" fn {}", interner.get(e.abi), interner.get(e.name.sym));
        }
        DeclKind::ExternStruct(e) => {
            let _ = writeln!(out, "extern \"{}\" struct {}", interner.get(e.abi), interner.get(e.name.sym));
        }
        DeclKind::ExternUnion(e) => {
            let _ = writeln!(out, "extern \"{}\" union {}", interner.get(e.abi), interner.get(e.name.sym));
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImportDecl {
    pub span: Span,
    pub path: Vec<IdentSpan>,
    pub tail: ImportTail,
}

#[derive(Clone, Debug)]
pub enum ImportTail {
    None,
    Alias(IdentSpan),
    Items(Vec<ImportItem>),
}

#[derive(Clone, Debug)]
pub struct ImportItem {
    pub span: Span,
    pub name: IdentSpan,
    pub alias: Option<IdentSpan>,
}

#[derive(Clone, Debug)]
pub struct Attr {
    pub span: Span,
    pub name: IdentSpan,
    pub args: Vec<AttrArg>,
    pub has_parens: bool,
}

#[derive(Clone, Debug)]
pub struct AttrArg {
    pub span: Span,
    pub name: Option<IdentSpan>,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct Decl {
    pub span: Span,
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub kind: DeclKind,
}

#[derive(Clone, Debug)]
pub enum DeclKind {
    Fn(FnDecl),
    Sys(SysDecl),
    Struct(StructDecl),
    Pool(PoolDecl),
    Enum(EnumDecl),
    Union(UnionDecl),
    Bitfield(BitfieldDecl),
    Iface(IfaceDecl),
    Impl(ImplDecl),
    Error(ErrorDecl),
    TypeAlias(TypeAliasDecl),
    Const(ConstDecl),
    ExternFn(ExternFnDecl),
    ExternStruct(ExternAggDecl),
    ExternUnion(ExternAggDecl),
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub is_comptime: bool,
    pub is_view: bool,
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
    pub effects: Vec<Attr>,
    pub where_clauses: Vec<Expr>,
    pub ensures_clauses: Vec<Expr>,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub struct SysDecl {
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub params: Vec<SysParam>,
    pub ret_type: Option<TypeExpr>,
    pub effects: Vec<Attr>,
    pub where_clauses: Vec<Expr>,
    pub ensures_clauses: Vec<Expr>,
    pub body: Block,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParamModifier {
    Mut,
    From,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub span: Span,
    pub attrs: Vec<Attr>,
    pub is_comptime: bool,
    pub modifier: Option<ParamModifier>,
    pub name: IdentSpan,
    pub is_self: bool,
    pub access: Option<AccessMode>,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug)]
pub struct SysParam {
    pub span: Span,
    pub attrs: Vec<Attr>,
    pub is_comptime: bool,
    pub modifier: Option<ParamModifier>,
    pub name: IdentSpan,
    pub is_self: bool,
    pub access: Option<AccessMode>,
    pub ty: TypeExpr,
    pub columns: Vec<IdentSpan>,
}

#[derive(Clone, Debug)]
pub struct StructDecl {
    pub is_linear: bool,
    pub is_view: bool,
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub where_clauses: Vec<Expr>,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct PoolDecl {
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub where_clauses: Vec<Expr>,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub span: Span,
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: IdentSpan,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct EnumDecl {
    pub name: IdentSpan,
    pub repr_int: Option<IdentSpan>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Clone, Debug)]
pub struct EnumVariant {
    pub span: Span,
    pub name: IdentSpan,
    pub value: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct UnionDecl {
    pub is_linear: bool,
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub where_clauses: Vec<Expr>,
    pub variants: Vec<UnionVariant>,
}

#[derive(Clone, Debug)]
pub struct UnionVariant {
    pub span: Span,
    pub name: IdentSpan,
    pub ty: Option<TypeExpr>,
}

#[derive(Clone, Debug)]
pub struct BitfieldDecl {
    pub name: IdentSpan,
    pub base_int: IdentSpan,
    pub members: Vec<BitfieldMember>,
}

#[derive(Clone, Debug)]
pub struct BitfieldMember {
    pub span: Span,
    pub name: IdentSpan,
    pub ty: IdentSpan,
    pub bits: Option<(Expr, Expr)>,
}

#[derive(Clone, Debug)]
pub struct IfaceDecl {
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub methods: Vec<IfaceMethod>,
}

#[derive(Clone, Debug)]
pub struct IfaceMethod {
    pub span: Span,
    pub attrs: Vec<Attr>,
    pub is_view: bool,
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
    pub effects: Vec<Attr>,
    pub where_clauses: Vec<Expr>,
    pub ensures_clauses: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub struct ImplDecl {
    pub generics: Vec<GenericParam>,
    pub iface_type: Option<TypeExpr>,
    pub target_type: TypeExpr,
    pub where_clauses: Vec<Expr>,
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug)]
pub struct ErrorDecl {
    pub name: IdentSpan,
    pub variants: Vec<IdentSpan>,
}

#[derive(Clone, Debug)]
pub struct TypeAliasDecl {
    pub name: IdentSpan,
    pub generics: Vec<GenericParam>,
    pub where_clauses: Vec<Expr>,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: IdentSpan,
    pub ty: TypeExpr,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct ExternFnDecl {
    pub abi: Symbol,
    pub name: IdentSpan,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
}

#[derive(Clone, Debug)]
pub struct ExternAggDecl {
    pub abi: Symbol,
    pub name: IdentSpan,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct GenericParam {
    pub span: Span,
    pub is_comptime: bool,
    pub name: IdentSpan,
    pub comptime_ty: Option<TypeExpr>,
    pub bounds: Vec<TypeExpr>,
}

#[derive(Clone, Debug)]
pub struct GenericArg {
    pub span: Span,
    pub kind: GenericArgKind,
}

#[derive(Clone, Debug)]
pub enum GenericArgKind {
    Type(TypeExpr),
    Expr(Expr),
    TypeOrConst(Vec<IdentSpan>),
}

#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub span: Span,
    pub kind: TypeKind,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    ErrorUnion {
        ok: Option<Box<TypeExpr>>,
        errors: Vec<TypeExpr>,
    },
    Optional(Box<TypeExpr>),
    Borrow {
        is_mut: bool,
        target: BorrowTarget,
    },
    Array {
        len: Box<Expr>,
        elem: Box<TypeExpr>,
    },
    Dyn(TypePath),
    FnPtr {
        params: Vec<TypeExpr>,
        ret: Option<Box<TypeExpr>>,
        effects: Vec<Attr>,
    },
    InlineStruct(Vec<Field>),
    Tuple(Vec<TypeExpr>),
    Void,
    Never,
    TypeType,
    SelfType,
    Inferred,
    Path(TypePath),
    Error,
}

#[derive(Clone, Debug)]
pub enum BorrowTarget {
    Single(Box<TypeExpr>),
    Slice(Box<TypeExpr>),
    Str,
}

#[derive(Clone, Debug)]
pub struct TypePath {
    pub segments: Vec<IdentSpan>,
    pub generics: Vec<GenericArg>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    pub tail_expr: Option<Box<Expr>>,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RangeKind {
    Exclusive,
    Inclusive,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let {
        attrs: Vec<Attr>,
        pat: Pat,
        ty: Option<TypeExpr>,
        init: Expr,
    },
    Var {
        attrs: Vec<Attr>,
        name: IdentSpan,
        ty: Option<TypeExpr>,
        init: Expr,
    },
    Const(ConstDecl),
    Defer(DeferBody),
    Errdefer {
        err_bind: Option<IdentSpan>,
        body: DeferBody,
    },
    Jump(JumpAction),
    If {
        is_comptime: bool,
        data: IfExpr,
    },
    Match {
        scrutinee: Expr,
        arms: Vec<MatchArm>,
    },
    While {
        label: Option<IdentSpan>,
        cond: Expr,
        where_clauses: Vec<Expr>,
        body: Block,
    },
    For {
        label: Option<IdentSpan>,
        pat: Pat,
        iter_start: Expr,
        iter_end: Option<(RangeKind, Expr)>,
        body: Block,
    },
    ParFor {
        pat: Pat,
        iter_start: Expr,
        iter_end: Option<(RangeKind, Expr)>,
        body: Block,
    },
    Loop {
        label: Option<IdentSpan>,
        body: Block,
    },
    ComptimeBlock(Block),
    RawBlock(Block),
    Assign {
        lhs: Expr,
        op: AssignOp,
        rhs: Expr,
    },
    Discard(Expr),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub enum DeferBody {
    Block(Block),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub enum JumpAction {
    Return(Option<Box<Expr>>),
    Break {
        label: Option<IdentSpan>,
        value: Option<Box<Expr>>,
    },
    Continue {
        label: Option<IdentSpan>,
    },
    Unreachable,
}

#[derive(Clone, Debug)]
pub struct IfExpr {
    pub span: Span,
    pub cond: IfCond,
    pub then_block: Block,
    pub else_branch: Option<ElseBranch>,
}

#[derive(Clone, Debug)]
pub enum IfCond {
    Expr(Box<Expr>),
    Let { pat: Pat, expr: Box<Expr> },
}

#[derive(Clone, Debug)]
pub enum ElseBranch {
    Block(Block),
    If { is_comptime: bool, data: Box<IfExpr> },
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub span: Span,
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: MatchArmBody,
}

#[derive(Clone, Debug)]
pub enum MatchArmBody {
    Block(Block),
    Expr(Expr),
    Jump(JumpAction),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    AddWrapAssign,
    SubWrapAssign,
    MulWrapAssign,
    AddSatAssign,
    SubSatAssign,
    MulSatAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Borrow,
    BorrowMut,
    Move,
    Copy,
    Try,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    BitOr,
    BitXor,
    BitAnd,
    Shl,
    Shr,
    ShlWrap,
    Add,
    Sub,
    AddWrap,
    SubWrap,
    AddSat,
    SubSat,
    AddCheck,
    SubCheck,
    Mul,
    Div,
    Mod,
    MulWrap,
    MulSat,
    MulCheck,
    DivCheck,
    ModCheck,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    IntLit {
        raw: Symbol,
        suffix: Option<Symbol>,
    },
    FloatLit {
        raw: Symbol,
        suffix: Option<Symbol>,
    },
    StringLit(String),
    RawStringLit(String),
    ByteStringLit(Vec<u8>),
    CharLit(char),
    ByteCharLit(u8),
    BoolLit(bool),
    Null,
    SelfVal,
    ReturnVal,
    Ident(Symbol),
    EnumLit(IdentSpan),
    AnonStructLit(Vec<FieldInit>),
    NamedStructLit {
        target: Box<Expr>,
        fields: Vec<FieldInit>,
    },
    ArrayLit(Vec<Expr>),
    ArrayRepeat {
        val: Box<Expr>,
        count: Box<Expr>,
    },
    Tuple(Vec<Expr>),
    Paren(Box<Expr>),
    Closure {
        params: Vec<Param>,
        body: Block,
    },
    If {
        is_comptime: bool,
        data: Box<IfExpr>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Loop {
        label: Option<IdentSpan>,
        body: Block,
    },
    ComptimeBlock(Block),
    ComptimeExpr(Box<Expr>),
    Builtin {
        name: IdentSpan,
        args: Vec<GenericArg>,
    },
    Asm {
        template: String,
        operands: Vec<AsmOperand>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Is {
        expr: Box<Expr>,
        pat: Pat,
    },
    Orelse {
        lhs: Box<Expr>,
        rhs: CatchBody,
    },
    Catch {
        lhs: Box<Expr>,
        err_bind: Option<IdentSpan>,
        rhs: CatchBody,
    },
    Field {
        base: Box<Expr>,
        field: IdentSpan,
    },
    Deref(Box<Expr>),
    GenericApply {
        base: Box<Expr>,
        args: Vec<GenericArg>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Slice {
        base: Box<Expr>,
        lo: Option<Box<Expr>>,
        hi: Option<Box<Expr>>,
    },
    Error,
}

#[derive(Clone, Debug)]
pub enum CatchBody {
    Block(Block),
    Expr(Box<Expr>),
    Jump(JumpAction),
}

#[derive(Clone, Debug)]
pub struct FieldInit {
    pub span: Span,
    pub name: Option<IdentSpan>,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct AsmOperand {
    pub span: Span,
    pub kind: AsmOperandKind,
}

#[derive(Clone, Debug)]
pub enum AsmOperandKind {
    Named {
        dir: IdentSpan,
        constraint: Expr,
        arg: Option<Expr>,
    },
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Pat {
    pub span: Span,
    pub kind: PatKind,
}

#[derive(Clone, Debug)]
pub enum PatKind {
    Wildcard,
    Ident(Symbol),
    IntLit {
        neg: bool,
        raw: Symbol,
        range_end: Option<(RangeKind, bool, Symbol)>,
    },
    CharLit {
        ch: char,
        range_end: Option<(RangeKind, char)>,
    },
    BoolLit(bool),
    Null,
    Variant {
        name: IdentSpan,
        args: Vec<Pat>,
    },
    Struct {
        path: Option<TypePath>,
        fields: Vec<FieldPat>,
    },
    Tuple(Vec<Pat>),
    Error,
}

#[derive(Clone, Debug)]
pub struct FieldPat {
    pub span: Span,
    pub name: IdentSpan,
    pub pat: Option<Pat>,
}

fn push_indent(depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn write_ident_path(segs: &[IdentSpan], interner: &Interner, out: &mut String) {
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        out.push_str(interner.get(seg.sym));
    }
}

fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_attr(attr: &Attr, interner: &Interner, out: &mut String) {
    out.push('@');
    out.push_str(interner.get(attr.name.sym));
    if attr.has_parens || !attr.args.is_empty() {
        out.push('(');
        for (i, arg) in attr.args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            if let Some(n) = arg.name {
                out.push_str(interner.get(n.sym));
                out.push('=');
            }
            write_expr_inline(&arg.value, interner, out);
        }
        out.push(')');
    }
}

fn write_attrs_inline(attrs: &[Attr], interner: &Interner, out: &mut String) {
    for attr in attrs {
        write_attr(attr, interner, out);
        out.push(' ');
    }
}

fn write_generic_params(params: &[GenericParam], interner: &Interner, out: &mut String) {
    if params.is_empty() {
        return;
    }
    out.push('[');
    for (i, gp) in params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        if gp.is_comptime {
            out.push_str("comptime ");
        }
        out.push_str(interner.get(gp.name.sym));
        if let Some(cty) = &gp.comptime_ty {
            out.push_str(": ");
            write_type_inline(cty, interner, out);
        }
        if !gp.bounds.is_empty() {
            out.push_str(": ");
            for (j, b) in gp.bounds.iter().enumerate() {
                if j > 0 {
                    out.push_str(" + ");
                }
                write_type_inline(b, interner, out);
            }
        }
    }
    out.push(']');
}

fn write_generic_args(args: &[GenericArg], interner: &Interner, out: &mut String) {
    if args.is_empty() {
        return;
    }
    out.push('[');
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match &arg.kind {
            GenericArgKind::Type(ty) => {
                out.push_str("ty:");
                write_type_inline(ty, interner, out);
            }
            GenericArgKind::Expr(e) => {
                out.push_str("expr:");
                write_expr_inline(e, interner, out);
            }
            GenericArgKind::TypeOrConst(segs) => {
                out.push_str("sym:");
                write_ident_path(segs, interner, out);
            }
        }
    }
    out.push(']');
}

fn write_param_inline(p: &Param, interner: &Interner, out: &mut String) {
    out.push('(');
    write_attrs_inline(&p.attrs, interner, out);
    if p.is_comptime {
        out.push_str("comptime ");
    }
    if let Some(m) = p.modifier {
        match m {
            ParamModifier::Mut => out.push_str("mut "),
            ParamModifier::From => out.push_str("from "),
        }
    }
    if let Some(acc) = p.access {
        match acc {
            AccessMode::Read => out.push_str("read "),
            AccessMode::Write => out.push_str("write "),
        }
    }
    if p.is_self {
        out.push_str("self");
    } else {
        out.push_str(interner.get(p.name.sym));
    }
    out.push_str(": ");
    write_type_inline(&p.ty, interner, out);
    out.push(')');
}

fn write_sys_param_inline(p: &SysParam, interner: &Interner, out: &mut String) {
    out.push('(');
    write_attrs_inline(&p.attrs, interner, out);
    if p.is_comptime {
        out.push_str("comptime ");
    }
    if let Some(m) = p.modifier {
        match m {
            ParamModifier::Mut => out.push_str("mut "),
            ParamModifier::From => out.push_str("from "),
        }
    }
    if let Some(acc) = p.access {
        match acc {
            AccessMode::Read => out.push_str("read "),
            AccessMode::Write => out.push_str("write "),
        }
    }
    if p.is_self {
        out.push_str("self");
    } else {
        out.push_str(interner.get(p.name.sym));
    }
    out.push_str(": ");
    write_type_inline(&p.ty, interner, out);
    if !p.columns.is_empty() {
        out.push_str(" cols[");
        for (i, c) in p.columns.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(interner.get(c.sym));
        }
        out.push(']');
    }
    out.push(')');
}

fn write_field_inline(f: &Field, interner: &Interner, out: &mut String) {
    out.push_str("(field ");
    write_attrs_inline(&f.attrs, interner, out);
    if f.is_pub {
        out.push_str("pub ");
    }
    out.push_str(interner.get(f.name.sym));
    out.push_str(": ");
    write_type_inline(&f.ty, interner, out);
    if let Some(def) = &f.default {
        out.push_str(" = ");
        write_expr_inline(def, interner, out);
    }
    out.push(')');
}

fn write_type_path_inline(tp: &TypePath, interner: &Interner, out: &mut String) {
    write_ident_path(&tp.segments, interner, out);
    write_generic_args(&tp.generics, interner, out);
}

fn write_type_inline(ty: &TypeExpr, interner: &Interner, out: &mut String) {
    match &ty.kind {
        TypeKind::ErrorUnion { ok, errors } => {
            out.push_str("(err-union ");
            if let Some(ok_ty) = ok {
                write_type_inline(ok_ty, interner, out);
            } else {
                out.push('_');
            }
            for e in errors {
                out.push(' ');
                write_type_inline(e, interner, out);
            }
            out.push(')');
        }
        TypeKind::Optional(inner) => {
            out.push_str("(opt ");
            write_type_inline(inner, interner, out);
            out.push(')');
        }
        TypeKind::Borrow { is_mut, target } => {
            out.push_str(if *is_mut { "(borrow-mut " } else { "(borrow " });
            match target {
                BorrowTarget::Single(inner) => {
                    out.push_str("single ");
                    write_type_inline(inner, interner, out);
                }
                BorrowTarget::Slice(inner) => {
                    out.push_str("slice ");
                    write_type_inline(inner, interner, out);
                }
                BorrowTarget::Str => {
                    out.push_str("str");
                }
            }
            out.push(')');
        }
        TypeKind::Array { len, elem } => {
            out.push_str("(array ");
            write_expr_inline(len, interner, out);
            out.push(' ');
            write_type_inline(elem, interner, out);
            out.push(')');
        }
        TypeKind::Dyn(tp) => {
            out.push_str("(dyn ");
            write_type_path_inline(tp, interner, out);
            out.push(')');
        }
        TypeKind::FnPtr { params, ret, effects } => {
            out.push_str("(fn-ptr (");
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                write_type_inline(p, interner, out);
            }
            out.push_str(") ");
            if let Some(r) = ret {
                write_type_inline(r, interner, out);
            } else {
                out.push('_');
            }
            if !effects.is_empty() {
                out.push_str(" (");
                for (i, ef) in effects.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    write_attr(ef, interner, out);
                }
                out.push(')');
            }
            out.push(')');
        }
        TypeKind::InlineStruct(fields) => {
            out.push_str("(inline-struct");
            for f in fields {
                out.push(' ');
                write_field_inline(f, interner, out);
            }
            out.push(')');
        }
        TypeKind::Tuple(elems) => {
            out.push_str("(tuple");
            for e in elems {
                out.push(' ');
                write_type_inline(e, interner, out);
            }
            out.push(')');
        }
        TypeKind::Void => out.push_str("void"),
        TypeKind::Never => out.push_str("never"),
        TypeKind::TypeType => out.push_str("type"),
        TypeKind::SelfType => out.push_str("Self"),
        TypeKind::Inferred => out.push('_'),
        TypeKind::Path(tp) => write_type_path_inline(tp, interner, out),
        TypeKind::Error => out.push_str("<ty-err>"),
    }
}

fn write_pat_inline(pat: &Pat, interner: &Interner, out: &mut String) {
    match &pat.kind {
        PatKind::Wildcard => out.push('_'),
        PatKind::Ident(sym) => {
            let _ = write!(out, "(pat-ident {})", interner.get(*sym));
        }
        PatKind::IntLit { neg, raw, range_end } => {
            out.push_str("(pat-int ");
            if *neg {
                out.push('-');
            }
            out.push_str(interner.get(*raw));
            if let Some((rk, rneg, rraw)) = range_end {
                out.push_str(match rk {
                    RangeKind::Exclusive => "..",
                    RangeKind::Inclusive => "..=",
                });
                if *rneg {
                    out.push('-');
                }
                out.push_str(interner.get(*rraw));
            }
            out.push(')');
        }
        PatKind::CharLit { ch, range_end } => {
            let _ = write!(out, "(pat-char {}", *ch as u32);
            if let Some((rk, end_ch)) = range_end {
                let op = match rk {
                    RangeKind::Exclusive => "..",
                    RangeKind::Inclusive => "..=",
                };
                let _ = write!(out, "{}{}", op, *end_ch as u32);
            }
            out.push(')');
        }
        PatKind::BoolLit(b) => {
            let _ = write!(out, "(pat-bool {b})");
        }
        PatKind::Null => out.push_str("(pat-null)"),
        PatKind::Variant { name, args } => {
            let _ = write!(out, "(pat-variant .{}", interner.get(name.sym));
            for a in args {
                out.push(' ');
                write_pat_inline(a, interner, out);
            }
            out.push(')');
        }
        PatKind::Struct { path, fields } => {
            out.push_str("(pat-struct ");
            if let Some(tp) = path {
                write_type_path_inline(tp, interner, out);
            } else {
                out.push('_');
            }
            for fp in fields {
                let _ = write!(out, " ({}", interner.get(fp.name.sym));
                if let Some(sub) = &fp.pat {
                    out.push(' ');
                    write_pat_inline(sub, interner, out);
                }
                out.push(')');
            }
            out.push(')');
        }
        PatKind::Tuple(pats) => {
            out.push_str("(pat-tuple");
            for p in pats {
                out.push(' ');
                write_pat_inline(p, interner, out);
            }
            out.push(')');
        }
        PatKind::Error => out.push_str("(pat-err)"),
    }
}

fn write_jump_inline(j: &JumpAction, interner: &Interner, out: &mut String) {
    match j {
        JumpAction::Return(val) => {
            out.push_str("(return");
            if let Some(e) = val {
                out.push(' ');
                write_expr_inline(e, interner, out);
            }
            out.push(')');
        }
        JumpAction::Break { label, value } => {
            out.push_str("(break");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            if let Some(v) = value {
                out.push(' ');
                write_expr_inline(v, interner, out);
            }
            out.push(')');
        }
        JumpAction::Continue { label } => {
            out.push_str("(continue");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(')');
        }
        JumpAction::Unreachable => out.push_str("(unreachable)"),
    }
}

fn write_if_cond_inline(cond: &IfCond, interner: &Interner, out: &mut String) {
    match cond {
        IfCond::Expr(e) => write_expr_inline(e, interner, out),
        IfCond::Let { pat, expr } => {
            out.push_str("(if-let ");
            write_pat_inline(pat, interner, out);
            out.push(' ');
            write_expr_inline(expr, interner, out);
            out.push(')');
        }
    }
}

fn write_catch_body_inline(cb: &CatchBody, interner: &Interner, out: &mut String) {
    match cb {
        CatchBody::Block(b) => write_block_inline(b, interner, out),
        CatchBody::Expr(e) => write_expr_inline(e, interner, out),
        CatchBody::Jump(j) => write_jump_inline(j, interner, out),
    }
}

fn write_match_arm_inline(arm: &MatchArm, interner: &Interner, out: &mut String) {
    out.push_str("(arm ");
    write_pat_inline(&arm.pat, interner, out);
    if let Some(g) = &arm.guard {
        out.push_str(" (guard ");
        write_expr_inline(g, interner, out);
        out.push(')');
    }
    out.push(' ');
    match &arm.body {
        MatchArmBody::Block(b) => write_block_inline(b, interner, out),
        MatchArmBody::Expr(e) => write_expr_inline(e, interner, out),
        MatchArmBody::Jump(j) => write_jump_inline(j, interner, out),
    }
    out.push(')');
}

fn write_field_init_inline(fi: &FieldInit, interner: &Interner, out: &mut String) {
    out.push('(');
    if let Some(n) = fi.name {
        out.push_str(interner.get(n.sym));
        out.push_str(": ");
    }
    write_expr_inline(&fi.value, interner, out);
    out.push(')');
}

fn write_stmt_inline(stmt: &Stmt, interner: &Interner, out: &mut String) {
    match &stmt.kind {
        StmtKind::Let { attrs, pat, ty, init } => {
            out.push_str("(let ");
            write_attrs_inline(attrs, interner, out);
            write_pat_inline(pat, interner, out);
            if let Some(t) = ty {
                out.push_str(" : ");
                write_type_inline(t, interner, out);
            }
            out.push_str(" = ");
            write_expr_inline(init, interner, out);
            out.push(')');
        }
        StmtKind::Var { attrs, name, ty, init } => {
            out.push_str("(var ");
            write_attrs_inline(attrs, interner, out);
            out.push_str(interner.get(name.sym));
            if let Some(t) = ty {
                out.push_str(" : ");
                write_type_inline(t, interner, out);
            }
            out.push_str(" = ");
            write_expr_inline(init, interner, out);
            out.push(')');
        }
        StmtKind::Const(c) => {
            let _ = write!(out, "(const {} : ", interner.get(c.name.sym));
            write_type_inline(&c.ty, interner, out);
            out.push_str(" = ");
            write_expr_inline(&c.value, interner, out);
            out.push(')');
        }
        StmtKind::Defer(body) => {
            out.push_str("(defer ");
            match body {
                DeferBody::Block(b) => write_block_inline(b, interner, out),
                DeferBody::Expr(e) => write_expr_inline(e, interner, out),
            }
            out.push(')');
        }
        StmtKind::Errdefer { err_bind, body } => {
            out.push_str("(errdefer");
            if let Some(eb) = err_bind {
                let _ = write!(out, " {}", interner.get(eb.sym));
            }
            out.push(' ');
            match body {
                DeferBody::Block(b) => write_block_inline(b, interner, out),
                DeferBody::Expr(e) => write_expr_inline(e, interner, out),
            }
            out.push(')');
        }
        StmtKind::Jump(j) => write_jump_inline(j, interner, out),
        StmtKind::If { is_comptime, data } => {
            write_if_expr_inline(*is_comptime, data, interner, out);
        }
        StmtKind::Match { scrutinee, arms } => {
            out.push_str("(match ");
            write_expr_inline(scrutinee, interner, out);
            for arm in arms {
                out.push(' ');
                write_match_arm_inline(arm, interner, out);
            }
            out.push(')');
        }
        StmtKind::While { label, cond, where_clauses, body } => {
            out.push_str("(while");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_expr_inline(cond, interner, out);
            if !where_clauses.is_empty() {
                out.push_str(" (where");
                for w in where_clauses {
                    out.push(' ');
                    write_expr_inline(w, interner, out);
                }
                out.push(')');
            }
            out.push(' ');
            write_block_inline(body, interner, out);
            out.push(')');
        }
        StmtKind::For { label, pat, iter_start, iter_end, body } => {
            out.push_str("(for");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_pat_inline(pat, interner, out);
            out.push(' ');
            write_expr_inline(iter_start, interner, out);
            if let Some((rk, end_e)) = iter_end {
                out.push_str(match rk {
                    RangeKind::Exclusive => " .. ",
                    RangeKind::Inclusive => " ..= ",
                });
                write_expr_inline(end_e, interner, out);
            }
            out.push(' ');
            write_block_inline(body, interner, out);
            out.push(')');
        }
        StmtKind::ParFor { pat, iter_start, iter_end, body } => {
            out.push_str("(par-for ");
            write_pat_inline(pat, interner, out);
            out.push(' ');
            write_expr_inline(iter_start, interner, out);
            if let Some((rk, end_e)) = iter_end {
                out.push_str(match rk {
                    RangeKind::Exclusive => " .. ",
                    RangeKind::Inclusive => " ..= ",
                });
                write_expr_inline(end_e, interner, out);
            }
            out.push(' ');
            write_block_inline(body, interner, out);
            out.push(')');
        }
        StmtKind::Loop { label, body } => {
            out.push_str("(loop");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_block_inline(body, interner, out);
            out.push(')');
        }
        StmtKind::ComptimeBlock(b) => {
            out.push_str("(comptime-block ");
            write_block_inline(b, interner, out);
            out.push(')');
        }
        StmtKind::RawBlock(b) => {
            out.push_str("(raw-block ");
            write_block_inline(b, interner, out);
            out.push(')');
        }
        StmtKind::Assign { lhs, op, rhs } => {
            let _ = write!(out, "(assign {:?} ", op);
            write_expr_inline(lhs, interner, out);
            out.push(' ');
            write_expr_inline(rhs, interner, out);
            out.push(')');
        }
        StmtKind::Discard(e) => {
            out.push_str("(discard ");
            write_expr_inline(e, interner, out);
            out.push(')');
        }
        StmtKind::Expr(e) => {
            out.push_str("(expr ");
            write_expr_inline(e, interner, out);
            out.push(')');
        }
    }
}

fn write_block_inline(b: &Block, interner: &Interner, out: &mut String) {
    out.push_str("(block");
    for s in &b.stmts {
        out.push(' ');
        write_stmt_inline(s, interner, out);
    }
    if let Some(t) = &b.tail_expr {
        out.push_str(" (tail ");
        write_expr_inline(t, interner, out);
        out.push(')');
    }
    out.push(')');
}

fn write_if_expr_inline(is_comptime: bool, data: &IfExpr, interner: &Interner, out: &mut String) {
    out.push_str(if is_comptime { "(comptime-if " } else { "(if " });
    write_if_cond_inline(&data.cond, interner, out);
    out.push(' ');
    write_block_inline(&data.then_block, interner, out);
    if let Some(eb) = &data.else_branch {
        out.push_str(" (else ");
        match eb {
            ElseBranch::Block(b) => write_block_inline(b, interner, out),
            ElseBranch::If { is_comptime: ic, data: id } => {
                write_if_expr_inline(*ic, id, interner, out);
            }
        }
        out.push(')');
    }
    out.push(')');
}

fn write_expr_inline(expr: &Expr, interner: &Interner, out: &mut String) {
    match &expr.kind {
        ExprKind::IntLit { raw, suffix } => {
            let _ = write!(out, "(int {}", interner.get(*raw));
            if let Some(suf) = suffix {
                let _ = write!(out, "_{}", interner.get(*suf));
            }
            out.push(')');
        }
        ExprKind::FloatLit { raw, suffix } => {
            let _ = write!(out, "(float {}", interner.get(*raw));
            if let Some(suf) = suffix {
                let _ = write!(out, "_{}", interner.get(*suf));
            }
            out.push(')');
        }
        ExprKind::StringLit(s) => {
            out.push_str("(str ");
            write_escaped_str(s, out);
            out.push(')');
        }
        ExprKind::RawStringLit(s) => {
            out.push_str("(rawstr ");
            write_escaped_str(s, out);
            out.push(')');
        }
        ExprKind::ByteStringLit(bs) => {
            out.push_str("(bstr ");
            for b in bs {
                let _ = write!(out, "{:02x}", b);
            }
            out.push(')');
        }
        ExprKind::CharLit(c) => {
            let _ = write!(out, "(char {})", *c as u32);
        }
        ExprKind::ByteCharLit(b) => {
            let _ = write!(out, "(bchar {})", b);
        }
        ExprKind::BoolLit(b) => {
            let _ = write!(out, "(bool {b})");
        }
        ExprKind::Null => out.push_str("(null)"),
        ExprKind::SelfVal => out.push_str("(self)"),
        ExprKind::ReturnVal => out.push_str("(return-val)"),
        ExprKind::Ident(sym) => {
            let _ = write!(out, "(ident {})", interner.get(*sym));
        }
        ExprKind::EnumLit(id) => {
            let _ = write!(out, "(enum-lit .{})", interner.get(id.sym));
        }
        ExprKind::AnonStructLit(fields) => {
            out.push_str("(anon-struct");
            for fi in fields {
                out.push(' ');
                write_field_init_inline(fi, interner, out);
            }
            out.push(')');
        }
        ExprKind::NamedStructLit { target, fields } => {
            out.push_str("(struct-lit ");
            write_expr_inline(target, interner, out);
            for fi in fields {
                out.push(' ');
                write_field_init_inline(fi, interner, out);
            }
            out.push(')');
        }
        ExprKind::ArrayLit(elems) => {
            out.push_str("(array");
            for e in elems {
                out.push(' ');
                write_expr_inline(e, interner, out);
            }
            out.push(')');
        }
        ExprKind::ArrayRepeat { val, count } => {
            out.push_str("(array-repeat ");
            write_expr_inline(val, interner, out);
            out.push(' ');
            write_expr_inline(count, interner, out);
            out.push(')');
        }
        ExprKind::Tuple(elems) => {
            out.push_str("(tuple");
            for e in elems {
                out.push(' ');
                write_expr_inline(e, interner, out);
            }
            out.push(')');
        }
        ExprKind::Paren(inner) => {
            out.push_str("(paren ");
            write_expr_inline(inner, interner, out);
            out.push(')');
        }
        ExprKind::Closure { params, body } => {
            out.push_str("(closure (");
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                write_param_inline(p, interner, out);
            }
            out.push_str(") ");
            write_block_inline(body, interner, out);
            out.push(')');
        }
        ExprKind::If { is_comptime, data } => {
            write_if_expr_inline(*is_comptime, data, interner, out);
        }
        ExprKind::Match { scrutinee, arms } => {
            out.push_str("(match ");
            write_expr_inline(scrutinee, interner, out);
            for arm in arms {
                out.push(' ');
                write_match_arm_inline(arm, interner, out);
            }
            out.push(')');
        }
        ExprKind::Loop { label, body } => {
            out.push_str("(loop");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_block_inline(body, interner, out);
            out.push(')');
        }
        ExprKind::ComptimeBlock(b) => {
            out.push_str("(comptime-block ");
            write_block_inline(b, interner, out);
            out.push(')');
        }
        ExprKind::ComptimeExpr(e) => {
            out.push_str("(comptime ");
            write_expr_inline(e, interner, out);
            out.push(')');
        }
        ExprKind::Builtin { name, args } => {
            let _ = write!(out, "(builtin @{}", interner.get(name.sym));
            if !args.is_empty() {
                out.push(' ');
                write_generic_args(args, interner, out);
            }
            out.push(')');
        }
        ExprKind::Asm { template, operands } => {
            out.push_str("(asm ");
            write_escaped_str(template, out);
            for op in operands {
                out.push(' ');
                match &op.kind {
                    AsmOperandKind::Named { dir, constraint, arg } => {
                        let _ = write!(out, "({} ", interner.get(dir.sym));
                        write_expr_inline(constraint, interner, out);
                        if let Some(a) = arg {
                            out.push(' ');
                            write_expr_inline(a, interner, out);
                        }
                        out.push(')');
                    }
                    AsmOperandKind::Expr(e) => write_expr_inline(e, interner, out),
                }
            }
            out.push(')');
        }
        ExprKind::Unary { op, expr } => {
            let _ = write!(out, "(unary {:?} ", op);
            write_expr_inline(expr, interner, out);
            out.push(')');
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let _ = write!(out, "(bin {:?} ", op);
            write_expr_inline(lhs, interner, out);
            out.push(' ');
            write_expr_inline(rhs, interner, out);
            out.push(')');
        }
        ExprKind::Is { expr, pat } => {
            out.push_str("(is ");
            write_expr_inline(expr, interner, out);
            out.push(' ');
            write_pat_inline(pat, interner, out);
            out.push(')');
        }
        ExprKind::Orelse { lhs, rhs } => {
            out.push_str("(orelse ");
            write_expr_inline(lhs, interner, out);
            out.push(' ');
            write_catch_body_inline(rhs, interner, out);
            out.push(')');
        }
        ExprKind::Catch { lhs, err_bind, rhs } => {
            out.push_str("(catch ");
            write_expr_inline(lhs, interner, out);
            if let Some(eb) = err_bind {
                let _ = write!(out, " |{}|", interner.get(eb.sym));
            }
            out.push(' ');
            write_catch_body_inline(rhs, interner, out);
            out.push(')');
        }
        ExprKind::Field { base, field } => {
            out.push_str("(field ");
            write_expr_inline(base, interner, out);
            let _ = write!(out, " .{})", interner.get(field.sym));
        }
        ExprKind::Deref(inner) => {
            out.push_str("(deref ");
            write_expr_inline(inner, interner, out);
            out.push(')');
        }
        ExprKind::GenericApply { base, args } => {
            out.push_str("(generic-apply ");
            write_expr_inline(base, interner, out);
            out.push(' ');
            write_generic_args(args, interner, out);
            out.push(')');
        }
        ExprKind::Call { callee, args } => {
            out.push_str("(call ");
            write_expr_inline(callee, interner, out);
            for a in args {
                out.push(' ');
                write_expr_inline(a, interner, out);
            }
            out.push(')');
        }
        ExprKind::Index { base, index } => {
            out.push_str("(index ");
            write_expr_inline(base, interner, out);
            out.push(' ');
            write_expr_inline(index, interner, out);
            out.push(')');
        }
        ExprKind::Slice { base, lo, hi } => {
            out.push_str("(slice ");
            write_expr_inline(base, interner, out);
            out.push(' ');
            if let Some(l) = lo {
                write_expr_inline(l, interner, out);
            } else {
                out.push('_');
            }
            out.push(' ');
            if let Some(h) = hi {
                write_expr_inline(h, interner, out);
            } else {
                out.push('_');
            }
            out.push(')');
        }
        ExprKind::Error => out.push_str("(expr-err)"),
    }
}

fn dump_block_lines(b: &Block, interner: &Interner, depth: usize, out: &mut String) {
    push_indent(depth, out);
    out.push_str("(block\n");
    for s in &b.stmts {
        dump_stmt_lines(s, interner, depth + 1, out);
    }
    if let Some(t) = &b.tail_expr {
        push_indent(depth + 1, out);
        out.push_str("(tail ");
        write_expr_inline(t, interner, out);
        out.push_str(")\n");
    }
    push_indent(depth, out);
    out.push_str(")\n");
}

fn dump_if_lines(is_comptime: bool, data: &IfExpr, interner: &Interner, depth: usize, out: &mut String) {
    push_indent(depth, out);
    out.push_str(if is_comptime { "(comptime-if " } else { "(if " });
    write_if_cond_inline(&data.cond, interner, out);
    out.push('\n');
    dump_block_lines(&data.then_block, interner, depth + 1, out);
    if let Some(eb) = &data.else_branch {
        push_indent(depth + 1, out);
        out.push_str("(else\n");
        match eb {
            ElseBranch::Block(b) => dump_block_lines(b, interner, depth + 2, out),
            ElseBranch::If { is_comptime: ic, data: id } => {
                dump_if_lines(*ic, id, interner, depth + 2, out);
            }
        }
        push_indent(depth + 1, out);
        out.push_str(")\n");
    }
    push_indent(depth, out);
    out.push_str(")\n");
}

fn dump_stmt_lines(stmt: &Stmt, interner: &Interner, depth: usize, out: &mut String) {
    match &stmt.kind {
        StmtKind::If { is_comptime, data } => {
            dump_if_lines(*is_comptime, data, interner, depth, out);
        }
        StmtKind::Match { scrutinee, arms } => {
            push_indent(depth, out);
            out.push_str("(match ");
            write_expr_inline(scrutinee, interner, out);
            out.push('\n');
            for arm in arms {
                push_indent(depth + 1, out);
                out.push_str("(arm ");
                write_pat_inline(&arm.pat, interner, out);
                if let Some(g) = &arm.guard {
                    out.push_str(" (guard ");
                    write_expr_inline(g, interner, out);
                    out.push(')');
                }
                match &arm.body {
                    MatchArmBody::Block(b) => {
                        out.push('\n');
                        dump_block_lines(b, interner, depth + 2, out);
                        push_indent(depth + 1, out);
                        out.push_str(")\n");
                    }
                    MatchArmBody::Expr(e) => {
                        out.push(' ');
                        write_expr_inline(e, interner, out);
                        out.push_str(")\n");
                    }
                    MatchArmBody::Jump(j) => {
                        out.push(' ');
                        write_jump_inline(j, interner, out);
                        out.push_str(")\n");
                    }
                }
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::While { label, cond, where_clauses, body } => {
            push_indent(depth, out);
            out.push_str("(while");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_expr_inline(cond, interner, out);
            if !where_clauses.is_empty() {
                out.push_str(" (where");
                for w in where_clauses {
                    out.push(' ');
                    write_expr_inline(w, interner, out);
                }
                out.push(')');
            }
            out.push('\n');
            dump_block_lines(body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::For { label, pat, iter_start, iter_end, body } => {
            push_indent(depth, out);
            out.push_str("(for");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push(' ');
            write_pat_inline(pat, interner, out);
            out.push(' ');
            write_expr_inline(iter_start, interner, out);
            if let Some((rk, end_e)) = iter_end {
                out.push_str(match rk {
                    RangeKind::Exclusive => " .. ",
                    RangeKind::Inclusive => " ..= ",
                });
                write_expr_inline(end_e, interner, out);
            }
            out.push('\n');
            dump_block_lines(body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::ParFor { pat, iter_start, iter_end, body } => {
            push_indent(depth, out);
            out.push_str("(par-for ");
            write_pat_inline(pat, interner, out);
            out.push(' ');
            write_expr_inline(iter_start, interner, out);
            if let Some((rk, end_e)) = iter_end {
                out.push_str(match rk {
                    RangeKind::Exclusive => " .. ",
                    RangeKind::Inclusive => " ..= ",
                });
                write_expr_inline(end_e, interner, out);
            }
            out.push('\n');
            dump_block_lines(body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::Loop { label, body } => {
            push_indent(depth, out);
            out.push_str("(loop");
            if let Some(l) = label {
                let _ = write!(out, " '{}", interner.get(l.sym));
            }
            out.push('\n');
            dump_block_lines(body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::ComptimeBlock(b) => {
            push_indent(depth, out);
            out.push_str("(comptime-block\n");
            dump_block_lines(b, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::RawBlock(b) => {
            push_indent(depth, out);
            out.push_str("(raw-block\n");
            dump_block_lines(b, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::Defer(DeferBody::Block(b)) => {
            push_indent(depth, out);
            out.push_str("(defer\n");
            dump_block_lines(b, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        StmtKind::Errdefer { err_bind, body: DeferBody::Block(b) } => {
            push_indent(depth, out);
            out.push_str("(errdefer");
            if let Some(eb) = err_bind {
                let _ = write!(out, " {}", interner.get(eb.sym));
            }
            out.push('\n');
            dump_block_lines(b, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        _ => {
            push_indent(depth, out);
            write_stmt_inline(stmt, interner, out);
            out.push('\n');
        }
    }
}

fn write_clauses_suffix(
    effects: &[Attr],
    where_clauses: &[Expr],
    ensures_clauses: &[Expr],
    interner: &Interner,
    out: &mut String,
) {
    if !effects.is_empty() {
        out.push_str(" (effects");
        for ef in effects {
            out.push(' ');
            write_attr(ef, interner, out);
        }
        out.push(')');
    }
    if !where_clauses.is_empty() {
        out.push_str(" (where");
        for w in where_clauses {
            out.push(' ');
            write_expr_inline(w, interner, out);
        }
        out.push(')');
    }
    if !ensures_clauses.is_empty() {
        out.push_str(" (ensures");
        for e in ensures_clauses {
            out.push(' ');
            write_expr_inline(e, interner, out);
        }
        out.push(')');
    }
}

fn dump_decl_full(d: &Decl, interner: &Interner, depth: usize, out: &mut String) {
    push_indent(depth, out);
    out.push('(');
    write_attrs_inline(&d.attrs, interner, out);
    if d.is_pub {
        out.push_str("pub ");
    }
    match &d.kind {
        DeclKind::Fn(f) => {
            if f.is_comptime {
                out.push_str("comptime ");
            }
            if f.is_view {
                out.push_str("view ");
            }
            out.push_str("fn ");
            out.push_str(interner.get(f.name.sym));
            write_generic_params(&f.generics, interner, out);
            out.push_str(" (params");
            for p in &f.params {
                out.push(' ');
                write_param_inline(p, interner, out);
            }
            out.push(')');
            if let Some(rt) = &f.ret_type {
                out.push_str(" -> ");
                write_type_inline(rt, interner, out);
            }
            write_clauses_suffix(&f.effects, &f.where_clauses, &f.ensures_clauses, interner, out);
            out.push('\n');
            dump_block_lines(&f.body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Sys(s) => {
            out.push_str("sys ");
            out.push_str(interner.get(s.name.sym));
            write_generic_params(&s.generics, interner, out);
            out.push_str(" (params");
            for p in &s.params {
                out.push(' ');
                write_sys_param_inline(p, interner, out);
            }
            out.push(')');
            if let Some(rt) = &s.ret_type {
                out.push_str(" -> ");
                write_type_inline(rt, interner, out);
            }
            write_clauses_suffix(&s.effects, &s.where_clauses, &s.ensures_clauses, interner, out);
            out.push('\n');
            dump_block_lines(&s.body, interner, depth + 1, out);
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Struct(s) => {
            if s.is_linear {
                out.push_str("linear ");
            }
            if s.is_view {
                out.push_str("view ");
            }
            out.push_str("struct ");
            out.push_str(interner.get(s.name.sym));
            write_generic_params(&s.generics, interner, out);
            write_clauses_suffix(&[], &s.where_clauses, &[], interner, out);
            out.push('\n');
            for f in &s.fields {
                push_indent(depth + 1, out);
                write_field_inline(f, interner, out);
                out.push('\n');
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Pool(p) => {
            out.push_str("pool ");
            out.push_str(interner.get(p.name.sym));
            write_generic_params(&p.generics, interner, out);
            write_clauses_suffix(&[], &p.where_clauses, &[], interner, out);
            out.push('\n');
            for f in &p.fields {
                push_indent(depth + 1, out);
                write_field_inline(f, interner, out);
                out.push('\n');
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Enum(e) => {
            out.push_str("enum ");
            out.push_str(interner.get(e.name.sym));
            if let Some(r) = e.repr_int {
                let _ = write!(out, " : {}", interner.get(r.sym));
            }
            out.push('\n');
            for v in &e.variants {
                push_indent(depth + 1, out);
                let _ = write!(out, "(variant {}", interner.get(v.name.sym));
                if let Some(val) = &v.value {
                    out.push_str(" = ");
                    write_expr_inline(val, interner, out);
                }
                out.push_str(")\n");
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Union(u) => {
            if u.is_linear {
                out.push_str("linear ");
            }
            out.push_str("union ");
            out.push_str(interner.get(u.name.sym));
            write_generic_params(&u.generics, interner, out);
            write_clauses_suffix(&[], &u.where_clauses, &[], interner, out);
            out.push('\n');
            for v in &u.variants {
                push_indent(depth + 1, out);
                let _ = write!(out, "(variant {}", interner.get(v.name.sym));
                if let Some(ty) = &v.ty {
                    out.push_str(": ");
                    write_type_inline(ty, interner, out);
                }
                out.push_str(")\n");
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Bitfield(b) => {
            let _ = writeln!(
                out,
                "bitfield {} : {}",
                interner.get(b.name.sym),
                interner.get(b.base_int.sym)
            );
            for m in &b.members {
                push_indent(depth + 1, out);
                let _ = write!(
                    out,
                    "(member {}: {}",
                    interner.get(m.name.sym),
                    interner.get(m.ty.sym)
                );
                if let Some((lo, hi)) = &m.bits {
                    out.push_str(" bits ");
                    write_expr_inline(lo, interner, out);
                    out.push_str("..");
                    write_expr_inline(hi, interner, out);
                }
                out.push_str(")\n");
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Iface(i) => {
            out.push_str("iface ");
            out.push_str(interner.get(i.name.sym));
            write_generic_params(&i.generics, interner, out);
            out.push('\n');
            for m in &i.methods {
                push_indent(depth + 1, out);
                out.push('(');
                write_attrs_inline(&m.attrs, interner, out);
                if m.is_view {
                    out.push_str("view ");
                }
                out.push_str("fn ");
                out.push_str(interner.get(m.name.sym));
                write_generic_params(&m.generics, interner, out);
                out.push_str(" (params");
                for p in &m.params {
                    out.push(' ');
                    write_param_inline(p, interner, out);
                }
                out.push(')');
                if let Some(rt) = &m.ret_type {
                    out.push_str(" -> ");
                    write_type_inline(rt, interner, out);
                }
                write_clauses_suffix(&m.effects, &m.where_clauses, &m.ensures_clauses, interner, out);
                out.push_str(")\n");
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Impl(i) => {
            out.push_str("impl");
            write_generic_params(&i.generics, interner, out);
            out.push(' ');
            if let Some(ifc) = &i.iface_type {
                write_type_inline(ifc, interner, out);
                out.push_str(" for ");
            }
            write_type_inline(&i.target_type, interner, out);
            write_clauses_suffix(&[], &i.where_clauses, &[], interner, out);
            out.push('\n');
            for sd in &i.decls {
                dump_decl_full(sd, interner, depth + 1, out);
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::Error(e) => {
            out.push_str("error ");
            out.push_str(interner.get(e.name.sym));
            for v in &e.variants {
                let _ = write!(out, " {}", interner.get(v.sym));
            }
            out.push_str(")\n");
        }
        DeclKind::TypeAlias(t) => {
            out.push_str("type ");
            out.push_str(interner.get(t.name.sym));
            write_generic_params(&t.generics, interner, out);
            write_clauses_suffix(&[], &t.where_clauses, &[], interner, out);
            out.push_str(" = ");
            write_type_inline(&t.ty, interner, out);
            out.push_str(")\n");
        }
        DeclKind::Const(c) => {
            let _ = write!(out, "const {}: ", interner.get(c.name.sym));
            write_type_inline(&c.ty, interner, out);
            out.push_str(" = ");
            write_expr_inline(&c.value, interner, out);
            out.push_str(")\n");
        }
        DeclKind::ExternFn(e) => {
            let _ = write!(
                out,
                "extern \"{}\" fn {} (params",
                interner.get(e.abi),
                interner.get(e.name.sym)
            );
            for p in &e.params {
                out.push(' ');
                write_param_inline(p, interner, out);
            }
            out.push(')');
            if let Some(rt) = &e.ret_type {
                out.push_str(" -> ");
                write_type_inline(rt, interner, out);
            }
            out.push_str(")\n");
        }
        DeclKind::ExternStruct(e) => {
            let _ = writeln!(
                out,
                "extern \"{}\" struct {}",
                interner.get(e.abi),
                interner.get(e.name.sym)
            );
            for f in &e.fields {
                push_indent(depth + 1, out);
                write_field_inline(f, interner, out);
                out.push('\n');
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
        DeclKind::ExternUnion(e) => {
            let _ = writeln!(
                out,
                "extern \"{}\" union {}",
                interner.get(e.abi),
                interner.get(e.name.sym)
            );
            for f in &e.fields {
                push_indent(depth + 1, out);
                write_field_inline(f, interner, out);
                out.push('\n');
            }
            push_indent(depth, out);
            out.push_str(")\n");
        }
    }
}
