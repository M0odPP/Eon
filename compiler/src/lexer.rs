use std::fmt;

use crate::diag::{DiagBag, ErrorCode};
use crate::span::{FileId, SourceFile, Span};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumLitId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StrLitId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ByteStrLitId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumLitData {
    pub raw_start: u32,
    pub raw_end: u32,
    pub suffix_start: u32,
    pub suffix_end: u32,
    pub has_suffix: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DocComment {
    pub is_module_doc: bool,
    pub span: Span,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Identifiers & special token `_`
    Ident,
    Underscore,
    Label,

    // Literals
    IntLit(NumLitId),
    FloatLit(NumLitId),
    StringLit(StrLitId),
    RawStringLit(StrLitId),
    ByteStringLit(ByteStrLitId),
    CharLit(char),
    ByteCharLit(u8),

    // Keywords: Declarations (21)
    KwModule,
    KwImport,
    KwPub,
    KwFn,
    KwSys,
    KwLet,
    KwVar,
    KwConst,
    KwType,
    KwStruct,
    KwLinear,
    KwPool,
    KwEnum,
    KwUnion,
    KwBitfield,
    KwIface,
    KwImpl,
    KwError,
    KwExtern,
    KwComptime,
    KwView,

    // Keywords: Control flow (16)
    KwIf,
    KwElse,
    KwMatch,
    KwWhile,
    KwFor,
    KwPar,
    KwLoop,
    KwBreak,
    KwContinue,
    KwReturn,
    KwDefer,
    KwErrdefer,
    KwTry,
    KwCatch,
    KwOrelse,
    KwUnreachable,

    // Keywords: Modifiers & contracts (9)
    KwMut,
    KwMove,
    KwCopy,
    KwWhere,
    KwEnsures,
    KwIn,
    KwAs,
    KwIs,
    KwFrom,

    // Keywords: Literals & special names (8)
    KwTrue,
    KwFalse,
    KwNull,
    KwVoid,
    KwNever,
    KwSelf,
    KwSelfType,
    KwDyn,

    // Keywords: HAL-only (3)
    KwRaw,
    KwAsm,
    KwVolatile,

    // Delimiters & punctuation
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    Semi,      // ;
    Colon,     // :
    Comma,     // ,
    Dot,       // .
    DotStar,   // .*
    DotDot,    // ..
    DotDotEq,  // ..=
    Arrow,     // ->
    FatArrow,  // =>
    At,        // @
    Question,  // ?

    // Operators
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    PlusWrap,    // +%
    MinusWrap,   // -%
    StarWrap,    // *%
    PlusSat,     // +|
    MinusSat,    // -|
    StarSat,     // *|
    PlusCheck,   // +?
    MinusCheck,  // -?
    StarCheck,   // *?
    SlashCheck,  // /?
    PercentCheck,// %?

    Shl,     // <<
    Shr,     // >>
    ShlWrap, // <<%

    Amp,    // &
    Pipe,   // |
    Caret,  // ^
    Tilde,  // ~
    Bang,   // !
    AndAnd, // &&
    OrOr,   // ||

    Eq,    // =
    EqEq,  // ==
    NotEq, // !=
    Lt,    // <
    LtEq,  // <=
    Gt,    // >
    GtEq,  // >=

    // Compound assignments
    PlusEq,      // +=
    MinusEq,     // -=
    StarEq,      // *=
    PlusWrapEq,  // +%=
    MinusWrapEq, // -%=
    StarWrapEq,  // *%=
    PlusSatEq,   // +|=
    MinusSatEq,  // -|=
    StarSatEq,   // *|=
    AmpEq,       // &=
    PipeEq,      // |=
    CaretEq,     // ^=
    ShlEq,       // <<=
    ShrEq,       // >>=

    Error,
    Eof,
}

impl TokenKind {
    pub fn from_keyword(s: &str) -> Option<TokenKind> {
        Some(match s {
            // Declarations
            "module" => Self::KwModule,
            "import" => Self::KwImport,
            "pub" => Self::KwPub,
            "fn" => Self::KwFn,
            "sys" => Self::KwSys,
            "let" => Self::KwLet,
            "var" => Self::KwVar,
            "const" => Self::KwConst,
            "type" => Self::KwType,
            "struct" => Self::KwStruct,
            "linear" => Self::KwLinear,
            "pool" => Self::KwPool,
            "enum" => Self::KwEnum,
            "union" => Self::KwUnion,
            "bitfield" => Self::KwBitfield,
            "iface" => Self::KwIface,
            "impl" => Self::KwImpl,
            "error" => Self::KwError,
            "extern" => Self::KwExtern,
            "comptime" => Self::KwComptime,
            "view" => Self::KwView,

            // Control flow
            "if" => Self::KwIf,
            "else" => Self::KwElse,
            "match" => Self::KwMatch,
            "while" => Self::KwWhile,
            "for" => Self::KwFor,
            "par" => Self::KwPar,
            "loop" => Self::KwLoop,
            "break" => Self::KwBreak,
            "continue" => Self::KwContinue,
            "return" => Self::KwReturn,
            "defer" => Self::KwDefer,
            "errdefer" => Self::KwErrdefer,
            "try" => Self::KwTry,
            "catch" => Self::KwCatch,
            "orelse" => Self::KwOrelse,
            "unreachable" => Self::KwUnreachable,

            // Modifiers & contracts
            "mut" => Self::KwMut,
            "move" => Self::KwMove,
            "copy" => Self::KwCopy,
            "where" => Self::KwWhere,
            "ensures" => Self::KwEnsures,
            "in" => Self::KwIn,
            "as" => Self::KwAs,
            "is" => Self::KwIs,
            "from" => Self::KwFrom,

            // Literals & special names
            "true" => Self::KwTrue,
            "false" => Self::KwFalse,
            "null" => Self::KwNull,
            "void" => Self::KwVoid,
            "never" => Self::KwNever,
            "self" => Self::KwSelf,
            "Self" => Self::KwSelfType,
            "dyn" => Self::KwDyn,

            // HAL-only
            "raw" => Self::KwRaw,
            "asm" => Self::KwAsm,
            "volatile" => Self::KwVolatile,

            _ => return None,
        })
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Ident => "identifier",
            Self::Underscore => "`_`",
            Self::Label => "loop label",
            Self::IntLit(_) => "integer literal",
            Self::FloatLit(_) => "float literal",
            Self::StringLit(_) => "string literal",
            Self::RawStringLit(_) => "raw string literal",
            Self::ByteStringLit(_) => "byte string literal",
            Self::CharLit(_) => "character literal",
            Self::ByteCharLit(_) => "byte character literal",
            Self::KwModule => "`module`",
            Self::KwImport => "`import`",
            Self::KwPub => "`pub`",
            Self::KwFn => "`fn`",
            Self::KwSys => "`sys`",
            Self::KwLet => "`let`",
            Self::KwVar => "`var`",
            Self::KwConst => "`const`",
            Self::KwType => "`type`",
            Self::KwStruct => "`struct`",
            Self::KwLinear => "`linear`",
            Self::KwPool => "`pool`",
            Self::KwEnum => "`enum`",
            Self::KwUnion => "`union`",
            Self::KwBitfield => "`bitfield`",
            Self::KwIface => "`iface`",
            Self::KwImpl => "`impl`",
            Self::KwError => "`error`",
            Self::KwExtern => "`extern`",
            Self::KwComptime => "`comptime`",
            Self::KwView => "`view`",
            Self::KwIf => "`if`",
            Self::KwElse => "`else`",
            Self::KwMatch => "`match`",
            Self::KwWhile => "`while`",
            Self::KwFor => "`for`",
            Self::KwPar => "`par`",
            Self::KwLoop => "`loop`",
            Self::KwBreak => "`break`",
            Self::KwContinue => "`continue`",
            Self::KwReturn => "`return`",
            Self::KwDefer => "`defer`",
            Self::KwErrdefer => "`errdefer`",
            Self::KwTry => "`try`",
            Self::KwCatch => "`catch`",
            Self::KwOrelse => "`orelse`",
            Self::KwUnreachable => "`unreachable`",
            Self::KwMut => "`mut`",
            Self::KwMove => "`move`",
            Self::KwCopy => "`copy`",
            Self::KwWhere => "`where`",
            Self::KwEnsures => "`ensures`",
            Self::KwIn => "`in`",
            Self::KwAs => "`as`",
            Self::KwIs => "`is`",
            Self::KwFrom => "`from`",
            Self::KwTrue => "`true`",
            Self::KwFalse => "`false`",
            Self::KwNull => "`null`",
            Self::KwVoid => "`void`",
            Self::KwNever => "`never`",
            Self::KwSelf => "`self`",
            Self::KwSelfType => "`Self`",
            Self::KwDyn => "`dyn`",
            Self::KwRaw => "`raw`",
            Self::KwAsm => "`asm`",
            Self::KwVolatile => "`volatile`",
            Self::LParen => "`(`",
            Self::RParen => "`)`",
            Self::LBracket => "`[`",
            Self::RBracket => "`]`",
            Self::LBrace => "`{`",
            Self::RBrace => "`}`",
            Self::Semi => "`;`",
            Self::Colon => "`:`",
            Self::Comma => "`,`",
            Self::Dot => "`.`",
            Self::DotStar => "`.*`",
            Self::DotDot => "`..`",
            Self::DotDotEq => "`..=`",
            Self::Arrow => "`->`",
            Self::FatArrow => "`=>`",
            Self::At => "`@`",
            Self::Question => "`?`",
            Self::Plus => "`+`",
            Self::Minus => "`-`",
            Self::Star => "`*`",
            Self::Slash => "`/`",
            Self::Percent => "`%`",
            Self::PlusWrap => "`+%`",
            Self::MinusWrap => "`-%`",
            Self::StarWrap => "`*%`",
            Self::PlusSat => "`+|`",
            Self::MinusSat => "`-|`",
            Self::StarSat => "`*|`",
            Self::PlusCheck => "`+?`",
            Self::MinusCheck => "`-?`",
            Self::StarCheck => "`*?`",
            Self::SlashCheck => "`/?`",
            Self::PercentCheck => "`%?`",
            Self::Shl => "`<<`",
            Self::Shr => "`>>`",
            Self::ShlWrap => "`<<%`",
            Self::Amp => "`&`",
            Self::Pipe => "`|`",
            Self::Caret => "`^`",
            Self::Tilde => "`~`",
            Self::Bang => "`!`",
            Self::AndAnd => "`&&`",
            Self::OrOr => "`||`",
            Self::Eq => "`=`",
            Self::EqEq => "`==`",
            Self::NotEq => "`!=`",
            Self::Lt => "`<`",
            Self::LtEq => "`<=`",
            Self::Gt => "`>`",
            Self::GtEq => "`>=`",
            Self::PlusEq => "`+=`",
            Self::MinusEq => "`-=`",
            Self::StarEq => "`*=`",
            Self::PlusWrapEq => "`+%=`",
            Self::MinusWrapEq => "`-%=`",
            Self::StarWrapEq => "`*%=`",
            Self::PlusSatEq => "`+|=`",
            Self::MinusSatEq => "`-|=`",
            Self::StarSatEq => "`*|=`",
            Self::AmpEq => "`&=`",
            Self::PipeEq => "`|=`",
            Self::CaretEq => "`^=`",
            Self::ShlEq => "`<<=`",
            Self::ShrEq => "`>>=`",
            Self::Error => "invalid token",
            Self::Eof => "end of file",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

pub struct LexedFile {
    pub tokens: Vec<Token>,
    pub num_lits: Vec<NumLitData>,
    pub string_lits: Vec<String>,
    pub byte_string_lits: Vec<Vec<u8>>,
    pub doc_comments: Vec<DocComment>,
}

impl LexedFile {
    pub fn num_raw<'a>(&self, src: &'a str, id: NumLitId) -> &'a str {
        let d = self.num_lits[id.0 as usize];
        &src[d.raw_start as usize..d.raw_end as usize]
    }

    pub fn num_suffix<'a>(&self, src: &'a str, id: NumLitId) -> Option<&'a str> {
        let d = self.num_lits[id.0 as usize];
        if d.has_suffix {
            Some(&src[d.suffix_start as usize..d.suffix_end as usize])
        } else {
            None
        }
    }
}

pub fn lex_file(file: &SourceFile, diags: &mut DiagBag) -> LexedFile {
    for issue in &file.utf8_issues {
        diags.error(ErrorCode::E0001, issue.span, issue.message);
    }
    let mut lexer = Lexer::new(file, diags);
    lexer.check_trojan_source();
    lexer.lex_all();
    lexer.finish()
}

struct Lexer<'a> {
    file_id: FileId,
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    diags: &'a mut DiagBag,
    tokens: Vec<Token>,
    num_lits: Vec<NumLitData>,
    string_lits: Vec<String>,
    byte_string_lits: Vec<Vec<u8>>,
    doc_comments: Vec<DocComment>,
}

impl<'a> Lexer<'a> {
    fn new(file: &'a SourceFile, diags: &'a mut DiagBag) -> Self {
        let src = file.text();
        Self {
            file_id: file.id,
            src,
            bytes: src.as_bytes(),
            pos: 0,
            diags,
            tokens: Vec::with_capacity(src.len() / 6 + 16),
            num_lits: Vec::new(),
            string_lits: Vec::new(),
            byte_string_lits: Vec::new(),
            doc_comments: Vec::new(),
        }
    }

    fn finish(self) -> LexedFile {
        LexedFile {
            tokens: self.tokens,
            num_lits: self.num_lits,
            string_lits: self.string_lits,
            byte_string_lits: self.byte_string_lits,
            doc_comments: self.doc_comments,
        }
    }

    #[inline]
    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file_id, start as u32, end as u32)
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn check_trojan_source(&mut self) {
        // Every bidi control code point (U+202A..=U+202E, U+2066..=U+2069) starts with 0xE2 in UTF-8.
        let mut i = 0;
        while i + 2 < self.bytes.len() {
            if self.bytes[i] == 0xE2 {
                let b1 = self.bytes[i + 1];
                let b2 = self.bytes[i + 2];
                let is_bidi = (b1 == 0x80 && (0xAA..=0xAE).contains(&b2))
                    || (b1 == 0x81 && (0xA6..=0xA9).contains(&b2));
                if is_bidi {
                    let span = self.span(i, i + 3);
                    self.diags.error(
                        ErrorCode::E0003,
                        span,
                        "forbidden Unicode bidirectional control character (Trojan Source)",
                    );
                    i += 3;
                    continue;
                }
            }
            i += 1;
        }
    }

    fn lex_all(&mut self) {
        while self.pos < self.bytes.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.bytes.len() {
                break;
            }
            let start = self.pos;
            let b = self.bytes[self.pos];
            match b {
                b'b' if self.peek_at(1) == Some(b'"') => self.lex_byte_string(start),
                b'b' if self.peek_at(1) == Some(b'\'') => self.lex_byte_char(start),
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.lex_ident_or_keyword(start),
                b'0'..=b'9' => self.lex_number(start),
                b'"' => self.lex_string(start),
                b'\'' => self.lex_char_or_label(start),
                b'\\' if self.peek_at(1) == Some(b'\\') => self.lex_raw_string(start),
                b'\\' => {
                    self.pos += 1;
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(start, self.pos),
                        "unexpected `\\`; raw multiline strings start with `\\\\`",
                    );
                    self.tokens.push(Token {
                        kind: TokenKind::Error,
                        span: self.span(start, self.pos),
                    });
                }
                b'(' => self.push_punct(TokenKind::LParen, 1),
                b')' => self.push_punct(TokenKind::RParen, 1),
                b'[' => self.push_punct(TokenKind::LBracket, 1),
                b']' => self.push_punct(TokenKind::RBracket, 1),
                b'{' => self.push_punct(TokenKind::LBrace, 1),
                b'}' => self.push_punct(TokenKind::RBrace, 1),
                b';' => self.push_punct(TokenKind::Semi, 1),
                b':' => self.push_punct(TokenKind::Colon, 1),
                b',' => self.push_punct(TokenKind::Comma, 1),
                b'@' => self.push_punct(TokenKind::At, 1),
                b'?' => self.push_punct(TokenKind::Question, 1),
                b'~' => self.push_punct(TokenKind::Tilde, 1),

                b'.' => match self.peek_at(1) {
                    Some(b'.') => {
                        if self.peek_at(2) == Some(b'=') {
                            self.push_punct(TokenKind::DotDotEq, 3);
                        } else {
                            self.push_punct(TokenKind::DotDot, 2);
                        }
                    }
                    Some(b'*') => self.push_punct(TokenKind::DotStar, 2),
                    Some(b'0'..=b'9') => {
                        self.pos += 1;
                        while matches!(self.peek(), Some(b'0'..=b'9' | b'_')) {
                            self.pos += 1;
                        }
                        let span = self.span(start, self.pos);
                        self.diags.error(
                            ErrorCode::E0004,
                            span,
                            "float literal cannot start with `.`; add a leading zero like `0.5`",
                        );
                        self.tokens.push(Token {
                            kind: TokenKind::Error,
                            span,
                        });
                    }
                    _ => self.push_punct(TokenKind::Dot, 1),
                },

                b'+' => match self.peek_at(1) {
                    Some(b'%') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::PlusWrapEq, 3)
                    }
                    Some(b'|') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::PlusSatEq, 3)
                    }
                    Some(b'%') => self.push_punct(TokenKind::PlusWrap, 2),
                    Some(b'|') => self.push_punct(TokenKind::PlusSat, 2),
                    Some(b'?') => self.push_punct(TokenKind::PlusCheck, 2),
                    Some(b'=') => self.push_punct(TokenKind::PlusEq, 2),
                    _ => self.push_punct(TokenKind::Plus, 1),
                },

                b'-' => match self.peek_at(1) {
                    Some(b'>') => self.push_punct(TokenKind::Arrow, 2),
                    Some(b'%') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::MinusWrapEq, 3)
                    }
                    Some(b'|') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::MinusSatEq, 3)
                    }
                    Some(b'%') => self.push_punct(TokenKind::MinusWrap, 2),
                    Some(b'|') => self.push_punct(TokenKind::MinusSat, 2),
                    Some(b'?') => self.push_punct(TokenKind::MinusCheck, 2),
                    Some(b'=') => self.push_punct(TokenKind::MinusEq, 2),
                    _ => self.push_punct(TokenKind::Minus, 1),
                },

                b'*' => match self.peek_at(1) {
                    Some(b'%') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::StarWrapEq, 3)
                    }
                    Some(b'|') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::StarSatEq, 3)
                    }
                    Some(b'%') => self.push_punct(TokenKind::StarWrap, 2),
                    Some(b'|') => self.push_punct(TokenKind::StarSat, 2),
                    Some(b'?') => self.push_punct(TokenKind::StarCheck, 2),
                    Some(b'=') => self.push_punct(TokenKind::StarEq, 2),
                    _ => self.push_punct(TokenKind::Star, 1),
                },

                b'/' => match self.peek_at(1) {
                    Some(b'?') => self.push_punct(TokenKind::SlashCheck, 2),
                    _ => self.push_punct(TokenKind::Slash, 1),
                },

                b'%' => match self.peek_at(1) {
                    Some(b'?') => self.push_punct(TokenKind::PercentCheck, 2),
                    _ => self.push_punct(TokenKind::Percent, 1),
                },

                b'<' => match self.peek_at(1) {
                    Some(b'<') if self.peek_at(2) == Some(b'%') => {
                        self.push_punct(TokenKind::ShlWrap, 3)
                    }
                    Some(b'<') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::ShlEq, 3)
                    }
                    Some(b'<') => self.push_punct(TokenKind::Shl, 2),
                    Some(b'=') => self.push_punct(TokenKind::LtEq, 2),
                    _ => self.push_punct(TokenKind::Lt, 1),
                },

                b'>' => match self.peek_at(1) {
                    Some(b'>') if self.peek_at(2) == Some(b'=') => {
                        self.push_punct(TokenKind::ShrEq, 3)
                    }
                    Some(b'>') => self.push_punct(TokenKind::Shr, 2),
                    Some(b'=') => self.push_punct(TokenKind::GtEq, 2),
                    _ => self.push_punct(TokenKind::Gt, 1),
                },

                b'=' => match self.peek_at(1) {
                    Some(b'=') => self.push_punct(TokenKind::EqEq, 2),
                    Some(b'>') => self.push_punct(TokenKind::FatArrow, 2),
                    _ => self.push_punct(TokenKind::Eq, 1),
                },

                b'!' => match self.peek_at(1) {
                    Some(b'=') => self.push_punct(TokenKind::NotEq, 2),
                    _ => self.push_punct(TokenKind::Bang, 1),
                },

                b'&' => match self.peek_at(1) {
                    Some(b'&') => self.push_punct(TokenKind::AndAnd, 2),
                    Some(b'=') => self.push_punct(TokenKind::AmpEq, 2),
                    _ => self.push_punct(TokenKind::Amp, 1),
                },

                b'|' => match self.peek_at(1) {
                    Some(b'|') => self.push_punct(TokenKind::OrOr, 2),
                    Some(b'=') => self.push_punct(TokenKind::PipeEq, 2),
                    _ => self.push_punct(TokenKind::Pipe, 1),
                },

                b'^' => match self.peek_at(1) {
                    Some(b'=') => self.push_punct(TokenKind::CaretEq, 2),
                    _ => self.push_punct(TokenKind::Caret, 1),
                },

                0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x7F => {
                    self.pos += 1;
                    self.diags.error(
                        ErrorCode::E0002,
                        self.span(start, self.pos),
                        format!("forbidden ASCII control byte `0x{:02X}` in source code", b),
                    );
                }

                _ => {
                    let ch = self.next_char_unchecked();
                    let span = self.span(start, self.pos);
                    if !is_bidi_control(ch) {
                        self.diags.error(
                            ErrorCode::E0004,
                            span,
                            format!("unexpected character `{}` (identifiers must be ASCII)", ch),
                        );
                        self.tokens.push(Token {
                            kind: TokenKind::Error,
                            span,
                        });
                    }
                }
            }
        }

        let eof_pos = self.bytes.len();
        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: self.span(eof_pos, eof_pos),
        });
    }

    #[inline]
    fn push_punct(&mut self, kind: TokenKind, len: usize) {
        let start = self.pos;
        self.pos += len;
        self.tokens.push(Token {
            kind,
            span: self.span(start, self.pos),
        });
    }

    #[inline]
    fn next_char_unchecked(&mut self) -> char {
        let ch = self.src[self.pos..].chars().next().unwrap();
        self.pos += ch.len_utf8();
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    self.pos += 1;
                }
                b'/' if self.peek_at(1) == Some(b'/') => {
                    let start = self.pos;
                    let is_doc_outer =
                        self.peek_at(2) == Some(b'/') && self.peek_at(3) != Some(b'/');
                    let is_doc_inner = self.peek_at(2) == Some(b'!');
                    self.pos += 2;
                    while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                        let b = self.bytes[self.pos];
                        if is_forbidden_ascii_ctrl(b) && b != b'\r' {
                            let cstart = self.pos;
                            self.pos += 1;
                            self.diags.error(
                                ErrorCode::E0002,
                                self.span(cstart, self.pos),
                                format!("forbidden ASCII control byte `0x{:02X}` in comment", b),
                            );
                        } else {
                            self.pos += 1;
                        }
                    }
                    if is_doc_outer || is_doc_inner {
                        self.doc_comments.push(DocComment {
                            is_module_doc: is_doc_inner,
                            span: self.span(start, self.pos),
                        });
                    }
                }
                b'/' if self.peek_at(1) == Some(b'*') => {
                    let start = self.pos;
                    self.pos += 2;
                    let mut depth = 1usize;
                    while self.pos < self.bytes.len() && depth > 0 {
                        if self.bytes[self.pos] == b'/' && self.peek_at(1) == Some(b'*') {
                            depth += 1;
                            self.pos += 2;
                        } else if self.bytes[self.pos] == b'*' && self.peek_at(1) == Some(b'/') {
                            depth -= 1;
                            self.pos += 2;
                        } else {
                            let b = self.bytes[self.pos];
                            if is_forbidden_ascii_ctrl(b) && b != b'\r' {
                                let cstart = self.pos;
                                self.pos += 1;
                                self.diags.error(
                                    ErrorCode::E0002,
                                    self.span(cstart, self.pos),
                                    format!(
                                        "forbidden ASCII control byte `0x{:02X}` in comment",
                                        b
                                    ),
                                );
                            } else {
                                self.pos += 1;
                            }
                        }
                    }
                    if depth > 0 {
                        self.diags.error(
                            ErrorCode::E0004,
                            self.span(start, self.pos),
                            "unterminated block comment `/*`",
                        );
                    }
                }
                _ => break,
            }
        }
    }

    fn lex_ident_or_keyword(&mut self, start: usize) {
        self.pos += 1;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        let kind = if text == "_" {
            TokenKind::Underscore
        } else if let Some(kw) = TokenKind::from_keyword(text) {
            kw
        } else {
            TokenKind::Ident
        };
        self.tokens.push(Token {
            kind,
            span: self.span(start, self.pos),
        });
    }

    fn lex_char_or_label(&mut self, start: usize) {
        self.pos += 1; // consume opening `'`
        if self.pos >= self.bytes.len() || self.bytes[self.pos] == b'\n' || self.bytes[self.pos] == b'\r' {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "unterminated character literal or loop label",
            );
            return;
        }
        if self.bytes[self.pos] == b'\'' {
            self.pos += 1;
            let span = self.span(start, self.pos);
            self.diags.error(
                ErrorCode::E0004,
                span,
                "empty character literal `''`",
            );
            self.tokens.push(Token {
                kind: TokenKind::CharLit('\0'),
                span,
            });
            return;
        }
        if self.bytes[self.pos] == b'\\' {
            let ch = self.lex_escape_char(false, start);
            if self.peek() == Some(b'\'') {
                self.pos += 1;
                self.tokens.push(Token {
                    kind: TokenKind::CharLit(ch.unwrap_or('\0')),
                    span: self.span(start, self.pos),
                });
            } else {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "unterminated character literal",
                );
            }
            return;
        }

        let first = self.next_char_unchecked();
        if self.peek() == Some(b'\'') {
            self.pos += 1;
            self.tokens.push(Token {
                kind: TokenKind::CharLit(first),
                span: self.span(start, self.pos),
            });
            return;
        }

        if first.is_ascii_alphabetic() || first == '_' {
            while let Some(b) = self.peek() {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b'\'') {
                self.pos += 1;
                let span = self.span(start, self.pos);
                self.diags.error(
                    ErrorCode::E0004,
                    span,
                    "character literal may contain only one Unicode scalar; use `\"...\"` for strings",
                );
                self.tokens.push(Token {
                    kind: TokenKind::CharLit('\0'),
                    span,
                });
                return;
            }
            self.tokens.push(Token {
                kind: TokenKind::Label,
                span: self.span(start, self.pos),
            });
            return;
        }

        self.diags.error(
            ErrorCode::E0004,
            self.span(start, self.pos),
            "unterminated character literal",
        );
    }

    fn lex_byte_char(&mut self, start: usize) {
        self.pos += 2; // consume `b'`
        if self.pos >= self.bytes.len() || matches!(self.bytes[self.pos], b'\n' | b'\r') {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "unterminated byte character literal",
            );
            return;
        }
        if self.bytes[self.pos] == b'\'' {
            self.pos += 1;
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "empty byte character literal `b''`",
            );
            return;
        }
        let val = if self.bytes[self.pos] == b'\\' {
            self.lex_escape_byte(start)
        } else {
            let b = self.bytes[self.pos];
            if b >= 0x80 {
                let _ = self.next_char_unchecked();
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "non-ASCII character in byte literal `b'...'`; use `\\xHH`",
                );
                None
            } else {
                if is_forbidden_ascii_ctrl(b) {
                    self.diags.error(
                        ErrorCode::E0002,
                        self.span(self.pos, self.pos + 1),
                        "forbidden ASCII control character in byte literal",
                    );
                }
                self.pos += 1;
                Some(b)
            }
        };

        if self.peek() == Some(b'\'') {
            self.pos += 1;
            self.tokens.push(Token {
                kind: TokenKind::ByteCharLit(val.unwrap_or(0)),
                span: self.span(start, self.pos),
            });
        } else {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "unterminated byte character literal",
            );
        }
    }

    fn lex_string(&mut self, start: usize) {
        self.pos += 1; // opening `"`
        let mut out = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "unterminated string literal",
                );
                break;
            }
            let b = self.bytes[self.pos];
            match b {
                b'"' => {
                    self.pos += 1;
                    break;
                }
                b'\n' | b'\r' => {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(start, self.pos),
                        "unterminated string literal before end of line; use `\\\\` for multiline strings",
                    );
                    break;
                }
                b'\\' => {
                    if let Some(ch) = self.lex_escape_char(false, start) {
                        out.push(ch);
                    }
                }
                b if is_forbidden_ascii_ctrl(b) => {
                    self.diags.error(
                        ErrorCode::E0002,
                        self.span(self.pos, self.pos + 1),
                        format!("forbidden ASCII control byte `0x{:02X}` in string literal", b),
                    );
                    self.pos += 1;
                }
                0x00..=0x7F => {
                    out.push(b as char);
                    self.pos += 1;
                }
                _ => {
                    let ch = self.next_char_unchecked();
                    out.push(ch);
                }
            }
        }
        let id = StrLitId(self.string_lits.len() as u32);
        self.string_lits.push(out);
        self.tokens.push(Token {
            kind: TokenKind::StringLit(id),
            span: self.span(start, self.pos),
        });
    }

    fn lex_byte_string(&mut self, start: usize) {
        self.pos += 2; // consume `b"`
        let mut out = Vec::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "unterminated byte string literal",
                );
                break;
            }
            let b = self.bytes[self.pos];
            match b {
                b'"' => {
                    self.pos += 1;
                    break;
                }
                b'\n' | b'\r' => {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(start, self.pos),
                        "unterminated byte string literal before end of line",
                    );
                    break;
                }
                b'\\' => {
                    if let Some(byte) = self.lex_escape_byte(start) {
                        out.push(byte);
                    }
                }
                b if is_forbidden_ascii_ctrl(b) => {
                    self.diags.error(
                        ErrorCode::E0002,
                        self.span(self.pos, self.pos + 1),
                        format!("forbidden ASCII control byte `0x{:02X}` in byte string", b),
                    );
                    self.pos += 1;
                }
                0x00..=0x7F => {
                    out.push(b);
                    self.pos += 1;
                }
                _ => {
                    let cstart = self.pos;
                    let _ = self.next_char_unchecked();
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(cstart, self.pos),
                        "non-ASCII character in byte string literal `b\"...\"`; use `\\xHH`",
                    );
                }
            }
        }
        let id = ByteStrLitId(self.byte_string_lits.len() as u32);
        self.byte_string_lits.push(out);
        self.tokens.push(Token {
            kind: TokenKind::ByteStringLit(id),
            span: self.span(start, self.pos),
        });
    }

    fn lex_raw_string(&mut self, start: usize) {
        let mut out = String::new();
        let mut first_line = true;
        loop {
            // Current position is at `\\`
            self.pos += 2;
            if !first_line {
                out.push('\n');
            }
            first_line = false;
            let line_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                let b = self.bytes[self.pos];
                if is_forbidden_ascii_ctrl(b) && b != b'\r' {
                    self.diags.error(
                        ErrorCode::E0002,
                        self.span(self.pos, self.pos + 1),
                        format!("forbidden ASCII control byte `0x{:02X}` in raw string", b),
                    );
                }
                self.pos += 1;
            }
            let mut line_end = self.pos;
            if line_end > line_start && self.bytes[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            out.push_str(&self.src[line_start..line_end]);

            // Look ahead past `\n` and horizontal whitespace for another `\\` line.
            if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
                let mut look = self.pos + 1;
                while look < self.bytes.len() && matches!(self.bytes[look], b' ' | b'\t') {
                    look += 1;
                }
                if look + 1 < self.bytes.len()
                    && self.bytes[look] == b'\\'
                    && self.bytes[look + 1] == b'\\'
                {
                    self.pos = look;
                    continue;
                }
            }
            break;
        }
        let end = self.pos;
        let id = StrLitId(self.string_lits.len() as u32);
        self.string_lits.push(out);
        self.tokens.push(Token {
            kind: TokenKind::RawStringLit(id),
            span: self.span(start, end),
        });
    }

    fn lex_escape_char(&mut self, _allow_high_hex: bool, lit_start: usize) -> Option<char> {
        let esc_start = self.pos;
        self.pos += 1; // consume `\`
        let Some(b) = self.peek() else {
            self.diags.error(
                ErrorCode::E0004,
                self.span(lit_start, self.pos),
                "unterminated escape sequence",
            );
            return None;
        };
        self.pos += 1;
        match b {
            b'n' => Some('\n'),
            b'r' => Some('\r'),
            b't' => Some('\t'),
            b'\\' => Some('\\'),
            b'"' => Some('"'),
            b'\'' => Some('\''),
            b'0' => Some('\0'),
            b'x' => {
                let val = self.read_hex_byte(esc_start)?;
                if val > 0x7F {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(esc_start, self.pos),
                        "`\\xHH` escape in UTF-8 literal must be ASCII `0x00..=0x7F`; use `\\u{...}`",
                    );
                    return None;
                }
                Some(val as char)
            }
            b'u' => {
                if self.peek() != Some(b'{') {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(esc_start, self.pos),
                        "expected `{` after `\\u`",
                    );
                    return None;
                }
                self.pos += 1;
                let hex_start = self.pos;
                let mut val: u32 = 0;
                let mut count = 0u32;
                while let Some(c) = self.peek() {
                    if c == b'}' {
                        break;
                    }
                    if let Some(d) = hex_digit(c) {
                        if count < 6 {
                            val = (val << 4) | d as u32;
                        }
                        count += 1;
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if self.peek() != Some(b'}') || count == 0 || count > 6 {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(esc_start, self.pos.max(hex_start)),
                        "invalid `\\u{...}` Unicode escape (expected 1 to 6 hex digits)",
                    );
                    return None;
                }
                self.pos += 1;
                match char::from_u32(val) {
                    Some(ch) => Some(ch),
                    None => {
                        self.diags.error(
                            ErrorCode::E0004,
                            self.span(esc_start, self.pos),
                            format!("`\\u{{{:X}}}` is not a valid Unicode scalar value", val),
                        );
                        None
                    }
                }
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(esc_start, self.pos),
                    format!("unknown escape sequence `\\{}`", b as char),
                );
                None
            }
        }
    }

    fn lex_escape_byte(&mut self, lit_start: usize) -> Option<u8> {
        let esc_start = self.pos;
        self.pos += 1; // consume `\`
        let Some(b) = self.peek() else {
            self.diags.error(
                ErrorCode::E0004,
                self.span(lit_start, self.pos),
                "unterminated escape sequence",
            );
            return None;
        };
        self.pos += 1;
        match b {
            b'n' => Some(b'\n'),
            b'r' => Some(b'\r'),
            b't' => Some(b'\t'),
            b'\\' => Some(b'\\'),
            b'"' => Some(b'"'),
            b'\'' => Some(b'\''),
            b'0' => Some(b'\0'),
            b'x' => self.read_hex_byte(esc_start),
            b'u' => {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(esc_start, self.pos),
                    "`\\u{...}` escapes are not allowed in byte literals; use `\\xHH`",
                );
                None
            }
            _ => {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(esc_start, self.pos),
                    format!("unknown escape sequence `\\{}`", b as char),
                );
                None
            }
        }
    }

    fn read_hex_byte(&mut self, esc_start: usize) -> Option<u8> {
        let (Some(h1), Some(h2)) = (
            self.peek().and_then(hex_digit),
            self.peek_at(1).and_then(hex_digit),
        ) else {
            self.diags.error(
                ErrorCode::E0004,
                self.span(esc_start, self.pos),
                "`\\x` escape requires exactly two hexadecimal digits",
            );
            return None;
        };
        self.pos += 2;
        Some((h1 << 4) | h2)
    }

    fn lex_number(&mut self, start: usize) {
        let mut is_float = false;
        let mut radix = 10u32;

        if self.bytes[self.pos] == b'0' {
            match self.peek_at(1) {
                Some(b'x') => {
                    radix = 16;
                    self.pos += 2;
                }
                Some(b'b') => {
                    radix = 2;
                    self.pos += 2;
                }
                Some(b'o') => {
                    radix = 8;
                    self.pos += 2;
                }
                _ => {}
            }
        }

        let digits_start = self.pos;
        let digit_count = self.consume_digits(radix);
        if digit_count == 0 {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "numeric literal has no digits after base prefix",
            );
        } else if self.bytes[digits_start] == b'_' {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "leading `_` immediately after base prefix is not allowed",
            );
        }

        // Check fractional part `.` (never consume `..`)
        if (radix == 10 || radix == 16) && self.peek() == Some(b'.') && self.peek_at(1) != Some(b'.')
        {
            match self.peek_at(1) {
                Some(b) if is_digit_in_radix(b, radix) => {
                    is_float = true;
                    self.pos += 1; // consume `.`
                    let frac_count = self.consume_digits(radix);
                    if frac_count == 0 {
                        self.diags.error(
                            ErrorCode::E0004,
                            self.span(start, self.pos),
                            "missing digits after decimal point",
                        );
                    }
                }
                _ => {
                    // `5.` without fractional digits is forbidden by spec 01 §2.5.
                    self.pos += 1;
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(start, self.pos),
                        "trailing `.` in numeric literal is forbidden; write e.g. `5.0`",
                    );
                }
            }
        }

        // Check exponent (`e`/`E` for decimal, `p`/`P` for hex float)
        let exp_match = match radix {
            10 => matches!(self.peek(), Some(b'e' | b'E')),
            16 => matches!(self.peek(), Some(b'p' | b'P')),
            _ => false,
        };
        if exp_match {
            if radix == 16 && !is_float {
                // `0x1p+3` is also valid hex float
                is_float = true;
            } else {
                is_float = true;
            }
            self.pos += 1; // consume e/E/p/P
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_digits = self.consume_digits(10);
            if exp_digits == 0 {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "missing exponent digits in floating-point literal",
                );
            }
        } else if radix == 16 && is_float {
            self.diags.error(
                ErrorCode::E0004,
                self.span(start, self.pos),
                "hexadecimal floating-point literal requires a binary exponent (`p+...`)",
            );
        }

        let raw_end = self.pos;

        // Check optional type suffix: `_u32` or `u32` (for hex, only `_` before `f..` to avoid ambiguity)
        let mut has_suffix = false;
        let mut suffix_start = self.pos;
        let mut suffix_end = self.pos;

        let starts_suffix = match self.peek() {
            Some(b'_') => true,
            Some(b) if b.is_ascii_alphabetic() => true,
            _ => false,
        };

        if starts_suffix {
            if self.peek() == Some(b'_') {
                self.pos += 1;
            }
            suffix_start = self.pos;
            while let Some(b) = self.peek() {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            suffix_end = self.pos;
            let suffix = &self.src[suffix_start..suffix_end];
            if suffix.is_empty() {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(start, self.pos),
                    "trailing `_` at end of numeric literal",
                );
            } else if is_float {
                if !is_valid_float_suffix(suffix) {
                    self.diags.error(
                        ErrorCode::E0004,
                        self.span(suffix_start, suffix_end),
                        format!("invalid floating-point type suffix `{}`", suffix),
                    );
                } else {
                    has_suffix = true;
                }
            } else if !is_valid_int_suffix(suffix) {
                self.diags.error(
                    ErrorCode::E0004,
                    self.span(suffix_start, suffix_end),
                    format!("invalid integer type suffix `{}`", suffix),
                );
            } else {
                has_suffix = true;
            }
        }

        let id = NumLitId(self.num_lits.len() as u32);
        self.num_lits.push(NumLitData {
            raw_start: start as u32,
            raw_end: raw_end as u32,
            suffix_start: suffix_start as u32,
            suffix_end: suffix_end as u32,
            has_suffix,
        });
        let kind = if is_float {
            TokenKind::FloatLit(id)
        } else {
            TokenKind::IntLit(id)
        };
        self.tokens.push(Token {
            kind,
            span: self.span(start, self.pos),
        });
    }

    fn consume_digits(&mut self, radix: u32) -> usize {
        let mut count = 0;
        while let Some(b) = self.peek() {
            if is_digit_in_radix(b, radix) {
                count += 1;
                self.pos += 1;
            } else if b == b'_' {
                // Stop before `_suffix` such as `42_u32` or `1.0_f32`.
                match self.peek_at(1) {
                    Some(next) if is_digit_in_radix(next, radix) || next == b'_' => {
                        self.pos += 1;
                    }
                    _ => break,
                }
            } else {
                break;
            }
        }
        count
    }
}

#[inline]
fn is_forbidden_ascii_ctrl(b: u8) -> bool {
    matches!(b, 0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x7F)
}

#[inline]
fn is_bidi_control(ch: char) -> bool {
    matches!(ch, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}')
}

#[inline]
fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[inline]
fn is_digit_in_radix(b: u8, radix: u32) -> bool {
    match radix {
        2 => matches!(b, b'0' | b'1'),
        8 => matches!(b, b'0'..=b'7'),
        10 => b.is_ascii_digit(),
        16 => b.is_ascii_hexdigit(),
        _ => false,
    }
}

fn is_valid_float_suffix(s: &str) -> bool {
    matches!(s, "f16" | "f32" | "f64")
}

fn is_valid_int_suffix(s: &str) -> bool {
    if matches!(
        s,
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u128" | "usize"
    ) {
        return true;
    }
    if let Some(rest) = s.strip_prefix('u').or_else(|| s.strip_prefix('i'))
        && let Ok(bits) = rest.parse::<u32>()
    {
        return (1..=64).contains(&bits);
    }
    false
}
