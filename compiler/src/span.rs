use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

impl FileId {
    pub const DUMMY: FileId = FileId(u32::MAX);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const DUMMY: Span = Span {
        file: FileId::DUMMY,
        start: 0,
        end: 0,
    };

    #[inline]
    pub const fn new(file: FileId, start: u32, end: u32) -> Self {
        Self { file, start, end }
    }

    #[inline]
    pub fn merge(self, other: Span) -> Span {
        if self.file == FileId::DUMMY {
            return other;
        }
        if other.file == FileId::DUMMY {
            return self;
        }
        debug_assert_eq!(self.file, other.file);
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file({}):{}..{}", self.file.0, self.start, self.end)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FileProfile {
    Safe,
    Hal,
}

impl FileProfile {
    pub fn from_path(path: &str) -> Self {
        if path.ends_with(".hal.eon") {
            Self::Hal
        } else {
            Self::Safe
        }
    }
}

#[derive(Clone, Debug)]
pub struct Utf8Issue {
    pub span: Span,
    pub message: &'static str,
}

pub struct SourceFile {
    pub id: FileId,
    pub name: String,
    pub profile: FileProfile,
    text: String,
    line_starts: Vec<u32>,
    pub utf8_issues: Vec<Utf8Issue>,
}

impl SourceFile {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }

    /// Returns 1-based `(line, column)` where column counts Unicode scalar values.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let off = offset.min(self.text.len() as u32);
        let idx = match self.line_starts.binary_search(&off) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let line_start = self.line_starts[idx] as usize;
        let clamped_end = clamp_char_boundary(&self.text, off as usize);
        let prefix = &self.text[line_start..clamped_end.max(line_start)];
        let col = prefix.chars().count() as u32 + 1;
        (idx as u32 + 1, col)
    }

    pub fn line_text(&self, line_1based: u32) -> &str {
        if line_1based == 0 {
            return "";
        }
        let idx = (line_1based - 1) as usize;
        if idx >= self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[idx] as usize;
        let end = if idx + 1 < self.line_starts.len() {
            (self.line_starts[idx + 1] as usize).saturating_sub(1)
        } else {
            self.text.len()
        };
        let raw = &self.text[start..end.max(start)];
        raw.strip_suffix('\r').unwrap_or(raw)
    }
}

fn clamp_char_boundary(s: &str, mut pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

#[derive(Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub fn add_file_str(&mut self, name: impl Into<String>, src: &str) -> FileId {
        self.add_file_bytes(name, src.as_bytes().to_vec())
    }

    pub fn add_file_bytes(&mut self, name: impl Into<String>, raw: Vec<u8>) -> FileId {
        let name = name.into();
        let profile = FileProfile::from_path(&name);
        let id = FileId(self.files.len() as u32);
        let mut utf8_issues = Vec::new();
        let mut clean = raw;

        if clean.starts_with(&[0xEF, 0xBB, 0xBF]) {
            utf8_issues.push(Utf8Issue {
                span: Span::new(id, 0, 3),
                message: "UTF-8 BOM (EF BB BF) is forbidden in Eon source files",
            });
            clean[0] = b' ';
            clean[1] = b' ';
            clean[2] = b' ';
        }

        let text = match String::from_utf8(clean) {
            Ok(s) => s,
            Err(err) => {
                let mut bytes = err.into_bytes();
                let mut pos = 0usize;
                let mut first = true;
                while pos < bytes.len() {
                    match std::str::from_utf8(&bytes[pos..]) {
                        Ok(_) => break,
                        Err(e) => {
                            let bad_start = pos + e.valid_up_to();
                            let bad_len = e.error_len().unwrap_or(bytes.len() - bad_start).max(1);
                            let bad_end = (bad_start + bad_len).min(bytes.len());
                            if first {
                                utf8_issues.push(Utf8Issue {
                                    span: Span::new(id, bad_start as u32, bad_end as u32),
                                    message: "source file is not valid UTF-8",
                                });
                                first = false;
                            }
                            for b in &mut bytes[bad_start..bad_end] {
                                *b = b' ';
                            }
                            pos = bad_end;
                        }
                    }
                }
                String::from_utf8(bytes).unwrap_or_default()
            }
        };

        let mut line_starts = vec![0u32];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }

        self.files.push(SourceFile {
            id,
            name,
            profile,
            text,
            line_starts,
            utf8_issues,
        });
        id
    }

    pub fn file(&self, id: FileId) -> &SourceFile {
        &self.files[id.0 as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub u32);

impl Symbol {
    pub const EMPTY: Symbol = Symbol(0);
}

/// Deterministic string interner using FNV-1a open addressing (no RandomState).
pub struct Interner {
    strings: Vec<String>,
    buckets: Vec<u32>,
    hashes: Vec<u64>,
}

const EMPTY_SLOT: u32 = u32::MAX;

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        let mut this = Self {
            strings: Vec::with_capacity(256),
            buckets: vec![EMPTY_SLOT; 512],
            hashes: Vec::with_capacity(256),
        };
        let empty = this.intern("");
        debug_assert_eq!(empty, Symbol::EMPTY);
        this
    }

    #[inline]
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        let hash = Self::fnv1a(s.as_bytes());
        if (self.strings.len() + 1) * 10 >= self.buckets.len() * 7 {
            self.rehash();
        }
        let mask = self.buckets.len() - 1;
        let mut idx = (hash as usize) & mask;
        loop {
            let slot = self.buckets[idx];
            if slot == EMPTY_SLOT {
                let sym = Symbol(self.strings.len() as u32);
                self.strings.push(s.to_owned());
                self.hashes.push(hash);
                self.buckets[idx] = sym.0;
                return sym;
            }
            if self.hashes[slot as usize] == hash && self.strings[slot as usize] == s {
                return Symbol(slot);
            }
            idx = (idx + 1) & mask;
        }
    }

    #[inline]
    pub fn get(&self, sym: Symbol) -> &str {
        &self.strings[sym.0 as usize]
    }

    fn rehash(&mut self) {
        let new_cap = (self.buckets.len() * 2).max(16);
        let mask = new_cap - 1;
        let mut new_buckets = vec![EMPTY_SLOT; new_cap];
        for (i, &h) in self.hashes.iter().enumerate() {
            let mut idx = (h as usize) & mask;
            while new_buckets[idx] != EMPTY_SLOT {
                idx = (idx + 1) & mask;
            }
            new_buckets[idx] = i as u32;
        }
        self.buckets = new_buckets;
    }
}
