//! Built-in ELF64 linker & DWARF5 `.debug_line` generator (zero external linker / libc).
//! Supports Linux x86_64 position-independent executables (`ET_DYN` PIE), shared libraries
//! (`ET_DYN` `.so`), and freestanding bare-metal kernel ELF64 images (with optional Multiboot2).

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ElfMode {
    #[default]
    LinuxPie,
    LinuxShared { soname: Option<String> },
    Freestanding { load_addr: u64, multiboot2: bool },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RelocSection {
    Text,
    Rodata,
    DataRelRo,
    Data,
    Bss,
    Tdata,
    Tbss,
}

#[derive(Clone, Debug)]
pub enum RelocKind {
    /// 32-bit PC-relative fixup at `offset`: `S + addend - (P + 4)` (standard x86_64 `rel32`).
    Rel32,
    /// 64-bit pointer fixup at `offset`: `S + addend` (patched directly in freestanding, or
    /// emitted as `R_X86_64_RELATIVE` / `R_X86_64_64` in `.rela.dyn` for PIE/shared `DataRelRo`/`Data`).
    Abs64,
    /// 32-bit PC-relative offset to a `.got` entry containing the target symbol's address.
    GotPcRel32,
    /// 32-bit PC-relative fixup targeting the `.plt` stub of an external dynamic symbol
    /// (or directly targeting a locally defined symbol).
    Plt32,
    /// 32-bit signed offset from `fs:0` TLS block (x86_64 Variant II:
    /// `tls_offset - align_up(tls_memsz, tls_align)`).
    TpOff32,
}

#[derive(Clone, Debug)]
pub struct ElfReloc {
    pub section: RelocSection,
    pub offset: usize,
    pub kind: RelocKind,
    pub target_sym: String,
    pub addend: i64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineEntry {
    pub code_offset: u32,
    pub file_idx: u32,
    pub line: u32,
    pub col: u32,
}

#[derive(Clone, Debug)]
pub struct ElfSymbolDef {
    pub name: String,
    pub section: RelocSection,
    pub offset: usize,
    pub size: u64,
    pub is_func: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ElfLinkInput {
    pub mode: ElfMode,
    pub entry_symbol: String,
    pub text: Vec<u8>,
    pub rodata: Vec<u8>,
    pub data_rel_ro: Vec<u8>,
    pub data: Vec<u8>,
    pub bss_size: usize,
    pub tdata: Vec<u8>,
    pub tbss_size: usize,
    pub symbols: Vec<ElfSymbolDef>,
    pub relocs: Vec<ElfReloc>,
    pub source_files: Vec<String>,
    pub line_entries: Vec<LineEntry>,
    pub emit_debug_line: bool,
    pub needed_libs: Vec<String>,
    pub exported_symbols: BTreeSet<String>,
    pub external_symbols: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct LinkedElf {
    pub bytes: Vec<u8>,
    pub entry_vaddr: u64,
    pub text_vaddr: u64,
    pub rodata_vaddr: u64,
    pub data_rel_ro_vaddr: u64,
    pub data_vaddr: u64,
    pub bss_vaddr: u64,
    pub tdata_vaddr: u64,
    pub tls_memsz: u64,
    pub symbol_vaddrs: BTreeMap<String, u64>,
}

const PAGE_SIZE: usize = 0x1000;
const TLS_ALIGN: usize = 64;

const EHDR_SIZE: usize = 64;
const PHDR_SIZE: usize = 56;
const SHDR_SIZE: usize = 64;
const SYM_SIZE: usize = 24;
const RELA_SIZE: usize = 24;
const DYN_SIZE: usize = 16;

const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const EM_X86_64: u16 = 62;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;
const PT_PHDR: u32 = 6;
const PT_TLS: u32 = 7;
const PT_GNU_STACK: u32 = 0x6474_E551;
const PT_GNU_RELRO: u32 = 0x6474_E552;

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_DYNAMIC: u32 = 6;
const SHT_NOBITS: u32 = 8;
const SHT_DYNSYM: u32 = 11;
const SHT_GNU_HASH: u32 = 0x6FFF_FFF6;

const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;
const SHF_TLS: u64 = 0x400;

const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_TLS: u8 = 6;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;

const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_PLTRELSZ: i64 = 2;
const DT_PLTGOT: i64 = 3;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_SONAME: i64 = 14;
const DT_PLTREL: i64 = 20;
const DT_JMPREL: i64 = 23;
const DT_FLAGS: i64 = 30;
const DT_GNU_HASH: i64 = 0x6FFF_FEF5;
const DT_RELACOUNT: i64 = 0x6FFF_FFF9;
const DT_FLAGS_1: i64 = 0x6FFF_FFFB;

const DT_RELA_VAL: u64 = 7;
const DF_BIND_NOW: u64 = 0x8;
const DF_1_NOW: u64 = 0x1;
const DF_1_PIE: u64 = 0x0800_0000;

const INTERP_PATH: &[u8] = b"/lib64/ld-linux-x86-64.so.2\0";

pub fn link_elf64(input: &ElfLinkInput) -> Result<LinkedElf, String> {
    match &input.mode {
        ElfMode::Freestanding { load_addr, multiboot2 } => {
            link_freestanding(input, *load_addr, *multiboot2)
        }
        ElfMode::LinuxPie | ElfMode::LinuxShared { .. } => link_dynamic_elf(input),
    }
}

#[derive(Default)]
struct SectionIndices {
    text: Option<u16>,
    rodata: Option<u16>,
    data_rel_ro: Option<u16>,
    data: Option<u16>,
    bss: Option<u16>,
    tdata: Option<u16>,
    tbss: Option<u16>,
}

impl SectionIndices {
    fn get(&self, sec: RelocSection) -> Option<u16> {
        match sec {
            RelocSection::Text => self.text,
            RelocSection::Rodata => self.rodata,
            RelocSection::DataRelRo => self.data_rel_ro,
            RelocSection::Data => self.data,
            RelocSection::Bss => self.bss,
            RelocSection::Tdata => self.tdata,
            RelocSection::Tbss => self.tbss,
        }
    }
}

#[derive(Copy, Clone)]
struct SectionAddrs {
    text: u64,
    rodata: u64,
    data_rel_ro: u64,
    data: u64,
    bss: u64,
    tdata: u64,
    tbss_tls_offset: usize,
}

impl SectionAddrs {
    fn sym_vaddr(&self, sec: RelocSection, offset: usize) -> u64 {
        match sec {
            RelocSection::Text => self.text + offset as u64,
            RelocSection::Rodata => self.rodata + offset as u64,
            RelocSection::DataRelRo => self.data_rel_ro + offset as u64,
            RelocSection::Data => self.data + offset as u64,
            RelocSection::Bss => self.bss + offset as u64,
            RelocSection::Tdata => self.tdata + offset as u64,
            RelocSection::Tbss => self.tdata + (self.tbss_tls_offset + offset) as u64,
        }
    }

    fn tls_offset(&self, sec: RelocSection, offset: usize) -> Option<u64> {
        match sec {
            RelocSection::Tdata => Some(offset as u64),
            RelocSection::Tbss => Some((self.tbss_tls_offset + offset) as u64),
            _ => None,
        }
    }
}

fn link_freestanding(
    input: &ElfLinkInput,
    base_vaddr: u64,
    use_multiboot2: bool,
) -> Result<LinkedElf, String> {
    let has_tls = !input.tdata.is_empty() || input.tbss_size > 0;
    let has_data_rel_ro =
        !input.data_rel_ro.is_empty() || has_sym_or_reloc_in(input, RelocSection::DataRelRo);
    let has_data = !input.data.is_empty() || has_sym_or_reloc_in(input, RelocSection::Data);
    let has_bss = input.bss_size > 0 || has_sym_or_reloc_in(input, RelocSection::Bss);
    let has_tdata = !input.tdata.is_empty() || has_sym_or_reloc_in(input, RelocSection::Tdata);
    let has_tbss = input.tbss_size > 0 || has_sym_or_reloc_in(input, RelocSection::Tbss);
    let got_syms = collect_got_symbols(input);
    let has_got = !got_syms.is_empty();

    let phnum: u16 = if has_tls { 2 } else { 1 };
    let phdr_off: usize = EHDR_SIZE;
    let mut cur_off = align_up(phdr_off + (phnum as usize) * PHDR_SIZE, 16);

    let mb2_bytes: Vec<u8> = if use_multiboot2 {
        build_multiboot2_header()
    } else {
        Vec::new()
    };
    let mb2_off = cur_off;
    if !mb2_bytes.is_empty() {
        cur_off = align_up(cur_off + mb2_bytes.len(), 16);
    }

    let text_off = cur_off;
    let text_vaddr = base_vaddr + text_off as u64;
    let mut text = input.text.clone();
    cur_off = align_up(text_off + text.len(), 8);

    let rodata_off = cur_off;
    let rodata_vaddr = base_vaddr + rodata_off as u64;
    let mut rodata = input.rodata.clone();
    cur_off = rodata_off + rodata.len();

    let mut data_rel_ro_off = cur_off;
    let mut data_rel_ro = input.data_rel_ro.clone();
    if has_data_rel_ro {
        cur_off = align_up(cur_off, 8);
        data_rel_ro_off = cur_off;
        cur_off += data_rel_ro.len();
    }
    let data_rel_ro_vaddr = base_vaddr + data_rel_ro_off as u64;

    let mut data_off = cur_off;
    let mut data = input.data.clone();
    if has_data {
        cur_off = align_up(cur_off, 8);
        data_off = cur_off;
        cur_off += data.len();
    }
    let data_vaddr = base_vaddr + data_off as u64;

    let mut tdata_off = cur_off;
    let mut tdata = input.tdata.clone();
    if has_tls {
        cur_off = align_up(cur_off, TLS_ALIGN);
        tdata_off = cur_off;
        cur_off += tdata.len();
    }
    let tdata_vaddr = base_vaddr + tdata_off as u64;
    let tbss_tls_offset = tdata.len();
    let tls_memsz = (tdata.len() + input.tbss_size) as u64;

    let mut got_off = cur_off;
    if has_got {
        cur_off = align_up(cur_off, 8);
        got_off = cur_off;
        cur_off += got_syms.len() * 8;
    }
    let got_vaddr = base_vaddr + got_off as u64;
    let mut got = vec![0u8; got_syms.len() * 8];

    let load_filesz = cur_off;
    let (bss_vaddr, load_memsz) = if has_bss {
        let bss_aligned_off = align_up(cur_off, 16);
        (
            base_vaddr + bss_aligned_off as u64,
            bss_aligned_off + input.bss_size,
        )
    } else {
        (base_vaddr + cur_off as u64, load_filesz)
    };

    let addrs = SectionAddrs {
        text: text_vaddr,
        rodata: rodata_vaddr,
        data_rel_ro: data_rel_ro_vaddr,
        data: data_vaddr,
        bss: bss_vaddr,
        tdata: tdata_vaddr,
        tbss_tls_offset,
    };

    let mut symbol_vaddrs: BTreeMap<String, u64> = BTreeMap::new();
    let mut tls_offsets: BTreeMap<String, u64> = BTreeMap::new();
    for s in &input.symbols {
        symbol_vaddrs.insert(s.name.clone(), addrs.sym_vaddr(s.section, s.offset));
        if let Some(toff) = addrs.tls_offset(s.section, s.offset) {
            tls_offsets.insert(s.name.clone(), toff);
        }
    }
    insert_synthetic_symbols(
        &mut symbol_vaddrs,
        base_vaddr,
        base_vaddr,
        base_vaddr,
        tdata_vaddr,
        tdata.len() as u64,
    );

    let entry_vaddr = *symbol_vaddrs.get(&input.entry_symbol).ok_or_else(|| {
        format!(
            "ELF64 linker error: unresolved entry symbol `{}`",
            input.entry_symbol
        )
    })?;

    let mut got_index: BTreeMap<&str, usize> = BTreeMap::new();
    for (idx, name) in got_syms.iter().enumerate() {
        let target_vaddr = *symbol_vaddrs.get(name).ok_or_else(|| {
            format!("ELF64 linker error: unresolved GOT target symbol `{name}`")
        })?;
        got[idx * 8..(idx + 1) * 8].copy_from_slice(&target_vaddr.to_le_bytes());
        got_index.insert(name.as_str(), idx);
    }

    let mut bufs = RelocBuffers {
        text: &mut text,
        rodata: &mut rodata,
        data_rel_ro: &mut data_rel_ro,
        data: &mut data,
        tdata: &mut tdata,
        addrs: &addrs,
        allow_tdata_reloc: true,
    };
    let tls_aligned_memsz = align_up(tls_memsz as usize, TLS_ALIGN) as i128;
    for rel in &input.relocs {
        let (buf, sec_vaddr) = bufs.slice_and_vaddr(rel.section)?;
        let place_vaddr = sec_vaddr + rel.offset as u64;
        match rel.kind {
            RelocKind::Rel32 | RelocKind::Plt32 => {
                let target_vaddr = *symbol_vaddrs.get(&rel.target_sym).ok_or_else(|| {
                    format!(
                        "ELF64 linker error: unresolved symbol `{}` referenced from {:?}+0x{:x}",
                        rel.target_sym, rel.section, rel.offset
                    )
                })?;
                patch_rel32(
                    buf,
                    rel.offset,
                    place_vaddr,
                    target_vaddr,
                    rel.addend,
                    &rel.target_sym,
                )?;
            }
            RelocKind::Abs64 => {
                let target_vaddr = *symbol_vaddrs.get(&rel.target_sym).ok_or_else(|| {
                    format!(
                        "ELF64 linker error: unresolved symbol `{}` referenced from {:?}+0x{:x}",
                        rel.target_sym, rel.section, rel.offset
                    )
                })?;
                if rel.offset + 8 > buf.len() {
                    return Err(format!(
                        "ELF64 linker error: Abs64 offset 0x{:x} out of range",
                        rel.offset
                    ));
                }
                let val = ((target_vaddr as i128) + (rel.addend as i128)) as u64;
                buf[rel.offset..rel.offset + 8].copy_from_slice(&val.to_le_bytes());
            }
            RelocKind::GotPcRel32 => {
                let idx = *got_index.get(rel.target_sym.as_str()).ok_or_else(|| {
                    format!(
                        "ELF64 linker error: missing GOT slot for `{}`",
                        rel.target_sym
                    )
                })?;
                let slot_vaddr = got_vaddr + (idx * 8) as u64;
                patch_rel32(
                    buf,
                    rel.offset,
                    place_vaddr,
                    slot_vaddr,
                    rel.addend,
                    &rel.target_sym,
                )?;
            }
            RelocKind::TpOff32 => {
                patch_tpoff32(buf, rel.offset, &tls_offsets, &rel.target_sym, rel.addend, tls_aligned_memsz)?;
            }
        }
    }

    // Assemble loaded image.
    let mut out = vec![0u8; load_filesz];
    if !mb2_bytes.is_empty() {
        out[mb2_off..mb2_off + mb2_bytes.len()].copy_from_slice(&mb2_bytes);
    }
    out[text_off..text_off + text.len()].copy_from_slice(&text);
    out[rodata_off..rodata_off + rodata.len()].copy_from_slice(&rodata);
    if has_data_rel_ro {
        out[data_rel_ro_off..data_rel_ro_off + data_rel_ro.len()].copy_from_slice(&data_rel_ro);
    }
    if has_data {
        out[data_off..data_off + data.len()].copy_from_slice(&data);
    }
    if !tdata.is_empty() {
        out[tdata_off..tdata_off + tdata.len()].copy_from_slice(&tdata);
    }
    if has_got {
        out[got_off..got_off + got.len()].copy_from_slice(&got);
    }

    let mut shstrtab = vec![0u8];
    let mut shdrs = vec![Shdr::null()];
    let mut sec_indices = SectionIndices {
        text: Some(shdrs.len() as u16),
        ..Default::default()
    };
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".text") as u32,
        kind: SHT_PROGBITS,
        flags: SHF_ALLOC | SHF_EXECINSTR,
        addr: text_vaddr,
        offset: text_off as u64,
        size: text.len() as u64,
        link: 0,
        info: 0,
        addralign: 16,
        entsize: 0,
    });
    sec_indices.rodata = Some(shdrs.len() as u16);
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".rodata") as u32,
        kind: SHT_PROGBITS,
        flags: SHF_ALLOC,
        addr: rodata_vaddr,
        offset: rodata_off as u64,
        size: rodata.len() as u64,
        link: 0,
        info: 0,
        addralign: 8,
        entsize: 0,
    });
    if has_data_rel_ro {
        sec_indices.data_rel_ro = Some(shdrs.len() as u16);
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".data.rel.ro") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: data_rel_ro_vaddr,
            offset: data_rel_ro_off as u64,
            size: data_rel_ro.len() as u64,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 0,
        });
    }
    if has_data {
        sec_indices.data = Some(shdrs.len() as u16);
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".data") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: data_vaddr,
            offset: data_off as u64,
            size: data.len() as u64,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 0,
        });
    }
    if has_tdata {
        sec_indices.tdata = Some(shdrs.len() as u16);
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".tdata") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE | SHF_TLS,
            addr: tdata_vaddr,
            offset: tdata_off as u64,
            size: tdata.len() as u64,
            link: 0,
            info: 0,
            addralign: TLS_ALIGN as u64,
            entsize: 0,
        });
    }
    if has_tbss {
        sec_indices.tbss = Some(shdrs.len() as u16);
        let tbss_align = tls_tbss_align(tdata.len());
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".tbss") as u32,
            kind: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE | SHF_TLS,
            addr: tdata_vaddr + tdata.len() as u64,
            offset: (tdata_off + tdata.len()) as u64,
            size: input.tbss_size as u64,
            link: 0,
            info: 0,
            addralign: tbss_align,
            entsize: 0,
        });
    }
    if has_got {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".got") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: got_vaddr,
            offset: got_off as u64,
            size: got.len() as u64,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 8,
        });
    }
    if has_bss {
        sec_indices.bss = Some(shdrs.len() as u16);
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".bss") as u32,
            kind: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: bss_vaddr,
            offset: load_filesz as u64,
            size: input.bss_size as u64,
            link: 0,
            info: 0,
            addralign: 16,
            entsize: 0,
        });
    }

    let (shdr_off, shstrtab_idx) = append_debug_and_symtab_sections(
        &mut out,
        input,
        text_vaddr,
        text.len() as u64,
        &addrs,
        &sec_indices,
        &mut shstrtab,
        &mut shdrs,
    );

    let ehdr = build_ehdr(
        ET_EXEC,
        entry_vaddr,
        phdr_off as u64,
        shdr_off as u64,
        phnum,
        shdrs.len() as u16,
        shstrtab_idx,
    );
    out[0..EHDR_SIZE].copy_from_slice(&ehdr);

    let mut phdrs = Vec::with_capacity((phnum as usize) * PHDR_SIZE);
    write_phdr(
        &mut phdrs,
        PT_LOAD,
        PF_R | PF_W | PF_X,
        0,
        base_vaddr,
        base_vaddr,
        load_filesz as u64,
        load_memsz as u64,
        PAGE_SIZE as u64,
    );
    if has_tls {
        write_phdr(
            &mut phdrs,
            PT_TLS,
            PF_R,
            tdata_off as u64,
            tdata_vaddr,
            tdata_vaddr,
            tdata.len() as u64,
            tls_memsz,
            TLS_ALIGN as u64,
        );
    }
    out[phdr_off..phdr_off + phdrs.len()].copy_from_slice(&phdrs);

    Ok(LinkedElf {
        bytes: out,
        entry_vaddr,
        text_vaddr,
        rodata_vaddr,
        data_rel_ro_vaddr,
        data_vaddr,
        bss_vaddr,
        tdata_vaddr,
        tls_memsz,
        symbol_vaddrs,
    })
}

fn link_dynamic_elf(input: &ElfLinkInput) -> Result<LinkedElf, String> {
    let is_pie = matches!(input.mode, ElfMode::LinuxPie);
    let soname = match &input.mode {
        ElfMode::LinuxShared { soname } => soname.as_deref(),
        _ => None,
    };
    let has_interp = is_pie && !input.needed_libs.is_empty();
    let has_tls = !input.tdata.is_empty() || input.tbss_size > 0;

    // Map local symbol definitions (later definitions override earlier ones just like symbol_vaddrs).
    let mut local_defs: BTreeMap<String, &ElfSymbolDef> = BTreeMap::new();
    for s in &input.symbols {
        local_defs.insert(s.name.clone(), s);
    }

    // Deduplicate external dynamic symbols (skipping anything locally defined).
    let mut ext_seen: BTreeSet<String> = BTreeSet::new();
    let mut ext_syms: Vec<String> = Vec::new();
    for ext in &input.external_symbols {
        if !local_defs.contains_key(ext) && ext_seen.insert(ext.clone()) {
            ext_syms.push(ext.clone());
        }
    }

    // Determine whether each external symbol is called via PLT (to mark STT_FUNC in .dynsym).
    let mut ext_is_func: BTreeSet<&str> = BTreeSet::new();
    for rel in &input.relocs {
        if matches!(rel.kind, RelocKind::Plt32) && ext_seen.contains(&rel.target_sym) {
            ext_is_func.insert(rel.target_sym.as_str());
        }
    }

    let has_plt = !ext_syms.is_empty();

    // Collect deduplicated GOT slots.
    let got_syms = collect_got_symbols(input);
    let has_got = !got_syms.is_empty();

    // Validate exported symbols and prepare `.gnu.hash` bucket sorting.
    let mut exported_names: Vec<String> = Vec::with_capacity(input.exported_symbols.len());
    for name in &input.exported_symbols {
        if !local_defs.contains_key(name) {
            return Err(format!(
                "ELF64 linker error: exported symbol `{name}` is not defined"
            ));
        }
        exported_names.push(name.clone());
    }

    let nbuckets: u32 = exported_names.len().max(1) as u32;
    let symoffset: u32 = 1 + ext_syms.len() as u32;
    exported_names.sort_by_key(|name| (gnu_hash(name) % nbuckets, name.clone()));

    // Check which optional loadable data sections are emitted.
    let has_tdata = !input.tdata.is_empty() || has_sym_or_reloc_in(input, RelocSection::Tdata);
    let has_tbss = input.tbss_size > 0 || has_sym_or_reloc_in(input, RelocSection::Tbss);
    let has_data_rel_ro =
        !input.data_rel_ro.is_empty() || has_sym_or_reloc_in(input, RelocSection::DataRelRo);
    let has_data = !input.data.is_empty() || has_sym_or_reloc_in(input, RelocSection::Data);
    let has_bss = input.bss_size > 0 || has_sym_or_reloc_in(input, RelocSection::Bss);

    // Precompute exact section header indices (1-based) so `.dynsym` / `.gnu.hash` / `.dynamic`
    // can reference them deterministically before layout.
    let mut next_shndx: u16 = 1;
    let mut alloc_shndx = || {
        let idx = next_shndx;
        next_shndx += 1;
        idx
    };

    let _interp_shndx = if has_interp { Some(alloc_shndx()) } else { None };
    let _gnu_hash_shndx = alloc_shndx();
    let dynsym_shndx = alloc_shndx();
    let dynstr_shndx = alloc_shndx();
    let _rela_dyn_shndx = alloc_shndx();
    let _rela_plt_shndx = alloc_shndx();
    let rodata_shndx = alloc_shndx();
    let text_shndx = alloc_shndx();
    let _plt_shndx = if has_plt { Some(alloc_shndx()) } else { None };
    let tdata_shndx = if has_tdata { Some(alloc_shndx()) } else { None };
    let tbss_shndx = if has_tbss { Some(alloc_shndx()) } else { None };
    let data_rel_ro_shndx = if has_data_rel_ro { Some(alloc_shndx()) } else { None };
    let _dynamic_shndx = alloc_shndx();
    let _got_shndx = if has_got { Some(alloc_shndx()) } else { None };
    let got_plt_shndx = if has_plt { Some(alloc_shndx()) } else { None };
    let data_shndx = if has_data { Some(alloc_shndx()) } else { None };
    let bss_shndx = if has_bss { Some(alloc_shndx()) } else { None };

    let sec_indices = SectionIndices {
        text: Some(text_shndx),
        rodata: Some(rodata_shndx),
        data_rel_ro: data_rel_ro_shndx,
        data: data_shndx,
        bss: bss_shndx,
        tdata: tdata_shndx,
        tbss: tbss_shndx,
    };

    // Build `.dynstr` and record string offsets.
    let mut dynstr = vec![0u8];
    let mut needed_str_offs: Vec<u64> = Vec::with_capacity(input.needed_libs.len());
    for lib in &input.needed_libs {
        needed_str_offs.push(push_cstr(&mut dynstr, lib) as u64);
    }
    let soname_str_off = soname.map(|s| push_cstr(&mut dynstr, s) as u64);

    let mut ext_dynsym_idx: BTreeMap<String, u32> = BTreeMap::new();
    let mut ext_str_offs: Vec<u32> = Vec::with_capacity(ext_syms.len());
    for (i, ext) in ext_syms.iter().enumerate() {
        ext_str_offs.push(push_cstr(&mut dynstr, ext) as u32);
        ext_dynsym_idx.insert(ext.clone(), 1 + i as u32);
    }
    let mut exp_str_offs: Vec<u32> = Vec::with_capacity(exported_names.len());
    for exp in &exported_names {
        exp_str_offs.push(push_cstr(&mut dynstr, exp) as u32);
    }

    // Build `.gnu.hash` bytes (depends only on exported names and symoffset).
    let gnu_hash_bytes = build_gnu_hash_section(&exported_names, symoffset, nbuckets);
    let dynsym_count = 1 + ext_syms.len() + exported_names.len();
    let dynsym_len = dynsym_count * SYM_SIZE;

    // Count exact `.rela.dyn` entries and `R_X86_64_RELATIVE` entries before layout!
    let mut relacount: usize = 0;
    let mut non_rel_rela_dyn_count: usize = 0;
    for sym in &got_syms {
        if local_defs.contains_key(sym) || is_synthetic_sym(sym) {
            relacount += 1;
        } else if ext_dynsym_idx.contains_key(sym) {
            non_rel_rela_dyn_count += 1;
        } else {
            return Err(format!(
                "ELF64 linker error: unresolved GOT target symbol `{sym}`"
            ));
        }
    }
    for rel in &input.relocs {
        if matches!(rel.kind, RelocKind::Abs64) {
            match rel.section {
                RelocSection::DataRelRo | RelocSection::Data => {
                    if local_defs.contains_key(&rel.target_sym)
                        || is_synthetic_sym(&rel.target_sym)
                    {
                        relacount += 1;
                    } else if ext_dynsym_idx.contains_key(&rel.target_sym) {
                        non_rel_rela_dyn_count += 1;
                    } else {
                        return Err(format!(
                            "ELF64 linker error: unresolved symbol `{}` referenced from {:?}+0x{:x}",
                            rel.target_sym, rel.section, rel.offset
                        ));
                    }
                }
                RelocSection::Text | RelocSection::Rodata | RelocSection::Tdata => {
                    return Err(format!(
                        "ELF64 linker error: Abs64 relocation in {:?} is not supported in PIE/Shared mode (use DataRelRo, Data, or GotPcRel32)",
                        rel.section
                    ));
                }
                RelocSection::Bss | RelocSection::Tbss => {
                    return Err(format!(
                        "ELF64 linker error: cannot apply relocation inside NOBITS section {:?}",
                        rel.section
                    ));
                }
            }
        }
    }
    let rela_dyn_count = relacount + non_rel_rela_dyn_count;
    let rela_dyn_len = rela_dyn_count * RELA_SIZE;
    let rela_plt_len = ext_syms.len() * RELA_SIZE;

    // Determine exact `.dynamic` tag count so its byte size is fixed before layout.
    let dyn_tag_count = needed_str_offs.len()
        + usize::from(soname_str_off.is_some())
        + 7 // DT_FLAGS, DT_FLAGS_1, DT_GNU_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_SYMENT
        + if has_plt { 4 } else { 0 } // DT_PLTGOT, DT_PLTRELSZ, DT_PLTREL, DT_JMPREL
        + if rela_dyn_count > 0 { 3 } else { 0 } // DT_RELA, DT_RELASZ, DT_RELAENT
        + usize::from(relacount > 0) // DT_RELACOUNT
        + 1; // DT_NULL
    let dynamic_len = dyn_tag_count * DYN_SIZE;

    // Determine exact program header count:
    // PT_PHDR, [PT_INTERP], 3x PT_LOAD, PT_DYNAMIC, [PT_TLS], PT_GNU_STACK, PT_GNU_RELRO
    let phnum: u16 = 7 + u16::from(has_interp) + u16::from(has_tls);
    let phdr_off: usize = EHDR_SIZE;
    let mut cur_off = phdr_off + (phnum as usize) * PHDR_SIZE;

    // ------------------------------------------------------------------------
    // Segment 0 (PF_R, file_off == vaddr):
    //   ehdr, phdrs, [.interp], .gnu.hash, .dynsym, .dynstr, .rela.dyn, .rela.plt, .rodata
    // ------------------------------------------------------------------------
    let mut interp_off = cur_off;
    if has_interp {
        cur_off = align_up(cur_off, 16);
        interp_off = cur_off;
        cur_off += INTERP_PATH.len();
    }
    let interp_vaddr = interp_off as u64;

    cur_off = align_up(cur_off, 8);
    let gnu_hash_off = cur_off;
    let gnu_hash_vaddr = gnu_hash_off as u64;
    cur_off += gnu_hash_bytes.len();

    cur_off = align_up(cur_off, 8);
    let dynsym_off = cur_off;
    let dynsym_vaddr = dynsym_off as u64;
    cur_off += dynsym_len;

    let dynstr_off = cur_off;
    let dynstr_vaddr = dynstr_off as u64;
    cur_off += dynstr.len();

    cur_off = align_up(cur_off, 8);
    let rela_dyn_off = cur_off;
    let rela_dyn_vaddr = rela_dyn_off as u64;
    cur_off += rela_dyn_len;

    cur_off = align_up(cur_off, 8);
    let rela_plt_off = cur_off;
    let rela_plt_vaddr = rela_plt_off as u64;
    cur_off += rela_plt_len;

    cur_off = align_up(cur_off, 16);
    let rodata_off = cur_off;
    let rodata_vaddr = rodata_off as u64;
    let mut rodata = input.rodata.clone();
    cur_off += rodata.len();

    let seg0_filesz = cur_off;
    let seg0_memsz = seg0_filesz;

    // ------------------------------------------------------------------------
    // Segment 1 (PF_R | PF_X, page-aligned, file_off == vaddr):
    //   .text, [.plt]
    // ------------------------------------------------------------------------
    cur_off = align_up(cur_off, PAGE_SIZE);
    let seg1_off = cur_off;
    let seg1_vaddr = seg1_off as u64;

    let text_off = cur_off;
    let text_vaddr = text_off as u64;
    let mut text = input.text.clone();
    cur_off += text.len();

    let mut plt_off = cur_off;
    let plt_len = if has_plt { 16 + ext_syms.len() * 16 } else { 0 };
    if has_plt {
        cur_off = align_up(cur_off, 16);
        plt_off = cur_off;
        cur_off += plt_len;
    }
    let plt_vaddr = plt_off as u64;
    let seg1_filesz = cur_off - seg1_off;
    let seg1_memsz = seg1_filesz;

    // ------------------------------------------------------------------------
    // Segment 2 (PF_R | PF_W, page-aligned, file_off == vaddr):
    //   RELRO prefix: [.tdata], [.data.rel.ro], .dynamic, [.got], padded to PAGE_SIZE
    //   Writable suffix: [.got.plt], [.data], [.bss]
    // ------------------------------------------------------------------------
    cur_off = align_up(cur_off, PAGE_SIZE);
    let seg2_off = cur_off;
    let seg2_vaddr = seg2_off as u64;

    // seg2_off is PAGE_SIZE-aligned, hence already TLS_ALIGN (64)-aligned.
    let tdata_off = cur_off;
    let tdata_vaddr = tdata_off as u64;
    let tdata = input.tdata.clone();
    cur_off += tdata.len();
    let tbss_tls_offset = tdata.len();
    let tls_memsz = (tdata.len() + input.tbss_size) as u64;

    let mut data_rel_ro_off = cur_off;
    let mut data_rel_ro = input.data_rel_ro.clone();
    if has_data_rel_ro {
        cur_off = align_up(cur_off, 16);
        data_rel_ro_off = cur_off;
        cur_off += data_rel_ro.len();
    }
    let data_rel_ro_vaddr = data_rel_ro_off as u64;

    cur_off = align_up(cur_off, 16);
    let dynamic_off = cur_off;
    let dynamic_vaddr = dynamic_off as u64;
    cur_off += dynamic_len;

    let mut got_off = cur_off;
    if has_got {
        cur_off = align_up(cur_off, 8);
        got_off = cur_off;
        cur_off += got_syms.len() * 8;
    }
    let got_vaddr = got_off as u64;
    let mut got = vec![0u8; got_syms.len() * 8];

    // Pad RELRO region to an exact page boundary so PT_GNU_RELRO mprotect never touches
    // `.got.plt`, `.data`, or `.bss`.
    cur_off = align_up(cur_off, PAGE_SIZE);
    let relro_size = cur_off - seg2_off;

    let got_plt_off = cur_off;
    let got_plt_vaddr = got_plt_off as u64;
    if has_plt {
        cur_off += (3 + ext_syms.len()) * 8;
    }

    let mut data_off = cur_off;
    let mut data = input.data.clone();
    if has_data {
        cur_off = align_up(cur_off, 16);
        data_off = cur_off;
        cur_off += data.len();
    }
    let data_vaddr = data_off as u64;

    let seg2_filesz = cur_off - seg2_off;
    let (bss_vaddr, seg2_memsz) = if has_bss {
        let bss_aligned_off = align_up(cur_off, 16);
        (
            bss_aligned_off as u64,
            (bss_aligned_off + input.bss_size) - seg2_off,
        )
    } else {
        (cur_off as u64, seg2_filesz)
    };

    // Populate symbol maps.
    let addrs = SectionAddrs {
        text: text_vaddr,
        rodata: rodata_vaddr,
        data_rel_ro: data_rel_ro_vaddr,
        data: data_vaddr,
        bss: bss_vaddr,
        tdata: tdata_vaddr,
        tbss_tls_offset,
    };

    let mut symbol_vaddrs: BTreeMap<String, u64> = BTreeMap::new();
    let mut tls_offsets: BTreeMap<String, u64> = BTreeMap::new();
    for s in &input.symbols {
        symbol_vaddrs.insert(s.name.clone(), addrs.sym_vaddr(s.section, s.offset));
        if let Some(toff) = addrs.tls_offset(s.section, s.offset) {
            tls_offsets.insert(s.name.clone(), toff);
        }
    }
    insert_synthetic_symbols(
        &mut symbol_vaddrs,
        0,
        rela_dyn_vaddr,
        rela_dyn_vaddr + rela_dyn_len as u64,
        tdata_vaddr,
        tdata.len() as u64,
    );

    let entry_vaddr = match &input.mode {
        ElfMode::LinuxShared { .. } => {
            if input.entry_symbol.is_empty() {
                0
            } else {
                symbol_vaddrs.get(&input.entry_symbol).copied().unwrap_or(0)
            }
        }
        ElfMode::LinuxPie => *symbol_vaddrs.get(&input.entry_symbol).ok_or_else(|| {
            format!(
                "ELF64 linker error: unresolved entry symbol `{}`",
                input.entry_symbol
            )
        })?,
        ElfMode::Freestanding { .. } => unreachable!(),
    };

    // Build `.dynsym`.
    let mut dynsym = Vec::with_capacity(dynsym_len);
    dynsym.extend_from_slice(&[0u8; SYM_SIZE]); // STN_UNDEF
    for (i, ext) in ext_syms.iter().enumerate() {
        let st_type = if ext_is_func.contains(ext.as_str()) {
            STT_FUNC
        } else {
            STT_NOTYPE
        };
        push_elf_sym(&mut dynsym, ext_str_offs[i], (STB_GLOBAL << 4) | st_type, 0, 0, 0);
    }
    for (i, exp) in exported_names.iter().enumerate() {
        let def = local_defs[exp];
        let shndx = sec_indices.get(def.section).ok_or_else(|| {
            format!(
                "ELF64 linker error: internal error missing section index for `{exp}`"
            )
        })?;
        let (st_type, st_value) = match def.section {
            RelocSection::Tdata | RelocSection::Tbss => {
                (STT_TLS, tls_offsets[exp])
            }
            _ => (
                if def.is_func { STT_FUNC } else { STT_OBJECT },
                addrs.sym_vaddr(def.section, def.offset),
            ),
        };
        push_elf_sym(
            &mut dynsym,
            exp_str_offs[i],
            (STB_GLOBAL << 4) | st_type,
            shndx,
            st_value,
            def.size,
        );
    }
    debug_assert_eq!(dynsym.len(), dynsym_len);

    // Build `.plt`, `.got.plt`, and `.rela.plt`.
    let (plt_bytes, got_plt_bytes, rela_plt_bytes, ext_plt_vaddr) = if has_plt {
        let (plt, got_plt) =
            build_plt_and_got_plt(plt_vaddr, got_plt_vaddr, dynamic_vaddr, ext_syms.len())?;
        let mut rela_plt = Vec::with_capacity(rela_plt_len);
        let mut plt_map: BTreeMap<&str, u64> = BTreeMap::new();
        for (i, ext) in ext_syms.iter().enumerate() {
            let dynsym_idx = ext_dynsym_idx[ext];
            let slot_vaddr = got_plt_vaddr + ((3 + i) * 8) as u64;
            write_rela(&mut rela_plt, slot_vaddr, dynsym_idx, R_X86_64_JUMP_SLOT, 0);
            plt_map.insert(ext.as_str(), plt_vaddr + ((1 + i) * 16) as u64);
        }
        (plt, got_plt, rela_plt, plt_map)
    } else {
        (Vec::new(), Vec::new(), Vec::new(), BTreeMap::new())
    };

    // Build `.rela.dyn` (partitioned with all `R_X86_64_RELATIVE` entries first for `DT_RELACOUNT`).
    let mut rela_dyn_relative: Vec<(u64, i64)> = Vec::with_capacity(relacount);
    let mut rela_dyn_other: Vec<(u64, u32, u32, i64)> = Vec::with_capacity(non_rel_rela_dyn_count);

    let mut got_index: BTreeMap<&str, usize> = BTreeMap::new();
    for (idx, sym) in got_syms.iter().enumerate() {
        let slot_vaddr = got_vaddr + (idx * 8) as u64;
        if let Some(&target_vaddr) = symbol_vaddrs.get(sym) {
            got[idx * 8..(idx + 1) * 8].copy_from_slice(&target_vaddr.to_le_bytes());
            rela_dyn_relative.push((slot_vaddr, target_vaddr as i64));
        } else {
            let dynsym_idx = ext_dynsym_idx[sym];
            rela_dyn_other.push((slot_vaddr, dynsym_idx, R_X86_64_GLOB_DAT, 0));
        }
        got_index.insert(sym.as_str(), idx);
    }

    // Apply all static/dynamic relocations.
    let mut dummy_tdata: [u8; 0] = [];
    let mut bufs = RelocBuffers {
        text: &mut text,
        rodata: &mut rodata,
        data_rel_ro: &mut data_rel_ro,
        data: &mut data,
        tdata: &mut dummy_tdata,
        addrs: &addrs,
        allow_tdata_reloc: false,
    };
    let tls_aligned_memsz = align_up(tls_memsz as usize, TLS_ALIGN) as i128;
    for rel in &input.relocs {
        let (buf, sec_vaddr) = bufs.slice_and_vaddr(rel.section)?;
        let place_vaddr = sec_vaddr + rel.offset as u64;
        match rel.kind {
            RelocKind::Rel32 => {
                let target_vaddr = *symbol_vaddrs.get(&rel.target_sym).ok_or_else(|| {
                    format!(
                        "ELF64 linker error: unresolved local symbol `{}` referenced from {:?}+0x{:x}",
                        rel.target_sym, rel.section, rel.offset
                    )
                })?;
                patch_rel32(
                    buf,
                    rel.offset,
                    place_vaddr,
                    target_vaddr,
                    rel.addend,
                    &rel.target_sym,
                )?;
            }
            RelocKind::Plt32 => {
                let target_vaddr = if let Some(&v) = symbol_vaddrs.get(&rel.target_sym) {
                    v
                } else if let Some(&plt_stub) = ext_plt_vaddr.get(rel.target_sym.as_str()) {
                    plt_stub
                } else {
                    return Err(format!(
                        "ELF64 linker error: unresolved symbol `{}` referenced from {:?}+0x{:x}",
                        rel.target_sym, rel.section, rel.offset
                    ));
                };
                patch_rel32(
                    buf,
                    rel.offset,
                    place_vaddr,
                    target_vaddr,
                    rel.addend,
                    &rel.target_sym,
                )?;
            }
            RelocKind::Abs64 => {
                if rel.offset + 8 > buf.len() {
                    return Err(format!(
                        "ELF64 linker error: Abs64 offset 0x{:x} out of range",
                        rel.offset
                    ));
                }
                if let Some(&target_vaddr) = symbol_vaddrs.get(&rel.target_sym) {
                    let val = ((target_vaddr as i128) + (rel.addend as i128)) as i64;
                    buf[rel.offset..rel.offset + 8].copy_from_slice(&(val as u64).to_le_bytes());
                    rela_dyn_relative.push((place_vaddr, val));
                } else {
                    let dynsym_idx = ext_dynsym_idx[&rel.target_sym];
                    buf[rel.offset..rel.offset + 8].copy_from_slice(&0u64.to_le_bytes());
                    rela_dyn_other.push((place_vaddr, dynsym_idx, R_X86_64_64, rel.addend));
                }
            }
            RelocKind::GotPcRel32 => {
                let idx = *got_index.get(rel.target_sym.as_str()).ok_or_else(|| {
                    format!(
                        "ELF64 linker error: missing GOT slot for `{}`",
                        rel.target_sym
                    )
                })?;
                let slot_vaddr = got_vaddr + (idx * 8) as u64;
                patch_rel32(
                    buf,
                    rel.offset,
                    place_vaddr,
                    slot_vaddr,
                    rel.addend,
                    &rel.target_sym,
                )?;
            }
            RelocKind::TpOff32 => {
                patch_tpoff32(buf, rel.offset, &tls_offsets, &rel.target_sym, rel.addend, tls_aligned_memsz)?;
            }
        }
    }

    let mut rela_dyn_bytes = Vec::with_capacity(rela_dyn_len);
    for (r_offset, r_addend) in &rela_dyn_relative {
        write_rela(&mut rela_dyn_bytes, *r_offset, 0, R_X86_64_RELATIVE, *r_addend);
    }
    for (r_offset, sym_idx, r_type, r_addend) in &rela_dyn_other {
        write_rela(&mut rela_dyn_bytes, *r_offset, *sym_idx, *r_type, *r_addend);
    }
    debug_assert_eq!(rela_dyn_bytes.len(), rela_dyn_len);

    // Build `.dynamic`.
    let mut dynamic_bytes = Vec::with_capacity(dynamic_len);
    for &lib_off in &needed_str_offs {
        write_dyn(&mut dynamic_bytes, DT_NEEDED, lib_off);
    }
    if let Some(soname_off) = soname_str_off {
        write_dyn(&mut dynamic_bytes, DT_SONAME, soname_off);
    }
    write_dyn(&mut dynamic_bytes, DT_FLAGS, DF_BIND_NOW);
    write_dyn(
        &mut dynamic_bytes,
        DT_FLAGS_1,
        DF_1_NOW | if is_pie { DF_1_PIE } else { 0 },
    );
    write_dyn(&mut dynamic_bytes, DT_GNU_HASH, gnu_hash_vaddr);
    write_dyn(&mut dynamic_bytes, DT_STRTAB, dynstr_vaddr);
    write_dyn(&mut dynamic_bytes, DT_SYMTAB, dynsym_vaddr);
    write_dyn(&mut dynamic_bytes, DT_STRSZ, dynstr.len() as u64);
    write_dyn(&mut dynamic_bytes, DT_SYMENT, SYM_SIZE as u64);
    if has_plt {
        write_dyn(&mut dynamic_bytes, DT_PLTGOT, got_plt_vaddr);
        write_dyn(&mut dynamic_bytes, DT_PLTRELSZ, rela_plt_len as u64);
        write_dyn(&mut dynamic_bytes, DT_PLTREL, DT_RELA_VAL);
        write_dyn(&mut dynamic_bytes, DT_JMPREL, rela_plt_vaddr);
    }
    if rela_dyn_count > 0 {
        write_dyn(&mut dynamic_bytes, DT_RELA, rela_dyn_vaddr);
        write_dyn(&mut dynamic_bytes, DT_RELASZ, rela_dyn_len as u64);
        write_dyn(&mut dynamic_bytes, DT_RELAENT, RELA_SIZE as u64);
        if relacount > 0 {
            write_dyn(&mut dynamic_bytes, DT_RELACOUNT, relacount as u64);
        }
    }
    write_dyn(&mut dynamic_bytes, DT_NULL, 0);
    debug_assert_eq!(dynamic_bytes.len(), dynamic_len);

    // Assemble loaded file bytes `[0 .. seg2_off + seg2_filesz]`.
    let total_load_filesz = seg2_off + seg2_filesz;
    let mut out = vec![0u8; total_load_filesz];
    if has_interp {
        out[interp_off..interp_off + INTERP_PATH.len()].copy_from_slice(INTERP_PATH);
    }
    out[gnu_hash_off..gnu_hash_off + gnu_hash_bytes.len()].copy_from_slice(&gnu_hash_bytes);
    out[dynsym_off..dynsym_off + dynsym.len()].copy_from_slice(&dynsym);
    out[dynstr_off..dynstr_off + dynstr.len()].copy_from_slice(&dynstr);
    out[rela_dyn_off..rela_dyn_off + rela_dyn_bytes.len()].copy_from_slice(&rela_dyn_bytes);
    if !rela_plt_bytes.is_empty() {
        out[rela_plt_off..rela_plt_off + rela_plt_bytes.len()].copy_from_slice(&rela_plt_bytes);
    }
    out[rodata_off..rodata_off + rodata.len()].copy_from_slice(&rodata);
    out[text_off..text_off + text.len()].copy_from_slice(&text);
    if !plt_bytes.is_empty() {
        out[plt_off..plt_off + plt_bytes.len()].copy_from_slice(&plt_bytes);
    }
    if !tdata.is_empty() {
        out[tdata_off..tdata_off + tdata.len()].copy_from_slice(&tdata);
    }
    if has_data_rel_ro {
        out[data_rel_ro_off..data_rel_ro_off + data_rel_ro.len()].copy_from_slice(&data_rel_ro);
    }
    out[dynamic_off..dynamic_off + dynamic_bytes.len()].copy_from_slice(&dynamic_bytes);
    if has_got {
        out[got_off..got_off + got.len()].copy_from_slice(&got);
    }
    if !got_plt_bytes.is_empty() {
        out[got_plt_off..got_plt_off + got_plt_bytes.len()].copy_from_slice(&got_plt_bytes);
    }
    if has_data {
        out[data_off..data_off + data.len()].copy_from_slice(&data);
    }

    // Build section headers in exact precomputed order.
    let mut shstrtab = vec![0u8];
    let mut shdrs = vec![Shdr::null()];

    if has_interp {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".interp") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC,
            addr: interp_vaddr,
            offset: interp_off as u64,
            size: INTERP_PATH.len() as u64,
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });
    }
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".gnu.hash") as u32,
        kind: SHT_GNU_HASH,
        flags: SHF_ALLOC,
        addr: gnu_hash_vaddr,
        offset: gnu_hash_off as u64,
        size: gnu_hash_bytes.len() as u64,
        link: dynsym_shndx as u32,
        info: 0,
        addralign: 8,
        entsize: 0,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".dynsym") as u32,
        kind: SHT_DYNSYM,
        flags: SHF_ALLOC,
        addr: dynsym_vaddr,
        offset: dynsym_off as u64,
        size: dynsym.len() as u64,
        link: dynstr_shndx as u32,
        info: 1, // first non-local symbol
        addralign: 8,
        entsize: SYM_SIZE as u64,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".dynstr") as u32,
        kind: SHT_STRTAB,
        flags: SHF_ALLOC,
        addr: dynstr_vaddr,
        offset: dynstr_off as u64,
        size: dynstr.len() as u64,
        link: 0,
        info: 0,
        addralign: 1,
        entsize: 0,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".rela.dyn") as u32,
        kind: SHT_RELA,
        flags: SHF_ALLOC,
        addr: rela_dyn_vaddr,
        offset: rela_dyn_off as u64,
        size: rela_dyn_bytes.len() as u64,
        link: dynsym_shndx as u32,
        info: 0,
        addralign: 8,
        entsize: RELA_SIZE as u64,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".rela.plt") as u32,
        kind: SHT_RELA,
        flags: SHF_ALLOC | if got_plt_shndx.is_some() { SHF_INFO_LINK } else { 0 },
        addr: rela_plt_vaddr,
        offset: rela_plt_off as u64,
        size: rela_plt_bytes.len() as u64,
        link: dynsym_shndx as u32,
        info: got_plt_shndx.unwrap_or(0) as u32,
        addralign: 8,
        entsize: RELA_SIZE as u64,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".rodata") as u32,
        kind: SHT_PROGBITS,
        flags: SHF_ALLOC,
        addr: rodata_vaddr,
        offset: rodata_off as u64,
        size: rodata.len() as u64,
        link: 0,
        info: 0,
        addralign: 16,
        entsize: 0,
    });
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".text") as u32,
        kind: SHT_PROGBITS,
        flags: SHF_ALLOC | SHF_EXECINSTR,
        addr: text_vaddr,
        offset: text_off as u64,
        size: text.len() as u64,
        link: 0,
        info: 0,
        addralign: 16,
        entsize: 0,
    });
    if has_plt {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".plt") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            addr: plt_vaddr,
            offset: plt_off as u64,
            size: plt_bytes.len() as u64,
            link: 0,
            info: 0,
            addralign: 16,
            entsize: 16,
        });
    }
    if has_tdata {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".tdata") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE | SHF_TLS,
            addr: tdata_vaddr,
            offset: tdata_off as u64,
            size: tdata.len() as u64,
            link: 0,
            info: 0,
            addralign: TLS_ALIGN as u64,
            entsize: 0,
        });
    }
    if has_tbss {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".tbss") as u32,
            kind: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE | SHF_TLS,
            addr: tdata_vaddr + tdata.len() as u64,
            offset: (tdata_off + tdata.len()) as u64,
            size: input.tbss_size as u64,
            link: 0,
            info: 0,
            addralign: tls_tbss_align(tdata.len()),
            entsize: 0,
        });
    }
    if has_data_rel_ro {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".data.rel.ro") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: data_rel_ro_vaddr,
            offset: data_rel_ro_off as u64,
            size: data_rel_ro.len() as u64,
            link: 0,
            info: 0,
            addralign: 16,
            entsize: 0,
        });
    }
    shdrs.push(Shdr {
        name: push_cstr(&mut shstrtab, ".dynamic") as u32,
        kind: SHT_DYNAMIC,
        flags: SHF_ALLOC | SHF_WRITE,
        addr: dynamic_vaddr,
        offset: dynamic_off as u64,
        size: dynamic_bytes.len() as u64,
        link: dynstr_shndx as u32,
        info: 0,
        addralign: 16,
        entsize: DYN_SIZE as u64,
    });
    if has_got {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".got") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: got_vaddr,
            offset: got_off as u64,
            size: got.len() as u64,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 8,
        });
    }
    if has_plt {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".got.plt") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: got_plt_vaddr,
            offset: got_plt_off as u64,
            size: got_plt_bytes.len() as u64,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 8,
        });
    }
    if has_data {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".data") as u32,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: data_vaddr,
            offset: data_off as u64,
            size: data.len() as u64,
            link: 0,
            info: 0,
            addralign: 16,
            entsize: 0,
        });
    }
    if has_bss {
        shdrs.push(Shdr {
            name: push_cstr(&mut shstrtab, ".bss") as u32,
            kind: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            addr: bss_vaddr,
            offset: total_load_filesz as u64,
            size: input.bss_size as u64,
            link: 0,
            info: 0,
            addralign: 16,
            entsize: 0,
        });
    }

    debug_assert_eq!(shdrs.len(), next_shndx as usize);

    let (shdr_off, shstrtab_idx) = append_debug_and_symtab_sections(
        &mut out,
        input,
        text_vaddr,
        text.len() as u64,
        &addrs,
        &sec_indices,
        &mut shstrtab,
        &mut shdrs,
    );

    let ehdr = build_ehdr(
        ET_DYN,
        entry_vaddr,
        phdr_off as u64,
        shdr_off as u64,
        phnum,
        shdrs.len() as u16,
        shstrtab_idx,
    );
    out[0..EHDR_SIZE].copy_from_slice(&ehdr);

    let mut phdrs = Vec::with_capacity((phnum as usize) * PHDR_SIZE);
    let phdr_table_bytes = (phnum as u64) * (PHDR_SIZE as u64);
    write_phdr(
        &mut phdrs,
        PT_PHDR,
        PF_R,
        phdr_off as u64,
        phdr_off as u64,
        phdr_off as u64,
        phdr_table_bytes,
        phdr_table_bytes,
        8,
    );
    if has_interp {
        write_phdr(
            &mut phdrs,
            PT_INTERP,
            PF_R,
            interp_off as u64,
            interp_vaddr,
            interp_vaddr,
            INTERP_PATH.len() as u64,
            INTERP_PATH.len() as u64,
            1,
        );
    }
    // Segment 0: PF_R
    write_phdr(
        &mut phdrs,
        PT_LOAD,
        PF_R,
        0,
        0,
        0,
        seg0_filesz as u64,
        seg0_memsz as u64,
        PAGE_SIZE as u64,
    );
    // Segment 1: PF_R | PF_X
    write_phdr(
        &mut phdrs,
        PT_LOAD,
        PF_R | PF_X,
        seg1_off as u64,
        seg1_vaddr,
        seg1_vaddr,
        seg1_filesz as u64,
        seg1_memsz as u64,
        PAGE_SIZE as u64,
    );
    // Segment 2: PF_R | PF_W
    write_phdr(
        &mut phdrs,
        PT_LOAD,
        PF_R | PF_W,
        seg2_off as u64,
        seg2_vaddr,
        seg2_vaddr,
        seg2_filesz as u64,
        seg2_memsz as u64,
        PAGE_SIZE as u64,
    );
    write_phdr(
        &mut phdrs,
        PT_DYNAMIC,
        PF_R | PF_W,
        dynamic_off as u64,
        dynamic_vaddr,
        dynamic_vaddr,
        dynamic_bytes.len() as u64,
        dynamic_bytes.len() as u64,
        8,
    );
    if has_tls {
        write_phdr(
            &mut phdrs,
            PT_TLS,
            PF_R,
            tdata_off as u64,
            tdata_vaddr,
            tdata_vaddr,
            tdata.len() as u64,
            tls_memsz,
            TLS_ALIGN as u64,
        );
    }
    write_phdr(&mut phdrs, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 0, 16);
    write_phdr(
        &mut phdrs,
        PT_GNU_RELRO,
        PF_R,
        seg2_off as u64,
        seg2_vaddr,
        seg2_vaddr,
        relro_size as u64,
        relro_size as u64,
        1,
    );
    debug_assert_eq!(phdrs.len(), (phnum as usize) * PHDR_SIZE);
    out[phdr_off..phdr_off + phdrs.len()].copy_from_slice(&phdrs);

    Ok(LinkedElf {
        bytes: out,
        entry_vaddr,
        text_vaddr,
        rodata_vaddr,
        data_rel_ro_vaddr,
        data_vaddr,
        bss_vaddr,
        tdata_vaddr,
        tls_memsz,
        symbol_vaddrs,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_debug_and_symtab_sections(
    out: &mut Vec<u8>,
    input: &ElfLinkInput,
    text_vaddr: u64,
    text_len: u64,
    addrs: &SectionAddrs,
    sec_indices: &SectionIndices,
    shstrtab: &mut Vec<u8>,
    shdrs: &mut Vec<Shdr>,
) -> (usize, u16) {
    let debug_line = if input.emit_debug_line {
        build_dwarf5_debug_line(text_vaddr, text_len, &input.source_files, &input.line_entries)
    } else {
        Vec::new()
    };
    let mut debug_sections: Vec<(&'static str, Vec<u8>)> = Vec::new();
    if !debug_line.is_empty() {
        let primary = input.source_files.first().map(String::as_str).unwrap_or("<eon>");
        let primary = primary.strip_prefix("./").unwrap_or(primary);
        let (abbrev, info) = build_dwarf5_debug_info(text_vaddr, text_len, primary);
        debug_sections.push((".debug_abbrev", abbrev));
        debug_sections.push((".debug_info", info));
        debug_sections.push((".debug_line", debug_line));
    }

    let mut strtab = vec![0u8];
    let mut symtab = vec![0u8; SYM_SIZE]; // STN_UNDEF
    for s in &input.symbols {
        let name_off = push_cstr(&mut strtab, &s.name) as u32;
        let shndx = sec_indices.get(s.section).unwrap_or(0);
        let (st_type, st_val) = match s.section {
            RelocSection::Tdata | RelocSection::Tbss => {
                (STT_TLS, addrs.tls_offset(s.section, s.offset).unwrap_or(0))
            }
            _ => (
                if s.is_func { STT_FUNC } else { STT_OBJECT },
                addrs.sym_vaddr(s.section, s.offset),
            ),
        };
        push_elf_sym(
            &mut symtab,
            name_off,
            (STB_GLOBAL << 4) | st_type,
            shndx,
            st_val,
            s.size,
        );
    }

    let shn_debug: Vec<u32> = debug_sections
        .iter()
        .map(|(name, _)| push_cstr(shstrtab, name) as u32)
        .collect();
    let shn_symtab = push_cstr(shstrtab, ".symtab") as u32;
    let shn_strtab = push_cstr(shstrtab, ".strtab") as u32;
    let shn_shstrtab = push_cstr(shstrtab, ".shstrtab") as u32;

    let mut debug_offs: Vec<usize> = Vec::with_capacity(debug_sections.len());
    for (_, bytes) in &debug_sections {
        pad_to(out, 4);
        debug_offs.push(out.len());
        out.extend_from_slice(bytes);
    }

    pad_to(out, 8);
    let symtab_off = out.len();
    out.extend_from_slice(&symtab);

    let strtab_off = out.len();
    out.extend_from_slice(&strtab);

    let shstrtab_off = out.len();
    out.extend_from_slice(shstrtab);

    pad_to(out, 8);
    let shdr_off = out.len();

    for (((_, bytes), &name), &off) in debug_sections.iter().zip(&shn_debug).zip(&debug_offs) {
        shdrs.push(Shdr {
            name,
            kind: SHT_PROGBITS,
            flags: 0,
            addr: 0,
            offset: off as u64,
            size: bytes.len() as u64,
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });
    }

    let strtab_idx = (shdrs.len() + 1) as u32;
    shdrs.push(Shdr {
        name: shn_symtab,
        kind: SHT_SYMTAB,
        flags: 0,
        addr: 0,
        offset: symtab_off as u64,
        size: symtab.len() as u64,
        link: strtab_idx,
        info: 1,
        addralign: 8,
        entsize: SYM_SIZE as u64,
    });
    shdrs.push(Shdr {
        name: shn_strtab,
        kind: SHT_STRTAB,
        flags: 0,
        addr: 0,
        offset: strtab_off as u64,
        size: strtab.len() as u64,
        link: 0,
        info: 0,
        addralign: 1,
        entsize: 0,
    });
    let shstrtab_idx = shdrs.len() as u16;
    shdrs.push(Shdr {
        name: shn_shstrtab,
        kind: SHT_STRTAB,
        flags: 0,
        addr: 0,
        offset: shstrtab_off as u64,
        size: shstrtab.len() as u64,
        link: 0,
        info: 0,
        addralign: 1,
        entsize: 0,
    });

    for sh in shdrs.iter() {
        sh.write_to(out);
    }

    (shdr_off, shstrtab_idx)
}

fn has_sym_or_reloc_in(input: &ElfLinkInput, sec: RelocSection) -> bool {
    input.symbols.iter().any(|s| s.section == sec)
        || input.relocs.iter().any(|r| r.section == sec)
}

fn collect_got_symbols(input: &ElfLinkInput) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::new();
    for rel in &input.relocs {
        if matches!(rel.kind, RelocKind::GotPcRel32) && seen.insert(rel.target_sym.clone()) {
            out.push(rel.target_sym.clone());
        }
    }
    out
}

fn is_synthetic_sym(name: &str) -> bool {
    matches!(
        name,
        "__eon_ehdr_start"
            | "__eon_rela_dyn_start"
            | "__eon_rela_dyn_end"
            | "__eon_tdata_start"
            | "__eon_tdata_end"
    )
}

fn insert_synthetic_symbols(
    symbol_vaddrs: &mut BTreeMap<String, u64>,
    ehdr_start: u64,
    rela_dyn_start: u64,
    rela_dyn_end: u64,
    tdata_start: u64,
    tdata_len: u64,
) {
    symbol_vaddrs
        .entry("__eon_ehdr_start".to_owned())
        .or_insert(ehdr_start);
    symbol_vaddrs
        .entry("__eon_rela_dyn_start".to_owned())
        .or_insert(rela_dyn_start);
    symbol_vaddrs
        .entry("__eon_rela_dyn_end".to_owned())
        .or_insert(rela_dyn_end);
    symbol_vaddrs
        .entry("__eon_tdata_start".to_owned())
        .or_insert(tdata_start);
    symbol_vaddrs
        .entry("__eon_tdata_end".to_owned())
        .or_insert(tdata_start + tdata_len);
}

fn tls_tbss_align(tdata_len: usize) -> u64 {
    if tdata_len == 0 {
        TLS_ALIGN as u64
    } else {
        1u64 << tdata_len.trailing_zeros().min(6)
    }
}

fn patch_rel32(
    buf: &mut [u8],
    offset: usize,
    place_vaddr: u64,
    target_vaddr: u64,
    addend: i64,
    sym: &str,
) -> Result<(), String> {
    if offset + 4 > buf.len() {
        return Err(format!(
            "ELF64 linker error: Rel32/Plt32/GotPcRel32 offset 0x{offset:x} out of range"
        ));
    }
    let disp = (target_vaddr as i128) + (addend as i128) - ((place_vaddr as i128) + 4);
    let disp32 = i32::try_from(disp).map_err(|_| {
        format!("ELF64 linker error: 32-bit PC-relative overflow ({disp}) for symbol `{sym}`")
    })?;
    buf[offset..offset + 4].copy_from_slice(&disp32.to_le_bytes());
    Ok(())
}

struct RelocBuffers<'a> {
    text: &'a mut [u8],
    rodata: &'a mut [u8],
    data_rel_ro: &'a mut [u8],
    data: &'a mut [u8],
    tdata: &'a mut [u8],
    addrs: &'a SectionAddrs,
    allow_tdata_reloc: bool,
}

impl<'a> RelocBuffers<'a> {
    fn slice_and_vaddr(&mut self, sec: RelocSection) -> Result<(&mut [u8], u64), String> {
        match sec {
            RelocSection::Text => Ok((self.text, self.addrs.text)),
            RelocSection::Rodata => Ok((self.rodata, self.addrs.rodata)),
            RelocSection::DataRelRo => Ok((self.data_rel_ro, self.addrs.data_rel_ro)),
            RelocSection::Data => Ok((self.data, self.addrs.data)),
            RelocSection::Tdata => {
                if self.allow_tdata_reloc {
                    Ok((self.tdata, self.addrs.tdata))
                } else {
                    Err("ELF64 linker error: relocation inside `.tdata` template is not supported in PIE/Shared mode".to_owned())
                }
            }
            RelocSection::Bss | RelocSection::Tbss => Err(format!(
                "ELF64 linker error: cannot apply relocation inside NOBITS section {sec:?}"
            )),
        }
    }
}

fn patch_tpoff32(
    buf: &mut [u8],
    offset: usize,
    tls_offsets: &BTreeMap<String, u64>,
    sym: &str,
    addend: i64,
    tls_aligned_memsz: i128,
) -> Result<(), String> {
    let toff = *tls_offsets.get(sym).ok_or_else(|| {
        format!("ELF64 linker error: TpOff32 relocation targeting non-TLS or unresolved symbol `{sym}`")
    })?;
    if offset + 4 > buf.len() {
        return Err(format!(
            "ELF64 linker error: TpOff32 offset 0x{offset:x} out of range"
        ));
    }
    let disp = (toff as i128) + (addend as i128) - tls_aligned_memsz;
    let disp32 = i32::try_from(disp).map_err(|_| {
        format!("ELF64 linker error: TpOff32 overflow ({disp}) for symbol `{sym}`")
    })?;
    buf[offset..offset + 4].copy_from_slice(&disp32.to_le_bytes());
    Ok(())
}

fn gnu_hash(name: &str) -> u32 {
    let mut h: u32 = 5381;
    for &b in name.as_bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

fn build_gnu_hash_section(exported_sorted: &[String], symoffset: u32, nbuckets: u32) -> Vec<u8> {
    let bloom_size: u32 = if exported_sorted.len() <= 8 {
        1
    } else {
        (exported_sorted.len() / 8).next_power_of_two().max(1) as u32
    };
    let bloom_shift: u32 = 6;

    let mut bloom = vec![0u64; bloom_size as usize];
    let mut buckets = vec![0u32; nbuckets as usize];
    let mut chains = Vec::with_capacity(exported_sorted.len());

    let hashes: Vec<u32> = exported_sorted.iter().map(|s| gnu_hash(s)).collect();
    for (i, &h) in hashes.iter().enumerate() {
        let dynsym_idx = symoffset + i as u32;
        let bloom_idx = ((h as usize) / 64) & ((bloom_size as usize) - 1);
        bloom[bloom_idx] |= (1u64 << (h % 64)) | (1u64 << ((h >> bloom_shift) % 64));

        let b = (h % nbuckets) as usize;
        if buckets[b] == 0 {
            buckets[b] = dynsym_idx;
        }
        let is_last = i + 1 == hashes.len() || (hashes[i + 1] % nbuckets) as usize != b;
        chains.push(if is_last { h | 1 } else { h & !1 });
    }

    let mut out = Vec::with_capacity(
        16 + (bloom.len() * 8) + (buckets.len() * 4) + (chains.len() * 4),
    );
    out.extend_from_slice(&nbuckets.to_le_bytes());
    out.extend_from_slice(&symoffset.to_le_bytes());
    out.extend_from_slice(&bloom_size.to_le_bytes());
    out.extend_from_slice(&bloom_shift.to_le_bytes());
    for w in bloom {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for b in buckets {
        out.extend_from_slice(&b.to_le_bytes());
    }
    for c in chains {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

fn build_plt_and_got_plt(
    plt_vaddr: u64,
    got_plt_vaddr: u64,
    dynamic_vaddr: u64,
    ext_count: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut plt = Vec::with_capacity((1 + ext_count) * 16);
    let mut got_plt = Vec::with_capacity((3 + ext_count) * 8);

    // GOT[0] = &_DYNAMIC, GOT[1] = 0 (link_map), GOT[2] = 0 (dl_runtime_resolve).
    got_plt.extend_from_slice(&dynamic_vaddr.to_le_bytes());
    got_plt.extend_from_slice(&0u64.to_le_bytes());
    got_plt.extend_from_slice(&0u64.to_le_bytes());

    // PLT0 (16 B):
    //   ff 35 <rel32 to GOT[1]>   (push [rip + GOT+8])
    //   ff 25 <rel32 to GOT[2]>   (jmp  [rip + GOT+16])
    //   0f 1f 40 00               (nop4)
    let disp_got1 = i32::try_from((got_plt_vaddr + 8) as i128 - (plt_vaddr + 6) as i128)
        .map_err(|_| "ELF64 linker error: PLT0 -> GOT[1] displacement overflow".to_owned())?;
    let disp_got2 = i32::try_from((got_plt_vaddr + 16) as i128 - (plt_vaddr + 12) as i128)
        .map_err(|_| "ELF64 linker error: PLT0 -> GOT[2] displacement overflow".to_owned())?;
    plt.extend_from_slice(&[0xFF, 0x35]);
    plt.extend_from_slice(&disp_got1.to_le_bytes());
    plt.extend_from_slice(&[0xFF, 0x25]);
    plt.extend_from_slice(&disp_got2.to_le_bytes());
    plt.extend_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);

    for i in 0..ext_count {
        let stub_vaddr = plt_vaddr + ((1 + i) * 16) as u64;
        let slot_vaddr = got_plt_vaddr + ((3 + i) * 8) as u64;
        let disp_slot = i32::try_from(slot_vaddr as i128 - (stub_vaddr + 6) as i128)
            .map_err(|_| "ELF64 linker error: PLT -> GOT.PLT displacement overflow".to_owned())?;
        let disp_plt0 = i32::try_from(plt_vaddr as i128 - (stub_vaddr + 16) as i128)
            .map_err(|_| "ELF64 linker error: PLT -> PLT0 displacement overflow".to_owned())?;

        // jmp qword ptr [rip + got_plt[3 + i]]
        plt.extend_from_slice(&[0xFF, 0x25]);
        plt.extend_from_slice(&disp_slot.to_le_bytes());
        // push i
        plt.push(0x68);
        plt.extend_from_slice(&(i as u32).to_le_bytes());
        // jmp PLT0
        plt.push(0xE9);
        plt.extend_from_slice(&disp_plt0.to_le_bytes());

        // Lazy fallback value points to `push i` right after the indirect jump.
        got_plt.extend_from_slice(&(stub_vaddr + 6).to_le_bytes());
    }

    Ok((plt, got_plt))
}

fn push_elf_sym(
    symtab: &mut Vec<u8>,
    name_off: u32,
    st_info: u8,
    shndx: u16,
    st_value: u64,
    st_size: u64,
) {
    symtab.extend_from_slice(&name_off.to_le_bytes());
    symtab.push(st_info);
    symtab.push(0); // st_other = STV_DEFAULT
    symtab.extend_from_slice(&shndx.to_le_bytes());
    symtab.extend_from_slice(&st_value.to_le_bytes());
    symtab.extend_from_slice(&st_size.to_le_bytes());
}

fn write_rela(out: &mut Vec<u8>, r_offset: u64, sym_idx: u32, r_type: u32, r_addend: i64) {
    let r_info: u64 = ((sym_idx as u64) << 32) | (r_type as u64);
    out.extend_from_slice(&r_offset.to_le_bytes());
    out.extend_from_slice(&r_info.to_le_bytes());
    out.extend_from_slice(&r_addend.to_le_bytes());
}

fn write_dyn(out: &mut Vec<u8>, d_tag: i64, d_val: u64) {
    out.extend_from_slice(&d_tag.to_le_bytes());
    out.extend_from_slice(&d_val.to_le_bytes());
}

fn build_ehdr(
    e_type: u16,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_phnum: u16,
    e_shnum: u16,
    e_shstrndx: u16,
) -> Vec<u8> {
    let mut ehdr = Vec::with_capacity(EHDR_SIZE);
    ehdr.extend_from_slice(&ELF_MAGIC);
    ehdr.push(ELFCLASS64);
    ehdr.push(ELFDATA2LSB);
    ehdr.push(EV_CURRENT);
    ehdr.push(0); // ELFOSABI_NONE
    ehdr.extend_from_slice(&[0u8; 8]);
    ehdr.extend_from_slice(&e_type.to_le_bytes());
    ehdr.extend_from_slice(&EM_X86_64.to_le_bytes());
    ehdr.extend_from_slice(&(EV_CURRENT as u32).to_le_bytes());
    ehdr.extend_from_slice(&e_entry.to_le_bytes());
    ehdr.extend_from_slice(&e_phoff.to_le_bytes());
    ehdr.extend_from_slice(&e_shoff.to_le_bytes());
    ehdr.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    ehdr.extend_from_slice(&(EHDR_SIZE as u16).to_le_bytes());
    ehdr.extend_from_slice(&(PHDR_SIZE as u16).to_le_bytes());
    ehdr.extend_from_slice(&e_phnum.to_le_bytes());
    ehdr.extend_from_slice(&(SHDR_SIZE as u16).to_le_bytes());
    ehdr.extend_from_slice(&e_shnum.to_le_bytes());
    ehdr.extend_from_slice(&e_shstrndx.to_le_bytes());
    debug_assert_eq!(ehdr.len(), EHDR_SIZE);
    ehdr
}

#[allow(clippy::too_many_arguments)]
fn write_phdr(
    out: &mut Vec<u8>,
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
) {
    out.extend_from_slice(&p_type.to_le_bytes());
    out.extend_from_slice(&p_flags.to_le_bytes());
    out.extend_from_slice(&p_offset.to_le_bytes());
    out.extend_from_slice(&p_vaddr.to_le_bytes());
    out.extend_from_slice(&p_paddr.to_le_bytes());
    out.extend_from_slice(&p_filesz.to_le_bytes());
    out.extend_from_slice(&p_memsz.to_le_bytes());
    out.extend_from_slice(&p_align.to_le_bytes());
}

/// Build a valid Multiboot2 header (24 bytes, 8-byte aligned) for freestanding bare-metal kernels.
fn build_multiboot2_header() -> Vec<u8> {
    let magic: u32 = 0xE852_50D6;
    let arch: u32 = 0; // x86 / protected-mode entry
    let header_len: u32 = 24;
    let checksum: u32 = 0u32.wrapping_sub(magic.wrapping_add(arch).wrapping_add(header_len));
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&magic.to_le_bytes());
    v.extend_from_slice(&arch.to_le_bytes());
    v.extend_from_slice(&header_len.to_le_bytes());
    v.extend_from_slice(&checksum.to_le_bytes());
    // End tag: type=0 (u16), flags=0 (u16), size=8 (u32)
    v.extend_from_slice(&0u16.to_le_bytes());
    v.extend_from_slice(&0u16.to_le_bytes());
    v.extend_from_slice(&8u32.to_le_bytes());
    v
}

/// Build a standard DWARF Version 5 `.debug_line` section compatible with `gdb`, `lldb`, and `readelf -wl`.
pub fn build_dwarf5_debug_line(
    text_vaddr: u64,
    text_len: u64,
    source_files: &[String],
    line_entries: &[LineEntry],
) -> Vec<u8> {
    let files: Vec<&str> = if source_files.is_empty() {
        vec!["<eon>"]
    } else {
        // Paths are relative to directory entry 0 (`.`); a leading `./` would show up as `././x.eon`.
        source_files.iter().map(|s| s.strip_prefix("./").unwrap_or(s)).collect()
    };

    // Build header body (everything after `header_length` field).
    let mut hdr = vec![
        1u8,           // minimum_instruction_length
        1u8,           // maximum_operations_per_instruction
        1u8,           // default_is_stmt
        (-5i8) as u8,  // line_base
        14u8,          // line_range
        13u8,          // opcode_base
    ];
    hdr.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]); // standard_opcode_lengths (1..12)

    // Directory table (DWARF5): 1 format entry (DW_LNCT_path=1, DW_FORM_string=8), 1 dir "."
    hdr.push(1u8); // directory_entry_format_count
    write_uleb128(&mut hdr, 1); // DW_LNCT_path
    write_uleb128(&mut hdr, 8); // DW_FORM_string
    write_uleb128(&mut hdr, 1); // directories_count
    hdr.extend_from_slice(b".\0");

    // File name table (DWARF5): 2 format entries (DW_LNCT_path + DW_LNCT_directory_index)
    hdr.push(2u8); // file_name_entry_format_count
    write_uleb128(&mut hdr, 1); // DW_LNCT_path
    write_uleb128(&mut hdr, 8); // DW_FORM_string
    write_uleb128(&mut hdr, 2); // DW_LNCT_directory_index
    write_uleb128(&mut hdr, 15); // DW_FORM_udata
    write_uleb128(&mut hdr, files.len() as u64);
    for f in &files {
        hdr.extend_from_slice(f.as_bytes());
        hdr.push(0);
        write_uleb128(&mut hdr, 0); // directory index 0
    }

    // Build line number program opcodes.
    let mut prog = Vec::new();
    let mut sorted = line_entries.to_vec();
    sorted.sort_by_key(|e| (e.code_offset, e.file_idx, e.line, e.col));
    sorted.dedup_by_key(|e| e.code_offset);

    // DW_LNE_set_address(text_vaddr)
    prog.push(0u8);
    write_uleb128(&mut prog, 9);
    prog.push(2u8); // DW_LNE_set_address
    prog.extend_from_slice(&text_vaddr.to_le_bytes());

    let mut cur_pc: u64 = 0;
    // DWARF5 §6.2.2: the state machine's `file` register starts at 1, not 0. Starting our shadow
    // copy at 1 guarantees that entries for file 0 (the primary source) emit `DW_LNS_set_file 0`;
    // otherwise every row was silently attributed to the second file in the table.
    let mut cur_file: u32 = 1;
    let mut cur_line: i64 = 1;
    let mut cur_col: u32 = 0;

    for e in &sorted {
        let target_pc = e.code_offset as u64;
        if target_pc < cur_pc {
            continue;
        }
        let target_file = e.file_idx.min((files.len().saturating_sub(1)) as u32);
        if target_file != cur_file {
            prog.push(4u8); // DW_LNS_set_file
            write_uleb128(&mut prog, target_file as u64);
            cur_file = target_file;
        }
        if e.col != cur_col {
            prog.push(5u8); // DW_LNS_set_column
            write_uleb128(&mut prog, e.col as u64);
            cur_col = e.col;
        }
        let target_line = e.line.max(1) as i64;
        let dline = target_line - cur_line;
        if dline != 0 {
            prog.push(3u8); // DW_LNS_advance_line
            write_sleb128(&mut prog, dline);
            cur_line = target_line;
        }
        let dpc = target_pc - cur_pc;
        if dpc != 0 {
            prog.push(2u8); // DW_LNS_advance_pc
            write_uleb128(&mut prog, dpc);
            cur_pc = target_pc;
        }
        prog.push(1u8); // DW_LNS_copy
    }

    if text_len > cur_pc {
        prog.push(2u8); // DW_LNS_advance_pc
        write_uleb128(&mut prog, text_len - cur_pc);
    }
    // DW_LNE_end_sequence
    prog.push(0u8);
    prog.push(1u8);
    prog.push(1u8);

    // Assemble complete DWARF5 line unit.
    let unit_body_len = 2 + 1 + 1 + 4 + hdr.len() + prog.len();
    let mut out = Vec::with_capacity(4 + unit_body_len);
    out.extend_from_slice(&(unit_body_len as u32).to_le_bytes()); // unit_length
    out.extend_from_slice(&5u16.to_le_bytes()); // version = 5
    out.push(8u8); // address_size = 8
    out.push(0u8); // segment_selector_size = 0
    out.extend_from_slice(&(hdr.len() as u32).to_le_bytes()); // header_length
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&prog);
    out
}

/// Build a minimal DWARF5 `.debug_abbrev` + `.debug_info` pair: one `DW_TAG_compile_unit` covering
/// the whole `.text` and pointing at `.debug_line` offset 0 via `DW_AT_stmt_list`.
///
/// Debuggers (`gdb`, `lldb`) only consult line tables that are referenced from a compile unit,
/// so `.debug_line` alone is invisible to `break file.eon:N`. Paths stay relative (`comp_dir = "."`)
/// to keep builds bit-for-bit reproducible.
pub fn build_dwarf5_debug_info(text_vaddr: u64, text_len: u64, primary_file: &str) -> (Vec<u8>, Vec<u8>) {
    const DW_TAG_COMPILE_UNIT: u64 = 0x11;
    const DW_CHILDREN_NO: u8 = 0;
    const DW_AT_NAME: u64 = 0x03;
    const DW_AT_STMT_LIST: u64 = 0x10;
    const DW_AT_LOW_PC: u64 = 0x11;
    const DW_AT_HIGH_PC: u64 = 0x12;
    const DW_AT_LANGUAGE: u64 = 0x13;
    const DW_AT_COMP_DIR: u64 = 0x1b;
    const DW_AT_PRODUCER: u64 = 0x25;
    const DW_FORM_ADDR: u64 = 0x01;
    const DW_FORM_DATA2: u64 = 0x05;
    const DW_FORM_DATA8: u64 = 0x07;
    const DW_FORM_STRING: u64 = 0x08;
    const DW_FORM_SEC_OFFSET: u64 = 0x17;
    const DW_UT_COMPILE: u8 = 0x01;
    // No registered DW_LANG for Eon yet; C11 gives debuggers sane expression/value printing.
    const DW_LANG_C11: u16 = 0x1d;

    let mut abbrev = Vec::new();
    write_uleb128(&mut abbrev, 1); // abbrev code
    write_uleb128(&mut abbrev, DW_TAG_COMPILE_UNIT);
    abbrev.push(DW_CHILDREN_NO);
    for (at, form) in [
        (DW_AT_PRODUCER, DW_FORM_STRING),
        (DW_AT_LANGUAGE, DW_FORM_DATA2),
        (DW_AT_NAME, DW_FORM_STRING),
        (DW_AT_COMP_DIR, DW_FORM_STRING),
        (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
        (DW_AT_LOW_PC, DW_FORM_ADDR),
        (DW_AT_HIGH_PC, DW_FORM_DATA8),
    ] {
        write_uleb128(&mut abbrev, at);
        write_uleb128(&mut abbrev, form);
    }
    abbrev.extend_from_slice(&[0, 0]); // end of attribute list
    abbrev.push(0); // end of abbreviation table

    let mut die = Vec::new();
    write_uleb128(&mut die, 1);
    die.extend_from_slice(concat!("eonc ", env!("CARGO_PKG_VERSION")).as_bytes());
    die.push(0);
    die.extend_from_slice(&DW_LANG_C11.to_le_bytes());
    die.extend_from_slice(primary_file.as_bytes());
    die.push(0);
    die.extend_from_slice(b".\0");
    die.extend_from_slice(&0u32.to_le_bytes()); // stmt_list -> .debug_line + 0
    die.extend_from_slice(&text_vaddr.to_le_bytes());
    die.extend_from_slice(&text_len.to_le_bytes()); // high_pc as length (DWARF4+ class constant)

    let unit_body_len = 2 + 1 + 1 + 4 + die.len();
    let mut info = Vec::with_capacity(4 + unit_body_len);
    info.extend_from_slice(&(unit_body_len as u32).to_le_bytes());
    info.extend_from_slice(&5u16.to_le_bytes()); // version
    info.push(DW_UT_COMPILE);
    info.push(8); // address_size
    info.extend_from_slice(&0u32.to_le_bytes()); // debug_abbrev_offset
    info.extend_from_slice(&die);
    (abbrev, info)
}

fn write_uleb128(out: &mut Vec<u8>, mut val: u64) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if val == 0 {
            break;
        }
    }
}

fn write_sleb128(out: &mut Vec<u8>, mut val: i64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        let done = (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0);
        if done {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

struct Shdr {
    name: u32,
    kind: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
}

impl Shdr {
    fn null() -> Self {
        Self {
            name: 0,
            kind: SHT_NULL,
            flags: 0,
            addr: 0,
            offset: 0,
            size: 0,
            link: 0,
            info: 0,
            addralign: 0,
            entsize: 0,
        }
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.name.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&self.link.to_le_bytes());
        out.extend_from_slice(&self.info.to_le_bytes());
        out.extend_from_slice(&self.addralign.to_le_bytes());
        out.extend_from_slice(&self.entsize.to_le_bytes());
    }
}

fn push_cstr(buf: &mut Vec<u8>, s: &str) -> usize {
    let off = buf.len();
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
    off
}

fn align_up(v: usize, a: usize) -> usize {
    if a <= 1 { v } else { (v + a - 1) & !(a - 1) }
}

fn pad_to(buf: &mut Vec<u8>, align: usize) {
    let target = align_up(buf.len(), align);
    buf.resize(target, 0);
}
