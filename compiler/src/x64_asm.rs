//! Direct x86_64 machine-code encoder, inline mini-assembler, and standalone
//! zero-libc Linux x86_64 runtime helper emitters (NO external assembler or libc).

use crate::elf64::RelocKind;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Reg {
    #[inline]
    pub fn id(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn low(self) -> u8 {
        (self as u8) & 7
    }

    #[inline]
    pub fn hi(self) -> bool {
        (self as u8) >= 8
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Xmm {
    Xmm0 = 0,
    Xmm1 = 1,
    Xmm2 = 2,
    Xmm3 = 3,
    Xmm4 = 4,
    Xmm5 = 5,
    Xmm6 = 6,
    Xmm7 = 7,
}

impl Xmm {
    #[inline]
    pub fn id(self) -> u8 {
        self as u8
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    O = 0x0,
    No = 0x1,
    B = 0x2,  // unsigned <  (also C / carry set)
    Ae = 0x3, // unsigned >= (also NC / carry clear)
    E = 0x4,  // ==
    Ne = 0x5, // !=
    Be = 0x6, // unsigned <=
    A = 0x7,  // unsigned >
    P = 0xA,  // parity (unordered NaN after ucomi*)
    Np = 0xB, // no parity (ordered after ucomi*)
    L = 0xC,  // signed <
    Ge = 0xD, // signed >=
    Le = 0xE, // signed <=
    G = 0xF,  // signed >
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Label(pub usize);

#[derive(Clone, Debug)]
pub struct FnReloc {
    pub offset: usize,
    pub kind: RelocKind,
    pub target_sym: String,
    pub addend: i64,
}

pub struct X64Buf {
    pub code: Vec<u8>,
    pub relocs: Vec<FnReloc>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label)>,
}

impl Default for X64Buf {
    fn default() -> Self {
        Self::new()
    }
}

impl X64Buf {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            relocs: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.code.len()
    }

    pub fn new_label(&mut self) -> Label {
        let id = self.labels.len();
        self.labels.push(None);
        Label(id)
    }

    pub fn bind_label(&mut self, l: Label) {
        self.labels[l.0] = Some(self.code.len());
    }

    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<FnReloc>), String> {
        for (patch_off, lbl) in self.fixups {
            let Some(target) = self.labels[lbl.0] else {
                return Err(format!("x86_64 asm error: unbound local label {}", lbl.0));
            };
            let disp = (target as i64) - ((patch_off as i64) + 4);
            let disp32 = i32::try_from(disp)
                .map_err(|_| format!("x86_64 asm error: branch displacement {disp} out of range"))?;
            self.code[patch_off..patch_off + 4].copy_from_slice(&disp32.to_le_bytes());
        }
        Ok((self.code, self.relocs))
    }

    #[inline]
    pub fn emit_u8(&mut self, b: u8) {
        self.code.push(b);
    }

    #[inline]
    pub fn emit_u32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn emit_i32(&mut self, v: i32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn emit_u64(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_rex(&mut self, w: bool, r: bool, x: bool, b: bool, force: bool) {
        let v = 0x40
            | (if w { 0x08 } else { 0 })
            | (if r { 0x04 } else { 0 })
            | (if x { 0x02 } else { 0 })
            | (if b { 0x01 } else { 0 });
        if force || v != 0x40 {
            self.emit_u8(v);
        }
    }

    #[inline]
    fn modrm(m: u8, reg: u8, rm: u8) -> u8 {
        ((m & 3) << 6) | ((reg & 7) << 3) | (rm & 7)
    }

    fn emit_mem(&mut self, reg_field: u8, base: Reg, disp: i32) {
        let bl = base.low();
        if disp == 0 && bl != 5 {
            self.emit_u8(Self::modrm(0, reg_field, bl));
            if bl == 4 {
                self.emit_u8(0x24); // SIB for [rsp]/[r12]
            }
        } else if (-128..=127).contains(&disp) {
            self.emit_u8(Self::modrm(1, reg_field, bl));
            if bl == 4 {
                self.emit_u8(0x24);
            }
            self.emit_u8(disp as i8 as u8);
        } else {
            self.emit_u8(Self::modrm(2, reg_field, bl));
            if bl == 4 {
                self.emit_u8(0x24);
            }
            self.emit_i32(disp);
        }
    }

    // --- Basic stack & control flow ---

    pub fn push(&mut self, r: Reg) {
        if r.hi() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x50 + r.low());
    }

    pub fn pop(&mut self, r: Reg) {
        if r.hi() {
            self.emit_u8(0x41);
        }
        self.emit_u8(0x58 + r.low());
    }

    pub fn ret(&mut self) {
        self.emit_u8(0xC3);
    }

    pub fn iretq(&mut self) {
        self.emit_u8(0x48);
        self.emit_u8(0xCF);
    }

    pub fn nop(&mut self) {
        self.emit_u8(0x90);
    }

    pub fn hlt(&mut self) {
        self.emit_u8(0xF4);
    }

    pub fn cli(&mut self) {
        self.emit_u8(0xFA);
    }

    pub fn sti(&mut self) {
        self.emit_u8(0xFB);
    }

    pub fn clc(&mut self) {
        self.emit_u8(0xF8);
    }

    pub fn stc(&mut self) {
        self.emit_u8(0xF9);
    }

    pub fn pause(&mut self) {
        self.emit_u8(0xF3);
        self.emit_u8(0x90);
    }

    pub fn mfence(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0xAE);
        self.emit_u8(0xF0);
    }

    pub fn lfence(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0xAE);
        self.emit_u8(0xE8);
    }

    pub fn sfence(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0xAE);
        self.emit_u8(0xF8);
    }

    pub fn ud2(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0x0B);
    }

    pub fn syscall(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0x05);
    }

    pub fn jmp_label(&mut self, l: Label) {
        self.emit_u8(0xE9);
        let off = self.pos();
        self.emit_i32(0);
        self.fixups.push((off, l));
    }

    pub fn jcc_label(&mut self, cc: Cond, l: Label) {
        self.emit_u8(0x0F);
        self.emit_u8(0x80 | (cc as u8));
        let off = self.pos();
        self.emit_i32(0);
        self.fixups.push((off, l));
    }

    pub fn jmp_sym(&mut self, sym: impl Into<String>) {
        self.emit_u8(0xE9);
        let off = self.pos();
        self.emit_i32(0);
        self.relocs.push(FnReloc {
            offset: off,
            kind: RelocKind::Rel32,
            target_sym: sym.into(),
            addend: 0,
        });
    }

    pub fn call_sym(&mut self, sym: impl Into<String>) {
        self.emit_u8(0xE8);
        let off = self.pos();
        self.emit_i32(0);
        self.relocs.push(FnReloc {
            offset: off,
            kind: RelocKind::Rel32,
            target_sym: sym.into(),
            addend: 0,
        });
    }

    pub fn call_label(&mut self, l: Label) {
        self.emit_u8(0xE8);
        let off = self.pos();
        self.emit_i32(0);
        self.fixups.push((off, l));
    }

    pub fn call_reg(&mut self, r: Reg) {
        self.emit_rex(false, false, false, r.hi(), false);
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(3, 2, r.low()));
    }

    fn push_sym_reloc(&mut self, kind: RelocKind, sym: impl Into<String>, addend: i64) {
        let off = self.pos();
        self.emit_i32(0);
        self.relocs.push(FnReloc {
            offset: off,
            kind,
            target_sym: sym.into(),
            addend,
        });
    }

    pub fn call_rip_sym(&mut self, sym: impl Into<String>, addend: i64) {
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(0, 2, 5));
        self.push_sym_reloc(RelocKind::Rel32, sym, addend);
    }

    pub fn jmp_rip_sym(&mut self, sym: impl Into<String>, addend: i64) {
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(0, 4, 5));
        self.push_sym_reloc(RelocKind::Rel32, sym, addend);
    }

    pub fn call_got_sym(&mut self, sym: impl Into<String>, addend: i64) {
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(0, 2, 5));
        self.push_sym_reloc(RelocKind::GotPcRel32, sym, addend);
    }

    pub fn jmp_got_sym(&mut self, sym: impl Into<String>, addend: i64) {
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(0, 4, 5));
        self.push_sym_reloc(RelocKind::GotPcRel32, sym, addend);
    }

    pub fn mov_got_load(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8B);
        self.emit_u8(Self::modrm(0, dst.low(), 5));
        self.push_sym_reloc(RelocKind::GotPcRel32, sym, addend);
    }

    pub fn call_plt_sym(&mut self, sym: impl Into<String>) {
        self.emit_u8(0xE8);
        self.push_sym_reloc(RelocKind::Plt32, sym, 0);
    }

    pub fn jmp_plt_sym(&mut self, sym: impl Into<String>) {
        self.emit_u8(0xE9);
        self.push_sym_reloc(RelocKind::Plt32, sym, 0);
    }

    pub fn jmp_reg(&mut self, r: Reg) {
        self.emit_rex(false, false, false, r.hi(), false);
        self.emit_u8(0xFF);
        self.emit_u8(Self::modrm(3, 4, r.low()));
    }

    pub fn lea_rip_sym(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8D);
        self.emit_u8(Self::modrm(0, dst.low(), 5));
        self.push_sym_reloc(RelocKind::Rel32, sym, addend);
    }

    pub fn lea_rip_bss_or_sym(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.lea_rip_sym(dst, sym, addend);
    }

    pub fn lea_rip_label(&mut self, dst: Reg, l: Label) {
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8D);
        self.emit_u8(Self::modrm(0, dst.low(), 5));
        let off = self.pos();
        self.emit_i32(0);
        self.fixups.push((off, l));
    }

    pub fn mov_rip_load(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8B);
        self.emit_u8(Self::modrm(0, dst.low(), 5));
        self.push_sym_reloc(RelocKind::Rel32, sym, addend);
    }

    pub fn mov_rip_store(&mut self, sym: impl Into<String>, addend: i64, src: Reg) {
        self.emit_rex(true, src.hi(), false, false, true);
        self.emit_u8(0x89);
        self.emit_u8(Self::modrm(0, src.low(), 5));
        self.push_sym_reloc(RelocKind::Rel32, sym, addend);
    }

    /// `mov dst, qword ptr fs:[disp32]` (uses SIB `0x25` so mod=0 rm=4 encodes `[disp32]`, not `[rip+disp32]`).
    pub fn load_fs_disp(&mut self, dst: Reg, disp32: i32) {
        self.emit_u8(0x64);
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8B);
        self.emit_u8(Self::modrm(0, dst.low(), 4));
        self.emit_u8(0x25);
        self.emit_i32(disp32);
    }

    /// `mov qword ptr fs:[disp32], src`.
    pub fn store_fs_disp(&mut self, disp32: i32, src: Reg) {
        self.emit_u8(0x64);
        self.emit_rex(true, src.hi(), false, false, true);
        self.emit_u8(0x89);
        self.emit_u8(Self::modrm(0, src.low(), 4));
        self.emit_u8(0x25);
        self.emit_i32(disp32);
    }

    /// `mov dst, qword ptr fs:[tpoff32(sym) + addend]` (ELF64 Variant II TLS access).
    pub fn load_fs_tpoff(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.emit_u8(0x64);
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x8B);
        self.emit_u8(Self::modrm(0, dst.low(), 4));
        self.emit_u8(0x25);
        self.push_sym_reloc(RelocKind::TpOff32, sym, addend);
    }

    /// `mov qword ptr fs:[tpoff32(sym) + addend], src` (ELF64 Variant II TLS access).
    pub fn store_fs_tpoff(&mut self, sym: impl Into<String>, addend: i64, src: Reg) {
        self.emit_u8(0x64);
        self.emit_rex(true, src.hi(), false, false, true);
        self.emit_u8(0x89);
        self.emit_u8(Self::modrm(0, src.low(), 4));
        self.emit_u8(0x25);
        self.push_sym_reloc(RelocKind::TpOff32, sym, addend);
    }

    /// Computes address of TLS variable `sym` into `dst` via `mov dst, fs:[0]; lea dst, [dst + tpoff32]`.
    pub fn lea_fs_tpoff(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.load_fs_disp(dst, 0);
        self.emit_rex(true, dst.hi(), false, dst.hi(), true);
        self.emit_u8(0x8D);
        self.emit_u8(Self::modrm(2, dst.low(), dst.low()));
        if dst.low() == 4 {
            self.emit_u8(0x24);
        }
        self.push_sym_reloc(RelocKind::TpOff32, sym, addend);
    }

    pub fn store_imm32_mem(&mut self, base: Reg, disp: i32, imm: i32) {
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0xC7);
        self.emit_mem(0, base, disp);
        self.emit_i32(imm);
    }

    pub fn store_imm64_sext_mem(&mut self, base: Reg, disp: i32, imm: i32) {
        self.emit_rex(true, false, false, base.hi(), true);
        self.emit_u8(0xC7);
        self.emit_mem(0, base, disp);
        self.emit_i32(imm);
    }

    pub fn cpuid(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0xA2);
    }

    pub fn rdtsc(&mut self) {
        self.emit_u8(0x0F);
        self.emit_u8(0x31);
    }

    pub fn stmxcsr(&mut self, base: Reg, disp: i32) {
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0xAE);
        self.emit_mem(3, base, disp);
    }

    pub fn ldmxcsr(&mut self, base: Reg, disp: i32) {
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0xAE);
        self.emit_mem(2, base, disp);
    }

    pub fn fnstcw(&mut self, base: Reg, disp: i32) {
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0xD9);
        self.emit_mem(7, base, disp);
    }

    pub fn fldcw(&mut self, base: Reg, disp: i32) {
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0xD9);
        self.emit_mem(5, base, disp);
    }

    pub fn mov_abs_sym(&mut self, dst: Reg, sym: impl Into<String>, addend: i64) {
        self.emit_rex(true, false, false, dst.hi(), true);
        self.emit_u8(0xB8 + dst.low());
        let off = self.pos();
        self.emit_u64(0);
        self.relocs.push(FnReloc {
            offset: off,
            kind: RelocKind::Abs64,
            target_sym: sym.into(),
            addend,
        });
    }

    // --- Moves & Loads / Stores ---

    pub fn mov_rr(&mut self, dst: Reg, src: Reg) {
        if dst == src {
            return;
        }
        self.emit_rex(true, src.hi(), false, dst.hi(), true);
        self.emit_u8(0x89);
        self.emit_u8(Self::modrm(3, src.low(), dst.low()));
    }

    pub fn mov_ri64(&mut self, dst: Reg, val: u64) {
        if val == 0 {
            self.xor_rr32(dst, dst);
        } else if val <= u32::MAX as u64 {
            self.emit_rex(false, false, false, dst.hi(), false);
            self.emit_u8(0xB8 + dst.low());
            self.emit_u32(val as u32);
        } else if (val as i64) >= i32::MIN as i64 && (val as i64) <= i32::MAX as i64 {
            self.emit_rex(true, false, false, dst.hi(), true);
            self.emit_u8(0xC7);
            self.emit_u8(Self::modrm(3, 0, dst.low()));
            self.emit_i32(val as i64 as i32);
        } else {
            self.emit_rex(true, false, false, dst.hi(), true);
            self.emit_u8(0xB8 + dst.low());
            self.emit_u64(val);
        }
    }

    pub fn lea_mem(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.emit_rex(true, dst.hi(), false, base.hi(), true);
        self.emit_u8(0x8D);
        self.emit_mem(dst.low(), base, disp);
    }

    /// Load `bytes` (1, 2, 4, or 8) from `[base + disp]` into `dst` (zero-extended for 1/2/4).
    pub fn load_mem(&mut self, dst: Reg, base: Reg, disp: i32, bytes: usize, signed: bool) {
        match (bytes, signed) {
            (8, _) => {
                self.emit_rex(true, dst.hi(), false, base.hi(), true);
                self.emit_u8(0x8B);
                self.emit_mem(dst.low(), base, disp);
            }
            (4, false) => {
                self.emit_rex(false, dst.hi(), false, base.hi(), false);
                self.emit_u8(0x8B);
                self.emit_mem(dst.low(), base, disp);
            }
            (4, true) => {
                self.emit_rex(true, dst.hi(), false, base.hi(), true);
                self.emit_u8(0x63); // movsxd
                self.emit_mem(dst.low(), base, disp);
            }
            (2, false) => {
                self.emit_rex(false, dst.hi(), false, base.hi(), false);
                self.emit_u8(0x0F);
                self.emit_u8(0xB7); // movzx r32, r/m16
                self.emit_mem(dst.low(), base, disp);
            }
            (2, true) => {
                self.emit_rex(true, dst.hi(), false, base.hi(), true);
                self.emit_u8(0x0F);
                self.emit_u8(0xBF); // movsx r64, r/m16
                self.emit_mem(dst.low(), base, disp);
            }
            (1, false) => {
                self.emit_rex(false, dst.hi(), false, base.hi(), false);
                self.emit_u8(0x0F);
                self.emit_u8(0xB6); // movzx r32, r/m8
                self.emit_mem(dst.low(), base, disp);
            }
            (1, true) => {
                self.emit_rex(true, dst.hi(), false, base.hi(), true);
                self.emit_u8(0x0F);
                self.emit_u8(0xBE); // movsx r64, r/m8
                self.emit_mem(dst.low(), base, disp);
            }
            _ => {
                self.emit_rex(true, dst.hi(), false, base.hi(), true);
                self.emit_u8(0x8B);
                self.emit_mem(dst.low(), base, disp);
            }
        }
    }

    /// Store `bytes` (1, 2, 4, or 8) from `src` to `[base + disp]`.
    pub fn store_mem(&mut self, base: Reg, disp: i32, src: Reg, bytes: usize) {
        match bytes {
            8 => {
                self.emit_rex(true, src.hi(), false, base.hi(), true);
                self.emit_u8(0x89);
                self.emit_mem(src.low(), base, disp);
            }
            4 => {
                self.emit_rex(false, src.hi(), false, base.hi(), false);
                self.emit_u8(0x89);
                self.emit_mem(src.low(), base, disp);
            }
            2 => {
                self.emit_u8(0x66);
                self.emit_rex(false, src.hi(), false, base.hi(), false);
                self.emit_u8(0x89);
                self.emit_mem(src.low(), base, disp);
            }
            1 => {
                let need_rex = src.id() >= 4 || base.hi();
                self.emit_rex(false, src.hi(), false, base.hi(), need_rex);
                self.emit_u8(0x88);
                self.emit_mem(src.low(), base, disp);
            }
            _ => {
                self.emit_rex(true, src.hi(), false, base.hi(), true);
                self.emit_u8(0x89);
                self.emit_mem(src.low(), base, disp);
            }
        }
    }

    /// Sign- or zero-extend `src` of width `bits` (8, 16, 32, 64) into 64-bit `dst`.
    pub fn extend_reg(&mut self, dst: Reg, src: Reg, bits: u8, signed: bool) {
        match (bits, signed) {
            (8, false) => {
                let need_rex = src.id() >= 4 || dst.hi();
                self.emit_rex(false, dst.hi(), false, src.hi(), need_rex);
                self.emit_u8(0x0F);
                self.emit_u8(0xB6);
                self.emit_u8(Self::modrm(3, dst.low(), src.low()));
            }
            (8, true) => {
                self.emit_rex(true, dst.hi(), false, src.hi(), true);
                self.emit_u8(0x0F);
                self.emit_u8(0xBE);
                self.emit_u8(Self::modrm(3, dst.low(), src.low()));
            }
            (16, false) => {
                self.emit_rex(false, dst.hi(), false, src.hi(), false);
                self.emit_u8(0x0F);
                self.emit_u8(0xB7);
                self.emit_u8(Self::modrm(3, dst.low(), src.low()));
            }
            (16, true) => {
                self.emit_rex(true, dst.hi(), false, src.hi(), true);
                self.emit_u8(0x0F);
                self.emit_u8(0xBF);
                self.emit_u8(Self::modrm(3, dst.low(), src.low()));
            }
            (32, false) => {
                // `mov dst32, src32` zero-extends to 64-bit.
                self.emit_rex(false, src.hi(), false, dst.hi(), false);
                self.emit_u8(0x89);
                self.emit_u8(Self::modrm(3, src.low(), dst.low()));
            }
            (32, true) => {
                self.emit_rex(true, dst.hi(), false, src.hi(), true);
                self.emit_u8(0x63);
                self.emit_u8(Self::modrm(3, dst.low(), src.low()));
            }
            _ => self.mov_rr(dst, src),
        }
    }

    // --- Integer ALU (64-bit) ---

    fn alu_rr(&mut self, op: u8, dst: Reg, src: Reg) {
        self.emit_rex(true, src.hi(), false, dst.hi(), true);
        self.emit_u8(op);
        self.emit_u8(Self::modrm(3, src.low(), dst.low()));
    }

    fn alu_ri32(&mut self, ext: u8, dst: Reg, imm: i32) {
        self.emit_rex(true, false, false, dst.hi(), true);
        if (-128..=127).contains(&imm) {
            self.emit_u8(0x83);
            self.emit_u8(Self::modrm(3, ext, dst.low()));
            self.emit_u8(imm as i8 as u8);
        } else {
            self.emit_u8(0x81);
            self.emit_u8(Self::modrm(3, ext, dst.low()));
            self.emit_i32(imm);
        }
    }

    pub fn add_rr(&mut self, dst: Reg, src: Reg) { self.alu_rr(0x01, dst, src); }
    pub fn sub_rr(&mut self, dst: Reg, src: Reg) { self.alu_rr(0x29, dst, src); }
    pub fn and_rr(&mut self, dst: Reg, src: Reg) { self.alu_rr(0x21, dst, src); }
    pub fn or_rr(&mut self, dst: Reg, src: Reg) { self.alu_rr(0x09, dst, src); }
    pub fn xor_rr(&mut self, dst: Reg, src: Reg) { self.alu_rr(0x31, dst, src); }
    pub fn cmp_rr(&mut self, lhs: Reg, rhs: Reg) { self.alu_rr(0x39, lhs, rhs); }

    pub fn xor_rr32(&mut self, dst: Reg, src: Reg) {
        self.emit_rex(false, src.hi(), false, dst.hi(), false);
        self.emit_u8(0x31);
        self.emit_u8(Self::modrm(3, src.low(), dst.low()));
    }

    pub fn add_ri32(&mut self, dst: Reg, imm: i32) {
        if imm != 0 {
            self.alu_ri32(0, dst, imm);
        }
    }
    pub fn sub_ri32(&mut self, dst: Reg, imm: i32) {
        if imm != 0 {
            self.alu_ri32(5, dst, imm);
        }
    }
    pub fn and_ri32(&mut self, dst: Reg, imm: i32) { self.alu_ri32(4, dst, imm); }
    pub fn or_ri32(&mut self, dst: Reg, imm: i32) { self.alu_ri32(1, dst, imm); }
    pub fn xor_ri32(&mut self, dst: Reg, imm: i32) { self.alu_ri32(6, dst, imm); }
    pub fn cmp_ri32(&mut self, dst: Reg, imm: i32) { self.alu_ri32(7, dst, imm); }

    pub fn test_rr(&mut self, a: Reg, b: Reg) {
        self.emit_rex(true, b.hi(), false, a.hi(), true);
        self.emit_u8(0x85);
        self.emit_u8(Self::modrm(3, b.low(), a.low()));
    }

    pub fn test_rr32(&mut self, a: Reg, b: Reg) {
        self.emit_rex(false, b.hi(), false, a.hi(), false);
        self.emit_u8(0x85);
        self.emit_u8(Self::modrm(3, b.low(), a.low()));
    }

    pub fn neg_r(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xF7);
        self.emit_u8(Self::modrm(3, 3, r.low()));
    }

    pub fn not_r(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xF7);
        self.emit_u8(Self::modrm(3, 2, r.low()));
    }

    pub fn imul_rr(&mut self, dst: Reg, src: Reg) {
        self.emit_rex(true, dst.hi(), false, src.hi(), true);
        self.emit_u8(0x0F);
        self.emit_u8(0xAF);
        self.emit_u8(Self::modrm(3, dst.low(), src.low()));
    }

    pub fn imul_rri32(&mut self, dst: Reg, src: Reg, imm: i32) {
        self.emit_rex(true, dst.hi(), false, src.hi(), true);
        self.emit_u8(0x69);
        self.emit_u8(Self::modrm(3, dst.low(), src.low()));
        self.emit_i32(imm);
    }

    /// Unsigned 64x64 -> 128-bit multiply `rdx:rax = rax * r` (sets CF/OF if rdx != 0).
    pub fn mul_r(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xF7);
        self.emit_u8(Self::modrm(3, 4, r.low()));
    }

    pub fn cqo(&mut self) {
        self.emit_u8(0x48);
        self.emit_u8(0x99);
    }

    pub fn idiv_r(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xF7);
        self.emit_u8(Self::modrm(3, 7, r.low()));
    }

    pub fn div_r(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xF7);
        self.emit_u8(Self::modrm(3, 6, r.low()));
    }

    pub fn shl_cl(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xD3);
        self.emit_u8(Self::modrm(3, 4, r.low()));
    }

    pub fn shr_cl(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xD3);
        self.emit_u8(Self::modrm(3, 5, r.low()));
    }

    pub fn sar_cl(&mut self, r: Reg) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xD3);
        self.emit_u8(Self::modrm(3, 7, r.low()));
    }

    pub fn shl_imm(&mut self, r: Reg, imm: u8) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xC1);
        self.emit_u8(Self::modrm(3, 4, r.low()));
        self.emit_u8(imm);
    }

    pub fn shr_imm(&mut self, r: Reg, imm: u8) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xC1);
        self.emit_u8(Self::modrm(3, 5, r.low()));
        self.emit_u8(imm);
    }

    pub fn sar_imm(&mut self, r: Reg, imm: u8) {
        self.emit_rex(true, false, false, r.hi(), true);
        self.emit_u8(0xC1);
        self.emit_u8(Self::modrm(3, 7, r.low()));
        self.emit_u8(imm);
    }

    pub fn setcc(&mut self, cc: Cond, dst: Reg) {
        let need_rex = dst.id() >= 4;
        self.emit_rex(false, false, false, dst.hi(), need_rex);
        self.emit_u8(0x0F);
        self.emit_u8(0x90 | (cc as u8));
        self.emit_u8(Self::modrm(3, 0, dst.low()));
        // Zero-extend low byte to 64-bit.
        self.extend_reg(dst, dst, 8, false);
    }

    pub fn cmovcc(&mut self, cc: Cond, dst: Reg, src: Reg) {
        self.emit_rex(true, dst.hi(), false, src.hi(), true);
        self.emit_u8(0x0F);
        self.emit_u8(0x40 | (cc as u8));
        self.emit_u8(Self::modrm(3, dst.low(), src.low()));
    }

    // --- Bulk memory ops (`rep movsb` / `rep stosb` / `rep stosq`) ---

    pub fn rep_movsb(&mut self) {
        self.emit_u8(0xF3);
        self.emit_u8(0xA4);
    }

    pub fn rep_stosb(&mut self) {
        self.emit_u8(0xF3);
        self.emit_u8(0xAA);
    }

    pub fn rep_stosq(&mut self) {
        self.emit_u8(0xF3);
        self.emit_u8(0x48);
        self.emit_u8(0xAB);
    }

    // --- Bit manipulation & byte-swap intrinsics ---

    fn emit_bit_cnt(&mut self, f3_prefix: bool, op2: u8, is_64: bool, dst: Reg, src: Reg) {
        if f3_prefix {
            self.emit_u8(0xF3);
        }
        self.emit_rex(is_64, dst.hi(), false, src.hi(), is_64);
        self.emit_u8(0x0F);
        self.emit_u8(op2);
        self.emit_u8(Self::modrm(3, dst.low(), src.low()));
    }

    pub fn popcnt(&mut self, is_64: bool, dst: Reg, src: Reg) {
        self.emit_bit_cnt(true, 0xB8, is_64, dst, src);
    }

    pub fn lzcnt(&mut self, is_64: bool, dst: Reg, src: Reg) {
        self.emit_bit_cnt(true, 0xBD, is_64, dst, src);
    }

    pub fn tzcnt(&mut self, is_64: bool, dst: Reg, src: Reg) {
        self.emit_bit_cnt(true, 0xBC, is_64, dst, src);
    }

    pub fn bsf(&mut self, is_64: bool, dst: Reg, src: Reg) {
        self.emit_bit_cnt(false, 0xBC, is_64, dst, src);
    }

    pub fn bsr(&mut self, is_64: bool, dst: Reg, src: Reg) {
        self.emit_bit_cnt(false, 0xBD, is_64, dst, src);
    }

    pub fn bswap(&mut self, bytes: usize, r: Reg) {
        match bytes {
            8 => {
                self.emit_rex(true, false, false, r.hi(), true);
                self.emit_u8(0x0F);
                self.emit_u8(0xC8 + r.low());
            }
            4 => {
                self.emit_rex(false, false, false, r.hi(), false);
                self.emit_u8(0x0F);
                self.emit_u8(0xC8 + r.low());
            }
            2 => {
                self.emit_u8(0x66);
                self.emit_rex(false, false, false, r.hi(), false);
                self.emit_u8(0xC1);
                self.emit_u8(Self::modrm(3, 0, r.low()));
                self.emit_u8(0x08);
            }
            _ => {}
        }
    }

    // --- Memory compare & atomic instructions ---

    pub fn cmp_mi32(&mut self, base: Reg, disp: i32, imm: i32, bytes: usize) {
        if bytes == 1 {
            self.emit_rex(false, false, false, base.hi(), false);
            self.emit_u8(0x80);
            self.emit_mem(7, base, disp);
            self.emit_u8(imm as i8 as u8);
            return;
        }
        if bytes == 2 {
            self.emit_u8(0x66);
        }
        self.emit_rex(bytes == 8, false, false, base.hi(), bytes == 8);
        if (-128..=127).contains(&imm) {
            self.emit_u8(0x83);
            self.emit_mem(7, base, disp);
            self.emit_u8(imm as i8 as u8);
        } else {
            self.emit_u8(0x81);
            self.emit_mem(7, base, disp);
            if bytes == 2 {
                self.code.extend_from_slice(&(imm as i16).to_le_bytes());
            } else {
                self.emit_i32(imm);
            }
        }
    }

    fn emit_rm_mem(&mut self, spec: (bool, Option<u8>, u8, u8), base: Reg, disp: i32, reg: Reg, bytes: usize) {
        let (lock, op_2b, op8, op_wide) = spec;
        if lock {
            self.emit_u8(0xF0);
        }
        match bytes {
            8 => {
                self.emit_rex(true, reg.hi(), false, base.hi(), true);
                if let Some(p) = op_2b { self.emit_u8(p); }
                self.emit_u8(op_wide);
                self.emit_mem(reg.low(), base, disp);
            }
            4 => {
                self.emit_rex(false, reg.hi(), false, base.hi(), false);
                if let Some(p) = op_2b { self.emit_u8(p); }
                self.emit_u8(op_wide);
                self.emit_mem(reg.low(), base, disp);
            }
            2 => {
                self.emit_u8(0x66);
                self.emit_rex(false, reg.hi(), false, base.hi(), false);
                if let Some(p) = op_2b { self.emit_u8(p); }
                self.emit_u8(op_wide);
                self.emit_mem(reg.low(), base, disp);
            }
            _ => {
                let need_rex = reg.id() >= 4 || base.hi();
                self.emit_rex(false, reg.hi(), false, base.hi(), need_rex);
                if let Some(p) = op_2b { self.emit_u8(p); }
                self.emit_u8(op8);
                self.emit_mem(reg.low(), base, disp);
            }
        }
    }

    /// `lock xadd [ptr + disp], val` (returns old value in `val`).
    pub fn lock_xadd_disp(&mut self, ptr: Reg, disp: i32, val: Reg, bytes: usize) {
        self.emit_rm_mem((true, Some(0x0F), 0xC0, 0xC1), ptr, disp, val, bytes);
    }

    /// `lock xadd [ptr], val` (returns old value in `val`).
    pub fn lock_xadd(&mut self, ptr: Reg, val: Reg, bytes: usize) {
        self.lock_xadd_disp(ptr, 0, val, bytes);
    }

    /// `lock cmpxchg [ptr + disp], desired` (compares `rax` with `[ptr + disp]`, sets ZF=1 on success).
    pub fn lock_cmpxchg_disp(&mut self, ptr: Reg, disp: i32, desired: Reg, bytes: usize) {
        self.emit_rm_mem((true, Some(0x0F), 0xB0, 0xB1), ptr, disp, desired, bytes);
    }

    /// `lock cmpxchg [ptr], desired` (compares `rax` with `[ptr]`, sets ZF=1 on success).
    pub fn lock_cmpxchg(&mut self, ptr: Reg, desired: Reg, bytes: usize) {
        self.lock_cmpxchg_disp(ptr, 0, desired, bytes);
    }

    /// `xchg [ptr + disp], val` (implicitly atomic on x86_64; returns old value in `val`).
    pub fn xchg_mem_disp(&mut self, ptr: Reg, disp: i32, val: Reg, bytes: usize) {
        self.emit_rm_mem((false, None, 0x86, 0x87), ptr, disp, val, bytes);
    }

    /// `xchg [ptr], val` (implicitly atomic on x86_64; returns old value in `val`).
    pub fn xchg_mem(&mut self, ptr: Reg, val: Reg, bytes: usize) {
        self.xchg_mem_disp(ptr, 0, val, bytes);
    }

    // --- SSE2 scalar float ops (`f32` / `f64`) ---

    pub fn movss_load(&mut self, dst: Xmm, base: Reg, disp: i32) {
        self.emit_u8(0xF3);
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x10);
        self.emit_mem(dst.id(), base, disp);
    }

    pub fn movss_store(&mut self, base: Reg, disp: i32, src: Xmm) {
        self.emit_u8(0xF3);
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x11);
        self.emit_mem(src.id(), base, disp);
    }

    pub fn movsd_load(&mut self, dst: Xmm, base: Reg, disp: i32) {
        self.emit_u8(0xF2);
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x10);
        self.emit_mem(dst.id(), base, disp);
    }

    pub fn movsd_store(&mut self, base: Reg, disp: i32, src: Xmm) {
        self.emit_u8(0xF2);
        self.emit_rex(false, false, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x11);
        self.emit_mem(src.id(), base, disp);
    }

    /// `movdqu [base + disp], xmmN` for any `N` in `0..16` (used to spill Win64 callee-saved
    /// `xmm6..xmm15`, which the `Xmm` enum used by codegen does not cover).
    pub fn movdqu_store_n(&mut self, base: Reg, disp: i32, xmm_n: u8) {
        debug_assert!(xmm_n < 16);
        self.emit_u8(0xF3);
        self.emit_rex(false, xmm_n >= 8, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x7F);
        self.emit_mem(xmm_n & 7, base, disp);
    }

    /// `movdqu xmmN, [base + disp]` for any `N` in `0..16`.
    pub fn movdqu_load_n(&mut self, xmm_n: u8, base: Reg, disp: i32) {
        debug_assert!(xmm_n < 16);
        self.emit_u8(0xF3);
        self.emit_rex(false, xmm_n >= 8, false, base.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x6F);
        self.emit_mem(xmm_n & 7, base, disp);
    }

    pub fn movd_xmm_r32(&mut self, dst: Xmm, src: Reg) {
        self.emit_u8(0x66);
        self.emit_rex(false, false, false, src.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x6E);
        self.emit_u8(Self::modrm(3, dst.id(), src.low()));
    }

    pub fn movd_r32_xmm(&mut self, dst: Reg, src: Xmm) {
        self.emit_u8(0x66);
        self.emit_rex(false, false, false, dst.hi(), false);
        self.emit_u8(0x0F);
        self.emit_u8(0x7E);
        self.emit_u8(Self::modrm(3, src.id(), dst.low()));
    }

    pub fn movq_xmm_r64(&mut self, dst: Xmm, src: Reg) {
        self.emit_u8(0x66);
        self.emit_rex(true, false, false, src.hi(), true);
        self.emit_u8(0x0F);
        self.emit_u8(0x6E);
        self.emit_u8(Self::modrm(3, dst.id(), src.low()));
    }

    pub fn movq_r64_xmm(&mut self, dst: Reg, src: Xmm) {
        self.emit_u8(0x66);
        self.emit_rex(true, false, false, dst.hi(), true);
        self.emit_u8(0x0F);
        self.emit_u8(0x7E);
        self.emit_u8(Self::modrm(3, src.id(), dst.low()));
    }

    pub fn sse_bin(&mut self, is_f64: bool, opcode: u8, dst: Xmm, src: Xmm) {
        self.emit_u8(if is_f64 { 0xF2 } else { 0xF3 });
        self.emit_u8(0x0F);
        self.emit_u8(opcode);
        self.emit_u8(Self::modrm(3, dst.id(), src.id()));
    }

    /// Scalar `sqrtss` / `sqrtsd` (`dst = sqrt(src)`).
    pub fn sqrt_f(&mut self, is_f64: bool, dst: Xmm, src: Xmm) {
        self.sse_bin(is_f64, 0x51, dst, src);
    }

    /// FMA3 `vfmadd213ss` / `vfmadd213sd`: `dst = (dst * src1) + src2`.
    pub fn vfmadd213(&mut self, is_f64: bool, dst: Xmm, src1: Xmm, src2: Xmm) {
        self.emit_u8(0xC4);
        self.emit_u8(0xE2);
        let vex2 = ((u8::from(is_f64)) << 7) | (((!src1.id()) & 0x0F) << 3) | 0x01;
        self.emit_u8(vex2);
        self.emit_u8(0xA9);
        self.emit_u8(Self::modrm(3, dst.id(), src2.id()));
    }

    /// Packed 128-bit unaligned float load: `movups xmm, [base + disp]` (`0F 10 /r`).
    pub fn movups_load(&mut self, dst: Xmm, base: Reg, disp: i32) {
        if base.hi() {
            self.emit_rex(false, false, false, true, false);
        }
        self.emit_u8(0x0F);
        self.emit_u8(0x10);
        self.emit_mem(dst.id(), base, disp);
    }

    /// Packed 128-bit unaligned float store: `movups [base + disp], xmm` (`0F 11 /r`).
    pub fn movups_store(&mut self, base: Reg, disp: i32, src: Xmm) {
        if base.hi() {
            self.emit_rex(false, false, false, true, false);
        }
        self.emit_u8(0x0F);
        self.emit_u8(0x11);
        self.emit_mem(src.id(), base, disp);
    }

    /// Packed 128-bit `f32x4` arithmetic: `addps (0x58)`, `subps (0x5C)`, `mulps (0x59)`, `divps (0x5E)`, `minps (0x5D)`, `maxps (0x5F)`.
    pub fn sse_ps_bin(&mut self, opcode: u8, dst: Xmm, src: Xmm) {
        self.emit_u8(0x0F);
        self.emit_u8(opcode);
        self.emit_u8(Self::modrm(3, dst.id(), src.id()));
    }

    /// Packed 128-bit `66 0F <op>` SSE2 ops (`addpd/subpd/mulpd/divpd/minpd/maxpd`,
    /// `paddb/psubb/paddd/psubd/pand/por/pxor/pcmpeqb/pcmpeqd/punpcklqdq`).
    pub fn sse_66_bin(&mut self, opcode: u8, dst: Xmm, src: Xmm) {
        self.emit_u8(0x66);
        self.emit_u8(0x0F);
        self.emit_u8(opcode);
        self.emit_u8(Self::modrm(3, dst.id(), src.id()));
    }

    /// `psllw xmm, imm8` (`66 0F 71 /6 ib`).
    pub fn psllw_imm(&mut self, dst: Xmm, imm: u8) {
        self.emit_u8(0x66);
        self.emit_u8(0x0F);
        self.emit_u8(0x71);
        self.emit_u8(Self::modrm(3, 6, dst.id()));
        self.emit_u8(imm);
    }

    /// `pmovmskb r32, xmm` (`66 [REX.R] 0F D7 /r`, zero-extends to r64).
    pub fn pmovmskb(&mut self, dst: Reg, src: Xmm) {
        self.emit_u8(0x66);
        if dst.hi() {
            self.emit_rex(false, true, false, false, false);
        }
        self.emit_u8(0x0F);
        self.emit_u8(0xD7);
        self.emit_u8(Self::modrm(3, dst.low(), src.id()));
    }

    pub fn ucomi(&mut self, is_f64: bool, a: Xmm, b: Xmm) {
        if is_f64 {
            self.emit_u8(0x66);
        }
        self.emit_u8(0x0F);
        self.emit_u8(0x2E);
        self.emit_u8(Self::modrm(3, a.id(), b.id()));
    }

    pub fn cvtsi2f(&mut self, is_f64: bool, dst: Xmm, src: Reg) {
        self.emit_u8(if is_f64 { 0xF2 } else { 0xF3 });
        self.emit_rex(true, false, false, src.hi(), true);
        self.emit_u8(0x0F);
        self.emit_u8(0x2A);
        self.emit_u8(Self::modrm(3, dst.id(), src.low()));
    }

    pub fn cvttf2si(&mut self, is_f64: bool, dst: Reg, src: Xmm) {
        self.emit_u8(if is_f64 { 0xF2 } else { 0xF3 });
        self.emit_rex(true, dst.hi(), false, false, true);
        self.emit_u8(0x0F);
        self.emit_u8(0x2C);
        self.emit_u8(Self::modrm(3, dst.low(), src.id()));
    }

    pub fn cvtss2sd(&mut self, dst: Xmm, src: Xmm) {
        self.emit_u8(0xF3);
        self.emit_u8(0x0F);
        self.emit_u8(0x5A);
        self.emit_u8(Self::modrm(3, dst.id(), src.id()));
    }

    pub fn cvtsd2ss(&mut self, dst: Xmm, src: Xmm) {
        self.emit_u8(0xF2);
        self.emit_u8(0x0F);
        self.emit_u8(0x5A);
        self.emit_u8(Self::modrm(3, dst.id(), src.id()));
    }
}

/// Built-in x86_64 mini-assembler for `.hal.eon` `asm("...")` templates.
pub fn encode_inline_asm(buf: &mut X64Buf, template: &str) -> Result<(), String> {
    for raw_line in template.split([';', '\n']) {
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let mnem = parts.next().unwrap_or("").to_ascii_lowercase();
        let ops = parts.next().unwrap_or("").trim();
        match mnem.as_str() {
            "nop" => buf.nop(),
            "hlt" => buf.hlt(),
            "cli" => buf.cli(),
            "sti" => buf.sti(),
            "pause" => buf.pause(),
            "mfence" => buf.mfence(),
            "lfence" => buf.lfence(),
            "sfence" => buf.sfence(),
            "ret" => buf.ret(),
            "iretq" => buf.iretq(),
            "syscall" => buf.syscall(),
            "ud2" => buf.ud2(),
            "clc" => buf.clc(),
            "stc" => buf.stc(),
            "cpuid" => buf.cpuid(),
            "rdtsc" => buf.rdtsc(),
            "rdmsr" => {
                buf.emit_u8(0x0F);
                buf.emit_u8(0x32);
            }
            "wrmsr" => {
                buf.emit_u8(0x0F);
                buf.emit_u8(0x30);
            }
            "in" => {
                // `in al, dx` -> 0xEC
                buf.emit_u8(0xEC);
            }
            "out" => {
                // `out dx, al` -> 0xEE
                buf.emit_u8(0xEE);
            }
            "int" => {
                let vec = parse_asm_imm(ops).unwrap_or(3) as u8;
                buf.emit_u8(0xCD);
                buf.emit_u8(vec);
            }
            "push" => {
                if let Some(r) = parse_reg64(ops) {
                    buf.push(r);
                }
            }
            "pop" => {
                if let Some(r) = parse_reg64(ops) {
                    buf.pop(r);
                }
            }
            "mov" => {
                let mut sp = ops.split(',');
                let dst_s = sp.next().unwrap_or("").trim();
                let src_s = sp.next().unwrap_or("").trim();
                if let Some(dst) = parse_reg64(dst_s) {
                    if let Some(src) = parse_reg64(src_s) {
                        buf.mov_rr(dst, src);
                    } else if let Some(imm) = parse_asm_imm(src_s) {
                        buf.mov_ri64(dst, imm);
                    }
                }
            }
            "xor" => {
                let mut sp = ops.split(',');
                let dst_s = sp.next().unwrap_or("").trim();
                let src_s = sp.next().unwrap_or("").trim();
                if let (Some(dst), Some(src)) = (parse_reg64(dst_s), parse_reg64(src_s)) {
                    buf.xor_rr(dst, src);
                }
            }
            _ => {
                return Err(format!("x86_64 mini-assembler: unsupported mnemonic `{mnem}` in `{line}`"));
            }
        }
    }
    Ok(())
}

fn parse_reg64(s: &str) -> Option<Reg> {
    match s.trim().trim_start_matches('%').to_ascii_lowercase().as_str() {
        "rax" | "eax" => Some(Reg::Rax),
        "rcx" | "ecx" => Some(Reg::Rcx),
        "rdx" | "edx" => Some(Reg::Rdx),
        "rbx" | "ebx" => Some(Reg::Rbx),
        "rsp" | "esp" => Some(Reg::Rsp),
        "rbp" | "ebp" => Some(Reg::Rbp),
        "rsi" | "esi" => Some(Reg::Rsi),
        "rdi" | "edi" => Some(Reg::Rdi),
        "r8" => Some(Reg::R8),
        "r9" => Some(Reg::R9),
        "r10" => Some(Reg::R10),
        "r11" => Some(Reg::R11),
        "r12" => Some(Reg::R12),
        "r13" => Some(Reg::R13),
        "r14" => Some(Reg::R14),
        "r15" => Some(Reg::R15),
        _ => None,
    }
}

fn parse_asm_imm(s: &str) -> Option<u64> {
    let t = s.trim().trim_start_matches('$');
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

/// Emit a Win64-to-EonCall ABI adapter trampoline for a target taking `n_int_args` (`0..=8`)
/// integer/pointer arguments and returning a scalar in `rax` (no `! E`, no floats).
///
/// Register-convention gaps bridged here:
/// * Args: Win64 `rcx, rdx, r8, r9, [rsp+40..]` -> EonCall `rdi, rsi, rdx, rcx, r8, r9, r10, r11`.
/// * Win64 treats `rdi`, `rsi`, `r15` and `xmm6..xmm15` as callee-saved, but EonCall does not
///   (`rdi`/`rsi` are argument registers, `r15` carries the error code, all XMM are volatile),
///   so the trampoline saves and restores them. `rbx`, `rbp`, `r12..r14` are callee-saved in both.
/// * Stack: `rsp` is 16-byte aligned before the inner `call` (EonCall needs no shadow space).
///
/// Frame (`rbp`-relative): `[rbp+16..48)` caller shadow space, `[rbp+48 + 8*k]` Win64 stack arg
/// `4 + k`; below `rbp`: saved `rdi, rsi, r15`, then 8 B padding + 160 B of `xmm6..xmm15`.
pub fn emit_win64_to_eon_trampoline(
    target_sym: &str,
    n_int_args: usize,
) -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    if n_int_args > 8 {
        return Err(format!(
            "Win64 trampoline for `{target_sym}`: {n_int_args} args exceed the 8 EonCall argument registers"
        ));
    }
    const XMM_SAVE_BYTES: i32 = 10 * 16;
    const PAD: i32 = 8; // 3 pushes after `push rbp` leave rsp = 8 (mod 16)
    let mut b = X64Buf::new();
    b.push(Reg::Rbp);
    b.mov_rr(Reg::Rbp, Reg::Rsp);
    b.push(Reg::Rdi);
    b.push(Reg::Rsi);
    b.push(Reg::R15);
    b.sub_ri32(Reg::Rsp, XMM_SAVE_BYTES + PAD);
    for i in 0..10u8 {
        b.movdqu_store_n(Reg::Rsp, i32::from(i) * 16, 6 + i);
    }
    // Order matters: each source is read before it is overwritten.
    b.mov_rr(Reg::Rdi, Reg::Rcx);
    b.mov_rr(Reg::Rsi, Reg::Rdx);
    b.mov_rr(Reg::Rdx, Reg::R8);
    b.mov_rr(Reg::Rcx, Reg::R9);
    let stack_arg_regs = [Reg::R8, Reg::R9, Reg::R10, Reg::R11];
    for (k, &r) in stack_arg_regs.iter().enumerate().take(n_int_args.saturating_sub(4)) {
        b.load_mem(r, Reg::Rbp, 48 + 8 * k as i32, 8, false);
    }
    b.call_sym(target_sym);
    for i in 0..10u8 {
        b.movdqu_load_n(6 + i, Reg::Rsp, i32::from(i) * 16);
    }
    b.add_ri32(Reg::Rsp, XMM_SAVE_BYTES + PAD);
    b.pop(Reg::R15);
    b.pop(Reg::Rsi);
    b.pop(Reg::Rdi);
    b.pop(Reg::Rbp);
    b.ret();
    b.finish()
}

// ============================================================================
// Standalone Zero-Libc Linux x86_64 Runtime Helpers (emitted ON-DEMAND only!)
// ============================================================================

/// `(symbol_name, size_bytes, align_bytes)` for every `.bss` reservation used by
/// the zero-C Linux x86_64 runtime helpers.
pub const EON_RT_BSS_SYMBOLS: &[(&str, usize, usize)] = &[
    ("__eon_vdso_clock_gettime", 8, 8),
    ("__eon_envp", 8, 8),
    ("__eon_alloc_tls_fallback", 128, 16),
    ("__eon_main_tcb", 128, 64),
    ("__eon_med_span_lock", 8, 8),
    ("__eon_med_chunks", 2304, 8),
    ("__eon_san_lock", 8, 8),
    ("__eon_san_quarantine", 1032, 8),
    ("__eon_san_altstack", 16384, 16),
    ("__eon_sched_state", 4096, 64),
    ("__eon_cpu_affinity_mask", 128, 8),
];

/// Returns the `.bss` layout descriptors needed by the runtime helper blobs.
pub fn rt_bss_layout() -> &'static [(&'static str, usize, usize)] {
    EON_RT_BSS_SYMBOLS
}

/// Transitive helper symbol dependencies so Wave 2 can tree-shake cleanly.
pub fn rt_helper_deps(name: &str) -> &'static [&'static str] {
    match name {
        "eon_rt_print_bytes"
        | "eon_rt_eprint_str"
        | "eon_rt_eprint_bytes"
        | "eon_rt_eprint_nl" => &["eon_rt_write_fd"],
        "eon_par_for" | "eon_sched_run_dag" => &["eon_sched_init"],
        "eon_run_fibers_iouring" => &[
            "eon_iouring_create",
            "eon_iouring_destroy",
            "eon_fiber_switch",
        ],
        _ => &[],
    }
}

/// Assembled machine-code blob and its relocations.
pub type RtBlobResult = Result<(Vec<u8>, Vec<FnReloc>), String>;
/// Named assembled runtime helper `(symbol, code, relocs)`.
pub type RtNamedBlob = (&'static str, Vec<u8>, Vec<FnReloc>);

/// Emit any standalone Linux runtime helper by its symbol name.
pub fn emit_rt_helper_by_name(name: &str) -> Option<RtBlobResult> {
    Some(match name {
        "eon_alloc_raw" => emit_rt_alloc_raw(),
        "eon_free_raw" => emit_rt_free_raw(),
        "eon_alloc_raw_san" => emit_rt_alloc_raw_san(),
        "eon_free_raw_san" => emit_rt_free_raw_san(),
        "eon_arena_alloc_aligned" => emit_rt_arena_alloc(),
        "eon_rt_arena_reset_san" => emit_rt_arena_reset_san(),
        "eon_rt_install_san_handler" => emit_rt_install_san_handler(),
        "eon_rt_tls_init" => emit_rt_tls_init(),
        "eon_rt_parse_vdso" => emit_rt_parse_vdso(),
        "eon_rt_now_ns" => emit_rt_now_ns(),
        "eon_rt_openat2_beneath" => emit_rt_openat2_beneath(),
        "eon_rt_read_fd" => emit_rt_read_fd(),
        "eon_rt_write_fd" => emit_rt_write_fd(),
        "eon_rt_close_fd" => emit_rt_close_fd(),
        "eon_rt_print_str" => emit_rt_print_str(),
        "eon_rt_print_bytes" => emit_rt_print_bytes(),
        "eon_rt_print_u64" => emit_rt_print_u64(),
        "eon_rt_print_i64" => emit_rt_print_i64(),
        "eon_rt_print_f32" => emit_rt_print_f32(),
        "eon_rt_print_nl" => emit_rt_print_nl(),
        "eon_rt_eprint_str" => emit_rt_eprint_str(),
        "eon_rt_eprint_bytes" => emit_rt_eprint_bytes(),
        "eon_rt_eprint_nl" => emit_rt_eprint_nl(),
        "eon_rt_validate_utf8" => emit_rt_validate_utf8(),
        "eon_rt_is_dir" => emit_rt_is_dir(),
        "eon_rt_mkdir_all" => emit_rt_mkdir_all(),
        "eon_rt_spawn_wait" => emit_rt_spawn_wait(),
        "eon_f16_to_f32" => emit_rt_f16_to_f32(),
        "eon_f32_to_f16" => emit_rt_f32_to_f16(),
        "eon_sched_init" => emit_rt_sched_init(),
        "eon_sched_shutdown" => emit_rt_sched_shutdown(),
        "eon_par_for" => emit_rt_par_for(),
        "eon_sched_run_dag" => emit_rt_sched_run_dag(),
        "eon_fiber_switch" => emit_rt_fiber_switch(),
        "eon_iouring_create" => emit_rt_iouring_create(),
        "eon_iouring_destroy" => emit_rt_iouring_destroy(),
        "eon_run_fibers_iouring" => emit_rt_run_fibers_iouring(),
        _ => return None,
    })
}

const SLAB_MAGIC: u64 = 0xE04E_534C_4142_3030; // "EONSLAB0"
const MED_CHUNK_BYTES: u64 = 2_097_152; // 2 MiB (512 * 4096)
const MED_CHUNK_STRIDE: i32 = 72; // 8B base_ptr + 64B bitmap
const MED_CHUNK_COUNT: u64 = 32;
const SAN_QUARANTINE_SLOTS: u64 = 64;
const SAN_POISON_BYTE: u64 = 0xDE;

// __eon_sched_state field offsets (4096 B in .bss)
const SCHED_OFF_INIT: i32 = 0; // u64: 0=uninit, 1=initing, 2=ready
const SCHED_OFF_NUM_WORKERS: i32 = 8;
const SCHED_OFF_SHUTDOWN: i32 = 16;
const SCHED_OFF_WORK_GEN: i32 = 24;
const SCHED_OFF_TASK_KIND: i32 = 32; // 1=par_for, 2=dag
const SCHED_OFF_TASK_CTX: i32 = 40;
const SCHED_OFF_TASK_FN: i32 = 48;
const SCHED_OFF_TASK_TOTAL: i32 = 56;
const SCHED_OFF_TASK_CHUNK: i32 = 64;
const SCHED_OFF_TASK_NEXT: i32 = 72;
const SCHED_OFF_TASK_DONE: i32 = 80;
const SCHED_OFF_ACTIVE_WORKERS: i32 = 88;
const SCHED_OFF_DAG_NODES: i32 = 96;
const SCHED_OFF_DAG_COUNT: i32 = 104;
const SCHED_OFF_CALL_LOCK: i32 = 112;
const SCHED_OFF_IN_TASK: i32 = 120;
const SCHED_OFF_TASK_CLOSED: i32 = 128;
const SCHED_OFF_DAG_RT: i32 = 136;
const SCHED_OFF_WORKER_TCBS: i32 = 192; // 8 * 128 = 1024 B (192..1216)
const SCHED_OFF_ONLINE_CPUS: i32 = 1216; // 8 * 8 = 64 B (1216..1280)
const SCHED_OFF_DAG_RT_INLINE: i32 = 1280; // 176 * 16 = 2816 B (1280..4096)
const SCHED_MAX_WORKERS: u64 = 8;
const SCHED_MAX_INLINE_DAG: u64 = 176;
const WORKER_STACK_BYTES: u64 = 1_048_576;
const CLONE_WORKER_FLAGS: u64 = 0x000D_0F00; // VM|FS|FILES|SIGHAND|THREAD|SYSVSEM|SETTLS

fn emit_spin_lock(b: &mut X64Buf, lock_sym: &str) {
    let l_acq = b.new_label();
    let l_spin = b.new_label();
    b.lea_rip_sym(Reg::R8, lock_sym, 0);
    b.bind_label(l_acq);
    b.mov_ri64(Reg::Rax, 1);
    b.xchg_mem(Reg::R8, Reg::Rax, 8);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_spin);
    let l_wait = b.new_label();
    b.bind_label(l_wait);
    b.pause();
    b.cmp_mi32(Reg::R8, 0, 0, 8);
    b.jcc_label(Cond::Ne, l_wait);
    b.jmp_label(l_acq);
    b.bind_label(l_spin);
}

fn emit_spin_unlock(b: &mut X64Buf, lock_sym: &str) {
    b.lea_rip_sym(Reg::R8, lock_sym, 0);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.xchg_mem(Reg::R8, Reg::Rax, 8);
}

/// Ensures main thread's `FS.base` is initialized (lazy one-time fallback if `_start`
/// did not already call `eon_rt_tls_init`) and loads `tcb` pointer into `dst`.
/// Clobbers `Rax, Rcx, Rdx, Rsi, Rdi, R11` ONLY on the cold first-time path; on hot
/// path clobbers `dst` and `Rcx` (only if `dst != Rcx`) and preserves `Rbx, R8..R15`.
fn emit_load_tcb(b: &mut X64Buf, dst: Reg) {
    let l_fs_ok = b.new_label();
    let l_have_tcb = b.new_label();
    b.lea_rip_sym(Reg::Rax, "__eon_main_tcb", 0);
    b.cmp_mi32(Reg::Rax, 0, 0, 8);
    b.jcc_label(Cond::Ne, l_fs_ok);
    b.store_mem(Reg::Rax, 0, Reg::Rax, 8);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.mov_ri64(Reg::Rdi, 0x1002); // ARCH_SET_FS
    b.mov_ri64(Reg::Rax, 158);    // SYS_arch_prctl
    b.syscall();
    b.bind_label(l_fs_ok);
    b.load_fs_disp(dst, 0);
    b.test_rr(dst, dst);
    b.jcc_label(Cond::Ne, l_have_tcb);
    b.lea_rip_sym(dst, "__eon_alloc_tls_fallback", 0);
    b.bind_label(l_have_tcb);
}

/// Charge `size_reg` (`R13`, >= 1) across `cap_reg` (`R12`) -> parent chain.
/// On budget failure sets `R15` = failing node and jumps to `l_fail`.
/// Preserves `R12, R13, R14`; clobbers `Rax, Rbx, Rcx, Rdx, R8, R9`.
fn emit_alloc_cap_charge(b: &mut X64Buf, l_fail: Label) {
    let l_walk = b.new_label();
    let l_done = b.new_label();
    let l_nolimit = b.new_label();
    let l_cas = b.new_label();
    let l_charged = b.new_label();
    let l_peak = b.new_label();
    let l_next = b.new_label();
    let l_budget_fail = b.new_label();

    b.mov_rr(Reg::Rbx, Reg::R12);
    b.xor_rr32(Reg::R15, Reg::R15);

    b.bind_label(l_walk);
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_done);

    b.load_mem(Reg::R8, Reg::Rbx, 24, 8, false); // budget_limit
    b.test_rr(Reg::R8, Reg::R8);
    b.jcc_label(Cond::E, l_nolimit);

    // Budgeted CAS loop on [rbx + 8] (used_bytes)
    b.load_mem(Reg::Rax, Reg::Rbx, 8, 8, false);
    b.bind_label(l_cas);
    b.mov_rr(Reg::Rcx, Reg::Rax);
    b.add_rr(Reg::Rcx, Reg::R13);
    b.jcc_label(Cond::B, l_budget_fail); // CF=1 unsigned overflow
    b.cmp_rr(Reg::Rcx, Reg::R8);
    b.jcc_label(Cond::A, l_budget_fail);
    b.lock_cmpxchg_disp(Reg::Rbx, 8, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_cas);
    b.jmp_label(l_charged);

    // Unlimited path: lock xadd
    b.bind_label(l_nolimit);
    b.mov_rr(Reg::Rax, Reg::R13);
    b.lock_xadd_disp(Reg::Rbx, 8, Reg::Rax, 8);
    b.mov_rr(Reg::Rcx, Reg::Rax);
    b.add_rr(Reg::Rcx, Reg::R13);

    b.bind_label(l_charged);
    // live_allocs += 1
    b.mov_ri64(Reg::R9, 1);
    b.lock_xadd_disp(Reg::Rbx, 0, Reg::R9, 8);
    // peak_bytes = max(peak_bytes, rcx)
    b.load_mem(Reg::Rax, Reg::Rbx, 16, 8, false);
    b.bind_label(l_peak);
    b.cmp_rr(Reg::Rcx, Reg::Rax);
    b.jcc_label(Cond::Be, l_next);
    b.lock_cmpxchg_disp(Reg::Rbx, 16, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_peak);

    b.bind_label(l_next);
    b.load_mem(Reg::Rbx, Reg::Rbx, 32, 8, false); // parent_ptr
    b.jmp_label(l_walk);

    b.bind_label(l_budget_fail);
    b.mov_rr(Reg::R15, Reg::Rbx);
    b.jmp_label(l_fail);

    b.bind_label(l_done);
}

/// Roll back `AllocCap` chain starting at `R12` stopping before `R15` (or `0`),
/// subtracting `1` live alloc and `R13` used bytes.
fn emit_alloc_cap_rollback(b: &mut X64Buf) {
    let l_loop = b.new_label();
    let l_done = b.new_label();
    b.mov_rr(Reg::Rbx, Reg::R12);
    b.bind_label(l_loop);
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_done);
    b.cmp_rr(Reg::Rbx, Reg::R15);
    b.jcc_label(Cond::E, l_done);
    b.mov_ri64(Reg::Rax, (-1i64) as u64);
    b.lock_xadd_disp(Reg::Rbx, 0, Reg::Rax, 8);
    b.mov_rr(Reg::Rax, Reg::R13);
    b.neg_r(Reg::Rax);
    b.lock_xadd_disp(Reg::Rbx, 8, Reg::Rax, 8);
    b.load_mem(Reg::Rbx, Reg::Rbx, 32, 8, false);
    b.jmp_label(l_loop);
    b.bind_label(l_done);
}

/// Decrement `live_allocs` and `used_bytes` (`size` in `R14`) up `R12` -> parent chain.
fn emit_alloc_cap_release(b: &mut X64Buf) {
    let l_walk = b.new_label();
    let l_done = b.new_label();
    let l_live = b.new_label();
    let l_live_done = b.new_label();
    let l_used = b.new_label();
    let l_used_zero = b.new_label();
    let l_used_cas = b.new_label();

    b.mov_rr(Reg::Rbx, Reg::R12);
    b.bind_label(l_walk);
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_done);

    // live_allocs = old > 0 ? old - 1 : 0
    b.load_mem(Reg::Rax, Reg::Rbx, 0, 8, false);
    b.bind_label(l_live);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_live_done);
    b.lea_mem(Reg::Rcx, Reg::Rax, -1);
    b.lock_cmpxchg_disp(Reg::Rbx, 0, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_live);
    b.bind_label(l_live_done);

    // used_bytes = old >= r14 ? old - r14 : 0
    b.load_mem(Reg::Rax, Reg::Rbx, 8, 8, false);
    b.bind_label(l_used);
    b.cmp_rr(Reg::Rax, Reg::R14);
    b.jcc_label(Cond::B, l_used_zero);
    b.mov_rr(Reg::Rcx, Reg::Rax);
    b.sub_rr(Reg::Rcx, Reg::R14);
    b.jmp_label(l_used_cas);
    b.bind_label(l_used_zero);
    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    b.bind_label(l_used_cas);
    b.lock_cmpxchg_disp(Reg::Rbx, 8, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_used);

    b.load_mem(Reg::Rbx, Reg::Rbx, 32, 8, false);
    b.jmp_label(l_walk);
    b.bind_label(l_done);
}

/// Emit bitmap bit-run set (`set=true`) or clear (`set=false`) in `[chunk + 8 .. chunk + 72]`.
/// Inputs: `R9 = chunk_ptr`, `Rdi = start_bit`, `Rbx = npages` (1..=64). Clobbers `Rax, Rcx, Rdx, Rsi, Rdi, R8, R10`.
fn emit_med_bitmap_update(b: &mut X64Buf, set: bool) {
    let l_loop = b.new_label();
    b.lea_mem(Reg::Rsi, Reg::R9, 8);
    b.mov_rr(Reg::R8, Reg::Rbx);
    b.bind_label(l_loop);
    b.mov_rr(Reg::Rax, Reg::Rdi);
    b.shr_imm(Reg::Rax, 6);
    b.shl_imm(Reg::Rax, 3);
    b.add_rr(Reg::Rax, Reg::Rsi);
    b.mov_rr(Reg::Rcx, Reg::Rdi);
    b.and_ri32(Reg::Rcx, 63);
    b.mov_ri64(Reg::Rdx, 1);
    b.shl_cl(Reg::Rdx);
    b.load_mem(Reg::R10, Reg::Rax, 0, 8, false);
    if set {
        b.or_rr(Reg::R10, Reg::Rdx);
    } else {
        b.not_r(Reg::Rdx);
        b.and_rr(Reg::R10, Reg::Rdx);
    }
    b.store_mem(Reg::Rax, 0, Reg::R10, 8);
    b.add_ri32(Reg::Rdi, 1);
    b.sub_ri32(Reg::R8, 1);
    b.jcc_label(Cond::Ne, l_loop);
}

/// `void *eon_alloc_raw(EonAllocCap *cap [rdi], size_t size [rsi], size_t align [rdx])`
/// Fast TLS slab allocator (`16..=2048 B`, 8 power-of-2 classes, zero per-object header),
/// spin-locked 2 MiB medium page-bitmap cache (`2049..=262144 B`), and large `SYS_mmap`,
/// with hierarchical `AllocCap` budget enforcement and zeroed return guarantees.
pub fn emit_rt_alloc_raw() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::R12, Reg::Rdi);
    // r13 = max(size, 1), r14 = max(align, 1)
    b.mov_ri64(Reg::Rax, 1);
    b.mov_rr(Reg::R13, Reg::Rsi);
    b.test_rr(Reg::R13, Reg::R13);
    b.cmovcc(Cond::E, Reg::R13, Reg::Rax);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.test_rr(Reg::R14, Reg::R14);
    b.cmovcc(Cond::E, Reg::R14, Reg::Rax);

    let l_rollback_fail = b.new_label();
    emit_alloc_cap_charge(&mut b, l_rollback_fail);

    let l_medium = b.new_label();
    let l_large = b.new_label();
    let l_done = b.new_label();

    // Small TLS slab fast path: eff_size <= 2048 && eff_align <= 2048
    b.cmp_ri32(Reg::R13, 2048);
    b.jcc_label(Cond::A, l_medium);
    b.cmp_ri32(Reg::R14, 2048);
    b.jcc_label(Cond::A, l_medium);

    // eff = max(r13, r14) -> slab class 0..=7
    b.mov_rr(Reg::R8, Reg::R13);
    b.cmp_rr(Reg::R8, Reg::R14);
    b.cmovcc(Cond::B, Reg::R8, Reg::R14);
    b.xor_rr32(Reg::Rbx, Reg::Rbx);
    let mut thresh = 16i32;
    let l_cls_done = b.new_label();
    for cls in 0..7u64 {
        b.cmp_ri32(Reg::R8, thresh);
        b.jcc_label(Cond::Be, l_cls_done);
        b.mov_ri64(Reg::Rbx, cls + 1);
        thresh *= 2;
    }
    b.bind_label(l_cls_done);

    // r15 = csz = 16 << class
    b.mov_ri64(Reg::R15, 16);
    b.mov_rr(Reg::Rcx, Reg::Rbx);
    b.shl_cl(Reg::R15);

    emit_load_tcb(&mut b, Reg::R9);
    // r8 = &tcb.freelists[class] = r9 + 16 + rbx*8
    b.lea_mem(Reg::R8, Reg::R9, 16);
    b.mov_rr(Reg::R10, Reg::Rbx);
    b.shl_imm(Reg::R10, 3);
    b.add_rr(Reg::R8, Reg::R10);

    b.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
    let l_slab_refill = b.new_label();
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_slab_refill);

    // Pop head and zero csz bytes (SlotPool/Arena/Buf expect zero-initialized memory!)
    b.load_mem(Reg::Rcx, Reg::Rax, 0, 8, false);
    b.store_mem(Reg::R8, 0, Reg::Rcx, 8);
    b.mov_rr(Reg::R9, Reg::Rax);
    b.mov_rr(Reg::Rdi, Reg::Rax);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_rr(Reg::Rcx, Reg::R15);
    b.shr_imm(Reg::Rcx, 3);
    b.rep_stosq();
    b.mov_rr(Reg::Rax, Reg::R9);
    b.jmp_label(l_done);

    // Refill: SYS_mmap 64 KiB (16 * 4 KiB pages)
    b.bind_label(l_slab_refill);
    b.mov_rr(Reg::R9, Reg::R8); // save freelist slot ptr across syscall
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, 65_536);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    // Note: R9 was clobbered by mmap offset 0! Save slot ptr in r14? No, recompute from TCB!
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_mmap_fail = b.new_label();
    b.jcc_label(Cond::Ae, l_mmap_fail);

    // Carve 16 pages: [page+0]=SLAB_MAGIC, [page+8]=class, blocks at [page+csz .. page+4096)
    b.mov_rr(Reg::Rsi, Reg::Rax); // cur page
    b.xor_rr32(Reg::Rdx, Reg::Rdx); // head = 0
    b.mov_ri64(Reg::Rcx, 16); // page counter
    b.mov_ri64(Reg::R10, SLAB_MAGIC);
    let l_page_lp = b.new_label();
    let l_blk_lp = b.new_label();
    let l_blk_end = b.new_label();
    b.bind_label(l_page_lp);
    b.store_mem(Reg::Rsi, 0, Reg::R10, 8);
    b.store_mem(Reg::Rsi, 8, Reg::Rbx, 8);
    b.mov_rr(Reg::Rdi, Reg::Rsi);
    b.add_rr(Reg::Rdi, Reg::R15); // blk = page + csz
    b.lea_mem(Reg::R11, Reg::Rsi, 4096); // page_end
    b.bind_label(l_blk_lp);
    b.cmp_rr(Reg::Rdi, Reg::R11);
    b.jcc_label(Cond::Ae, l_blk_end);
    b.store_mem(Reg::Rdi, 0, Reg::Rdx, 8);
    b.mov_rr(Reg::Rdx, Reg::Rdi);
    b.add_rr(Reg::Rdi, Reg::R15);
    b.jmp_label(l_blk_lp);
    b.bind_label(l_blk_end);
    b.add_ri32(Reg::Rsi, 4096);
    b.sub_ri32(Reg::Rcx, 1);
    b.jcc_label(Cond::Ne, l_page_lp);

    // Pop 1st block from rdx, store remaining list in tcb.freelists[rbx], zero first qword of ret
    b.load_mem(Reg::R10, Reg::Rdx, 0, 8, false);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rdx, 0, Reg::Rax, 8);
    b.load_fs_disp(Reg::R9, 0);
    b.test_rr(Reg::R9, Reg::R9);
    let l_refill_tcb = b.new_label();
    b.jcc_label(Cond::Ne, l_refill_tcb);
    b.lea_rip_sym(Reg::R9, "__eon_alloc_tls_fallback", 0);
    b.bind_label(l_refill_tcb);
    b.lea_mem(Reg::R8, Reg::R9, 16);
    b.shl_imm(Reg::Rbx, 3);
    b.add_rr(Reg::R8, Reg::Rbx);
    b.store_mem(Reg::R8, 0, Reg::R10, 8);
    b.mov_rr(Reg::Rax, Reg::Rdx);
    b.jmp_label(l_done);

    // Medium path: 2049 <= r13 <= 262144 and r14 <= 4096
    b.bind_label(l_medium);
    b.cmp_ri32(Reg::R13, 262_144);
    b.jcc_label(Cond::A, l_large);
    b.cmp_ri32(Reg::R14, 4096);
    b.jcc_label(Cond::A, l_large);

    b.lea_mem(Reg::Rbx, Reg::R13, 4095);
    b.shr_imm(Reg::Rbx, 12); // npages in 1..=64
    emit_spin_lock(&mut b, "__eon_med_span_lock");
    b.lea_rip_sym(Reg::R9, "__eon_med_chunks", 0);
    b.xor_rr32(Reg::R10, Reg::R10); // empty_slot = 0
    b.xor_rr32(Reg::R8, Reg::R8);   // chunk_idx = 0..32

    let l_c_loop = b.new_label();
    let l_c_next = b.new_label();
    let l_c_done = b.new_label();
    let l_bit_lp = b.new_label();
    let l_bit_test = b.new_label();
    let l_bit_one = b.new_label();
    let l_bit_adv = b.new_label();
    let l_found_run = b.new_label();

    b.bind_label(l_c_loop);
    b.cmp_ri32(Reg::R8, MED_CHUNK_COUNT as i32);
    b.jcc_label(Cond::Ae, l_c_done);
    b.load_mem(Reg::R11, Reg::R9, 0, 8, false);
    b.test_rr(Reg::R11, Reg::R11);
    let l_scan_bm = b.new_label();
    b.jcc_label(Cond::Ne, l_scan_bm);
    b.test_rr(Reg::R10, Reg::R10);
    b.jcc_label(Cond::Ne, l_c_next);
    b.mov_rr(Reg::R10, Reg::R9);
    b.jmp_label(l_c_next);

    b.bind_label(l_scan_bm);
    b.xor_rr32(Reg::Rdi, Reg::Rdi); // bit = 0..512
    b.xor_rr32(Reg::Rdx, Reg::Rdx); // run = 0
    b.bind_label(l_bit_lp);
    b.cmp_ri32(Reg::Rdi, 512);
    b.jcc_label(Cond::Ae, l_c_next);
    b.mov_rr(Reg::Rsi, Reg::Rdi);
    b.shr_imm(Reg::Rsi, 6);
    b.shl_imm(Reg::Rsi, 3);
    b.lea_mem(Reg::Rax, Reg::R9, 8);
    b.add_rr(Reg::Rax, Reg::Rsi);
    b.load_mem(Reg::Rax, Reg::Rax, 0, 8, false);
    // Fast skip full u64 words at 64-bit boundaries
    b.mov_rr(Reg::Rcx, Reg::Rdi);
    b.and_ri32(Reg::Rcx, 63);
    b.jcc_label(Cond::Ne, l_bit_test);
    b.cmp_ri32(Reg::Rax, -1);
    b.jcc_label(Cond::Ne, l_bit_test);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.add_ri32(Reg::Rdi, 64);
    b.jmp_label(l_bit_lp);

    b.bind_label(l_bit_test);
    b.shr_cl(Reg::Rax);
    b.and_ri32(Reg::Rax, 1);
    b.jcc_label(Cond::Ne, l_bit_one);
    b.add_ri32(Reg::Rdx, 1);
    b.cmp_rr(Reg::Rdx, Reg::Rbx);
    b.jcc_label(Cond::E, l_found_run);
    b.jmp_label(l_bit_adv);
    b.bind_label(l_bit_one);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.bind_label(l_bit_adv);
    b.add_ri32(Reg::Rdi, 1);
    b.jmp_label(l_bit_lp);

    b.bind_label(l_c_next);
    b.add_ri32(Reg::R9, MED_CHUNK_STRIDE);
    b.add_ri32(Reg::R8, 1);
    b.jmp_label(l_c_loop);

    // Found contiguous run ending at bit Rdi
    b.bind_label(l_found_run);
    b.add_ri32(Reg::Rdi, 1);
    b.sub_rr(Reg::Rdi, Reg::Rbx); // start_bit
    b.mov_rr(Reg::R15, Reg::Rdi);
    // Save base_ptr in r11? Wait: emit_med_bitmap_update preserves R11!
    emit_med_bitmap_update(&mut b, true);
    b.shl_imm(Reg::R15, 12);
    b.add_rr(Reg::R15, Reg::R11); // ret_ptr
    emit_spin_unlock(&mut b, "__eon_med_span_lock");
    // Zero npages * 4096 bytes before returning reused medium span
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_rr(Reg::Rcx, Reg::Rbx);
    b.shl_imm(Reg::Rcx, 9);
    b.rep_stosq();
    b.mov_rr(Reg::Rax, Reg::R15);
    b.jmp_label(l_done);

    // No free run in active chunks; allocate new 2 MiB chunk if empty slot available
    b.bind_label(l_c_done);
    let l_med_full = b.new_label();
    b.test_rr(Reg::R10, Reg::R10);
    b.jcc_label(Cond::E, l_med_full);
    b.mov_rr(Reg::R15, Reg::R10); // save slot ptr across syscall
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, MED_CHUNK_BYTES);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_med_mmap_fail = b.new_label();
    b.jcc_label(Cond::Ae, l_med_mmap_fail);

    b.store_mem(Reg::R15, 0, Reg::Rax, 8);
    b.mov_rr(Reg::R9, Reg::Rax); // save ret_ptr
    // Set low `rbx` bits (1..=64) in bitmap[0]
    let l_all64 = b.new_label();
    let l_bm_w = b.new_label();
    b.cmp_ri32(Reg::Rbx, 64);
    b.jcc_label(Cond::E, l_all64);
    b.mov_ri64(Reg::Rdx, 1);
    b.mov_rr(Reg::Rcx, Reg::Rbx);
    b.shl_cl(Reg::Rdx);
    b.sub_ri32(Reg::Rdx, 1);
    b.jmp_label(l_bm_w);
    b.bind_label(l_all64);
    b.mov_ri64(Reg::Rdx, u64::MAX);
    b.bind_label(l_bm_w);
    b.store_mem(Reg::R15, 8, Reg::Rdx, 8);
    emit_spin_unlock(&mut b, "__eon_med_span_lock");
    b.mov_rr(Reg::Rax, Reg::R9);
    b.jmp_label(l_done);

    b.bind_label(l_med_mmap_fail);
    emit_spin_unlock(&mut b, "__eon_med_span_lock");
    b.jmp_label(l_mmap_fail);

    b.bind_label(l_med_full);
    emit_spin_unlock(&mut b, "__eon_med_span_lock");

    // Large path: SYS_mmap(align_up(eff_size, 4096))
    b.bind_label(l_large);
    b.lea_mem(Reg::R15, Reg::R13, 4095);
    b.and_ri32(Reg::R15, !4095); // len
    let l_lg_overalign = b.new_label();
    b.cmp_ri32(Reg::R14, 4096);
    b.jcc_label(Cond::A, l_lg_overalign);

    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_rr(Reg::Rsi, Reg::R15);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ae, l_mmap_fail);
    b.jmp_label(l_done);

    // Rare >4096 alignment: map len + align - 4096, trim unaligned prefix & suffix
    b.bind_label(l_lg_overalign);
    b.mov_rr(Reg::Rbx, Reg::R15);
    b.add_rr(Reg::Rbx, Reg::R14);
    b.sub_ri32(Reg::Rbx, 4096); // extra_len
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ae, l_mmap_fail);
    // base = rax; aligned = (base + align - 1) & ~(align - 1)
    b.mov_rr(Reg::R8, Reg::Rax); // base
    b.lea_mem(Reg::Rcx, Reg::R14, -1);
    b.lea_mem(Reg::R9, Reg::R8, 0);
    b.add_rr(Reg::R9, Reg::Rcx);
    b.not_r(Reg::Rcx);
    b.and_rr(Reg::R9, Reg::Rcx); // aligned
    // Trim prefix [base .. aligned)
    let l_no_pfx = b.new_label();
    b.cmp_rr(Reg::R9, Reg::R8);
    b.jcc_label(Cond::E, l_no_pfx);
    b.mov_rr(Reg::Rsi, Reg::R9);
    b.sub_rr(Reg::Rsi, Reg::R8);
    b.mov_rr(Reg::Rdi, Reg::R8);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_no_pfx);
    // Trim suffix [aligned + len .. base + extra_len)
    b.add_rr(Reg::R8, Reg::Rbx); // orig_end
    b.mov_rr(Reg::Rdi, Reg::R9);
    b.add_rr(Reg::Rdi, Reg::R15); // used_end
    let l_no_sfx = b.new_label();
    b.cmp_rr(Reg::R8, Reg::Rdi);
    b.jcc_label(Cond::Be, l_no_sfx);
    b.mov_rr(Reg::Rsi, Reg::R8);
    b.sub_rr(Reg::Rsi, Reg::Rdi);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_no_sfx);
    b.mov_rr(Reg::Rax, Reg::R9);
    b.jmp_label(l_done);

    b.bind_label(l_mmap_fail);
    b.xor_rr32(Reg::R15, Reg::R15);
    b.bind_label(l_rollback_fail);
    emit_alloc_cap_rollback(&mut b);
    b.xor_rr32(Reg::Rax, Reg::Rax);

    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.ret();
    b.finish()
}

/// `void eon_free_raw(EonAllocCap *cap [rdi], void *ptr [rsi], size_t size [rdx])`
pub fn emit_rt_free_raw() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret = b.new_label();
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_ret);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);

    b.mov_rr(Reg::R12, Reg::Rdi);
    b.mov_rr(Reg::R13, Reg::Rsi);
    b.mov_ri64(Reg::Rax, 1);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.test_rr(Reg::R14, Reg::R14);
    b.cmovcc(Cond::E, Reg::R14, Reg::Rax);

    let l_page_aligned = b.new_label();
    let l_update_cap = b.new_label();

    // Check (ptr & 4095) != 0 -> TLS slab object
    b.mov_rr(Reg::Rax, Reg::R13);
    b.and_ri32(Reg::Rax, 4095);
    b.jcc_label(Cond::E, l_page_aligned);

    b.mov_rr(Reg::R8, Reg::R13);
    b.and_ri32(Reg::R8, !4095); // page_base
    b.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
    b.mov_ri64(Reg::Rcx, SLAB_MAGIC);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ne, l_update_cap);
    b.load_mem(Reg::Rbx, Reg::R8, 8, 8, false); // class
    b.cmp_ri32(Reg::Rbx, 8);
    b.jcc_label(Cond::Ae, l_update_cap);

    emit_load_tcb(&mut b, Reg::R9);
    b.lea_mem(Reg::R10, Reg::R9, 16);
    b.shl_imm(Reg::Rbx, 3);
    b.add_rr(Reg::R10, Reg::Rbx);
    b.load_mem(Reg::Rax, Reg::R10, 0, 8, false);
    b.store_mem(Reg::R13, 0, Reg::Rax, 8);
    b.store_mem(Reg::R10, 0, Reg::R13, 8);
    b.jmp_label(l_update_cap);

    // Page-aligned: check medium 2 MiB chunk table, else SYS_munmap
    b.bind_label(l_page_aligned);
    b.lea_mem(Reg::Rbx, Reg::R14, 4095);
    b.shr_imm(Reg::Rbx, 12); // npages
    let l_munmap = b.new_label();
    b.cmp_ri32(Reg::Rbx, 64);
    b.jcc_label(Cond::A, l_munmap);

    emit_spin_lock(&mut b, "__eon_med_span_lock");
    b.lea_rip_sym(Reg::R9, "__eon_med_chunks", 0);
    b.xor_rr32(Reg::R8, Reg::R8);
    let l_scan = b.new_label();
    let l_next = b.new_label();
    let l_not_med = b.new_label();
    b.bind_label(l_scan);
    b.cmp_ri32(Reg::R8, MED_CHUNK_COUNT as i32);
    b.jcc_label(Cond::Ae, l_not_med);
    b.load_mem(Reg::R11, Reg::R9, 0, 8, false);
    b.test_rr(Reg::R11, Reg::R11);
    b.jcc_label(Cond::E, l_next);
    b.cmp_rr(Reg::R13, Reg::R11);
    b.jcc_label(Cond::B, l_next);
    b.mov_ri64(Reg::Rax, MED_CHUNK_BYTES);
    b.add_rr(Reg::Rax, Reg::R11);
    b.cmp_rr(Reg::R13, Reg::Rax);
    b.jcc_label(Cond::Ae, l_next);

    // Found chunk: clear npages bits at (ptr - base) >> 12
    b.mov_rr(Reg::Rdi, Reg::R13);
    b.sub_rr(Reg::Rdi, Reg::R11);
    b.shr_imm(Reg::Rdi, 12);
    emit_med_bitmap_update(&mut b, false);
    emit_spin_unlock(&mut b, "__eon_med_span_lock");
    b.jmp_label(l_update_cap);

    b.bind_label(l_next);
    b.add_ri32(Reg::R9, MED_CHUNK_STRIDE);
    b.add_ri32(Reg::R8, 1);
    b.jmp_label(l_scan);

    b.bind_label(l_not_med);
    emit_spin_unlock(&mut b, "__eon_med_span_lock");

    b.bind_label(l_munmap);
    b.mov_rr(Reg::Rdi, Reg::R13);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.shl_imm(Reg::Rsi, 12);
    b.mov_ri64(Reg::Rax, 11); // SYS_munmap
    b.syscall();

    b.bind_label(l_update_cap);
    emit_alloc_cap_release(&mut b);

    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.bind_label(l_ret);
    b.ret();
    b.finish()
}

/// `void *eon_alloc_raw_san(EonAllocCap *cap [rdi], size_t size [rsi], size_t align [rdx])`
/// `--sanitize` allocator: places every buffer right against a `PROT_NONE` trailing guard page
/// so any 1-byte overflow triggers an instant hardware `SIGSEGV`.
pub fn emit_rt_alloc_raw_san() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::R12, Reg::Rdi);
    b.mov_ri64(Reg::Rax, 1);
    b.mov_rr(Reg::R13, Reg::Rsi);
    b.test_rr(Reg::R13, Reg::R13);
    b.cmovcc(Cond::E, Reg::R13, Reg::Rax);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.test_rr(Reg::R14, Reg::R14);
    b.cmovcc(Cond::E, Reg::R14, Reg::Rax);
    // Clamp alignment <= 4096 so free_raw_san can always recover base via align_up(ptr+size, 4096)
    b.mov_ri64(Reg::Rax, 4096);
    b.cmp_ri32(Reg::R14, 4096);
    b.cmovcc(Cond::A, Reg::R14, Reg::Rax);

    let l_rollback = b.new_label();
    emit_alloc_cap_charge(&mut b, l_rollback);

    // data_pages = align_up(r13, 4096), total_map = data_pages + 4096
    b.lea_mem(Reg::Rbx, Reg::R13, 4095);
    b.and_ri32(Reg::Rbx, !4095); // rbx = data_pages
    b.lea_mem(Reg::Rsi, Reg::Rbx, 4096);
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();

    let l_mmap_fail = b.new_label();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ae, l_mmap_fail);

    // mprotect(base + data_pages, 4096, PROT_NONE)
    b.mov_rr(Reg::R15, Reg::Rax);
    b.add_rr(Reg::R15, Reg::Rbx); // r15 = guard_start
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.mov_ri64(Reg::Rsi, 4096);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.mov_ri64(Reg::Rax, 10); // SYS_mprotect
    b.syscall();

    // ret = (guard_start - eff_size) & ~(eff_align - 1)
    b.mov_rr(Reg::Rax, Reg::R15);
    b.sub_rr(Reg::Rax, Reg::R13);
    b.lea_mem(Reg::Rcx, Reg::R14, -1);
    b.not_r(Reg::Rcx);
    b.and_rr(Reg::Rax, Reg::Rcx);
    let l_done = b.new_label();
    b.jmp_label(l_done);

    b.bind_label(l_mmap_fail);
    b.xor_rr32(Reg::R15, Reg::R15);
    b.bind_label(l_rollback);
    emit_alloc_cap_rollback(&mut b);
    b.xor_rr32(Reg::Rax, Reg::Rax);

    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.ret();
    b.finish()
}

/// `void eon_free_raw_san(EonAllocCap *cap [rdi], void *ptr [rsi], size_t size [rdx])`
/// `--sanitize` deallocator: poisons buffer with `0xDE`, locks pages to `PROT_NONE`,
/// and pushes into the 64-slot quarantine ring so UAF faults immediately without VMA leaks.
pub fn emit_rt_free_raw_san() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret = b.new_label();
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_ret);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::R12, Reg::Rdi);
    b.mov_rr(Reg::R13, Reg::Rsi);
    b.mov_ri64(Reg::Rax, 1);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.test_rr(Reg::R14, Reg::R14);
    b.cmovcc(Cond::E, Reg::R14, Reg::Rax);

    // Poison [ptr .. ptr + eff_size] with 0xDE
    b.mov_rr(Reg::Rdi, Reg::R13);
    b.mov_ri64(Reg::Rax, SAN_POISON_BYTE);
    b.mov_rr(Reg::Rcx, Reg::R14);
    b.rep_stosb();

    // guard_start = align_up(ptr + eff_size, 4096)
    // data_pages = align_up(eff_size, 4096)
    // base = guard_start - data_pages, total_map = data_pages + 4096
    b.mov_rr(Reg::Rbx, Reg::R14);
    b.add_ri32(Reg::Rbx, 4095);
    b.and_ri32(Reg::Rbx, !4095); // rbx = data_pages

    b.mov_rr(Reg::R15, Reg::R13);
    b.add_rr(Reg::R15, Reg::R14);
    b.add_ri32(Reg::R15, 4095);
    b.and_ri32(Reg::R15, !4095); // guard_start
    b.sub_rr(Reg::R15, Reg::Rbx); // r15 = base

    // mprotect(base, data_pages, PROT_NONE)
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.mov_ri64(Reg::Rax, 10); // SYS_mprotect
    b.syscall();

    // Quarantine ring push: spin-lock, swap (base, total_len) at slot, unlock, munmap old
    b.add_ri32(Reg::Rbx, 4096); // rbx = total_map
    emit_spin_lock(&mut b, "__eon_san_lock");
    b.lea_rip_sym(Reg::R9, "__eon_san_quarantine", 0);
    b.load_mem(Reg::R10, Reg::R9, 0, 8, false); // head idx
    b.mov_rr(Reg::R11, Reg::R10);
    b.add_ri32(Reg::R11, 1);
    b.and_ri32(Reg::R11, (SAN_QUARANTINE_SLOTS - 1) as i32);
    b.store_mem(Reg::R9, 0, Reg::R11, 8);

    b.and_ri32(Reg::R10, (SAN_QUARANTINE_SLOTS - 1) as i32);
    b.shl_imm(Reg::R10, 4); // * 16
    b.lea_mem(Reg::R9, Reg::R9, 8);
    b.add_rr(Reg::R9, Reg::R10);
    b.load_mem(Reg::R10, Reg::R9, 0, 8, false); // old_base
    b.load_mem(Reg::R11, Reg::R9, 8, 8, false); // old_len
    b.store_mem(Reg::R9, 0, Reg::R15, 8);
    b.store_mem(Reg::R9, 8, Reg::Rbx, 8);
    b.mov_rr(Reg::R15, Reg::R10);
    b.mov_rr(Reg::Rbx, Reg::R11);
    emit_spin_unlock(&mut b, "__eon_san_lock");

    let l_no_evict = b.new_label();
    b.test_rr(Reg::R15, Reg::R15);
    b.jcc_label(Cond::E, l_no_evict);
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_ri64(Reg::Rax, 11); // SYS_munmap
    b.syscall();
    b.bind_label(l_no_evict);

    emit_alloc_cap_release(&mut b);

    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.bind_label(l_ret);
    b.ret();
    b.finish()
}

/// `void *eon_arena_alloc_aligned(EonArena *a [rdi], size_t size [rsi], size_t align [rdx])`
/// EonArena layout: [0]=buf, [8]=cap, [16]=used.
pub fn emit_rt_arena_alloc() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_fail = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_fail);
    b.load_mem(Reg::R8, Reg::Rdi, 0, 8, false); // buf
    b.test_rr(Reg::R8, Reg::R8);
    b.jcc_label(Cond::E, l_fail);

    // mask = align <= 1 ? 0 : align - 1
    let l_m_ok = b.new_label();
    b.cmp_ri32(Reg::Rdx, 1);
    b.jcc_label(Cond::A, l_m_ok);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    let l_m_done = b.new_label();
    b.jmp_label(l_m_done);
    b.bind_label(l_m_ok);
    b.sub_ri32(Reg::Rdx, 1);
    b.bind_label(l_m_done);

    // base = (used + mask) & ~mask
    b.load_mem(Reg::R9, Reg::Rdi, 16, 8, false); // used
    b.add_rr(Reg::R9, Reg::Rdx);
    b.not_r(Reg::Rdx);
    b.and_rr(Reg::R9, Reg::Rdx); // r9 = base

    // cap = [rdi+8]; if size > cap || base > cap - size => fail
    b.load_mem(Reg::R10, Reg::Rdi, 8, 8, false); // cap
    b.cmp_rr(Reg::Rsi, Reg::R10);
    b.jcc_label(Cond::A, l_fail);
    b.sub_rr(Reg::R10, Reg::Rsi); // cap - size
    b.cmp_rr(Reg::R9, Reg::R10);
    b.jcc_label(Cond::A, l_fail);

    // p = buf + base; used = base + size; return p
    b.mov_rr(Reg::Rax, Reg::R8);
    b.add_rr(Reg::Rax, Reg::R9);
    b.add_rr(Reg::R9, Reg::Rsi);
    b.store_mem(Reg::Rdi, 16, Reg::R9, 8);
    b.ret();

    b.bind_label(l_fail);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.ret();
    b.finish()
}

/// `void eon_rt_arena_reset_san(EonArena *a [rdi])`
/// Poisons `[buf .. buf + used]` with `0xDE` before resetting `used = 0`.
pub fn emit_rt_arena_reset_san() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_done = b.new_label();
    let l_zero_only = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_done);
    b.mov_rr(Reg::R8, Reg::Rdi);
    b.load_mem(Reg::Rdx, Reg::R8, 0, 8, false);  // buf
    b.load_mem(Reg::Rcx, Reg::R8, 16, 8, false); // used
    b.test_rr(Reg::Rdx, Reg::Rdx);
    b.jcc_label(Cond::E, l_zero_only);
    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::E, l_zero_only);
    b.mov_rr(Reg::Rdi, Reg::Rdx);
    b.mov_ri64(Reg::Rax, SAN_POISON_BYTE);
    b.rep_stosb();
    b.bind_label(l_zero_only);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R8, 16, Reg::Rax, 8);
    b.bind_label(l_done);
    b.ret();
    b.finish()
}

/// `void eon_rt_install_san_handler(void)`
/// Installs `SYS_sigaltstack` (`__eon_san_altstack`) and `SYS_rt_sigaction` handlers
/// for `SIGSEGV (11)`, `SIGILL (4)`, and `SIGBUS (7)` printing fault address/RIP/backtrace.
pub fn emit_rt_install_san_handler() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_sig_handler = b.new_label();
    let l_sigreturn = b.new_label();
    let l_write_hex = b.new_label();
    let l_str_hdr = b.new_label();
    let l_str_addr = b.new_label();
    let l_str_rip = b.new_label();
    let l_str_bt = b.new_label();
    let l_str_nl = b.new_label();

    b.sub_ri32(Reg::Rsp, 40);
    // stack_t { ss_sp, ss_flags(0), ss_size(16384) }
    b.lea_rip_sym(Reg::Rax, "__eon_san_altstack", 0);
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 8);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rsp, 8, Reg::Rax, 8);
    b.mov_ri64(Reg::Rax, 16_384);
    b.store_mem(Reg::Rsp, 16, Reg::Rax, 8);
    b.mov_ri64(Reg::Rax, 131); // SYS_sigaltstack
    b.mov_rr(Reg::Rdi, Reg::Rsp);
    b.xor_rr32(Reg::Rsi, Reg::Rsi);
    b.syscall();

    // Kernel struct sigaction { sa_handler, sa_flags, sa_restorer, sa_mask }
    b.lea_rip_label(Reg::Rax, l_sig_handler);
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 8);
    b.mov_ri64(Reg::Rax, 0x0C00_0004); // SA_RESTORER | SA_ONSTACK | SA_SIGINFO
    b.store_mem(Reg::Rsp, 8, Reg::Rax, 8);
    b.lea_rip_label(Reg::Rax, l_sigreturn);
    b.store_mem(Reg::Rsp, 16, Reg::Rax, 8);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rsp, 24, Reg::Rax, 8);

    for sig in [11u64, 4, 7] {
        b.mov_ri64(Reg::Rax, 13); // SYS_rt_sigaction
        b.mov_ri64(Reg::Rdi, sig);
        b.mov_rr(Reg::Rsi, Reg::Rsp);
        b.xor_rr32(Reg::Rdx, Reg::Rdx);
        b.mov_ri64(Reg::R10, 8); // sizeof(kernel sigset_t)
        b.syscall();
    }

    b.add_ri32(Reg::Rsp, 40);
    b.ret();

    // __eon_san_sigreturn
    b.bind_label(l_sigreturn);
    b.mov_ri64(Reg::Rax, 15); // SYS_rt_sigreturn
    b.syscall();
    b.ud2();

    // Helper: write "0x" + 16 hex digits of Rdi to stderr (fd=2)
    b.bind_label(l_write_hex);
    b.sub_ri32(Reg::Rsp, 24);
    b.mov_ri64(Reg::Rax, u64::from_le_bytes(*b"0x\0\0\0\0\0\0"));
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 2);
    b.mov_ri64(Reg::Rcx, 16);
    let l_hex_lp = b.new_label();
    let l_hex_alpha = b.new_label();
    let l_hex_st = b.new_label();
    b.bind_label(l_hex_lp);
    b.mov_rr(Reg::Rax, Reg::Rdi);
    b.and_ri32(Reg::Rax, 0xF);
    b.cmp_ri32(Reg::Rax, 10);
    b.jcc_label(Cond::Ae, l_hex_alpha);
    b.add_ri32(Reg::Rax, b'0' as i32);
    b.jmp_label(l_hex_st);
    b.bind_label(l_hex_alpha);
    b.add_ri32(Reg::Rax, (b'a' as i32) - 10);
    b.bind_label(l_hex_st);
    b.lea_mem(Reg::Rdx, Reg::Rsp, 1);
    b.add_rr(Reg::Rdx, Reg::Rcx);
    b.store_mem(Reg::Rdx, 0, Reg::Rax, 1);
    b.shr_imm(Reg::Rdi, 4);
    b.sub_ri32(Reg::Rcx, 1);
    b.jcc_label(Cond::Ne, l_hex_lp);
    b.mov_ri64(Reg::Rax, 1);
    b.mov_ri64(Reg::Rdi, 2);
    b.mov_rr(Reg::Rsi, Reg::Rsp);
    b.mov_ri64(Reg::Rdx, 18);
    b.syscall();
    b.add_ri32(Reg::Rsp, 24);
    b.ret();

    // __eon_san_sig_handler(sig: rdi, siginfo: rsi, ucontext: rdx)
    b.bind_label(l_sig_handler);
    b.mov_rr(Reg::R12, Reg::Rdi);
    b.mov_rr(Reg::R13, Reg::Rsi);
    b.mov_rr(Reg::R14, Reg::Rdx);
    // Align stack for local calls
    b.and_ri32(Reg::Rsp, !15);

    let emit_stderr = |buf: &mut X64Buf, lbl: Label, len: u64| {
        buf.mov_ri64(Reg::Rax, 1);
        buf.mov_ri64(Reg::Rdi, 2);
        buf.lea_rip_label(Reg::Rsi, lbl);
        buf.mov_ri64(Reg::Rdx, len);
        buf.syscall();
    };

    emit_stderr(&mut b, l_str_hdr, 23);
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.call_label(l_write_hex);

    emit_stderr(&mut b, l_str_addr, 6);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    let l_no_si = b.new_label();
    b.test_rr(Reg::R13, Reg::R13);
    b.jcc_label(Cond::E, l_no_si);
    b.load_mem(Reg::Rdi, Reg::R13, 16, 8, false); // siginfo.si_addr
    b.bind_label(l_no_si);
    b.call_label(l_write_hex);

    emit_stderr(&mut b, l_str_rip, 5);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    let l_no_uc = b.new_label();
    b.test_rr(Reg::R14, Reg::R14);
    b.jcc_label(Cond::E, l_no_uc);
    b.load_mem(Reg::Rdi, Reg::R14, 168, 8, false); // ucontext.uc_mcontext.rip
    b.bind_label(l_no_uc);
    b.call_label(l_write_hex);
    emit_stderr(&mut b, l_str_nl, 1);

    // Frame-pointer backtrace walk (guarded against corrupted/overflowed stack)
    let l_bt_done = b.new_label();
    b.test_rr(Reg::R14, Reg::R14);
    b.jcc_label(Cond::E, l_bt_done);
    b.load_mem(Reg::Rbx, Reg::R14, 120, 8, false); // REG_RBP
    b.load_mem(Reg::R13, Reg::R14, 160, 8, false); // REG_RSP
    b.mov_ri64(Reg::R15, 16);
    let l_bt_lp = b.new_label();
    b.bind_label(l_bt_lp);
    b.cmp_rr(Reg::Rbx, Reg::R13);
    b.jcc_label(Cond::B, l_bt_done);
    b.mov_ri64(Reg::Rax, 0x10_0000);
    b.add_rr(Reg::Rax, Reg::R13);
    b.cmp_rr(Reg::Rbx, Reg::Rax);
    b.jcc_label(Cond::Ae, l_bt_done);
    b.mov_rr(Reg::Rax, Reg::Rbx);
    b.and_ri32(Reg::Rax, 7);
    b.jcc_label(Cond::Ne, l_bt_done);

    emit_stderr(&mut b, l_str_bt, 6);
    b.load_mem(Reg::Rdi, Reg::Rbx, 8, 8, false);
    b.call_label(l_write_hex);
    emit_stderr(&mut b, l_str_nl, 1);

    b.load_mem(Reg::Rax, Reg::Rbx, 0, 8, false);
    b.cmp_rr(Reg::Rax, Reg::Rbx);
    b.jcc_label(Cond::Be, l_bt_done);
    b.mov_rr(Reg::Rbx, Reg::Rax);
    b.sub_ri32(Reg::R15, 1);
    b.jcc_label(Cond::Ne, l_bt_lp);

    b.bind_label(l_bt_done);
    b.lea_mem(Reg::Rdi, Reg::R12, 128);
    b.mov_ri64(Reg::Rax, 231); // SYS_exit_group
    b.syscall();
    b.ud2();

    b.bind_label(l_str_hdr);
    b.code.extend_from_slice(b"EON SANITIZER TRAP sig=");
    b.bind_label(l_str_addr);
    b.code.extend_from_slice(b" addr=");
    b.bind_label(l_str_rip);
    b.code.extend_from_slice(b" rip=");
    b.bind_label(l_str_bt);
    b.code.extend_from_slice(b"  bt: ");
    b.bind_label(l_str_nl);
    b.code.extend_from_slice(b"\n");

    b.finish()
}

/// `void eon_rt_tls_init(void)`
/// Initializes `__eon_main_tcb` self-pointer and sets main thread's `FS.base` via `SYS_arch_prctl`.
pub fn emit_rt_tls_init() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.lea_rip_sym(Reg::Rsi, "__eon_main_tcb", 0);
    b.store_mem(Reg::Rsi, 0, Reg::Rsi, 8);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rsi, 8, Reg::Rax, 8);
    b.mov_ri64(Reg::Rdi, 0x1002); // ARCH_SET_FS
    b.mov_ri64(Reg::Rax, 158);    // SYS_arch_prctl
    b.syscall();
    b.ret();
    b.finish()
}

/// `void *eon_rt_parse_vdso(const void *vdso_ehdr [rdi])`
/// Parses in-memory Linux ELF64 vDSO, resolves `__vdso_clock_gettime`, caches it in
/// `__eon_vdso_clock_gettime`, and returns the pointer (or `0` on failure).
pub fn emit_rt_parse_vdso() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_fail = b.new_label();
    let l_str_name = b.new_label();

    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_fail);
    b.load_mem(Reg::Rax, Reg::Rdi, 0, 4, false);
    b.mov_ri64(Reg::Rcx, 0x464C_457F); // "\x7fELF"
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ne, l_fail);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.load_mem(Reg::R8, Reg::Rdi, 32, 8, false);  // e_phoff
    b.add_rr(Reg::R8, Reg::Rdi);                  // ph_ptr
    b.load_mem(Reg::R9, Reg::Rdi, 54, 2, false);  // e_phentsize
    b.load_mem(Reg::R10, Reg::Rdi, 56, 2, false); // e_phnum

    b.xor_rr32(Reg::R12, Reg::R12); // load_bias
    b.xor_rr32(Reg::Rbx, Reg::Rbx); // got_load = 0
    b.xor_rr32(Reg::R13, Reg::R13); // dyn_ptr = 0

    let l_ph_lp = b.new_label();
    let l_ph_next = b.new_label();
    let l_ph_done = b.new_label();
    b.bind_label(l_ph_lp);
    b.test_rr(Reg::R10, Reg::R10);
    b.jcc_label(Cond::E, l_ph_done);
    b.load_mem(Reg::Rax, Reg::R8, 0, 4, false); // p_type
    b.cmp_ri32(Reg::Rax, 1); // PT_LOAD
    let l_not_load = b.new_label();
    b.jcc_label(Cond::Ne, l_not_load);
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::Ne, l_ph_next);
    b.mov_rr(Reg::R12, Reg::Rdi);
    b.load_mem(Reg::Rcx, Reg::R8, 16, 8, false); // p_vaddr
    b.sub_rr(Reg::R12, Reg::Rcx); // load_bias = vdso - p_vaddr
    b.mov_ri64(Reg::Rbx, 1);
    b.jmp_label(l_ph_next);

    b.bind_label(l_not_load);
    b.cmp_ri32(Reg::Rax, 2); // PT_DYNAMIC
    b.jcc_label(Cond::Ne, l_ph_next);
    b.load_mem(Reg::R13, Reg::R8, 16, 8, false); // p_vaddr

    b.bind_label(l_ph_next);
    b.add_rr(Reg::R8, Reg::R9);
    b.sub_ri32(Reg::R10, 1);
    b.jmp_label(l_ph_lp);

    b.bind_label(l_ph_done);
    let l_fail_pop = b.new_label();
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_fail_pop);
    b.test_rr(Reg::R13, Reg::R13);
    b.jcc_label(Cond::E, l_fail_pop);

    // Relocate d_val helper inline: if val >= rdi && val < rdi + 0x10_0000 { val } else { val + r12 }
    let reloc_val = |buf: &mut X64Buf, reg: Reg| {
        let l_rel = buf.new_label();
        let l_ok = buf.new_label();
        buf.cmp_rr(reg, Reg::Rdi);
        buf.jcc_label(Cond::B, l_rel);
        buf.mov_ri64(Reg::Rax, 0x10_0000);
        buf.add_rr(Reg::Rax, Reg::Rdi);
        buf.cmp_rr(reg, Reg::Rax);
        buf.jcc_label(Cond::B, l_ok);
        buf.bind_label(l_rel);
        buf.add_rr(reg, Reg::R12);
        buf.bind_label(l_ok);
    };

    reloc_val(&mut b, Reg::R13);

    // Walk .dynamic to find DT_HASH (4), DT_STRTAB (5), DT_SYMTAB (6)
    b.xor_rr32(Reg::R8, Reg::R8);   // sysv_hash
    b.xor_rr32(Reg::R14, Reg::R14); // strtab
    b.xor_rr32(Reg::R15, Reg::R15); // symtab
    let l_dyn_lp = b.new_label();
    let l_dyn_next = b.new_label();
    let l_dyn_done = b.new_label();
    b.bind_label(l_dyn_lp);
    b.load_mem(Reg::Rcx, Reg::R13, 0, 8, false); // d_tag
    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::E, l_dyn_done);
    b.load_mem(Reg::Rdx, Reg::R13, 8, 8, false); // d_val
    b.cmp_ri32(Reg::Rcx, 4);
    let l_not_hash = b.new_label();
    b.jcc_label(Cond::Ne, l_not_hash);
    b.mov_rr(Reg::R8, Reg::Rdx);
    b.jmp_label(l_dyn_next);
    b.bind_label(l_not_hash);
    b.cmp_ri32(Reg::Rcx, 5);
    let l_not_str = b.new_label();
    b.jcc_label(Cond::Ne, l_not_str);
    b.mov_rr(Reg::R14, Reg::Rdx);
    b.jmp_label(l_dyn_next);
    b.bind_label(l_not_str);
    b.cmp_ri32(Reg::Rcx, 6);
    b.jcc_label(Cond::Ne, l_dyn_next);
    b.mov_rr(Reg::R15, Reg::Rdx);
    b.bind_label(l_dyn_next);
    b.add_ri32(Reg::R13, 16);
    b.jmp_label(l_dyn_lp);

    b.bind_label(l_dyn_done);
    b.test_rr(Reg::R8, Reg::R8);
    b.jcc_label(Cond::E, l_fail_pop);
    b.test_rr(Reg::R14, Reg::R14);
    b.jcc_label(Cond::E, l_fail_pop);
    b.test_rr(Reg::R15, Reg::R15);
    b.jcc_label(Cond::E, l_fail_pop);

    reloc_val(&mut b, Reg::R8);
    reloc_val(&mut b, Reg::R14);
    reloc_val(&mut b, Reg::R15);

    // nchain = dword [sysv_hash + 4]
    b.load_mem(Reg::R9, Reg::R8, 4, 4, false);
    b.lea_rip_label(Reg::R10, l_str_name); // target "__vdso_clock_gettime\0"
    b.xor_rr32(Reg::Rbx, Reg::Rbx);       // sym idx

    let l_sym_lp = b.new_label();
    let l_sym_next = b.new_label();
    let l_cmp_lp = b.new_label();
    b.bind_label(l_sym_lp);
    b.cmp_rr(Reg::Rbx, Reg::R9);
    b.jcc_label(Cond::Ae, l_fail_pop);
    b.imul_rri32(Reg::Rax, Reg::Rbx, 24);
    b.add_rr(Reg::Rax, Reg::R15); // Elf64_Sym*
    b.load_mem(Reg::R11, Reg::Rax, 8, 8, false); // st_value
    b.test_rr(Reg::R11, Reg::R11);
    b.jcc_label(Cond::E, l_sym_next);
    b.load_mem(Reg::Rsi, Reg::Rax, 0, 4, false); // st_name
    b.add_rr(Reg::Rsi, Reg::R14);                // name ptr

    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    b.bind_label(l_cmp_lp);
    b.mov_rr(Reg::Rdx, Reg::Rsi);
    b.add_rr(Reg::Rdx, Reg::Rcx);
    b.load_mem(Reg::Rax, Reg::Rdx, 0, 1, false);
    b.mov_rr(Reg::Rdx, Reg::R10);
    b.add_rr(Reg::Rdx, Reg::Rcx);
    b.load_mem(Reg::Rdx, Reg::Rdx, 0, 1, false);
    b.cmp_rr(Reg::Rax, Reg::Rdx);
    b.jcc_label(Cond::Ne, l_sym_next);
    b.add_ri32(Reg::Rcx, 1);
    b.cmp_ri32(Reg::Rcx, 21); // len("__vdso_clock_gettime\0")
    b.jcc_label(Cond::B, l_cmp_lp);

    // Match! fn_addr = load_bias + st_value
    b.mov_rr(Reg::Rax, Reg::R12);
    b.add_rr(Reg::Rax, Reg::R11);
    b.mov_rip_store("__eon_vdso_clock_gettime", 0, Reg::Rax);
    let l_done = b.new_label();
    b.jmp_label(l_done);

    b.bind_label(l_sym_next);
    b.add_ri32(Reg::Rbx, 1);
    b.jmp_label(l_sym_lp);

    b.bind_label(l_fail_pop);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.ret();

    b.bind_label(l_fail);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.ret();

    b.bind_label(l_str_name);
    b.code.extend_from_slice(b"__vdso_clock_gettime\0");

    b.finish()
}

/// `u64 eon_rt_now_ns(void)`
/// Uses cached `__eon_vdso_clock_gettime` in userspace when resolved (~3 ns),
/// transparently falling back to `SYS_clock_gettime(CLOCK_MONOTONIC=1, &ts)`.
pub fn emit_rt_now_ns() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 24); // aligned 16B before call; timespec at [rsp..rsp+16]
    let l_fallback = b.new_label();
    let l_compute = b.new_label();

    b.mov_rip_load(Reg::Rax, "__eon_vdso_clock_gettime", 0);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_fallback);
    b.mov_ri64(Reg::Rdi, 1); // CLOCK_MONOTONIC
    b.mov_rr(Reg::Rsi, Reg::Rsp);
    b.call_reg(Reg::Rax);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_compute);

    b.bind_label(l_fallback);
    b.mov_ri64(Reg::Rax, 228); // SYS_clock_gettime
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_rr(Reg::Rsi, Reg::Rsp);
    b.syscall();

    b.bind_label(l_compute);
    b.load_mem(Reg::Rax, Reg::Rsp, 0, 8, false); // tv_sec
    b.imul_rri32(Reg::Rax, Reg::Rax, 1_000_000_000);
    b.load_mem(Reg::Rdx, Reg::Rsp, 8, 8, false); // tv_nsec
    b.add_rr(Reg::Rax, Reg::Rdx);
    b.add_ri32(Reg::Rsp, 24);
    b.ret();
    b.finish()
}

/// `i64 eon_rt_openat2_beneath(i32 dirfd [rdi], const char *path [rsi], u64 flags [rdx], u64 mode [rcx])`
/// Invokes Linux `SYS_openat2(437)` with `RESOLVE_BENEATH` (`0x08`) to block `..` / symlink escapes.
pub fn emit_rt_openat2_beneath() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 24);
    b.store_mem(Reg::Rsp, 0, Reg::Rdx, 8); // open_how.flags
    b.store_mem(Reg::Rsp, 8, Reg::Rcx, 8); // open_how.mode
    b.mov_ri64(Reg::Rax, 8);               // RESOLVE_BENEATH
    b.store_mem(Reg::Rsp, 16, Reg::Rax, 8);
    b.mov_rr(Reg::Rdx, Reg::Rsp);
    b.mov_ri64(Reg::R10, 24); // sizeof(struct open_how)
    b.mov_ri64(Reg::Rax, 437); // SYS_openat2
    b.syscall();
    b.add_ri32(Reg::Rsp, 24);
    b.ret();
    b.finish()
}

/// `isize eon_rt_read_fd(i32 fd [rdi], u8 *buf [rsi], usize len [rdx])`
pub fn emit_rt_read_fd() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.xor_rr32(Reg::Rax, Reg::Rax); // SYS_read (0)
    b.syscall();
    b.ret();
    b.finish()
}

/// `isize eon_rt_write_fd(i32 fd [rdi], const u8 *buf [rsi], usize len [rdx])`
pub fn emit_rt_write_fd() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.mov_ri64(Reg::Rax, 1); // SYS_write (1)
    b.syscall();
    b.ret();
    b.finish()
}

/// `i64 eon_rt_close_fd(i32 fd [rdi])`
pub fn emit_rt_close_fd() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.mov_ri64(Reg::Rax, 3); // SYS_close (3)
    b.syscall();
    b.ret();
    b.finish()
}

/// `void eon_rt_print_str(const u8 *ptr [rdi], usize len [rsi])`
pub fn emit_rt_print_str() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_done = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_done);
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_done);
    // SYS_write(fd=1, buf=rsi, count=rdx)
    b.mov_rr(Reg::Rdx, Reg::Rsi);
    b.mov_rr(Reg::Rsi, Reg::Rdi);
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_ri64(Reg::Rax, 1);
    b.syscall();
    b.bind_label(l_done);
    b.ret();
    b.finish()
}

/// `void eon_rt_print_nl(void)`
pub fn emit_rt_print_nl() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 16);
    b.mov_ri64(Reg::Rax, 10); // '\n'
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 1);
    b.mov_ri64(Reg::Rax, 1);
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_rr(Reg::Rsi, Reg::Rsp);
    b.mov_ri64(Reg::Rdx, 1);
    b.syscall();
    b.add_ri32(Reg::Rsp, 16);
    b.ret();
    b.finish()
}

/// Internal emitter for `void fn(const u8 *ptr [rdi], usize len [rsi])` writing all bytes to `fd`
/// via `eon_rt_write_fd`, retrying on `-EINTR` and handling partial writes.
fn emit_rt_write_all_to_fd(fd: u64) -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret_fast = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_ret_fast);
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_ret_fast);

    // 3 pushes + 8B return address => 32B => RSP 16-byte aligned before `call eon_rt_write_fd`.
    b.push(Reg::Rbp);
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.mov_rr(Reg::Rbx, Reg::Rdi); // ptr
    b.mov_rr(Reg::R12, Reg::Rsi); // remaining

    let l_loop = b.new_label();
    let l_adv = b.new_label();
    let l_done = b.new_label();
    b.bind_label(l_loop);
    b.test_rr(Reg::R12, Reg::R12);
    b.jcc_label(Cond::E, l_done);
    b.mov_ri64(Reg::Rdi, fd);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_rr(Reg::Rdx, Reg::R12);
    b.call_sym("eon_rt_write_fd");
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::G, l_adv);
    b.cmp_ri32(Reg::Rax, -4); // EINTR
    b.jcc_label(Cond::E, l_loop);
    b.jmp_label(l_done);

    b.bind_label(l_adv);
    b.add_rr(Reg::Rbx, Reg::Rax);
    b.sub_rr(Reg::R12, Reg::Rax);
    b.jmp_label(l_loop);

    b.bind_label(l_done);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.pop(Reg::Rbp);
    b.bind_label(l_ret_fast);
    b.ret();
    b.finish()
}

/// `void eon_rt_print_bytes(const u8 *ptr [rdi], usize len [rsi])`
pub fn emit_rt_print_bytes() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    emit_rt_write_all_to_fd(1)
}

/// `void eon_rt_eprint_str(const u8 *ptr [rdi], usize len [rsi])`
pub fn emit_rt_eprint_str() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    emit_rt_write_all_to_fd(2)
}

/// `void eon_rt_eprint_bytes(const u8 *ptr [rdi], usize len [rsi])`
pub fn emit_rt_eprint_bytes() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    emit_rt_write_all_to_fd(2)
}

/// `void eon_rt_eprint_nl(void)`
pub fn emit_rt_eprint_nl() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    // 24B + 8B return address => 32B => RSP 16-byte aligned before `call eon_rt_write_fd`.
    b.sub_ri32(Reg::Rsp, 24);
    b.mov_ri64(Reg::Rax, 10); // '\n'
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 1);
    b.mov_ri64(Reg::Rdi, 2);
    b.mov_rr(Reg::Rsi, Reg::Rsp);
    b.mov_ri64(Reg::Rdx, 1);
    b.call_sym("eon_rt_write_fd");
    b.add_ri32(Reg::Rsp, 24);
    b.ret();
    b.finish()
}

/// `u64 eon_rt_validate_utf8(const u8 *ptr [rdi], usize len [rsi])`
/// Returns `1` if `ptr[0..len]` is strictly valid RFC 3629 UTF-8, `0` otherwise.
pub fn emit_rt_validate_utf8() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_loop = b.new_label();
    let l_step1 = b.new_label();
    let l_chk3 = b.new_label();
    let l_chk3_ed = b.new_label();
    let l_chk3_norm = b.new_label();
    let l_chk4 = b.new_label();
    let l_chk4_f4 = b.new_label();
    let l_chk4_norm = b.new_label();
    let l_valid = b.new_label();
    let l_invalid = b.new_label();

    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_valid);
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_invalid);

    b.xor_rr32(Reg::Rcx, Reg::Rcx); // i = 0

    b.bind_label(l_loop);
    b.cmp_rr(Reg::Rcx, Reg::Rsi);
    b.jcc_label(Cond::Ae, l_valid);
    b.mov_rr(Reg::R11, Reg::Rdi);
    b.add_rr(Reg::R11, Reg::Rcx);
    b.load_mem(Reg::Rax, Reg::R11, 0, 1, false); // b0

    // 1-byte ASCII: 0x00..=0x7F
    b.cmp_ri32(Reg::Rax, 0x80);
    b.jcc_label(Cond::B, l_step1);

    // 0x80..=0xC1 invalid (continuation or overlong 2-byte lead)
    b.cmp_ri32(Reg::Rax, 0xC2);
    b.jcc_label(Cond::B, l_invalid);

    // 2-byte: 0xC2..=0xDF
    b.cmp_ri32(Reg::Rax, 0xE0);
    b.jcc_label(Cond::Ae, l_chk3);
    b.mov_rr(Reg::Rdx, Reg::Rsi);
    b.sub_rr(Reg::Rdx, Reg::Rcx);
    b.cmp_ri32(Reg::Rdx, 2);
    b.jcc_label(Cond::B, l_invalid);
    b.load_mem(Reg::R8, Reg::R11, 1, 1, false); // b1
    b.and_ri32(Reg::R8, 0xC0);
    b.cmp_ri32(Reg::R8, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 2);
    b.jmp_label(l_loop);

    // 3-byte: 0xE0..=0xEF
    b.bind_label(l_chk3);
    b.cmp_ri32(Reg::Rax, 0xF0);
    b.jcc_label(Cond::Ae, l_chk4);
    b.mov_rr(Reg::Rdx, Reg::Rsi);
    b.sub_rr(Reg::Rdx, Reg::Rcx);
    b.cmp_ri32(Reg::Rdx, 3);
    b.jcc_label(Cond::B, l_invalid);
    b.load_mem(Reg::R8, Reg::R11, 1, 1, false); // b1
    b.load_mem(Reg::R9, Reg::R11, 2, 1, false); // b2
    b.and_ri32(Reg::R9, 0xC0);
    b.cmp_ri32(Reg::R9, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);

    b.cmp_ri32(Reg::Rax, 0xE0);
    b.jcc_label(Cond::Ne, l_chk3_ed);
    // E0: b1 in 0xA0..=0xBF ((b1 & 0xE0) == 0xA0)
    b.and_ri32(Reg::R8, 0xE0);
    b.cmp_ri32(Reg::R8, 0xA0);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 3);
    b.jmp_label(l_loop);

    b.bind_label(l_chk3_ed);
    b.cmp_ri32(Reg::Rax, 0xED);
    b.jcc_label(Cond::Ne, l_chk3_norm);
    // ED: b1 in 0x80..=0x9F ((b1 & 0xE0) == 0x80), rejects UTF-16 surrogates U+D800..U+DFFF
    b.and_ri32(Reg::R8, 0xE0);
    b.cmp_ri32(Reg::R8, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 3);
    b.jmp_label(l_loop);

    b.bind_label(l_chk3_norm);
    // E1..EC, EE..EF: b1 in 0x80..=0xBF
    b.and_ri32(Reg::R8, 0xC0);
    b.cmp_ri32(Reg::R8, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 3);
    b.jmp_label(l_loop);

    // 4-byte: 0xF0..=0xF4
    b.bind_label(l_chk4);
    b.cmp_ri32(Reg::Rax, 0xF5);
    b.jcc_label(Cond::Ae, l_invalid);
    b.mov_rr(Reg::Rdx, Reg::Rsi);
    b.sub_rr(Reg::Rdx, Reg::Rcx);
    b.cmp_ri32(Reg::Rdx, 4);
    b.jcc_label(Cond::B, l_invalid);
    b.load_mem(Reg::R8, Reg::R11, 1, 1, false);  // b1
    b.load_mem(Reg::R9, Reg::R11, 2, 1, false);  // b2
    b.load_mem(Reg::R10, Reg::R11, 3, 1, false); // b3
    b.and_ri32(Reg::R9, 0xC0);
    b.cmp_ri32(Reg::R9, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.and_ri32(Reg::R10, 0xC0);
    b.cmp_ri32(Reg::R10, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);

    b.cmp_ri32(Reg::Rax, 0xF0);
    b.jcc_label(Cond::Ne, l_chk4_f4);
    // F0: b1 in 0x90..=0xBF (rejects overlong 4-byte encodings)
    b.cmp_ri32(Reg::R8, 0x90);
    b.jcc_label(Cond::B, l_invalid);
    b.cmp_ri32(Reg::R8, 0xBF);
    b.jcc_label(Cond::A, l_invalid);
    b.add_ri32(Reg::Rcx, 4);
    b.jmp_label(l_loop);

    b.bind_label(l_chk4_f4);
    b.cmp_ri32(Reg::Rax, 0xF4);
    b.jcc_label(Cond::Ne, l_chk4_norm);
    // F4: b1 in 0x80..=0x8F ((b1 & 0xF0) == 0x80), rejects > U+10FFFF
    b.and_ri32(Reg::R8, 0xF0);
    b.cmp_ri32(Reg::R8, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 4);
    b.jmp_label(l_loop);

    b.bind_label(l_chk4_norm);
    // F1..F3: b1 in 0x80..=0xBF
    b.and_ri32(Reg::R8, 0xC0);
    b.cmp_ri32(Reg::R8, 0x80);
    b.jcc_label(Cond::Ne, l_invalid);
    b.add_ri32(Reg::Rcx, 4);
    b.jmp_label(l_loop);

    b.bind_label(l_step1);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_loop);

    b.bind_label(l_valid);
    b.mov_ri64(Reg::Rax, 1);
    b.ret();

    b.bind_label(l_invalid);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.ret();
    b.finish()
}

/// `u64 eon_rt_is_dir(i64 dirfd [rdi], const u8 *path [rsi], usize len [rdx])`
/// Returns `1` if `path` exists and is a directory (following symlinks), else `0`.
pub fn emit_rt_is_dir() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_copy = b.new_label();
    let l_copy_done = b.new_label();
    let l_false = b.new_label();
    let l_ret = b.new_label();

    b.push(Reg::Rbp);
    b.mov_rr(Reg::Rbp, Reg::Rsp);
    // Reserve 4256B (16B aligned): [rsp..rsp+160) = struct stat (144B), [rsp+160..rsp+4256) = path_buf (4096B).
    b.sub_ri32(Reg::Rsp, 4096);
    b.store_mem(Reg::Rsp, 0, Reg::Rdi, 8); // stack page probe
    b.sub_ri32(Reg::Rsp, 160);

    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_false);
    b.test_rr(Reg::Rdx, Reg::Rdx);
    b.jcc_label(Cond::E, l_false);
    b.mov_ri64(Reg::Rax, 4096);
    b.cmp_rr(Reg::Rdx, Reg::Rax);
    b.jcc_label(Cond::Ae, l_false);

    // Copy path into [rsp + 160], rejecting embedded NUL.
    b.lea_mem(Reg::R8, Reg::Rsp, 160);
    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    b.bind_label(l_copy);
    b.cmp_rr(Reg::Rcx, Reg::Rdx);
    b.jcc_label(Cond::Ae, l_copy_done);
    b.mov_rr(Reg::R9, Reg::Rsi);
    b.add_rr(Reg::R9, Reg::Rcx);
    b.load_mem(Reg::Rax, Reg::R9, 0, 1, false);
    b.test_rr32(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_false);
    b.mov_rr(Reg::R10, Reg::R8);
    b.add_rr(Reg::R10, Reg::Rcx);
    b.store_mem(Reg::R10, 0, Reg::Rax, 1);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_copy);

    b.bind_label(l_copy_done);
    b.mov_rr(Reg::R10, Reg::R8);
    b.add_rr(Reg::R10, Reg::Rdx);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R10, 0, Reg::Rax, 1); // NUL terminator

    // SYS_newfstatat(262)(dirfd=rdi, path=r8, stat_buf=rsp, flags=0)
    b.mov_rr(Reg::Rsi, Reg::R8);
    b.mov_rr(Reg::Rdx, Reg::Rsp);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.mov_ri64(Reg::Rax, 262);
    b.syscall();
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_false);

    // Check (st_mode & S_IFMT) == S_IFDIR (u32 at stat_buf + 24)
    b.load_mem(Reg::Rax, Reg::Rsp, 24, 4, false);
    b.and_ri32(Reg::Rax, 0o170000);
    b.cmp_ri32(Reg::Rax, 0o040000);
    b.jcc_label(Cond::Ne, l_false);
    b.mov_ri64(Reg::Rax, 1);
    b.jmp_label(l_ret);

    b.bind_label(l_false);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.bind_label(l_ret);
    b.mov_rr(Reg::Rsp, Reg::Rbp);
    b.pop(Reg::Rbp);
    b.ret();
    b.finish()
}

/// `i64 eon_rt_mkdir_all(i64 dirfd [rdi], const u8 *path [rsi], usize len [rdx])`
/// Recursively creates `path` and missing parents with mode `0o755`. Returns `0` or `-errno`.
pub fn emit_rt_mkdir_all() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_copy = b.new_label();
    let l_copy_done = b.new_label();
    let l_walk = b.new_label();
    let l_walk_next = b.new_label();
    let l_walk_done = b.new_label();
    let l_eexist = b.new_label();
    let l_inval = b.new_label();
    let l_ret = b.new_label();

    b.push(Reg::Rbp);
    b.mov_rr(Reg::Rbp, Reg::Rsp);
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    // Reserve 4256B: [rsp..rsp+160) = struct stat, [rsp+160..rsp+4256) = path_buf.
    b.sub_ri32(Reg::Rsp, 4096);
    b.store_mem(Reg::Rsp, 0, Reg::Rdi, 8); // stack page probe
    b.sub_ri32(Reg::Rsp, 160);

    b.mov_rr(Reg::Rbx, Reg::Rdi); // dirfd
    b.mov_rr(Reg::R13, Reg::Rdx); // path_len
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_inval);
    b.test_rr(Reg::R13, Reg::R13);
    b.jcc_label(Cond::E, l_inval);
    b.mov_ri64(Reg::Rax, 4096);
    b.cmp_rr(Reg::R13, Reg::Rax);
    b.jcc_label(Cond::Ae, l_inval);

    // Copy path into r12 = [rsp + 160], rejecting embedded NUL.
    b.lea_mem(Reg::R12, Reg::Rsp, 160);
    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    b.bind_label(l_copy);
    b.cmp_rr(Reg::Rcx, Reg::R13);
    b.jcc_label(Cond::Ae, l_copy_done);
    b.mov_rr(Reg::R8, Reg::Rsi);
    b.add_rr(Reg::R8, Reg::Rcx);
    b.load_mem(Reg::Rax, Reg::R8, 0, 1, false);
    b.test_rr32(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_inval);
    b.mov_rr(Reg::R9, Reg::R12);
    b.add_rr(Reg::R9, Reg::Rcx);
    b.store_mem(Reg::R9, 0, Reg::Rax, 1);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_copy);

    b.bind_label(l_copy_done);
    b.mov_rr(Reg::R9, Reg::R12);
    b.add_rr(Reg::R9, Reg::R13);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R9, 0, Reg::Rax, 1); // NUL terminator

    // Walk i = 1..path_len; whenever buf[i] == '/' && buf[i-1] != '/', mkdirat prefix.
    b.mov_ri64(Reg::R14, 1);
    b.bind_label(l_walk);
    b.cmp_rr(Reg::R14, Reg::R13);
    b.jcc_label(Cond::Ae, l_walk_done);
    b.mov_rr(Reg::R8, Reg::R12);
    b.add_rr(Reg::R8, Reg::R14);
    b.load_mem(Reg::Rax, Reg::R8, 0, 1, false);
    b.cmp_ri32(Reg::Rax, b'/' as i32);
    b.jcc_label(Cond::Ne, l_walk_next);
    b.load_mem(Reg::Rax, Reg::R8, -1, 1, false);
    b.cmp_ri32(Reg::Rax, b'/' as i32);
    b.jcc_label(Cond::E, l_walk_next);

    // Temporarily NUL-terminate prefix at buf[i].
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R8, 0, Reg::Rax, 1);
    // SYS_mkdirat(258)(dirfd=rbx, path=r12, mode=0o755)
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.mov_rr(Reg::Rsi, Reg::R12);
    b.mov_ri64(Reg::Rdx, 0o755);
    b.mov_ri64(Reg::Rax, 258);
    b.syscall();
    // Restore '/' at buf[i].
    b.mov_rr(Reg::R8, Reg::R12);
    b.add_rr(Reg::R8, Reg::R14);
    b.mov_ri64(Reg::Rdx, b'/' as u64);
    b.store_mem(Reg::R8, 0, Reg::Rdx, 1);

    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_walk_next);
    b.cmp_ri32(Reg::Rax, -17); // -EEXIST
    b.jcc_label(Cond::E, l_walk_next);
    b.jmp_label(l_ret);

    b.bind_label(l_walk_next);
    b.add_ri32(Reg::R14, 1);
    b.jmp_label(l_walk);

    // Final SYS_mkdirat(258)(dirfd=rbx, path=r12, mode=0o755).
    b.bind_label(l_walk_done);
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.mov_rr(Reg::Rsi, Reg::R12);
    b.mov_ri64(Reg::Rdx, 0o755);
    b.mov_ri64(Reg::Rax, 258);
    b.syscall();
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_ret);
    b.cmp_ri32(Reg::Rax, -17); // -EEXIST
    b.jcc_label(Cond::Ne, l_ret);

    // On -EEXIST, verify existing entry is a directory (matching std::fs::create_dir_all).
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.mov_rr(Reg::Rsi, Reg::R12);
    b.mov_rr(Reg::Rdx, Reg::Rsp);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.mov_ri64(Reg::Rax, 262); // SYS_newfstatat
    b.syscall();
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_eexist);
    b.load_mem(Reg::Rax, Reg::Rsp, 24, 4, false);
    b.and_ri32(Reg::Rax, 0o170000);
    b.cmp_ri32(Reg::Rax, 0o040000);
    b.jcc_label(Cond::Ne, l_eexist);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.jmp_label(l_ret);

    b.bind_label(l_eexist);
    b.mov_ri64(Reg::Rax, (-17i64) as u64);
    b.jmp_label(l_ret);

    b.bind_label(l_inval);
    b.mov_ri64(Reg::Rax, (-22i64) as u64);
    b.bind_label(l_ret);
    b.lea_mem(Reg::Rsp, Reg::Rbp, -32);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.pop(Reg::Rbp);
    b.ret();
    b.finish()
}

/// `i64 eon_rt_spawn_wait(const u8 *path [rdi], usize path_len [rsi], const u8 *args_nul [rdx], usize args_len [rcx])`
/// Forks, executes `path` with NUL-delimited `args_nul` (or `[path, NULL]` if empty) and inherited
/// `__eon_envp` (or `[NULL]`), waits for child completion, and returns exit code (`>= 0`) or `-errno`.
pub fn emit_rt_spawn_wait() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_copy_path = b.new_label();
    let l_copy_path_done = b.new_label();
    let l_no_args = b.new_label();
    let l_probe_lp = b.new_label();
    let l_probe_done = b.new_label();
    let l_arg_lp = b.new_label();
    let l_skip_nul = b.new_label();
    let l_arg_done = b.new_label();
    let l_have_argv = b.new_label();
    let l_have_envp = b.new_label();
    let l_child = b.new_label();
    let l_wait = b.new_label();
    let l_sig = b.new_label();
    let l_inval = b.new_label();
    let l_ret = b.new_label();

    b.push(Reg::Rbp);
    b.mov_rr(Reg::Rbp, Reg::Rsp);
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);
    b.sub_ri32(Reg::Rsp, 8); // align to rbp - 48
    // Fixed stack frame:
    //   [rbp - 56]           = wstatus (8B)
    //   [rbp - 64]           = null_envp (8B, zero)
    //   [rbp - 2128..-64]    = argv[0..=256] (2064B)
    //   [rbp - 6224..-2128]  = path_buf (4096B)
    b.sub_ri32(Reg::Rsp, 4096);
    b.store_mem(Reg::Rsp, 0, Reg::Rdi, 8);
    b.sub_ri32(Reg::Rsp, 2080);
    b.store_mem(Reg::Rsp, 0, Reg::Rdi, 8);

    b.mov_rr(Reg::Rbx, Reg::Rdx); // args_ptr
    b.mov_rr(Reg::R14, Reg::Rcx); // args_len
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rbp, -56, Reg::Rax, 8); // wstatus = 0
    b.store_mem(Reg::Rbp, -64, Reg::Rax, 8); // null_envp = NULL

    // Validate path: non-null, 1..=4095, no embedded NUL.
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_inval);
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_inval);
    b.mov_ri64(Reg::Rax, 4096);
    b.cmp_rr(Reg::Rsi, Reg::Rax);
    b.jcc_label(Cond::Ae, l_inval);

    b.lea_mem(Reg::R8, Reg::Rbp, -6224); // path_buf
    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    b.bind_label(l_copy_path);
    b.cmp_rr(Reg::Rcx, Reg::Rsi);
    b.jcc_label(Cond::Ae, l_copy_path_done);
    b.mov_rr(Reg::R9, Reg::Rdi);
    b.add_rr(Reg::R9, Reg::Rcx);
    b.load_mem(Reg::Rax, Reg::R9, 0, 1, false);
    b.test_rr32(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_inval);
    b.mov_rr(Reg::R10, Reg::R8);
    b.add_rr(Reg::R10, Reg::Rcx);
    b.store_mem(Reg::R10, 0, Reg::Rax, 1);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_copy_path);

    b.bind_label(l_copy_path_done);
    b.mov_rr(Reg::R10, Reg::R8);
    b.add_rr(Reg::R10, Reg::Rsi);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R10, 0, Reg::Rax, 1); // NUL-terminate path_buf

    // Build argv table at r12 = rbp - 2128; r13 = argc.
    b.lea_mem(Reg::R12, Reg::Rbp, -2128);
    b.xor_rr32(Reg::R13, Reg::R13);

    b.test_rr(Reg::R14, Reg::R14);
    b.jcc_label(Cond::E, l_no_args);
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_inval);
    b.mov_ri64(Reg::Rax, 1_048_576);
    b.cmp_rr(Reg::R14, Reg::Rax);
    b.jcc_label(Cond::Ae, l_inval);

    // Dynamically reserve align_up(args_len + 1, 16) bytes on stack with 4KiB page probes.
    b.lea_mem(Reg::R8, Reg::R14, 16);
    b.and_ri32(Reg::R8, !15);
    b.bind_label(l_probe_lp);
    b.cmp_ri32(Reg::R8, 4096);
    b.jcc_label(Cond::Be, l_probe_done);
    b.sub_ri32(Reg::Rsp, 4096);
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 8);
    b.sub_ri32(Reg::R8, 4096);
    b.jmp_label(l_probe_lp);
    b.bind_label(l_probe_done);
    b.sub_rr(Reg::Rsp, Reg::R8);
    b.store_mem(Reg::Rsp, 0, Reg::Rax, 8);

    // Copy args_nul into [rsp] and append a safety NUL at [rsp + args_len].
    b.mov_rr(Reg::Rdi, Reg::Rsp);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_rr(Reg::Rcx, Reg::R14);
    b.rep_movsb();
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rdi, 0, Reg::Rax, 1);

    // Scan NUL-terminated entries in args_copy[0..args_len].
    b.xor_rr32(Reg::R8, Reg::R8); // i = 0
    b.bind_label(l_arg_lp);
    b.cmp_rr(Reg::R8, Reg::R14);
    b.jcc_label(Cond::Ae, l_arg_done);
    b.cmp_ri32(Reg::R13, 256);
    b.jcc_label(Cond::Ae, l_inval);
    b.mov_rr(Reg::Rax, Reg::Rsp);
    b.add_rr(Reg::Rax, Reg::R8); // &args_copy[i]
    b.mov_rr(Reg::R9, Reg::R13);
    b.shl_imm(Reg::R9, 3);
    b.add_rr(Reg::R9, Reg::R12);
    b.store_mem(Reg::R9, 0, Reg::Rax, 8); // argv[argc] = &args_copy[i]
    b.add_ri32(Reg::R13, 1);

    b.bind_label(l_skip_nul);
    b.mov_rr(Reg::Rax, Reg::Rsp);
    b.add_rr(Reg::Rax, Reg::R8);
    b.load_mem(Reg::Rcx, Reg::Rax, 0, 1, false);
    b.add_ri32(Reg::R8, 1);
    b.test_rr32(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::Ne, l_skip_nul);
    b.jmp_label(l_arg_lp);

    b.bind_label(l_arg_done);
    b.bind_label(l_no_args);
    b.test_rr(Reg::R13, Reg::R13);
    b.jcc_label(Cond::Ne, l_have_argv);
    b.lea_mem(Reg::Rax, Reg::Rbp, -6224); // default argv[0] = path_nul
    b.store_mem(Reg::R12, 0, Reg::Rax, 8);
    b.mov_ri64(Reg::R13, 1);

    b.bind_label(l_have_argv);
    b.mov_rr(Reg::R9, Reg::R13);
    b.shl_imm(Reg::R9, 3);
    b.add_rr(Reg::R9, Reg::R12);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R9, 0, Reg::Rax, 8); // argv[argc] = NULL

    // Load inherited envp from __eon_envp, falling back to [NULL].
    b.mov_rip_load(Reg::R15, "__eon_envp", 0);
    b.test_rr(Reg::R15, Reg::R15);
    b.jcc_label(Cond::Ne, l_have_envp);
    b.lea_mem(Reg::R15, Reg::Rbp, -64);
    b.bind_label(l_have_envp);

    // SYS_fork(57)
    b.mov_ri64(Reg::Rax, 57);
    b.syscall();
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::L, l_ret);
    b.jcc_label(Cond::E, l_child);

    // Parent: SYS_wait4(61)(child_pid, &wstatus, 0, NULL) retrying on -EINTR (-4).
    b.mov_rr(Reg::Rbx, Reg::Rax);
    b.bind_label(l_wait);
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.lea_mem(Reg::Rsi, Reg::Rbp, -56);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.mov_ri64(Reg::Rax, 61);
    b.syscall();
    b.cmp_ri32(Reg::Rax, -4); // -EINTR
    b.jcc_label(Cond::E, l_wait);
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::L, l_ret);

    // Decode wstatus: if (wstatus & 0x7F) == 0 -> (wstatus >> 8) & 0xFF, else 128 + (wstatus & 0x7F).
    b.load_mem(Reg::Rax, Reg::Rbp, -56, 4, false);
    b.mov_rr(Reg::Rdx, Reg::Rax);
    b.and_ri32(Reg::Rdx, 0x7F);
    b.test_rr(Reg::Rdx, Reg::Rdx);
    b.jcc_label(Cond::Ne, l_sig);
    b.shr_imm(Reg::Rax, 8);
    b.and_ri32(Reg::Rax, 0xFF);
    b.jmp_label(l_ret);

    b.bind_label(l_sig);
    b.lea_mem(Reg::Rax, Reg::Rdx, 128);
    b.jmp_label(l_ret);

    // Child: SYS_execve(59)(path_nul, argv, envp); on failure exit(127).
    b.bind_label(l_child);
    b.lea_mem(Reg::Rdi, Reg::Rbp, -6224);
    b.mov_rr(Reg::Rsi, Reg::R12);
    b.mov_rr(Reg::Rdx, Reg::R15);
    b.mov_ri64(Reg::Rax, 59);
    b.syscall();
    b.mov_ri64(Reg::Rdi, 127);
    b.mov_ri64(Reg::Rax, 231); // SYS_exit_group
    b.syscall();
    b.ud2();

    b.bind_label(l_inval);
    b.mov_ri64(Reg::Rax, (-22i64) as u64);
    b.bind_label(l_ret);
    b.lea_mem(Reg::Rsp, Reg::Rbp, -40);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.pop(Reg::Rbp);
    b.ret();
    b.finish()
}

/// `void eon_rt_print_u64(u64 val [rdi])`
pub fn emit_rt_print_u64() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 32);
    b.lea_mem(Reg::Rsi, Reg::Rsp, 32); // cursor at end
    b.mov_rr(Reg::Rax, Reg::Rdi);
    b.mov_ri64(Reg::Rcx, 10);

    let l_loop = b.new_label();
    b.bind_label(l_loop);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.div_r(Reg::Rcx); // rax = q, rdx = r
    b.add_ri32(Reg::Rdx, b'0' as i32);
    b.sub_ri32(Reg::Rsi, 1);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_loop);

    // count = (rsp + 32) - rsi
    b.lea_mem(Reg::Rdx, Reg::Rsp, 32);
    b.sub_rr(Reg::Rdx, Reg::Rsi);
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_ri64(Reg::Rax, 1); // SYS_write
    b.syscall();
    b.add_ri32(Reg::Rsp, 32);
    b.ret();
    b.finish()
}

/// `void eon_rt_print_i64(i64 val [rdi])`
pub fn emit_rt_print_i64() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 32);
    b.lea_mem(Reg::Rsi, Reg::Rsp, 32);
    b.xor_rr32(Reg::R8, Reg::R8); // is_neg = 0
    b.mov_rr(Reg::Rax, Reg::Rdi);
    let l_pos = b.new_label();
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::Ge, l_pos);
    b.mov_ri64(Reg::R8, 1);
    b.neg_r(Reg::Rax); // handles i64::MIN too as unsigned 2^63!
    b.bind_label(l_pos);

    b.mov_ri64(Reg::Rcx, 10);
    let l_loop = b.new_label();
    b.bind_label(l_loop);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.div_r(Reg::Rcx);
    b.add_ri32(Reg::Rdx, b'0' as i32);
    b.sub_ri32(Reg::Rsi, 1);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_loop);

    let l_write = b.new_label();
    b.test_rr(Reg::R8, Reg::R8);
    b.jcc_label(Cond::E, l_write);
    b.sub_ri32(Reg::Rsi, 1);
    b.mov_ri64(Reg::Rdx, b'-' as u64);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);

    b.bind_label(l_write);
    b.lea_mem(Reg::Rdx, Reg::Rsp, 32);
    b.sub_rr(Reg::Rdx, Reg::Rsi);
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_ri64(Reg::Rax, 1);
    b.syscall();
    b.add_ri32(Reg::Rsp, 32);
    b.ret();
    b.finish()
}

/// `void eon_rt_print_f32(f32 val [xmm0])` — prints `%.2f` matching C11 oracle!
pub fn emit_rt_print_f32() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.sub_ri32(Reg::Rsp, 40);
    // Convert f32 in xmm0 -> f64 in xmm0, multiply by 100.0, round to nearest int64!
    b.cvtss2sd(Xmm::Xmm0, Xmm::Xmm0);
    b.mov_ri64(Reg::Rax, 100.0f64.to_bits());
    b.movq_xmm_r64(Xmm::Xmm1, Reg::Rax);
    b.sse_bin(true, 0x59, Xmm::Xmm0, Xmm::Xmm1); // mulsd xmm0, 100.0
    // Add +/- 0.5 for rounding
    b.mov_ri64(Reg::Rax, 0.5f64.to_bits());
    b.movq_xmm_r64(Xmm::Xmm1, Reg::Rax);
    b.mov_ri64(Reg::Rax, 0);
    b.movq_xmm_r64(Xmm::Xmm2, Reg::Rax);
    b.ucomi(true, Xmm::Xmm0, Xmm::Xmm2);
    let l_neg = b.new_label();
    let l_cvt = b.new_label();
    b.jcc_label(Cond::B, l_neg);
    b.sse_bin(true, 0x58, Xmm::Xmm0, Xmm::Xmm1); // +0.5
    b.jmp_label(l_cvt);
    b.bind_label(l_neg);
    b.sse_bin(true, 0x5C, Xmm::Xmm0, Xmm::Xmm1); // -0.5
    b.bind_label(l_cvt);
    b.cvttf2si(true, Reg::Rax, Xmm::Xmm0); // scaled = round(v * 100)

    b.xor_rr32(Reg::R8, Reg::R8);
    let l_pos = b.new_label();
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::Ge, l_pos);
    b.mov_ri64(Reg::R8, 1);
    b.neg_r(Reg::Rax);
    b.bind_label(l_pos);

    b.lea_mem(Reg::Rsi, Reg::Rsp, 32);
    b.mov_ri64(Reg::Rcx, 10);
    // 2 fractional digits
    for _ in 0..2 {
        b.xor_rr32(Reg::Rdx, Reg::Rdx);
        b.div_r(Reg::Rcx);
        b.add_ri32(Reg::Rdx, b'0' as i32);
        b.sub_ri32(Reg::Rsi, 1);
        b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);
    }
    b.sub_ri32(Reg::Rsi, 1);
    b.mov_ri64(Reg::Rdx, b'.' as u64);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);
    // Integer part
    let l_loop = b.new_label();
    b.bind_label(l_loop);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.div_r(Reg::Rcx);
    b.add_ri32(Reg::Rdx, b'0' as i32);
    b.sub_ri32(Reg::Rsi, 1);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_loop);

    let l_write = b.new_label();
    b.test_rr(Reg::R8, Reg::R8);
    b.jcc_label(Cond::E, l_write);
    b.sub_ri32(Reg::Rsi, 1);
    b.mov_ri64(Reg::Rdx, b'-' as u64);
    b.store_mem(Reg::Rsi, 0, Reg::Rdx, 1);

    b.bind_label(l_write);
    b.lea_mem(Reg::Rdx, Reg::Rsp, 32);
    b.sub_rr(Reg::Rdx, Reg::Rsi);
    b.mov_ri64(Reg::Rdi, 1);
    b.mov_ri64(Reg::Rax, 1);
    b.syscall();
    b.add_ri32(Reg::Rsp, 40);
    b.ret();
    b.finish()
}

/// `f32 eon_f16_to_f32(u16 h in rdi)` -> IEEE-754 binary16 -> binary32 in `xmm0`.
/// Exact bit-for-bit match with `eir_interp::f16_bits_to_f32`.
pub fn emit_rt_f16_to_f32() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    // rdx = sign = ((rdi >> 15) & 1) << 31
    b.mov_rr(Reg::Rdx, Reg::Rdi);
    b.shr_imm(Reg::Rdx, 15);
    b.and_ri32(Reg::Rdx, 1);
    b.shl_imm(Reg::Rdx, 31);
    // rcx = exp = (rdi >> 10) & 0x1F
    b.mov_rr(Reg::Rcx, Reg::Rdi);
    b.shr_imm(Reg::Rcx, 10);
    b.and_ri32(Reg::Rcx, 0x1F);
    // rax = frac = rdi & 0x03FF
    b.mov_rr(Reg::Rax, Reg::Rdi);
    b.and_ri32(Reg::Rax, 0x03FF);

    let l_exp_nonzero = b.new_label();
    let l_subnormal = b.new_label();
    let l_sub_loop = b.new_label();
    let l_exp_inf = b.new_label();
    let l_done = b.new_label();

    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::Ne, l_exp_nonzero);
    // exp == 0
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_subnormal);
    // +-0.0
    b.mov_rr(Reg::Rax, Reg::Rdx);
    b.jmp_label(l_done);

    // Subnormal binary16 -> normal f32
    b.bind_label(l_subnormal);
    b.mov_ri64(Reg::Rsi, 113); // 127 - 14
    b.bind_label(l_sub_loop);
    b.mov_rr(Reg::Rdi, Reg::Rax);
    b.and_ri32(Reg::Rdi, 0x0400);
    let l_sub_done = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::Ne, l_sub_done);
    b.shl_imm(Reg::Rax, 1);
    b.sub_ri32(Reg::Rsi, 1);
    b.jmp_label(l_sub_loop);
    b.bind_label(l_sub_done);
    b.and_ri32(Reg::Rax, 0x03FF);
    b.shl_imm(Reg::Rax, 13);
    b.shl_imm(Reg::Rsi, 23);
    b.or_rr(Reg::Rax, Reg::Rsi);
    b.or_rr(Reg::Rax, Reg::Rdx);
    b.jmp_label(l_done);

    b.bind_label(l_exp_nonzero);
    b.cmp_ri32(Reg::Rcx, 31);
    b.jcc_label(Cond::E, l_exp_inf);
    // Normal: sign | ((exp + 112) << 23) | (frac << 13)
    b.add_ri32(Reg::Rcx, 112);
    b.shl_imm(Reg::Rcx, 23);
    b.shl_imm(Reg::Rax, 13);
    b.or_rr(Reg::Rax, Reg::Rcx);
    b.or_rr(Reg::Rax, Reg::Rdx);
    b.jmp_label(l_done);

    // Inf / NaN: sign | (0xFF << 23) | (frac << 13)
    b.bind_label(l_exp_inf);
    b.shl_imm(Reg::Rax, 13);
    b.mov_ri64(Reg::Rcx, 0xFFu64 << 23);
    b.or_rr(Reg::Rax, Reg::Rcx);
    b.or_rr(Reg::Rax, Reg::Rdx);

    b.bind_label(l_done);
    b.movd_xmm_r32(Xmm::Xmm0, Reg::Rax);
    b.ret();
    b.finish()
}

/// `u16 eon_f32_to_f16(f32 v in xmm0)` -> IEEE-754 binary32 -> binary16 (round-to-nearest-even) in `rax`.
/// Exact bit-for-bit match with `eir_interp::f32_to_f16_bits`.
pub fn emit_rt_f32_to_f16() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.movd_r32_xmm(Reg::Rdi, Xmm::Xmm0);
    // rdx = sign = (bits >> 16) & 0x8000
    b.mov_rr(Reg::Rdx, Reg::Rdi);
    b.shr_imm(Reg::Rdx, 16);
    b.and_ri32(Reg::Rdx, 0x8000);
    // rcx = exp = (bits >> 23) & 0xFF
    b.mov_rr(Reg::Rcx, Reg::Rdi);
    b.shr_imm(Reg::Rcx, 23);
    b.and_ri32(Reg::Rcx, 0xFF);
    // rsi = frac = bits & 0x007F_FFFF
    b.mov_rr(Reg::Rsi, Reg::Rdi);
    b.and_ri32(Reg::Rsi, 0x007F_FFFF);

    let l_not_inf = b.new_label();
    let l_inf_zero_frac = b.new_label();
    let l_ovf = b.new_label();
    let l_und = b.new_label();
    let l_sub = b.new_label();
    let l_norm_no_round = b.new_label();
    let l_norm_round = b.new_label();

    // if exp == 0xFF
    b.cmp_ri32(Reg::Rcx, 0xFF);
    b.jcc_label(Cond::Ne, l_not_inf);
    b.mov_rr(Reg::Rax, Reg::Rdx);
    b.or_ri32(Reg::Rax, 0x7C00);
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_inf_zero_frac);
    b.shr_imm(Reg::Rsi, 13);
    let l_nan_nz = b.new_label();
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::Ne, l_nan_nz);
    b.mov_ri64(Reg::Rsi, 1);
    b.bind_label(l_nan_nz);
    b.or_rr(Reg::Rax, Reg::Rsi);
    b.bind_label(l_inf_zero_frac);
    b.ret();

    // unbiased = exp - 127
    b.bind_label(l_not_inf);
    b.sub_ri32(Reg::Rcx, 127);
    b.cmp_ri32(Reg::Rcx, 15);
    b.jcc_label(Cond::G, l_ovf);
    b.cmp_ri32(Reg::Rcx, -24);
    b.jcc_label(Cond::L, l_und);
    b.cmp_ri32(Reg::Rcx, -14);
    b.jcc_label(Cond::L, l_sub);

    // Normal: h_exp = (unbiased + 15) << 10, mant = frac >> 13, rem = frac & 0x1FFF
    b.add_ri32(Reg::Rcx, 15);
    b.shl_imm(Reg::Rcx, 10);
    b.mov_rr(Reg::Rax, Reg::Rsi);
    b.shr_imm(Reg::Rax, 13); // mant
    b.and_ri32(Reg::Rsi, 0x1FFF); // rem
    b.cmp_ri32(Reg::Rsi, 0x1000);
    b.jcc_label(Cond::G, l_norm_round);
    b.jcc_label(Cond::L, l_norm_no_round);
    // rem == 0x1000 -> round to even if (mant & 1) != 0
    b.mov_rr(Reg::Rdi, Reg::Rax);
    b.and_ri32(Reg::Rdi, 1);
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_norm_no_round);
    b.bind_label(l_norm_round);
    b.add_ri32(Reg::Rax, 1);
    b.bind_label(l_norm_no_round);
    b.add_rr(Reg::Rax, Reg::Rcx);
    b.or_rr(Reg::Rax, Reg::Rdx);
    b.and_ri32(Reg::Rax, 0xFFFF);
    b.ret();

    // Subnormal: full = frac | 0x0080_0000; shift = (-14 - unbiased) + 13 = -1 - unbiased
    b.bind_label(l_sub);
    b.or_ri32(Reg::Rsi, 0x0080_0000); // full
    b.neg_r(Reg::Rcx);
    b.sub_ri32(Reg::Rcx, 1); // shift in rcx (14..24)
    b.mov_rr(Reg::Rax, Reg::Rsi);
    b.shr_cl(Reg::Rax); // mant = full >> shift
    // half = 1 << (shift - 1), mask = (1 << shift) - 1
    b.mov_ri64(Reg::R8, 1);
    b.shl_cl(Reg::R8); // 1 << shift
    b.mov_rr(Reg::R9, Reg::R8);
    b.shr_imm(Reg::R9, 1); // half = 1 << (shift - 1)
    b.sub_ri32(Reg::R8, 1); // mask
    b.and_rr(Reg::Rsi, Reg::R8); // rem = full & mask
    let l_sub_no_round = b.new_label();
    let l_sub_round = b.new_label();
    b.cmp_rr(Reg::Rsi, Reg::R9);
    b.jcc_label(Cond::A, l_sub_round);
    b.jcc_label(Cond::B, l_sub_no_round);
    b.mov_rr(Reg::Rdi, Reg::Rax);
    b.and_ri32(Reg::Rdi, 1);
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_sub_no_round);
    b.bind_label(l_sub_round);
    b.add_ri32(Reg::Rax, 1);
    b.bind_label(l_sub_no_round);
    b.or_rr(Reg::Rax, Reg::Rdx);
    b.and_ri32(Reg::Rax, 0xFFFF);
    b.ret();

    b.bind_label(l_ovf);
    b.mov_rr(Reg::Rax, Reg::Rdx);
    b.or_ri32(Reg::Rax, 0x7C00);
    b.ret();

    b.bind_label(l_und);
    b.mov_rr(Reg::Rax, Reg::Rdx);
    b.ret();

    b.finish()
}

// --- Multi-core scheduler (`par for` + DAG `Schedule`), Static Fibers & `io_uring` ---

/// Shared `par_for` chunk-stealing loop used by both caller and worker threads.
/// Assumes `sched_reg` holds `&__eon_sched_state` and `rsp % 16 == 0` before `call`.
/// Preserves `sched_reg`, `Rbx`, `R13`; clobbers `R14, R15` + caller-saved.
fn emit_par_for_work_loop(b: &mut X64Buf, sched_reg: Reg) {
    let l_loop = b.new_label();
    let l_done = b.new_label();
    let l_no_ovf = b.new_label();
    b.load_mem(Reg::R14, sched_reg, SCHED_OFF_TASK_FN, 8, false);
    b.load_mem(Reg::R15, sched_reg, SCHED_OFF_TASK_CTX, 8, false);

    b.bind_label(l_loop);
    b.load_mem(Reg::Rax, sched_reg, SCHED_OFF_TASK_CHUNK, 8, false);
    b.lock_xadd_disp(sched_reg, SCHED_OFF_TASK_NEXT, Reg::Rax, 8);
    b.load_mem(Reg::Rdx, sched_reg, SCHED_OFF_TASK_TOTAL, 8, false);
    b.cmp_rr(Reg::Rax, Reg::Rdx);
    b.jcc_label(Cond::Ae, l_done);

    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.load_mem(Reg::Rcx, sched_reg, SCHED_OFF_TASK_CHUNK, 8, false);
    b.add_rr(Reg::Rsi, Reg::Rcx);
    b.jcc_label(Cond::Ae, l_no_ovf); // no carry -> didn't wrap
    b.mov_rr(Reg::Rsi, Reg::Rdx);
    b.bind_label(l_no_ovf);
    b.cmp_rr(Reg::Rsi, Reg::Rdx);
    b.cmovcc(Cond::A, Reg::Rsi, Reg::Rdx);

    b.mov_rr(Reg::Rdi, Reg::Rax);
    b.mov_rr(Reg::Rdx, Reg::R15);
    b.call_reg(Reg::R14);
    b.jmp_label(l_loop);

    b.bind_label(l_done);
}

/// Shared DAG execution loop over `EonDagNode` (32 B: `[0]=fn, [8]=arg, [16]=init_deps:u32,
/// [20]=num_succs:u32, [24]=succs_ptr`) and runtime slots `{ rem_deps: i64, claimed: u64 }`.
/// Assumes `sched_reg` holds `&__eon_sched_state` and `rsp % 16 == 0`.
/// Preserves `sched_reg` and `Rbx`/`R13`.
fn emit_dag_work_loop(b: &mut X64Buf, sched_reg: Reg) {
    let l_outer = b.new_label();
    let l_scan = b.new_label();
    let l_next_i = b.new_label();
    let l_pause = b.new_label();
    let l_done = b.new_label();

    b.push(Reg::R13);
    b.push(Reg::Rbx);
    b.load_mem(Reg::R13, sched_reg, SCHED_OFF_DAG_NODES, 8, false);
    b.load_mem(Reg::R14, sched_reg, SCHED_OFF_DAG_RT, 8, false);
    b.load_mem(Reg::R15, sched_reg, SCHED_OFF_DAG_COUNT, 8, false);

    b.bind_label(l_outer);
    b.load_mem(Reg::Rax, sched_reg, SCHED_OFF_TASK_DONE, 8, false);
    b.cmp_rr(Reg::Rax, Reg::R15);
    b.jcc_label(Cond::Ae, l_done);

    b.xor_rr32(Reg::Rbx, Reg::Rbx);
    b.bind_label(l_scan);
    b.cmp_rr(Reg::Rbx, Reg::R15);
    b.jcc_label(Cond::Ae, l_pause);

    b.mov_rr(Reg::R8, Reg::Rbx);
    b.shl_imm(Reg::R8, 4);
    b.add_rr(Reg::R8, Reg::R14); // rt = &dag_rt[i]
    b.load_mem(Reg::Rax, Reg::R8, 0, 8, false); // rem_deps
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_next_i);
    b.load_mem(Reg::Rax, Reg::R8, 8, 8, false); // claimed
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::Ne, l_next_i);

    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_ri64(Reg::Rcx, 1);
    b.lock_cmpxchg_disp(Reg::R8, 8, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_next_i);

    // Claimed node i: invoke node.fn_ptr(arg_ptr != 0 ? arg_ptr : task_ctx)
    b.mov_rr(Reg::R9, Reg::Rbx);
    b.shl_imm(Reg::R9, 5);
    b.add_rr(Reg::R9, Reg::R13);
    b.load_mem(Reg::Rax, Reg::R9, 0, 8, false);
    b.load_mem(Reg::Rdi, Reg::R9, 8, 8, false);
    let l_have_arg = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::Ne, l_have_arg);
    b.load_mem(Reg::Rdi, sched_reg, SCHED_OFF_TASK_CTX, 8, false);
    b.bind_label(l_have_arg);
    b.call_reg(Reg::Rax);

    // Decrement rem_deps on all successors
    b.mov_rr(Reg::R9, Reg::Rbx);
    b.shl_imm(Reg::R9, 5);
    b.add_rr(Reg::R9, Reg::R13);
    b.load_mem(Reg::Rcx, Reg::R9, 20, 4, false); // num_succs
    let l_mark_done = b.new_label();
    b.test_rr(Reg::Rcx, Reg::Rcx);
    b.jcc_label(Cond::E, l_mark_done);
    b.load_mem(Reg::Rsi, Reg::R9, 24, 8, false); // succs_ptr
    let l_succ_lp = b.new_label();
    b.bind_label(l_succ_lp);
    b.load_mem(Reg::Rdx, Reg::Rsi, 0, 4, false);
    b.shl_imm(Reg::Rdx, 4);
    b.add_rr(Reg::Rdx, Reg::R14);
    b.mov_ri64(Reg::Rax, (-1i64) as u64);
    b.lock_xadd_disp(Reg::Rdx, 0, Reg::Rax, 8);
    b.add_ri32(Reg::Rsi, 4);
    b.sub_ri32(Reg::Rcx, 1);
    b.jcc_label(Cond::Ne, l_succ_lp);

    b.bind_label(l_mark_done);
    b.mov_ri64(Reg::Rax, 1);
    b.lock_xadd_disp(sched_reg, SCHED_OFF_TASK_DONE, Reg::Rax, 8);
    b.jmp_label(l_outer);

    b.bind_label(l_next_i);
    b.add_ri32(Reg::Rbx, 1);
    b.jmp_label(l_scan);

    b.bind_label(l_pause);
    b.pause();
    b.jmp_label(l_outer);

    b.bind_label(l_done);
    b.pop(Reg::Rbx);
    b.pop(Reg::R13);
}

/// `void eon_sched_init(void)`
/// Detects online CPUs via `SYS_sched_getaffinity`, spawns 1:1 pinned worker threads
/// via `SYS_clone(..., CLONE_SETTLS)` + `SYS_sched_setaffinity`, parked on `SYS_futex`.
pub fn emit_rt_sched_init() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret_fast = b.new_label();
    let l_worker_entry = b.new_label();

    b.lea_rip_sym(Reg::R8, "__eon_sched_state", 0);
    b.cmp_mi32(Reg::R8, SCHED_OFF_INIT, 2, 8);
    b.jcc_label(Cond::E, l_ret_fast);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::R12, Reg::R8);
    // Try claiming init_done: 0 -> 1
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_ri64(Reg::Rcx, 1);
    b.lock_cmpxchg_disp(Reg::R12, SCHED_OFF_INIT, Reg::Rcx, 8);
    let l_do_init = b.new_label();
    let l_done = b.new_label();
    b.jcc_label(Cond::E, l_do_init);

    // Another thread is initializing; wait until init_done == 2
    let l_wait_ready = b.new_label();
    b.bind_label(l_wait_ready);
    b.pause();
    b.cmp_mi32(Reg::R12, SCHED_OFF_INIT, 2, 8);
    b.jcc_label(Cond::Ne, l_wait_ready);
    b.jmp_label(l_done);

    b.bind_label(l_do_init);
    // Ensure main TLS is ready before workers launch
    emit_load_tcb(&mut b, Reg::Rax);

    // Query online CPU affinity mask: SYS_sched_getaffinity(204, 0, 128, &mask)
    b.lea_rip_sym(Reg::R13, "__eon_cpu_affinity_mask", 0);
    b.mov_ri64(Reg::Rax, 204);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, 128);
    b.mov_rr(Reg::Rdx, Reg::R13);
    b.syscall();

    b.xor_rr32(Reg::R14, Reg::R14); // num_cpus = 0
    let l_cpu_fallback = b.new_label();
    let l_cpus_ready = b.new_label();
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::Le, l_cpu_fallback);

    // Scan up to 1024 bits (128 bytes) for up to 8 online CPUs
    b.xor_rr32(Reg::Rcx, Reg::Rcx); // cpu bit index 0..1024
    let l_cpu_scan = b.new_label();
    let l_cpu_next = b.new_label();
    b.bind_label(l_cpu_scan);
    b.cmp_ri32(Reg::Rcx, 1024);
    b.jcc_label(Cond::Ae, l_cpus_ready);
    b.cmp_ri32(Reg::R14, SCHED_MAX_WORKERS as i32);
    b.jcc_label(Cond::Ae, l_cpus_ready);
    b.mov_rr(Reg::Rax, Reg::Rcx);
    b.shr_imm(Reg::Rax, 6);
    b.shl_imm(Reg::Rax, 3);
    b.add_rr(Reg::Rax, Reg::R13);
    b.load_mem(Reg::Rdx, Reg::Rax, 0, 8, false);
    b.mov_rr(Reg::R8, Reg::Rcx);
    b.and_ri32(Reg::R8, 63);
    // Shift right by R8 using Cl (save rcx in r9)
    b.mov_rr(Reg::R9, Reg::Rcx);
    b.mov_rr(Reg::Rcx, Reg::R8);
    b.shr_cl(Reg::Rdx);
    b.mov_rr(Reg::Rcx, Reg::R9);
    b.and_ri32(Reg::Rdx, 1);
    b.jcc_label(Cond::E, l_cpu_next);
    // Record online_cpus[r14] = cpu_bit
    b.mov_rr(Reg::Rax, Reg::R14);
    b.shl_imm(Reg::Rax, 3);
    b.lea_mem(Reg::R8, Reg::R12, SCHED_OFF_ONLINE_CPUS);
    b.add_rr(Reg::R8, Reg::Rax);
    b.store_mem(Reg::R8, 0, Reg::Rcx, 8);
    b.add_ri32(Reg::R14, 1);
    b.bind_label(l_cpu_next);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_cpu_scan);

    b.bind_label(l_cpus_ready);
    b.test_rr(Reg::R14, Reg::R14);
    let l_spawn_start = b.new_label();
    b.jcc_label(Cond::Ne, l_spawn_start);
    b.bind_label(l_cpu_fallback);
    b.mov_ri64(Reg::R14, 1);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R12, SCHED_OFF_ONLINE_CPUS, Reg::Rax, 8);

    // Spawn w = 0 .. R14 workers
    b.bind_label(l_spawn_start);
    b.xor_rr32(Reg::Rbx, Reg::Rbx); // w = 0
    let l_spawn_lp = b.new_label();
    let l_spawn_done = b.new_label();
    b.bind_label(l_spawn_lp);
    b.cmp_rr(Reg::Rbx, Reg::R14);
    b.jcc_label(Cond::Ae, l_spawn_done);

    // worker TCB = sched + SCHED_OFF_WORKER_TCBS + w * 128
    b.mov_rr(Reg::R15, Reg::Rbx);
    b.shl_imm(Reg::R15, 7);
    b.lea_mem(Reg::R8, Reg::R12, SCHED_OFF_WORKER_TCBS);
    b.add_rr(Reg::R15, Reg::R8);
    b.store_mem(Reg::R15, 0, Reg::R15, 8); // self_ptr
    b.store_mem(Reg::R15, 8, Reg::Rbx, 8); // worker_id
    b.mov_rr(Reg::Rax, Reg::Rbx);
    b.shl_imm(Reg::Rax, 3);
    b.lea_mem(Reg::R8, Reg::R12, SCHED_OFF_ONLINE_CPUS);
    b.add_rr(Reg::R8, Reg::Rax);
    b.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
    b.store_mem(Reg::R15, 80, Reg::Rax, 8); // cpu_id

    // Allocate 64 KiB stack for worker
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, WORKER_STACK_BYTES);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.jcc_label(Cond::Ae, l_spawn_done);

    b.store_mem(Reg::R15, 88, Reg::Rax, 8); // stack_base
    b.mov_ri64(Reg::Rcx, WORKER_STACK_BYTES - 16);
    b.add_rr(Reg::Rax, Reg::Rcx); // child_rsp (16B aligned)

    // active_workers += 1 before clone so shutdown cannot race
    b.mov_ri64(Reg::Rcx, 1);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_ACTIVE_WORKERS, Reg::Rcx, 8);

    b.mov_ri64(Reg::Rdi, CLONE_WORKER_FLAGS);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.mov_rr(Reg::R8, Reg::R15); // tls = worker tcb
    b.mov_ri64(Reg::Rax, 56);    // SYS_clone
    b.syscall();

    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_worker_entry);
    let l_clone_ok = b.new_label();
    b.cmp_ri32(Reg::Rax, 0);
    b.jcc_label(Cond::G, l_clone_ok);
    b.mov_ri64(Reg::Rcx, (-1i64) as u64);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_ACTIVE_WORKERS, Reg::Rcx, 8);
    b.jmp_label(l_spawn_done);

    b.bind_label(l_clone_ok);
    b.add_ri32(Reg::Rbx, 1);
    b.jmp_label(l_spawn_lp);

    b.bind_label(l_spawn_done);
    b.store_mem(Reg::R12, SCHED_OFF_NUM_WORKERS, Reg::Rbx, 8);
    b.mov_ri64(Reg::Rax, 2);
    b.xchg_mem_disp(Reg::R12, SCHED_OFF_INIT, Reg::Rax, 8);

    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.bind_label(l_ret_fast);
    b.ret();

    // --- Worker Thread Entry (child after SYS_clone; rsp == child_rsp, %16 == 0) ---
    b.bind_label(l_worker_entry);
    b.load_fs_disp(Reg::Rbx, 0); // rbx = worker TCB
    b.sub_ri32(Reg::Rsp, 128);
    b.mov_rr(Reg::Rdi, Reg::Rsp);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_ri64(Reg::Rcx, 16);
    b.rep_stosq();
    b.load_mem(Reg::Rcx, Reg::Rbx, 80, 8, false); // cpu_id
    b.and_ri32(Reg::Rcx, 1023);
    b.mov_rr(Reg::Rax, Reg::Rcx);
    b.shr_imm(Reg::Rax, 6);
    b.shl_imm(Reg::Rax, 3);
    b.add_rr(Reg::Rax, Reg::Rsp);
    b.and_ri32(Reg::Rcx, 63);
    b.mov_ri64(Reg::Rdx, 1);
    b.shl_cl(Reg::Rdx);
    b.store_mem(Reg::Rax, 0, Reg::Rdx, 8);
    b.mov_ri64(Reg::Rax, 203); // SYS_sched_setaffinity
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, 128);
    b.mov_rr(Reg::Rdx, Reg::Rsp);
    b.syscall();
    b.add_ri32(Reg::Rsp, 128);

    b.lea_rip_sym(Reg::R12, "__eon_sched_state", 0);
    b.xor_rr32(Reg::R13, Reg::R13); // last_gen = 0

    let l_wloop = b.new_label();
    let l_wexit = b.new_label();
    let l_w_skip = b.new_label();
    let l_w_dag = b.new_label();
    let l_w_leave = b.new_label();

    b.bind_label(l_wloop);
    b.cmp_mi32(Reg::R12, SCHED_OFF_SHUTDOWN, 0, 8);
    b.jcc_label(Cond::Ne, l_wexit);

    b.load_mem(Reg::Rax, Reg::R12, SCHED_OFF_WORK_GEN, 4, false);
    b.cmp_rr(Reg::Rax, Reg::R13);
    let l_w_work = b.new_label();
    b.jcc_label(Cond::Ne, l_w_work);

    // Park on futex(&work_gen, FUTEX_WAIT_PRIVATE, last_gen)
    b.cmp_mi32(Reg::R12, SCHED_OFF_SHUTDOWN, 0, 8);
    b.jcc_label(Cond::Ne, l_wexit);
    b.lea_mem(Reg::Rdi, Reg::R12, SCHED_OFF_WORK_GEN);
    b.mov_ri64(Reg::Rsi, 128); // FUTEX_WAIT_PRIVATE
    b.mov_rr(Reg::Rdx, Reg::R13);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.xor_rr32(Reg::R8, Reg::R8);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.mov_ri64(Reg::Rax, 202); // SYS_futex
    b.syscall();
    b.jmp_label(l_wloop);

    b.bind_label(l_w_work);
    b.mov_rr(Reg::R13, Reg::Rax);
    b.mov_ri64(Reg::Rax, 1);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_IN_TASK, Reg::Rax, 8);
    b.cmp_mi32(Reg::R12, SCHED_OFF_SHUTDOWN, 0, 8);
    b.jcc_label(Cond::Ne, l_w_skip);
    b.cmp_mi32(Reg::R12, SCHED_OFF_TASK_CLOSED, 0, 8);
    b.jcc_label(Cond::Ne, l_w_skip);
    b.load_mem(Reg::Rax, Reg::R12, SCHED_OFF_WORK_GEN, 4, false);
    b.cmp_rr(Reg::Rax, Reg::R13);
    b.jcc_label(Cond::Ne, l_w_skip);

    b.load_mem(Reg::Rax, Reg::R12, SCHED_OFF_TASK_KIND, 8, false);
    b.cmp_ri32(Reg::Rax, 2);
    b.jcc_label(Cond::E, l_w_dag);
    b.cmp_ri32(Reg::Rax, 1);
    b.jcc_label(Cond::Ne, l_w_leave);
    emit_par_for_work_loop(&mut b, Reg::R12);
    b.jmp_label(l_w_leave);

    b.bind_label(l_w_dag);
    emit_dag_work_loop(&mut b, Reg::R12);

    b.bind_label(l_w_leave);
    b.bind_label(l_w_skip);
    b.mov_ri64(Reg::Rax, (-1i64) as u64);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_IN_TASK, Reg::Rax, 8);
    b.jmp_label(l_wloop);

    b.bind_label(l_wexit);
    b.mov_ri64(Reg::Rax, (-1i64) as u64);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_ACTIVE_WORKERS, Reg::Rax, 8);
    b.mov_ri64(Reg::Rax, 60); // SYS_exit (single thread, NOT exit_group!)
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.syscall();
    b.ud2();

    b.finish()
}

/// `void eon_sched_shutdown(void)`
/// Signals all worker threads to exit and waits until `active_workers == 0`.
pub fn emit_rt_sched_shutdown() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_done = b.new_label();
    let l_wait = b.new_label();

    b.lea_rip_sym(Reg::R8, "__eon_sched_state", 0);
    b.cmp_mi32(Reg::R8, SCHED_OFF_INIT, 2, 8);
    b.jcc_label(Cond::Ne, l_done);

    b.mov_ri64(Reg::Rax, 1);
    b.xchg_mem_disp(Reg::R8, SCHED_OFF_SHUTDOWN, Reg::Rax, 8);
    b.mov_ri64(Reg::Rax, 1);
    b.lock_xadd_disp(Reg::R8, SCHED_OFF_WORK_GEN, Reg::Rax, 8);

    b.lea_mem(Reg::Rdi, Reg::R8, SCHED_OFF_WORK_GEN);
    b.mov_ri64(Reg::Rsi, 129); // FUTEX_WAKE_PRIVATE
    b.mov_ri64(Reg::Rdx, 0x7FFF_FFFF);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.xor_rr32(Reg::R8, Reg::R8);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.mov_ri64(Reg::Rax, 202);
    b.syscall();

    b.lea_rip_sym(Reg::R8, "__eon_sched_state", 0);
    b.bind_label(l_wait);
    b.cmp_mi32(Reg::R8, SCHED_OFF_ACTIVE_WORKERS, 0, 8);
    b.jcc_label(Cond::E, l_done);
    b.pause();
    b.jmp_label(l_wait);

    b.bind_label(l_done);
    b.ret();
    b.finish()
}

/// `void eon_par_for(size_t n [rdi], size_t chunk [rsi], void (*fn)(size_t, size_t, void *) [rdx], void *ctx [rcx])`
pub fn emit_rt_par_for() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret_zero = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_ret_zero);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::Rbx, Reg::Rdi); // n
    b.mov_ri64(Reg::Rax, 1);
    b.mov_rr(Reg::R13, Reg::Rsi); // eff_chunk = max(chunk, 1)
    b.test_rr(Reg::R13, Reg::R13);
    b.cmovcc(Cond::E, Reg::R13, Reg::Rax);
    b.mov_rr(Reg::R14, Reg::Rdx); // fn_ptr
    b.mov_rr(Reg::R15, Reg::Rcx); // ctx

    // Sequential fast path for tiny loops (n <= chunk)
    let l_seq = b.new_label();
    let l_done = b.new_label();
    b.cmp_rr(Reg::Rbx, Reg::R13);
    b.jcc_label(Cond::Be, l_seq);

    b.call_sym("eon_sched_init");
    b.lea_rip_sym(Reg::R12, "__eon_sched_state", 0);

    // Non-blocking try_lock on SCHED_OFF_CALL_LOCK so nested par_for inside workers never deadlocks
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_ri64(Reg::Rcx, 1);
    b.lock_cmpxchg_disp(Reg::R12, SCHED_OFF_CALL_LOCK, Reg::Rcx, 8);
    b.jcc_label(Cond::Ne, l_seq);

    // Publish task fields before bumping work_gen
    b.mov_ri64(Reg::Rax, 1);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_KIND, Reg::Rax, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_CTX, Reg::R15, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_FN, Reg::R14, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_TOTAL, Reg::Rbx, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_CHUNK, Reg::R13, 8);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_NEXT, Reg::Rax, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_DONE, Reg::Rax, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_CLOSED, Reg::Rax, 8);

    b.mov_ri64(Reg::Rax, 1);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_WORK_GEN, Reg::Rax, 8);

    // Wake workers via futex
    b.lea_mem(Reg::Rdi, Reg::R12, SCHED_OFF_WORK_GEN);
    b.mov_ri64(Reg::Rsi, 129); // FUTEX_WAKE_PRIVATE
    b.mov_ri64(Reg::Rdx, 0x7FFF_FFFF);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.xor_rr32(Reg::R8, Reg::R8);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.mov_ri64(Reg::Rax, 202);
    b.syscall();

    // Caller participates in chunk stealing too
    emit_par_for_work_loop(&mut b, Reg::R12);

    // Close epoch and drain in-flight workers before caller frees stack ctx
    b.mov_ri64(Reg::Rax, 1);
    b.xchg_mem_disp(Reg::R12, SCHED_OFF_TASK_CLOSED, Reg::Rax, 8);
    let l_drain = b.new_label();
    b.bind_label(l_drain);
    b.cmp_mi32(Reg::R12, SCHED_OFF_IN_TASK, 0, 8);
    let l_drained = b.new_label();
    b.jcc_label(Cond::E, l_drained);
    b.pause();
    b.jmp_label(l_drain);
    b.bind_label(l_drained);

    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.xchg_mem_disp(Reg::R12, SCHED_OFF_CALL_LOCK, Reg::Rax, 8);
    b.jmp_label(l_done);

    b.bind_label(l_seq);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_rr(Reg::Rdx, Reg::R15);
    b.call_reg(Reg::R14);

    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.bind_label(l_ret_zero);
    b.ret();
    b.finish()
}

/// `void eon_sched_run_dag(const EonDagNode *nodes [rdi], size_t num_nodes [rsi], void *ctx [rdx])`
/// `EonDagNode` (32 B): `[0]=fn_ptr, [8]=arg_ptr, [16]=init_deps:u32, [20]=num_succs:u32, [24]=succs_ptr`.
pub fn emit_rt_sched_run_dag() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret_empty = b.new_label();
    b.test_rr(Reg::Rsi, Reg::Rsi);
    b.jcc_label(Cond::E, l_ret_empty);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);

    b.mov_rr(Reg::R13, Reg::Rdi); // nodes
    b.mov_rr(Reg::R14, Reg::Rsi); // num_nodes
    b.mov_rr(Reg::R15, Reg::Rdx); // ctx

    // Single-node fast path
    let l_multi = b.new_label();
    let l_done = b.new_label();
    b.cmp_ri32(Reg::R14, 1);
    b.jcc_label(Cond::A, l_multi);
    b.load_mem(Reg::Rax, Reg::R13, 0, 8, false);
    b.load_mem(Reg::Rdi, Reg::R13, 8, 8, false);
    let l_have_1 = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::Ne, l_have_1);
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.bind_label(l_have_1);
    b.call_reg(Reg::Rax);
    b.jmp_label(l_done);

    b.bind_label(l_multi);
    b.call_sym("eon_sched_init");
    b.lea_rip_sym(Reg::R12, "__eon_sched_state", 0);

    // Acquire call lock (spin if another thread runs a Schedule; recursive DAG runs inline via try)
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.mov_ri64(Reg::Rcx, 1);
    b.lock_cmpxchg_disp(Reg::R12, SCHED_OFF_CALL_LOCK, Reg::Rcx, 8);
    let l_got_lock = b.new_label();
    b.jcc_label(Cond::E, l_got_lock);

    // Fallback sequential wave execution for nested DAG call
    b.xor_rr32(Reg::Rbx, Reg::Rbx);
    let l_seq_lp = b.new_label();
    b.bind_label(l_seq_lp);
    b.cmp_rr(Reg::Rbx, Reg::R14);
    b.jcc_label(Cond::Ae, l_done);
    b.mov_rr(Reg::R8, Reg::Rbx);
    b.shl_imm(Reg::R8, 5);
    b.add_rr(Reg::R8, Reg::R13);
    b.load_mem(Reg::Rax, Reg::R8, 0, 8, false);
    b.load_mem(Reg::Rdi, Reg::R8, 8, 8, false);
    let l_sq_arg = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::Ne, l_sq_arg);
    b.mov_rr(Reg::Rdi, Reg::R15);
    b.bind_label(l_sq_arg);
    b.call_reg(Reg::Rax);
    b.add_ri32(Reg::Rbx, 1);
    b.jmp_label(l_seq_lp);

    b.bind_label(l_got_lock);
    // Use inline dag_rt if <= 176 nodes, else temporary SYS_mmap
    b.xor_rr32(Reg::Rbx, Reg::Rbx); // mmap_sz (0 => inline)
    let l_use_inline = b.new_label();
    let l_rt_ready = b.new_label();
    b.cmp_ri32(Reg::R14, SCHED_MAX_INLINE_DAG as i32);
    b.jcc_label(Cond::Be, l_use_inline);
    b.mov_rr(Reg::Rsi, Reg::R14);
    b.shl_imm(Reg::Rsi, 4);
    b.add_ri32(Reg::Rsi, 4095);
    b.and_ri32(Reg::Rsi, !4095);
    b.mov_rr(Reg::Rbx, Reg::Rsi);
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.jmp_label(l_rt_ready);

    b.bind_label(l_use_inline);
    b.lea_mem(Reg::Rax, Reg::R12, SCHED_OFF_DAG_RT_INLINE);

    b.bind_label(l_rt_ready);
    b.store_mem(Reg::R12, SCHED_OFF_DAG_RT, Reg::Rax, 8);
    // Initialize dag_rt[i] = { rem_deps = nodes[i].init_deps, claimed = 0 }
    b.xor_rr32(Reg::Rcx, Reg::Rcx);
    let l_init_rt = b.new_label();
    b.bind_label(l_init_rt);
    b.cmp_rr(Reg::Rcx, Reg::R14);
    let l_init_done = b.new_label();
    b.jcc_label(Cond::Ae, l_init_done);
    b.mov_rr(Reg::R8, Reg::Rcx);
    b.shl_imm(Reg::R8, 5);
    b.add_rr(Reg::R8, Reg::R13);
    b.load_mem(Reg::Rdx, Reg::R8, 16, 4, false); // init_deps
    b.mov_rr(Reg::R9, Reg::Rcx);
    b.shl_imm(Reg::R9, 4);
    b.add_rr(Reg::R9, Reg::Rax);
    b.store_mem(Reg::R9, 0, Reg::Rdx, 8);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.store_mem(Reg::R9, 8, Reg::Rdx, 8);
    b.add_ri32(Reg::Rcx, 1);
    b.jmp_label(l_init_rt);

    b.bind_label(l_init_done);
    b.mov_ri64(Reg::Rax, 2);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_KIND, Reg::Rax, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_CTX, Reg::R15, 8);
    b.store_mem(Reg::R12, SCHED_OFF_DAG_NODES, Reg::R13, 8);
    b.store_mem(Reg::R12, SCHED_OFF_DAG_COUNT, Reg::R14, 8);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_DONE, Reg::Rax, 8);
    b.store_mem(Reg::R12, SCHED_OFF_TASK_CLOSED, Reg::Rax, 8);

    b.mov_ri64(Reg::Rax, 1);
    b.lock_xadd_disp(Reg::R12, SCHED_OFF_WORK_GEN, Reg::Rax, 8);
    b.lea_mem(Reg::Rdi, Reg::R12, SCHED_OFF_WORK_GEN);
    b.mov_ri64(Reg::Rsi, 129);
    b.mov_ri64(Reg::Rdx, 0x7FFF_FFFF);
    b.xor_rr32(Reg::R10, Reg::R10);
    b.xor_rr32(Reg::R8, Reg::R8);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.mov_ri64(Reg::Rax, 202);
    b.syscall();

    emit_dag_work_loop(&mut b, Reg::R12);

    b.mov_ri64(Reg::Rax, 1);
    b.xchg_mem_disp(Reg::R12, SCHED_OFF_TASK_CLOSED, Reg::Rax, 8);
    let l_drain = b.new_label();
    let l_drained = b.new_label();
    b.bind_label(l_drain);
    b.cmp_mi32(Reg::R12, SCHED_OFF_IN_TASK, 0, 8);
    b.jcc_label(Cond::E, l_drained);
    b.pause();
    b.jmp_label(l_drain);
    b.bind_label(l_drained);

    let l_no_unmap = b.new_label();
    b.test_rr(Reg::Rbx, Reg::Rbx);
    b.jcc_label(Cond::E, l_no_unmap);
    b.load_mem(Reg::Rdi, Reg::R12, SCHED_OFF_DAG_RT, 8, false);
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_no_unmap);

    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.xchg_mem_disp(Reg::R12, SCHED_OFF_CALL_LOCK, Reg::Rax, 8);

    b.bind_label(l_done);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.bind_label(l_ret_empty);
    b.ret();
    b.finish()
}

/// `void eon_fiber_switch(FiberContext *from [rdi], const FiberContext *to [rsi])`
/// Layout (64 B): `[0]=rbx, [8]=rbp, [16]=r12, [24]=r13, [32]=r14, [40]=r15, [48]=rsp, [56]=rip`.
pub fn emit_rt_fiber_switch() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_restore = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_restore);

    b.store_mem(Reg::Rdi, 0, Reg::Rbx, 8);
    b.store_mem(Reg::Rdi, 8, Reg::Rbp, 8);
    b.store_mem(Reg::Rdi, 16, Reg::R12, 8);
    b.store_mem(Reg::Rdi, 24, Reg::R13, 8);
    b.store_mem(Reg::Rdi, 32, Reg::R14, 8);
    b.store_mem(Reg::Rdi, 40, Reg::R15, 8);
    b.load_mem(Reg::Rax, Reg::Rsp, 0, 8, false); // return address
    b.store_mem(Reg::Rdi, 56, Reg::Rax, 8);
    b.lea_mem(Reg::Rax, Reg::Rsp, 8); // caller rsp after ret
    b.store_mem(Reg::Rdi, 48, Reg::Rax, 8);

    b.bind_label(l_restore);
    b.load_mem(Reg::Rbx, Reg::Rsi, 0, 8, false);
    b.load_mem(Reg::Rbp, Reg::Rsi, 8, 8, false);
    b.load_mem(Reg::R12, Reg::Rsi, 16, 8, false);
    b.load_mem(Reg::R13, Reg::Rsi, 24, 8, false);
    b.load_mem(Reg::R14, Reg::Rsi, 32, 8, false);
    b.load_mem(Reg::R15, Reg::Rsi, 40, 8, false);
    b.load_mem(Reg::Rsp, Reg::Rsi, 48, 8, false);
    b.load_mem(Reg::Rax, Reg::Rsi, 56, 8, false);
    b.jmp_reg(Reg::Rax);
    b.finish()
}

/// `void *eon_iouring_create(u32 entries [rdi])`
/// Sets up real kernel `SYS_io_uring_setup(425)` and mmaps SQ ring, CQ ring, and SQE array.
pub fn emit_rt_iouring_create() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);

    b.mov_ri64(Reg::Rax, 8);
    b.mov_rr(Reg::Rbx, Reg::Rdi);
    b.cmp_ri32(Reg::Rbx, 8);
    b.cmovcc(Cond::B, Reg::Rbx, Reg::Rax);

    // Allocate 4096 B for EonIoUring handle (+0..128) + io_uring_params (+128..248)
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rsi, 4096);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_ret = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_ret);
    b.mov_rr(Reg::R12, Reg::Rax);

    // SYS_io_uring_setup(entries, &params)
    b.mov_rr(Reg::Rdi, Reg::Rbx);
    b.lea_mem(Reg::Rsi, Reg::R12, 128);
    b.mov_ri64(Reg::Rax, 425);
    b.syscall();
    b.cmp_ri32(Reg::Rax, 0);
    let l_fail_unmap_hdr = b.new_label();
    b.jcc_label(Cond::L, l_fail_unmap_hdr);
    b.mov_rr(Reg::R13, Reg::Rax); // fd
    b.store_mem(Reg::R12, 0, Reg::R13, 8);

    b.load_mem(Reg::Rax, Reg::R12, 128, 4, false); // params.sq_entries
    b.store_mem(Reg::R12, 120, Reg::Rax, 8);

    // 1. mmap SQ ring: len = params.sq_off.array + sq_entries * 4
    b.shl_imm(Reg::Rax, 2);
    b.load_mem(Reg::Rcx, Reg::R12, 128 + 64, 4, false);
    b.add_rr(Reg::Rax, Reg::Rcx);
    b.store_mem(Reg::R12, 16, Reg::Rax, 8);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 1); // MAP_SHARED
    b.mov_rr(Reg::R8, Reg::R13);
    b.xor_rr32(Reg::R9, Reg::R9); // IORING_OFF_SQ_RING
    b.mov_ri64(Reg::Rax, 9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_close = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_close);
    b.store_mem(Reg::R12, 8, Reg::Rax, 8); // sq_ring

    // 2. mmap CQ ring: len = params.cq_off.cqes + cq_entries * 16
    b.load_mem(Reg::Rax, Reg::R12, 128 + 4, 4, false);
    b.shl_imm(Reg::Rax, 4);
    b.load_mem(Reg::Rcx, Reg::R12, 128 + 100, 4, false);
    b.add_rr(Reg::Rax, Reg::Rcx);
    b.store_mem(Reg::R12, 32, Reg::Rax, 8);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 1);
    b.mov_rr(Reg::R8, Reg::R13);
    b.mov_ri64(Reg::R9, 0x0800_0000); // IORING_OFF_CQ_RING
    b.mov_ri64(Reg::Rax, 9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_sq = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_sq);
    b.store_mem(Reg::R12, 24, Reg::Rax, 8); // cq_ring

    // 3. mmap SQEs: len = sq_entries * 64
    b.load_mem(Reg::Rax, Reg::R12, 120, 8, false);
    b.shl_imm(Reg::Rax, 6);
    b.store_mem(Reg::R12, 48, Reg::Rax, 8);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 1);
    b.mov_rr(Reg::R8, Reg::R13);
    b.mov_ri64(Reg::R9, 0x1000_0000); // IORING_OFF_SQES
    b.mov_ri64(Reg::Rax, 9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_cq = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_cq);
    b.store_mem(Reg::R12, 40, Reg::Rax, 8); // sqes

    // Populate cached ring pointers
    b.load_mem(Reg::R8, Reg::R12, 8, 8, false);   // sq_ring
    b.load_mem(Reg::R9, Reg::R12, 24, 8, false);  // cq_ring
    b.load_mem(Reg::Rax, Reg::R12, 128 + 40, 4, false);
    b.add_rr(Reg::Rax, Reg::R8);
    b.store_mem(Reg::R12, 56, Reg::Rax, 8); // sq_head_ptr
    b.load_mem(Reg::Rax, Reg::R12, 128 + 44, 4, false);
    b.add_rr(Reg::Rax, Reg::R8);
    b.store_mem(Reg::R12, 64, Reg::Rax, 8); // sq_tail_ptr
    b.load_mem(Reg::Rax, Reg::R12, 128 + 48, 4, false);
    b.add_rr(Reg::Rax, Reg::R8);
    b.load_mem(Reg::Rax, Reg::Rax, 0, 4, false);
    b.store_mem(Reg::R12, 72, Reg::Rax, 8); // sq_mask
    b.load_mem(Reg::Rax, Reg::R12, 128 + 64, 4, false);
    b.add_rr(Reg::Rax, Reg::R8);
    b.store_mem(Reg::R12, 80, Reg::Rax, 8); // sq_array_ptr

    b.load_mem(Reg::Rax, Reg::R12, 128 + 80, 4, false);
    b.add_rr(Reg::Rax, Reg::R9);
    b.store_mem(Reg::R12, 88, Reg::Rax, 8); // cq_head_ptr
    b.load_mem(Reg::Rax, Reg::R12, 128 + 84, 4, false);
    b.add_rr(Reg::Rax, Reg::R9);
    b.store_mem(Reg::R12, 96, Reg::Rax, 8); // cq_tail_ptr
    b.load_mem(Reg::Rax, Reg::R12, 128 + 88, 4, false);
    b.add_rr(Reg::Rax, Reg::R9);
    b.load_mem(Reg::Rax, Reg::Rax, 0, 4, false);
    b.store_mem(Reg::R12, 104, Reg::Rax, 8); // cq_mask
    b.load_mem(Reg::Rax, Reg::R12, 128 + 100, 4, false);
    b.add_rr(Reg::Rax, Reg::R9);
    b.store_mem(Reg::R12, 112, Reg::Rax, 8); // cqes_ptr

    b.mov_rr(Reg::Rax, Reg::R12);
    let l_done = b.new_label();
    b.jmp_label(l_done);

    b.bind_label(l_fail_cq);
    b.load_mem(Reg::Rdi, Reg::R12, 24, 8, false);
    b.load_mem(Reg::Rsi, Reg::R12, 32, 8, false);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_fail_sq);
    b.load_mem(Reg::Rdi, Reg::R12, 8, 8, false);
    b.load_mem(Reg::Rsi, Reg::R12, 16, 8, false);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_fail_close);
    b.mov_rr(Reg::Rdi, Reg::R13);
    b.mov_ri64(Reg::Rax, 3);
    b.syscall();
    b.bind_label(l_fail_unmap_hdr);
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.mov_ri64(Reg::Rsi, 4096);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.bind_label(l_fail_ret);
    b.xor_rr32(Reg::Rax, Reg::Rax);

    b.bind_label(l_done);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.ret();
    b.finish()
}

/// `void eon_iouring_destroy(void *ring_state [rdi])`
pub fn emit_rt_iouring_destroy() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_ret);

    b.push(Reg::R12);
    b.mov_rr(Reg::R12, Reg::Rdi);
    for (p_off, sz_off) in [(40i32, 48i32), (24, 32), (8, 16)] {
        let l_skip = b.new_label();
        b.load_mem(Reg::Rdi, Reg::R12, p_off, 8, false);
        b.test_rr(Reg::Rdi, Reg::Rdi);
        b.jcc_label(Cond::E, l_skip);
        b.load_mem(Reg::Rsi, Reg::R12, sz_off, 8, false);
        b.mov_ri64(Reg::Rax, 11);
        b.syscall();
        b.bind_label(l_skip);
    }
    b.load_mem(Reg::Rdi, Reg::R12, 0, 8, false);
    b.mov_ri64(Reg::Rax, 3); // SYS_close
    b.syscall();
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.mov_ri64(Reg::Rsi, 4096);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();
    b.pop(Reg::R12);
    b.bind_label(l_ret);
    b.ret();
    b.finish()
}

/// `u64 eon_run_fibers_iouring(size_t count [rdi], size_t stack_bytes [rsi], void *entry_fn [rdx], void *arg_ptr [rcx], u64 sanitize [r8])`
/// Allocates `count` static-stack fibers (with `PROT_NONE` guard pages when `sanitize != 0`),
/// executes two real `eon_fiber_switch` context switches per fiber across batched `IORING_OP_NOP`
/// submissions/completions via `SYS_io_uring_enter`, and returns completed fiber count.
pub fn emit_rt_run_fibers_iouring() -> Result<(Vec<u8>, Vec<FnReloc>), String> {
    let mut b = X64Buf::new();
    let l_ret_zero = b.new_label();
    b.test_rr(Reg::Rdi, Reg::Rdi);
    b.jcc_label(Cond::E, l_ret_zero);

    b.push(Reg::Rbx);
    b.push(Reg::R12);
    b.push(Reg::R13);
    b.push(Reg::R14);
    b.push(Reg::R15);
    b.sub_ri32(Reg::Rsp, 128); // [rsp..64]=main_ctx, locals at +64..+128

    b.mov_rr(Reg::R15, Reg::Rdi); // count
    b.store_mem(Reg::Rsp, 80, Reg::Rdx, 8); // entry_fn
    b.store_mem(Reg::Rsp, 88, Reg::Rcx, 8); // arg_ptr
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::Rsp, 96, Reg::Rax, 8); // completed = 0
    b.store_mem(Reg::Rsp, 112, Reg::R8, 8); // sanitize

    // eff_stack = align_up(max(stack_bytes, 4096), 4096)
    b.mov_ri64(Reg::Rax, 4096);
    b.cmp_ri32(Reg::Rsi, 4096);
    b.cmovcc(Cond::B, Reg::Rsi, Reg::Rax);
    b.add_ri32(Reg::Rsi, 4095);
    b.and_ri32(Reg::Rsi, !4095);
    // stride = eff_stack + (sanitize != 0 ? 4096 : 0)
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.test_rr(Reg::R8, Reg::R8);
    b.cmovcc(Cond::Ne, Reg::Rdx, Reg::Rax);
    b.add_rr(Reg::Rsi, Reg::Rdx);
    b.store_mem(Reg::Rsp, 104, Reg::Rsi, 8); // stride

    // desc_sz = align_up(count * 80, 4096)
    b.imul_rri32(Reg::Rsi, Reg::R15, 80);
    b.add_ri32(Reg::Rsi, 4095);
    b.and_ri32(Reg::Rsi, !4095);
    b.store_mem(Reg::Rsp, 64, Reg::Rsi, 8);
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_0 = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_0);
    b.mov_rr(Reg::R13, Reg::Rax); // desc_table

    // arena_sz = count * stride
    b.load_mem(Reg::Rsi, Reg::Rsp, 104, 8, false);
    b.mov_rr(Reg::Rax, Reg::R15);
    b.mul_r(Reg::Rsi); // rax = count * stride
    b.store_mem(Reg::Rsp, 72, Reg::Rax, 8);
    b.mov_rr(Reg::Rsi, Reg::Rax);
    b.mov_ri64(Reg::Rax, 9);
    b.xor_rr32(Reg::Rdi, Reg::Rdi);
    b.mov_ri64(Reg::Rdx, 3);
    b.mov_ri64(Reg::R10, 0x22);
    b.mov_ri64(Reg::R8, u64::MAX);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.syscall();
    b.mov_ri64(Reg::Rcx, (-4096i64) as u64);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    let l_fail_desc = b.new_label();
    b.jcc_label(Cond::Ae, l_fail_desc);
    b.mov_rr(Reg::R14, Reg::Rax); // stack_arena

    // Create io_uring with 128 entries
    b.mov_ri64(Reg::Rdi, 128);
    b.call_sym("eon_iouring_create");
    let l_fail_arena = b.new_label();
    b.test_rr(Reg::Rax, Reg::Rax);
    b.jcc_label(Cond::E, l_fail_arena);
    b.mov_rr(Reg::R12, Reg::Rax); // ring

    // Initialize all `count` FiberContext descriptors (and guard pages if sanitize != 0)
    let l_fiber_tramp = b.new_label();
    b.xor_rr32(Reg::Rbx, Reg::Rbx); // i = 0
    let l_init_lp = b.new_label();
    let l_init_done = b.new_label();
    let l_no_guard = b.new_label();
    b.load_mem(Reg::R8, Reg::Rsp, 104, 8, false);  // stride
    b.load_mem(Reg::R9, Reg::Rsp, 112, 8, false);  // sanitize
    b.bind_label(l_init_lp);
    b.cmp_rr(Reg::Rbx, Reg::R15);
    b.jcc_label(Cond::Ae, l_init_done);

    // stack_slice = r14 + i * stride
    b.mov_rr(Reg::R10, Reg::Rbx);
    b.imul_rr(Reg::R10, Reg::R8);
    b.add_rr(Reg::R10, Reg::R14);

    b.test_rr(Reg::R9, Reg::R9);
    b.jcc_label(Cond::E, l_no_guard);
    // Protect guard page [stack_slice .. stack_slice + 4096)
    b.push(Reg::R8);
    b.push(Reg::R9);
    b.push(Reg::R10);
    b.mov_rr(Reg::Rdi, Reg::R10);
    b.mov_ri64(Reg::Rsi, 4096);
    b.xor_rr32(Reg::Rdx, Reg::Rdx);
    b.mov_ri64(Reg::Rax, 10); // SYS_mprotect
    b.syscall();
    b.pop(Reg::R10);
    b.pop(Reg::R9);
    b.pop(Reg::R8);
    b.bind_label(l_no_guard);

    b.imul_rri32(Reg::R11, Reg::Rbx, 80);
    b.add_rr(Reg::R11, Reg::R13); // desc
    b.store_mem(Reg::R11, 0, Reg::Rbx, 8); // rbx = fiber_id
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.store_mem(Reg::R11, 8, Reg::Rax, 8); // rbp = 0
    b.store_mem(Reg::R11, 16, Reg::R11, 8); // r12 = desc
    b.load_mem(Reg::Rax, Reg::Rsp, 80, 8, false);
    b.store_mem(Reg::R11, 24, Reg::Rax, 8); // r13 = entry_fn
    b.lea_mem(Reg::Rax, Reg::Rsp, 0);
    b.store_mem(Reg::R11, 32, Reg::Rax, 8); // r14 = main_ctx_ptr
    b.load_mem(Reg::Rax, Reg::Rsp, 88, 8, false);
    b.store_mem(Reg::R11, 40, Reg::Rax, 8); // r15 = arg_ptr
    b.mov_rr(Reg::Rax, Reg::R10);
    b.add_rr(Reg::Rax, Reg::R8);
    b.store_mem(Reg::R11, 48, Reg::Rax, 8); // rsp = stack_top
    b.lea_rip_label(Reg::Rax, l_fiber_tramp);
    b.store_mem(Reg::R11, 56, Reg::Rax, 8); // rip = fiber_trampoline

    b.add_ri32(Reg::Rbx, 1);
    b.jmp_label(l_init_lp);

    // Batched io_uring execution loop (base_idx in rbx)
    b.bind_label(l_init_done);
    b.xor_rr32(Reg::Rbx, Reg::Rbx);
    let l_batch_lp = b.new_label();
    let l_batch_done = b.new_label();
    b.bind_label(l_batch_lp);
    b.cmp_rr(Reg::Rbx, Reg::R15);
    b.jcc_label(Cond::Ae, l_batch_done);

    // batch = min(count - base_idx, sq_entries)
    b.mov_rr(Reg::Rax, Reg::R15);
    b.sub_rr(Reg::Rax, Reg::Rbx);
    b.load_mem(Reg::Rcx, Reg::R12, 120, 8, false);
    b.cmp_rr(Reg::Rax, Reg::Rcx);
    b.cmovcc(Cond::A, Reg::Rax, Reg::Rcx);
    b.store_mem(Reg::Rsp, 120, Reg::Rax, 8); // batch

    // Start each fiber in batch and queue IORING_OP_NOP SQE
    b.xor_rr32(Reg::R8, Reg::R8); // k = 0
    let l_sub_lp = b.new_label();
    let l_sub_done = b.new_label();
    b.bind_label(l_sub_lp);
    b.load_mem(Reg::Rax, Reg::Rsp, 120, 8, false);
    b.cmp_rr(Reg::R8, Reg::Rax);
    b.jcc_label(Cond::Ae, l_sub_done);

    b.lea_mem(Reg::Rdi, Reg::Rsp, 0); // main_ctx
    b.mov_rr(Reg::Rsi, Reg::Rbx);
    b.add_rr(Reg::Rsi, Reg::R8); // fid = base + k
    b.push(Reg::R8);
    b.imul_rri32(Reg::Rsi, Reg::Rsi, 80);
    b.add_rr(Reg::Rsi, Reg::R13);
    b.call_sym("eon_fiber_switch");
    b.pop(Reg::R8);

    // Queue SQE for fid = base + k
    b.load_mem(Reg::R9, Reg::R12, 64, 8, false);  // sq_tail_ptr
    b.load_mem(Reg::R10, Reg::R9, 0, 4, false);   // sq_tail
    b.add_rr(Reg::R10, Reg::R8);                  // tail + k
    b.load_mem(Reg::R11, Reg::R12, 72, 8, false); // sq_mask
    b.and_rr(Reg::R10, Reg::R11);                 // slot
    b.load_mem(Reg::Rdi, Reg::R12, 40, 8, false); // sqes
    b.mov_rr(Reg::Rsi, Reg::R10);
    b.shl_imm(Reg::Rsi, 6);
    b.add_rr(Reg::Rdi, Reg::Rsi); // sqe
    b.xor_rr32(Reg::Rax, Reg::Rax);
    for q in 0..8i32 {
        b.store_mem(Reg::Rdi, q * 8, Reg::Rax, 8);
    }
    b.mov_rr(Reg::Rax, Reg::Rbx);
    b.add_rr(Reg::Rax, Reg::R8); // user_data = fid
    b.store_mem(Reg::Rdi, 32, Reg::Rax, 8);
    b.load_mem(Reg::Rdx, Reg::R12, 80, 8, false); // sq_array
    b.mov_rr(Reg::Rcx, Reg::R10);
    b.shl_imm(Reg::Rcx, 2);
    b.add_rr(Reg::Rdx, Reg::Rcx);
    b.store_mem(Reg::Rdx, 0, Reg::R10, 4);

    b.add_ri32(Reg::R8, 1);
    b.jmp_label(l_sub_lp);

    b.bind_label(l_sub_done);
    b.load_mem(Reg::R9, Reg::R12, 64, 8, false);
    b.load_mem(Reg::Rax, Reg::R9, 0, 4, false);
    b.load_mem(Reg::Rcx, Reg::Rsp, 120, 8, false);
    b.add_rr(Reg::Rax, Reg::Rcx);
    b.sfence();
    b.store_mem(Reg::R9, 0, Reg::Rax, 4); // publish sq_tail

    // Enter kernel for `batch` SQEs and `batch` CQEs
    b.load_mem(Reg::Rdi, Reg::R12, 0, 8, false); // fd
    b.mov_rr(Reg::Rsi, Reg::Rcx);                // to_submit = batch
    b.mov_rr(Reg::Rdx, Reg::Rcx);                // min_complete = batch
    b.mov_ri64(Reg::R10, 1);                     // IORING_ENTER_GETEVENTS
    b.xor_rr32(Reg::R8, Reg::R8);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.mov_ri64(Reg::Rax, 426); // SYS_io_uring_enter
    b.syscall();

    // Drain CQEs and resume matching fibers
    b.xor_rr32(Reg::R8, Reg::R8); // drained = 0
    let l_drain_lp = b.new_label();
    let l_cqe_lp = b.new_label();
    let l_cqe_empty = b.new_label();
    b.bind_label(l_drain_lp);
    b.load_mem(Reg::Rax, Reg::Rsp, 120, 8, false); // batch
    b.cmp_rr(Reg::R8, Reg::Rax);
    let l_batch_next = b.new_label();
    b.jcc_label(Cond::Ae, l_batch_next);

    b.bind_label(l_cqe_lp);
    b.load_mem(Reg::R9, Reg::R12, 88, 8, false); // cq_head_ptr
    b.load_mem(Reg::R10, Reg::R9, 0, 4, false);  // cq_head
    b.load_mem(Reg::R11, Reg::R12, 96, 8, false); // cq_tail_ptr
    b.load_mem(Reg::Rcx, Reg::R11, 0, 4, false); // cq_tail
    b.lfence();
    b.cmp_rr(Reg::R10, Reg::Rcx);
    b.jcc_label(Cond::E, l_cqe_empty);

    b.load_mem(Reg::Rdi, Reg::R12, 104, 8, false); // cq_mask
    b.mov_rr(Reg::Rsi, Reg::R10);
    b.and_rr(Reg::Rsi, Reg::Rdi);
    b.shl_imm(Reg::Rsi, 4);
    b.load_mem(Reg::Rdx, Reg::R12, 112, 8, false); // cqes_ptr
    b.add_rr(Reg::Rdx, Reg::Rsi);

    b.load_mem(Reg::Rdi, Reg::Rdx, 0, 8, false); // user_data = fid
    b.load_mem(Reg::Rax, Reg::Rdx, 8, 4, true);  // cqe.res
    b.add_ri32(Reg::R10, 1);
    b.sfence();
    b.store_mem(Reg::R9, 0, Reg::R10, 4); // advance cq_head

    b.imul_rri32(Reg::Rsi, Reg::Rdi, 80);
    b.add_rr(Reg::Rsi, Reg::R13);
    b.store_mem(Reg::Rsi, 72, Reg::Rax, 8);
    b.push(Reg::R8);
    b.push(Reg::Rsi);
    b.lea_mem(Reg::Rdi, Reg::Rsp, 16); // main_ctx (+16 past 2 pushes!)
    b.call_sym("eon_fiber_switch");
    b.pop(Reg::Rsi);
    b.pop(Reg::R8);

    b.load_mem(Reg::Rax, Reg::Rsi, 72, 8, false);
    b.load_mem(Reg::Rcx, Reg::Rsp, 96, 8, false);
    b.add_rr(Reg::Rcx, Reg::Rax);
    b.store_mem(Reg::Rsp, 96, Reg::Rcx, 8);
    b.add_ri32(Reg::R8, 1);
    b.load_mem(Reg::Rax, Reg::Rsp, 120, 8, false);
    b.cmp_rr(Reg::R8, Reg::Rax);
    b.jcc_label(Cond::B, l_cqe_lp);
    b.jmp_label(l_batch_next);

    b.bind_label(l_cqe_empty);
    // Still waiting on remaining CQEs (e.g., EINTR): enter with GETEVENTS
    b.load_mem(Reg::Rdx, Reg::Rsp, 120, 8, false);
    b.sub_rr(Reg::Rdx, Reg::R8);
    b.load_mem(Reg::Rdi, Reg::R12, 0, 8, false);
    b.xor_rr32(Reg::Rsi, Reg::Rsi);
    b.mov_ri64(Reg::R10, 1);
    b.xor_rr32(Reg::R9, Reg::R9);
    b.push(Reg::R8);
    b.xor_rr32(Reg::R8, Reg::R8);
    b.mov_ri64(Reg::Rax, 426);
    b.syscall();
    b.pop(Reg::R8);
    b.jmp_label(l_drain_lp);

    b.bind_label(l_batch_next);
    b.load_mem(Reg::Rax, Reg::Rsp, 120, 8, false);
    b.add_rr(Reg::Rbx, Reg::Rax);
    b.jmp_label(l_batch_lp);

    // Cleanup ring, stack arena, descriptor table, return completed count
    b.bind_label(l_batch_done);
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.call_sym("eon_iouring_destroy");

    b.bind_label(l_fail_arena);
    b.mov_rr(Reg::Rdi, Reg::R14);
    b.load_mem(Reg::Rsi, Reg::Rsp, 72, 8, false);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();

    b.bind_label(l_fail_desc);
    b.mov_rr(Reg::Rdi, Reg::R13);
    b.load_mem(Reg::Rsi, Reg::Rsp, 64, 8, false);
    b.mov_ri64(Reg::Rax, 11);
    b.syscall();

    b.bind_label(l_fail_0);
    b.load_mem(Reg::Rax, Reg::Rsp, 96, 8, false);
    b.add_ri32(Reg::Rsp, 128);
    b.pop(Reg::R15);
    b.pop(Reg::R14);
    b.pop(Reg::R13);
    b.pop(Reg::R12);
    b.pop(Reg::Rbx);
    b.ret();

    b.bind_label(l_ret_zero);
    b.xor_rr32(Reg::Rax, Reg::Rax);
    b.ret();

    // --- Static-stack fiber trampoline (entered via eon_fiber_switch with rsp % 16 == 0) ---
    b.bind_label(l_fiber_tramp);
    b.sub_ri32(Reg::Rsp, 16);
    b.store_mem(Reg::Rsp, 0, Reg::Rbx, 8);
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.mov_rr(Reg::Rsi, Reg::R14);
    b.call_sym("eon_fiber_switch");

    // Resumed after CQE arrival
    let l_skip_entry = b.new_label();
    b.test_rr(Reg::R13, Reg::R13);
    b.jcc_label(Cond::E, l_skip_entry);
    b.mov_rr(Reg::Rdi, Reg::R15); // arg_ptr
    b.load_mem(Reg::Rsi, Reg::R12, 72, 4, true); // cqe_res
    b.mov_rr(Reg::Rdx, Reg::Rbx); // fiber_id
    b.call_reg(Reg::R13);
    b.bind_label(l_skip_entry);

    b.mov_ri64(Reg::Rax, 1);
    b.store_mem(Reg::R12, 72, Reg::Rax, 8);
    b.add_ri32(Reg::Rsp, 16);
    b.mov_rr(Reg::Rdi, Reg::R12);
    b.mov_rr(Reg::Rsi, Reg::R14);
    b.call_sym("eon_fiber_switch");
    b.ud2();

    b.finish()
}

/// Emit all scheduler, fiber, and `io_uring` runtime helpers as `(sym_name, code, relocs)`.
pub fn emit_rt_sched_helpers() -> Result<Vec<RtNamedBlob>, String> {
    let mut out = Vec::with_capacity(8);
    for (name, res) in [
        ("eon_sched_init", emit_rt_sched_init()),
        ("eon_sched_shutdown", emit_rt_sched_shutdown()),
        ("eon_par_for", emit_rt_par_for()),
        ("eon_sched_run_dag", emit_rt_sched_run_dag()),
        ("eon_fiber_switch", emit_rt_fiber_switch()),
        ("eon_iouring_create", emit_rt_iouring_create()),
        ("eon_iouring_destroy", emit_rt_iouring_destroy()),
        ("eon_run_fibers_iouring", emit_rt_run_fibers_iouring()),
    ] {
        let (code, relocs) = res?;
        out.push((name, code, relocs));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_new_x86_instructions_accurately() {
        let mut b = X64Buf::new();
        b.load_fs_disp(Reg::Rax, 0);
        b.load_fs_disp(Reg::R10, 0x18);
        b.store_fs_disp(0, Reg::R11);
        b.vfmadd213(false, Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
        b.vfmadd213(true, Xmm::Xmm3, Xmm::Xmm4, Xmm::Xmm5);
        b.popcnt(true, Reg::R10, Reg::R11);
        b.popcnt(false, Reg::R10, Reg::R11);
        b.lzcnt(true, Reg::Rax, Reg::Rbx);
        b.tzcnt(false, Reg::R8, Reg::R9);
        b.bswap(8, Reg::Rax);
        b.bswap(8, Reg::R12);
        b.bswap(4, Reg::R12);
        b.bswap(2, Reg::R12);
        b.stmxcsr(Reg::R12, 16);
        b.ldmxcsr(Reg::Rsp, 8);
        b.fnstcw(Reg::Rbp, -4);
        b.fldcw(Reg::Rdi, 0);
        b.rep_stosq();
        let (code, relocs) = b.finish().unwrap();
        assert!(relocs.is_empty());
        assert_eq!(
            code,
            vec![
                0x64, 0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, // mov rax, fs:[0]
                0x64, 0x4C, 0x8B, 0x14, 0x25, 0x18, 0x00, 0x00, 0x00, // mov r10, fs:[0x18]
                0x64, 0x4C, 0x89, 0x1C, 0x25, 0x00, 0x00, 0x00, 0x00, // mov fs:[0], r11
                0xC4, 0xE2, 0x71, 0xA9, 0xC2, // vfmadd213ss xmm0, xmm1, xmm2
                0xC4, 0xE2, 0xD9, 0xA9, 0xDD, // vfmadd213sd xmm3, xmm4, xmm5
                0xF3, 0x4D, 0x0F, 0xB8, 0xD3, // popcnt r10, r11
                0xF3, 0x45, 0x0F, 0xB8, 0xD3, // popcnt r10d, r11d
                0xF3, 0x48, 0x0F, 0xBD, 0xC3, // lzcnt rax, rbx
                0xF3, 0x45, 0x0F, 0xBC, 0xC1, // tzcnt r8d, r9d
                0x48, 0x0F, 0xC8, // bswap rax
                0x49, 0x0F, 0xCC, // bswap r12
                0x41, 0x0F, 0xCC, // bswap r12d
                0x66, 0x41, 0xC1, 0xC4, 0x08, // rol r12w, 8
                0x41, 0x0F, 0xAE, 0x5C, 0x24, 0x10, // stmxcsr [r12+16]
                0x0F, 0xAE, 0x54, 0x24, 0x08, // ldmxcsr [rsp+8]
                0xD9, 0x7D, 0xFC, // fnstcw [rbp-4]
                0xD9, 0x2F, // fldcw [rdi]
                0xF3, 0x48, 0xAB, // rep stosq
            ]
        );
    }

    #[test]
    fn encodes_rip_relocs_and_displacements() {
        let mut b = X64Buf::new();
        b.mov_rip_load(Reg::R10, "__eon_vdso_clock_gettime", 0);
        b.mov_rip_store("__eon_vdso_clock_gettime", 0, Reg::Rax);
        b.call_rip_sym("puts@GOTPCREL", 0);
        b.jmp_rip_sym("exit@GOTPCREL", 0);
        b.jmp_reg(Reg::R11);
        b.lock_xadd_disp(Reg::Rbp, -16, Reg::Rcx, 8);
        b.lock_cmpxchg_disp(Reg::Rsp, 24, Reg::Rdx, 8);
        b.cmp_mi32(Reg::R12, 8, 2, 8);
        let (code, relocs) = b.finish().unwrap();
        assert_eq!(relocs.len(), 4);
        assert!(!code.is_empty());
    }

    #[test]
    fn all_runtime_helpers_emit_pic_without_abs64_relocs() {
        let names = [
            "eon_alloc_raw",
            "eon_free_raw",
            "eon_alloc_raw_san",
            "eon_free_raw_san",
            "eon_arena_alloc_aligned",
            "eon_rt_arena_reset_san",
            "eon_rt_install_san_handler",
            "eon_rt_tls_init",
            "eon_rt_parse_vdso",
            "eon_rt_now_ns",
            "eon_rt_openat2_beneath",
            "eon_rt_read_fd",
            "eon_rt_write_fd",
            "eon_rt_close_fd",
            "eon_rt_print_str",
            "eon_rt_print_bytes",
            "eon_rt_print_u64",
            "eon_rt_print_i64",
            "eon_rt_print_f32",
            "eon_rt_print_nl",
            "eon_rt_eprint_str",
            "eon_rt_eprint_bytes",
            "eon_rt_eprint_nl",
            "eon_rt_validate_utf8",
            "eon_rt_is_dir",
            "eon_rt_mkdir_all",
            "eon_rt_spawn_wait",
            "eon_f16_to_f32",
            "eon_f32_to_f16",
            "eon_sched_init",
            "eon_sched_shutdown",
            "eon_par_for",
            "eon_sched_run_dag",
            "eon_fiber_switch",
            "eon_iouring_create",
            "eon_iouring_destroy",
            "eon_run_fibers_iouring",
        ];
        for name in names {
            let (code, relocs) = emit_rt_helper_by_name(name)
                .unwrap_or_else(|| panic!("missing helper `{name}`"))
                .unwrap_or_else(|e| panic!("helper `{name}` failed: {e}"));
            assert!(!code.is_empty(), "empty helper {name}");
            for r in &relocs {
                assert!(
                    matches!(r.kind, RelocKind::Rel32),
                    "helper `{name}` emitted non-PIC relocation {:?}",
                    r
                );
            }
        }
        assert_eq!(emit_rt_sched_helpers().unwrap().len(), 8);
    }
}
