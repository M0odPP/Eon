#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Instant;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

const TARGET_LEN: usize = 16 * 1024 * 1024; // 16 MiB exact
const ITERS: usize = 8;

// 3 fixed multi-byte needles (exact UTF-8 bytes)
const NEEDLE_0: &[u8] = b"Eon_SIMD_v02";
// "café—π=3.14" = 63 61 66 C3 A9 E2 80 94 CF 80 3D 33 2E 31 34 (15 bytes)
const NEEDLE_1: &[u8] = &[
    b'c', b'a', b'f', 0xC3, 0xA9, 0xE2, 0x80, 0x94, 0xCF, 0x80, b'=', b'3', b'.', b'1', b'4',
];
// "🚀launch!" = F0 9F 9A 80 6C 61 75 6E 63 68 21 (11 bytes)
const NEEDLE_2: &[u8] = &[
    0xF0, 0x9F, 0x9A, 0x80, b'l', b'a', b'u', b'n', b'c', b'h', b'!',
];

const ASCII_PROSE: [&[u8]; 8] = [
    b"Linear ownership and capability-based APIs eliminate hidden global state in systems code. ",
    b"Structure-of-arrays pools align hot columns on cache lines for predictable traversal speed. ",
    b"Explicit wrapping arithmetic and static bounds proofs prevent silent integer wrap bugs. ",
    b"Deterministic benchmarks validate checksums across safe Rust and safe Eon implementations. ",
    b"Compile-time relational verification removes redundant bounds checks inside tight loops. ",
    b"Arena allocators reclaim entire parse trees in constant time with a single pointer reset. ",
    b"Vectorized byte scanning accelerates UTF-8 validation and substring search on x86_64 CPUs. ",
    b"Zero-copy byte slices keep memory traffic low while parsing structured text documents. ",
];

// 2-byte UTF-8: " résumé naïve café ΔΩ ДЖ "
const UTF8_2B: &[u8] = &[
    b' ', b'r', 0xC3, 0xA9, b's', b'u', b'm', 0xC3, 0xA9, b' ', b'n', b'a', 0xC3, 0xAF, b'v', b'e',
    b' ', b'c', b'a', b'f', 0xC3, 0xA9, b' ', 0xCE, 0x94, 0xCE, 0xA9, b' ', 0xD0, 0x94, 0xD0, 0x96,
    b' ',
];

// 3-byte UTF-8: " 漢字データ €100 — 日本語 "
const UTF8_3B: &[u8] = &[
    b' ', 0xE6, 0xBC, 0xA2, 0xE5, 0xAD, 0x97, 0xE3, 0x83, 0x87, 0xE3, 0x83, 0xBC, 0xE3, 0x82, 0xBF,
    b' ', 0xE2, 0x82, 0xAC, b'1', b'0', b'0', b' ', 0xE2, 0x80, 0x94, b' ', 0xE6, 0x97, 0xA5, 0xE6,
    0x9C, 0xAC, 0xE8, 0xAA, 0x9E, b' ',
];

// 4-byte UTF-8: " 🚀🦀𝄞 systems 🔥✨ "
const UTF8_4B: &[u8] = &[
    b' ', 0xF0, 0x9F, 0x9A, 0x80, 0xF0, 0x9F, 0xA6, 0x80, 0xF0, 0x9D, 0x84, 0x9E, b' ', b's', b'y',
    b's', b't', b'e', b'm', b's', b' ', 0xF0, 0x9F, 0x94, 0xA5, 0xE2, 0x9C, 0xA8, b' ',
];

// Decoys: " Eon_SIMX café—X 🚀land! "
const DECOYS: &[u8] = &[
    b' ', b'E', b'o', b'n', b'_', b'S', b'I', b'M', b'X', b' ', b'c', b'a', b'f', 0xC3, 0xA9, 0xE2,
    0x80, 0x94, b'X', b' ', 0xF0, 0x9F, 0x9A, 0x80, b'l', b'a', b'n', b'd', b'!', b' ',
];

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn generate_utf8_buffer(target_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_len);
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;

    while out.len() + 128 <= target_len {
        let r = next_u64(&mut rng) % 100;
        if r < 70 {
            let idx = (next_u64(&mut rng) & 7) as usize;
            out.extend_from_slice(ASCII_PROSE[idx]);
        } else if r < 81 {
            out.extend_from_slice(UTF8_2B);
        } else if r < 90 {
            out.extend_from_slice(UTF8_3B);
        } else if r < 96 {
            out.extend_from_slice(UTF8_4B);
        } else if r == 96 {
            out.extend_from_slice(DECOYS);
        } else if r == 97 {
            out.push(b'[');
            out.extend_from_slice(NEEDLE_0);
            out.extend_from_slice(b"] ");
        } else if r == 98 {
            out.push(b'<');
            out.extend_from_slice(NEEDLE_1);
            out.extend_from_slice(b"> ");
        } else {
            out.push(b'(');
            out.extend_from_slice(NEEDLE_2);
            out.extend_from_slice(b") ");
        }
    }
    while out.len() < target_len {
        out.push(b' ');
    }
    out
}

#[inline]
fn fnv_mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(FNV_PRIME)
}

#[inline]
fn validate_and_count_utf8(buf: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(buf).ok()?;
    Some(s.chars().count() as u64)
}

#[inline]
fn match_at(haystack: &[u8], pos: usize, needle: &[u8]) -> bool {
    let nlen = needle.len();
    if pos + nlen > haystack.len() {
        return false;
    }
    haystack[pos + nlen - 1] == needle[nlen - 1]
        && &haystack[pos + 1..pos + nlen - 1] == &needle[1..nlen - 1]
}

fn scan_three_needles(haystack: &[u8], n0: &[u8], n1: &[u8], n2: &[u8]) -> u64 {
    let f0 = n0[0];
    let f1 = n1[0];
    let f2 = n2[0];

    let mut h0 = FNV_OFFSET;
    let mut h1 = FNV_OFFSET;
    let mut h2 = FNV_OFFSET;
    let mut c0: u64 = 0;
    let mut c1: u64 = 0;
    let mut c2: u64 = 0;

    let mut base = 0usize;
    let chunks = haystack.chunks_exact(16);
    let rem_start = chunks.len() * 16;

    for chunk in chunks {
        let mut m0: u32 = 0;
        let mut m1: u32 = 0;
        let mut m2: u32 = 0;
        for j in 0..16 {
            let b = chunk[j];
            m0 |= ((b == f0) as u32) << j;
            m1 |= ((b == f1) as u32) << j;
            m2 |= ((b == f2) as u32) << j;
        }
        if (m0 | m1 | m2) != 0 {
            while m0 != 0 {
                let pos = base + m0.trailing_zeros() as usize;
                if match_at(haystack, pos, n0) {
                    h0 = fnv_mix(h0, (pos as u64).wrapping_add(0x1000_0001));
                    c0 = c0.wrapping_add(1);
                }
                m0 &= m0 - 1;
            }
            while m1 != 0 {
                let pos = base + m1.trailing_zeros() as usize;
                if match_at(haystack, pos, n1) {
                    h1 = fnv_mix(h1, (pos as u64).wrapping_add(0x2000_0002));
                    c1 = c1.wrapping_add(1);
                }
                m1 &= m1 - 1;
            }
            while m2 != 0 {
                let pos = base + m2.trailing_zeros() as usize;
                if match_at(haystack, pos, n2) {
                    h2 = fnv_mix(h2, (pos as u64).wrapping_add(0x3000_0003));
                    c2 = c2.wrapping_add(1);
                }
                m2 &= m2 - 1;
            }
        }
        base += 16;
    }

    let mut pos = rem_start;
    while pos < haystack.len() {
        let b = haystack[pos];
        if b == f0 && match_at(haystack, pos, n0) {
            h0 = fnv_mix(h0, (pos as u64).wrapping_add(0x1000_0001));
            c0 = c0.wrapping_add(1);
        }
        if b == f1 && match_at(haystack, pos, n1) {
            h1 = fnv_mix(h1, (pos as u64).wrapping_add(0x2000_0002));
            c1 = c1.wrapping_add(1);
        }
        if b == f2 && match_at(haystack, pos, n2) {
            h2 = fnv_mix(h2, (pos as u64).wrapping_add(0x3000_0003));
            c2 = c2.wrapping_add(1);
        }
        pos += 1;
    }

    let mut h = fnv_mix(h0, c0);
    h = fnv_mix(fnv_mix(h, h1), c1);
    h = fnv_mix(fnv_mix(h, h2), c2);
    h
}

fn main() {
    let buf = generate_utf8_buffer(TARGET_LEN);

    let t0 = Instant::now();
    let mut checksum: u64 = FNV_OFFSET;
    for iter in 0..ITERS {
        let data = black_box(buf.as_slice());
        let cp_count = validate_and_count_utf8(data).expect("valid UTF-8");
        let scan_h = scan_three_needles(data, NEEDLE_0, NEEDLE_1, NEEDLE_2);
        let h = fnv_mix(fnv_mix(FNV_OFFSET, cp_count), scan_h);
        checksum = fnv_mix(checksum, h).wrapping_add(iter as u64);
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    black_box(checksum);

    println!("RESULT utf8_scan checksum={} elapsed_ns={}", checksum, elapsed_ns);
}
