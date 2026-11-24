#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::time::Instant;

#[inline]
fn rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn hex_nibble(v: u64) -> u8 {
    let n = (v & 15) as u8;
    if n < 10 {
        48u8.wrapping_add(n)
    } else {
        87u8.wrapping_add(n)
    }
}

#[inline]
fn alpha_char(v: u64) -> u8 {
    let idx = (v & 31) as u8;
    if idx < 26 {
        97u8.wrapping_add(idx)
    } else if idx == 31 {
        95u8
    } else {
        48u8.wrapping_add(idx.wrapping_sub(26))
    }
}

fn push_prefix(buf: &mut Vec<u8>, kind: usize) {
    let s: &[u8] = match kind {
        0 => b"sym_",
        1 => b"mod_",
        2 => b"fn_",
        3 => b"type_",
        4 => b"var_",
        5 => b"cfg_",
        6 => b"ast_",
        _ => b"pkg_",
    };
    buf.extend_from_slice(s);
}

#[inline]
fn sample_zipf(rng: &mut u64) -> u32 {
    let r = rng_next(rng);
    let tier = r & 255;
    if tier < 96 {
        ((r >> 8) & 127) as u32
    } else if tier < 168 {
        128u32.wrapping_add(((r >> 8) & 1023) as u32)
    } else if tier < 216 {
        1152u32.wrapping_add(((r >> 8) & 4095) as u32)
    } else {
        ((r >> 8) & 32767) as u32
    }
}

struct StringInterner {
    map: HashMap<String, u32>,
    arena: String,
    spans: Vec<(u32, u32)>,
}

impl StringInterner {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            arena: String::new(),
            spans: Vec::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.spans.len() as u32;
        let start = self.arena.len() as u32;
        self.arena.push_str(s);
        self.spans.push((start, s.len() as u32));
        self.map.insert(s.to_owned(), id);
        id
    }

    #[inline]
    fn lookup(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }

    #[inline]
    fn get(&self, id: u32) -> Option<&[u8]> {
        let &(off, len) = self.spans.get(id as usize)?;
        let o = off as usize;
        let l = len as usize;
        Some(&self.arena.as_bytes()[o..o + l])
    }
}

fn main() {
    const NUM_UNIQUE: usize = 32768;
    const NUM_MISS_KEYS: usize = 2048;
    const TOTAL_STRINGS: usize = NUM_UNIQUE + NUM_MISS_KEYS;
    const NUM_EVENTS: usize = 400_000;
    const NUM_PROBES: usize = 100_000;

    // 1. Pre-generate deterministic string corpus and Zipf-skewed event/probe streams (excluded from timer).
    let mut key_bytes: Vec<u8> = Vec::with_capacity(TOTAL_STRINGS * 28);
    let mut key_offs: Vec<u32> = Vec::with_capacity(TOTAL_STRINGS);
    let mut key_lens: Vec<u32> = Vec::with_capacity(TOTAL_STRINGS);

    for k in 0..TOTAL_STRINGS {
        let start = key_bytes.len();
        let ku = k as u64;
        let h0 = splitmix64(ku ^ 0x9E3779B97F4A7C15);
        let h1 = splitmix64(h0);
        push_prefix(&mut key_bytes, k & 7);

        let body_len = 8usize + (((h0 >> 4) & 15) as usize);
        let mut s = h1;
        for _ in 0..body_len {
            key_bytes.push(alpha_char(s));
            s = splitmix64(s.wrapping_add(1));
        }
        key_bytes.push(95);
        key_bytes.push(hex_nibble(ku >> 16));
        key_bytes.push(hex_nibble(ku >> 12));
        key_bytes.push(hex_nibble(ku >> 8));
        key_bytes.push(hex_nibble(ku >> 4));
        key_bytes.push(hex_nibble(ku));

        let end = key_bytes.len();
        key_offs.push(start as u32);
        key_lens.push((end - start) as u32);
    }

    let key_str = String::from_utf8(key_bytes).expect("ASCII corpus");

    let mut rng: u64 = 0x517CC1B727220A95;
    let mut events: Vec<u32> = Vec::with_capacity(NUM_EVENTS);
    for i in 0..NUM_EVENTS {
        if i < 262144 && (i & 7) == 0 {
            let seq = i >> 3;
            let idx = (seq.wrapping_mul(19937).wrapping_add(13) & 32767) as u32;
            events.push(idx);
        } else {
            events.push(sample_zipf(&mut rng));
        }
    }

    let mut probes: Vec<u32> = Vec::with_capacity(NUM_PROBES);
    for _ in 0..NUM_PROBES {
        let r = rng_next(&mut rng);
        if (r & 4) == 0 && ((r >> 3) & 3) == 0 {
            let midx = (NUM_UNIQUE as u32).wrapping_add(((r >> 8) & 2047) as u32);
            probes.push(midx);
        } else {
            probes.push(sample_zipf(&mut rng));
        }
    }

    // 2. Timed interning + HashMap insert/update/lookup/delete workload.
    let t0 = Instant::now();

    let mut interner = StringInterner::new();
    let mut sym_stats: HashMap<u64, u64> = HashMap::new();
    let mut pair_stats: HashMap<u64, u64> = HashMap::new();

    let mut checksum: u64 = 0xCBF29CE484222325;

    // Pass 1: 400,000 Zipf-skewed intern + dual HashMap upsert operations.
    for (i, &key_idx_u32) in events.iter().enumerate() {
        let key_idx = key_idx_u32 as usize;
        let off = key_offs[key_idx] as usize;
        let len = key_lens[key_idx] as usize;
        let s = &key_str[off..off + len];

        let sym_id = interner.intern(s);
        let sym_u64 = sym_id as u64;
        let iu = i as u64;

        checksum = (checksum ^ sym_u64).wrapping_mul(0x100000001B3);

        let delta = iu.wrapping_mul(3) ^ 0x517CC1B727220A95;
        if let Some(v) = sym_stats.get_mut(&sym_u64) {
            *v = v.wrapping_add(delta);
        } else {
            sym_stats.insert(sym_u64, delta | 1);
        }

        let k2 = (sym_u64 << 2) | (iu & 3);
        let pdelta = sym_u64.wrapping_mul(0x9E3779B97F4A7C15) ^ iu;
        if let Some(v) = pair_stats.get_mut(&k2) {
            *v = v.wrapping_add(pdelta);
        } else {
            pair_stats.insert(k2, pdelta);
        }
    }

    // Pass 2: 100,000 mixed hit/miss Interner + HashMap lookups.
    let mut miss_count: u64 = 0;
    for (p, &key_idx_u32) in probes.iter().enumerate() {
        let key_idx = key_idx_u32 as usize;
        let off = key_offs[key_idx] as usize;
        let len = key_lens[key_idx] as usize;
        let s = &key_str[off..off + len];
        let pu = p as u64;

        if let Some(id) = interner.lookup(s) {
            let id_u64 = id as u64;
            let v1 = sym_stats.get(&id_u64).copied().unwrap_or(0);
            let k2 = (id_u64 << 2) | (pu & 3);
            let v2 = pair_stats.get(&k2).copied().unwrap_or(0);
            checksum = (checksum ^ v1.wrapping_add(v2).wrapping_add(id_u64)).wrapping_mul(0x100000001B3);
        } else {
            miss_count = miss_count.wrapping_add(1);
            checksum = (checksum ^ pu.wrapping_add(0xFF)).wrapping_mul(0x100000001B3);
        }
    }

    // Pass 3: Delete every 4th key from sym_stats, then verify reverse lookup + ordered final scan.
    let uniq_len = interner.len();
    for id in 0..uniq_len {
        if (id & 3) == 0 {
            let rem = sym_stats.remove(&(id as u64)).unwrap_or(0);
            checksum = (checksum ^ rem).wrapping_mul(0x100000001B3);
        }
    }

    for id in 0..uniq_len {
        let id_u32 = id as u32;
        let id_u64 = id as u64;
        if let Some(raw) = interner.get(id_u32) {
            if !raw.is_empty() {
                let first = raw[0] as u64;
                let last = raw[raw.len() - 1] as u64;
                let meta = ((raw.len() as u64) << 16) | (first << 8) | last;
                checksum = (checksum ^ meta).wrapping_mul(0x100000001B3);
            }
        }
        let sv = sym_stats.get(&id_u64).copied().unwrap_or(0xA5A5);
        let pv = pair_stats.get(&((id_u64 << 2) | (id_u64 & 3))).copied().unwrap_or(0x5A5A);
        checksum = (checksum ^ sv ^ pv).wrapping_mul(0x100000001B3);
    }

    checksum ^= ((uniq_len as u64) << 32) ^ ((sym_stats.len() as u64) << 16) ^ miss_count;

    drop(pair_stats);
    drop(sym_stats);
    drop(interner);

    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    println!("RESULT hashmap_intern checksum={checksum} elapsed_ns={elapsed_ns}");
}
