#![forbid(unsafe_code)]

use std::time::Instant;

const BATCH_SIZE: usize = 32;
const NUM_BATCHES: usize = 15_625; // 15,625 * 32 = 500,000 Vec<u8> alloc/free cycles
const ARENA_FRAMES: usize = 2_000;
const ARENA_CAP: usize = 262_144; // 256 KiB per frame arena
const SLICES_PER_FRAME: usize = 256; // 2,000 * 256 = 512,000 temporary arena slices

#[inline(always)]
fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn stamp_buf(s: &mut [u8], tag: u8) {
    let n = s.len();
    if n >= 32 {
        s[0] = tag;
        let q1 = n >> 2;
        let q2 = n >> 1;
        let last = n - 1;
        s[q1] = tag ^ 0x5A;
        s[q2] = tag.wrapping_add(0x33);
        s[last] = tag ^ 0xA5;
    }
}

#[inline]
fn checksum_buf(s: &[u8]) -> u64 {
    let n = s.len();
    let mut acc: u64 = n as u64;
    let mut i: usize = 0;
    while i < n {
        acc = acc.wrapping_mul(131).wrapping_add(s[i] as u64);
        i = i.wrapping_add(16);
    }
    if n > 0 {
        let last = n - 1;
        acc ^= (s[last] as u64) << 24;
    }
    acc
}

#[inline]
fn write_and_sum_slice(s: &mut [u8], seed: u64) -> u64 {
    let n = s.len();
    let base = (seed & 255) as u8;
    let mut i: usize = 0;
    while i < n {
        let b = base.wrapping_add((i & 255) as u8);
        s[i] = b;
        i = i.wrapping_add(8);
    }
    if n > 0 {
        let last = n - 1;
        s[last] = base ^ 0xC3;
    }
    let mut acc: u64 = n as u64;
    let mut j: usize = 0;
    while j < n {
        acc = acc.wrapping_mul(31).wrapping_add(s[j] as u64);
        j = j.wrapping_add(8);
    }
    if n > 0 {
        let last = n - 1;
        acc ^= (s[last] as u64) << 16;
    }
    acc
}

struct BumpArena {
    buf: Vec<u8>,
    used: usize,
}

impl BumpArena {
    #[inline]
    fn new(cap: usize) -> Self {
        Self {
            buf: vec![0u8; cap],
            used: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.used = 0;
    }

    #[inline]
    fn alloc_bytes(&mut self, len: usize) -> &mut [u8] {
        let aligned = (self.used + 15) & !15;
        let end = aligned + len;
        assert!(end <= self.buf.len(), "BumpArena OOM");
        self.used = end;
        &mut self.buf[aligned..end]
    }
}

fn main() {
    // Untimed setup: 1024-byte deterministic payload template.
    let mut pattern_buf = Vec::with_capacity(1024);
    for i in 0..1024usize {
        let v = (mix64((i as u64) ^ 0x517C_C1B7_2722_0A95) & 255) as u8;
        pattern_buf.push(v);
    }
    let pattern_slice = pattern_buf.as_slice();

    let t0 = Instant::now();
    let mut checksum: u64 = 0;
    let mut rng: u64 = 0x243F_6A88_85A3_08D3;

    // Part (a): 500,000 mixed-size (32 B .. 1024 B) Vec<u8> alloc/write/read/free cycles.
    let mut live: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);

    for batch in 0..NUM_BATCHES {
        for j in 0..BATCH_SIZE {
            rng = mix64(rng);
            let cls = (rng & 31) as usize;
            let sz = 32usize.wrapping_add(cls.wrapping_mul(32));

            let mut buf = Vec::<u8>::with_capacity(sz);
            buf.extend_from_slice(&pattern_slice[..sz]);

            let tag = (((batch as u64).wrapping_add(j as u64) ^ rng) & 255) as u8;
            stamp_buf(buf.as_mut_slice(), tag);
            live.push(buf);
        }

        let mut batch_acc: u64 = 0;
        while let Some(popped) = live.pop() {
            batch_acc = batch_acc.wrapping_add(checksum_buf(popped.as_slice()));
            drop(popped);
        }
        checksum = checksum.wrapping_add(mix64(batch_acc ^ (batch as u64)));
    }
    drop(live);

    // Part (b): 2,000 frame Arena cycles allocating 256 mixed temporary slices each.
    let mut arena = BumpArena::new(ARENA_CAP);

    for frame in 0..ARENA_FRAMES {
        if frame > 0 && (frame % 100) == 0 {
            arena = BumpArena::new(ARENA_CAP);
        } else {
            arena.reset();
        }

        let mut frame_acc: u64 = 0;
        for _ in 0..SLICES_PER_FRAME {
            rng = mix64(rng);
            let sz = 32usize.wrapping_add(((rng & 15) as usize).wrapping_mul(32));
            let sl = arena.alloc_bytes(sz);
            frame_acc = frame_acc.wrapping_add(write_and_sum_slice(sl, rng));
        }
        checksum = checksum.wrapping_add(mix64(frame_acc ^ ((frame as u64).wrapping_add(0x10_0000))));
    }

    drop(arena);

    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    println!(
        "RESULT alloc_churn checksum={} elapsed_ns={}",
        checksum, elapsed_ns
    );
}
