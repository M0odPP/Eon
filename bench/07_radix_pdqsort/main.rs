#![forbid(unsafe_code)]

use std::time::Instant;

const N_RADIX: usize = 1_000_000;
const N_RECORDS: usize = 200_000;

#[derive(Clone, Copy)]
struct Record {
    key: u64,
    payload: u64,
}

struct Rng {
    state: u64,
}

impl Rng {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[inline(never)]
fn radix_sort_u64(keys: &mut [u64], scratch: &mut [u64]) {
    let n = keys.len();
    debug_assert_eq!(scratch.len(), n);

    let mut hist = [[0usize; 256]; 8];
    for &v in keys.iter() {
        hist[0][(v & 0xFF) as usize] += 1;
        hist[1][((v >> 8) & 0xFF) as usize] += 1;
        hist[2][((v >> 16) & 0xFF) as usize] += 1;
        hist[3][((v >> 24) & 0xFF) as usize] += 1;
        hist[4][((v >> 32) & 0xFF) as usize] += 1;
        hist[5][((v >> 40) & 0xFF) as usize] += 1;
        hist[6][((v >> 48) & 0xFF) as usize] += 1;
        hist[7][((v >> 56) & 0xFF) as usize] += 1;
    }

    for pass in 0..8usize {
        let shift = pass * 8;
        let mut offsets = [0usize; 256];
        let mut sum = 0usize;
        for b in 0..256 {
            offsets[b] = sum;
            sum += hist[pass][b];
        }

        if (pass & 1) == 0 {
            for &v in keys.iter() {
                let b = ((v >> shift) & 0xFF) as usize;
                let pos = offsets[b];
                offsets[b] = pos + 1;
                scratch[pos] = v;
            }
        } else {
            for &v in scratch.iter() {
                let b = ((v >> shift) & 0xFF) as usize;
                let pos = offsets[b];
                offsets[b] = pos + 1;
                keys[pos] = v;
            }
        }
    }
}

fn main() {
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);

    let mut keys = Vec::with_capacity(N_RADIX);
    for _ in 0..N_RADIX {
        keys.push(rng.next());
    }
    let mut scratch = vec![0u64; N_RADIX];

    let mut records = Vec::with_capacity(N_RECORDS);
    for i in 0..N_RECORDS {
        let key = rng.next();
        let rot = (key << 17) | (key >> 47);
        let payload = (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ rot;
        records.push(Record { key, payload });
    }

    let t0 = Instant::now();
    radix_sort_u64(&mut keys, &mut scratch);
    records.sort_unstable_by_key(|r| r.key);
    let elapsed_ns = t0.elapsed().as_nanos() as u64;

    let mut ok = true;
    for i in 1..N_RADIX {
        if keys[i - 1] > keys[i] {
            ok = false;
            break;
        }
    }
    if ok {
        for i in 1..N_RECORDS {
            if records[i - 1].key > records[i].key {
                ok = false;
                break;
            }
        }
    }
    assert!(ok, "sort verification failed");

    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &v in &keys {
        h = (h ^ v).wrapping_mul(0x0000_0100_0000_01B3);
    }
    for r in &records {
        h = (h ^ r.key).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ r.payload).wrapping_mul(0x0000_0100_0000_01B3);
    }

    println!("RESULT radix_pdqsort checksum={} elapsed_ns={}", h, elapsed_ns);
}
