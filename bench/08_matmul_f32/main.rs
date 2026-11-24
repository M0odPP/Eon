#![forbid(unsafe_code)]

use std::time::Instant;

const N: usize = 512;
const MATMUL_REPS: usize = 3;
const BI: usize = 32;
const BK: usize = 32;
const BJ: usize = 64;

const DOT_LEN: usize = 65_536;
const DOT_U_BANK: usize = 16;
const DOT_V_BANK: usize = 32;

#[inline(never)]
fn matmul_add_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut ib = 0usize;
    while ib < n {
        let i_end = (ib + BI).min(n);
        let mut kb = 0usize;
        while kb < n {
            let k_end = (kb + BK).min(n);
            let mut jb = 0usize;
            while jb < n {
                let j_end = (jb + BJ).min(n);
                for i in ib..i_end {
                    let a_row = &a[i * n..(i + 1) * n];
                    let c_tile = &mut c[i * n + jb..i * n + j_end];
                    for k in kb..k_end {
                        let aik = a_row[k];
                        let b_tile = &b[k * n + jb..k * n + j_end];
                        for (cv, &bv) in c_tile.iter_mut().zip(b_tile.iter()) {
                            *cv += aik * bv;
                        }
                    }
                }
                jb += BJ;
            }
            kb += BK;
        }
        ib += BI;
    }
}

#[inline(never)]
fn dot_f32(u: &[f32], v: &[f32]) -> f32 {
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;
    let mut acc4 = 0.0_f32;
    let mut acc5 = 0.0_f32;
    let mut acc6 = 0.0_f32;
    let mut acc7 = 0.0_f32;

    for (cu, cv) in u.chunks_exact(8).zip(v.chunks_exact(8)) {
        acc0 += cu[0] * cv[0];
        acc1 += cu[1] * cv[1];
        acc2 += cu[2] * cv[2];
        acc3 += cu[3] * cv[3];
        acc4 += cu[4] * cv[4];
        acc5 += cu[5] * cv[5];
        acc6 += cu[6] * cv[6];
        acc7 += cu[7] * cv[7];
    }
    ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))
}

fn main() {
    let mut a = vec![0.0_f32; N * N];
    let mut b = vec![0.0_f32; N * N];
    let mut c = vec![0.0_f32; N * N];

    for i in 0..N {
        for j in 0..N {
            let av = (((i.wrapping_mul(17).wrapping_add(j.wrapping_mul(31)).wrapping_add(3)) & 15) as i32 - 8) as f32 * 0.25_f32;
            let bv = (((i.wrapping_mul(29).wrapping_add(j.wrapping_mul(13)).wrapping_add(7)) & 15) as i32 - 8) as f32 * 0.25_f32;
            a[i * N + j] = av;
            b[i * N + j] = bv;
        }
    }

    let mut u_bank = vec![0.0_f32; DOT_U_BANK * DOT_LEN];
    let mut v_bank = vec![0.0_f32; DOT_V_BANK * DOT_LEN];
    for p in 0..DOT_U_BANK {
        for m in 0..DOT_LEN {
            let uv = (((p.wrapping_mul(19).wrapping_add(m.wrapping_mul(7)).wrapping_add(1)) & 7) as i32 - 4) as f32 * 0.25_f32;
            u_bank[p * DOT_LEN + m] = uv;
        }
    }
    for q in 0..DOT_V_BANK {
        for m in 0..DOT_LEN {
            let vv = (((q.wrapping_mul(23).wrapping_add(m.wrapping_mul(11)).wrapping_add(5)) & 7) as i32 - 4) as f32 * 0.25_f32;
            v_bank[q * DOT_LEN + m] = vv;
        }
    }

    let mut dot_out = vec![0.0_f32; DOT_U_BANK * DOT_V_BANK];

    let t0 = Instant::now();
    c.fill(0.0_f32);
    for rep in 0..MATMUL_REPS {
        a[rep * N + rep] += 0.25_f32;
        matmul_add_tiled(&a, &b, &mut c, N);
    }
    for p in 0..DOT_U_BANK {
        let u = &u_bank[p * DOT_LEN..(p + 1) * DOT_LEN];
        for q in 0..DOT_V_BANK {
            let v = &v_bank[q * DOT_LEN..(q + 1) * DOT_LEN];
            dot_out[p * DOT_V_BANK + q] = dot_f32(u, v);
        }
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;

    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &x in &c {
        h = (h ^ (x.to_bits() as u64)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    for &d in &dot_out {
        h = (h ^ (d.to_bits() as u64)).wrapping_mul(0x0000_0100_0000_01B3);
    }

    println!("RESULT matmul_f32 checksum={} elapsed_ns={}", h, elapsed_ns);
}
