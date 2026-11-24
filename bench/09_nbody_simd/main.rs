#![forbid(unsafe_code)]

use std::time::Instant;

const N: usize = 1536;
const STEPS: usize = 40;
const EPS: f32 = 0.0625_f32;
const DT: f32 = 0.0078125_f32;

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

    #[inline]
    fn next_f32_sym(&mut self, scale: f32) -> f32 {
        let u = (self.next() & 4095) as i32 - 2048;
        (u as f32) * scale
    }
}

#[inline(never)]
fn nbody_step(
    px: &mut [f32],
    py: &mut [f32],
    pz: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    mass: &[f32],
    ax: &mut [f32],
    ay: &mut [f32],
    az: &mut [f32],
) {
    let n = px.len();
    ax.fill(0.0_f32);
    ay.fill(0.0_f32);
    az.fill(0.0_f32);

    const BLOCK: usize = 64;
    let mut ib = 0usize;
    while ib < n {
        let i_end = (ib + BLOCK).min(n);
        let px_b = &px[ib..i_end];
        let py_b = &py[ib..i_end];
        let pz_b = &pz[ib..i_end];
        let ax_b = &mut ax[ib..i_end];
        let ay_b = &mut ay[ib..i_end];
        let az_b = &mut az[ib..i_end];

        for j in 0..n {
            let xj = px[j];
            let yj = py[j];
            let zj = pz[j];
            let mj = mass[j];

            for i in 0..px_b.len() {
                let dx = xj - px_b[i];
                let dy = yj - py_b[i];
                let dz = zj - pz_b[i];
                let dist2 = dx * dx + (dy * dy + (dz * dz + EPS));
                let inv_d = 1.0_f32 / dist2.sqrt();
                let inv_d3 = (inv_d * inv_d) * inv_d;
                let s = mj * inv_d3;
                ax_b[i] = dx * s + ax_b[i];
                ay_b[i] = dy * s + ay_b[i];
                az_b[i] = dz * s + az_b[i];
            }
        }
        ib += BLOCK;
    }

    for i in 0..n {
        vx[i] = ax[i] * DT + vx[i];
        vy[i] = ay[i] * DT + vy[i];
        vz[i] = az[i] * DT + vz[i];
        px[i] = vx[i] * DT + px[i];
        py[i] = vy[i] * DT + py[i];
        pz[i] = vz[i] * DT + pz[i];
    }
}

fn main() {
    let mut rng = Rng::new(0xD1B5_4A32_D192_ED03);

    let mut px = Vec::with_capacity(N);
    let mut py = Vec::with_capacity(N);
    let mut pz = Vec::with_capacity(N);
    let mut vx = Vec::with_capacity(N);
    let mut vy = Vec::with_capacity(N);
    let mut vz = Vec::with_capacity(N);
    let mut mass = Vec::with_capacity(N);

    for _ in 0..N {
        px.push(rng.next_f32_sym(1.0_f32 / 256.0_f32));
        py.push(rng.next_f32_sym(1.0_f32 / 256.0_f32));
        pz.push(rng.next_f32_sym(1.0_f32 / 256.0_f32));
        vx.push(rng.next_f32_sym(1.0_f32 / 8192.0_f32));
        vy.push(rng.next_f32_sym(1.0_f32 / 8192.0_f32));
        vz.push(rng.next_f32_sym(1.0_f32 / 8192.0_f32));
        let m_u = (rng.next() & 1023) as i32;
        mass.push(0.5_f32 + (m_u as f32) * (1.0_f32 / 1024.0_f32));
    }

    let mut ax = vec![0.0_f32; N];
    let mut ay = vec![0.0_f32; N];
    let mut az = vec![0.0_f32; N];

    let t0 = Instant::now();
    for _ in 0..STEPS {
        nbody_step(
            &mut px, &mut py, &mut pz, &mut vx, &mut vy, &mut vz, &mass, &mut ax, &mut ay, &mut az,
        );
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;

    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for i in 0..N {
        let qpx = (px[i] * 1024.0_f32) as i64 as u64;
        let qpy = (py[i] * 1024.0_f32) as i64 as u64;
        let qpz = (pz[i] * 1024.0_f32) as i64 as u64;
        let qvx = (vx[i] * 1024.0_f32) as i64 as u64;
        let qvy = (vy[i] * 1024.0_f32) as i64 as u64;
        let qvz = (vz[i] * 1024.0_f32) as i64 as u64;
        h = (h ^ qpx).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ qpy).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ qpz).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ qvx).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ qvy).wrapping_mul(0x0000_0100_0000_01B3);
        h = (h ^ qvz).wrapping_mul(0x0000_0100_0000_01B3);
    }

    println!("RESULT nbody_simd checksum={} elapsed_ns={}", h, elapsed_ns);
}
