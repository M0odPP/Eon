#![forbid(unsafe_code)]

use std::time::Instant;

struct Bodies {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,
    inv_mass: Vec<f32>,
    drag: Vec<f32>,
}

impl Bodies {
    fn with_capacity(n: usize) -> Self {
        Self {
            pos_x: Vec::with_capacity(n),
            pos_y: Vec::with_capacity(n),
            pos_z: Vec::with_capacity(n),
            vel_x: Vec::with_capacity(n),
            vel_y: Vec::with_capacity(n),
            vel_z: Vec::with_capacity(n),
            inv_mass: Vec::with_capacity(n),
            drag: Vec::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.pos_x.len()
    }
}

fn sys_apply_forces(b: &mut Bodies) {
    let dt: f32 = 0.016;
    let grav: f32 = -9.81;
    let n = b.len();
    let pos_x = &b.pos_x[..n];
    let pos_y = &b.pos_y[..n];
    let pos_z = &b.pos_z[..n];
    let inv_mass = &b.inv_mass[..n];
    let drag = &b.drag[..n];
    let vel_x = &mut b.vel_x[..n];
    let vel_y = &mut b.vel_y[..n];
    let vel_z = &mut b.vel_z[..n];

    for i in 0..n {
        let px = pos_x[i];
        let py = pos_y[i];
        let pz = pos_z[i];
        let im = inv_mass[i];
        let dr = drag[i];

        let dx = 0.0_f32 - px;
        let dy = 25.0_f32 - py;
        let dz = 0.0_f32 - pz;
        let dist_sq = ((dx * dx) + (dy * dy)) + ((dz * dz) + 4.0_f32);
        let inv_d = 1.0_f32 / dist_sq.sqrt();
        let pull = 12.0_f32 * inv_d;
        let swirl = 6.0_f32 * inv_d;

        let ax = ((dx * pull) - (dz * swirl)) * im;
        let ay = grav + ((dy * pull) * im);
        let az = ((dz * pull) + (dx * swirl)) * im;

        let damp = 1.0_f32 - (dr * dt);
        vel_x[i] = (vel_x[i] + (ax * dt)) * damp;
        vel_y[i] = (vel_y[i] + (ay * dt)) * damp;
        vel_z[i] = (vel_z[i] + (az * dt)) * damp;
    }
}

fn sys_integrate_pos(b: &mut Bodies) {
    let dt: f32 = 0.016;
    let n = b.len();
    let vel_x = &b.vel_x[..n];
    let vel_y = &b.vel_y[..n];
    let vel_z = &b.vel_z[..n];
    let pos_x = &mut b.pos_x[..n];
    let pos_y = &mut b.pos_y[..n];
    let pos_z = &mut b.pos_z[..n];

    for i in 0..n {
        pos_x[i] = pos_x[i] + (vel_x[i] * dt);
        pos_y[i] = pos_y[i] + (vel_y[i] * dt);
        pos_z[i] = pos_z[i] + (vel_z[i] * dt);
    }
}

fn sys_collide_bounds(b: &mut Bodies) {
    let min_xz: f32 = -50.0;
    let max_xz: f32 = 50.0;
    let min_y: f32 = 0.0;
    let max_y: f32 = 100.0;
    let rest: f32 = -0.75;
    let n = b.len();
    let pos_x = &mut b.pos_x[..n];
    let pos_y = &mut b.pos_y[..n];
    let pos_z = &mut b.pos_z[..n];
    let vel_x = &mut b.vel_x[..n];
    let vel_y = &mut b.vel_y[..n];
    let vel_z = &mut b.vel_z[..n];

    for i in 0..n {
        let px = pos_x[i];
        if px < min_xz {
            pos_x[i] = min_xz;
            vel_x[i] = vel_x[i] * rest;
        } else if px > max_xz {
            pos_x[i] = max_xz;
            vel_x[i] = vel_x[i] * rest;
        }

        let py = pos_y[i];
        if py < min_y {
            pos_y[i] = min_y;
            vel_y[i] = vel_y[i] * rest;
        } else if py > max_y {
            pos_y[i] = max_y;
            vel_y[i] = vel_y[i] * rest;
        }

        let pz = pos_z[i];
        if pz < min_xz {
            pos_z[i] = min_xz;
            vel_z[i] = vel_z[i] * rest;
        } else if pz > max_xz {
            pos_z[i] = max_xz;
            vel_z[i] = vel_z[i] * rest;
        }
    }
}

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
fn rng_unit_f32(state: &mut u64) -> f32 {
    let r = (rng_next(state) >> 40) as u32;
    (r as f32) * (1.0_f32 / 16777216.0_f32)
}

fn main() {
    const NUM_BODIES: usize = 200_000;
    const NUM_TICKS: usize = 60;

    let mut bodies = Bodies::with_capacity(NUM_BODIES);

    // Deterministic body initialization (excluded from timed section).
    let mut rng: u64 = 0xA5A5F0F012345678;
    for _ in 0..NUM_BODIES {
        let px = (rng_unit_f32(&mut rng) * 96.0_f32) - 48.0_f32;
        let py = (rng_unit_f32(&mut rng) * 90.0_f32) + 5.0_f32;
        let pz = (rng_unit_f32(&mut rng) * 96.0_f32) - 48.0_f32;
        let vx = (rng_unit_f32(&mut rng) * 20.0_f32) - 10.0_f32;
        let vy = (rng_unit_f32(&mut rng) * 20.0_f32) - 10.0_f32;
        let vz = (rng_unit_f32(&mut rng) * 20.0_f32) - 10.0_f32;
        let inv_mass = 0.25_f32 + (rng_unit_f32(&mut rng) * 1.75_f32);
        let drag = 0.05_f32 + (rng_unit_f32(&mut rng) * 0.25_f32);

        bodies.pos_x.push(px);
        bodies.pos_y.push(py);
        bodies.pos_z.push(pz);
        bodies.vel_x.push(vx);
        bodies.vel_y.push(vy);
        bodies.vel_z.push(vz);
        bodies.inv_mass.push(inv_mass);
        bodies.drag.push(drag);
    }

    // Timed physics simulation loop.
    let t0 = Instant::now();
    for _ in 0..NUM_TICKS {
        sys_apply_forces(&mut bodies);
        sys_integrate_pos(&mut bodies);
        sys_collide_bounds(&mut bodies);
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;

    // Bit-exact IEEE-754 checksum over all body positions and velocities.
    let mut checksum: u64 = 0xCBF29CE484222325;
    for i in 0..bodies.len() {
        let b0 = ((bodies.pos_x[i].to_bits() as u64) << 32) | (bodies.pos_y[i].to_bits() as u64);
        let b1 = ((bodies.pos_z[i].to_bits() as u64) << 32) | (bodies.vel_x[i].to_bits() as u64);
        let b2 = ((bodies.vel_y[i].to_bits() as u64) << 32) | (bodies.vel_z[i].to_bits() as u64);
        checksum = (checksum ^ b0).wrapping_mul(0x100000001B3);
        checksum = (checksum ^ b1).wrapping_mul(0x100000001B3);
        checksum = (checksum ^ b2).wrapping_mul(0x100000001B3);
    }

    println!("RESULT ecs_physics checksum={checksum} elapsed_ns={elapsed_ns}");
}
