#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::time::Instant;

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r' | 0x0B | 0x0C)
}

fn count_bytes(data: &[u8]) -> (u64, u64, u64) {
    let mut lines: u64 = 0;
    let mut words: u64 = 0;
    let mut prev_ws: u64 = 1;

    let mut chunks = data.chunks_exact(32);
    for ch in &mut chunks {
        let mut nl_m: u32 = 0;
        let mut ws_m: u32 = 0;
        for (bit, &b) in ch.iter().enumerate() {
            let nl = (b == b'\n') as u32;
            let ws = is_ws(b) as u32;
            nl_m |= nl << bit;
            ws_m |= ws << bit;
        }
        let non_ws = !ws_m;
        let prev_shifted = (ws_m << 1) | (prev_ws as u32);
        let starts = non_ws & prev_shifted;
        lines += nl_m.count_ones() as u64;
        words += starts.count_ones() as u64;
        prev_ws = ((ws_m >> 31) & 1) as u64;
    }
    for &b in chunks.remainder() {
        if b == b'\n' {
            lines += 1;
        }
        let ws = is_ws(b) as u64;
        if ws == 0 && prev_ws == 1 {
            words += 1;
        }
        prev_ws = ws;
    }
    (lines, words, data.len() as u64)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ewc_rs [--bench N] <file...>");
        std::process::exit(1);
    }
    let mut idx = 1usize;
    let mut repeat: u64 = 1;
    if args[1] == "--bench" && args.len() >= 4 {
        repeat = args[2].parse::<u64>().unwrap_or(1).max(1);
        idx = 3;
    }
    let t0 = Instant::now();
    let mut total = (0u64, 0u64, 0u64);
    for path in &args[idx..] {
        let data = fs::read(path).expect("failed to read file");
        let mut c = (0u64, 0u64, 0u64);
        for _ in 0..repeat {
            c = count_bytes(&data);
        }
        total.0 += c.0;
        total.1 += c.1;
        total.2 += c.2;
        if repeat == 1 {
            println!("{} {} {} {}", c.0, c.1, c.2, path);
        }
    }
    let ns = t0.elapsed().as_nanos() as u64;
    if repeat > 1 {
        println!(
            "ewc_rs bench: lines={} words={} bytes={} reps={} ns={}",
            total.0, total.1, total.2, repeat, ns
        );
    }
}
