#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

fn has_regex_meta(pat: &[u8]) -> bool {
    pat.iter().any(|&c| matches!(c, b'^' | b'$' | b'.' | b'*' | b'+' | b'?'))
}

fn find_line_end(data: &[u8], start: usize) -> usize {
    match data[start..].iter().position(|&b| b == b'\n') {
        Some(off) => start + off,
        None => data.len(),
    }
}

fn find_line_start(data: &[u8], pos: usize) -> usize {
    match data[..pos].iter().rposition(|&b| b == b'\n') {
        Some(idx) => idx + 1,
        None => 0,
    }
}

fn count_literal_lines(data: &[u8], pat: &[u8], print_lines: bool) -> u64 {
    let plen = pat.len();
    if plen == 0 || data.len() < plen {
        return 0;
    }
    let first = pat[0];
    let last = pat[plen - 1];
    let last_off = plen - 1;
    let limit = data.len() - plen;
    let mut matches: u64 = 0;
    let mut i = 0usize;
    let mut out = io::BufWriter::new(io::stdout().lock());

    while i <= limit {
        if let Some(rel) = data[i..=limit].iter().position(|&b| b == first) {
            let pos = i + rel;
            if data[pos + last_off] == last && &data[pos..pos + plen] == pat {
                matches += 1;
                let le = find_line_end(data, pos + plen);
                if print_lines {
                    let ls = find_line_start(data, pos);
                    let _ = out.write_all(&data[ls..le]);
                    let _ = out.write_all(b"\n");
                }
                i = if le < data.len() { le + 1 } else { data.len() };
            } else {
                i = pos + 1;
            }
        } else {
            break;
        }
    }
    matches
}

#[inline]
fn char_matches(p: u8, c: u8) -> bool {
    p == b'.' || p == c
}

fn regex_match_here(pat: &[u8], pi: usize, line: &[u8], li: usize) -> bool {
    if pi >= pat.len() {
        return true;
    }
    if pat[pi] == b'$' && pi + 1 == pat.len() {
        return li == line.len();
    }
    if pi + 1 < pat.len() {
        let q = pat[pi + 1];
        if q == b'*' {
            let mut cur = li;
            loop {
                if regex_match_here(pat, pi + 2, line, cur) {
                    return true;
                }
                if cur < line.len() && char_matches(pat[pi], line[cur]) {
                    cur += 1;
                } else {
                    return false;
                }
            }
        }
        if q == b'+' {
            if li >= line.len() || !char_matches(pat[pi], line[li]) {
                return false;
            }
            let mut cur = li + 1;
            loop {
                if regex_match_here(pat, pi + 2, line, cur) {
                    return true;
                }
                if cur < line.len() && char_matches(pat[pi], line[cur]) {
                    cur += 1;
                } else {
                    return false;
                }
            }
        }
        if q == b'?' {
            if li < line.len() && char_matches(pat[pi], line[li]) {
                if regex_match_here(pat, pi + 2, line, li + 1) {
                    return true;
                }
            }
            return regex_match_here(pat, pi + 2, line, li);
        }
    }
    if li < line.len() && char_matches(pat[pi], line[li]) {
        return regex_match_here(pat, pi + 1, line, li + 1);
    }
    false
}

fn regex_match_line(pat: &[u8], line: &[u8]) -> bool {
    if pat.is_empty() {
        return true;
    }
    if pat[0] == b'^' {
        return regex_match_here(pat, 1, line, 0);
    }
    for i in 0..=line.len() {
        if regex_match_here(pat, 0, line, i) {
            return true;
        }
    }
    false
}

fn count_regex_lines(data: &[u8], pat: &[u8], print_lines: bool) -> u64 {
    let mut matches: u64 = 0;
    let mut pos = 0usize;
    let mut out = io::BufWriter::new(io::stdout().lock());
    while pos < data.len() {
        let le = find_line_end(data, pos);
        let line = &data[pos..le];
        if regex_match_line(pat, line) {
            matches += 1;
            if print_lines {
                let _ = out.write_all(line);
                let _ = out.write_all(b"\n");
            }
        }
        pos = if le < data.len() { le + 1 } else { data.len() };
    }
    matches
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut count_only = false;
    let mut repeat: u64 = 1;
    let mut idx = 1usize;
    while idx < args.len() {
        if args[idx] == "-c" {
            count_only = true;
            idx += 1;
            continue;
        }
        if args[idx] == "--bench" && idx + 1 < args.len() {
            repeat = args[idx + 1].parse::<u64>().unwrap_or(1).max(1);
            count_only = true;
            idx += 2;
            continue;
        }
        break;
    }
    if idx + 1 >= args.len() {
        eprintln!("usage: egrep_rs [-c] [--bench N] <pattern> <file>");
        std::process::exit(1);
    }
    let pat = args[idx].as_bytes();
    let data = fs::read(&args[idx + 1]).expect("failed to read file");
    let use_regex = has_regex_meta(pat);

    let t0 = Instant::now();
    let mut total: u64 = 0;
    for _ in 0..repeat {
        total = if use_regex {
            count_regex_lines(&data, pat, !count_only)
        } else {
            count_literal_lines(&data, pat, !count_only)
        };
    }
    let ns = t0.elapsed().as_nanos() as u64;
    if repeat > 1 {
        println!("egrep_rs bench: matches={} reps={} ns={}", total, repeat, ns);
    } else if count_only {
        println!("{}", total);
    }
}
