#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Instant;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

const NUM_RECORDS: usize = 4100;
const ITERS: usize = 15;
const MAX_DEPTH: u32 = 64;
const NONE_IDX: u32 = u32::MAX;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum JsonKind {
    Null = 0,
    Bool = 1,
    Int = 2,
    Str = 3,
    Array = 4,
    Object = 5,
}

#[derive(Clone, Copy)]
struct JsonNode {
    kind: JsonKind,
    bool_val: bool,
    int_val: i64,
    key_start: u32,
    key_len: u32,
    str_start: u32,
    str_len: u32,
    child_count: u32,
    first_child: u32,
    next_sibling: u32,
}

#[derive(Clone, Copy)]
struct StrSpan {
    start: u32,
    len: u32,
}

struct ParserState {
    pos: usize,
    depth: u32,
}

#[derive(Debug)]
enum JsonError {
    UnexpectedEof,
    UnexpectedToken,
    InvalidEscape,
    InvalidNumber,
    TrailingData,
    DepthLimit,
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn push_u64_dec(out: &mut Vec<u8>, mut val: u64) {
    if val == 0 {
        out.push(b'0');
        return;
    }
    let mut digits = [0u8; 24];
    let mut count = 0usize;
    while val > 0 {
        digits[count] = b'0' + (val % 10) as u8;
        val /= 10;
        count += 1;
    }
    while count > 0 {
        count -= 1;
        out.push(digits[count]);
    }
}

fn push_i64_dec(out: &mut Vec<u8>, val: i64) {
    if val < 0 {
        out.push(b'-');
        let mag = 0u64.wrapping_sub(val as u64);
        push_u64_dec(out, mag);
    } else {
        push_u64_dec(out, val as u64);
    }
}

fn push_gen_string(out: &mut Vec<u8>, rng: &mut u64, with_escapes: bool) {
    out.push(b'"');
    let n = 6usize + (next_u64(rng) % 18) as usize;
    for _ in 0..n {
        let v = (next_u64(rng) % 64) as u8;
        if with_escapes && v == 0 {
            out.extend_from_slice(b"\\\"");
        } else if with_escapes && v == 1 {
            out.extend_from_slice(b"\\\\");
        } else if with_escapes && v == 2 {
            out.extend_from_slice(b"\\n");
        } else if with_escapes && v == 3 {
            out.extend_from_slice(b"\\t");
        } else {
            let c = v % 36;
            if c < 26 {
                out.push(b'a' + c);
            } else {
                out.push(b'0' + (c - 26));
            }
        }
    }
    out.push(b'"');
}

fn generate_json_doc(num_records: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_records * 620 + 128);
    let mut rng: u64 = 0x1234_5678_9ABC_DEF0;

    out.extend_from_slice(b"{\"version\": 2, \"generator\": \"eon-bench-01\", \"records\": [\n");
    for r in 0..num_records {
        if r > 0 {
            out.extend_from_slice(b",\n");
        }
        out.extend_from_slice(b"  {\"id\": ");
        push_u64_dec(&mut out, r as u64);
        out.extend_from_slice(b", \"active\": ");
        if (next_u64(&mut rng) & 1) == 0 {
            out.extend_from_slice(b"true");
        } else {
            out.extend_from_slice(b"false");
        }
        out.extend_from_slice(b", \"tag\": ");
        push_gen_string(&mut out, &mut rng, true);

        out.extend_from_slice(b", \"score\": ");
        let score = (next_u64(&mut rng) % 1_000_000_001) as i64 - 500_000_000;
        push_i64_dec(&mut out, score);

        out.extend_from_slice(b", \"meta\": {\"zone\": ");
        push_gen_string(&mut out, &mut rng, false);
        out.extend_from_slice(b", \"flags\": [");
        for fi in 0..4 {
            if fi > 0 {
                out.extend_from_slice(b", ");
            }
            let fv = next_u64(&mut rng) % 3;
            if fv == 0 {
                out.extend_from_slice(b"true");
            } else if fv == 1 {
                out.extend_from_slice(b"false");
            } else {
                out.extend_from_slice(b"null");
            }
        }
        out.extend_from_slice(b"], \"counts\": [");
        for ci in 0..6 {
            if ci > 0 {
                out.extend_from_slice(b", ");
            }
            let cv = (next_u64(&mut rng) % 200_001) as i64 - 100_000;
            push_i64_dec(&mut out, cv);
        }
        out.extend_from_slice(b"]}, \"items\": [");
        let n_items = 2usize + (next_u64(&mut rng) % 3) as usize;
        for ii in 0..n_items {
            if ii > 0 {
                out.extend_from_slice(b", ");
            }
            out.extend_from_slice(b"{\"k\": ");
            push_gen_string(&mut out, &mut rng, false);
            out.extend_from_slice(b", \"v\": ");
            let iv = (next_u64(&mut rng) % 2_000_001) as i64 - 1_000_000;
            push_i64_dec(&mut out, iv);
            out.extend_from_slice(b", \"ok\": ");
            if (next_u64(&mut rng) & 1) == 0 {
                out.extend_from_slice(b"true");
            } else {
                out.extend_from_slice(b"false");
            }
            out.extend_from_slice(b", \"note\": ");
            push_gen_string(&mut out, &mut rng, true);
            out.push(b'}');
        }
        out.extend_from_slice(b"], \"extra\": [");
        let e0 = (next_u64(&mut rng) % 99_991) as i64;
        push_i64_dec(&mut out, e0);
        out.extend_from_slice(b", ");
        push_gen_string(&mut out, &mut rng, true);
        out.extend_from_slice(b", null, [");
        let e1 = (next_u64(&mut rng) % 65_536) as i64 - 32_768;
        let e2 = (next_u64(&mut rng) % 65_536) as i64 - 32_768;
        push_i64_dec(&mut out, e1);
        out.extend_from_slice(b", ");
        push_i64_dec(&mut out, e2);
        out.extend_from_slice(b"]]}");
    }
    out.extend_from_slice(b"\n]}\n");
    out
}

#[inline]
fn skip_ws(src: &[u8], st: &mut ParserState) {
    let mut p = st.pos;
    let n = src.len();
    while p < n {
        let b = src[p];
        if b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' {
            break;
        }
        p += 1;
    }
    st.pos = p;
}

#[inline]
fn parse_string_span(src: &[u8], st: &mut ParserState) -> Result<StrSpan, JsonError> {
    let p0 = st.pos;
    let n = src.len();
    if p0 >= n || src[p0] != b'"' {
        return Err(JsonError::UnexpectedToken);
    }
    let mut p = p0 + 1;
    let start = p;
    while p < n {
        let ch = src[p];
        if ch == b'"' {
            st.pos = p + 1;
            return Ok(StrSpan {
                start: start as u32,
                len: (p - start) as u32,
            });
        }
        if ch == b'\\' {
            let next_p = p + 1;
            if next_p >= n {
                return Err(JsonError::UnexpectedEof);
            }
            let esc = src[next_p];
            if esc != b'"'
                && esc != b'\\'
                && esc != b'/'
                && esc != b'b'
                && esc != b'f'
                && esc != b'n'
                && esc != b'r'
                && esc != b't'
            {
                return Err(JsonError::InvalidEscape);
            }
            p = next_p + 1;
            continue;
        }
        if ch < 0x20 {
            return Err(JsonError::UnexpectedToken);
        }
        p += 1;
    }
    st.pos = p;
    Err(JsonError::UnexpectedEof)
}

#[inline]
fn parse_int(src: &[u8], st: &mut ParserState) -> Result<i64, JsonError> {
    let mut p = st.pos;
    let n = src.len();
    if p >= n {
        return Err(JsonError::UnexpectedEof);
    }
    let neg = if src[p] == b'-' {
        p += 1;
        true
    } else {
        false
    };
    let start = p;
    let mut acc: i64 = 0;
    while p < n {
        let ch = src[p];
        if ch < b'0' || ch > b'9' {
            break;
        }
        acc = acc.wrapping_mul(10).wrapping_add((ch - b'0') as i64);
        p += 1;
    }
    st.pos = p;
    if p == start {
        return Err(JsonError::InvalidNumber);
    }
    if neg {
        Ok(0i64.wrapping_sub(acc))
    } else {
        Ok(acc)
    }
}

#[inline]
fn push_node(nodes: &mut Vec<JsonNode>, node: JsonNode) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(node);
    id
}

fn parse_value(
    src: &[u8],
    st: &mut ParserState,
    nodes: &mut Vec<JsonNode>,
    key_start: u32,
    key_len: u32,
) -> Result<u32, JsonError> {
    skip_ws(src, st);
    let p = st.pos;
    let n = src.len();
    if p >= n {
        return Err(JsonError::UnexpectedEof);
    }
    if st.depth >= MAX_DEPTH {
        return Err(JsonError::DepthLimit);
    }
    let ch = src[p];

    match ch {
        b'"' => {
            let sp = parse_string_span(src, st)?;
            Ok(push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Str,
                    bool_val: false,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: sp.start,
                    str_len: sp.len,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            ))
        }
        b'-' | b'0'..=b'9' => {
            let val = parse_int(src, st)?;
            Ok(push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Int,
                    bool_val: false,
                    int_val: val,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            ))
        }
        b't' => {
            if p + 4 > n || &src[p..p + 4] != b"true" {
                return Err(JsonError::UnexpectedToken);
            }
            st.pos = p + 4;
            Ok(push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Bool,
                    bool_val: true,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            ))
        }
        b'f' => {
            if p + 5 > n || &src[p..p + 5] != b"false" {
                return Err(JsonError::UnexpectedToken);
            }
            st.pos = p + 5;
            Ok(push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Bool,
                    bool_val: false,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            ))
        }
        b'n' => {
            if p + 4 > n || &src[p..p + 4] != b"null" {
                return Err(JsonError::UnexpectedToken);
            }
            st.pos = p + 4;
            Ok(push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Null,
                    bool_val: false,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            ))
        }
        b'[' => {
            st.pos = p + 1;
            st.depth += 1;
            let arr_id = push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Array,
                    bool_val: false,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            );
            skip_ws(src, st);
            if st.pos < n && src[st.pos] == b']' {
                st.pos += 1;
                st.depth -= 1;
                return Ok(arr_id);
            }
            let mut count: u32 = 0;
            let mut prev_child: u32 = NONE_IDX;
            loop {
                let elem_id = parse_value(src, st, nodes, 0, 0)?;
                if prev_child != NONE_IDX {
                    nodes[prev_child as usize].next_sibling = elem_id;
                } else {
                    nodes[arr_id as usize].first_child = elem_id;
                }
                prev_child = elem_id;
                count = count.wrapping_add(1);
                skip_ws(src, st);
                let pc = st.pos;
                if pc >= n {
                    return Err(JsonError::UnexpectedEof);
                }
                if src[pc] == b']' {
                    st.pos = pc + 1;
                    break;
                }
                if src[pc] != b',' {
                    return Err(JsonError::UnexpectedToken);
                }
                st.pos = pc + 1;
            }
            nodes[arr_id as usize].child_count = count;
            st.depth -= 1;
            Ok(arr_id)
        }
        b'{' => {
            st.pos = p + 1;
            st.depth += 1;
            let obj_id = push_node(
                nodes,
                JsonNode {
                    kind: JsonKind::Object,
                    bool_val: false,
                    int_val: 0,
                    key_start,
                    key_len,
                    str_start: 0,
                    str_len: 0,
                    child_count: 0,
                    first_child: NONE_IDX,
                    next_sibling: NONE_IDX,
                },
            );
            skip_ws(src, st);
            if st.pos < n && src[st.pos] == b'}' {
                st.pos += 1;
                st.depth -= 1;
                return Ok(obj_id);
            }
            let mut count: u32 = 0;
            let mut prev_child: u32 = NONE_IDX;
            loop {
                skip_ws(src, st);
                let key = parse_string_span(src, st)?;
                skip_ws(src, st);
                let p_colon = st.pos;
                if p_colon >= n || src[p_colon] != b':' {
                    return Err(JsonError::UnexpectedToken);
                }
                st.pos = p_colon + 1;
                let val_id = parse_value(src, st, nodes, key.start, key.len)?;
                if prev_child != NONE_IDX {
                    nodes[prev_child as usize].next_sibling = val_id;
                } else {
                    nodes[obj_id as usize].first_child = val_id;
                }
                prev_child = val_id;
                count = count.wrapping_add(1);
                skip_ws(src, st);
                let pc = st.pos;
                if pc >= n {
                    return Err(JsonError::UnexpectedEof);
                }
                if src[pc] == b'}' {
                    st.pos = pc + 1;
                    break;
                }
                if src[pc] != b',' {
                    return Err(JsonError::UnexpectedToken);
                }
                st.pos = pc + 1;
            }
            nodes[obj_id as usize].child_count = count;
            st.depth -= 1;
            Ok(obj_id)
        }
        _ => Err(JsonError::UnexpectedToken),
    }
}

fn parse_json_doc(input: &[u8], nodes: &mut Vec<JsonNode>) -> Result<u32, JsonError> {
    let mut st = ParserState { pos: 0, depth: 0 };
    let root = parse_value(input, &mut st, nodes, 0, 0)?;
    skip_ws(input, &mut st);
    if st.pos != input.len() {
        return Err(JsonError::TrailingData);
    }
    Ok(root)
}

#[inline]
fn fnv_mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(FNV_PRIME)
}

fn walk_dom(src: &[u8], nodes: &[JsonNode], id: u32, depth: u64, mut h: u64) -> u64 {
    let n = &nodes[id as usize];
    h = fnv_mix(h, ((n.kind as u64) << 8) ^ depth);
    if n.key_len > 0 {
        let ks = n.key_start as usize;
        h = fnv_mix(h, n.key_len as u64);
        if ks < src.len() {
            h = fnv_mix(h, src[ks] as u64);
        }
    }
    match n.kind {
        JsonKind::Null => {
            h = fnv_mix(h, 0xA5A5);
        }
        JsonKind::Bool => {
            h = fnv_mix(h, if n.bool_val { 0x1111 } else { 0x2222 });
        }
        JsonKind::Int => {
            h = fnv_mix(h, n.int_val as u64);
        }
        JsonKind::Str => {
            let ss = n.str_start as usize;
            let sl = n.str_len as usize;
            h = fnv_mix(h, sl as u64);
            if sl > 0 && ss + sl <= src.len() {
                h = fnv_mix(h, ((src[ss] as u64) << 8) | (src[ss + sl - 1] as u64));
            }
        }
        JsonKind::Array | JsonKind::Object => {
            h = fnv_mix(h, n.child_count as u64);
            let mut cur = n.first_child;
            while cur != NONE_IDX {
                h = walk_dom(src, nodes, cur, depth.wrapping_add(1), h);
                cur = nodes[cur as usize].next_sibling;
            }
            h = fnv_mix(h, 0x5555 ^ depth);
        }
    }
    h
}

fn main() {
    let doc = generate_json_doc(NUM_RECORDS);
    let mut nodes: Vec<JsonNode> = Vec::with_capacity(262_144);

    let t0 = Instant::now();
    let mut checksum: u64 = FNV_OFFSET;
    for iter in 0..ITERS {
        let src = black_box(doc.as_slice());
        nodes.clear();
        let root = parse_json_doc(src, &mut nodes).expect("valid generated JSON");
        let h = walk_dom(src, &nodes, root, 0, FNV_OFFSET);
        let h2 = fnv_mix(h, nodes.len() as u64);
        checksum = fnv_mix(checksum, h2).wrapping_add(iter as u64);
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    black_box(checksum);

    println!("RESULT json_dom checksum={} elapsed_ns={}", checksum, elapsed_ns);
}
