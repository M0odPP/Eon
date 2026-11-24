#![forbid(unsafe_code)]

use std::thread;
use std::time::Instant;

const NUM_NODES: usize = 262_144;
const NODE_MASK: u64 = 262_143;
const NUM_ITERS: usize = 20;
const INIT_RANK: u64 = 1_048_576;
const TELEPORT_BASE: u64 = 157_286;

struct GraphNodes {
    in_off: Vec<u32>,
    in_deg: Vec<u32>,
    out_deg: Vec<u32>,
    label: Vec<u32>,
    rank: Vec<u64>,
    contrib: Vec<u64>,
}

#[derive(Clone, Copy)]
struct NodeOut {
    node_idx: u32,
    next_label: u32,
    next_rank: u64,
    next_contrib: u64,
    hash: u64,
}

#[inline(always)]
fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn step_node(slot: &mut NodeOut, nodes: &GraphNodes, in_edges: &[u32]) {
    let u = slot.node_idx as usize;
    let off = nodes.in_off[u] as usize;
    let deg = nodes.in_deg[u] as usize;

    let mut sum_contrib: u64 = 0;
    let mut best_label: u32 = nodes.label[u];

    let edges = &in_edges[off..off + deg];
    for &src_u32 in edges {
        let src = src_u32 as usize;
        sum_contrib = sum_contrib.wrapping_add(nodes.contrib[src]);
        let sl = nodes.label[src];
        if sl < best_label {
            best_label = sl;
        }
    }

    let new_rank = ((sum_contrib.wrapping_mul(217)) >> 8).wrapping_add(TELEPORT_BASE);
    let od = nodes.out_deg[u] as u64;
    let new_contrib = if od > 0 { new_rank / od } else { new_rank };
    let h = mix64(
        new_rank
            ^ ((best_label as u64) << 32)
            ^ ((u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    );

    slot.next_rank = new_rank;
    slot.next_contrib = new_contrib;
    slot.next_label = best_label;
    slot.hash = h;
}

fn main() {
    // 1. Deterministic CSR incoming-edge graph construction (untimed setup).
    let mut in_off = Vec::with_capacity(NUM_NODES);
    let mut in_deg = Vec::with_capacity(NUM_NODES);
    let mut out_deg = vec![0u32; NUM_NODES];
    let mut in_edges = Vec::with_capacity(NUM_NODES * 16);

    let mut edge_cursor: u32 = 0;
    for u in 0..NUM_NODES {
        let uu = u as u64;
        let h0 = mix64(uu ^ 0xA076_1D64_78BD_642F);
        let deg_u64 = 8u64.wrapping_add(h0 & 7);
        let deg_u32 = deg_u64 as u32;
        in_off.push(edge_cursor);
        in_deg.push(deg_u32);

        let mut k: u64 = 0;
        while k < deg_u64 {
            let src_u64 = if (k & 1) == 0 {
                let sh = (k & 7).wrapping_mul(6);
                let step = 1u64.wrapping_add((h0 >> sh) & 63);
                uu.wrapping_add(step) & NODE_MASK
            } else {
                mix64(h0 ^ (k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15))) & NODE_MASK
            };
            let src_u32 = src_u64 as u32;
            in_edges.push(src_u32);
            out_deg[src_u64 as usize] = out_deg[src_u64 as usize].wrapping_add(1);
            k = k.wrapping_add(1);
        }
        edge_cursor = edge_cursor.wrapping_add(deg_u32);
    }

    for od in &mut out_deg {
        if *od == 0 {
            *od = 1;
        }
    }

    // 2. Populate SoA GraphNodes vectors.
    let mut label = Vec::with_capacity(NUM_NODES);
    let mut rank = Vec::with_capacity(NUM_NODES);
    let mut contrib = Vec::with_capacity(NUM_NODES);
    let mut out_buf = Vec::with_capacity(NUM_NODES);

    for u in 0..NUM_NODES {
        let uu = u as u64;
        let od = out_deg[u] as u64;
        let init_contrib = if od > 0 { INIT_RANK / od } else { INIT_RANK };
        let init_label = (mix64(uu ^ 0xE703_7ED1_A0B4_28DB) & 0x3FFF_FFFF) as u32;

        label.push(init_label);
        rank.push(INIT_RANK);
        contrib.push(init_contrib);
        out_buf.push(NodeOut {
            node_idx: u as u32,
            next_label: init_label,
            next_rank: INIT_RANK,
            next_contrib: init_contrib,
            hash: 0,
        });
    }

    let mut nodes = GraphNodes {
        in_off,
        in_deg,
        out_deg,
        label,
        rank,
        contrib,
    };

    let mut iter_hashes = vec![0u64; NUM_NODES];
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let chunk_size = NUM_NODES.div_ceil(n_threads);

    // 3. Timed workload: 20 iterations of parallel CSR PageRank + Label Propagation + Reduction.
    let t0 = Instant::now();
    let mut checksum: u64 = 0;

    for iter in 0..NUM_ITERS {
        let nodes_ref = &nodes;
        let edges_ref = in_edges.as_slice();
        thread::scope(|s| {
            for chunk in out_buf.chunks_mut(chunk_size) {
                s.spawn(move || {
                    for slot in chunk.iter_mut() {
                        step_node(slot, nodes_ref, edges_ref);
                    }
                });
            }
        });

        for u in 0..NUM_NODES {
            let s = out_buf[u];
            nodes.rank[u] = s.next_rank;
            nodes.contrib[u] = s.next_contrib;
            nodes.label[u] = s.next_label;
            iter_hashes[u] = s.hash;
        }

        // Parallel reduction matching `.par_reduce(.AddWrap, 0)` and `.par_reduce(.BitXor, 0)`.
        let (iter_sum, iter_xor) = thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for chunk in iter_hashes.chunks(chunk_size) {
                handles.push(s.spawn(move || {
                    let mut acc_sum: u64 = 0;
                    let mut acc_xor: u64 = 0;
                    for &v in chunk {
                        acc_sum = acc_sum.wrapping_add(v);
                        acc_xor ^= v;
                    }
                    (acc_sum, acc_xor)
                }));
            }
            let mut total_sum: u64 = 0;
            let mut total_xor: u64 = 0;
            for h in handles {
                let (ps, px) = h.join().unwrap();
                total_sum = total_sum.wrapping_add(ps);
                total_xor ^= px;
            }
            (total_sum, total_xor)
        });

        checksum = checksum.wrapping_add(mix64(iter_sum ^ iter_xor ^ ((iter as u64) + 1)));
    }

    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    println!(
        "RESULT par_reduce_graph checksum={} elapsed_ns={}",
        checksum, elapsed_ns
    );
}
