#![forbid(unsafe_code)]

use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
struct GenId {
    idx: u32,
    gen: u32,
}

#[derive(Clone, Copy)]
struct Widget {
    tag: u32,
    min_w: u32,
    min_h: u32,
    pad: u32,
    gap: u32,
    flex: u32,
    is_row: u8,
    parent: Option<GenId>,
    first_child: Option<GenId>,
    next_sibling: Option<GenId>,
}

struct Slot {
    gen: u32, // odd = occupied, even = free
    val: Widget,
}

struct SlotPool {
    slots: Vec<Slot>,
    free_stack: Vec<u32>,
}

impl SlotPool {
    fn with_capacity(cap: usize) -> Self {
        let mut slots = Vec::with_capacity(cap);
        let mut free_stack = Vec::with_capacity(cap);
        let empty_widget = Widget {
            tag: 0,
            min_w: 0,
            min_h: 0,
            pad: 0,
            gap: 0,
            flex: 0,
            is_row: 0,
            parent: None,
            first_child: None,
            next_sibling: None,
        };
        for i in 0..cap {
            slots.push(Slot {
                gen: 0,
                val: empty_widget,
            });
            free_stack.push((cap - 1 - i) as u32);
        }
        Self { slots, free_stack }
    }

    #[inline]
    fn insert(&mut self, w: Widget) -> GenId {
        if let Some(idx) = self.free_stack.pop() {
            let slot = &mut self.slots[idx as usize];
            let new_gen = slot.gen.wrapping_add(1);
            slot.gen = new_gen;
            slot.val = w;
            GenId { idx, gen: new_gen }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot { gen: 1, val: w });
            GenId { idx, gen: 1 }
        }
    }

    #[inline]
    fn remove(&mut self, id: GenId) -> Option<Widget> {
        if (id.gen & 1) == 0 {
            return None;
        }
        let slot = self.slots.get_mut(id.idx as usize)?;
        if slot.gen != id.gen {
            return None;
        }
        slot.gen = slot.gen.wrapping_add(1);
        self.free_stack.push(id.idx);
        Some(slot.val)
    }

    #[inline]
    fn get(&self, id: GenId) -> Option<&Widget> {
        if (id.gen & 1) == 0 {
            return None;
        }
        let slot = self.slots.get(id.idx as usize)?;
        if slot.gen == id.gen {
            Some(&slot.val)
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(&mut self, id: GenId) -> Option<&mut Widget> {
        if (id.gen & 1) == 0 {
            return None;
        }
        let slot = self.slots.get_mut(id.idx as usize)?;
        if slot.gen == id.gen {
            Some(&mut slot.val)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
struct LayoutBox {
    widget: GenId,
    tag: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    measure_w: u32,
    measure_h: u32,
    flex: u32,
    pad: u32,
    gap: u32,
    is_row: u8,
    first_child: u32,
    next_sibling: u32,
}

const NO_BOX: u32 = u32::MAX;

#[inline]
fn rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn unlink_child(tree: &mut SlotPool, parent_id: GenId, target: GenId) {
    let first_opt = tree.get(parent_id).and_then(|p| p.first_child);
    let target_next = tree.get(target).and_then(|t| t.next_sibling);

    let Some(first) = first_opt else {
        return;
    };
    if first == target {
        if let Some(p) = tree.get_mut(parent_id) {
            p.first_child = target_next;
        }
        return;
    }

    let mut prev = first;
    loop {
        let next_opt = tree.get(prev).and_then(|c| c.next_sibling);
        let Some(next_id) = next_opt else {
            break;
        };
        if next_id == target {
            if let Some(p) = tree.get_mut(prev) {
                p.next_sibling = target_next;
            }
            return;
        }
        prev = next_id;
    }
}

fn main() {
    const NUM_CONTAINERS: usize = 4096;
    const NUM_LEAVES: usize = 12288;
    const TOTAL_WIDGETS: usize = 16384;
    const NUM_FRAMES: usize = 120;
    const CHURN_PER_FRAME: usize = 64;
    const MUTATIONS_PER_FRAME: usize = 128;
    const VIEW_W: u32 = 1920;
    const VIEW_H: u32 = 1080;

    let mut tree = SlotPool::with_capacity(TOTAL_WIDGETS);
    let mut containers: Vec<GenId> = Vec::with_capacity(NUM_CONTAINERS);
    let mut leaves: Vec<GenId> = Vec::with_capacity(NUM_LEAVES);
    let mut boxes: Vec<LayoutBox> = Vec::with_capacity(TOTAL_WIDGETS);

    let mut rng: u64 = 0xD1B54A32D192ED03;
    let mut next_tag: u32 = 1;

    // Build deterministic persistent widget tree (excluded from timed section).
    for i in 0..NUM_CONTAINERS {
        let r = rng_next(&mut rng);
        let min_w = 24u32.wrapping_add((r & 63) as u32);
        let min_h = 16u32.wrapping_add(((r >> 6) & 31) as u32);
        let pad = (((r >> 12) & 3) as u32).wrapping_add(1);
        let gap = ((r >> 14) & 3) as u32;
        let flex = ((r >> 16) & 3) as u32;
        let is_row = if (i & 1) == 0 { 0u8 } else { 1u8 };
        let tag = next_tag;
        next_tag = next_tag.wrapping_add(1);

        let parent_opt = if i > 0 {
            Some(containers[(i - 1) >> 2])
        } else {
            None
        };

        let wid = tree.insert(Widget {
            tag,
            min_w,
            min_h,
            pad,
            gap,
            flex,
            is_row,
            parent: parent_opt,
            first_child: None,
            next_sibling: None,
        });

        if let Some(pid) = parent_opt {
            let old_first = tree.get(pid).and_then(|p| p.first_child);
            if let Some(w) = tree.get_mut(wid) {
                w.next_sibling = old_first;
            }
            if let Some(p) = tree.get_mut(pid) {
                p.first_child = Some(wid);
            }
        }
        containers.push(wid);
    }

    for k in 0..NUM_LEAVES {
        let r = rng_next(&mut rng);
        let min_w = 16u32.wrapping_add((r & 63) as u32);
        let min_h = 12u32.wrapping_add(((r >> 6) & 31) as u32);
        let flex = ((r >> 12) & 3) as u32;
        let tag = next_tag;
        next_tag = next_tag.wrapping_add(1);

        let pidx = k & 4095;
        let pid = containers[pidx];
        let old_first = tree.get(pid).and_then(|p| p.first_child);
        let wid = tree.insert(Widget {
            tag,
            min_w,
            min_h,
            pad: 0,
            gap: 0,
            flex,
            is_row: 0,
            parent: Some(pid),
            first_child: None,
            next_sibling: old_first,
        });
        if let Some(p) = tree.get_mut(pid) {
            p.first_child = Some(wid);
        }
        leaves.push(wid);
    }

    let root = containers[0];

    // Timed multi-frame layout solve + tree churn workload.
    let t0 = Instant::now();
    let mut checksum: u64 = 0xCBF29CE484222325;

    for _frame in 0..NUM_FRAMES {
        // 1. Dynamic leaf remove + ABA verification + re-insert churn in generational SlotPool.
        for _c in 0..CHURN_PER_FRAME {
            let li = (rng_next(&mut rng) as usize) % NUM_LEAVES;
            let old_id = leaves[li];

            let old_parent_opt = tree.get(old_id).and_then(|w| w.parent);
            if let Some(old_pid) = old_parent_opt {
                unlink_child(&mut tree, old_pid, old_id);
            }
            let _ = tree.remove(old_id);

            // Confirm generational ABA rejection on stale handle.
            if tree.get(old_id).is_some() {
                checksum ^= 0xDEAD;
            } else {
                checksum = checksum.wrapping_add(1);
            }

            let new_pidx = (rng_next(&mut rng) as usize) & 4095;
            let new_pid = containers[new_pidx];
            let r = rng_next(&mut rng);
            let min_w = 16u32.wrapping_add((r & 63) as u32);
            let min_h = 12u32.wrapping_add(((r >> 6) & 31) as u32);
            let flex = ((r >> 12) & 3) as u32;
            let tag = next_tag;
            next_tag = next_tag.wrapping_add(1);

            let old_first = tree.get(new_pid).and_then(|p| p.first_child);
            let new_id = tree.insert(Widget {
                tag,
                min_w,
                min_h,
                pad: 0,
                gap: 0,
                flex,
                is_row: 0,
                parent: Some(new_pid),
                first_child: None,
                next_sibling: old_first,
            });
            if let Some(p) = tree.get_mut(new_pid) {
                p.first_child = Some(new_id);
            }
            leaves[li] = new_id;
        }

        // 2. Per-frame widget size & flex mutations.
        for m in 0..MUTATIONS_PER_FRAME {
            let r = rng_next(&mut rng);
            if (m & 3) == 0 {
                let ci = (r as usize) & 4095;
                let cid = containers[ci];
                if let Some(w) = tree.get_mut(cid) {
                    w.min_w = 20u32.wrapping_add(((r >> 16) & 127) as u32);
                    w.min_h = 16u32.wrapping_add(((r >> 24) & 63) as u32);
                    w.flex = ((r >> 32) & 3) as u32;
                }
            } else {
                let li = (r as usize) % NUM_LEAVES;
                let lid = leaves[li];
                if let Some(w) = tree.get_mut(lid) {
                    w.min_w = 12u32.wrapping_add(((r >> 16) & 95) as u32);
                    w.min_h = 10u32.wrapping_add(((r >> 24) & 47) as u32);
                    w.flex = ((r >> 32) & 3) as u32;
                }
            }
        }

        // 3. Reset per-frame layout Vec and build BFS topological snapshot.
        boxes.clear();

        if let Some(rrow) = tree.get(root) {
            boxes.push(LayoutBox {
                widget: root,
                tag: rrow.tag,
                x: 0,
                y: 0,
                w: rrow.min_w,
                h: rrow.min_h,
                measure_w: 0,
                measure_h: 0,
                flex: rrow.flex,
                pad: rrow.pad,
                gap: rrow.gap,
                is_row: rrow.is_row,
                first_child: NO_BOX,
                next_sibling: NO_BOX,
            });
        }

        let mut qi: usize = 0;
        while qi < boxes.len() {
            let wid = boxes[qi].widget;
            let mut child_opt = tree.get(wid).and_then(|w| w.first_child);
            let mut last_box = NO_BOX;
            while let Some(cid) = child_opt {
                if let Some(crow) = tree.get(cid) {
                    let cbox = boxes.len() as u32;
                    boxes.push(LayoutBox {
                        widget: cid,
                        tag: crow.tag,
                        x: 0,
                        y: 0,
                        w: crow.min_w,
                        h: crow.min_h,
                        measure_w: 0,
                        measure_h: 0,
                        flex: crow.flex,
                        pad: crow.pad,
                        gap: crow.gap,
                        is_row: crow.is_row,
                        first_child: NO_BOX,
                        next_sibling: NO_BOX,
                    });
                    if last_box != NO_BOX {
                        boxes[last_box as usize].next_sibling = cbox;
                    } else {
                        boxes[qi].first_child = cbox;
                    }
                    last_box = cbox;
                    child_opt = crow.next_sibling;
                } else {
                    break;
                }
            }
            qi += 1;
        }

        let nboxes = boxes.len();

        // 4. Pass 1 (Bottom-Up): compute intrinsic (measure_w, measure_h) from leaves to root.
        for rev in (0..nboxes).rev() {
            let is_row = boxes[rev].is_row;
            let pad = boxes[rev].pad;
            let gap = boxes[rev].gap;

            let mut content_w: u32 = 0;
            let mut content_h: u32 = 0;
            let mut child_count: u32 = 0;

            let mut c_idx = boxes[rev].first_child;
            while c_idx != NO_BOX {
                let c = &boxes[c_idx as usize];
                let cmw = c.measure_w;
                let cmh = c.measure_h;
                if is_row == 1 {
                    content_w = content_w.wrapping_add(cmw);
                    if cmh > content_h {
                        content_h = cmh;
                    }
                } else {
                    if cmw > content_w {
                        content_w = cmw;
                    }
                    content_h = content_h.wrapping_add(cmh);
                }
                child_count = child_count.wrapping_add(1);
                c_idx = c.next_sibling;
            }

            if child_count > 1 {
                let gap_total = child_count.wrapping_sub(1).wrapping_mul(gap);
                if is_row == 1 {
                    content_w = content_w.wrapping_add(gap_total);
                } else {
                    content_h = content_h.wrapping_add(gap_total);
                }
            }

            let pad2 = pad.wrapping_mul(2);
            let req_w = content_w.wrapping_add(pad2);
            let req_h = content_h.wrapping_add(pad2);
            let b = &mut boxes[rev];
            b.measure_w = req_w.max(b.w);
            b.measure_h = req_h.max(b.h);
        }

        // 5. Pass 2 (Top-Down): flex distribution & rect placement from root to leaves.
        if nboxes > 0 {
            let rmw = boxes[0].measure_w;
            let rmh = boxes[0].measure_h;
            boxes[0].x = 0;
            boxes[0].y = 0;
            boxes[0].w = rmw.max(VIEW_W);
            boxes[0].h = rmh.max(VIEW_H);
        }

        for i in 0..nboxes {
            let first_c = boxes[i].first_child;
            if first_c == NO_BOX {
                continue;
            }

            let bx = boxes[i].x;
            let by = boxes[i].y;
            let bw = boxes[i].w;
            let bh = boxes[i].h;
            let pad = boxes[i].pad;
            let gap = boxes[i].gap;
            let is_row = boxes[i].is_row;

            let pad2 = pad.wrapping_mul(2);
            let inner_x = bx.wrapping_add(pad);
            let inner_y = by.wrapping_add(pad);
            let inner_w = if bw > pad2 { bw.wrapping_sub(pad2) } else { 0 };
            let inner_h = if bh > pad2 { bh.wrapping_sub(pad2) } else { 0 };

            let mut total_main: u32 = 0;
            let mut total_flex: u32 = 0;
            let mut child_count: u32 = 0;

            let mut c_idx = first_c;
            while c_idx != NO_BOX {
                let c = &boxes[c_idx as usize];
                if is_row == 1 {
                    total_main = total_main.wrapping_add(c.measure_w);
                } else {
                    total_main = total_main.wrapping_add(c.measure_h);
                }
                total_flex = total_flex.wrapping_add(c.flex);
                child_count = child_count.wrapping_add(1);
                c_idx = c.next_sibling;
            }

            if child_count > 1 {
                total_main = total_main.wrapping_add(child_count.wrapping_sub(1).wrapping_mul(gap));
            }

            let avail_main = if is_row == 1 { inner_w } else { inner_h };
            let extra = if avail_main > total_main {
                avail_main.wrapping_sub(total_main)
            } else {
                0
            };
            let total_flex_u64 = total_flex as u64;
            let extra_u64 = extra as u64;

            let mut cur = if is_row == 1 { inner_x } else { inner_y };
            c_idx = first_c;
            while c_idx != NO_BOX {
                let c = &mut boxes[c_idx as usize];
                let cf = c.flex;
                let bonus = if total_flex_u64 > 0 && cf > 0 {
                    (extra_u64.wrapping_mul(cf as u64) / total_flex_u64) as u32
                } else {
                    0
                };
                let cmw = c.measure_w;
                let cmh = c.measure_h;
                if is_row == 1 {
                    let cw = cmw.wrapping_add(bonus);
                    let ch = inner_h.max(cmh);
                    c.x = cur;
                    c.y = inner_y;
                    c.w = cw;
                    c.h = ch;
                    cur = cur.wrapping_add(cw).wrapping_add(gap);
                } else {
                    let cw = inner_w.max(cmw);
                    let ch = cmh.wrapping_add(bonus);
                    c.x = inner_x;
                    c.y = cur;
                    c.w = cw;
                    c.h = ch;
                    cur = cur.wrapping_add(ch).wrapping_add(gap);
                }
                c_idx = c.next_sibling;
            }
        }

        // 6. Hash computed (tag, x, y, w, h) rects into checksum.
        let mut frame_hash: u64 = 0x9E3779B97F4A7C15;
        for b in &boxes {
            let v1 = ((b.tag as u64) << 32) | (b.x as u64);
            let v2 = ((b.y as u64) << 32) | (b.w as u64);
            let v3 = b.h as u64;
            frame_hash = (frame_hash ^ v1).wrapping_mul(0x100000001B3);
            frame_hash = (frame_hash ^ v2).wrapping_mul(0x100000001B3);
            frame_hash = (frame_hash ^ v3).wrapping_mul(0x100000001B3);
        }
        checksum = (checksum ^ frame_hash).wrapping_mul(0x100000001B3);
    }

    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    println!("RESULT ui_layout checksum={checksum} elapsed_ns={elapsed_ns}");
}
