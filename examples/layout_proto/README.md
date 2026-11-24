# EonUI Layout Prototype

Console layout prototype compiled and executed through the zero-C Eon pipeline (`Typed CFG -> CFG Analysis -> SSA EIR -> DBM Verifier -> Native x86_64 / Built-in ELF64 PIE Linker` or `EIR-Interp`), with zero `clang`, `gcc`, `ld`, or `libc` dependencies on Linux `x86_64`.

## Build & Run (Zero-C)

```bash
# Default zero-C native ELF64 PIE executable (ElfMode::LinuxPie)
eonc build examples/layout_proto -o layout_proto
./layout_proto

# Build and run directly on native x86_64
eonc run examples/layout_proto

# Reference SSA EIR interpreter (semantic oracle)
eonc run --interp examples/layout_proto

# Native build with built-in memory sanitizer (PROT_NONE guard pages, 0xDE poison, 64-entry quarantine, sigaltstack backtraces)
eonc build --sanitize examples/layout_proto -o layout_proto_san
./layout_proto_san
```

## Workload & Architecture
- **Persistent scene tree:** `SlotPool.[TreeNode]` with `Link[Self]` columns (`parent`, `first_child`, `next_sibling`, resolved to 8-byte `GenId[TreeNode]`). Exercises ABA protection (insert -> remove -> stale `get` rejected), tree mutation on every frame (`tree.width[right_pane]` toggled and settled to canonical `10` on frame 99), and deterministic cleanup via `destroy(alloc)` + `errdefer`.
- **Per-frame layout buffers:** `FrameArena` (4096 B) backing `ArenaPool.[LayoutBox]` (`Link[Self]` resolved to 4-byte `Id[boxes]`) reset across 100 frames without additional heap allocations.
- **2D rasterization:** 8x24 ASCII framebuffer (`[192]u8`), combining checked linear indexing in `fill_rect` (`grid.at_mut(idx)`) with DBM-proven zero-check indexing (`gy * stride + gx` and `grid[cell_u.widen.[usize]()]`) in the final raster pass.
- **Error-path resource safety:** `errdefer_probe_step(alloc, true)` forces an error return after allocating a `FrameArena` and asserts `err_cleanup_leaks=0` via `alloc.live_allocs()`.

## Deterministic Workload Invariants
Verified across `EIR-Interp`, native `x86_64`, and `--sanitize`:
- `frames=100`, `boxes=5`
- Frame arena footprint: `arena_used=384` / `arena_cap=4096`, `arena_after_reset=0`
- Allocator peak / leaks: `peak_bytes=4488`, `err_cleanup_leaks=0`, `final_live_allocs=0`
- ABA stale-handle rejection: `aba_rejected=1`

## Phase 3A -> Phase 3B / Phase 6 Evolution
1. **Checked vs. DBM-proven arithmetic (`ZM-08`):** `fill_rect` uses explicit checked operators (`+?`, `*?`) + `try` propagation, while the final raster loop (`for gy in 0_u32..8_u32` / `for gx in 0_u32..24_u32`) uses relational DBM bounds (`gy * stride + gx`) with zero runtime overflow checks.
2. **Checked vs. DBM-proven slice indexing (`ZM-07`):** `grid.at_mut(idx)` exercises checked slice access in `fill_rect`, while `grid[cell_u.widen.[usize]()]` in the raster pass compiles without bounds checks via DBM verification.
3. **Explicit signedness/width conversions (`ZM-05`):** Layout coordinates use `u32` with explicit `.widen.[usize]()` / `.widen.[u64]()` conversions.
4. **Data layout split:** Separating long-lived tree state (`SlotPool` with generational IDs) from transient frame geometry (`ArenaPool` with branded 4-byte IDs and O(1) `reset()`) eliminates per-frame heap traffic and keeps `peak_bytes` flat across all 100 frames.
