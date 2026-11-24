# `compiler/` (`eonc0`) — Bootstrap Seed Only

As of **Phase 8** (Triple Bootstrap `SHA-256(eonc2) == SHA-256(eonc3)` verified bit-for-bit on Linux `x86_64`), the Rust compiler in `compiler/` is frozen in **Bootstrap Seed Only** mode.

- Role: compile `selfhost/` (`eonc0` -> `eonc1`) as the reproducible Stage-0 seed.
- Policy: all new compiler passes, the optimizing pipeline (`EIR-Opt` + `MachIR`, Phase 9), low-level Linux/microkernel features (Phase 10), and future target backends live exclusively in `selfhost/` written in 100% Safe Eon.
- Verification: `python3 tools/bootstrap_stage3.py` verifies `eonc0 -> eonc1 -> eonc2 -> eonc3` and asserts `SHA-256(eonc2) == SHA-256(eonc3)`.
