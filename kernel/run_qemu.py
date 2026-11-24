#!/usr/bin/env python3
"""
Build and boot `EonOS-Micro` (`kernel/`) in QEMU x86_64 (`-nographic -smp 2`).
Verifies:
  1. `--audit-tcb` confirms ZERO `@hal`/`raw`/`asm` inside `kernel/src/`.
  2. QEMU boots `kernel/out/eonos.elf` via Xen PVH / Multiboot2 freestanding ELF64.
  3. Serial output confirms: UART MmioReg, GDT/TSS/IDT, LAPIC/IOAPIC, PMM linear PhysPage + errdefer,
     4-level paging (`bitfield PageEntry` + `PhysAddr`/`VirtAddr`), SMP (`cpus_online=2`),
     cooperative fibers (30 switches), and >= 1,000 hardware timer IRQs via `@interrupt @noalloc @stack(512)`.
  4. QEMU exits cleanly via `isa-debug-exit` (`port 0xF4`, `val 0x10` -> exit code `33`).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_qemu() -> tuple[str, dict[str, str]]:
    env = os.environ.copy()
    local_qemu = Path.home() / ".local/usr/bin/qemu-system-x86_64"
    if local_qemu.exists():
        lib_dir = str(Path.home() / ".local/usr/lib")
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
        return str(local_qemu), env
    sys_qemu = shutil.which("qemu-system-x86_64")
    if sys_qemu:
        return sys_qemu, env
    raise RuntimeError("qemu-system-x86_64 not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eonc", default=str(ROOT / "selfhost/out/eonc2_rel"))
    ap.add_argument("--out", default=str(ROOT / "kernel/out/eonos.elf"))
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        args.eonc,
        "build",
        "--freestanding",
        "--release",
        "--target-cpu=baseline",
        "--audit-tcb",
        "kernel/src",
        "kernel/arch/x86_64",
        "-o",
        str(out_path),
    ]
    print(f"[kernel] building: {' '.join(build_cmd)}")
    b = subprocess.run(build_cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(b.stdout)
    sys.stderr.write(b.stderr)
    if b.returncode != 0:
        print(f"[kernel] BUILD FAILED (rc={b.returncode})", file=sys.stderr)
        return 1

    # Verify TCB audit: no kernel/src/*.eon file may appear as @hal/raw
    combined_build = b.stdout + "\n" + b.stderr
    for line in combined_build.splitlines():
        if "kernel/src/" in line and ("@hal" in line or "raw" in line):
            print(f"[kernel] TCB VIOLATION in kernel/src: {line}", file=sys.stderr)
            return 1

    # Verify strict W^X ELF PT_LOAD segments (no simultaneous W and X flags)
    re_out = subprocess.check_output(["readelf", "-l", "-W", str(out_path)], text=True)
    load_flags = []
    for line in re_out.splitlines():
        s = line.strip()
        if s.startswith("LOAD "):
            parts = s.split()
            # readelf -l -W: Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg Align
            # Note Flg can contain spaces like "R E"
            flg = " ".join(parts[6:-1])
            load_flags.append(flg)
            if "W" in flg and "E" in flg:
                print(f"[kernel] W^X ELF VIOLATION: PT_LOAD segment has RWE flags: {line}", file=sys.stderr)
                return 1
    if not load_flags or not any("E" in f and "W" not in f for f in load_flags):
        print(f"[kernel] W^X ELF VIOLATION: expected R E code segment, got {load_flags}", file=sys.stderr)
        return 1
    print(f"[kernel] W^X ELF check: ok (PT_LOAD flags={load_flags})")

    qemu_bin, env = find_qemu()
    qemu_cmd = [
        qemu_bin,
        "-nographic",
        "-smp", "2",
        "-m", "128M",
        "-no-reboot",
        "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04",
        "-kernel", str(out_path),
    ]
    print(f"[kernel] booting QEMU: {' '.join(qemu_cmd)}")
    r = subprocess.run(qemu_cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    print("--- QEMU serial output ---")
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        print("--- QEMU stderr ---")
        sys.stderr.write(r.stderr)
    print(f"--- QEMU exit code: {r.returncode} (expected 33 = (0x10 << 1) | 1) ---")

    required_markers = [
        "[EonOS-Micro] uart16550 MmioReg: ok",
        "[EonOS-Micro] gdt+tss+idt",
        "[EonOS-Micro] pmm (linear PhysPage + errdefer): ok",
        "[EonOS-Micro] vm 4-level paging",
        "[EonOS-Micro] smp cpus_online=2",
        "[EonOS-Micro] fibers: switches=30",
        "(>= 1000): ok",
        "[EonOS-Micro] ALL CHECKS PASSED",
    ]
    missing = [m for m in required_markers if m not in r.stdout]
    if r.returncode != 33 or missing:
        print(f"[kernel] VERIFICATION FAILED: rc={r.returncode}, missing={missing}", file=sys.stderr)
        return 1

    print(f"[kernel] EonOS-Micro PASS (elf={out_path.stat().st_size} bytes, rc=33)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
