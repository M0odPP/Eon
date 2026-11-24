#!/usr/bin/env python3
"""
Phase 10.2 Verification & Benchmark Suite:
  - Zero-libc ELF audit (`readelf -d`) on `ewc`, `egrep`, `ehttpd`
  - Binary size comparison (Eon static-PIE zero-libc vs Rust opt-level=3 fat-LTO)
  - `ewc` & `egrep` throughput (GB/s), peak RSS, and bit-exact match verification
  - `ehttpd` (multi-core `io_uring`) vs safe Rust HTTP/1.1 server: RPS, p50/p99 latency, peak RSS
  - `--sanitize` stress tests on `ewc`, `egrep`, and `ehttpd`
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "utils" / "out"
RUST_FLAGS = ["-C", "opt-level=3", "-C", "lto=fat", "-C", "codegen-units=1", "-C", "panic=abort", "-C", "target-cpu=native"]

LOADGEN_C = r"""
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int g_port = 18080;
static int g_reqs_per_conn = 2500;
static const char REQ[] = "GET /bench HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n";

struct WorkerArg {
    int id;
    uint64_t *lats_ns;
    int completed;
    int errors;
};

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *worker_fn(void *p) {
    struct WorkerArg *a = (struct WorkerArg *)p;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { a->errors++; return NULL; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)g_port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        a->errors++;
        close(fd);
        return NULL;
    }
    char rbuf[512];
    size_t req_len = sizeof(REQ) - 1;
    for (int i = 0; i < g_reqs_per_conn; i++) {
        uint64_t t0 = now_ns();
        ssize_t w = send(fd, REQ, req_len, 0);
        if (w != (ssize_t)req_len) { a->errors++; break; }
        ssize_t got = 0;
        while (got < 80) {
            ssize_t r = recv(fd, rbuf + got, sizeof(rbuf) - (size_t)got, 0);
            if (r <= 0) { a->errors++; goto done; }
            got += r;
            if (memmem(rbuf, (size_t)got, "Hello, Eon!\r\n", 13) != NULL) break;
        }
        uint64_t t1 = now_ns();
        a->lats_ns[a->completed++] = t1 - t0;
    }
done:
    close(fd);
    return NULL;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: loadgen <port> <conns> <reqs_per_conn>\n"); return 1; }
    g_port = atoi(argv[1]);
    int nconns = atoi(argv[2]);
    g_reqs_per_conn = atoi(argv[3]);
    pthread_t *ths = calloc((size_t)nconns, sizeof(pthread_t));
    struct WorkerArg *args = calloc((size_t)nconns, sizeof(struct WorkerArg));
    for (int i = 0; i < nconns; i++) {
        args[i].id = i;
        args[i].lats_ns = malloc((size_t)g_reqs_per_conn * sizeof(uint64_t));
    }
    uint64_t t0 = now_ns();
    for (int i = 0; i < nconns; i++) pthread_create(&ths[i], NULL, worker_fn, &args[i]);
    for (int i = 0; i < nconns; i++) pthread_join(ths[i], NULL);
    uint64_t t1 = now_ns();

    int total = 0, errs = 0;
    for (int i = 0; i < nconns; i++) { total += args[i].completed; errs += args[i].errors; }
    uint64_t *all = malloc((size_t)(total > 0 ? total : 1) * sizeof(uint64_t));
    int pos = 0;
    for (int i = 0; i < nconns; i++) {
        memcpy(all + pos, args[i].lats_ns, (size_t)args[i].completed * sizeof(uint64_t));
        pos += args[i].completed;
        free(args[i].lats_ns);
    }
    qsort(all, (size_t)total, sizeof(uint64_t), cmp_u64);
    double secs = (double)(t1 - t0) / 1e9;
    double rps = secs > 0.0 ? (double)total / secs : 0.0;
    double p50_us = total > 0 ? (double)all[total / 2] / 1000.0 : 0.0;
    double p99_us = total > 0 ? (double)all[(int)((double)total * 0.99)] / 1000.0 : 0.0;
    printf("{\"reqs\":%d,\"errors\":%d,\"wall_s\":%.4f,\"rps\":%.1f,\"p50_us\":%.2f,\"p99_us\":%.2f}\n",
           total, errs, secs, rps, p50_us, p99_us);
    free(all); free(args); free(ths);
    return errs > 0 ? 2 : 0;
}
"""


def check_zero_libc(bin_path: Path) -> None:
    out = subprocess.check_output(["readelf", "-d", str(bin_path)], text=True)
    needed = [line.strip() for line in out.splitlines() if "(NEEDED)" in line]
    if needed:
        raise RuntimeError(f"{bin_path} has dynamic NEEDED dependencies: {needed}")
    ph = subprocess.check_output(["readelf", "-l", str(bin_path)], text=True)
    if "INTERP" in ph:
        raise RuntimeError(f"{bin_path} has PT_INTERP!")


def read_proc_vmhwm_kib(pid: int) -> int:
    try:
        txt = Path(f"/proc/{pid}/status").read_text()
        m = re.search(r"VmHWM:\s+(\d+)\s+kB", txt)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


def run_with_rss(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, status, rusage = os.wait4(proc.pid, 0)
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    proc.stdout.close()
    proc.stderr.close()
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError(f"cmd failed: {cmd} status={status} err={err}")
    return out.strip(), int(rusage.ru_maxrss)


def generate_corpus(path: Path, size_mb: int = 32) -> None:
    lines = [
        b"Linear ownership and capability-based APIs eliminate hidden global state in systems code.\n",
        b"Structure-of-arrays pools align hot columns on cache lines for predictable traversal speed.\n",
        b"Explicit wrapping arithmetic and static bounds proofs prevent silent integer wrap bugs.\n",
        b"Eon_SIMD_v10_marker_42: vectorized AVX2 scanning accelerates wc and grep on Linux without libc.\n",
        b"Compile-time relational DBM verification removes redundant bounds checks inside tight loops.\n",
        b"Arena allocators reclaim entire HTTP request frames in constant time with a single pointer reset.\n",
        b"Zero-copy mmap slices keep memory traffic minimal while scanning large log files and corpuses.\n",
        b"io_uring thread-per-core static fibers deliver low p99 tail latency with minimal RSS footprint.\n",
    ]
    block = b"".join(lines * 64)
    target = size_mb * 1024 * 1024
    with open(path, "wb") as f:
        written = 0
        while written < target:
            f.write(block)
            written += len(block)


def wait_port(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"server on port {port} did not become ready")


def send_quit(port: int, n: int = 1) -> None:
    for _ in range(n):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5) as s:
                s.sendall(b"GET /quit HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                s.recv(256)
        except OSError:
            pass


def bench_http_server(name: str, cmd, port: int, loadgen: Path, conns: int, reqs_per_conn: int):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait for READY line
        ready_line = proc.stdout.readline().strip()
        if "READY" not in ready_line:
            time.sleep(0.1)
        wait_port(port)
        # Warmup
        subprocess.run([str(loadgen), str(port), "8", "200"], check=True, capture_output=True)
        # Measured run
        out = subprocess.check_output([str(loadgen), str(port), str(conns), str(reqs_per_conn)], text=True).strip()
        stats = json.loads(out)
        stats["rss_kib"] = read_proc_vmhwm_kib(proc.pid)
        # Trigger graceful shutdown across workers
        send_quit(port, n=8)
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=2.0)
        return stats
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eonc", default=str(ROOT / "selfhost/out/eonc2_rel"))
    ap.add_argument("--save", default=str(ROOT / "utils/results/phase10_utils.json"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eonc = args.eonc

    print("=== 1. Building Eon zero-libc utilities (--release) ===")
    for name, src in [
        ("ewc", ["utils/ewc/main.eon", "utils/sys/linux_sys.hal.eon"]),
        ("egrep", ["utils/egrep/main.eon", "utils/sys/linux_sys.hal.eon"]),
        ("ehttpd", ["utils/ehttpd/main.eon", "utils/sys/linux_sys.hal.eon"]),
    ]:
        out_bin = OUT_DIR / name
        subprocess.run([eonc, "build", "--release", *src, "-o", str(out_bin)], cwd=ROOT, check=True)
        check_zero_libc(out_bin)
        print(f"  [OK] {name}: {out_bin.stat().st_size} bytes (ZERO libc, ZERO PT_INTERP)")

    print("=== 2. Building safe Rust (#![forbid(unsafe_code)]) references ===")
    for name in ("ewc", "egrep", "ehttpd"):
        out_bin = OUT_DIR / f"{name}_rs"
        subprocess.run(["rustc", *RUST_FLAGS, str(ROOT / "utils/rust_ref" / f"{name}.rs"), "-o", str(out_bin)], check=True)
        print(f"  [OK] {name}_rs: {out_bin.stat().st_size} bytes")

    print("=== 3. Building & running --sanitize stress tests ===")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        small = tdp / "small.txt"
        generate_corpus(small, size_mb=1)
        for name, src in [
            ("ewc_san", ["utils/ewc/main.eon", "utils/sys/linux_sys.hal.eon"]),
            ("egrep_san", ["utils/egrep/main.eon", "utils/sys/linux_sys.hal.eon"]),
            ("ehttpd_san", ["utils/ehttpd/main.eon", "utils/sys/linux_sys.hal.eon"]),
        ]:
            sb = tdp / name
            subprocess.run([eonc, "build", "--sanitize", *src, "-o", str(sb)], cwd=ROOT, check=True)
        subprocess.run([str(tdp / "ewc_san"), str(small)], check=True, capture_output=True)
        subprocess.run([str(tdp / "egrep_san"), "-c", "Eon_SIMD_v10_marker_42", str(small)], check=True, capture_output=True)
        subprocess.run([str(tdp / "egrep_san"), "-c", "^Eon_SIMD.*42:", str(small)], check=True, capture_output=True)
        # Stress ehttpd_san with 2,000 live HTTP requests + graceful shutdown (verifies live_allocs == 0!)
        loadgen = tdp / "loadgen"
        lg_c = tdp / "loadgen.c"
        lg_c.write_text(LOADGEN_C)
        subprocess.run(["cc", "-O2", "-pthread", str(lg_c), "-o", str(loadgen)], check=True)
        san_srv = subprocess.Popen(
            [str(tdp / "ehttpd_san"), "--port", "18189", "--threads", "1", "--max-reqs", "2000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        san_srv.stdout.readline()
        wait_port(18189)
        subprocess.run([str(loadgen), "18189", "4", "500"], check=True, capture_output=True)
        san_out, san_err = san_srv.communicate(timeout=5.0)
        if san_srv.returncode != 0 or "SANITIZER" in san_err.upper() or "LEAK" in san_err.upper():
            raise RuntimeError(f"ehttpd --sanitize failed rc={san_srv.returncode} err={san_err}")
        print(f"  [OK] --sanitize stress: ewc=PASS, egrep(literal+regex)=PASS, ehttpd(2000 reqs)=PASS ({san_out.strip()})")

        print("=== 4. Benchmarking ewc & egrep (32 MiB corpus x 20 reps) ===")
        corpus = tdp / "corpus_32m.txt"
        generate_corpus(corpus, size_mb=32)
        corpus_bytes = corpus.stat().st_size

        ewc_eon_out, ewc_eon_rss = run_with_rss([str(OUT_DIR / "ewc"), "--bench", "20", str(corpus)])
        ewc_rs_out, ewc_rs_rss = run_with_rss([str(OUT_DIR / "ewc_rs"), "--bench", "20", str(corpus)])
        eg_lit_eon_out, eg_lit_eon_rss = run_with_rss([str(OUT_DIR / "egrep"), "--bench", "20", "Eon_SIMD_v10_marker_42", str(corpus)])
        eg_lit_rs_out, eg_lit_rs_rss = run_with_rss([str(OUT_DIR / "egrep_rs"), "--bench", "20", "Eon_SIMD_v10_marker_42", str(corpus)])
        eg_re_eon_out, eg_re_eon_rss = run_with_rss([str(OUT_DIR / "egrep"), "--bench", "3", "^Eon_SIMD.*42:", str(corpus)])
        eg_re_rs_out, eg_re_rs_rss = run_with_rss([str(OUT_DIR / "egrep_rs"), "--bench", "3", "^Eon_SIMD.*42:", str(corpus)])

        def parse_kv(s):
            return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", s)}

        kv_ewc_e, kv_ewc_r = parse_kv(ewc_eon_out), parse_kv(ewc_rs_out)
        assert (kv_ewc_e["lines"], kv_ewc_e["words"], kv_ewc_e["bytes"]) == (kv_ewc_r["lines"], kv_ewc_r["words"], kv_ewc_r["bytes"]), "ewc mismatch!"
        kv_egl_e, kv_egl_r = parse_kv(eg_lit_eon_out), parse_kv(eg_lit_rs_out)
        assert kv_egl_e["matches"] == kv_egl_r["matches"] and kv_egl_e["matches"] > 0, "egrep literal mismatch!"
        kv_egr_e, kv_egr_r = parse_kv(eg_re_eon_out), parse_kv(eg_re_rs_out)
        assert kv_egr_e["matches"] == kv_egr_r["matches"] == kv_egl_e["matches"], "egrep regex mismatch!"

        print("=== 5. Benchmarking ehttpd (4-core io_uring) vs Rust ehttpd_rs (4-core) ===")
        http_eon = bench_http_server(
            "ehttpd",
            [str(OUT_DIR / "ehttpd"), "--port", "18190", "--threads", "4"],
            18190, loadgen, conns=32, reqs_per_conn=3000,
        )
        http_rs = bench_http_server(
            "ehttpd_rs",
            [str(OUT_DIR / "ehttpd_rs"), "--port", "18191", "--threads", "4"],
            18191, loadgen, conns=32, reqs_per_conn=3000,
        )

    def gbps(bytes_total, ns):
        return (bytes_total / (ns / 1e9)) / (1024**3) if ns > 0 else 0.0

    results = {
        "sizes_bytes": {
            "ewc_eon": (OUT_DIR / "ewc").stat().st_size,
            "ewc_rust": (OUT_DIR / "ewc_rs").stat().st_size,
            "egrep_eon": (OUT_DIR / "egrep").stat().st_size,
            "egrep_rust": (OUT_DIR / "egrep_rs").stat().st_size,
            "ehttpd_eon": (OUT_DIR / "ehttpd").stat().st_size,
            "ehttpd_rust": (OUT_DIR / "ehttpd_rs").stat().st_size,
        },
        "ewc": {
            "eon_gbps": round(gbps(corpus_bytes * 20, kv_ewc_e["ns"]), 3),
            "rust_gbps": round(gbps(corpus_bytes * 20, kv_ewc_r["ns"]), 3),
            "ratio_time": round(kv_ewc_e["ns"] / kv_ewc_r["ns"], 4),
            "eon_rss_kib": ewc_eon_rss,
            "rust_rss_kib": ewc_rs_rss,
            "lines": kv_ewc_e["lines"],
            "words": kv_ewc_e["words"],
            "bytes": kv_ewc_e["bytes"],
        },
        "egrep_literal": {
            "eon_gbps": round(gbps(corpus_bytes * 20, kv_egl_e["ns"]), 3),
            "rust_gbps": round(gbps(corpus_bytes * 20, kv_egl_r["ns"]), 3),
            "ratio_time": round(kv_egl_e["ns"] / kv_egl_r["ns"], 4),
            "eon_rss_kib": eg_lit_eon_rss,
            "rust_rss_kib": eg_lit_rs_rss,
            "matches": kv_egl_e["matches"],
        },
        "egrep_regex": {
            "eon_gbps": round(gbps(corpus_bytes * 3, kv_egr_e["ns"]), 3),
            "rust_gbps": round(gbps(corpus_bytes * 3, kv_egr_r["ns"]), 3),
            "ratio_time": round(kv_egr_e["ns"] / kv_egr_r["ns"], 4),
            "matches": kv_egr_e["matches"],
        },
        "ehttpd": {
            "eon": http_eon,
            "rust": http_rs,
            "rps_speedup": round(http_eon["rps"] / max(http_rs["rps"], 1.0), 3),
        },
        "sanitize": "PASS",
        "zero_libc": "PASS",
    }

    save_p = Path(args.save)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    save_p.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
