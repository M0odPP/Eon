# 07. Effect System, Capability-Based Security, and Non-Blocking I/O

This chapter addresses two critical requirements:
1. One and the same language runs inside an OS kernel interrupt handler, a UI application, and a third-party library without risk of covert data exfiltration or hidden allocation.
2. Programs handle tens of thousands of network and disk operations on a "1 thread = 1 core" scheduler **without `async/await` and without `Pin`**.

---

## 1. Effect System (Static Effect Checking)

Instead of splitting into a "safe dialect," a "kernel dialect," and a "shader dialect," Eon uses a **static effect system**. Every function carries an **effect set** verified across the entire call graph.

### 1.1. Built-In Effects and Restrictions

| Annotation | What the Compiler Guarantees (and Forbids in the Body + All Callees) |
| :--- | :--- |
| **`@pure`** | Zero side effects: no writes through `*mut`, no atomics, no I/O, no allocations. Output depends strictly on inputs. |
| **`@noalloc`** | Forbids calling any function that takes `AllocCap` / `*mut Allocator` or allocates on the heap. (Stack storage and a passed fixed-buffer `*mut Arena` are allowed, provided the arena does not grow via the OS.) |
| **`@noblock`** | Forbids any operation capable of suspending the current fiber/core (waiting on an I/O ring, waiting on a channel, `yield_fiber`). |
| **`@interrupt`** | Includes `@noalloc` + `@noblock` + forbids `@recursive` + mandates an explicit `@stack(N)` + saves all clobbered registers (including forbidding FPU/SIMD unless `@save_simd` is specified). |
| **`@stack(N)`** | Mathematically proven that the sum of stack frames of this function and **any** call chain from it (including `dyn Iface`) does not exceed `N` bytes (`E0701`). Forbids calling `@recursive`. |
| **`@recursive`** | Explicitly allows direct or mutual recursion (forbidden by default, `ZM-13`). Disables static `@stack(N)` computation. |
| **`@gpu`** | Function compiles to GPU code (SPIR-V/MSL/HLSL). Includes `@noalloc` + `@noblock` + forbids recursion + forbids syscalls and pointers to CPU memory. |

### 1.2. Effect Inference and Subtyping Rule
* Internal (private) functions may omit annotations — the compiler **automatically infers** their actual effects and exact stack size in bytes!
* Public functions (`pub fn`) and `iface` methods **must** declare their effect contracts explicitly so that an internal implementation change cannot accidentally break callers (for example, an OS kernel).
* **Subtyping rule:** a stricter function (e.g., `@pure @noalloc @stack(128)`) may always be called from a less strict one (an ordinary function), but **never vice versa** (`E0112`).

### 1.3. Effects of Closures and Higher-Order Functions
A closure (`|x| { ... }`) is a second-class value with its own effect set and stack usage. A higher-order function declares that its effects depend on a passed closure via the `@effects_of(f)` attribute, and its stack bound via `@stack_of(f)`:
```
pub fn drain[T](buf: LinearBuf[T], alloc: *mut AllocCap, f: fn(T))
    @effects_of(f) @stack(96 + @stack_of(f)) { ... }
```
Rules:
* The effects of calling `drain(...)` = the body's own effects (excluding the call to `f`) ∪ the effects of `f` **at the call site**. Therefore, `drain` with an allocating closure cannot be called from a `@noalloc` function (`E0112`), whereas calling it with a pure closure is allowed.
* `@stack_of(f)` is the computed stack ceiling of the closure; if the closure is fully inlined, the additive term is `0`. For non-inlined closures (`{ fn_ptr, env_ptr }`), `@stack(N)` verification is performed at the call site, where the closure body is known.
* For private functions, `@effects_of` / `@stack_of` are inferred automatically; for `pub` and `iface`, they must be specified explicitly. There are no other effect variables in the language.

---

## 2. Object-Capability Security

Because Eon has no global variables and no hidden syscalls (in `.eon`, you cannot "just call `open()` from libc"), access to the outside world is possible **solely through capability objects (`Capabilities`)**.

### 2.1. Program Entry Point
```
pub fn main(env: *mut Env) -> !AppError {
    // `Env` is a struct of capability VALUES (fields are values, not references; E0310 is not violated).
    // All world authorities originate ONLY here:
    //   env.alloc : AllocCap   (borrowed as &mut env.alloc)
    //   env.fs    : FsCap
    //   env.net   : NetCap
    //   env.clock : ClockCap   (TrivialCopy handle)
    //   env.gpu   : GpuCap
    //   env.io    : IoCap      (TrivialCopy handle to the current core's I/O ring; see §3.1)

    // Pass ONLY an arena to the JSON parser — it physically CANNOT open a file or socket!
    let doc = try json.parse(input_bytes, &mut frame_arena);

    // Give the networking module an attenuated capability restricted to ONE host:
    let api_cap = env.net.restrict_host("api.example.com", 443_u16);
    try client.fetch(api_cap, env.io);
    return;
}
```

### 2.2. Capability Attenuation
Every capability supports creating an **attenuated child capability**:
* `fs.sub_dir("assets/") -> DirCap` — a UI texture library sees **only** the `assets/` folder and physically cannot read `~/.ssh/id_rsa` (not even via `../../`, because paths resolve relative to an `openat` descriptor / EonOS capability handle).
* `alloc.sub_budget(16 * 1024 * 1024) -> AllocCap` — caps a subsystem to at most 16 MB of RAM (returning `OutOfMemory` on overflow, protecting the OS and UI against DoS).

---

## 3. Non-Blocking I/O Without `async/await`: I/O Rings + Static Fibers

In Rust, `async/await` splits the ecosystem into two incompatible worlds (sync vs. async), requires `Pin`/`Unpin`, produces enormous state machines, and leads to hidden `Box<dyn Future>` allocations. In Go, goroutines use growable stacks (requiring prologue checks on every function call and a complex runtime with a garbage collector).

Eon unites **two clean mechanisms** on top of a single "1 thread = 1 CPU core" scheduler:

### 3.1. Low Level: Asynchronous I/O Rings (`IoRing`)
A unified abstraction over:
* **EonOS:** native zero-copy IPC/NVMe/NIC rings directly in shared kernel memory.
* **Linux:** `io_uring` (0 syscalls on the hot path via `SQPOLL`).
* **Windows:** I/O Ring API (`CreateIoRing`) / IOCP.
* **macOS:** `kqueue` + I/O pool.

**Who holds what.** The ring belongs to the CPU core (the scheduler) and is modified only by code executing on that core. Application code **never** holds `*mut IoRing`: it holds `IoCap` — an 8-byte `TrivialCopy` handle to the current core's ring, which can be freely copied into hundreds of fibers (without violating `*mut` exclusivity, because `*mut` to the ring exists solely inside the scheduler, `core.sched`, `@hal`). Direct use of `*mut IoRing` is permitted only in `@noblock` code (drivers, high-load proxies), where suspension is impossible:
```
let req_id = try io.submit_read(fd, &mut dma_buf, offset);
// ... perform useful CPU work ...
let done = io.poll_completion();
```

#### Standard Binary Layout of `IoRing` Entries (64 Bytes = 1 Cache Line)
So that I/O rings work zero-copy without translation on EonOS and map 1-to-1 onto `io_uring` on Linux, submission queue entries (`IoSqe`) and completion queue entries (`IoCqe`) have a fixed aligned layout:
```
@align(64)
pub struct IoSqe {
    pub op:        IoOp;      // u8: Read, Write, Accept, Connect, Timeout, Fsync, IpcSend
    pub flags:     u8;        // FixedBuf, LinkNext, Drain
    pub _pad:      u16;
    pub fd:        u32;       // handle / fd / channel id
    pub offset:    u64;       // file offset or timeout in ns
    pub buf_addr:  u64;       // buffer address (verified by the compiler until CQE completion!)
    pub buf_len:   u32;
    pub op_flags:  u32;
    pub user_data: u64;       // request tag or FiberId for automatic wakeup
    pub _ext:      [24]u8;    // extension for network addresses / IPC tokens
}

@align(16)
pub struct IoCqe {
    pub user_data: u64;       // matches IoSqe.user_data
    pub result:    i32;       // byte count >= 0, or negative IoError code
    pub flags:     u32;       // More, BufferId
}
```
* **In-Flight Buffer Safety:** while an `IoSqe` request is inside the kernel/DMA, the buffer is passed to `IoRing` under **linear ownership** (`linear DmaSlice` or `linear BufSlice`) and returned to the caller together with the `IoCqe`! Freeing or overwriting a buffer while an NVMe drive or NIC is writing into it is **physically impossible at the type level** (solving the notorious incompatibility between `io_uring` and `Future` cancellation in Rust!).

### 3.2. Application and UI Level: Statically Checked Fibers (`Fiber`)
For regular code (network servers, UI asset loading, file operations), the programmer writes **plain linear code without a single `async`/`await` keyword**:

```
pub fn handle_client(conn: TcpStream, io: IoCap) -> !NetError @stack(4096) {
    errdefer conn.close(io);
    var buf: [2048]u8 = [0_u8; 2048];
    loop {
        // If data has not arrived yet, the fiber yields the physical core in ~15 ns
        // to another ready fiber or `par for` chunk! The OS thread NEVER blocks!
        let n = try conn.read(io, &mut buf[0..2048]);
        if n == 0_usize { break; }
        try conn.write_all(io, &buf[0..n]);
    }
    conn.close(io);
    return;
}
```

**Why Eon fibers eliminate every drawback of Go goroutines and C threads:**
1. **No stack overflow and no growable stacks (stack-growth checks are unnecessary!):**
   Because the Eon compiler **statically computes the exact stack size** of any non-`@recursive` function (including `dyn Iface` calls), when launching a fiber via `sched.spawn_fiber(handle_client, ...)`, the compiler **knows with certainty** that this fiber needs exactly `4096` bytes (+ a 256-byte safety margin for register saves)!
   - No stack-overflow checks in every function prologue (0% speed loss!).
   - No giant 2–8 MB per-thread stacks: 100,000 concurrent network connections with 4 KB fibers take only ~400 MB of RAM and are allocated from `SlotPool[FiberStack]` in `O(1)`.
2. **No function coloring:** `conn.read` is an ordinary function — it simply does **not** carry the `@noblock` effect. The compiler prevents calling it from `@interrupt` or while holding a `@noblock` spinlock (`E0112`)!
3. **Instant context switch (~15 ns):** only callee-saved CPU registers are saved (`rbx, rbp, r12-r15, rsp` on `x86_64` — just 56 bytes; `x19-x30, sp, d8-d15` on `ARM64`; `s0-s11, ra, sp` on `RISC-V`), with zero kernel syscalls and no FPU/AVX saves when a fiber suspends at a call site (where caller-saved registers are already dead under the `EonCall` ABI).
4. **Structured Concurrency:** fibers spawned inside a `with_scope(sched, |scope| { ... });` block are guaranteed to finish (or be cooperatively cancelled, running all `defer`/`errdefer` blocks and consuming all `linear` resources!) before `with_scope` exits. This lets fibers safely borrow `*T` from the parent stack without `Arc` (exact rules in §3.3)!

### 3.3. Fiber Spawning and Borrow Safety
Structured concurrency safety is **not derived** from the base rules of Chapter 03 (a fiber's closure and arguments are stored in the scheduler), so it is defined by two explicit constructs:
1. **`sched.spawn_fiber(f, args…)` (detached fiber).** Arguments must be strictly **owned** values (`move` / `TrivialCopy`, including `IoCap`). References `*T` / `*mut T`, borrowed values (03 §1.5), and capturing closures are forbidden (`E0731`): the fiber may outlive the frame that spawned it.
2. **`with_scope(sched, |scope| { scope.spawn(f, args…); … })`.** The sole construct that allows a fiber to borrow parent data. Rules:
   * borrowed arguments to `scope.spawn` are checked like ordinary call arguments, except the borrow is **extended until the end of `with_scope`**: `&mut x` locks `x` for the entire block, so two `scope.spawn` calls capturing `&mut x` yield `E0118`, whereas multiple shared `&x` borrows of the same data are permitted;
   * `with_scope` does not return control until all spawned fibers have completed (joined) — this is a **TCB axiom** (implemented in `core.sched`, included in `eon build --audit-tcb`) on which the soundness of the construct rests;
   * while the parent waits on the join, it executes no code and does not access the locked data.

### 3.4. Cancellation and Linear Resources
In Eon, there is no "sudden killing of a fiber mid-function" (which would break linearity). Cancellation is cooperative: when a `scope` is cancelled, any blocking operation on `IoRing` (e.g., `conn.read`) returns the standard error `IoError.Cancelled`, which cleanly unwinds all `errdefer` statements and guarantees release of every `linear` resource!

---

## 4. Specification Tests

### Must Compile
```
module example.effects_ok;

pub fn pure_math(a: f32, b: f32) -> f32 @pure @noalloc @noblock @stack(32) {
    return (a * a) + (b * b);
}

pub fn kernel_helper(x: f32) -> f32 @noalloc @noblock @stack(128) {
    return pure_math(x, 2.0_f32); // OK: pure_math's effects are stricter or equal
}
```

### Must Fail Compilation
```
module example.effects_fail;

pub fn blocking_read(io: IoCap) -> !IoError { ... }

pub fn bad_irq_handler(io: IoCap) -> !IoError @interrupt @stack(512) {
    return blocking_read(io); // ERROR E0112 (ZM-12): @interrupt requires @noblock,
                              // but blocking_read may suspend the fiber!
}

pub fn bad_recursive(n: u32) -> u32 @stack(256) {
    if n == 0_u32 { return 0_u32; }
    return bad_recursive(n - 1_u32); // ERROR E0113 (ZM-13): recursion without @recursive
                                     // and incompatible with static @stack(256)!
}
```
