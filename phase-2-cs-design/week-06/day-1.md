# Phase 2 · Week 06 · Day 1 — Processes, Threads & Scheduling

> Objective: Understand how an OS manages execution units and CPU time.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-7]]

## Concept

### Process vs Thread

A **process** is an isolated execution environment: its own virtual address space, file descriptor table, and OS bookkeeping. Spawning a process is heavyweight — the kernel copies (or copy-on-write forks) memory maps, duplicates the file table, and resets signal handlers.

A **thread** shares its parent process's address space. Multiple threads inside one process can read the same global variable without an IPC call. That sharing is both the power and the danger — a bug in one thread corrupts the whole process.

```
Process A                       Process B
┌───────────────────────┐       ┌───────────────────────┐
│ code  heap  stack      │       │ code  heap  stack      │
│   Thread-1             │       │   Thread-1             │
│   Thread-2  ←shared→   │       │   Thread-1             │
└───────────────────────┘       └───────────────────────┘
          ↕ IPC (pipe/socket/shm)         (isolated)
```

Key properties:

| Property | Process | Thread |
|----------|---------|--------|
| Memory | Isolated | Shared |
| Creation cost | High (fork/exec) | Low |
| Crash blast radius | Contained | Whole process |
| Communication | IPC | Shared memory |
| Python GIL impact | Not affected | Affected for CPU work |

### Python GIL

CPython has a **Global Interpreter Lock** — only one thread runs Python bytecode at a time. CPU-bound threads get no parallelism from `threading`. Solutions:
- CPU-bound work → `multiprocessing` (separate processes, no GIL).
- I/O-bound work → `threading` or `asyncio` (GIL released during I/O syscalls).
- Python 3.13+ introduces an experimental no-GIL build (PEP 703).

### CPU Scheduling

The **scheduler** decides which ready thread/process runs on which CPU core and for how long.

**Preemptive scheduling**: OS forcibly evicts a running thread when its **time quantum** (typ. 1–10 ms) expires or a higher-priority task becomes ready.

Common algorithms:

| Algorithm | Idea | Pros | Cons |
|-----------|------|------|------|
| Round-Robin | equal time slices | fair | no priority |
| Priority scheduling | highest priority first | responsive | starvation |
| CFS (Linux default) | virtual runtime tree | balanced | complex |
| MLFQ | multiple queues, demote on long bursts | adapts | complex |

**Context switch**: saving/restoring CPU registers and program counter when switching between threads. Costs ~microseconds but adds up under high contention.

## How It Works Internally

### Process Creation — fork/exec model

```
parent pid=100               child pid=101
fork()  ──────────────────►  copy-on-write address space
                             exec("python") → replace image
```

`fork()` duplicates the process. `exec()` replaces its memory image with a new program. Python's `multiprocessing` on Unix uses this; on Windows it uses `spawn` (no fork available).

### Thread Stack

Each thread gets its own **stack** (default 8 MB on Linux) carved out of the process's virtual address space. The heap is shared. Stack overflow in one thread triggers SIGSEGV for the whole process.

### Scheduler Data Structures

Linux CFS keeps a **red-black tree** of runnable tasks sorted by `vruntime` (CPU time weighted by priority). The leftmost node is always the next to run — O(log n) pick.

## Worked Example

```python
"""
day1_processes_threads.py — Compare process vs thread for CPU vs I/O work.
Python 3.12+
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


# ── CPU-bound task ─────────────────────────────────────────────────────────────

def cpu_task(n: int) -> float:
    """Compute sum of sqrt for n iterations — pure CPU work."""
    return sum(math.sqrt(i) for i in range(n))


def benchmark_cpu(n: int = 2_000_000, workers: int = 4) -> None:
    log.info("=== CPU-bound: threads vs processes (n=%d, workers=%d) ===", n, workers)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(cpu_task, [n] * workers))
    log.info("ThreadPool CPU time: %.3f s  (GIL = no speedup expected)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(cpu_task, [n] * workers))
    log.info("ProcessPool CPU time: %.3f s  (parallel, ~%dx faster)", time.perf_counter() - t0, workers)


# ── I/O-bound task ─────────────────────────────────────────────────────────────

def io_task(sleep_s: float) -> None:
    """Simulate I/O wait."""
    time.sleep(sleep_s)


def benchmark_io(tasks: int = 20, sleep_s: float = 0.05) -> None:
    log.info("=== I/O-bound: threads (n=%d, sleep=%.2fs each) ===", tasks, sleep_s)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=tasks) as pool:
        list(pool.map(io_task, [sleep_s] * tasks))
    log.info("ThreadPool I/O time: %.3f s  (GIL released during sleep → parallel)", time.perf_counter() - t0)


# ── Process identity ───────────────────────────────────────────────────────────

def show_identity() -> None:
    log.info("Main process PID: %d", os.getpid())

    def thread_fn() -> None:
        log.info("Thread in PID %d, thread id: %d", os.getpid(), threading.get_ident())

    t = threading.Thread(target=thread_fn)
    t.start()
    t.join()
    log.info("Thread shares PID — same process, different execution context")


if __name__ == "__main__":
    show_identity()
    benchmark_cpu()
    benchmark_io()
```

Run it and observe:
- `ThreadPool CPU` ≈ same as sequential (GIL prevents parallel bytecode).
- `ProcessPool CPU` ≈ 4× faster (4 real CPU cores).
- `ThreadPool I/O` completes in ≈ `sleep_s` not `tasks × sleep_s` (GIL released during `sleep`).

## Exercises

1. Modify `cpu_task` to also compute `math.factorial(1000)`. Re-run both executors. Does the ratio change?
2. Write a function `spawn_child()` that forks a child process using `multiprocessing.Process`, prints its own PID, then exits. The parent prints both PIDs.
3. Explain: why does `ThreadPoolExecutor` with 4 workers on a CPU-bound task sometimes run *slower* than a single thread?
4. What happens to open file handles shared between parent and child after a `fork`? Write a short paragraph and a one-sentence risk.
5. Research Python 3.13 PEP 703 (no-GIL). List two things that become easier and two that become harder.

(Solutions: ../solutions/week-06/day-1.md)

## Self-check

- [ ] I can explain the difference between a process and a thread in one sentence each.
- [ ] I can choose `threading` vs `multiprocessing` given a workload description.
- [ ] I can describe what a context switch is and why it has cost.
- [ ] I can explain why CPython's GIL matters for threading.
- [ ] I can run the worked example and interpret the timing output.

## Supplementary

- [Linux Scheduler (CFS) — kernel docs](https://www.kernel.org/doc/html/latest/scheduler/sched-design-CFS.html)
- [Python docs — multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [PEP 703 — Making the GIL Optional](https://peps.python.org/pep-0703/)
- [Operating Systems: Three Easy Pieces — Processes chapter](https://pages.cs.wisc.edu/~remzi/OSTEP/)
