# Solutions — Week 06 · Day 1 — Processes, Threads & Scheduling

## Exercise 1 — Add factorial to cpu_task

```python
def cpu_task(n: int) -> float:
    total = sum(math.sqrt(i) for i in range(n))
    total += float(math.factorial(1000) % 10**9)  # add factorial, keep as float
    return total
```

With the larger factorial, each task takes more CPU time. The ratio between ThreadPool and ProcessPool stays roughly the same or widens slightly, because more time per task means the overhead of process creation is a smaller fraction — and the GIL penalty remains proportional to total work.

## Exercise 2 — spawn_child with multiprocessing.Process

```python
import multiprocessing
import os

def child_fn() -> None:
    print(f"Child PID: {os.getpid()}")

def spawn_child() -> None:
    print(f"Parent PID: {os.getpid()}")
    p = multiprocessing.Process(target=child_fn)
    p.start()
    p.join()
    print(f"Child process exited with code {p.exitcode}")

if __name__ == "__main__":
    spawn_child()
```

Expected output: Parent and child PIDs differ. `p.join()` blocks until child exits.

## Exercise 3 — Why threads can be slower than single for CPU work

The GIL forces serialization: all 4 threads take turns running, each for up to 5 ms, then releasing the GIL. The overhead of:
- Acquiring/releasing the GIL on each switch.
- OS context switches (save/restore registers).
- Cache invalidation (each core may fetch the same cache lines).

...adds up and can make total wall-clock time *longer* than sequential execution. The GIL does not allow overlapping CPU computation — only I/O releases it.

## Exercise 4 — Open file handles after fork

After `fork()`, both parent and child have copies of all open file descriptors pointing to the **same underlying file descriptions** in the kernel (same offset, same position). Writing to the file in one process advances the shared file-position cursor. **Risk**: Both processes writing to the same log file or database socket leads to interleaved/corrupt output unless one side closes or re-opens the descriptor.

## Exercise 5 — PEP 703 (no-GIL Python 3.13)

**Easier with no GIL:**
1. CPU-bound threading becomes truly parallel — `ThreadPoolExecutor` can utilize multiple cores without `multiprocessing`.
2. Sharing large data structures (e.g., shared NumPy arrays) between threads requires no serialization workaround.

**Harder with no GIL:**
1. Existing C extensions that assumed GIL protection for their internal data structures may have race conditions — need auditing and rewriting.
2. Pure-Python code that relied on the GIL for atomic-looking operations (e.g., `list.append`, `dict.__setitem__`) may now need explicit locks to be safe.
