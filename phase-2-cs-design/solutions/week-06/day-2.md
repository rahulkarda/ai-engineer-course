# Solutions — Week 06 · Day 2 — Memory & Virtual Memory

## Exercise 1 — tracemalloc top 5 allocations

```python
import tracemalloc
tracemalloc.start()

buf = bytearray(256 * 1024 * 1024)
for i in range(0, len(buf), 4096):
    buf[i] = 1

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
print("=== Top 5 memory allocations ===")
for stat in top_stats[:5]:
    print(stat)
```

Expected output: the `bytearray` allocation dominates, traced to `overcommit_demo`.

## Exercise 2 — Why sys.getsizeof([1,2,3]) excludes integers

`sys.getsizeof` returns the **shallow** size: the `list` object header plus an array of 3 pointers (each 8 bytes on 64-bit). The integers 1, 2, 3 are separate objects on the heap — the list just holds references (pointers) to them. To get total (deep) memory, you would need to recursively sum `getsizeof` over all referenced objects. The `pympler` library's `asizeof` does this.

## Exercise 3 — Memory leak root causes and diagnostic tools

| Cause | Diagnostic tool |
|-------|----------------|
| Reference cycle not collected (e.g., `__del__` + cycle) | `gc.garbage` list + `gc.set_debug(gc.DEBUG_LEAK)` |
| Growing cache with no eviction (e.g., `dict` accumulating keys) | `tracemalloc` — snapshot diff over time |
| C extension not releasing memory (PyObject refcount bug) | `Valgrind` + Python's `sys.getrefcount`; also `memray` |

## Exercise 4 — RSS vs VSZ

- **VSZ (Virtual Size)**: total virtual address space size, including unmapped/swapped/demand-paged memory. Includes mmap regions, shared libraries, reserved but unwritten pages. Often >> physical RAM.
- **RSS (Resident Set Size)**: physical RAM frames currently backing the process's pages. Only pages that have been faulted in and not yet swapped out.

**RSS** better indicates actual physical memory consumption. VSZ can be misleadingly large (a 512 MiB `bytearray` just allocated, not written → large VSZ, tiny RSS increase).

## Exercise 5 — fork + exec and COW

COW benefits are largely **wasted** when `exec()` immediately follows `fork()`. COW defers copying pages until a write, but `exec()` discards the entire address space anyway, replacing it with the new program image. The kernel is smart enough to recognize this pattern (`vfork()` is the extreme case — parent suspends until child execs). The main residual benefit is avoiding the page-table duplication cost for the brief window between fork and exec.
