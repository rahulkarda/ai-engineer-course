# Phase 2 · Week 06 · Day 2 — Memory & Virtual Memory

> Objective: Understand how an OS provides each process its own memory space via virtual memory and paging.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]]

## Concept

### Physical vs Virtual Memory

Every process believes it owns the entire address space (e.g., 0 to 2⁶⁴ - 1 on a 64-bit system). This is a lie told by the OS — and a very useful one. The **Memory Management Unit (MMU)** hardware translates virtual addresses to physical RAM addresses on every memory access.

Benefits of virtual memory:
- **Isolation** — process A cannot accidentally read process B's memory.
- **Overcommit** — sum of all virtual spaces can exceed physical RAM.
- **Shared libraries** — one physical copy of `libc` mapped read-only into every process.
- **Simplified linking** — every process can link at address 0x400000 without conflicts.

### Paging

Memory is divided into fixed-size units called **pages** (typically 4 KiB on x86-64, 16 KiB on ARM64/macOS). The OS keeps a **page table** per process mapping virtual page number → physical frame number.

```
Virtual Address Space (process view)
 ┌─────────────────────────┐
 │ 0x0000 — text (code)    │  page 0   ─→  frame 42
 │ 0x1000 — data           │  page 1   ─→  frame 7
 │ 0x2000 — heap ↓         │  page 2   ─→  frame 99
 │   ...                   │  page 3   ─→  NOT PRESENT (swapped)
 │ stack ↑                 │  page N   ─→  frame 5
 └─────────────────────────┘
```

Page table entry flags: Present, Writable, User/Kernel, Accessed, Dirty.

### Page Faults

When a process accesses a page marked **not present**, the CPU raises a **page fault** exception. The OS handler:
1. Checks if the access is valid (within a mapped region).
2. If valid: loads the page from disk (swap/file), updates the page table, resumes the instruction.
3. If invalid: delivers `SIGSEGV` to the process (`Segmentation fault`).

**Demand paging**: pages are loaded lazily on first access, not at `exec` time.

### Page Replacement

When physical RAM is full and a new page is needed, the OS must **evict** a frame:
- **LRU (Least Recently Used)**: evict the page unused longest — good in theory, expensive in hardware.
- **Clock algorithm**: approximate LRU using a reference bit — practical.
- **Linux**: uses a two-list (active/inactive) clock variant.

Excessive eviction + reload = **thrashing** — CPU spends more time paging than executing.

### Memory Layout of a Process

```
High addresses
┌────────────────────┐  kernel space (inaccessible)
├────────────────────┤
│       stack        │  ↓ grows down; local vars, return addresses
│        ↕           │
│       heap         │  ↑ grows up; malloc/new
├────────────────────┤
│    BSS (uninit)    │  zero-init globals
├────────────────────┤
│    data (init)     │  initialized globals
├────────────────────┤
│       text         │  read-only executable code
└────────────────────┘
Low addresses (0x0 — null page, unmapped)
```

### Python Memory Model

Python objects live on the **heap**. CPython uses:
- **Reference counting** as primary GC — freed immediately when count hits 0.
- **Cycle collector** (generational) for reference cycles.
- **pymalloc** — a custom allocator in 8-byte arenas that sits on top of `malloc`.
- `sys.getsizeof(obj)` returns shallow byte size (excluding referenced objects).

## How It Works Internally

### TLB — Translation Lookaside Buffer

Walking a multi-level page table (4 levels on x86-64) on every memory access costs 4 memory reads. The CPU caches recent VA→PA translations in the **TLB** (~64–2048 entries). A **TLB miss** triggers a hardware page-table walk; a context switch flushes the TLB (or uses tagged entries — ASID).

### Copy-on-Write (COW)

After `fork()`, parent and child share physical frames — all mapped read-only. On first write by either, the MMU raises a protection fault; the OS copies the page to a new frame for the writer. This makes `fork()` fast even for large processes — most pages are never written.

### Huge Pages

Default pages are 4 KiB → large datasets need millions of TLB entries. **Huge pages** (2 MiB on x86-64) reduce TLB pressure for big buffers. Python's `mmap` and NumPy can use huge pages when available.

## Worked Example

```python
"""
day2_memory.py — Observe memory layout and reference counting.
Python 3.12+
"""
from __future__ import annotations

import ctypes
import gc
import logging
import resource
import sys
from collections.abc import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── Object sizes ───────────────────────────────────────────────────────────────

def show_sizes() -> None:
    objects: dict[str, object] = {
        "int(0)": 0,
        "int(2**30)": 2**30,
        "float": 3.14,
        "str('')": "",
        "str('hello')": "hello",
        "list[]": [],
        "list[1,2,3]": [1, 2, 3],
        "dict{}": {},
    }
    log.info("=== Object sizes (shallow) ===")
    for name, obj in objects.items():
        log.info("  %-20s %d bytes", name, sys.getsizeof(obj))


# ── Reference counting ─────────────────────────────────────────────────────────

def show_refcount() -> None:
    log.info("=== Reference counting ===")

    x: list[int] = [1, 2, 3]
    log.info("After x=[1,2,3]:    refcount=%d", sys.getrefcount(x))  # 2 (x + getrefcount arg)

    y = x  # noqa: F841
    log.info("After y=x:          refcount=%d", sys.getrefcount(x))  # 3

    del y
    log.info("After del y:        refcount=%d", sys.getrefcount(x))  # 2


# ── Memory usage ──────────────────────────────────────────────────────────────

def show_process_memory() -> None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss in bytes; Linux: in kilobytes
    log.info("Max RSS: %d KB", usage.ru_maxrss // 1024)


# ── Cycle detection ───────────────────────────────────────────────────────────

def demonstrate_cycle() -> None:
    log.info("=== Cycle collector ===")
    gc.disable()  # turn off automatic collection

    a: dict[str, object] = {}
    b: dict[str, object] = {}
    a["ref"] = b
    b["ref"] = a  # cycle: a → b → a

    id_a, id_b = id(a), id(b)
    del a, b
    # Objects still alive — reference cycle keeps them
    log.info("Cycle alive (gc disabled). Collected: %d", gc.collect())
    log.info("After gc.collect: objects freed")

    gc.enable()


# ── Virtual memory overcommit demo ────────────────────────────────────────────

def overcommit_demo(size_mb: int = 512) -> None:
    """Allocate a large bytearray but don't write to it — observe lazy pages."""
    import os

    log.info("=== Virtual memory overcommit ===")
    log.info("Allocating %d MiB bytearray (demand-paged — physical pages not yet claimed)", size_mb)
    buf = bytearray(size_mb * 1024 * 1024)
    show_process_memory()  # RSS should NOT increase by 512 MiB

    log.info("Writing every 4096th byte to force page-in...")
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    show_process_memory()  # RSS now increases


if __name__ == "__main__":
    show_sizes()
    show_refcount()
    show_process_memory()
    demonstrate_cycle()
    overcommit_demo(256)
```

Run and observe:
- `list[1,2,3]` is larger than `list[]` due to pointer slots.
- RSS barely changes after `bytearray` allocation; spikes after writing every 4096th byte.
- Cycle GC frees the cyclic objects only after explicit `gc.collect()`.

## Exercises

1. Use `tracemalloc` (stdlib) to trace the top 5 memory allocations in `overcommit_demo`. Print file, line, and byte count.
2. Explain in your own words why `sys.getsizeof([1,2,3])` does not include the memory used by the integers 1, 2, 3.
3. A Python server leaks memory over hours. List three likely root causes and one diagnostic tool for each.
4. What is the difference between RSS and VSZ in `ps` output? Which better indicates actual physical memory consumption?
5. A process calls `fork()` and immediately runs `exec()`. Does COW benefit us here? Why or why not?

(Solutions: ../solutions/week-06/day-2.md)

## Self-check

- [ ] I can explain virtual memory in one paragraph without consulting notes.
- [ ] I can describe what happens during a page fault step-by-step.
- [ ] I can use `sys.getsizeof` and `resource.getrusage` to inspect memory.
- [ ] I can explain Python reference counting and when the cycle collector is needed.
- [ ] I can define TLB and explain why it matters for performance.

## Supplementary

- [Operating Systems: Three Easy Pieces — Virtual Memory chapters](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [Python docs — gc module](https://docs.python.org/3/library/gc.html)
- [Python docs — tracemalloc](https://docs.python.org/3/library/tracemalloc.html)
- [Linux kernel — memory management overview](https://www.kernel.org/doc/html/latest/admin-guide/mm/index.html)
