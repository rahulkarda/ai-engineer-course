# Phase 2 · Week 06 · Day 7 — Project Write-Up, Integration & Reflection

> Objective: Consolidate the week's learnings by annotating your TCP server with OS/networking concept mappings, then complete the project.  ·  Time: ~3 hr project + see assessment.md  ·  Prereqs: [[week-06/day-6]]

## Project Deliverable

You will produce:
1. A fully working TCP echo + command server (`server.py`).
2. A concurrent benchmark client (`client.py`).
3. A concept mapping document (`concepts.md`).
4. A pytest test suite (`test_server.py`).

See [[project.md]] for the full spec and acceptance criteria.

---

## Concept Mapping

For each behavior you observed in Day 6's experiments, fill in the table below (write in your own words — this is your reference for the assessment).

| Observed behavior | OS/network concept | Why it happens |
|-------------------|--------------------|----------------|
| Second client blocks on single-threaded server | Single-threaded accept loop | `accept()` is blocking; only one connection processed at a time |
| Multiple clients served simultaneously after threading | Thread-per-connection model | Each thread runs independently, context-switched by OS scheduler |
| `ServerStats.active_connections` would race without mutex | Race condition on shared mutable state | Read-modify-write is not atomic; threads interleave |
| TIME_WAIT sockets after benchmark | TCP four-way FIN + TIME_WAIT state machine | 2×MSL wait prevents old segments from confusing new connections |
| Backlog overflow drops connections | TCP `listen()` backlog queue | OS drops SYNs when accept queue full |
| Latency rises with 100 concurrent clients | Context switch overhead + lock contention | More threads → more scheduling overhead and mutex wait time |
| `bytearray` allocation doesn't spike RSS | Demand paging / virtual memory | Pages not physically allocated until first write |
| TCP_NODELAY reduces latency for small messages | Nagle's algorithm disabled | Small writes sent immediately, not buffered for coalescing |

---

## Integration Checklist

Before submitting, verify each item:

- [ ] `server.py` starts without error: `python server.py --port 9002`
- [ ] `nc 127.0.0.1 9002` connects, `ECHO hello` returns `OK hello`, `QUIT` disconnects cleanly.
- [ ] `python client.py --clients 20 --requests 50` completes and prints p50/p95/p99.
- [ ] `python -m pytest test_server.py -v` passes all tests.
- [ ] `ServerStats` tests cover concurrent access (use `ThreadPoolExecutor`).
- [ ] Server handles `SIGINT` gracefully (no stack trace, logs "Shutting down").
- [ ] `concepts.md` filled with at least 6 rows.

---

## Concept Review: Week Summary

### Processes & Threads (Day 1)
- Process = isolated address space + resources.
- Thread = shared address space, separate stack.
- GIL prevents CPU-bound parallelism in CPython threads → use `multiprocessing`.
- I/O-bound → threads are fine (GIL released during I/O).
- Scheduler = preemptive, time-quantum-based (CFS on Linux).

### Memory (Day 2)
- Virtual memory = per-process illusion of the whole address space.
- Paging = 4 KiB blocks, page table maps virtual → physical frames.
- Page fault = access to unmapped/swapped page → kernel loads it.
- Python: reference counting + cycle GC; `pymalloc` arena.
- COW = fork copies pages lazily on first write.

### Concurrency Primitives (Day 3)
- Mutex: one thread at a time in critical section.
- Semaphore: counting limit on concurrent access.
- Condition variable: wait-for-condition with atomic lock release.
- Deadlock: all four Coffman conditions must hold; break circular wait with lock ordering.

### OSI / TCP-IP (Day 4)
- OSI: 7 layers (Application→Physical). TCP/IP: 4 layers.
- TCP: reliable, ordered, connection-oriented. 3-way handshake, flow control, congestion control.
- UDP: unreliable, connectionless, 8-byte header. Use for latency-sensitive, loss-tolerant traffic.
- Port = multiplexer; socket = 5-tuple.

### HTTP / TLS / DNS (Day 5)
- HTTP: stateless request-response. Methods, status codes, headers.
- HTTP/2: multiplexed streams, HPACK compression.
- TLS 1.3: 1 RTT handshake, ECDHE key exchange, forward secrecy, cert chain verification.
- DNS: hierarchical, cached, A/AAAA/CNAME/MX/TXT records, TTL-based invalidation.

---

## Reflection Questions

Answer these in your `concepts.md` before attempting the assessment.

1. Your TCP server handles 20 concurrent clients with threads. Name two alternative concurrency models you could use instead, and one trade-off each.
2. A production API server receives 10,000 req/s. Why would a thread-per-connection model fail at this scale, and what would you use instead?
3. A client reports "connection refused" errors. Walk through the network stack from application to kernel to identify possible causes.
4. Your HTTPS API returns data fine in a browser but your Python `socket` code raises `ssl.SSLCertVerificationError`. What are the possible causes?
5. Explain in one paragraph how the OS scheduler, TCP congestion control, and Python's GIL all interact when your benchmark client runs 100 concurrent threads against your server.

---

## Self-check

- [ ] I can describe the full lifecycle of a TCP connection (open → data → close) step by step.
- [ ] I can explain why thread-per-connection doesn't scale to millions of connections.
- [ ] I can connect each Day 6 experiment result to a specific OS or network concept.
- [ ] My test suite passes and covers thread-safety scenarios.
- [ ] I feel confident answering any of the Week 6 assessment questions.

## Supplementary

- [The C10K Problem — Dan Kegel (classic scaling essay)](http://www.kegel.com/c10k.html)
- [Python asyncio — for the next step beyond threads](https://docs.python.org/3/library/asyncio.html)
- [High Performance Browser Networking — Ilya Grigorik (free)](https://hpbn.co/)
