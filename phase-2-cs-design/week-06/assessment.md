# Phase 2 · Week 06 — Assessment

> Week: OS & Networking Fundamentals  ·  Time: ~1 hr  ·  Pass gate: 75%

## Conceptual (interview-style) — 25% of grade

Answer each question in 2–5 sentences. Be precise; use correct terminology.

1. Explain the difference between a **process** and a **thread**. When would you choose `multiprocessing.Process` over `threading.Thread` in Python?

2. Describe what happens step-by-step when a running thread tries to `acquire()` a mutex that another thread already holds. What OS mechanism is involved?

3. What are the four Coffman conditions for deadlock? Give a concrete code example that demonstrates circular wait.

4. Explain virtual memory and paging. What is a **page fault** and when does it occur?

5. Compare TCP and UDP: what guarantees does each provide and in what situations would you choose UDP over TCP?

6. Describe the TLS 1.3 handshake at a high level. What is **forward secrecy** and how does ECDHE provide it?

7. Trace a DNS lookup for `api.example.com`: list every step from browser to authoritative nameserver and back. Which steps involve network I/O?

8. What is `SO_REUSEADDR`? Why do server programs almost always set it?

---

## Coding Challenge — 40% of grade

### Spec: Concurrent Key-Value Store Server

Build a TCP server that implements a simple key-value store over the line protocol:

**Commands:**
```
SET key value\n      → OK\n
GET key\n            → OK value\n  or  ERR not found\n
DEL key\n            → OK\n        or  ERR not found\n
KEYS\n               → OK key1,key2,...\n  (empty → OK \n)
STATS\n              → OK connections=N cmds=M\n
QUIT\n               → BYE\n and close
```

**Acceptance criteria:**
- [ ] Server runs on a configurable host/port (`--host`, `--port` args).
- [ ] Handles at least 10 concurrent clients without data corruption (use a mutex on the store).
- [ ] All commands return correct responses per spec.
- [ ] `STATS` returns accurate `connections` (total ever accepted) and `cmds` (total commands processed).
- [ ] Server starts and binds within 2 seconds; gracefully shuts down on `SIGINT`.
- [ ] Passes a pytest suite you write that:
  - Tests each command via a real socket.
  - Tests concurrent `SET` from 10 threads and verifies final store state.
  - Tests `ERR not found` for a missing key.

**Code quality requirements:**
- Type hints on all functions.
- `logging` not `print`.
- `ServerStats` as a `@dataclass` with a `threading.Lock`.
- No global mutable state outside the dataclass.

---

## Debugging Challenge — 20% of grade

The following server code has **three bugs**. Find and fix them. Explain each fix with one sentence.

```python
import socket
import threading

store = {}

def handle(conn):
    while True:
        data = conn.recv(1024)
        cmd, _, arg = data.decode().partition(" ")
        if cmd == "SET":
            key, _, val = arg.partition(" ")
            store[key] = val
            conn.send(b"OK")
        elif cmd == "GET":
            val = store[cmd]          # bug
            conn.send(f"OK {val}")    # bug
        elif cmd == "QUIT":
            break

def serve():
    s = socket.socket()
    s.bind(("0.0.0.0", 9000))
    s.listen()                        # bug
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle, args=(conn,)).start()

serve()
```

---

## Code Quality & Testing — 15% of grade

Evaluated on the solution you produce for the coding challenge:

- Functions are typed and documented.
- Mutex used correctly (no bare `store[key]` accesses outside the lock).
- Tests are isolated (each test uses a fresh port or a fresh store instance).
- At least one test uses `concurrent.futures.ThreadPoolExecutor` to verify thread-safety.
- Follows project style: `logging`, no secrets, `pathlib` where paths used.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation (coding challenge) | 40% |
| Conceptual understanding (interview Q&A) | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-06/assessment.md
