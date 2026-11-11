# Solutions — Week 06 Assessment

## Conceptual Answers

### Q1 — Process vs Thread

A **process** is an OS-managed execution environment with its own isolated virtual address space, file descriptors, and bookkeeping (PCB). A **thread** shares the address space of its parent process — multiple threads inside one process read/write the same heap and globals.

Choose `multiprocessing.Process` over `threading.Thread` when:
- Work is **CPU-bound** — `threading` provides no parallelism in CPython due to the GIL; `multiprocessing` spawns separate interpreter processes, each with its own GIL.
- **Crash isolation** is needed — a crash in a subprocess doesn't kill the parent.
- The work is truly independent with limited shared state (IPC cost is acceptable).

### Q2 — Mutex acquire when already held

1. Thread B calls `lock.acquire()`.
2. CPython calls `pthread_mutex_lock` (POSIX) / `WaitForSingleObject` (Windows).
3. The OS checks the mutex's state: locked by Thread A.
4. Thread B is moved from RUNNING → BLOCKED; added to the mutex's wait queue (kernel object).
5. Thread B's CPU time slot is given to another runnable thread (context switch).
6. When Thread A calls `lock.release()`, the OS wakes the highest-priority waiter (Thread B), moves it to READY, and at the next scheduler tick it resumes and re-acquires the mutex.

Underlying mechanism: **futex** (Linux) — checks the lock state atomically in user space first; only makes a syscall (`futex_wait`) on contention.

### Q3 — Coffman conditions + circular wait example

**Four conditions** (all must hold for deadlock):
1. **Mutual exclusion** — resource held by ≤ 1 thread.
2. **Hold and wait** — thread holds ≥ 1 resource while waiting for another.
3. **No preemption** — resources can't be forcibly taken.
4. **Circular wait** — cycle in resource-allocation graph.

```python
import threading, time
m1, m2 = threading.Lock(), threading.Lock()

def thread_a():
    with m1:               # holds m1
        time.sleep(0.05)
        m2.acquire()       # waits for m2 (held by B)

def thread_b():
    with m2:               # holds m2
        time.sleep(0.05)
        m1.acquire()       # waits for m1 (held by A)

# Both threads run simultaneously → deadlock
```

### Q4 — Virtual memory and paging

**Virtual memory**: each process sees a private address space (0 to 2⁶⁴-1 on 64-bit) managed by the OS+MMU. Actual physical RAM frames are mapped to virtual pages via per-process **page tables**. This provides isolation, overcommit, and address-space-layout randomization.

**Paging**: address space divided into 4 KiB pages. Page table entry maps virtual page → physical frame + permission flags (present, writable, user).

**Page fault**: CPU attempts to access a virtual address whose page table entry is marked NOT PRESENT. MMU raises an exception; kernel handler:
1. Validates the access (is it in a mapped region?).
2. Loads the page from disk/swap into a free frame.
3. Updates the page table entry.
4. Restarts the faulting instruction.
If invalid: `SIGSEGV`.

### Q5 — TCP vs UDP

| | TCP | UDP |
|--|-----|-----|
| Connection | 3-way handshake | None |
| Delivery | Guaranteed (retransmit) | Best-effort |
| Order | In-order byte stream | Datagrams, any order |
| Flow control | Yes (window) | No |
| Header size | 20+ bytes | 8 bytes |

Choose UDP when:
- Low latency matters more than reliability (video calls, gaming — stale frame is worthless).
- Application implements its own reliability (QUIC/HTTP/3).
- Single-shot queries where retransmit logic is simpler than TCP setup (DNS queries).

### Q6 — TLS 1.3 handshake and forward secrecy

1. Client sends `ClientHello`: supported cipher suites, key_share (ECDHE public key), random nonce.
2. Server sends `ServerHello` (chosen suite, server ECDHE public key) + `Certificate` + `CertificateVerify` (proof of private key) + `Finished` — all **1 round-trip**.
3. Client verifies the certificate (chain to trusted root CA), computes shared secret via ECDHE, verifies `Finished` MAC.
4. Client sends its own `Finished`. Application data flows.

**Forward secrecy**: the session key is derived from **ephemeral** ECDHE key pairs generated fresh per session. The server's long-term private key only authenticates; it never contributes to the symmetric session key. If the long-term key is later compromised, past session keys cannot be derived — each session key existed only during that session.

### Q7 — DNS resolution for api.example.com

1. Browser checks its internal DNS cache (TTL, typ. 60 s). Hit → done.
2. Browser asks the OS stub resolver (libc `getaddrinfo`).
3. OS checks its own cache (nscd / systemd-resolved). Hit → done.
4. OS sends query to the **recursive resolver** (e.g., 8.8.8.8) via UDP port 53.
5. Recursive resolver (if uncached) queries a **root nameserver** (`.`): "who handles `.com`?"
6. Root responds: `.com` TLD nameservers.
7. Recursive resolver queries a **TLD nameserver**: "who handles `example.com`?"
8. TLD responds: `example.com`'s authoritative nameservers (NS records).
9. Recursive resolver queries the **authoritative nameserver** for `api.example.com` A record.
10. Authoritative responds with IP (and TTL).
11. Recursive caches result, returns to OS.
12. OS returns to browser, caches locally.

Steps 4–11 involve network I/O. Steps 1–3 are in-memory.

### Q8 — SO_REUSEADDR

`SO_REUSEADDR` allows a server socket to **bind to a port that has a socket in TIME_WAIT state** from a previous server run. Without it, restarting a server within 2×MSL (~2 min) after a crash fails with `Address already in use`. It also lets multiple sockets bind to the same port with different local addresses (multicast use case).

**Note**: `SO_REUSEADDR` does not bypass TIME_WAIT's protection against segment confusion — it only permits rebinding, and the kernel still tracks the old connection.

---

## Coding Challenge — Reference Solution

```python
"""
kv_server.py — TCP key-value store server (assessment reference solution).
Python 3.12+
"""
from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-15s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class KVStore:
    _data: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


@dataclass
class ServerStats:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_connections: int = 0
    total_commands: int = 0

    def record_connection(self) -> None:
        with self._lock:
            self.total_connections += 1

    def record_command(self) -> None:
        with self._lock:
            self.total_commands += 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.total_connections, self.total_commands


def recv_line(conn: socket.socket) -> str | None:
    buf = bytearray()
    while True:
        b = conn.recv(1)
        if not b:
            return None
        if b == b"\n":
            return buf.decode(errors="replace").strip()
        buf.extend(b)


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    store: KVStore,
    stats: ServerStats,
) -> None:
    stats.record_connection()
    log.info("Connect %s:%d", *addr)
    try:
        with conn:
            conn.sendall(b"READY\n")
            while True:
                line = recv_line(conn)
                if line is None:
                    break
                stats.record_command()
                parts = line.split(" ", 2)
                cmd = parts[0].upper()

                if cmd == "SET" and len(parts) >= 3:
                    store.set(parts[1], parts[2])
                    conn.sendall(b"OK\n")
                elif cmd == "GET" and len(parts) >= 2:
                    val = store.get(parts[1])
                    if val is not None:
                        conn.sendall(f"OK {val}\n".encode())
                    else:
                        conn.sendall(b"ERR not found\n")
                elif cmd == "DEL" and len(parts) >= 2:
                    if store.delete(parts[1]):
                        conn.sendall(b"OK\n")
                    else:
                        conn.sendall(b"ERR not found\n")
                elif cmd == "KEYS":
                    keys = ",".join(store.keys())
                    conn.sendall(f"OK {keys}\n".encode())
                elif cmd == "STATS":
                    conns, cmds = stats.snapshot()
                    conn.sendall(f"OK connections={conns} cmds={cmds}\n".encode())
                elif cmd == "QUIT":
                    conn.sendall(b"BYE\n")
                    break
                else:
                    conn.sendall(f"ERR unknown command: {cmd}\n".encode())
    except (ConnectionResetError, BrokenPipeError):
        log.warning("Client %s:%d reset", *addr)
    finally:
        log.info("Closed %s:%d", *addr)


def run_server(host: str, port: int) -> None:
    store = KVStore()
    stats = ServerStats()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(128)
    log.info("KV server on %s:%d", host, port)

    def on_sigint(signum: int, frame: object) -> None:
        log.info("Shutting down")
        server.close()

    signal.signal(signal.SIGINT, on_sigint)

    while True:
        try:
            conn, addr = server.accept()
        except OSError:
            break
        threading.Thread(
            target=handle_client,
            args=(conn, addr, store, stats),
            daemon=True,
            name=f"C-{addr[1]}",
        ).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args()
    run_server(args.host, args.port)
```

---

## Debugging Challenge — Three Bugs Fixed

```python
# Bug 1: store[cmd] should be store[key]
#   val = store[cmd]      ← cmd = "GET", key is in `arg`
#   Fix:
val = store.get(key, None)
if val is None:
    conn.send(b"ERR not found\n")
else:
    conn.send(f"OK {val}\n".encode())  # Bug 2 fix also here

# Bug 2: conn.send(f"OK {val}") sends a str, not bytes
#   Fix: encode the string
conn.sendall(f"OK {val}\n".encode())

# Bug 3: listen() without backlog argument is fine on Python (defaults to SOMAXCONN),
# but the real bug is missing SO_REUSEADDR — server can't restart quickly.
# Also: data = conn.recv(1024) returns bytes; .decode() strips \n but cmd/arg parsing
# assumes clean input. For robustness, strip the result.
# Full fixed handle():
def handle(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            break
        line = data.decode().strip()          # strip \r\n
        cmd, _, arg = line.partition(" ")
        if cmd == "SET":
            key, _, val = arg.partition(" ")
            store[key] = val
            conn.sendall(b"OK\n")
        elif cmd == "GET":
            key = arg.strip()                 # Bug 1 fix: use key not cmd
            val = store.get(key)
            if val is not None:
                conn.sendall(f"OK {val}\n".encode())  # Bug 2 fix: .encode()
            else:
                conn.sendall(b"ERR not found\n")
        elif cmd == "QUIT":
            break
    conn.close()

def serve():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Bug 3 fix
    s.bind(("0.0.0.0", 9000))
    s.listen(128)
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle, args=(conn,)).start()
```

**Bug summary:**
1. `store[cmd]` → should be `store[key]` (using wrong variable as dict key).
2. `conn.send(f"OK {val}")` → must encode to bytes: `conn.sendall(f"OK {val}\n".encode())`.
3. Missing `SO_REUSEADDR` → `Address already in use` on restart within TIME_WAIT window.
