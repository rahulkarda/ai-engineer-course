# Phase 2 · Week 06 — Project: TCP Echo App + Concurrent Handler

> Time: ~3 hr  ·  Prereqs: All of Week 06

## Goal

Build a production-quality (at this level) TCP key-value echo server with a concurrent client benchmark, map its behavior to OS and networking concepts, and write tests that verify correctness under concurrency.

## Deliverables

```
week-06-project/
├── server.py           # TCP server with KV protocol
├── client.py           # Benchmark client
├── test_server.py      # pytest suite (≥ 8 tests)
└── concepts.md         # concept mapping (filled table)
```

---

## Specification

### server.py

Implement a TCP server supporting this line-delimited protocol (same as assessment coding challenge):

```
SET key value\n      → OK\n
GET key\n            → OK value\n  |  ERR not found\n
DEL key\n            → OK\n        |  ERR not found\n
KEYS\n               → OK key1,key2,...\n
STATS\n              → OK connections=N cmds=M\n
QUIT\n               → BYE\n and close connection
```

Requirements:
- CLI: `python server.py [--host HOST] [--port PORT]`
- Thread-per-connection model (not async).
- `KVStore` class: thread-safe with `threading.Lock`; methods `set`, `get`, `delete`, `keys`.
- `ServerStats` dataclass: thread-safe; tracks `total_connections` and `total_commands`.
- Handle `SIGINT` → log "Shutting down" → `server.close()` → exit cleanly.
- Log every connection open/close and every command received.
- `SO_REUSEADDR` set on server socket.

### client.py

Implement a concurrent benchmark client:

```
python client.py [--host HOST] [--port PORT] [--clients N] [--requests N]
```

- Each client sends `N` SET commands (`SET bench_<i> value_<i>`), then `N` GET commands, then `QUIT`.
- Measure per-request round-trip latency (ms).
- After all clients finish, print:
  - Total requests, elapsed time, throughput (req/s).
  - Latency: p50, p95, p99, max.
- Uses `threading.Thread` (not asyncio).

### test_server.py

Pytest suite with at least **8 tests**:

1. Server starts and accepts connections.
2. `SET` then `GET` returns correct value.
3. `GET` on missing key returns `ERR not found`.
4. `DEL` removes key; subsequent `GET` returns `ERR not found`.
5. `KEYS` returns all set keys.
6. `STATS` increments `cmds` correctly over multiple commands.
7. `QUIT` closes connection cleanly (subsequent `recv` returns empty bytes).
8. Concurrent `SET` from 10 threads — all 10 keys present in store after join.

Use a **test fixture** that starts the server in a background thread, waits for it to bind, yields the port, then tears it down.

### concepts.md

A filled mapping table (at least 8 rows) connecting observed server behavior to the OS/networking concepts from the week. Template row:

```
| Behavior observed | Concept | Explanation |
```

---

## Scaffold

Use this scaffold to start `server.py`:

```python
"""
server.py — TCP KV store server.
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
    """Thread-safe in-memory key-value store."""
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
        byte = conn.recv(1)
        if not byte:
            return None
        if byte == b"\n":
            return buf.decode(errors="replace").strip()
        buf.extend(byte)


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
                # TODO: implement command dispatch
                response = b"ERR not implemented\n"
                conn.sendall(response)
    except (ConnectionResetError, BrokenPipeError):
        log.warning("Client %s:%d disconnected", *addr)
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

    def shutdown(signum: int, frame: object) -> None:  # noqa: ANN001
        log.info("Shutting down")
        server.close()

    signal.signal(signal.SIGINT, shutdown)

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

## Acceptance Criteria Summary

| Criterion | Pass condition |
|-----------|---------------|
| Server starts | Binds and logs within 2 s |
| Protocol correct | All 6 commands return correct responses |
| Thread-safe store | 10-thread concurrent SET test passes 100% of runs |
| Tests | ≥ 8 tests, all green |
| Graceful shutdown | SIGINT logs "Shutting down", no traceback |
| Benchmark | p50 latency < 5 ms on loopback with 20 clients |
| Concepts doc | ≥ 8 rows mapping behavior → concept |

---

## Extension Challenges (optional)

1. Replace `recv_line` byte-at-a-time with a buffered reader — measure the latency difference.
2. Add a `TTL` command: `EXPIRE key seconds\n` — key auto-deletes after `seconds`. Use `threading.Timer`.
3. Implement `INCR key\n` → atomically increments an integer value (error if not integer).
4. Add connection-level rate limiting: max 100 commands/second per client; return `ERR rate limited\n` when exceeded.
5. Rewrite the server using `asyncio` and compare throughput with the threaded version at 100 concurrent clients.

Project solutions: ../solutions/week-06/project.md
