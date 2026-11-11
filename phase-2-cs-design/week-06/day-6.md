# Phase 2 · Week 06 · Day 6 — Build: TCP Echo Server + Concurrent Request Handler

> Objective: Build a threaded TCP echo server and concurrent HTTP-like request handler from scratch.  ·  Time: ~4 hr (deep practice)  ·  Prereqs: [[week-06/day-4]], [[week-06/day-5]], [[week-06/day-3]]

## Overview

Today's 4-hour session builds a complete mini-server that exercises every concept from the week: sockets, TCP connection lifecycle, threading, mutexes, and basic protocol design. You will also run experiments to observe OS/network behavior directly.

**What you will build:**
1. A TCP echo server (single-threaded, then multi-threaded).
2. A simple line-delimited request protocol on top of TCP.
3. A concurrent connection handler with a shared state counter.
4. A benchmark client that sends N concurrent requests.
5. A load statistics reporter mapping results to OS concepts.

---

## Part 1 — Single-Threaded Echo Server (30 min)

Start with the simplest possible server to understand the socket lifecycle.

```python
"""
echo_server_v1.py — Single-threaded TCP echo server.
Run: python echo_server_v1.py
Test: nc 127.0.0.1 9000   (then type lines, Ctrl-D to disconnect)
"""
from __future__ import annotations

import logging
import socket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 9000
BUFSIZE = 4096


def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
    log.info("Connection from %s:%d", *addr)
    with conn:
        while True:
            data = conn.recv(BUFSIZE)
            if not data:
                break  # client closed connection (EOF)
            log.info("Received %d bytes from %s:%d", len(data), *addr)
            conn.sendall(data)  # echo back
    log.info("Connection closed: %s:%d", *addr)


def run_server() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)  # backlog = 5 pending connections
        log.info("Echo server listening on %s:%d", HOST, PORT)
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr)  # blocks — only one client at a time


if __name__ == "__main__":
    run_server()
```

**Experiment 1**: Open two terminal windows. Connect both with `nc 127.0.0.1 9000`. Observe that the second client hangs until the first disconnects. This is the single-threaded limitation.

---

## Part 2 — Multi-Threaded Echo Server (45 min)

```python
"""
echo_server_v2.py — Multi-threaded TCP echo server.
Each accepted connection gets its own thread.
"""
from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-15s %(message)s")
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 9001
BUFSIZE = 4096


@dataclass
class ServerStats:
    """Thread-safe connection counters."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_connections: int = 0
    active_connections: int = 0
    total_bytes: int = 0

    def connection_opened(self) -> None:
        with self._lock:
            self.total_connections += 1
            self.active_connections += 1

    def connection_closed(self, bytes_transferred: int) -> None:
        with self._lock:
            self.active_connections -= 1
            self.total_bytes += bytes_transferred

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total": self.total_connections,
                "active": self.active_connections,
                "bytes": self.total_bytes,
            }


def handle_client(conn: socket.socket, addr: tuple[str, int], stats: ServerStats) -> None:
    stats.connection_opened()
    bytes_received = 0
    log.info("Connection from %s:%d  [active=%d]", *addr, stats.snapshot()["active"])
    try:
        with conn:
            while True:
                data = conn.recv(BUFSIZE)
                if not data:
                    break
                bytes_received += len(data)
                conn.sendall(data)
    except ConnectionResetError:
        log.warning("Client %s:%d reset connection", *addr)
    finally:
        stats.connection_closed(bytes_received)
        log.info("Closed %s:%d  bytes=%d  stats=%s", *addr, bytes_received, stats.snapshot())


def run_server(max_threads: int = 50) -> None:
    stats = ServerStats()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(128)
        log.info("Multi-threaded echo server on %s:%d (max_threads=%d)", HOST, PORT, max_threads)
        while True:
            conn, addr = server.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, stats),
                daemon=True,
                name=f"Client-{addr[1]}",
            )
            t.start()


if __name__ == "__main__":
    run_server()
```

**Experiment 2**: Connect 5 clients simultaneously with `for i in {1..5}; do nc 127.0.0.1 9001 &; done`. All should connect and echo concurrently.

---

## Part 3 — Line-Protocol Request Handler (60 min)

Add a simple application-layer protocol on top of TCP: `COMMAND ARG\n`.

```python
"""
echo_server_v3.py — TCP server with line-delimited command protocol.

Protocol:
  ECHO <text>\n  → responds: OK <text>\n
  STATS\n        → responds: OK total=N active=M bytes=B\n
  QUIT\n         → responds: BYE\n and closes connection

Run: python echo_server_v3.py
Test: nc 127.0.0.1 9002
      ECHO hello world
      STATS
      QUIT
"""
from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-15s %(message)s")
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 9002


@dataclass
class ServerStats:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_connections: int = 0
    active_connections: int = 0
    total_bytes: int = 0

    def connection_opened(self) -> None:
        with self._lock:
            self.total_connections += 1
            self.active_connections += 1

    def connection_closed(self, bytes_transferred: int) -> None:
        with self._lock:
            self.active_connections -= 1
            self.total_bytes += bytes_transferred

    def to_string(self) -> str:
        with self._lock:
            return f"total={self.total_connections} active={self.active_connections} bytes={self.total_bytes}"


def recv_line(conn: socket.socket, max_bytes: int = 1024) -> str | None:
    """Read until \\n or connection close. Returns decoded line or None on disconnect."""
    buf = bytearray()
    while True:
        byte = conn.recv(1)
        if not byte:
            return None
        if byte == b"\n":
            return buf.decode(errors="replace").strip()
        buf.extend(byte)
        if len(buf) > max_bytes:
            return buf.decode(errors="replace").strip()


def handle_client(conn: socket.socket, addr: tuple[str, int], stats: ServerStats) -> None:
    stats.connection_opened()
    bytes_transferred = 0
    log.info("Connect from %s:%d", *addr)
    try:
        with conn:
            conn.sendall(b"READY\n")
            while True:
                line = recv_line(conn)
                if line is None:
                    break
                log.info("CMD from %s:%d → %r", *addr, line)
                parts = line.split(" ", 1)
                command = parts[0].upper()
                arg = parts[1] if len(parts) > 1 else ""

                if command == "ECHO":
                    response = f"OK {arg}\n"
                elif command == "STATS":
                    response = f"OK {stats.to_string()}\n"
                elif command == "QUIT":
                    conn.sendall(b"BYE\n")
                    bytes_transferred += 4
                    break
                else:
                    response = f"ERR unknown command: {command}\n"

                encoded = response.encode()
                conn.sendall(encoded)
                bytes_transferred += len(encoded)
    except (ConnectionResetError, BrokenPipeError):
        log.warning("Client %s:%d disconnected unexpectedly", *addr)
    finally:
        stats.connection_closed(bytes_transferred)
        log.info("Closed %s:%d", *addr)


def run_server() -> None:
    stats = ServerStats()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(128)
        log.info("Protocol server on %s:%d", HOST, PORT)
        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr, stats),
                daemon=True,
                name=f"C-{addr[1]}",
            ).start()


if __name__ == "__main__":
    run_server()
```

---

## Part 4 — Concurrent Benchmark Client (45 min)

```python
"""
benchmark_client.py — Concurrent TCP client that stress-tests the echo server.

Usage: python benchmark_client.py --host 127.0.0.1 --port 9002 --clients 20 --requests 50
"""
from __future__ import annotations

import argparse
import logging
import socket
import statistics
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-15s %(message)s")
log = logging.getLogger(__name__)


def client_session(
    host: str,
    port: int,
    n_requests: int,
    results: list[float],
    lock: threading.Lock,
) -> None:
    latencies: list[float] = []
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            # Read READY greeting
            sock.recv(10)
            for i in range(n_requests):
                msg = f"ECHO payload_{i}\n".encode()
                t0 = time.perf_counter()
                sock.sendall(msg)
                resp = b""
                while b"\n" not in resp:
                    chunk = sock.recv(256)
                    if not chunk:
                        break
                    resp += chunk
                latencies.append((time.perf_counter() - t0) * 1000)
            sock.sendall(b"QUIT\n")
    except Exception as exc:
        log.error("Client error: %s", exc)

    with lock:
        results.extend(latencies)


def run_benchmark(host: str, port: int, n_clients: int, n_requests: int) -> None:
    results: list[float] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=client_session,
            args=(host, port, n_requests, results, lock),
            name=f"Client-{i}",
        )
        for i in range(n_clients)
    ]

    log.info("Launching %d clients × %d requests each = %d total", n_clients, n_requests, n_clients * n_requests)
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    if results:
        log.info("=== Results ===")
        log.info("Total requests : %d", len(results))
        log.info("Total time     : %.3f s", elapsed)
        log.info("Throughput     : %.1f req/s", len(results) / elapsed)
        log.info("Latency p50    : %.2f ms", statistics.median(results))
        log.info("Latency p95    : %.2f ms", statistics.quantiles(results, n=20)[18])
        log.info("Latency p99    : %.2f ms", statistics.quantiles(results, n=100)[98])
        log.info("Latency max    : %.2f ms", max(results))
    else:
        log.warning("No results collected — check server is running")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.host, args.port, args.clients, args.requests)
```

---

## Part 5 — Experiments & Observations (60 min)

Run these experiments in order, recording your observations for the project write-up (Day 7).

### Experiment A — Thread count vs throughput
Run `benchmark_client.py` with `--clients 1 5 10 20 50 100`. Plot p50 latency and throughput against client count. At what point does performance degrade? Why?

### Experiment B — OS connection backlog
Start `echo_server_v2.py` with `server.listen(1)` (backlog=1). Run `--clients 50`. Observe connection refused errors. Restore backlog to 128.

### Experiment C — TIME_WAIT observation
After the benchmark, run `netstat -an | grep TIME_WAIT` (macOS/Linux). Count TIME_WAIT sockets. Explain why they exist.

### Experiment D — Nagle's algorithm
Add `conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)` to `handle_client` in v3. Re-run benchmark. Does latency change for small messages?

### Experiment E — CPU vs I/O characterization
Run `top` or `htop` while the benchmark runs. Is the bottleneck CPU, memory, or I/O? Which part of the code causes context switches?

---

## Exercises (complete after experiments)

1. Add a `PING` command to the v3 server that responds `PONG\n`. Add a test using `subprocess` to start the server and a client socket to verify the response.
2. The current `recv_line` reads one byte at a time — inefficient. Rewrite it to use a buffer (`bytearray`) and `recv(4096)`, splitting on `\n`.
3. Add a semaphore to `run_server` (v3) that limits concurrent connections to 10. What happens to the 11th client?
4. Add graceful shutdown: catch `SIGINT` in the server, stop accepting new connections, wait for existing threads to finish (use `threading.Event`).
5. Write a pytest suite with at least 3 tests for the `ServerStats` class, verifying thread-safety by running 100 concurrent `connection_opened` calls.

(Solutions: ../solutions/week-06/day-6.md)

## Self-check

- [ ] I can start the v3 server and interact with it manually using `nc`.
- [ ] I can explain why the single-threaded server blocks on the second client.
- [ ] I can run the benchmark client and interpret p50/p95/p99 latency.
- [ ] I can observe TIME_WAIT sockets and explain their purpose.
- [ ] I can implement thread-safe shared state using a mutex.

## Supplementary

- [Beej's Guide — server lifecycle](https://beej.us/guide/bgnet/html/#clientserver)
- [Python docs — socketserver module](https://docs.python.org/3/library/socketserver.html)
- [Measuring latency percentiles](https://www.brendangregg.com/FrequencyTrails/percentiles.html)
