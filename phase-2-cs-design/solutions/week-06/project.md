# Solutions — Week 06 Project: TCP KV Store

## Complete server.py

```python
"""
server.py — TCP KV store server (complete project solution).
Python 3.12+

Usage:
    python server.py [--host HOST] [--port PORT]
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
    """Thread-safe counters for connections and commands."""
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


def recv_line(conn: socket.socket, max_bytes: int = 4096) -> str | None:
    """Read until newline or EOF using a buffer (one syscall per chunk)."""
    buf = bytearray()
    while True:
        chunk = conn.recv(256)
        if not chunk:
            return None
        buf.extend(chunk)
        if b"\n" in buf:
            line, _, _ = buf.partition(b"\n")
            return line.decode(errors="replace").strip()
        if len(buf) > max_bytes:
            return buf.decode(errors="replace").strip()


def dispatch(
    cmd: str,
    args: list[str],
    store: KVStore,
    stats: ServerStats,
) -> bytes:
    """Process a single command, return response bytes."""
    if cmd == "SET":
        if len(args) < 2:
            return b"ERR SET requires key and value\n"
        store.set(args[0], " ".join(args[1:]))
        return b"OK\n"
    if cmd == "GET":
        if not args:
            return b"ERR GET requires key\n"
        val = store.get(args[0])
        return f"OK {val}\n".encode() if val is not None else b"ERR not found\n"
    if cmd == "DEL":
        if not args:
            return b"ERR DEL requires key\n"
        return b"OK\n" if store.delete(args[0]) else b"ERR not found\n"
    if cmd == "KEYS":
        keys = store.keys()
        return f"OK {','.join(keys)}\n".encode()
    if cmd == "STATS":
        conns, cmds = stats.snapshot()
        return f"OK connections={conns} cmds={cmds}\n".encode()
    if cmd == "QUIT":
        return b"__QUIT__"
    return f"ERR unknown command: {cmd}\n".encode()


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
                parts = line.split()
                if not parts:
                    continue
                cmd = parts[0].upper()
                args = parts[1:]
                log.info("CMD %s %s from %s:%d", cmd, args, *addr)
                response = dispatch(cmd, args, store, stats)
                if response == b"__QUIT__":
                    conn.sendall(b"BYE\n")
                    break
                conn.sendall(response)
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
    log.info("KV server listening on %s:%d", host, port)

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
    parser = argparse.ArgumentParser(description="TCP KV store server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args()
    run_server(args.host, args.port)
```

---

## Complete client.py

```python
"""
client.py — Concurrent benchmark client for KV server.
Python 3.12+

Usage:
    python client.py [--host HOST] [--port PORT] [--clients N] [--requests N]
"""
from __future__ import annotations

import argparse
import logging
import socket
import statistics
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def recv_response(sock: socket.socket) -> str:
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b or b == b"\n":
            return buf.decode(errors="replace").strip()
        buf.extend(b)


def client_session(
    host: str,
    port: int,
    n_requests: int,
    latencies: list[float],
    lock: threading.Lock,
) -> None:
    local: list[float] = []
    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            recv_response(sock)  # READY
            for i in range(n_requests):
                key = f"bench_{threading.get_ident()}_{i}"
                # SET
                t0 = time.perf_counter()
                sock.sendall(f"SET {key} value_{i}\n".encode())
                recv_response(sock)
                local.append((time.perf_counter() - t0) * 1000)
                # GET
                t0 = time.perf_counter()
                sock.sendall(f"GET {key}\n".encode())
                recv_response(sock)
                local.append((time.perf_counter() - t0) * 1000)
            sock.sendall(b"QUIT\n")
    except Exception as exc:
        log.error("Session error: %s", exc)
    finally:
        with lock:
            latencies.extend(local)


def run_benchmark(host: str, port: int, n_clients: int, n_requests: int) -> None:
    latencies: list[float] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=client_session,
            args=(host, port, n_requests, latencies, lock),
            name=f"BClient-{i}",
        )
        for i in range(n_clients)
    ]
    log.info("Starting %d clients × %d requests each", n_clients, n_requests)
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    if not latencies:
        log.warning("No latency data collected")
        return

    q = statistics.quantiles(latencies, n=100)
    log.info("=== Benchmark Results ===")
    log.info("Total requests : %d", len(latencies))
    log.info("Elapsed        : %.3f s", elapsed)
    log.info("Throughput     : %.1f req/s", len(latencies) / elapsed)
    log.info("Latency p50    : %.2f ms", statistics.median(latencies))
    log.info("Latency p95    : %.2f ms", q[94])
    log.info("Latency p99    : %.2f ms", q[98])
    log.info("Latency max    : %.2f ms", max(latencies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV server benchmark client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.host, args.port, args.clients, args.requests)
```

---

## Complete test_server.py

```python
"""
test_server.py — pytest suite for KV server.
Python 3.12+

Run: pytest test_server.py -v
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

import pytest

# Import server components directly for unit tests
# Adjust path if running from different directory
sys.path.insert(0, ".")
from server import KVStore, ServerStats, dispatch, run_server


# ── Fixtures ──────────────────────────────────────────────────────────────────

_SERVER_PORT = 19876  # use a non-standard port to avoid conflicts


@pytest.fixture(scope="module")
def server() -> Generator[int, None, None]:
    """Start a real server in a background daemon thread."""
    t = threading.Thread(
        target=run_server,
        args=("127.0.0.1", _SERVER_PORT),
        daemon=True,
        name="TestServer",
    )
    t.start()
    time.sleep(0.2)  # wait for bind
    yield _SERVER_PORT


def make_client(port: int) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    # read READY
    buf = b""
    while b"\n" not in buf:
        buf += sock.recv(64)
    return sock


def send_cmd(sock: socket.socket, cmd: str) -> str:
    sock.sendall(f"{cmd}\n".encode())
    buf = b""
    while b"\n" not in buf:
        buf += sock.recv(256)
    return buf.split(b"\n", 1)[0].decode().strip()


# ── Unit tests (no network) ───────────────────────────────────────────────────

def test_kvstore_set_get():
    store = KVStore()
    store.set("k", "v")
    assert store.get("k") == "v"


def test_kvstore_delete():
    store = KVStore()
    store.set("k", "v")
    assert store.delete("k") is True
    assert store.get("k") is None
    assert store.delete("k") is False


def test_kvstore_keys():
    store = KVStore()
    store.set("a", "1")
    store.set("b", "2")
    assert sorted(store.keys()) == ["a", "b"]


def test_server_stats_thread_safe():
    stats = ServerStats()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = [pool.submit(stats.record_connection) for _ in range(100)]
        for f in futs:
            f.result()
    assert stats.total_connections == 100


def test_dispatch_set_get():
    store, stats = KVStore(), ServerStats()
    assert dispatch("SET", ["mykey", "myval"], store, stats) == b"OK\n"
    assert dispatch("GET", ["mykey"], store, stats) == b"OK myval\n"


def test_dispatch_get_missing():
    store, stats = KVStore(), ServerStats()
    assert dispatch("GET", ["nokey"], store, stats) == b"ERR not found\n"


# ── Integration tests (real socket) ───────────────────────────────────────────

def test_set_and_get(server: int):
    with make_client(server) as s:
        assert send_cmd(s, "SET hello world") == "OK"
        assert send_cmd(s, "GET hello") == "OK world"


def test_get_missing_key(server: int):
    with make_client(server) as s:
        resp = send_cmd(s, "GET __nonexistent_key_xyz__")
        assert resp == "ERR not found"


def test_del_key(server: int):
    with make_client(server) as s:
        send_cmd(s, "SET delme value")
        assert send_cmd(s, "DEL delme") == "OK"
        assert send_cmd(s, "GET delme") == "ERR not found"


def test_del_missing(server: int):
    with make_client(server) as s:
        assert send_cmd(s, "DEL __nosuchkey__") == "ERR not found"


def test_keys_command(server: int):
    with make_client(server) as s:
        send_cmd(s, "SET k1 v1")
        send_cmd(s, "SET k2 v2")
        resp = send_cmd(s, "KEYS")
        assert resp.startswith("OK")
        key_list = resp[3:].split(",") if resp != "OK " else []
        assert "k1" in key_list
        assert "k2" in key_list


def test_stats_increments(server: int):
    with make_client(server) as s:
        r1 = send_cmd(s, "STATS")
        assert r1.startswith("OK connections=")
        send_cmd(s, "SET stat_key val")
        r2 = send_cmd(s, "STATS")
        cmds1 = int(r1.split("cmds=")[1])
        cmds2 = int(r2.split("cmds=")[1])
        assert cmds2 > cmds1


def test_quit_closes_connection(server: int):
    with make_client(server) as s:
        resp = send_cmd(s, "QUIT")
        assert resp == "BYE"
        time.sleep(0.05)
        s.settimeout(1)
        data = s.recv(64)
        assert data == b""  # server closed, EOF


def test_concurrent_set_thread_safe(server: int):
    """10 threads set distinct keys concurrently; all keys must be present."""
    def set_key(i: int) -> None:
        with make_client(server) as s:
            assert send_cmd(s, f"SET concurrent_{i} val_{i}") == "OK"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(set_key, i) for i in range(10)]
        for f in futs:
            f.result()

    with make_client(server) as s:
        for i in range(10):
            resp = send_cmd(s, f"GET concurrent_{i}")
            assert resp == f"OK val_{i}", f"Missing key concurrent_{i}"
```

---

## Sample concepts.md

```markdown
# Week 06 — Concept Mapping

| Observed behavior | Concept | Explanation |
|---|---|---|
| Single-threaded server blocks second client | Single-threaded accept loop | `accept()` blocks until first client disconnects; OS queues the second in the listen backlog |
| Multi-threaded server handles 10 clients at once | Thread-per-connection; OS preemptive scheduling | Each thread gets its own stack; CFS scheduler time-slices across all ready threads |
| `active_connections` corrupted without mutex | Race condition on shared mutable state | Read-modify-write is not atomic; threads interleave between load and store bytecodes |
| TIME_WAIT sockets after benchmark | TCP 4-way FIN + TIME_WAIT (2×MSL) | OS keeps socket in TIME_WAIT to handle retransmitted FINs and prevent stale segment confusion |
| `listen(1)` causes connection refusals under load | TCP accept backlog queue | Kernel drops SYN when queue full; client gets RST (ECONNREFUSED) |
| Latency degrades past 50 threads | Context-switch overhead + GIL contention | More threads = more scheduling overhead; GIL forces serial bytecode execution |
| Large `bytearray` allocation doesn't spike RSS | Virtual memory + demand paging | Pages allocated in virtual space but not mapped to physical frames until first write |
| Server binds immediately after crash with SO_REUSEADDR | TIME_WAIT + SO_REUSEADDR | Without flag: `EADDRINUSE` during TIME_WAIT window; with flag: kernel allows rebind |
```
