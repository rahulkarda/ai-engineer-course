# Solutions — Week 06 · Day 6 — TCP Echo Server Build

## Exercise 1 — PING/PONG command + subprocess test

**Server addition** (in v3 `handle_client`):
```python
elif command == "PING":
    response = "PONG\n"
```

**Test using subprocess:**
```python
import socket
import subprocess
import sys
import time
import pytest

@pytest.fixture(scope="module")
def server_proc():
    proc = subprocess.Popen(
        [sys.executable, "echo_server_v3.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.3)  # wait for bind
    yield proc
    proc.terminate()
    proc.wait()

def test_ping_pong(server_proc):
    with socket.create_connection(("127.0.0.1", 9002), timeout=2) as s:
        s.recv(10)  # READY
        s.sendall(b"PING\n")
        resp = b""
        while b"\n" not in resp:
            resp += s.recv(64)
        assert resp.strip() == b"PONG"
```

## Exercise 2 — Buffered recv_line

```python
def recv_line_buffered(conn: socket.socket, max_bytes: int = 4096) -> str | None:
    """Read bytes into a buffer, return complete line when \\n found."""
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return None
        buf.extend(chunk)
        if b"\n" in buf:
            line, _, _ = buf.partition(b"\n")
            return line.decode(errors="replace").strip()
        if len(buf) > max_bytes:
            return buf.decode(errors="replace").strip()
```

**Why faster**: one `recv(4096)` syscall vs up to 4096 `recv(1)` syscalls. Syscall overhead is ~0.1–1 µs each, so 4096 single-byte reads ≈ 0.4–4 ms overhead per line vs one bulk read.

**Caveat**: this implementation discards bytes after `\n`. For a real server reading multiple lines, maintain a per-connection buffer that preserves the remainder after the `\n`.

## Exercise 3 — Semaphore-limited connections

```python
import threading
import socket

MAX_CONNECTIONS = 10
sem = threading.Semaphore(MAX_CONNECTIONS)

def handle_client_limited(conn, addr, stats):
    acquired = sem.acquire(blocking=False)
    if not acquired:
        # Immediately reject
        try:
            conn.sendall(b"ERR server at capacity\n")
        finally:
            conn.close()
        return
    try:
        handle_client(conn, addr, stats)  # original handler
    finally:
        sem.release()
```

The 11th client receives `ERR server at capacity\n` and the connection is closed immediately. This is a **fail-fast** behavior, preferable to blocking the 11th client indefinitely (which would appear as a hang to the user).

## Exercise 4 — Graceful shutdown with SIGINT

```python
import signal
import threading

shutdown_event = threading.Event()

def handle_client_checked(conn, addr, stats):
    """Version that checks shutdown_event between iterations."""
    stats.connection_opened()
    try:
        with conn:
            conn.sendall(b"READY\n")
            while not shutdown_event.is_set():
                conn.settimeout(1.0)  # don't block forever
                try:
                    line = recv_line(conn)
                except TimeoutError:
                    continue
                if line is None:
                    break
                # ... process command ...
    finally:
        stats.connection_closed(0)

active_threads: list[threading.Thread] = []

def run_server_graceful(host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(128)

    def on_sigint(signum, frame):
        import logging
        logging.getLogger(__name__).info("Shutting down")
        shutdown_event.set()
        server.close()

    signal.signal(signal.SIGINT, on_sigint)

    stats = ServerStats()
    while not shutdown_event.is_set():
        try:
            conn, addr = server.accept()
        except OSError:
            break
        t = threading.Thread(target=handle_client_checked, args=(conn, addr, stats), daemon=True)
        active_threads.append(t)
        t.start()

    # Wait for all active connections to finish
    for t in active_threads:
        t.join(timeout=5)
```

## Exercise 5 — ServerStats thread-safety tests

```python
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from echo_server_v2 import ServerStats

def test_connection_opened_increments():
    s = ServerStats()
    s.connection_opened()
    assert s.total_connections == 1
    assert s.active_connections == 1

def test_connection_closed_decrements():
    s = ServerStats()
    s.connection_opened()
    s.connection_closed(100)
    assert s.active_connections == 0
    assert s.total_bytes == 100

def test_concurrent_open_thread_safe():
    s = ServerStats()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(s.connection_opened) for _ in range(100)]
        for f in futures:
            f.result()
    assert s.total_connections == 100
    assert s.active_connections == 100

def test_concurrent_open_close_thread_safe():
    s = ServerStats()
    def open_then_close():
        s.connection_opened()
        s.connection_closed(10)
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(open_then_close) for _ in range(100)]
        for f in futures:
            f.result()
    assert s.total_connections == 100
    assert s.active_connections == 0
    assert s.total_bytes == 1000

def test_snapshot_consistent():
    s = ServerStats()
    s.connection_opened()
    snap = s.snapshot()
    assert snap["total"] == 1
    assert snap["active"] == 1
    assert snap["bytes"] == 0
```
