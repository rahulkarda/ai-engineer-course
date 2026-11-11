# Solutions — Week 06 · Day 7 — Project Write-Up & Reflection

## Reflection Question Answers

### Q1 — Two alternative concurrency models

**1. asyncio (event loop / coroutines)**
- Trade-off: single thread handles thousands of connections cooperatively via `await`; no OS thread overhead; but CPU-bound work blocks the loop (need `run_in_executor`). Good fit for I/O-heavy servers.

**2. ProcessPoolExecutor + shared memory**
- Trade-off: bypasses GIL, true CPU parallelism, crash isolation per process; but shared state requires `multiprocessing.Manager` or shared memory objects, which are slower than in-process locks. Good for CPU-heavy tasks (ML inference per connection).

### Q2 — Thread-per-connection failure at 10,000 req/s

Each OS thread consumes ~8 MiB of stack space. 10,000 threads = ~80 GiB memory, far beyond typical machines. Additionally:
- Scheduling 10,000 threads causes massive context-switch overhead (each switch ~1–10 µs × 10,000 = 10–100 ms per "round").
- Lock contention on shared state grows with thread count.

**Alternative**: `asyncio` (or `epoll`/`kqueue`-based event loop) multiplexes thousands of connections onto a handful of threads. Python's `asyncio.start_server` handles 10,000+ concurrent connections on a few threads with minimal memory.

### Q3 — Connection refused diagnostic path

1. **Application layer**: server process not running → nothing listening on port.
2. **Transport layer**: server socket not bound to that port, or `listen()` backlog full and `tcp_abort_on_overflow` set.
3. **Network layer**: firewall (`iptables`, `pf`) dropping or rejecting SYNs to that port.
4. **Data link/routing**: wrong IP — packets not reaching host; NAT misconfiguration.

Diagnostics: `netstat -tlnp` (is port open?), `telnet host port` (test connectivity), `tcpdump` (SYN sent? RST received? SYN-ACK received?), `iptables -L` (firewall rules).

### Q4 — `ssl.SSLCertVerificationError` causes

1. **Self-signed certificate** — not in system trust store; fix: add cert to trust store or use `ssl.CERT_NONE` only for internal dev (never production).
2. **Expired certificate** — `notAfter` in the past; fix: renew the cert.
3. **Hostname mismatch** — cert's CN/SAN doesn't match the hostname you connected to; fix: use the correct hostname or update the cert.
4. **Missing intermediate CA** — server only sends leaf cert; client can't build chain to root; fix: configure server to send the full chain.
5. **System trust store outdated** — root CA not present in old Python/OS; fix: `pip install certifi` and use `ssl.create_default_context(cafile=certifi.where())`.

### Q5 — Interaction of scheduler, TCP congestion control, and GIL

When 100 client threads run concurrently against the server:

- **GIL** limits the server to executing one thread's Python bytecode at a time, releasing per 5 ms. Server threads waiting on `recv()` syscalls release the GIL — many threads can block in the kernel simultaneously (I/O-bound, GIL not held).
- **OS scheduler** (CFS) time-slices the server's runnable threads. With 100 client-side threads + server-side threads, the scheduler context-switches every ~5 ms per thread. More threads = more time spent switching, less time doing actual work.
- **TCP congestion control** (slow start + CWND) throttles each connection's send rate independently. With 100 connections on loopback, congestion is minimal (no loss), so CWND grows quickly to fill the socket buffer. However, if the server's accept queue or processing rate lags, clients' sockets buffer data and TCP flow control (advertised window = 0) stalls senders.

Net effect: server throughput peaks around 20–50 threads (I/O overlap benefit exceeds scheduling overhead), then degrades as context-switch cost and GIL contention outweigh the I/O parallelism benefit.
