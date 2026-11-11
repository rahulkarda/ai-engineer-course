# Phase 2 · Week 06 · Day 4 — OSI Model, TCP/IP & TCP vs UDP

> Objective: Map real network traffic to the OSI and TCP/IP models; contrast TCP and UDP.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]]

## Concept

### OSI Model (7 Layers)

The OSI model is a conceptual framework for network protocol design. Each layer provides services to the one above and uses services from the one below.

```
┌─────────────────────────────────────────────────────────┐
│  7 — Application    HTTP, DNS, SMTP, FTP                │
│  6 — Presentation   TLS/SSL, encoding (UTF-8, JSON)     │
│  5 — Session        WebSockets session management       │
│  4 — Transport      TCP, UDP — ports, reliability       │
│  3 — Network        IP — routing, addressing            │
│  2 — Data Link      Ethernet, Wi-Fi — MAC frames        │
│  1 — Physical       Bits on wire, fiber, radio          │
└─────────────────────────────────────────────────────────┘
```

Mnemonic (top-down): **A**ll **P**eople **S**eem **T**o **N**eed **D**ata **P**rocessing.

### TCP/IP Model (4 Layers)

The TCP/IP model collapses OSI into four practical layers:

```
TCP/IP              OSI equivalent
─────────────────   ──────────────────────────────────
Application       ← 7 + 6 + 5
Transport         ← 4
Internet          ← 3
Network Access    ← 2 + 1
```

TCP/IP is the internet's actual implementation; OSI is the theoretical reference.

### IP Addressing & Routing

Each host has an IP address (IPv4 = 32-bit, IPv6 = 128-bit). A **router** at Layer 3 forwards packets based on destination IP, using a **routing table** (longest prefix match). Subnets use CIDR notation: `192.168.1.0/24` = 256 addresses.

### TCP — Transmission Control Protocol

TCP provides **reliable, ordered, connection-oriented** byte-stream delivery.

**Three-way handshake** (connection establishment):
```
Client                    Server
   ──── SYN ──────────────►
   ◄─── SYN-ACK ───────────
   ──── ACK ──────────────►
         (connection established)
```

**Key features:**
- **Sequence numbers** — track byte position; detect duplicates and gaps.
- **Acknowledgements** — sender retransmits unacknowledged segments after timeout.
- **Flow control** — receiver advertises a **receive window** (sliding window); sender doesn't overwhelm receiver.
- **Congestion control** — slow start, congestion avoidance, fast retransmit (CUBIC, BBR on Linux).
- **Four-way FIN** teardown — ensures all data flushed before close.

**Header cost**: 20 bytes minimum per segment + per-connection kernel state.

### UDP — User Datagram Protocol

UDP provides **unreliable, connectionless** datagram delivery.

- No handshake — send immediately.
- No retransmission — if packet lost, it's gone.
- No ordering — datagrams may arrive out-of-order.
- **Header**: only 8 bytes (src port, dst port, length, checksum).
- Application must handle reliability if needed (e.g., QUIC builds its own on UDP).

### When to Use Each

| Use case | Protocol | Why |
|----------|----------|-----|
| Web (HTTP/1.1, 2) | TCP | Reliability matters |
| HTTP/3 / QUIC | UDP | Own retransmit on top |
| DNS queries | UDP | Small, one-shot |
| Video streaming | UDP | Prefer low latency over perfect delivery |
| Online gaming | UDP | Stale frames worthless |
| File transfer (FTP, SCP) | TCP | Every byte must arrive |
| Real-time comms (WebRTC) | UDP | Jitter > loss |

### Ports

Ports (16-bit, 0–65535) allow multiplexing many services on one IP:
- 0–1023: well-known (80=HTTP, 443=HTTPS, 22=SSH, 53=DNS, 25=SMTP).
- 1024–49151: registered.
- 49152–65535: ephemeral (client-side connections).

A **socket** is uniquely identified by the 5-tuple: `(protocol, src_ip, src_port, dst_ip, dst_port)`.

## How It Works Internally

### TCP Segment Journey

```
Application writes "GET / HTTP/1.1\r\n"
  ↓ TCP layer adds header (seq#, ACK, ports)
  ↓ IP layer adds header (src IP, dst IP, TTL)
  ↓ Ethernet adds frame (src MAC, dst MAC, CRC)
  ↓ Wi-Fi / cable → bits on wire
  ... routers strip/rewrite Ethernet header, decrement TTL, forward on IP
  ↓ Server NIC receives bits
  ↓ Ethernet driver checks CRC
  ↓ IP layer checks destination, hands to TCP
  ↓ TCP reassembles stream, delivers to socket buffer
  ↓ Application reads from socket
```

### Congestion Control — Slow Start

TCP starts conservatively: **cwnd = 1 MSS** (maximum segment size, ~1460 bytes). For each ACK, cwnd doubles (exponential) until **ssthresh** (slow start threshold), then increments by 1 MSS/RTT (linear). On packet loss, ssthresh is halved and slow start restarts.

```
cwnd
   |          /
   |        /
   |      /          (linear — congestion avoidance)
   |    *
   |   *   ← ssthresh (exponential growth stops)
   |  *
   | * * *   ← slow start
   +───────────────→ RTT
```

### Nagle's Algorithm

TCP by default buffers small writes and coalesces them into one segment — reduces packet count but adds latency. Applications requiring low-latency (SSH keystroke, gaming) set `TCP_NODELAY` socket option to disable it.

## Worked Example

```python
"""
day4_tcp_udp_inspect.py — Inspect TCP connection state and UDP echo.
Python 3.12+
"""
from __future__ import annotations

import logging
import os
import socket
import struct
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── TCP connection state inspection ───────────────────────────────────────────

def tcp_inspect(host: str = "example.com", port: int = 80) -> None:
    log.info("=== TCP three-way handshake to %s:%d ===", host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    t0 = time.perf_counter()
    try:
        sock.connect((host, port))
        rtt = (time.perf_counter() - t0) * 1000
        log.info("Connected — RTT approx %.1f ms", rtt)
        log.info("Local  addr: %s:%d", *sock.getsockname())
        log.info("Remote addr: %s:%d", *sock.getpeername())

        # Observe socket options
        rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        log.info("SO_RCVBUF=%d  SO_SNDBUF=%d", rcvbuf, sndbuf)
    finally:
        sock.close()
        log.info("Socket closed (FIN sequence triggered)")


# ── UDP echo ───────────────────────────────────────────────────────────────────

def udp_echo_server(host: str = "127.0.0.1", port: int = 9999) -> None:
    """Single-shot UDP echo server — receives one datagram and echoes it."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.settimeout(3)
        log.info("UDP server listening on %s:%d", host, port)
        try:
            data, addr = sock.recvfrom(1024)
            log.info("UDP server received %r from %s:%d", data.decode(), *addr)
            sock.sendto(data, addr)
        except TimeoutError:
            log.warning("UDP server timed out waiting for data")


def udp_echo_client(host: str = "127.0.0.1", port: int = 9999) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(3)
        message = b"hello UDP"
        sock.sendto(message, (host, port))
        log.info("UDP client sent: %r", message.decode())
        try:
            data, _ = sock.recvfrom(1024)
            log.info("UDP client received echo: %r", data.decode())
        except TimeoutError:
            log.warning("UDP client timed out — no echo received (expected if server not running)")


def udp_demo() -> None:
    """Run server and client in the same process for demo purposes."""
    import threading

    log.info("=== UDP echo demo ===")
    server_thread = threading.Thread(target=udp_echo_server, daemon=True)
    server_thread.start()
    time.sleep(0.05)  # let server bind
    udp_echo_client()
    server_thread.join(timeout=2)


# ── Port scanner (loopback only) ───────────────────────────────────────────────

def scan_local_ports(ports: list[int] | None = None) -> None:
    if ports is None:
        ports = [22, 80, 443, 8080, 5432, 6379, 9999]
    log.info("=== Port scan on 127.0.0.1 ===")
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(("127.0.0.1", port))
        status = "OPEN" if result == 0 else "closed"
        log.info("  port %5d : %s", port, status)
        sock.close()


if __name__ == "__main__":
    tcp_inspect()
    udp_demo()
    scan_local_ports()
```

## Exercises

1. Draw the full OSI stack for an HTTP GET request, labeling each layer's header/trailer added.
2. Why does UDP have a checksum if it's "unreliable"? What does the checksum verify vs what it doesn't?
3. Modify `tcp_inspect` to send an HTTP/1.0 GET request (`GET / HTTP/1.0\r\nHost: example.com\r\n\r\n`) and print the first 200 bytes of the response.
4. Calculate the maximum theoretical throughput of a TCP connection with RTT=100 ms and receive window=65535 bytes. Show your work.
5. What is the **TIME_WAIT** state in TCP? Why does it last 2×MSL and what problem does it prevent?

(Solutions: ../solutions/week-06/day-4.md)

## Self-check

- [ ] I can list the 7 OSI layers and one protocol per layer from memory.
- [ ] I can explain the TCP three-way handshake with a diagram.
- [ ] I can explain why UDP is faster and when that trade-off is correct.
- [ ] I can implement a basic UDP echo client/server in Python.
- [ ] I can explain TCP slow start and congestion avoidance.

## Supplementary

- [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
- [TCP/IP Illustrated Vol 1 — Stevens (classic reference)](https://www.pearson.com/en-us/subject-catalog/p/tcp-ip-illustrated-volume-1-the-protocols/P200000009790)
- [Python docs — socket](https://docs.python.org/3/library/socket.html)
- [QUIC and HTTP/3 explainer](https://http3-explained.haxx.se/)
