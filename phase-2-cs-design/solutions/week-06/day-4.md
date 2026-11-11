# Solutions — Week 06 · Day 4 — OSI Model, TCP/IP & TCP vs UDP

## Exercise 1 — OSI stack for HTTP GET

```
Client sends: GET /index.html HTTP/1.1

Layer 7 (Application):  HTTP request line + headers
Layer 6 (Presentation): Character encoding (UTF-8), no separate TLS here (plain HTTP)
Layer 5 (Session):       HTTP keep-alive session tracking
Layer 4 (Transport):     TCP segment — src_port=49152, dst_port=80, seq/ack, flags
Layer 3 (Network):       IP datagram — src_ip=192.168.1.2, dst_ip=93.184.216.34, TTL=64
Layer 2 (Data Link):     Ethernet frame — src_MAC, dst_MAC (gateway MAC), EtherType=0x0800
Layer 1 (Physical):      Electrical/optical signal, radio (Wi-Fi)

Each layer adds a header (and trailer at L2: CRC/FCS).
Reverse happens at server (de-encapsulation).
```

## Exercise 2 — UDP checksum purpose

The UDP checksum verifies the **integrity of the datagram's contents in transit** — bit flips introduced by network hardware, noisy links, or memory errors in routers. It does NOT:
- Guarantee delivery (a corrupted datagram is silently discarded, not retransmitted).
- Prevent out-of-order arrival.
- Detect loss.

The checksum covers the UDP header + payload + a pseudo-header (src/dst IP, protocol, length) — this lets it catch misdelivery (wrong destination IP/port) in addition to corruption.

## Exercise 3 — HTTP/1.0 GET and response

```python
def raw_http_get_with_request(host: str = "example.com") -> str:
    import socket
    request = "GET / HTTP/1.0\r\nHost: example.com\r\n\r\n"
    with socket.create_connection((host, 80), timeout=5) as sock:
        sock.sendall(request.encode())
        parts: list[bytes] = []
        while chunk := sock.recv(4096):
            parts.append(chunk)
    response = b"".join(parts).decode(errors="replace")
    print(response[:200])
    return response
```

Expected: `HTTP/1.0 200 OK\r\nContent-Type: text/html; ...` followed by HTML.

## Exercise 4 — Maximum TCP throughput calculation

**Formula (bandwidth-delay product):**
```
max_throughput = receive_window / RTT
              = 65535 bytes / 0.1 s
              = 655,350 bytes/s
              ≈ 640 KiB/s  ≈  5.1 Mbit/s
```

This is the constraint imposed by the receive window alone. To exceed it, TCP window scaling (RFC 7323) is needed, which extends the window to 1 GiB. Modern TCP stacks auto-negotiate window scaling, making actual throughput much higher on fast links.

## Exercise 5 — TIME_WAIT state

After the initiating side sends the final `ACK` in the four-way FIN exchange, it enters **TIME_WAIT** and stays there for **2 × MSL** (Maximum Segment Lifetime, typically 60 s on Linux → 2 min total).

**Why 2×MSL?**
1. Ensures the final `ACK` reaches the other side. If the `ACK` is lost and the remote sends a retransmitted `FIN`, the TIME_WAIT side must still be able to re-send the `ACK` — if it had already moved to CLOSED, it would send `RST`, which the remote might misinterpret.
2. Prevents old duplicate segments from a closed connection being misinterpreted by a new connection on the same 5-tuple. After 2×MSL, any segment from the old connection has expired in the network.

**Effect on servers**: a server that initiates millions of short connections (or the client side of a reverse proxy) can exhaust ephemeral ports due to TIME_WAIT accumulation. Mitigation: `SO_REUSEADDR`, `TCP_TIMEWAIT_REUSE` sysctl, or using the server as the passive closer.
