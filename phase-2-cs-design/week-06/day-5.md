# Phase 2 · Week 06 · Day 5 — HTTP/HTTPS, TLS Handshake, DNS & Sockets in Python

> Objective: Understand HTTP semantics, the TLS handshake, DNS resolution, and build sockets in Python.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-4]]

## Concept

### HTTP — HyperText Transfer Protocol

HTTP is a **stateless request-response application protocol** running over TCP (HTTP/1.1, 2) or UDP/QUIC (HTTP/3).

**Request structure:**
```
GET /api/users?page=1 HTTP/1.1\r\n
Host: api.example.com\r\n
Accept: application/json\r\n
Authorization: Bearer <token>\r\n
\r\n
```

**Response structure:**
```
HTTP/1.1 200 OK\r\n
Content-Type: application/json\r\n
Content-Length: 42\r\n
\r\n
{"users": [...]}
```

**Methods**: GET (idempotent, safe), POST (create), PUT (replace, idempotent), PATCH (partial update), DELETE (idempotent), OPTIONS, HEAD.

**Status code families:**
| Range | Meaning | Examples |
|-------|---------|---------|
| 1xx | Informational | 101 Upgrade |
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirect | 301 Moved, 302 Found, 304 Not Modified |
| 4xx | Client error | 400 Bad Request, 401 Unauthorized, 404 Not Found, 429 Rate Limited |
| 5xx | Server error | 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable |

**HTTP/1.1 vs HTTP/2 vs HTTP/3:**
| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---------|----------|--------|--------|
| Transport | TCP | TCP | UDP (QUIC) |
| Multiplexing | No (head-of-line blocking) | Yes (streams) | Yes + no HoL at transport |
| Header compression | No | HPACK | QPACK |
| Server push | No | Yes | Yes |

### HTTPS & TLS Handshake

HTTPS = HTTP + TLS (Transport Layer Security). TLS encrypts the TCP stream between client and server.

**TLS 1.3 handshake (simplified):**
```
Client                              Server
  ──── ClientHello ─────────────────►  (TLS version, cipher suites, key_share)
  ◄─── ServerHello ─────────────────  (chosen cipher, server key_share)
  ◄─── Certificate ─────────────────  (X.509 cert with public key)
  ◄─── CertificateVerify ───────────  (signature proving private key)
  ◄─── Finished (encrypted) ────────  (MAC of all handshake messages)
  ──── Finished (encrypted) ────────►
  ════ Application data (encrypted) ══
```

Key concepts:
- **Certificate**: X.509 — binds domain name to public key, signed by a Certificate Authority (CA).
- **Chain of trust**: CA hierarchy — root CAs (in OS/browser trust store) → intermediate CAs → leaf cert.
- **Key exchange**: TLS 1.3 uses ECDHE (Elliptic Curve Diffie-Hellman Ephemeral) → **forward secrecy**: compromising the server's long-term key doesn't decrypt past sessions.
- **HSTS** (HTTP Strict Transport Security): tells browser to always use HTTPS.

**TLS 1.2 vs 1.3:**
- 1.2: 2 round-trips before data. 1.3: 1 round-trip (or 0-RTT resumption).
- 1.3 removed weak ciphers (RC4, DES, SHA-1, RSA key exchange without FS).

### DNS — Domain Name System

DNS is a **distributed hierarchical database** mapping domain names to IP addresses.

**Resolution chain:**
```
Browser → OS resolver
       → Recursive resolver (ISP or 8.8.8.8)
         → Root nameserver (.)
           → TLD nameserver (.com)
             → Authoritative nameserver (example.com)
               → Returns A record: 93.184.216.34
```

**Record types:**
| Type | Purpose | Example |
|------|---------|---------|
| A | IPv4 address | `example.com → 93.184.216.34` |
| AAAA | IPv6 address | |
| CNAME | Canonical name (alias) | `www → example.com` |
| MX | Mail exchange | |
| TXT | Arbitrary text (SPF, DKIM) | |
| NS | Nameserver | |
| PTR | Reverse lookup (IP → name) | |

**TTL**: DNS records carry a time-to-live; resolvers cache until TTL expires. Low TTL = fresh but more queries.

### Python Sockets — Raw API

The `socket` module exposes POSIX sockets directly.

**Socket lifecycle:**
```
Server                           Client
socket()                         socket()
bind(address)
listen()
accept() ← blocks ─────────────  connect(server_addr)
recv()/send()  ◄───────────────► send()/recv()
close()                          close()
```

For UDP: skip `listen()`/`accept()`/`connect()`. Use `sendto()`/`recvfrom()`.

## How It Works Internally

### HTTP Keep-Alive & Connection Reuse

HTTP/1.1 defaults to `Connection: keep-alive` — the TCP connection stays open for multiple requests, amortizing three-way handshake cost. `requests` library manages a connection pool automatically.

### TLS Session Resumption

TLS 1.3 supports **session tickets** (0-RTT): client stores an encrypted token; on reconnect sends it with the first message, allowing data to flow in the same round-trip as the handshake. Trade-off: 0-RTT is vulnerable to replay attacks — safe only for idempotent requests.

### DNS Caching Layers

1. Browser cache (Chrome: 60 s default).
2. OS stub resolver cache.
3. Recursive resolver cache (respects TTL).
4. Authoritative server.

`dig +trace example.com` shows the full chain in your terminal.

## Worked Example

```python
"""
day5_http_dns_sockets.py — HTTP raw socket, DNS lookup, TLS inspection.
Python 3.12+
"""
from __future__ import annotations

import logging
import socket
import ssl
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── Raw HTTP over TCP socket ───────────────────────────────────────────────────

def raw_http_get(host: str = "example.com", path: str = "/", port: int = 80) -> str:
    """Send HTTP/1.0 GET using raw socket to see the protocol directly."""
    log.info("=== Raw HTTP GET %s%s ===", host, path)
    request = (
        f"GET {path} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(request.encode())
        response_parts: list[bytes] = []
        while chunk := sock.recv(4096):
            response_parts.append(chunk)
    response = b"".join(response_parts).decode(errors="replace")
    header_end = response.find("\r\n\r\n")
    headers = response[:header_end] if header_end != -1 else response[:200]
    log.info("Response headers:\n%s", headers)
    return response


# ── HTTPS with TLS ────────────────────────────────────────────────────────────

def https_get(host: str = "example.com", path: str = "/", port: int = 443) -> str:
    """HTTPS GET — observe TLS negotiation details."""
    log.info("=== HTTPS GET %s%s ===", host, path)
    context = ssl.create_default_context()  # verifies cert against system trust store
    request = (
        f"GET {path} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    with socket.create_connection((host, port), timeout=5) as raw_sock:
        with context.wrap_socket(raw_sock, server_hostname=host) as tls_sock:
            log.info("TLS version : %s", tls_sock.version())
            log.info("Cipher      : %s", tls_sock.cipher())
            cert = tls_sock.getpeercert()
            log.info("Subject     : %s", cert.get("subject"))
            log.info("Issuer      : %s", cert.get("issuer"))
            log.info("Not after   : %s", cert.get("notAfter"))
            tls_sock.sendall(request.encode())
            response_parts: list[bytes] = []
            while chunk := tls_sock.recv(4096):
                response_parts.append(chunk)
    response = b"".join(response_parts).decode(errors="replace")
    return response


# ── DNS resolution ────────────────────────────────────────────────────────────

def dns_lookup(hostname: str = "example.com") -> None:
    log.info("=== DNS lookup: %s ===", hostname)

    # getaddrinfo returns full socket info — handles IPv4 and IPv6
    results = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    for family, stype, proto, canonname, sockaddr in results:
        log.info(
            "family=%-8s addr=%-20s port=%d",
            socket.AddressFamily(family).name,
            sockaddr[0],
            sockaddr[1],
        )

    # Reverse lookup
    try:
        name, _, _ = socket.gethostbyaddr(results[0][4][0])
        log.info("Reverse DNS: %s → %s", results[0][4][0], name)
    except socket.herror:
        log.info("Reverse DNS: no PTR record")


# ── HTTP status code parser ────────────────────────────────────────────────────

def parse_status_line(response: str) -> tuple[str, int, str]:
    """Extract (HTTP version, status code, reason) from raw response."""
    first_line = response.split("\r\n", 1)[0]
    parts = first_line.split(" ", 2)
    return parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ""


if __name__ == "__main__":
    # HTTP
    raw_resp = raw_http_get()
    version, code, reason = parse_status_line(raw_resp)
    log.info("Parsed: %s %d %s", version, code, reason)

    # HTTPS + TLS inspection
    https_get()

    # DNS
    dns_lookup("google.com")
```

## Exercises

1. Extend `https_get` to follow HTTP 301/302 redirects (up to 5 hops). Test with `http://example.com` (no S).
2. Use `ssl.get_server_certificate(('example.com', 443))` and `ssl.DER_cert_to_PEM_cert` to save the certificate to a file. Parse the PEM with Python's `cryptography` library (install separately) and print the SANs (Subject Alternative Names).
3. Query DNS for `TXT` records of `google.com` using `dnspython` (`pip install dnspython`) and print any SPF record found.
4. Write a function that times the DNS resolution of a list of 20 popular domains and returns the slowest and fastest. Does caching affect repeat runs? How would you test that?
5. Explain why HTTPS is needed even when connecting to a known IP address (bypassing DNS). What attack does TLS's cert verification prevent?

(Solutions: ../solutions/week-06/day-5.md)

## Self-check

- [ ] I can describe the TLS 1.3 handshake in four steps.
- [ ] I can use `ssl.wrap_socket` to inspect a certificate's fields.
- [ ] I can list four DNS record types and their purpose.
- [ ] I can implement HTTP GET over a raw TCP socket.
- [ ] I can explain forward secrecy and why TLS 1.3 provides it.

## Supplementary

- [MDN — HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [TLS 1.3 RFC 8446](https://www.rfc-editor.org/rfc/rfc8446)
- [DNS and How it Works](https://howdns.works/)
- [Python docs — ssl module](https://docs.python.org/3/library/ssl.html)
- [Cloudflare Learning — What is DNS?](https://www.cloudflare.com/learning/dns/what-is-dns/)
