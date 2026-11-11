# Solutions — Week 06 · Day 5 — HTTP/HTTPS, TLS, DNS & Sockets

## Exercise 1 — Follow HTTP redirects

```python
def https_get_follow_redirects(
    url: str = "http://example.com/",
    max_hops: int = 5,
) -> str:
    import socket
    import ssl
    import urllib.parse

    for hop in range(max_hops):
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        request = f"GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n"

        raw = socket.create_connection((host, port), timeout=5)
        if parsed.scheme == "https":
            ctx = ssl.create_default_context()
            conn: socket.socket = ctx.wrap_socket(raw, server_hostname=host)
        else:
            conn = raw

        with conn:
            conn.sendall(request.encode())
            parts: list[bytes] = []
            while chunk := conn.recv(4096):
                parts.append(chunk)

        response = b"".join(parts).decode(errors="replace")
        status_line = response.split("\r\n", 1)[0]
        code = int(status_line.split()[1])

        if code in (301, 302, 303, 307, 308):
            for line in response.split("\r\n"):
                if line.lower().startswith("location:"):
                    url = line.split(":", 1)[1].strip()
                    break
        else:
            return response

    raise RuntimeError(f"Too many redirects (>{max_hops})")
```

## Exercise 2 — Certificate SAN parsing

```python
def get_cert_sans(host: str = "example.com") -> list[str]:
    import socket
    import ssl
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    pem = ssl.get_server_certificate((host, 443))
    cert = x509.load_pem_x509_certificate(pem.encode(), default_backend())
    san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    return san_ext.value.get_values_for_type(x509.DNSName)

# Output for example.com: ['www.example.com', 'example.com', ...]
```

## Exercise 3 — DNS TXT / SPF via dnspython

```python
def get_spf_record(domain: str = "google.com") -> list[str]:
    import dns.resolver

    answers = dns.resolver.resolve(domain, "TXT")
    spf_records = []
    for rdata in answers:
        for txt_string in rdata.strings:
            decoded = txt_string.decode()
            if decoded.startswith("v=spf1"):
                spf_records.append(decoded)
    return spf_records
```

## Exercise 4 — DNS resolution timing

```python
import socket
import statistics
import time

DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "reddit.com", "linkedin.com", "netflix.com",
    "github.com", "stackoverflow.com", "cloudflare.com", "microsoft.com", "apple.com",
    "openai.com", "anthropic.com", "python.org", "fastapi.tiangolo.com", "pydantic.dev",
]

def measure_dns(domains: list[str]) -> dict[str, float]:
    times: dict[str, float] = {}
    for d in domains:
        t0 = time.perf_counter()
        try:
            socket.getaddrinfo(d, 443)
        except socket.gaierror:
            pass
        times[d] = (time.perf_counter() - t0) * 1000
    return times

# First run
first = measure_dns(DOMAINS)
# Second run (OS/resolver cache active)
second = measure_dns(DOMAINS)

print(f"First run  p50={statistics.median(first.values()):.2f}ms")
print(f"Second run p50={statistics.median(second.values()):.2f}ms  (should be faster due to OS cache)")
```

**Caching effect**: second run is typically 10–100× faster because the OS stub resolver has cached results. To verify isolation, flush the DNS cache between runs (`sudo dscacheutil -flushcache` on macOS, `sudo systemd-resolve --flush-caches` on Linux).

## Exercise 5 — Why HTTPS needed even with known IP

**Man-in-the-Middle (MITM) attack**: even if you know the IP address `93.184.216.34`, an attacker on the network path (BGP hijack, ARP spoofing, rogue router) can intercept TCP connections at that IP and respond in your place. Without TLS, the attacker reads and modifies all traffic.

TLS's certificate verification prevents this by proving the server possesses the private key matching the certificate signed by a trusted CA for the domain. An attacker cannot forge this proof (they don't have the private key). Without cert verification (`ssl.CERT_NONE`), you'd have encryption but no authentication — vulnerable to MITM because the attacker can present their own cert.
