# Phase 7 · Week 02 · Day 2 — Load Balancing: L4/L7, Algorithms & Health Checks

> Objective: Understand how load balancers distribute traffic, the difference between L4 and L7, and how algorithms + health checks determine routing decisions.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

A **load balancer** sits in front of a pool of servers and distributes incoming requests so that no single server is overwhelmed. It is the traffic cop that makes horizontal scaling work.

### OSI layers relevant to LBs

```
Application  L7  ← HTTP, HTTPS, gRPC, WebSocket  (content-aware routing)
Transport    L4  ← TCP, UDP                        (IP + port routing only)
Network      L3
...
```

### L4 load balancing

Operates at **TCP/UDP level**. The LB sees source IP, destination IP, and port — nothing else. It cannot inspect HTTP headers, paths, or cookies.

- **Faster**: no packet reassembly or HTTP parsing.
- **Simpler**: works for any TCP-based protocol (DB proxies, raw TCP).
- **Blind to content**: cannot route `/api` vs `/static`, cannot do SSL termination without passing raw bytes.

Examples: AWS Network Load Balancer (NLB), HAProxy in TCP mode, Linux IPVS.

### L7 load balancing

Operates at the **HTTP/application layer**. The LB terminates the connection, reads the full HTTP request, and makes routing decisions based on:
- URL path (`/api/v1/*` → service-A, `/images/*` → CDN origin)
- Host header (virtual hosting)
- HTTP method
- Cookie / session affinity
- Request body (rare)

- **More capable**: SSL termination, path-based routing, A/B testing, request rewriting.
- **More expensive**: requires full connection + HTTP parse.
- **Can add latency**: ~1–2 ms per hop on modern hardware; negligible for most apps.

Examples: AWS Application Load Balancer (ALB), NGINX upstream, HAProxy HTTP mode, Envoy.

### Routing algorithms

| Algorithm | Description | Best for |
|-----------|-------------|----------|
| **Round-robin** | Cycle through backends in order | Homogeneous servers, uniform requests |
| **Weighted round-robin** | Cycle with frequency proportional to weight | Heterogeneous hardware |
| **Least connections** | Route to backend with fewest open connections | Long-lived or variable-cost requests |
| **Least response time** | Route to fastest-responding backend | Latency-sensitive, heterogeneous load |
| **IP hash** | `hash(client_ip) % N` | Session stickiness without a shared store |
| **Consistent hash** | Ring-based hash (see Day 6) | Caches, stateful backends, minimises reshuffling on scale events |
| **Random** | Pick uniformly at random | Simple, load-balances in expectation |

### Health checks

A backend that is down must be removed from rotation. The LB periodically probes each backend:

- **Active health check**: LB sends a synthetic request (e.g., `GET /healthz`) and expects a 200 response within a timeout.
- **Passive health check**: LB observes real traffic; if N consecutive requests to a backend fail, it marks it unhealthy.

Parameters:
- `interval`: how often to probe (e.g., 5 s)
- `timeout`: how long to wait for a response (e.g., 2 s)
- `healthy_threshold`: consecutive successes to mark healthy (e.g., 2)
- `unhealthy_threshold`: consecutive failures to mark unhealthy (e.g., 3)

## How it works internally

### Connection flow: L7

```
Client ──TCP SYN──► LB ──TCP SYN──► Backend-A
Client ◄─TCP SYN-ACK── LB ◄─TCP SYN-ACK── Backend-A
Client ─HTTP GET /api─► LB  [reads path, picks backend]
LB ─HTTP GET /api─► Backend-A
LB ◄─HTTP 200─── Backend-A
Client ◄─HTTP 200─── LB
```

The LB maintains **two separate TCP connections**: client↔LB and LB↔backend. This allows connection pooling (LB reuses backend connections) and SSL offload.

### Consistent hashing sketch

Standard modulo hashing: `backend = hash(key) % N`. Problem: when N changes (scale event), almost every key remaps → cache thrash.

Consistent hashing: place backends on a ring of integers [0, 2^32). A key maps to the first backend clockwise from `hash(key)`. Adding/removing a node remaps only `keys / N` on average — a fraction, not the whole set. Full implementation is Day 6's project component.

### Health check state machine

```
         ┌────────────────────────────┐
         ▼                            │
    [HEALTHY] ──fail×unhealthy_threshold──► [UNHEALTHY]
         ▲                            │
         └──pass×healthy_threshold────┘
```

During the window between "first failure" and "marked unhealthy", the LB may still send requests to the failing backend — this is the **flap window**. Setting `unhealthy_threshold=2` and `interval=5s` means up to 10 s of failures before removal.

## Worked example

```python
"""
day2_load_balancer.py
Minimal L7-style load balancer simulation in Python.
Supports round-robin and least-connections algorithms.
Includes active health checking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Health(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class Backend:
    host: str
    port: int
    weight: int = 1
    health: Health = Health.HEALTHY
    active_connections: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class Algorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"


class LoadBalancer:
    def __init__(
        self,
        backends: list[Backend],
        algorithm: Algorithm = Algorithm.ROUND_ROBIN,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
    ) -> None:
        self._backends = backends
        self._algorithm = algorithm
        self._rr_index = 0
        self._unhealthy_threshold = unhealthy_threshold
        self._healthy_threshold = healthy_threshold

    @property
    def healthy_backends(self) -> list[Backend]:
        return [b for b in self._backends if b.health == Health.HEALTHY]

    def pick(self) -> Backend | None:
        pool = self.healthy_backends
        if not pool:
            log.error("No healthy backends available")
            return None

        if self._algorithm == Algorithm.ROUND_ROBIN:
            backend = pool[self._rr_index % len(pool)]
            self._rr_index += 1
            return backend

        if self._algorithm == Algorithm.LEAST_CONNECTIONS:
            return min(pool, key=lambda b: b.active_connections)

        if self._algorithm == Algorithm.WEIGHTED_ROUND_ROBIN:
            # Expand pool by weight, then round-robin
            weighted: list[Backend] = []
            for b in pool:
                weighted.extend([b] * b.weight)
            backend = weighted[self._rr_index % len(weighted)]
            self._rr_index += 1
            return backend

        return pool[0]

    async def handle_request(self, request_id: str) -> str:
        backend = self.pick()
        if backend is None:
            return f"[{request_id}] ERROR: no backends"
        backend.active_connections += 1
        try:
            # Simulate backend processing
            await asyncio.sleep(0.005)
            result = f"[{request_id}] → {backend.address}"
            log.debug(result)
            return result
        finally:
            backend.active_connections -= 1

    async def health_check_loop(
        self,
        interval: float = 1.0,
        rounds: int = 5,
    ) -> None:
        """Simulates active health checking for `rounds` iterations."""
        for _ in range(rounds):
            await asyncio.sleep(interval)
            for backend in self._backends:
                ok = await self._probe(backend)
                if ok:
                    backend.consecutive_failures = 0
                    backend.consecutive_successes += 1
                    if backend.consecutive_successes >= self._healthy_threshold:
                        if backend.health != Health.HEALTHY:
                            log.info("Backend %s recovered → HEALTHY", backend.address)
                        backend.health = Health.HEALTHY
                else:
                    backend.consecutive_successes = 0
                    backend.consecutive_failures += 1
                    if backend.consecutive_failures >= self._unhealthy_threshold:
                        if backend.health != Health.UNHEALTHY:
                            log.warning(
                                "Backend %s failed %d times → UNHEALTHY",
                                backend.address,
                                backend.consecutive_failures,
                            )
                        backend.health = Health.UNHEALTHY

    async def _probe(self, backend: Backend) -> bool:
        """Mock health probe — override in production with real HTTP GET /healthz."""
        # Simulate port 8002 being down after t=2s
        if backend.port == 8002 and time.monotonic() % 10 > 2:
            return False
        return True


async def main() -> None:
    backends = [
        Backend(host="127.0.0.1", port=8001),
        Backend(host="127.0.0.1", port=8002),
        Backend(host="127.0.0.1", port=8003),
    ]
    lb = LoadBalancer(backends, Algorithm.LEAST_CONNECTIONS)

    health_task = asyncio.create_task(lb.health_check_loop(interval=0.5, rounds=10))

    for i in range(20):
        await asyncio.sleep(0.1)
        result = await lb.handle_request(f"req-{i:02d}")
        log.info(result)

    await health_task
    log.info(
        "Final health: %s",
        {b.address: b.health.value for b in backends},
    )


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. **Algorithm comparison**: modify `LoadBalancer` to add weighted least-connections (weight `active_connections / weight`). Write a test that verifies a backend with weight=2 receives approximately twice the connections of a weight=1 backend under the same load.

2. **Path-based routing (L7)**: extend `LoadBalancer.pick()` to accept an optional `path: str` parameter. Backends are tagged with a `path_prefix: str | None`. Requests matching a prefix route to the matching backend; all others use round-robin. Write tests.

3. **Health check timing**: with `unhealthy_threshold=3`, `healthy_threshold=2`, `interval=5s`: (a) how long can a failure go undetected? (b) how long until a recovered backend re-enters rotation? Show your arithmetic.

4. **NGINX upstream config**: write an NGINX `upstream` block (in a comment, not runnable) that configures three backends with least_conn, a passive health check (`max_fails=3 fail_timeout=30s`), and one backend as backup.

(Solutions: ../solutions/week-02/day-2.md)

## Self-check
- [ ] I can explain L4 vs L7 trade-offs and give a use case for each
- [ ] I can implement round-robin and least-connections algorithms
- [ ] I can describe a health check state machine including thresholds
- [ ] I know when consistent hashing is preferable to modulo hashing

## Supplementary links
- [HAProxy docs: load balancing algorithms](https://www.haproxy.com/documentation/haproxy-configuration-manual/latest/#4-balance)
- [NGINX upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)
- [AWS: ALB vs NLB](https://aws.amazon.com/elasticloadbalancing/features/)
- [Envoy load balancing policies](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers)
