# Solutions — Day 2: Load Balancing

## Exercise 1 — Weighted least-connections

```python
def pick_weighted_least_conn(pool: list[Backend]) -> Backend:
    """
    Weighted least-connections: route to backend with lowest
    (active_connections / weight) ratio.
    """
    return min(pool, key=lambda b: b.active_connections / b.weight)


# Test
import pytest
from dataclasses import dataclass, field

@pytest.mark.asyncio
async def test_weighted_least_conn_distribution():
    from day2_load_balancer import Backend, LoadBalancer, Algorithm, Health
    import asyncio

    # weight=2 should receive ~2× the connections of weight=1
    backends = [
        Backend("127.0.0.1", 8001, weight=1),
        Backend("127.0.0.1", 8002, weight=2),
    ]

    # Patch Algorithm to use weighted_least_conn
    lb = LoadBalancer(backends, Algorithm.WEIGHTED_ROUND_ROBIN)

    request_counts = {b.address: 0 for b in backends}
    for i in range(300):
        result = await lb.handle_request(f"req-{i}")
        addr = result.split("→ ")[1]
        request_counts[addr] += 1

    ratio = request_counts["127.0.0.1:8002"] / request_counts["127.0.0.1:8001"]
    assert 1.7 <= ratio <= 2.3, f"Expected ~2x, got {ratio:.2f}"
```

---

## Exercise 2 — Path-based routing

```python
@dataclass
class Backend:
    host: str
    port: int
    weight: int = 1
    health: Health = Health.HEALTHY
    active_connections: int = 0
    path_prefix: str | None = None   # new field

class LoadBalancerWithRouting(LoadBalancer):
    def pick(self, path: str | None = None) -> Backend | None:
        pool = self.healthy_backends
        if not pool:
            return None
        if path:
            # Find backends matching the path prefix
            matching = [b for b in pool if b.path_prefix and path.startswith(b.path_prefix)]
            if matching:
                return matching[0]
        # Fall back to round-robin
        backend = pool[self._rr_index % len(pool)]
        self._rr_index += 1
        return backend


def test_path_routing():
    api_backend = Backend("127.0.0.1", 8001, path_prefix="/api")
    static_backend = Backend("127.0.0.1", 8002, path_prefix="/static")
    default_backend = Backend("127.0.0.1", 8003)

    lb = LoadBalancerWithRouting([api_backend, static_backend, default_backend])
    assert lb.pick("/api/users").port == 8001
    assert lb.pick("/static/app.js").port == 8002
    assert lb.pick("/healthz").port == 8003  # round-robin
```

---

## Exercise 3 — Health check timing

- `unhealthy_threshold=3`, `interval=5s`:
  - **Max undetected failure duration**: probe fires every 5s; needs 3 consecutive failures → `3 × 5 = 15s` maximum (first failure could arrive just after a successful probe, so worst case is 14.999s to first miss plus 10s for the next two → up to 15s).
- `healthy_threshold=2`, `interval=5s`:
  - **Time to re-enter rotation after recovery**: 2 × 5 = 10s.

---

## Exercise 4 — NGINX upstream config

```nginx
# nginx.conf upstream block (illustrative)
upstream app_servers {
    least_conn;

    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8004 backup;  # only used when all above are down

    keepalive 32;  # keep 32 idle connections per worker
}
```

Notes:
- `max_fails=3`: passive health check — after 3 failures within `fail_timeout`, mark down.
- `fail_timeout=30s`: window for failure counting AND how long the server stays marked down.
- `backup`: standby server, receives traffic only when all primary servers are unavailable.
- `keepalive 32`: enable HTTP keepalive to backends (reduces TCP overhead).
