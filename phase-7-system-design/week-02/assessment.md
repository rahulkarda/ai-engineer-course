# Week 02 Assessment — Scalability: Load Balancing, Caching & CDN

> Time: ~1 hr  ·  Pass gate: 75%  ·  Answer key: ../solutions/week-02/assessment.md

## Conceptual (interview-style) — 25%

Answer each in 2–5 sentences. Be precise — vague answers score partial credit.

1. A service is stateful: it stores user sessions in process memory. You want to scale it from 1 to 5 replicas. What breaks, and what is the minimal change required to make it work?

2. Explain the difference between L4 and L7 load balancing. Give one concrete use case where you would choose L4 over L7.

3. A cache entry expires at 11:00:00.000 exactly. One thousand concurrent requests arrive at 11:00:00.001. Describe exactly what happens without stampede protection, and how a Redis SET NX lock solves it.

4. You have a cache with `allkeys-lru` eviction. Your access pattern is highly skewed (20% of keys account for 80% of reads). After a cold start, which policy — LRU or LFU — will have a better hit rate once the cache is warm? Why?

5. Explain how consistent hashing reduces key remapping during a scale event compared to modulo hashing. Use concrete numbers: 3 backends, 10,000 keys, add a 4th backend.

6. A team proposes write-back caching for their order management system. What risk do you raise and what architectural mitigation would you require before approving?

7. Your CDN serves `app.js` with `Cache-Control: max-age=86400`. You deploy a critical bug fix to `app.js`. Users still see the old file. What are two ways to solve this, and what are their trade-offs?

---

## Coding challenge — 40%

### Challenge A: Stampede-safe cache (required)

Implement `SafeCache` with the following interface:

```python
class SafeCache:
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...
    async def invalidate(self, key: str) -> None: ...
```

Requirements:
- Backed by an in-memory store (no Redis dependency in tests).
- `get` must implement cache-aside: on miss, fetch from a `DataBackend` (injected).
- On concurrent misses for the same key, exactly **one** coroutine queries the backend; others wait and receive the result once populated.
- `invalidate` deletes the cache entry and writes the new value to the backing store if provided.
- Track and expose `hit_rate` as a float.

Acceptance criteria:
```python
# Test: stampede prevention
db = MockDataBackend(delay=0.1)
cache = SafeCache(db)
results = await asyncio.gather(*[cache.get("key") for _ in range(100)])
assert db.call_count == 1       # exactly one DB call
assert all(r == expected for r in results)

# Test: hit rate after warm-up
for _ in range(10):
    await cache.get("key")
assert cache.hit_rate >= 0.9    # at least 9/10 requests are hits after warm-up
```

---

### Challenge B: Consistent hash ring (required)

Implement `HashRing` with:

```python
class HashRing:
    def add_node(self, node_id: str, weight: int = 1) -> None: ...
    def remove_node(self, node_id: str) -> None: ...
    def get_node(self, key: str) -> str | None: ...
```

Requirements:
- Virtual nodes: `vnodes_per_node * weight` positions per node.
- Deterministic: same key always maps to same node.
- On adding a node with N existing nodes and K keys, at most `K / (N+1) × 1.2` keys remap (at most 20% excess over ideal).

Acceptance criteria:
```python
ring = HashRing(vnodes_per_node=150)
for node in ["A", "B", "C"]:
    ring.add_node(node)

# Routing
assert ring.get_node("user:42") is not None  # returns a node

# Distribution: each node handles 20–47% of 10,000 keys
assignments = [ring.get_node(f"key:{i}") for i in range(10000)]
for node in ["A", "B", "C"]:
    pct = assignments.count(node) / 10000
    assert 0.20 <= pct <= 0.47, f"{node}: {pct:.2%}"

# Remapping on scale event
before = {k: ring.get_node(k) for k in [f"key:{i}" for i in range(10000)]}
ring.add_node("D")
after = {k: ring.get_node(k) for k in before}
remapped = sum(1 for k in before if before[k] != after[k])
assert remapped / 10000 <= 0.30  # ≤ 30% remapped
```

---

## Debugging challenge — 20%

The following service is deployed on 3 replicas and causes intermittent 401 errors. Find and fix all bugs. There are at least 3.

```python
import secrets, time
from fastapi import FastAPI, HTTPException, Header

app = FastAPI()
_sessions = {}   # bug 1

@app.post("/login")
def login(user_id: str):
    token = secrets.token_hex(16)
    _sessions[user_id] = token
    return {"token": token}

@app.get("/me")
def me(x_token: str = Header()):
    for uid, tok in _sessions.items():
        if tok == x_token:
            return {"user_id": uid}
    raise HTTPException(401)

@app.delete("/logout")
def logout(x_token: str = Header()):
    _sessions.clear()   # bug 2: clears ALL sessions
    return {"ok": True}

CACHE = {}

@app.get("/products/{pid}")
def get_product(pid: str):
    if pid in CACHE:       # bug 3: in-memory cache, not shared
        return CACHE[pid]
    result = fetch_from_db(pid)
    CACHE[pid] = result    # different replicas cache independently
    return result
```

For each bug: (a) explain exactly why it fails under horizontal scaling, (b) write the fix.

---

## Rubric

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | Challenges A and B pass acceptance tests without referencing course examples |
| Conceptual understanding | 25% | All 7 conceptual questions answered correctly and precisely |
| Debugging & problem-solving | 20% | All 3+ bugs identified with correct explanations and working fixes |
| Code quality & testing | 15% | Type hints, logging, no hardcoded secrets, at least 5 meaningful test assertions |

**Pass gate: ≥ 75% total. Independent implementation is mandatory — 0 points there = automatic fail.**

Answer key: ../solutions/week-02/assessment.md
