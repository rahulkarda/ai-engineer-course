# Solutions — Week 05 · Day 6 (Saturday Deep Practice Exercises)

---

## Exercise 1 — Race condition in non-atomic GET+INCR

**The race condition (without Lua)**:

```
Time    Pod A                           Pod B
t0      GET rl:user1:tokens → 1.0      GET rl:user1:tokens → 1.0
t1                                     (both see 1.0 token — still allowed)
t2      tokens = 1.0 - 1.0 = 0.0      tokens = 1.0 - 1.0 = 0.0
t3      SET rl:user1:tokens 0.0        SET rl:user1:tokens 0.0
```

**Result**: two requests both allowed when only one token was available. The user effectively gets double their quota. This is the classic **check-then-act** race condition (TOCTOU: Time Of Check To Time Of Use).

**What the Lua script prevents**: in Redis, a Lua script executes atomically — the Redis server runs the entire script without interleaving any other client commands. Between the HMGET (read tokens) and the HMSET (write new token count), no other client can modify the key. The two-phase check-and-update is guaranteed to see a consistent state.

**At the Redis command level**:
```
Without Lua: HMGET → [gap] → HMSET  ← another client's HMSET can land in the gap
With Lua:    [HMGET + HMSET in one script] ← indivisible
```

---

## Exercise 2 — Persisting trie state to disk

**Approaches**:

**Option A — Pickle (avoid)**
```python
import pickle
with open("trie.pkl", "wb") as f:
    pickle.dump(trie, f)
```
Fast to implement, but `pickle.load()` can execute arbitrary code on deserialization — never use for data from untrusted sources. Security risk if the snapshot file could be tampered with.

**Option B — JSON frequency list (recommended)**
Serialize the trie as a list of `(word, frequency)` pairs:
```python
import json
from pathlib import Path

def save_trie(trie: Trie, path: Path) -> None:
    data = []
    def _collect(node, prefix):
        if node.frequency > 0:
            data.append([prefix, node.frequency])
        for ch, child in node.children.items():
            _collect(child, prefix + ch)
    _collect(trie.root, "")
    path.write_text(json.dumps(data))

def load_trie(path: Path) -> Trie:
    t = Trie()
    for word, freq in json.loads(path.read_text()):
        t.insert(word, freq)
    return t
```
Human-readable, no code execution risk, compatible with offline frequency data pipelines.

**Option C — Protocol Buffers / MessagePack**
For large tries (millions of words), JSON is slow. MessagePack gives 2–5× faster serialization with compact binary output. The schema is just `repeated (string word, int32 freq)`.

**Recommendation**: JSON for small tries (< 100 K words), MessagePack or Parquet for production scale.

---

## Exercise 3 — Three-region typeahead consistency

**Acceptable consistency model**: eventual consistency with ~1–5 minute lag.

**Reasoning**: a user typing "new yor" in Tokyo vs. San Francisco will get slightly different suggestions if data hasn't replicated yet. This is not a safety-critical difference — a slightly stale suggestion list is acceptable.

**Sync strategy**:

1. **Read path**: each region serves from its local trie (in-memory, loaded from region-local snapshot). Sub-ms latency, fully independent of other regions.

2. **Write path**: `record_query("new york pizza")` events go to a global Kafka cluster with multi-region replication (Confluent Cloud, AWS MSK, or Kafka MirrorMaker 2).

3. **Trie rebuild**: each region runs an hourly batch job:
   - Reads last hour of query events from its local Kafka partition.
   - Merges frequency deltas into the snapshot file.
   - Reloads the trie into the running service (hot swap via a blue/green pointer).

4. **Conflict model**: last-write-wins on frequency updates. Since we're summing frequencies (not setting them), commutativity means the order of merge doesn't matter — final frequency converges regardless of region merge order.

---

## Exercise 4 — Per-endpoint rate limiting

**Config model change**:

```python
# src/rate_limiter/config.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class EndpointLimit(BaseModel):
    capacity: float
    rate: float   # tokens/s


class RateLimiterSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    default_capacity: float = 100.0
    default_rate: float = 1.67

    # Per-endpoint overrides
    endpoint_limits: dict[str, EndpointLimit] = {
        "/api/v1/search": EndpointLimit(capacity=50.0, rate=0.83),
        "/api/v1/upload": EndpointLimit(capacity=5.0, rate=0.02),
    }

    model_config = {"env_prefix": "RL_"}
```

**Key change in `allow()`**: include the endpoint path in the Redis key.

```python
def allow(self, user_key: str, endpoint: str, cost: float = 1.0) -> bool:
    # Composite key: rate limit is per (user, endpoint)
    redis_key = f"{user_key}:{endpoint.replace('/', '_')}"
    # Select config for this endpoint
    limit_config = self._settings.endpoint_limits.get(
        endpoint,
        EndpointLimit(
            capacity=self._settings.default_capacity,
            rate=self._settings.default_rate,
        ),
    )
    # ... rest of token bucket logic with limit_config
```

**Middleware change**:
```python
async def dispatch(self, request: Request, call_next):
    key = request.headers.get("X-User-Id") or request.client.host
    endpoint = request.url.path
    if not self._limiter.allow(key, endpoint):
        return Response(..., status_code=429)
    return await call_next(request)
```
