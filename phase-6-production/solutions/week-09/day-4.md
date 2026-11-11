# Solutions — Day 4: Caching + Rate Limiting

---

## Exercise 1: LRU eviction in `ExactCache` using `OrderedDict`

When the cache exceeds `max_size`, evict the oldest entry (least-recently-used order maintained by `OrderedDict.move_to_end`).

```python
# gateway/cache/exact_cache.py
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

from gateway.providers.base import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


def _cache_key(request: CompletionRequest) -> str:
    """Deterministic key from model + messages (ignores temperature for exact match)."""
    payload = {"model": request.model, "messages": request.messages}
    serialised = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class ExactCache:
    """
    In-memory exact-match cache with LRU eviction.

    OrderedDict preserves insertion order; move_to_end(key) promotes a key
    to 'most recently used'. popitem(last=False) removes the oldest (LRU).
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._store: OrderedDict[str, CompletionResponse] = OrderedDict()
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Exercise 1 — LRU eviction
    # ------------------------------------------------------------------
    def set(self, request: CompletionRequest, response: CompletionResponse) -> None:
        key = _cache_key(request)
        if key in self._store:
            # Update existing entry and promote to MRU position
            self._store.move_to_end(key)
            self._store[key] = response
            return

        if len(self._store) >= self._max_size:
            evicted_key, _ = self._store.popitem(last=False)  # remove LRU
            self._stats.evictions += 1
            logger.debug("cache evicted key=%s…", evicted_key[:12])

        self._store[key] = response
        logger.debug("cache set key=%s…", key[:12])

    def get(self, request: CompletionRequest) -> CompletionResponse | None:
        key = _cache_key(request)
        if key not in self._store:
            self._stats.misses += 1
            return None
        # Promote to MRU
        self._store.move_to_end(key)
        self._stats.hits += 1
        return self._store[key]

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Exercise 2 — CacheStats + get_stats()
    # ------------------------------------------------------------------
    def get_stats(self) -> CacheStats:
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
        )

    def reset_stats(self) -> None:
        self._stats = CacheStats()
```

---

## Exercise 2: `CacheStats` dataclass + `get_stats()`

Already embedded above. Standalone definition for clarity:

```python
# gateway/cache/stats.py
from dataclasses import dataclass


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits (0.0–1.0)."""
        return self.hits / self.total if self.total else 0.0
```

`get_stats()` returns a **copy** of the internal `CacheStats` so the caller cannot mutate the cache's internal counters.

---

## Exercise 3: `RateLimiter.wait_time()` — seconds until next token available

Returns `0.0` if a request is allowed right now; otherwise returns the number of seconds the caller must wait before the next token refills.

```python
# gateway/ratelimit/limiter.py
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter using a sliding window (deque of timestamps).

    rpm  — requests per minute (window = 60 s)
    """

    def __init__(self, rpm: int) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self._rpm = rpm
        self._window_s: float = 60.0
        # Per-tenant deque of request timestamps (monotonic)
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, tenant_id: str, now: float) -> None:
        """Remove timestamps outside the current window."""
        window = self._windows[tenant_id]
        cutoff = now - self._window_s
        while window and window[0] <= cutoff:
            window.popleft()

    def is_allowed(self, tenant_id: str) -> bool:
        now = time.monotonic()
        self._prune(tenant_id, now)
        if len(self._windows[tenant_id]) < self._rpm:
            self._windows[tenant_id].append(now)
            logger.debug("rate_limiter tenant=%s allowed count=%d", tenant_id, len(self._windows[tenant_id]))
            return True
        logger.debug("rate_limiter tenant=%s denied rpm=%d", tenant_id, self._rpm)
        return False

    # ------------------------------------------------------------------
    # Exercise 3 — wait_time()
    # ------------------------------------------------------------------
    def wait_time(self, tenant_id: str) -> float:
        """
        Return 0.0 if the tenant can send a request right now.
        Otherwise return the number of seconds until the oldest request
        in the window expires and a new token becomes available.
        """
        now = time.monotonic()
        self._prune(tenant_id, now)
        window = self._windows[tenant_id]

        if len(window) < self._rpm:
            return 0.0

        # Oldest timestamp in the window; it expires at oldest + window_s
        oldest = window[0]
        wait = (oldest + self._window_s) - now
        return max(0.0, wait)
```

Usage:

```python
limiter = RateLimiter(rpm=60)
secs = limiter.wait_time("tenant-abc")
if secs > 0:
    await asyncio.sleep(secs)
allowed = limiter.is_allowed("tenant-abc")
```

---

## Exercise 4: pytest — `RateLimiter(rpm=2)` — first two allowed, third denied

```python
# tests/unit/test_rate_limiter.py
import time
import pytest

from gateway.ratelimit.limiter import RateLimiter


def test_rate_limiter_rpm_2_third_call_denied() -> None:
    """RateLimiter(rpm=2): first two is_allowed() → True, third → False."""
    limiter = RateLimiter(rpm=2)
    tenant = "test-tenant"

    result_1 = limiter.is_allowed(tenant)
    result_2 = limiter.is_allowed(tenant)
    result_3 = limiter.is_allowed(tenant)

    assert result_1 is True
    assert result_2 is True
    assert result_3 is False


def test_wait_time_zero_when_under_limit() -> None:
    limiter = RateLimiter(rpm=10)
    # No requests yet — should be allowed immediately
    assert limiter.wait_time("tenant-a") == 0.0


def test_wait_time_positive_when_at_limit() -> None:
    limiter = RateLimiter(rpm=2)
    tenant = "tenant-b"

    limiter.is_allowed(tenant)
    limiter.is_allowed(tenant)

    # Now at limit — wait_time should be > 0
    wait = limiter.wait_time(tenant)
    assert wait > 0.0
    assert wait <= 60.0


def test_wait_time_zero_after_window_expires() -> None:
    """After window_s seconds, the bucket should refill and wait_time → 0."""
    limiter = RateLimiter(rpm=1)
    # Monkey-patch window to 0.05 s for a fast test
    limiter._window_s = 0.05
    tenant = "tenant-c"

    limiter.is_allowed(tenant)  # use the one token
    assert limiter.wait_time(tenant) > 0.0

    time.sleep(0.1)  # let the window expire

    assert limiter.wait_time(tenant) == 0.0


def test_multiple_tenants_independent() -> None:
    limiter = RateLimiter(rpm=1)
    assert limiter.is_allowed("a") is True
    assert limiter.is_allowed("b") is True  # different tenant — own bucket

    # Both buckets full now
    assert limiter.is_allowed("a") is False
    assert limiter.is_allowed("b") is False


# ---- LRU cache eviction tests (Exercise 1 + 2) ----

def test_exact_cache_evicts_lru_when_full() -> None:
    from gateway.cache.exact_cache import ExactCache
    from gateway.providers.base import CompletionRequest, CompletionResponse

    def _req(model: str) -> CompletionRequest:
        return CompletionRequest(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )

    def _resp(model: str) -> CompletionResponse:
        return CompletionResponse(
            content="ok", model=model, provider="anthropic",
            input_tokens=1, output_tokens=1, latency_ms=1.0,
        )

    cache = ExactCache(max_size=2)
    req_a, req_b, req_c = _req("claude-haiku-5-5"), _req("claude-sonnet-5-5"), _req("claude-opus-5-5")

    cache.set(req_a, _resp("claude-haiku-5-5"))
    cache.set(req_b, _resp("claude-sonnet-5-5"))
    assert len(cache) == 2

    # Adding a third entry should evict the LRU (req_a)
    cache.set(req_c, _resp("claude-opus-5-5"))
    assert len(cache) == 2
    assert cache.get(req_a) is None  # evicted
    assert cache.get(req_b) is not None
    assert cache.get(req_c) is not None

    stats = cache.get_stats()
    assert stats.evictions == 1


def test_exact_cache_stats_track_hits_and_misses() -> None:
    from gateway.cache.exact_cache import ExactCache
    from gateway.providers.base import CompletionRequest, CompletionResponse

    def _req(content: str) -> CompletionRequest:
        return CompletionRequest(
            model="claude-haiku-5-5",
            messages=[{"role": "user", "content": content}],
        )

    def _resp() -> CompletionResponse:
        return CompletionResponse(
            content="ok", model="claude-haiku-5-5", provider="anthropic",
            input_tokens=1, output_tokens=1, latency_ms=1.0,
        )

    cache = ExactCache(max_size=100)
    req = _req("hello")

    # Miss
    result = cache.get(req)
    assert result is None

    # Set then hit
    cache.set(req, _resp())
    result = cache.get(req)
    assert result is not None

    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_rate == pytest.approx(0.5)
```
