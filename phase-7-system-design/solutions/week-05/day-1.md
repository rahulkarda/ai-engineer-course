# Solutions — Week 05 · Day 1 (URL Shortener Exercises)

---

## Exercise 1 — MD5 hash failure modes

**Failure mode 1 — Hash collision**
Two distinct long URLs produce the same MD5 prefix. Example:
`MD5("https://a.com")[0:7]` and `MD5("https://b.com")[0:7]` both equal `"aB3kQ9x"`.
When the second URL is shortened, it overwrites the first row, silently breaking the first short URL.

**Fix**: on write, check if `code` exists with a different `long_url`. If yes, append a salt and rehash:
```python
code = md5(long_url)[0:7]
while db_exists(code) and db_get(code).long_url != long_url:
    long_url += "_"   # salt
    code = md5(long_url)[0:7]
```
This still has worst-case unbounded retries. Better: use KGS or Snowflake IDs (no collision by design).

**Failure mode 2 — Enumeration**
`MD5("https://example.com")[0:7]` = `"2f45a9c"`. An attacker cannot easily predict codes from MD5, but if they can observe that codes are deterministic hashes of URLs, they can pre-compute rainbow tables for common URLs (e.g. all government or financial URLs). This allows discovering private short URLs by knowing the original.

**Fix**: add a secret salt to the hash: `MD5(long_url + secret_salt)[0:7]`. Or use random codes (KGS).

---

## Exercise 2 — Cache invalidation race condition

**Flow**:
1. User calls `DELETE /{code}`.
2. Service deletes row from Postgres.
3. Service calls `DEL code:aB3kQ9x` on Redis.

**Race condition**:
Between step 2 and step 3, another request for `GET /aB3kQ9x` may:
- Miss the Redis cache (or hit a different Redis node on a partitioned cluster).
- Read from Postgres replica — but the delete has not replicated yet → finds the row → serves the redirect.
- A different request may also re-populate Redis from Postgres before step 3 fires → the deleted URL is back in cache.

**Prevention**:
- Write-through invalidation: delete from cache first, then from DB (but now a cache miss between delete and DB delete briefly serves the old URL from DB — acceptable for expiry-based deletes).
- Better: set a "tombstone" key in Redis (`SETEX deleted:aB3kQ9x 60 1`) and check it on every read before serving. Any read within 60 s gets a 404.
- For hard deletes: accept brief inconsistency (redirect for up to TTL seconds after delete) — common in practice.

---

## Exercise 3 — Viral traffic spike (500K req/s for one code)

**What happens**:
- CDN edge cache has a `302` cached → no origin hit. Most requests served from edge at ~5 ms.
- But if CDN is not caching 302 (by default CDNs don't cache redirects without explicit `Cache-Control`), all 500 K requests hit the Redirect Service.
- Redis serves from in-memory at ~0.1 ms → no bottleneck at 500 K req/s (Redis Cluster handles millions of ops/s).
- No Postgres hit needed if Redis is warm.

**Potential issue**: CDN misconfiguration or Redis cache cold-start (e.g. Redis restart). In that case, 500 K req/s all hit Postgres → each row lookup → read replica overwhelmed.

**Changes needed**:
1. Ensure CDN `Cache-Control: public, max-age=300` on 302 responses for non-tracking use cases.
2. For 302 (tracking mode): add Redis Cluster with read replicas; configure request coalescing — if 100 requests arrive simultaneously for a cold key, only one goes to Postgres and the rest wait.
3. Configure probabilistic cache refresh: when TTL < 10% remaining, one request in 100 triggers an async refresh, avoiding cache stampede.

---

## Exercise 4 — Stats endpoint design

**Aggregations to pre-compute** (real-time counter updates are expensive):
- `click_count` (approximate): maintain in Redis `INCR`, batch-flush to Postgres every 30 s.
- `click_count_by_country` (top 10 countries): Redis hash `HINCRBY stats:{code}:country {cc} 1`.
- `unique_clickers` (HyperLogLog): `PFADD uniq:{code} {ip_hash}` → `PFCOUNT uniq:{code}`.

**Avoid the `clicks` table being a bottleneck**:
- Do not query the `clicks` table on every `GET /stats` call — it can have billions of rows.
- Instead, answer stats from Redis counters (for last 24 h) + pre-aggregated daily rollup table in ClickHouse (for historical).
- Schema: `click_rollups(code, date, country, click_count)` — updated by a nightly batch job.

---

## Exercise 5 — encode_base62 / decode_base62 tests

```python
# tests/unit/test_base62.py
import pytest
from src.url_shortener.key_service import encode_base62, decode_base62

@pytest.mark.parametrize("n,expected_len", [
    (0, 1),
    (1, 1),
    (61, 1),
    (62, 2),
    (3_521_614_606_207, 7),  # 62^7 - 1
])
def test_encode_length(n: int, expected_len: int) -> None:
    code = encode_base62(n)
    assert len(code) == expected_len

@pytest.mark.parametrize("n", [0, 1, 61, 62, 100_000, 3_521_614_606_207])
def test_round_trip(n: int) -> None:
    assert decode_base62(encode_base62(n)) == n

def test_base62_charset() -> None:
    import string
    charset = string.ascii_letters + string.digits
    for char in encode_base62(999_999):
        assert char in charset
```

**Note on `encode_base62(0)`**: the modulo-loop approach exits immediately with empty `result`. The guard `if n == 0: return BASE62[0]` is required. Test this explicitly.
