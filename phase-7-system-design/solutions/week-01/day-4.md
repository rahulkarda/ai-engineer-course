# Solutions — Day 4: Latency Numbers + SLAs / SLOs

---

## Exercise 1 — Microservice chain latency + SLO derivation

```
Observed p99 per service: 50 ms
Chain p99 ≈ 3 × 50 ms = 150 ms  (additive approximation for independent services)

SLO: p99 < 100 ms
Required per service: 100 ms ÷ 3 ≈ 33 ms each

In practice, set internal SLOs at 30 ms (tighter) to leave headroom.
```

Note: the actual statistical tail is slightly worse than additive because each service's tail events are independent. For production SLO budgeting, use: target = SLO / (n × 1.2) as a conservative heuristic.

---

## Exercise 2 — Error budgets

```
SLO 99%     → 0.01 × 43,200 min = 432 min/month  (7.2 hours)
SLO 99.9%   → 0.001 × 43,200   = 43.2 min/month
SLO 99.99%  → 0.0001 × 43,200  = 4.32 min/month
SLO 99.999% → 0.00001 × 43,200 = 0.432 min/month (26 seconds)
```

Lesson: each additional 9 reduces the budget by 10×. Going from 99.9% to 99.99% is much harder than going from 99% to 99.9%.

---

## Exercise 3 — Redis vs in-process cache

```
Redis:        p99 = 0.5 ms
In-process:   p99 = 0.05 ms
SLO target:   p99 < 5 ms

Both options satisfy the SLO. The question is operational:

In-process cache:
  - No network hop (0.05 ms)
  - Each of N API server instances has its own cache — inconsistency risk
  - Cache is lost on process restart (cold start problem)
  - Memory competes with application heap

Redis:
  - Single shared cache — consistent across all instances
  - Survives process restarts
  - 0.5 ms latency comfortably within the 5 ms SLO
  - Adds operational dependency

Recommendation at 500 k requests/sec: Redis.
  0.5 ms per request × 500,000 = 250,000 ms of latency "spent" per second
  vs. 0.05 ms × 500,000 = 25,000 ms
  Difference: 225,000 ms/sec = 225 CPU-seconds/sec of extra wait
  At scale, the Redis hop adds ~4.5 ms to average request time across all requests
  — but still within SLO. Consistency benefit outweighs the latency cost.
```

---

## Exercise 4 — HDD random writes

```
HDD seek time: 10 ms per random seek
Max random writes/sec = 1,000 ms / 10 ms = 100 random writes/sec

This is why write-heavy applications on HDDs hit a hard ceiling.
Each write requires:
  1. Seek to the target track (10 ms)
  2. Rotational latency to the target sector (~5 ms average at 7200 RPM)
  3. Write (1–2 ms)
Total: ~15–17 ms per random write → ~60 random writes/sec realistic

Architectural fix:
  1. LSM-tree storage engine (RocksDB, Cassandra, ClickHouse):
     Buffers writes in memory (memtable), flushes as sequential
     append-only files (SSTs). Sequential write speed is ~100–200 MB/s
     on HDD vs. 1 MB/s for random writes.
  2. Write-ahead log (WAL) + batch commit:
     Group commits reduce seek count — one seek per batch of 1,000 writes.
  3. Use NVMe SSD: random write latency ~100 μs → 10,000 random writes/sec.
```
