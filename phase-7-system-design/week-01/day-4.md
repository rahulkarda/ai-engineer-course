# Phase 7 · Week 01 · Day 4 — Latency Numbers + SLAs / SLOs

> Objective: Internalise key latency numbers and understand how SLAs and SLOs shape architectural decisions.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

Two engineers propose different designs. One prefers a remote RPC between services; the other argues for co-location. Neither can win the argument without latency numbers. The numbers are the language of system design.

SLAs (Service Level Agreements) and SLOs (Service Level Objectives) translate business requirements into measurable engineering targets. They determine everything: replication strategy, caching depth, circuit-breaker thresholds.

---

## Latency Numbers Every Engineer Should Know

Jeff Dean's table, updated for modern hardware:

| Operation | Latency | Notes |
|---|---|---|
| L1 cache reference | 0.5 ns | |
| Branch misprediction | 5 ns | |
| L2 cache reference | 7 ns | 14× L1 |
| Mutex lock/unlock | 25 ns | |
| Main memory reference | 100 ns | 200× L1 |
| Compress 1 KB (Snappy) | 3 μs | |
| Send 1 KB over 1 Gbps network | 10 μs | |
| Read 4 KB from SSD | 150 μs | |
| Read 1 MB sequentially from memory | 250 μs | |
| Round-trip within same data centre | 500 μs | |
| Read 1 MB sequentially from SSD | 1 ms | 4× memory |
| Disk seek (HDD) | 10 ms | 20× SSD |
| Read 1 MB sequentially from HDD | 30 ms | |
| Send packet US → Europe → US | 150 ms | |
| DNS lookup (cold) | 20–120 ms | Varies |
| TLS handshake | 10–20 ms | Per new connection |

### Orders of magnitude to remember

```
L1 cache        ~1 ns
L2 cache        ~10 ns
RAM             ~100 ns
NVMe SSD        ~100 μs     (100,000× slower than RAM)
Intra-DC net    ~500 μs
HDD seek        ~10 ms      (100,000× slower than SSD random)
Cross-continent ~150 ms
```

### Practical implications

- A Redis GET over the network (~200 μs) is **500× faster** than a disk seek.
- A TLS handshake (10 ms) costs as much as 20,000 RAM accesses — keep connections alive.
- Cross-continent latency (150 ms) can dominate everything else for a geo-distributed system.
- Disk seeks kill sequential throughput — batch writes, use LSM-trees (Cassandra, RocksDB).

---

## SLA / SLO / SLI — The Definitions

| Term | Meaning | Who owns it |
|---|---|---|
| **SLI** (Service Level Indicator) | The metric being measured | Engineering |
| **SLO** (Service Level Objective) | The target value for an SLI | Engineering / Product |
| **SLA** (Service Level Agreement) | The contractual commitment to customers | Business / Legal |

The chain: SLA sets the outer bound → engineering sets internal SLOs tighter → SLIs measure reality.

```
SLA: "99.9% availability per month"
  └─► SLO: "99.95% availability per month" (tighter; gives headroom)
        └─► SLI: p(request succeeded) measured over rolling 30 days
```

If the SLO is breached, the SLA may be breached. Violating an SLA has financial consequences (credits, contract penalties).

### Error budget

```
Error budget = 1 - SLO

SLO 99.9% → error budget = 0.1% of the month
           = 0.001 × 30 × 24 × 60 min
           = 43.2 minutes/month
```

The error budget governs how often you can deploy, run experiments, and tolerate incidents. When the budget is spent, deployments stop until the budget refills.

---

## Latency SLOs

Availability is easy to measure. Latency SLOs require percentiles — averages hide tail latency.

| Percentile | Meaning |
|---|---|
| p50 | Median — half of requests are faster |
| p95 | 5% of requests are slower than this |
| p99 | 1% of requests are slower |
| p99.9 | 1 in 1,000 requests is slower |

### Why p99 matters more than average

Suppose your API serves 1,000 requests/sec. At p99 = 200 ms, **10 requests/sec** are experiencing >200 ms. Those are real users. Optimising the median while ignoring the tail means those users always suffer.

### Cascading latency

When service A calls B calls C, latency compounds:

```
p99 latency = p99(A) + p99(B) + p99(C)

If each is 100 ms at p99:
  p99 of the chain ≈ 300 ms

Worse: if each is independent, the p99 of the chain is actually ~p(1 - 0.01^3) 
at the tail — approximately 3× the individual p99 for short chains.
```

This is why microservices architectures with many hops struggle to meet tight latency SLOs.

---

## How Latency Numbers Drive Architecture

### Cache placement

```
Target: p99 < 10 ms

DB query (SSD-backed Postgres): ~5 ms average but p99 can spike to 50-200 ms under load
Redis GET (network): ~200 μs consistently

Decision: cache hot data in Redis; only miss to DB
```

### Connection pooling

```
TLS handshake: 10-20 ms
Connection pool reuse: ~0 ms overhead

Decision: always pool; never create a new connection per request at scale
```

### Async vs synchronous

```
Synchronous call to slow third-party API (p99 = 2 s):
  All threads in thread pool block for 2 s
  At 100 concurrent requests → thread exhaustion → cascade failure

Async with queue:
  Request enqueued instantly (< 1 ms)
  Worker processes when ready
  No thread exhaustion

Decision: anything > 100 ms that is not user-facing critical path → async
```

---

## Worked Example

**Problem:** Design an SLO strategy for a payments API.

```
Business SLA: 99.99% availability, p99 < 500 ms

Engineering SLOs (tighter):
  Availability SLO: 99.995%
  Latency SLO:      p99 < 300 ms (leaves budget for downstream variance)
  Error rate SLO:   < 0.1% 5xx errors

Error budget:
  99.995% → 0.005% downtime per month
  = 0.00005 × 43,200 min ≈ 2.16 minutes/month

Consequence: extremely limited deployment window. Use:
  - Blue-green deploys
  - Feature flags
  - Automated rollback on SLO breach

Latency breakdown budget (300 ms p99):
  TLS + auth middleware:    20 ms
  Validation + DB read:     50 ms
  Business logic:           30 ms
  DB write (synchronous):   80 ms
  Response serialisation:   10 ms
  Network (intra-DC):        5 ms
                          ─────────
  Total budget:            195 ms  (105 ms headroom to p99)

Decision: DB writes must be SSD-backed, replicated synchronously. 
          Async replication would violate the "no data loss" requirement for payments.
```

---

## Exercises

1. An application has three microservices in its critical path, each with p99 latency of 50 ms. Estimate the p99 latency of the full request. If the SLO is p99 < 100 ms, what must each service's p99 target be?
2. Calculate the error budget (in minutes/month) for SLOs of 99%, 99.9%, 99.99%, and 99.999%.
3. You have a choice between a Redis cache (p99 = 0.5 ms) and a co-located in-memory cache (p99 = 0.05 ms). The system processes 500 k requests/sec and needs p99 < 5 ms. Which do you choose and why? What is the latency budget saved?
4. A disk seek takes ~10 ms. Explain why a write-heavy database application on spinning disks can handle at most ~100 random writes/sec, and what architectural change resolves this.

(Solutions: ../solutions/week-01/day-4.md)

---

## Self-check

- [ ] I can recall 5+ key latency numbers from memory.
- [ ] I can explain the difference between SLA, SLO, and SLI with examples.
- [ ] I can calculate an error budget from an SLO.
- [ ] I can explain why tail latency (p99) matters more than average.
- [ ] I can derive SLO targets for individual services given an end-to-end SLO.

---

## Supplementary

- [Google SRE Book — Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)
- [Latency Numbers Every Programmer Should Know — Interactive](https://colin-scott.github.io/personal_website/research/interactive_latency.html)
- [High Scalability — Latency is Everywhere](http://highscalability.com/latency-everywhere-and-it-costs-you-sales-how-crush-it)
