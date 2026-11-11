# Phase 7 · Week 01 · Day 3 — Back-of-Envelope Estimation

> Objective: Perform fast, credible capacity estimates for QPS, storage, bandwidth, and memory.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

Back-of-envelope (BOE) estimation is the engineering equivalent of a Fermi problem. You don't need exact numbers — you need **order-of-magnitude accuracy** so you can make technology choices that won't fail in production.

If you pick a single-node Postgres when the workload is 500 k writes/sec, you've failed the design. BOE prevents that.

The entire estimation workflow:

```
1. Anchor on users / daily active users (DAU)
2. Derive QPS (queries per second)
3. Estimate storage
4. Estimate bandwidth
5. Estimate memory (for caches)
6. Draw conclusions (single server? sharding? CDN?)
```

---

## Numbers to Memorise

### Power of 2

| Power | Value | Human name |
|---|---|---|
| 2^10 | 1,024 | ~1 K |
| 2^20 | 1,048,576 | ~1 M |
| 2^30 | 1,073,741,824 | ~1 G |
| 2^40 | ~1 T | |

### Latency cheat sheet (preview — full table Day 4)

| Operation | Time |
|---|---|
| L1 cache reference | 0.5 ns |
| Main memory reference | 100 ns |
| SSD random read | 100 μs |
| Network round-trip (same DC) | 500 μs |
| Disk seek | 10 ms |

### Data sizes

| Unit | Bytes |
|---|---|
| ASCII char | 1 B |
| UUID | 16 B |
| 64-bit integer | 8 B |
| Typical row (relational) | 100–500 B |
| Small image (thumbnail) | 50 KB |
| Large image | 1–5 MB |
| Video (1 min, 720p) | ~100 MB |

---

## QPS Estimation

### Formula

```
QPS = (DAU × actions_per_day) / 86,400
```

86,400 seconds in a day. Use 100,000 as a round approximation.

### Peak QPS

Systems don't see uniform load. Assume peak = **2–5× average** for most consumer apps (10× for events like Black Friday).

### Example: URL shortener

```
DAU: 10 M users
Each user creates 1 short URL and clicks 10 links per day

Write QPS = 10M × 1 / 100,000 = 100 writes/sec
Read  QPS = 10M × 10 / 100,000 = 1,000 reads/sec
Peak read = 1,000 × 3 = ~3,000 reads/sec
```

Conclusion: reads dominate (10:1 ratio). Design must optimise for reads.

---

## Storage Estimation

### Formula

```
Storage = QPS × record_size × seconds_in_period
```

Or per-day:

```
Daily storage = daily_writes × record_size
Annual storage = daily_storage × 365
```

### Example: URL shortener

```
Record: short_code (7 bytes) + long_url (avg 100 bytes) + created_at (8 bytes) + user_id (8 bytes)
      ≈ 123 bytes ≈ 150 bytes (round up for overhead)

Daily writes: 100/sec × 86,400 = 8.64 M records/day ≈ 10 M/day
Daily storage: 10 M × 150 B = 1.5 GB/day
Annual storage: 1.5 × 365 ≈ 550 GB/year
5-year storage: ~2.75 TB

Conclusion: fits on a single large SSD-backed server; sharding not required for storage alone.
```

---

## Bandwidth Estimation

### Formula

```
Inbound  bandwidth = write QPS × avg_payload_size
Outbound bandwidth = read  QPS × avg_response_size
```

### Example: URL shortener

```
Inbound:  100 writes/sec × 500 B (URL + metadata) = 50 KB/s  — trivial
Outbound: 1,000 reads/sec × 100 B (redirect response) = 100 KB/s  — trivial

Conclusion: bandwidth is not a bottleneck for this system.
```

### Example: Video streaming platform

```
Read QPS: 100,000 concurrent streams × 5 Mbps (720p bitrate)
Outbound: 100,000 × 5 Mbps = 500 Gbps

Conclusion: this is why CDNs exist. A single data centre cannot deliver 500 Gbps.
```

---

## Memory Estimation (Caching)

Caching rule of thumb: cache the **20% of data that handles 80% of requests** (Pareto).

```
Memory needed = hot_fraction × dataset_size
```

### Example: URL shortener

```
Total dataset (5 years): 2.75 TB
Hot URLs (20%): 0.2 × 2.75 TB = 550 GB

A Redis cluster with 550 GB RAM can serve virtually all read traffic from cache.
Alternatively: cache only last-7-days URLs. 
  7 days × 1.5 GB/day = 10.5 GB — fits in a single Redis node.
```

---

## Worked Example End-to-End

**Problem:** "Design Twitter — estimate capacity."

```
Assumptions:
  DAU: 300 M
  Each user reads 100 tweets/day, writes 1 tweet/day (includes retweets)
  10% of users are heavy tweeters (10×)

Write QPS:
  300M × 1 / 100,000 = 3,000 writes/sec
  Peak:  3,000 × 3 = 9,000 writes/sec

Read QPS:
  300M × 100 / 100,000 = 300,000 reads/sec
  Peak: 300,000 × 3 = 900,000 reads/sec

Storage per tweet:
  tweet_id (8B) + user_id (8B) + text (280B) + timestamp (8B) + metadata (50B)
  ≈ 354 B ≈ 400 B

Daily write storage:
  3,000 × 86,400 × 400 B ≈ 100 GB/day

Annual: ~36 TB/year (text only; media is orders of magnitude more)

Media (assume 20% of tweets have a 1 MB image):
  3,000 × 0.2 × 86,400 × 1 MB ≈ 51 TB/day  →  CDN and object storage mandatory

Bandwidth:
  Inbound text: 3,000 × 400 B = 1.2 MB/s  — negligible
  Outbound:     900,000 × 400 B = 360 MB/s = ~3 Gbps  — manageable with load balancers

Cache:
  20% hot tweets: 0.2 × 36 TB/year → cache last 7 days
  7 × 100 GB/day = 700 GB RAM cluster
```

**Conclusions:**
- Read-heavy (100:1) → aggressive caching mandatory
- Media storage → object store (S3) + CDN, not relational DB
- 9 k writes/sec → standard DB cluster handles this (Postgres with replication)
- 900 k reads/sec → Redis required; cannot hit DB

---

## How it works internally

Interviewers don't verify your arithmetic. They watch:
1. Do you start from sensible anchors (DAU, actions/user)?
2. Do you state your assumptions out loud?
3. Do you draw conclusions from the numbers?

A wrong number with correct methodology beats a correct number that led nowhere.

---

## Exercises

1. Estimate QPS, storage, and bandwidth for "a photo-sharing app like Instagram" (DAU: 50 M; each user uploads 2 photos/day, views 50 photos/day; avg photo 2 MB; thumbnail 20 KB).
2. A URL shortener is estimated at 150 bytes per record and 100 M new URLs/year. How much storage is needed for 5 years? What's the read QPS if click-through rate is 10:1 write?
3. Calculate the peak write bandwidth for a system with 500 k concurrent IoT sensors, each sending a 200-byte event every 5 seconds.
4. A design team argues whether to cache the entire dataset or just the "hot 10%". The full dataset is 10 TB. Cache RAM costs $5/GB-month. Estimate the monthly cost difference between the two approaches.

(Solutions: ../solutions/week-01/day-3.md)

---

## Self-check

- [ ] I can estimate QPS from DAU and actions-per-day in under 30 seconds.
- [ ] I can derive daily and annual storage from a per-record size.
- [ ] I know when bandwidth becomes a bottleneck (video/media >> text).
- [ ] I can size a cache from dataset size and access distribution.

---

## Supplementary

- [System Design Primer — Back-of-the-envelope calculations](https://github.com/donnemartin/system-design-primer#back-of-the-envelope-calculations)
- [Numbers Every Programmer Should Know — Jeff Dean](https://brenocon.com/dean_perf.html)
