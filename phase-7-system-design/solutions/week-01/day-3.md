# Solutions — Day 3: Back-of-Envelope Estimation

---

## Exercise 1 — Photo-sharing app (Instagram)

**Assumptions stated explicitly:**
- DAU: 50 M
- Upload: 2 photos/user/day
- Views: 50 photos/user/day
- Full photo: 2 MB; thumbnail: 20 KB

**QPS:**
```
Write QPS = 50M × 2 / 100,000 = 1,000 uploads/sec
Read  QPS = 50M × 50 / 100,000 = 25,000 views/sec
Peak write = 1,000 × 3 = 3,000/sec
Peak read  = 25,000 × 3 = 75,000/sec
```

**Storage (daily):**
```
Full photos: 1,000/sec × 86,400 sec × 2 MB = 172 TB/day
Thumbnails:  1,000/sec × 86,400 sec × 20 KB = 1.72 TB/day
Total:       ~174 TB/day
Annual:      ~63 PB  → object storage + CDN mandatory
```

**Bandwidth:**
```
Inbound (uploads): 3,000 uploads/sec × 2 MB = 6 GB/s inbound
Outbound (views):  75,000 views/sec × 20 KB (thumbnail) = 1.5 GB/s
                   (full load is larger; CDN absorbs most of it)
```

**Conclusions:** Storage is the dominant constraint. Object store (S3) mandatory. CDN required for outbound bandwidth. Photos are never stored in relational DB.

---

## Exercise 2 — URL shortener

```
Records/year = 100 M URLs/year
5-year total = 500 M records × 150 bytes = 75 GB

Read QPS (10:1 click rate):
  100 M new URLs/year ÷ 365 days ≈ 274 K new URLs/day
  Write QPS = 274 K / 86,400 ≈ 3.2 writes/sec
  Read  QPS = 3.2 × 10 = 32 reads/sec
  (These are low; a "globally popular" URL shortener at 10 B redirects/day is a different scale)
```

Note: if 10 B redirects/day are assumed:
```
Read QPS = 10 B / 86,400 ≈ 115,700 reads/sec
Peak     = 115,700 × 3 ≈ 347,000 reads/sec
```

75 GB total storage fits on a single large SSD server. Sharding not needed for storage; caching needed for reads at 115 k+ QPS.

---

## Exercise 3 — IoT sensor peak bandwidth

```
Sensors: 500,000
Event size: 200 bytes
Frequency: 1 event every 5 seconds = 0.2 events/sec

Write QPS = 500,000 × 0.2 = 100,000 writes/sec
Bandwidth = 100,000 × 200 B = 20 MB/s = 160 Mbps inbound

Conclusion: manageable on a 1 Gbps uplink; write QPS of 100 k/sec
requires a time-series DB (InfluxDB, TimescaleDB) or stream processor
(Kafka + Flink), not a standard RDBMS.
```

---

## Exercise 4 — Cache cost comparison

```
Full dataset:  10 TB
Hot 10%:       1 TB

RAM cost: $5/GB-month

Full caching:  10,000 GB × $5 = $50,000/month
10% hot cache: 1,000 GB × $5  = $5,000/month

Monthly saving: $45,000/month

Decision: cache the hot 10%. The Pareto principle holds: 10% of data
serves 90%+ of traffic. The $45 k/month saving funds other infrastructure.
Tiered approach: hot cache in Redis; warm data on SSD-backed DB; cold in
cheaper HDD-backed storage.
```
