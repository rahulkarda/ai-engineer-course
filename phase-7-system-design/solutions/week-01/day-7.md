# Solutions — Day 7: Assessment Prep Review

---

## Framework review

**1. The 8 steps:**
Requirements → Constraints/NFRs → API Design → Data Model → High-Level Design → Deep Dives → Bottlenecks → Trade-offs.

**2. Danger of skipping requirements:**
Without requirements, every design decision is a guess. You may optimise for the wrong scale (10 k users vs. 10 M), choose the wrong consistency model, or over-engineer features that are out of scope. You lose credibility as an engineer who communicates before building.

**3. Step for Kafka vs. RabbitMQ:**
Step 6 (Deep Dives) or Step 7 (Bottlenecks). Specifically in the context of deep-diving the async messaging component or when discussing why RabbitMQ's per-message acknowledgement model doesn't support fan-out to 10 M subscribers.

---

## Requirements review

**4. Ride-sharing examples:**
- Functional: "A rider can request a ride from their current location to a destination."
- Non-functional: "The matching algorithm must return a driver assignment within 2 seconds (p99)."

**5. CARS:** Consistency · Availability · Read/Write ratio · Scale.

---

## Estimation review

**6. Messaging app QPS:**
```
Write: 50M × 5 / 100,000 = 2,500 writes/sec
Peak:  2,500 × 3 = 7,500 writes/sec
```

**7. Storage:**
```
Messages/day = 2,500 × 86,400 = 216 M messages/day
Daily storage = 216 M × 200 B = 43.2 GB/day
Annual storage = 43.2 × 365 ≈ 15.8 TB/year
```

---

## Latency review

**8. HDD vs SSD random read:**
HDD seek ~10 ms, SSD random read ~100–150 μs. HDD is approximately 100× slower for random access. Sequential reads are closer (HDD ~100 MB/s vs SSD ~500 MB/s).

**9. Three-service chain p99:**
50 ms × 3 = 150 ms (additive approximation). In practice slightly worse due to tail independence.

---

## SLA/SLO review

**10. Error budget for 99.95%:**
```
Downtime allowed = 0.0005 × 43,200 min = 21.6 min/month
```

**11. SLO vs SLA relationship:**
SLO is the internal engineering target; SLA is the external contractual commitment. The SLO is always set tighter than the SLA to provide a buffer — if the SLO is breached, engineering has time to respond before the SLA is violated.

---

## Diagramming review

**12. CDN-backed static file server:**

```
[ Browser ]
    │ HTTPS
    ▼
[ CDN Edge Node ]
    │ (cache miss only) HTTPS
    ▼
[ Origin Server ]
    │ HTTP (internal)
    ▼
( Object Store / S3 )

Flow:
  1. Browser requests /logo.png
  2. CDN edge: cache hit → return immediately (< 10 ms)
  3. CDN edge: cache miss → fetch from origin, cache with TTL, return
  4. Origin fetches from object store if not in local cache
```

---

## Trade-offs review

**13. Eventual consistency for analytics counter:**

> "I chose eventual consistency for the analytics click counter rather than strong consistency.
> Eventual consistency lets me write click events to a Kafka queue and aggregate asynchronously, which keeps the redirect path at p99 < 10 ms — no blocking DB write on the critical path.
> The cost is that click counts may lag by up to 60 seconds.
> That trade-off is acceptable because the NFRs explicitly state analytics can be eventually consistent within 60 seconds — the user clicking a link doesn't need to see an updated count immediately."
