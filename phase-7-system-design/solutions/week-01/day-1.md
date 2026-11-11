# Solutions — Day 1: The System-Design Interview Framework

---

## Exercise 1 — URL shortener framework outline

| Step | One-sentence answer |
|---|---|
| 1. Requirements | Users can shorten a URL and retrieve the original URL via the short code. |
| 2. NFRs | 100 M new URLs/day, 10 B redirects/day, redirect p99 < 10 ms, 99.99% availability. |
| 3. API | `POST /urls` (returns short code), `GET /{code}` (302 redirect). |
| 4. Data Model | `urls` table: `short_code`, `original_url`, `user_id`, `created_at`, `expires_at`. |
| 5. High-level | Client → CDN → LB → API → Redis cache → Postgres. |
| 6. Deep dives | Short-code generation (collision avoidance), redirect latency (cache warm-up). |
| 7. Bottlenecks | Redis cache size; DB write throughput at peak. |
| 8. Trade-offs | Hash vs. auto-increment code generation; SQL vs. NoSQL for URL store. |

---

## Exercise 2 — Time allocation for ride-sharing matching

Suggested allocation:
- Requirements: 3 min
- NFRs + estimation: 4 min
- API: 3 min
- Data model: 3 min
- High-level design: 5 min
- Deep dives: 15 min
- Bottlenecks + trade-offs: 5 min

**Most time on deep dives.** The matching algorithm (geospatial proximity, driver availability, ETA prediction) is the intellectual core of the problem. Generic CRUD is trivially easy; the matching logic is where interviewers probe depth.

---

## Exercise 3 — Kafka hot partitions

This question belongs to **Step 6 (Deep Dives)** — specifically the sub-problem of scaling the fan-out pipeline.

Model answer: "Hot partitions occur when a small number of topic IDs have disproportionate subscriber counts, routing all traffic to one Kafka partition and one consumer. The fix is to add a secondary partitioning key: for topics over a threshold (e.g., 1 M subscribers), assign a 'shard ID' suffix (topic_id:0, topic_id:1, ...) and spread messages across multiple partitions. Consumers aggregate across shards before delivery."

---

## Exercise 4 — Signals that the interviewer wants you to go deeper

1. They ask a "how would you handle...?" question about a specific component (e.g., "How would you handle the case where the cache is cold?").
2. They lean forward, stop writing, and maintain eye contact while you describe a component.
3. They repeat back your statement as a question: "So you said Redis — why Redis specifically?"

In all three cases: stop advancing to the next step, pivot to that component, and go one level deeper.
