# Week 01 Assessment — HLD Fundamentals & the System-Design Interview Method

> Pass gate: 75%  ·  Time allowed: 90 minutes  ·  Open notes: No

---

## Part 1: Conceptual (25 points)

Answer each question in 2–5 sentences. Precision beats length.

1. **(3 pts)** You are 10 minutes into a system-design interview and the interviewer has not interrupted you. You have just finished the requirements and estimation steps. What is the most dangerous mistake you could make in the next 5 minutes, and how do you avoid it?

2. **(3 pts)** Explain the difference between an SLO and an SLA. Give a concrete example where a team's SLO is tighter than the customer-facing SLA, and explain why that is deliberately engineered that way.

3. **(3 pts)** A colleague claims: "p50 latency for our API is 40 ms — we're well within our 200 ms SLO." Why is this statement insufficient? What metric do you actually need, and why?

4. **(3 pts)** You are designing a system and must choose between Cassandra (AP) and Postgres (CP). Your NFRs say: 10 k writes/sec, 100 k reads/sec, data must never be lost, but stale reads are acceptable for up to 5 seconds. Which do you choose? Justify using the CAP theorem.

5. **(4 pts)** Describe the two-tier fan-out strategy for delivering notifications to 10 M subscribers. Why does a naïve single-tier approach fail, and what specifically does the second tier fix?

6. **(3 pts)** A system's error budget is 0.1% per month. Your team just spent 0.08% on a deploy gone wrong. There are 10 days left in the month. What engineering decisions does this force?

7. **(3 pts)** Compare push vs. pull architectures for delivering real-time feed updates. Name one system that uses each approach and state the trade-off each accepts.

8. **(3 pts)** Why does cascading latency affect a microservices chain with 10 hops much more severely than one with 3 hops? Use numbers from the latency table to illustrate.

---

## Part 2: Design Challenge (40 points)

### Problem: Design a Pastebin Service

A Pastebin service allows users to paste text (up to 10 MB) and get a short URL. Anyone with the URL can read the paste. Pastes can optionally expire.

**Constraints:**
- 5 M DAU
- 1 M new pastes per day
- 100 M paste views per day
- Max paste size: 10 MB (text only)
- Retention: user-specified (1 hr / 1 day / 1 week / 1 month / forever)
- Availability: 99.9%
- Read latency: p99 < 200 ms for pastes ≤ 1 KB; p99 < 2 s for pastes up to 10 MB

**Deliverable:** Produce a written system-design document covering:

1. Functional requirements (include explicitly what is out of scope).
2. Non-functional requirements (use the provided constraints; derive any missing values).
3. Capacity estimation: QPS (read + write), storage (per day and 5-year total), bandwidth, cache sizing.
4. API design: at minimum, `POST /pastes` and `GET /pastes/{id}`.
5. Data model: schema with column names, types, and indices. Justify SQL vs. NoSQL.
6. High-level component diagram (ASCII or Mermaid). Label all arrows.
7. Deep dive on **one** of: paste storage strategy for large pastes, expiry/deletion mechanism, or read scaling.
8. Bottlenecks: identify two and state mitigations.
9. Trade-offs table: at least 4 rows.

**Scoring:**
- Requirements completeness: 5 pts
- Estimation accuracy and methodology: 8 pts
- API correctness: 5 pts
- Data model: 5 pts
- Diagram clarity: 5 pts
- Deep dive depth: 7 pts
- Bottlenecks + trade-offs: 5 pts

---

## Part 3: Debugging & Problem-Solving (20 points)

**Scenario:**

A URL shortener has the following architecture:

```
Client → CDN → API Service (3 instances) → Redis → Postgres
```

**Problem:** Users report that 0.5% of redirect requests return a 404 error for short codes that definitely exist. The errors are intermittent and seem to happen for codes created in the last 10 minutes. Codes older than 30 minutes never produce 404s.

Answer the following:

1. **(5 pts)** Propose a hypothesis for the root cause. Be specific — name the component and the failure mode.
2. **(5 pts)** Describe the exact sequence of events that leads to the error (trace through the architecture diagram above).
3. **(5 pts)** Propose a fix. Be specific about which component changes and how.
4. **(5 pts)** How would you test that your fix is correct without creating the real bug in production?

---

## Part 4: Code Quality (15 points)

Below is a Python implementation of a rate limiter using Redis. Review it and answer the questions.

```python
import redis
import time

r = redis.Redis(host="localhost", port=6379)


def check_rate_limit(user_id, limit=100, window=60):
    key = "ratelimit:" + user_id
    count = r.incr(key)
    if count == 1:
        r.expire(key, window)
    if count > limit:
        return False
    return True
```

1. **(3 pts)** Identify one correctness bug. Describe the exact scenario in which it manifests.
2. **(3 pts)** Identify one operational/security concern with how the Redis connection is managed.
3. **(3 pts)** The function returns `bool`. Rewrite the signature using Python 3.12+ type hints and Pydantic where appropriate. You do not need to fix the bugs — just improve the types.
4. **(3 pts)** Replace the `redis.Redis` global with a pattern that allows this to be unit-tested without a real Redis instance.
5. **(3 pts)** The function uses no logging. Add one `logger.info` and one `logger.warning` call that would be useful in production. Name the fields you would include.

---

## Rubric

| Dimension | Weight | Points |
|---|---|---|
| Independent implementation (design challenge) | 40% | 40 |
| Conceptual understanding | 25% | 25 |
| Debugging & problem-solving | 20% | 20 |
| Code quality & testing | 15% | 15 |
| **Total** | **100%** | **100** |

**Pass gate: 75 / 100**

Answer key: ../solutions/week-01/assessment.md
