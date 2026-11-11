# Phase 7 · Week 01 · Day 6 — Saturday Deep Practice: Write a Full System-Design Document

> Objective: Produce a complete, structured system-design document for a URL shortener using the 8-step framework.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

---

## Overview

Today is your first full-length system-design session. You will write a complete design document for a URL shortener — a canonical interview problem — by following every step of the framework you learned this week.

This is not a reading exercise. Put the material aside after reviewing the briefing, then write the document yourself. The goal is to build the **muscle memory** of moving through all 8 steps without forgetting any.

---

## The Problem Brief

> **Design a URL Shortener service (similar to bit.ly or TinyURL).**
>
> A URL shortener takes a long URL and returns a short alias. When a user visits the short alias, they are redirected to the original URL.
>
> The system should handle global traffic, be highly available, and redirect users with minimal latency.

---

## Session Structure (4 hours)

| Block | Duration | Task |
|---|---|---|
| Orientation | 15 min | Re-read the brief; sketch a rough outline; identify unknowns |
| Step 1–2 | 30 min | Write functional and non-functional requirements |
| Step 3–4 | 45 min | API design and data model |
| Step 5 | 45 min | High-level component diagram + brief component descriptions |
| Step 6 | 45 min | Two deep dives (choose: hashing strategy, redirect performance, or availability) |
| Step 7–8 | 30 min | Bottlenecks and trade-offs |
| Review | 30 min | Read your document; fill gaps; verify the diagram is self-consistent |

---

## Constraints (for the purposes of this exercise)

Use these numbers. Do not change them — they are your requirements:

- **100 M new URLs shortened per day**
- **10 B redirects per day**
- **Short code length:** your design must decide (5 chars? 7 chars? base62?)
- **URL retention:** 5 years
- **Availability:** 99.99%
- **Redirect latency:** p99 < 10 ms
- **Analytics:** page-view counts per short URL (eventually consistent is fine)

---

## Deliverable Format

Your document must include all of the following sections. Use headers exactly as shown.

```markdown
# System Design: URL Shortener

## 1. Requirements
### 1.1 Functional
### 1.2 Non-Functional
### 1.3 Out of Scope

## 2. Capacity Estimation
### 2.1 QPS
### 2.2 Storage
### 2.3 Bandwidth
### 2.4 Cache sizing

## 3. API Design
(REST or GraphQL — your choice, justify it)

## 4. Data Model
(SQL or NoSQL — your choice, justify it; show schema)

## 5. High-Level Design
(ASCII or Mermaid diagram + component descriptions)

## 6. Deep Dives
### 6.1 Short-code generation strategy
### 6.2 Redirect performance

## 7. Bottlenecks & Failure Modes

## 8. Trade-offs
(table format)
```

---

## Guidance: Short-Code Generation (study this before starting)

The trickiest part of this design is generating globally unique short codes without collisions.

Three approaches exist:

### Approach A — Hash + truncate

Hash the long URL (MD5 or SHA-256), take the first N characters of the base62 encoding.

```python
import hashlib
import base64

def shorten(url: str, length: int = 7) -> str:
    digest = hashlib.md5(url.encode()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode()
    return encoded[:length]
```

Problem: collisions. Two different URLs can produce the same 7-character prefix. Requires collision detection + retry.

### Approach B — Pre-generated ID pool

A separate service pre-generates random base62 codes and stores them in a "keys" table. The shortener claims a key atomically.

Pros: no collision risk, fast (no computation at request time).
Cons: operational complexity; key service is a single point of failure.

### Approach C — Auto-increment ID + base62 encode

Use a distributed counter (or DB auto-increment) and encode the integer in base62.

```python
def encode_base62(n: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    while n > 0:
        result.append(chars[n % 62])
        n //= 62
    return "".join(reversed(result)) or "0"
```

Base62 with 7 characters = 62^7 ≈ **3.5 trillion** unique codes. At 100 M/day, that's **95 years** of capacity.

Pros: simple, no collisions.
Cons: sequential IDs are guessable (enumerable). Mitigate by XOR-scrambling with a secret salt.

---

## Guidance: Redirect Performance

The redirect path is the most latency-sensitive operation. A 10 ms p99 target means:

- No cold DB queries on the hot path
- Every active short code must be in cache

Approach:
1. Client → CDN edge node → check edge cache
2. Cache miss → API server → Redis → cache hit
3. Redis miss → Postgres read replica → cache write-through → respond

With 10 B redirects/day, most popular codes are accessed millions of times. A small LRU cache handles the hot set.

---

## Exercises (embedded in the practice)

1. In your Capacity Estimation section, derive the minimum base62 code length to support 5 years of URLs at 100 M/day without collision.
2. In your Deep Dive 6.1, justify your chosen short-code generation approach using the trade-off formula from Day 5.
3. In your Bottlenecks section, identify what happens if the Redis cache cluster fails. Write a recovery strategy.
4. In your Trade-offs table, include: DB choice, code generation strategy, caching approach, and analytics storage.

(Solutions: ../solutions/week-01/day-6.md)

---

## Self-check

- [ ] My document covers all 8 framework steps.
- [ ] My capacity estimation shows calculations, not just final numbers.
- [ ] My API section shows HTTP methods, paths, request/response shapes.
- [ ] My diagram has labels on arrows and distinguishes different storage types.
- [ ] My trade-offs table has at least 4 rows with a "reason chosen" column.
- [ ] I can defend every decision if challenged in an interview.

---

## Supplementary

- [Designing a URL Shortening Service — System Design Primer](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/pastebin/README.md)
- [High Scalability — TinyURL Architecture](http://highscalability.com/blog/2009/11/25/tinys-not-so-tiny-url-shortening-service.html)
