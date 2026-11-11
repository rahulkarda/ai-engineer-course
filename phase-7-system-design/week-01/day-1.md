# Phase 7 · Week 01 · Day 1 — The System-Design Interview Framework

> Objective: Learn the 8-step SD interview method and practice driving a session end-to-end.  ·  Time: ~1 hr  ·  Prereqs: none (conceptual day)

---

## Concept

A system-design interview is a 45-minute conversation in which you demonstrate that you can take a vague problem statement and turn it into a credible architecture. Interviewers are not looking for the "right" answer — they're watching **how you think**: do you gather requirements before drawing boxes? Do you identify bottlenecks? Can you communicate trade-offs clearly?

The framework below is a repeatable process. Internalise it until it's automatic.

---

## The 8-Step Framework

```
Step 1  Requirements          2–3 min   What are we building? (functional)
Step 2  Constraints / NFRs    2–3 min   How big / fast / reliable? (non-functional)
Step 3  API Design            3–5 min   Public interface — endpoints, RPCs, events
Step 4  Data Model            3–5 min   Schema sketch — entities, relationships, indices
Step 5  High-Level Design     5–8 min   Component diagram — services, DBs, caches, queues
Step 6  Deep Dives            8–10 min  Pick 2-3 hard sub-problems; go deep on each
Step 7  Bottlenecks           3–5 min   Where does the system break? How to fix?
Step 8  Trade-offs            3–5 min   What did you give up? What would you change?
        ──────────────────────────────
        Total ≈ 35–45 min
```

This is a **communication protocol**, not a checklist. Interviewer cues should pull you deeper into any step.

---

## How it works internally

### Why this order matters

Each step's output feeds the next:

- Requirements → constrain the API surface
- API → reveals the data entities
- Data model → drives storage choices in the high-level design
- High-level design → exposes the components you need to deep-dive
- Deep dives → surface the bottlenecks
- Bottlenecks → force trade-off decisions

Skipping straight to the diagram is the single most common mistake. Without requirements, your diagram is a guess.

### The interviewer's mental model

Interviewers evaluate you on three axes:

| Axis | What they look for |
|---|---|
| Breadth | Do you know all the relevant building blocks? |
| Depth | Can you go deep on the hard parts? |
| Judgment | Do you make principled trade-off decisions? |

A candidate who draws a perfect diagram but can't explain *why* they chose Postgres over Cassandra fails on judgment.

### Time management

The most common failure mode is spending 20 minutes on requirements and leaving no time for deep dives. Use a mental timer:

```
0:00  Start — ask clarifying questions
5:00  Move to API design
10:00 High-level diagram on the board
20:00 Signal you're entering deep dives: "I'd like to go deeper on X and Y"
35:00 Bottlenecks + trade-offs
```

If the interviewer keeps asking questions at a step — stay there; they're steering you.

---

## Worked Example

**Problem:** "Design a system for sending push notifications."

Walk through the framework verbally:

### Step 1 — Requirements (functional)
- Users subscribe to topics
- Publishers push a message to a topic
- All subscribers receive the notification within a few seconds
- Notifications delivered to iOS, Android, Web

### Step 2 — Constraints / NFRs
- 50 M active users, 1 M publishers
- Peak: 100k messages/sec fan-out to 10 subscribers on average → 1 M delivery/sec
- P99 delivery latency < 5 s
- At-least-once delivery (idempotent on the device side)
- 99.9% availability

### Step 3 — API (REST sketch)
```
POST /subscriptions          body: { user_id, topic_id }
DELETE /subscriptions/{id}
POST /topics/{id}/messages   body: { title, body, data }
GET  /devices/{user_id}      → registered device tokens
```

### Step 4 — Data Model (sketch)
```
users(id, name)
devices(id, user_id, platform, token, updated_at)
topics(id, name, owner_id)
subscriptions(user_id, topic_id, created_at)  ← PK composite
messages(id, topic_id, payload, created_at)
deliveries(msg_id, device_id, status, attempted_at)
```

### Step 5 — High-Level Design

```
Publisher
    │
    ▼
[ API Gateway ]──────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
[ Message Service ]   ──push──►  [ Queue (Kafka) ]
                                         │
                              ┌──────────▼──────────┐
                              │  Fan-out Workers     │
                              │  (read subscriptions)│
                              └──────┬───────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
                  [APNs]          [FCM]          [Web Push]
```

### Step 6 — Deep Dives (pick one)

**Fan-out at scale:** For a topic with 10 M subscribers, fan-out is too slow in-line. Use a two-tier approach: a "fat fan-out" writes to a partitioned Kafka topic keyed by user shard; shard workers read and call APNs/FCM in micro-batches. This caps per-worker responsibility to ~100k users.

### Step 7 — Bottleneck

APNs/FCM rate-limit per certificate (~1k/sec). Mitigation: maintain a per-provider connection pool with retry + exponential backoff; bucket deliveries by priority.

### Step 8 — Trade-offs

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Queue | Kafka | SQS | Need topic-partitioned fan-out; Kafka's consumer groups fit |
| Delivery guarantee | At-least-once | Exactly-once | Simpler; dedup on device |
| Fan-out model | Pull (worker reads subs) | Push (write to per-user queue) | Read fan-out cheaper at this scale |

---

## Exercises

1. Apply the 8-step framework to "Design a URL shortener" — write a one-sentence answer for each step (don't build the full design yet; just practice the framework outline).
2. You're given 45 minutes. Sketch a rough time allocation for designing "a ride-sharing matching system" and justify which step you'd spend the most time on.
3. An interviewer asks "How would you handle hot partitions in your Kafka fan-out?" — identify which step that question belongs to and write a 3-sentence answer.
4. List three signals that tell you an interviewer wants you to go deeper on a specific component rather than moving to the next step.

(Solutions: ../solutions/week-01/day-1.md)

---

## Self-check

- [ ] I can recite the 8 steps from memory in under 30 seconds.
- [ ] I can explain why requirements come before the API design.
- [ ] I can describe what "judgment" means to a system-design interviewer.
- [ ] I can apply rough time allocations to each step for a 45-minute session.

---

## Supplementary

- [System Design Primer — The Interview Process](https://github.com/donnemartin/system-design-primer#the-system-design-interview)
- [Grokking Modern System Design Interview](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers)
