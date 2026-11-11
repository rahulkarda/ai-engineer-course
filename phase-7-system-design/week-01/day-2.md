# Phase 7 · Week 01 · Day 2 — Requirements: Functional vs Non-Functional

> Objective: Master the art of scoping a design problem through structured requirements gathering.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

Requirements are the contract between you and the interviewer. Getting them wrong wastes the remaining 40 minutes. Getting them right lets you make every downstream decision confidently.

There are two categories:

| Type | Question answered | Example |
|---|---|---|
| **Functional** | What does the system *do*? | "Users can upload a file up to 5 GB" |
| **Non-functional** | How well does it do it? | "Upload must succeed within 30 s on a 100 Mbps connection" |

Both are mandatory. A design without NFRs has no basis for technology choices. A design without functional requirements may solve the wrong problem.

---

## Functional Requirements

Functional requirements describe **user-observable behaviour**. Write them as user stories or capability statements:

- "A user can shorten a URL and receive a unique short code."
- "A user can retrieve the original URL by visiting the short code."
- "An admin can deactivate a short link."

### Scoping — what to exclude

In an interview, you will never build everything. State explicitly what is **out of scope**:

- Analytics dashboard → out of scope
- Custom vanity URLs → out of scope for v1
- OAuth login → assume authenticated token is passed

This is not laziness — it's professional scoping. Real engineers do it constantly.

### Clarifying questions to ask

```
Who are the users?        (consumers, businesses, internal tooling?)
What are the core flows?  (happy path only, or error recovery too?)
What's the scale?         (10 k users or 10 M?)
Are there existing systems to integrate with?
Any regulatory constraints? (GDPR, HIPAA, SOC 2?)
```

---

## Non-Functional Requirements

NFRs define the **quality attributes** of the system. They drive architecture.

### The CARS mnemonic

| Letter | Attribute | Example |
|---|---|---|
| **C** | Consistency | All replicas return the same data after a write |
| **A** | Availability | System is up 99.9% of the time |
| **R** | Read/Write ratio | 100:1 read-heavy |
| **S** | Scale | 100 M users, 500 writes/sec peak |

Add: **Latency** (P99 < 200 ms), **Durability** (no data loss), **Security** (encryption at rest/in transit).

### How NFRs drive architecture decisions

```
High read:write ratio        → caching layer (Redis), CDN for static assets
Strong consistency required  → single-writer DB, avoid eventual consistency
Geo-distributed users        → multi-region deployment, data residency rules
High durability              → replication factor ≥ 3, cross-region backups
Low latency                  → colocate compute + data, avoid network hops
```

### SLA numbers to know

| Availability | Downtime/year | Downtime/month |
|---|---|---|
| 99%     | 3.65 days | 7.3 hours |
| 99.9%   | 8.77 hours | 44 min |
| 99.99%  | 52 min | 4.4 min |
| 99.999% | 5.25 min | 26 sec |

Most consumer systems target 99.9%. Financial systems: 99.99%+.

---

## How it works internally

### The requirements-to-design pipeline

```
Functional reqs
    │
    ├─► identify entities (data model)
    ├─► identify operations (API)
    └─► identify flows (component diagram)

Non-functional reqs
    │
    ├─► scale numbers → capacity estimation (Day 3)
    ├─► latency targets → caching, co-location decisions
    ├─► availability → replication, failover strategy
    └─► consistency → CAP trade-off, DB choice
```

### Common mistakes

1. **Skipping NFRs entirely** — leads to a design that works at 100 users but fails at 100 M.
2. **Gold-plating functional scope** — adding real-time analytics when the problem says "users can shorten URLs". Signal you know what to cut.
3. **Vague NFRs** — "fast" and "reliable" are not requirements. "P99 latency < 200 ms" and "99.9% availability" are.

---

## Worked Example

**Problem:** "Design a collaborative document editor (like Google Docs)."

### Functional requirements (after asking clarifying questions)

Core:
- Users can create, edit, and delete documents.
- Multiple users can edit the same document simultaneously and see each other's changes in real-time.
- Changes are persisted automatically.

Out of scope (v1):
- Version history / undo beyond in-session
- Comments and suggestions mode
- Offline editing
- Mobile apps (web only)

### Non-functional requirements

| NFR | Target |
|---|---|
| Concurrent editors per doc | Up to 100 |
| Real-time sync latency | < 500 ms P99 |
| Total documents | 1 B |
| Active users (DAU) | 10 M |
| Write throughput | ~50 k ops/sec (assuming 5 ops/user/sec for active editors) |
| Availability | 99.99% |
| Data durability | No data loss (RPO = 0) |
| Consistency | Strong within a session; eventual across devices |

### What these NFRs imply

- 500 ms real-time → WebSocket, not polling
- 50 k ops/sec → append-only operation log, not full-document saves
- Strong within session → operational transformation (OT) or CRDT for conflict resolution
- No data loss → write-ahead log, synchronous replication before ack

---

## Exercises

1. You're designing "a file storage service (like Dropbox)". Write 5 functional requirements and 5 non-functional requirements. For each NFR, state one architectural implication.
2. An interviewer says "Design Twitter." List three clarifying questions you'd ask before writing a single requirement, and explain what the answer to each question would change about your design.
3. Classify these as Functional (F) or Non-Functional (NF):
   - "Users can follow other users."
   - "The feed must load in under 300 ms for 95% of requests."
   - "The system must support 10 k new user registrations per day."
   - "Posts can include images up to 20 MB."
   - "The system must be available in the EU and US with data residency."
4. You discover mid-design that a requirement is ambiguous: "notifications should be delivered quickly." How do you handle this in an interview? Write the exact question you'd ask.

(Solutions: ../solutions/week-01/day-2.md)

---

## Self-check

- [ ] I can distinguish functional from non-functional requirements with examples.
- [ ] I can derive architectural implications from NFR numbers.
- [ ] I can scope a problem by explicitly naming what's out of scope.
- [ ] I know the downtime budget for 99.9%, 99.99%, and 99.999% availability.

---

## Supplementary

- [System Design Primer — Performance vs Scalability](https://github.com/donnemartin/system-design-primer#performance-vs-scalability)
- [AWS Well-Architected Framework — Reliability Pillar](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html)
