# Week 01 Project — URL Shortener: Complete System-Design Document

> Deliverable: A production-grade written system-design document.  ·  Time: ~3 hr  ·  See day-6 for the session structure.

---

## Project Brief

Produce a complete, written system-design document for a URL shortener service. The document must follow the 8-step framework taught in Week 1 and satisfy the constraints below. This is the portfolio-quality artifact for Week 1.

Use the Day 6 session to draft the document and Day 7 to refine it.

---

## Constraints

| Parameter | Value |
|---|---|
| New URLs per day | 100 M |
| Redirects per day | 10 B |
| URL retention | 5 years |
| Availability | 99.99% |
| Redirect latency | p99 < 10 ms |
| Analytics (view counts) | Eventually consistent, updated within 60 s |
| Geography | Global (US, EU, APAC) |
| Short code length | Your design must justify the length |

---

## Required Sections

Your document must contain all of the following, in order:

### 1. Requirements

#### 1.1 Functional
List the user-observable capabilities the system must provide. Include at minimum: URL creation, URL resolution/redirect, URL expiry, and analytics.

#### 1.2 Non-Functional
Derive the NFRs from the constraints table above. Add any that are implied but not stated (e.g., data durability).

#### 1.3 Out of Scope
Explicitly list at least three features that are NOT in scope for this design.

---

### 2. Capacity Estimation

Show your working. Numbers without derivation do not count.

#### 2.1 QPS
- Average write QPS
- Average read QPS
- Peak write QPS (state your multiplier assumption)
- Peak read QPS

#### 2.2 Storage
- Per-record size (list every field, its type, and byte size)
- Daily storage added
- 5-year total storage
- Is sharding required? Justify.

#### 2.3 Bandwidth
- Inbound bandwidth (new URL creation)
- Outbound bandwidth (redirects)
- Is a CDN required? Justify.

#### 2.4 Cache Sizing
- What is the "hot set" of URLs?
- How much RAM is needed to cache it?
- What eviction policy do you choose and why?

---

### 3. API Design

Design the REST API. For each endpoint, specify:
- HTTP method + path
- Request body (JSON schema)
- Response body (JSON schema)
- HTTP status codes (success + error cases)

Minimum endpoints:
- `POST /urls` — create a short URL
- `GET /{short_code}` — redirect to original URL
- `DELETE /urls/{short_code}` — deactivate a short URL (authenticated)
- `GET /urls/{short_code}/stats` — view analytics

---

### 4. Data Model

Design the database schema. For each table:
- Column name, type, nullable, constraints
- Primary key
- Relevant indices (state the queries they serve)

Justify your choice of SQL vs. NoSQL. Include at minimum:
- `urls` table
- `analytics` table or design

---

### 5. High-Level Design

Draw a component diagram (ASCII or Mermaid) that includes:
- Global traffic path (client → CDN edge → load balancer → API → cache → DB)
- Write path (URL creation)
- Read/redirect path
- Analytics write path

Label all arrows with the protocol (HTTP, gRPC, SQL, pub/sub, etc.).

Follow the diagram with a brief description of each component's responsibility.

---

### 6. Deep Dives

Write two of the following. Each deep dive should be at least one full page.

#### Option A: Short-Code Generation Strategy
Compare the three approaches (hash-truncate, pre-generated key pool, base62 auto-increment). Choose one. Justify with a trade-off matrix. Address: collision probability, throughput under load, sequential enumeration risk.

#### Option B: Redirect Performance
Explain how you achieve p99 < 10 ms globally. Walk through the full request path for a cache hit and a cache miss. Include: CDN TTL strategy, Redis read path, DB fallback, connection pooling.

#### Option C: Availability and Failure Modes
Map out the failure modes for each component in your diagram. For each: impact, detection method, and recovery strategy. Address: how you achieve 99.99% with 4.38 min/month downtime budget.

---

### 7. Bottlenecks

Identify at least three bottlenecks in your design. For each:
- Component
- At what scale does it become a bottleneck?
- Mitigation strategy

---

### 8. Trade-offs

Write a trade-off table with at least 6 rows covering your major decisions.

| Decision | Option chosen | Alternative(s) | Reason |
|---|---|---|---|
| ... | ... | ... | ... |

---

## Evaluation Criteria

This project is evaluated on the same rubric as the assessment:

| Criterion | Points |
|---|---|
| Requirements completeness and scoping | 15 |
| Estimation: correct methodology and conclusions | 20 |
| API: completeness and REST correctness | 10 |
| Data model: schema quality, index justification | 10 |
| Diagram: clarity, labels, component coverage | 15 |
| Deep dives: technical depth and accuracy | 20 |
| Bottlenecks + trade-offs: specificity | 10 |
| **Total** | **100** |

Pass gate: 75 / 100

---

## Submission

Save your document as `week-01-project-submission.md` in:
`phase-7-system-design/week-01/`

Reference design: `../solutions/week-01/project.md`
