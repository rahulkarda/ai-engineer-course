# Phase 7 · Week 05 · Day 7 (Sunday) — Week Project + Assessment Prep

> Objective: Complete the week project, run a mock interview, and consolidate knowledge.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[week-05/day-6]]

---

## Overview

| Block | Time | Activity |
|---|---|---|
| A | 90 min | Week project: polish and finalize two designs + implementations |
| B | 45 min | Mock interview: present your designs aloud (solo or with a peer) |
| C | 30 min | Assessment warm-up: review conceptual questions |
| D | 15 min | Self-audit: code quality, test coverage, documentation |

---

## Block A — Week Project Finalization

### Deliverable checklist

Your week project consists of two polished design documents and two reference implementations. By end of today:

**Design Documents** (see `project.md` for full spec):
- [ ] Design 1 (Chat + Messaging): all 8 framework steps; ≥ 2 Mermaid/ASCII diagrams; bottleneck + trade-off tables.
- [ ] Design 2 (Distributed Rate Limiter): all 8 framework steps; ≥ 1 diagram; algorithm comparison table.
- [ ] Both designs cite at least one real-world system (Discord, Slack, Stripe, etc.) as a reference point.

**Reference Implementations**:
- [ ] `src/rate_limiter/`: `token_bucket.py`, `config.py`, `middleware.py`.
- [ ] `src/typeahead/`: `trie.py`, `service.py`, `api.py`.
- [ ] `tests/unit/test_token_bucket.py`: ≥ 8 passing tests.
- [ ] `tests/unit/test_trie.py`: ≥ 10 passing tests.
- [ ] `pyproject.toml` with ruff, black, pytest config.
- [ ] `.env.example` with `RL_REDIS_URL` placeholder (no real credentials).
- [ ] All files: type hints, `logging` not `print`, no secrets.

### Code quality audit

Run this before submitting:

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/ --strict

# Tests
pytest tests/unit/ -v --tb=short

# Coverage (aim for ≥ 80% on src/)
pytest tests/unit/ --cov=src --cov-report=term-missing
```

---

## Block B — Mock Interview: Present Your Designs

### Solo practice protocol

Set a timer for 45 minutes total. Pretend you are in a live interview.

**Warm-up (5 min)**: state the problem, ask 2–3 clarifying questions (even if no one answers them — the habit matters).

**Chat system design (20 min)**:
1. Requirements (2 min)
2. Estimation (2 min)
3. API + data model (4 min)
4. Architecture diagram — draw it while narrating (5 min)
5. Deep dives: message delivery, presence (5 min)
6. Wrap: bottlenecks + trade-offs (2 min)

**Rate limiter design (20 min)**: same structure.

### Evaluator rubric (self-grade)

| Behavior | Did you do it? |
|---|---|
| Asked clarifying questions before drawing | Y / N |
| Stated assumptions explicitly | Y / N |
| Drove the session (didn't wait to be prompted) | Y / N |
| Named at least one real-world system as reference | Y / N |
| Identified a non-obvious bottleneck | Y / N |
| Offered a trade-off without being asked | Y / N |
| Stayed within time | Y / N |

Score yourself: 7/7 = excellent; 5–6 = good; < 5 = repeat the mock.

---

## Block C — Assessment Warm-Up

Review these conceptual questions. Write short answers from memory before checking the course notes.

### System design rapid-fire

1. URL shortener: you need to store 50 billion short URLs. Postgres or Cassandra? Justify in two sentences.
2. News feed: user A follows 1 000 celebrities. What goes wrong with pure fanout-on-write? What's the fix?
3. Chat: your WebSocket gateway pod restarts. How do clients know to reconnect? How do they catch up on missed messages?
4. Rate limiter: explain why two Redis calls (`GET` then `INCR`) are not atomic and what problem this causes under concurrent load.
5. Geohash: precision 5 gives cells of ~5 km. A user wants results within 1 km. What precision do you use and how many cells do you query?
6. Typeahead: your trie holds 50 M words. Insertion of a new word costs O(word_length). What operation is expensive and often overlooked?
7. Notification: your push notification fails 3 times. What do you put in the dead-letter queue entry, and what does the on-call alert contain?

### Estimation practice

Without a calculator, estimate:
- How much RAM does a Redis sorted set of 10 M entries cost? (Each member: 30 B string + 8 B score)
- How many Redis keys does the typeahead service create for the top 1 M queries (avg length 7)?

---

## Block D — Self-Audit Checklist

Before finalizing your project submission:

**Architecture**
- [ ] No service is a god-object (> 3 responsibilities).
- [ ] Every stateful component (Cassandra, Redis, Postgres) has a failover strategy mentioned.
- [ ] Every async component (Kafka) has a dead-letter queue strategy mentioned.

**Code**
- [ ] All public functions have type annotations.
- [ ] No `print()` statements — `logging` only.
- [ ] No hardcoded credentials or URLs (use `os.getenv` or pydantic-settings).
- [ ] Tests use mocks for Redis and external services.
- [ ] At least one test for the sad path (rate limit hit, Redis down, empty trie).

**Documentation**
- [ ] `README.md` describes how to run both services locally.
- [ ] `.env.example` has all required env vars with descriptions.

---

## Exercises

1. Reflection: which of the five problems this week (URL shortener, news feed, chat, rate limiter, geo+typeahead) was hardest to estimate? Why?
2. Peer review: swap your two design documents with a classmate (or review your own as if you'd never seen them). Use the 8-step framework as a rubric and score each section 0–2.
3. Extend the typeahead `api.py` to add a `POST /record` endpoint that accepts `{"query": "...", "frequency": 1}` and calls `service.record_query`. Write the Pydantic request model and the endpoint handler.

(Solutions: ../solutions/week-05/day-7.md)

---

## Self-check
- [ ] Both design documents complete with all 8 steps.
- [ ] Both implementations pass their test suites.
- [ ] I can present either design verbally in 20 min without notes.
- [ ] I can explain every trade-off decision in my designs from first principles.

---

## Looking ahead

**Week 06** shifts to advanced distributed systems topics: consensus (Raft/Paxos), distributed transactions, and coordination patterns. The system-design problems will require deeper understanding of failure modes and consistency guarantees.

Preview: Zookeeper, etcd, two-phase commit, saga pattern — and when each is appropriate.
