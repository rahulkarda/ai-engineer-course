# Week 03 Assessment — Data Layer at Scale

> Time: ~90 min  ·  Open notes, closed AI  ·  Pass threshold: 75%

---

## Conceptual (interview-style)

Answer each question as you would in a live system-design interview: concise, precise, with trade-offs where relevant. Aim for 2–4 sentences per answer.

**Q1.** A candidate says: "I'll just use MongoDB because it scales horizontally." What follow-up questions would you ask to evaluate whether that's the right choice?

**Q2.** You have a `users` table with 50 million rows. A query `SELECT * FROM users WHERE last_name = 'Smith' AND country = 'DE'` is slow. You create an index on `(last_name, country)`. Your colleague says you should have created it on `(country, last_name)` instead. Who is right, and why?

**Q3.** Explain the difference between synchronous and asynchronous replication. Give one concrete scenario where you would choose synchronous, and one where you would choose asynchronous.

**Q4.** You have a 4-node consistent hash ring with 100 virtual nodes per physical node. You add a 5th node. Approximately what fraction of keys will move, and why?

**Q5.** Your social feed service receives 50 000 reads/s and 500 writes/s. A product manager asks why you can't "just use the primary database for reads." Give a technically precise answer covering at least two failure modes or scaling concerns.

**Q6.** Describe the CQRS pattern. What problem does it solve? What new problem does it introduce? Give one example of a read model that would not exist in the write model.

**Q7.** You notice your PostgreSQL replica has a replication lag of 8 seconds. List three possible causes and one mitigation for each.

**Q8.** A data warehouse team asks why you use PostgreSQL for the transactional service and ClickHouse for analytics. They suggest consolidating onto one system. How do you respond? Reference OLTP vs OLAP characteristics.

---

## Coding Challenge

### Scenario

You are building the data layer for **FeedApp**, a read-heavy social feed service. Requirements:

- 10 M daily active users, 100 M posts total
- Read:write ratio 200:1 on the timeline endpoint
- Reads must return within p99 150 ms end-to-end
- Writes must be durable (no data loss on single node failure)
- Eventual consistency on timeline reads is acceptable (up to 2 s)
- Users have at most 10 000 followers (no celebrity fan-out problem for this exercise)

### Task

Implement `src/feedapp/data_layer.py` containing:

**1. `FeedShardRouter` (extends or wraps `ShardingRouter`)**

- Partition key: `user_id` (string)
- Must support 8 shards
- `get_shard_for_user(user_id: str) -> str`

**2. `FeedReplicaRouter` (extends or wraps `ReplicaRouter`)**

- 3 replicas per shard, max lag 500 ms
- `get_read_node_for_shard(shard: str) -> str`
- `get_write_node_for_shard(shard: str) -> str`

**3. `TimelineReadModel` (CQRS read model)**

```python
from pydantic import BaseModel
from datetime import datetime

class TimelineEntry(BaseModel):
    post_id: str
    author_id: str
    author_name: str
    body_preview: str        # first 280 chars
    like_count: int
    created_at: datetime

class TimelineReadModel:
    """In-memory CQRS read model for development/testing.

    Production version would be backed by a Redis Sorted Set or
    a denormalised Postgres table populated via event projection.
    """

    def __init__(self) -> None: ...

    def apply_post_created(self, follower_ids: list[str], entry: TimelineEntry) -> None:
        """Project a PostCreated event into each follower's timeline."""
        ...

    def get_timeline(
        self, user_id: str, limit: int = 20, before: datetime | None = None
    ) -> list[TimelineEntry]:
        """Return up to `limit` entries for `user_id`, newest first.

        If `before` is supplied, return only entries older than that timestamp.
        """
        ...

    def entry_count(self, user_id: str) -> int:
        """Return total number of timeline entries for `user_id`."""
        ...
```

**4. Tests** in `tests/unit/test_feedapp_data_layer.py`:

Write at minimum 5 tests covering:

- Correct shard assignment is deterministic for a given `user_id`
- All 8 shards receive traffic for 800 distinct user IDs (±10%)
- Write node is always a leader (`-leader` suffix)
- `TimelineReadModel.apply_post_created` adds entry to all follower timelines
- `TimelineReadModel.get_timeline` returns entries in descending time order
- `TimelineReadModel.get_timeline` with `before` correctly paginates

### Acceptance criteria

- [ ] `FeedShardRouter.get_shard_for_user` is deterministic
- [ ] 8 shards receive roughly equal load over 800 keys (each shard 75–125 keys)
- [ ] `route_write` always returns a node ending in `-leader`
- [ ] `TimelineReadModel` stores entries per follower and retrieves in correct order
- [ ] Pagination with `before` works correctly
- [ ] All tests pass: `pytest tests/unit/test_feedapp_data_layer.py -v`
- [ ] No hardcoded secrets; any config via `os.getenv`
- [ ] All public methods type-annotated with Python 3.12+ syntax

---

## Rubric (pass >= 75%)

| Dimension | Weight | What "full marks" looks like |
|---|---|---|
| Independent implementation | 40% | All methods implemented without copying Day 6 solutions verbatim; shows understanding of the algorithms |
| Conceptual understanding | 25% | Conceptual answers are precise, include trade-offs, no factual errors |
| Debugging & problem-solving | 20% | Tests are meaningful (not just happy-path); edge cases handled (empty ring, all replicas lagging, empty timeline) |
| Code quality & testing | 15% | Type hints everywhere, `logging` not `print`, Pydantic models used for `TimelineEntry`, no god functions |

### Scoring guide

| Score | Outcome |
|---|---|
| 90–100% | Distinction — ready for senior data-layer design questions |
| 75–89% | Pass — solid foundation, revisit weak conceptual answers |
| 60–74% | Near-pass — re-read Days 3–5, re-attempt conceptual section |
| < 60% | Retry — spend another session on sharding + replication fundamentals |

Answer key: ../solutions/week-03/assessment.md
