# Solutions — Week 05 · Day 7 (Sunday Exercises)

---

## Exercise 1 — Hardest estimation reflection

**Expected insight (not a fixed answer — compare your reasoning)**:

The **news feed** estimation is typically hardest because the fanout multiplier makes numbers non-intuitive. Writing QPS = posts/day is small (~600/s for 50 M DAU), but fanout write QPS = posts/day × avg_followers can be 100–1 000× higher. Students often forget the multiplier and underestimate storage and write load by 2 orders of magnitude.

**Chat** is the second-hardest because of the dual storage system (messages in Cassandra + metadata in Postgres) and the non-obvious concurrent connection estimate (50 M WebSocket connections require a fundamentally different capacity model than stateless HTTP).

**URL shortener** is the easiest to estimate because the 1:1 mapping between reads and writes is intuitive.

---

## Exercise 2 — Peer review rubric

**8-step checklist for peer review** (score each 0–2):

| Step | 0 | 1 | 2 |
|---|---|---|---|
| Requirements | Missing or vague | Has functional but not non-functional | Both F+NF, with concrete numbers |
| Estimation | Missing | Numbers present but ungrounded | Math shown, realistic scale |
| API | Missing | Endpoint names only | Full payload examples + verbs |
| Data model | Missing | Table names only | Field types, PKs, indices, storage placement |
| Architecture | Missing | Box-and-line | Protocols labeled, data flows directional |
| Deep dives | Missing | Surface description | Shows failure mode + fix, not just "use Redis" |
| Bottlenecks | Missing | 1 row | ≥ 3 rows with scale thresholds |
| Trade-offs | Missing | Generic statements | Specific to this system's decisions |

**Scoring**: 16/16 = excellent, 12–15 = good, < 12 = needs revision.

---

## Exercise 3 — POST /record endpoint

```python
# src/typeahead/api.py — add to existing api.py

from pydantic import BaseModel, Field


class RecordQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100)
    frequency: int = Field(default=1, ge=1, le=1000)


class RecordQueryResponse(BaseModel):
    query: str
    frequency: int
    status: str


@app.post("/record", response_model=RecordQueryResponse, status_code=201)
def record_query(body: RecordQueryRequest) -> RecordQueryResponse:
    """Record a user query and update its frequency in the trie."""
    assert _service is not None
    _service.record_query(body.query, body.frequency)
    logger.info("Recorded query=%r frequency=%d", body.query, body.frequency)
    return RecordQueryResponse(
        query=body.query.lower(),
        frequency=body.frequency,
        status="recorded",
    )
```

**Test**:
```python
# tests/unit/test_typeahead_api.py
from fastapi.testclient import TestClient
from src.typeahead.api import app

client = TestClient(app)

def test_record_then_suggest():
    client.post("/record", json={"query": "pytest", "frequency": 50})
    resp = client.get("/suggest", params={"q": "py"})
    assert resp.status_code == 200
    assert "pytest" in resp.json()["suggestions"]

def test_record_invalid_query():
    resp = client.post("/record", json={"query": "", "frequency": 1})
    assert resp.status_code == 422  # validation error

def test_record_negative_frequency():
    resp = client.post("/record", json={"query": "test", "frequency": -1})
    assert resp.status_code == 422
```

**Notes**:
- The `lifespan` context manager initializes `_service`. In tests using `TestClient`, the lifespan runs automatically. If using an older version of FastAPI (pre-0.93), use `@app.on_event("startup")` instead.
- `assert _service is not None` is a type narrowing guard; in production code you'd raise `HTTPException(503)` if the service is not initialized.
