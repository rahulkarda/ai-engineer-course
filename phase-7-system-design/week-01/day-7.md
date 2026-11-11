# Phase 7 · Week 01 · Day 7 — Sunday: Project Completion + Assessment Prep

> Objective: Finalise the week project, review all concepts, and prepare for the assessment.  ·  Time: ~3 hr  ·  Prereqs: [[day-6]]

---

## Overview

Today has two parts:

1. **Project completion (2 hr):** Finish and refine your URL Shortener system-design document from Day 6. If you finished Day 6, spend today on depth: write the Python scaffold, add a metrics/observability section, and harden your trade-offs.
2. **Assessment prep (1 hr):** Self-test against the week's concepts before the formal assessment.

---

## Part 1: Project Work (2 hours)

### If you didn't finish Day 6

Complete the missing sections first. Use the deliverable format from Day 6.

### If you finished Day 6 — go deeper

Pick two of the following extensions and add them to your document:

#### Extension A: Observability

Add a section "Observability & Monitoring" that covers:
- Which metrics you would instrument (e.g., redirect latency p50/p95/p99, cache hit rate, error rate by HTTP status).
- Which SLIs map to your SLOs.
- How you would alert: on what threshold, with what escalation.

Example metrics code (Python 3.12+, using `logging` module):

```python
import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def timed_operation(operation_name: str) -> Generator[None, None, None]:
    """Context manager that logs the duration of a named operation."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "operation_complete",
            extra={
                "operation": operation_name,
                "duration_ms": round(duration_ms, 3),
            },
        )
```

#### Extension B: Failure Scenarios

Add a "Failure Analysis" section using this table template:

| Component | Failure mode | Impact | Detection | Recovery |
|---|---|---|---|---|
| Redis cache | All nodes down | All redirects hit DB; 10× read load | Cache miss rate → 100% alert | Fallback to DB reads; scale DB read replicas |
| Postgres primary | Crash | Writes fail; reads continue from replica | Healthcheck timeout | Promote replica; update DNS; restart writes |
| Short-code generator | Counter service down | Cannot create new short URLs | Error rate spike | Queue new-URL requests; drain queue when service recovers |

#### Extension C: Python Scaffold

Write a minimal FastAPI application skeleton for the two core endpoints. No business logic needed — just the structure.

```python
# src/url_shortener/api.py
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl

logger = logging.getLogger(__name__)

app = FastAPI(title="URL Shortener", version="0.1.0")


class ShortenRequest(BaseModel):
    url: HttpUrl
    custom_alias: str | None = None


class ShortenResponse(BaseModel):
    short_code: str
    short_url: str
    original_url: str


@app.post("/urls", response_model=ShortenResponse, status_code=status.HTTP_201_CREATED)
async def shorten_url(body: ShortenRequest) -> ShortenResponse:
    """Create a short alias for a long URL."""
    # TODO: call ShortCodeService.generate()
    # TODO: persist to DB, cache in Redis
    raise NotImplementedError("Implement short-code generation service")


@app.get("/{short_code}", status_code=status.HTTP_302_FOUND)
async def redirect(short_code: str) -> RedirectResponse:
    """Resolve short code and redirect to original URL."""
    # TODO: check Redis cache first
    # TODO: fallback to DB read
    # TODO: increment analytics counter (async, fire-and-forget)
    original_url = _resolve(short_code)
    if original_url is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Short code not found")
    logger.info("redirect", extra={"short_code": short_code, "target": original_url})
    return RedirectResponse(url=original_url)


def _resolve(short_code: str) -> str | None:
    """Placeholder: look up original URL by short code."""
    return None
```

---

## Part 2: Assessment Prep (1 hour)

Work through these review questions without notes. If you can't answer one, return to the relevant day.

### Framework review

1. Name the 8 steps of the system-design interview framework in order.
2. What is the danger of jumping to the component diagram without completing requirements?
3. At what step would you discuss the choice between Kafka and RabbitMQ?

### Requirements review

4. Give an example of a functional requirement and a non-functional requirement for a ride-sharing app.
5. What does the CARS mnemonic stand for?

### Estimation review

6. A system has 50 M DAU. Each user sends 5 messages per day. What is the average write QPS? What is peak QPS assuming 3× multiplier?
7. Each message record is 200 bytes. How much storage does the system need per day? Per year?

### Latency review

8. Approximately how much slower is an HDD seek compared to an SSD random read?
9. A chain of 3 microservices each has p99 = 50 ms. What is the approximate p99 of the full request?

### SLA/SLO review

10. An SLO is 99.95% availability. How many minutes of downtime are allowed per month?
11. What is the relationship between an SLO and an SLA?

### Diagramming review

12. Draw (in ASCII) a high-level component diagram for a CDN-backed static file server. Include: browser, CDN edge, origin server, object store.

### Trade-offs review

13. Use the trade-off formula to defend: "I chose eventual consistency for the analytics counter rather than strong consistency."

---

## Week Review Checklist

Before taking the assessment, confirm:

- [ ] I completed the URL shortener design document (all 8 sections).
- [ ] I can draw the high-level diagram from memory.
- [ ] I can explain every component and why it exists.
- [ ] I can defend every technology choice with a trade-off statement.
- [ ] I know the latency numbers for RAM, SSD, disk, and network.
- [ ] I can calculate an error budget from an SLO.

---

## Supplementary

- [Designing Data-Intensive Applications (Kleppmann) — Ch. 1: Reliable, Scalable, Maintainable](https://dataintensive.net/)
- [System Design Interview Vol.1 (Alex Xu) — Chapter 1: Scale from Zero](https://bytebytego.com/)
