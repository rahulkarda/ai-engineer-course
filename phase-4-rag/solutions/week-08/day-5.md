# Solutions — Week 08 · Day 5

## Exercise 1: Rate Limiting Middleware

Rate limiting protects the service from abuse and ensures fair resource allocation. The token-bucket algorithm provides bursty allowance while maintaining average rate. Using `os.getenv` for configuration keeps limits adjustable per deployment without code changes.

```python
# src/middleware/rate_limit.py
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
BURST_MULTIPLIER: float = float(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "1.5"))


class TokenBucket:
    """Token-bucket rate limiter for a single key."""

    def __init__(self, rate_per_second: float, burst_capacity: float) -> None:
        self.rate = rate_per_second
        self.capacity = burst_capacity
        self.tokens: float = burst_capacity
        self.last_refill: float = time.monotonic()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """Attempt to consume tokens. Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0

            wait = (tokens - self.tokens) / self.rate
            return False, wait


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP token-bucket rate limiting middleware."""

    def __init__(self, app, rate_per_minute: int | None = None) -> None:
        super().__init__(app)
        rpm = rate_per_minute or RATE_LIMIT_PER_MINUTE
        self._rate_per_second = rpm / 60.0
        self._burst = rpm * BURST_MULTIPLIER / 60.0
        self._buckets: dict[str, TokenBucket] = defaultdict(self._make_bucket)
        logger.info("Rate limiter initialized: %d req/min (burst x%.1f)", rpm, BURST_MULTIPLIER)

    def _make_bucket(self) -> TokenBucket:
        return TokenBucket(self._rate_per_second, self._burst)

    def _get_client_key(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting on health checks to avoid false degraded signals
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)

        key = self._get_client_key(request)
        allowed, retry_after = self._buckets[key].consume()

        if not allowed:
            logger.warning("Rate limit exceeded for client %s; retry_after=%.2fs", key, retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": round(retry_after, 2),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        response = await call_next(request)
        return response
```

```python
# src/main.py (registration snippet)
from fastapi import FastAPI
from src.middleware.rate_limit import RateLimitMiddleware

app = FastAPI()
app.add_middleware(RateLimitMiddleware)
```

---

## Exercise 2: PII Scrubber Tests

Tests verify that every scrubbing path works in isolation (one PII type at a time) and in combination. `pytest.mark.parametrize` avoids repetitive test stubs while keeping failures precise. All external I/O is absent — the scrubber is pure logic.

```python
# tests/unit/test_pii_scrubber.py
from __future__ import annotations

import re

import pytest


class PIIScrubber:
    """Scrub common PII patterns from text before logging or storing."""

    _EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
    _PHONE_RE = re.compile(
        r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    )
    _SSN_RE = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
    _CC_RE = re.compile(r"\b(?:\d{4}[- ]){3}\d{4}\b")

    EMAIL_MASK = "[EMAIL]"
    PHONE_MASK = "[PHONE]"
    SSN_MASK = "[SSN]"
    CC_MASK = "[CC]"

    def scrub(self, text: str) -> str:
        text = self._EMAIL_RE.sub(self.EMAIL_MASK, text)
        text = self._SSN_RE.sub(self.SSN_MASK, text)
        text = self._CC_RE.sub(self.CC_MASK, text)
        text = self._PHONE_RE.sub(self.PHONE_MASK, text)
        return text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def scrubber() -> PIIScrubber:
    return PIIScrubber()


# ---------------------------------------------------------------------------
# Email tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected_contains, should_not_contain",
    [
        ("Contact us at support@example.com for help.", "[EMAIL]", "support@example.com"),
        ("Emails: foo.bar+tag@sub.domain.org and baz@test.io", "[EMAIL]", "@"),
        ("No email here, just text.", "No email here", None),
    ],
)
def test_email_scrubbing(
    scrubber: PIIScrubber,
    text: str,
    expected_contains: str,
    should_not_contain: str | None,
) -> None:
    result = scrubber.scrub(text)
    assert expected_contains in result
    if should_not_contain:
        assert should_not_contain not in result


# ---------------------------------------------------------------------------
# Phone number tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_phone",
    [
        "555-123-4567",
        "(555) 123-4567",
        "+1 555 123 4567",
        "555.123.4567",
        "5551234567",
    ],
)
def test_phone_scrubbing(scrubber: PIIScrubber, raw_phone: str) -> None:
    text = f"Call me at {raw_phone} anytime."
    result = scrubber.scrub(text)
    assert "[PHONE]" in result
    assert raw_phone not in result


# ---------------------------------------------------------------------------
# SSN tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_ssn",
    [
        "123-45-6789",
        "123 45 6789",
    ],
)
def test_ssn_scrubbing(scrubber: PIIScrubber, raw_ssn: str) -> None:
    text = f"SSN on file: {raw_ssn}."
    result = scrubber.scrub(text)
    assert "[SSN]" in result
    assert raw_ssn not in result


# ---------------------------------------------------------------------------
# Combination tests — no PII in output
# ---------------------------------------------------------------------------

def test_no_pii_in_combined_output(scrubber: PIIScrubber) -> None:
    text = (
        "User john.doe@company.com, SSN 111-22-3333, "
        "phone (800) 555-9999 submitted a claim."
    )
    result = scrubber.scrub(text)

    pii_patterns = [
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
    ]
    for pattern in pii_patterns:
        assert not pattern.search(result), f"PII still present matching {pattern.pattern!r}"

    assert "[EMAIL]" in result
    assert "[SSN]" in result
    assert "[PHONE]" in result


def test_clean_text_unchanged(scrubber: PIIScrubber) -> None:
    text = "The document discusses machine learning and retrieval-augmented generation."
    assert scrubber.scrub(text) == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string(scrubber: PIIScrubber) -> None:
    assert scrubber.scrub("") == ""


def test_scrub_is_idempotent(scrubber: PIIScrubber) -> None:
    text = "email: user@domain.com, ssn: 999-88-7777"
    once = scrubber.scrub(text)
    twice = scrubber.scrub(once)
    assert once == twice
```

---

## Exercise 3: /health Endpoint with Downstream Connectivity Checks

The health endpoint drives load-balancer routing: return 503 when any critical dependency is unavailable so traffic stops landing on a broken replica. Async checks run concurrently for minimal latency. Tests mock all network calls so the suite never needs live infrastructure.

```python
# src/routers/health.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import StrEnum

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])

VECTOR_STORE_URL: str = os.getenv("VECTOR_STORE_URL", "http://chromadb:8000")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
HEALTH_TIMEOUT: float = float(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "3.0"))


class ServiceStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceCheck(BaseModel):
    name: str
    status: ServiceStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: ServiceStatus
    services: list[ServiceCheck]
    timestamp: float


async def _check_chromadb(client: httpx.AsyncClient) -> ServiceCheck:
    start = time.monotonic()
    try:
        resp = await client.get(f"{VECTOR_STORE_URL}/api/v1/heartbeat", timeout=HEALTH_TIMEOUT)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return ServiceCheck(name="chromadb", status=ServiceStatus.HEALTHY, latency_ms=round(latency, 2))
        return ServiceCheck(
            name="chromadb",
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency, 2),
            detail=f"HTTP {resp.status_code}",
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        logger.warning("ChromaDB health check failed: %s", exc)
        return ServiceCheck(
            name="chromadb",
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency, 2),
            detail=str(exc),
        )


async def _check_redis() -> ServiceCheck:
    """Ping Redis using a raw socket to avoid a hard redis-py dependency in this module."""
    import asyncio

    start = time.monotonic()
    host = REDIS_URL.replace("redis://", "").split(":")[0]
    port_str = REDIS_URL.replace("redis://", "").split(":")[-1].split("/")[0]
    port = int(port_str) if port_str.isdigit() else 6379

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=HEALTH_TIMEOUT
        )
        writer.write(b"PING\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(128), timeout=HEALTH_TIMEOUT)
        writer.close()
        await writer.wait_closed()
        latency = (time.monotonic() - start) * 1000
        if b"+PONG" in data:
            return ServiceCheck(name="redis", status=ServiceStatus.HEALTHY, latency_ms=round(latency, 2))
        return ServiceCheck(
            name="redis",
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency, 2),
            detail=f"unexpected response: {data[:40]}",
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        logger.warning("Redis health check failed: %s", exc)
        return ServiceCheck(
            name="redis",
            status=ServiceStatus.DEGRADED,
            latency_ms=round(latency, 2),
            detail=str(exc),
        )


@router.get("/health", response_model=HealthResponse, summary="Downstream connectivity check")
async def health_check() -> HealthResponse:
    async with httpx.AsyncClient() as client:
        chromadb_check, redis_check = await asyncio.gather(
            _check_chromadb(client),
            _check_redis(),
        )

    services = [chromadb_check, redis_check]
    all_healthy = all(s.status == ServiceStatus.HEALTHY for s in services)
    overall = ServiceStatus.HEALTHY if all_healthy else ServiceStatus.DEGRADED

    return HealthResponse(
        status=overall,
        services=services,
        timestamp=time.time(),
    )
```

```python
# src/main.py (registration snippet)
from src.routers.health import router as health_router

app.include_router(health_router)
```

Health endpoint returns `200` when all checks pass and `503` when any service is degraded. To make FastAPI return 503 on degraded status, use a dependency or a custom response class:

```python
# src/routers/health.py — add to the endpoint
from fastapi import Response

@router.get("/health")
async def health_check(response: Response) -> HealthResponse:
    ...
    if overall == ServiceStatus.DEGRADED:
        response.status_code = 503
    return HealthResponse(status=overall, services=services, timestamp=time.time())
```

```python
# tests/unit/test_health.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.routers.health import ServiceStatus


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _make_healthy_chromadb() -> AsyncMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=mock_resp)
    return client


@patch("src.routers.health.asyncio.open_connection")
@patch("src.routers.health.httpx.AsyncClient")
def test_health_all_healthy(mock_http_cls, mock_open_conn, client: TestClient) -> None:
    mock_http_cls.return_value = _make_healthy_chromadb()

    reader = AsyncMock()
    reader.read = AsyncMock(return_value=b"+PONG\r\n")
    writer = AsyncMock()
    writer.write = MagicMock()
    mock_open_conn.return_value = (reader, writer)

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == ServiceStatus.HEALTHY
    service_names = {s["name"] for s in body["services"]}
    assert "chromadb" in service_names
    assert "redis" in service_names


@patch("src.routers.health.asyncio.open_connection", side_effect=ConnectionRefusedError("refused"))
@patch("src.routers.health.httpx.AsyncClient")
def test_health_redis_down_returns_503(mock_http_cls, _mock_conn, client: TestClient) -> None:
    mock_http_cls.return_value = _make_healthy_chromadb()

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == ServiceStatus.DEGRADED
    redis_svc = next(s for s in body["services"] if s["name"] == "redis")
    assert redis_svc["status"] == ServiceStatus.DEGRADED
    assert redis_svc["detail"] is not None


@patch("src.routers.health.asyncio.open_connection")
@patch("src.routers.health.httpx.AsyncClient")
def test_health_chromadb_down_returns_503(mock_http_cls, mock_open_conn, client: TestClient) -> None:
    bad_resp = MagicMock()
    bad_resp.status_code = 503
    bad_client = AsyncMock()
    bad_client.__aenter__ = AsyncMock(return_value=bad_client)
    bad_client.__aexit__ = AsyncMock(return_value=False)
    bad_client.get = AsyncMock(return_value=bad_resp)
    mock_http_cls.return_value = bad_client

    reader = AsyncMock()
    reader.read = AsyncMock(return_value=b"+PONG\r\n")
    writer = AsyncMock()
    mock_open_conn.return_value = (reader, writer)

    response = client.get("/health")
    assert response.status_code == 503


@patch("src.routers.health.asyncio.open_connection")
@patch("src.routers.health.httpx.AsyncClient")
def test_health_response_schema(mock_http_cls, mock_open_conn, client: TestClient) -> None:
    mock_http_cls.return_value = _make_healthy_chromadb()
    reader = AsyncMock()
    reader.read = AsyncMock(return_value=b"+PONG\r\n")
    writer = AsyncMock()
    mock_open_conn.return_value = (reader, writer)

    response = client.get("/health")
    body = response.json()

    assert "status" in body
    assert "services" in body
    assert "timestamp" in body
    assert isinstance(body["timestamp"], float)
    for svc in body["services"]:
        assert "name" in svc
        assert "status" in svc
```

---

## Key concepts demonstrated

- Token-bucket algorithm allows short bursts while enforcing long-run average rate — more realistic than a sliding-window counter for API traffic
- Middleware placement: `add_middleware` wraps all routes; health/metrics paths are exempted to avoid masking real degradation
- `Retry-After` header is an HTTP standard; clients that respect it back off gracefully instead of hammering a 429 loop
- PII scrubber regex order matters — SSN before phone to avoid partial digit matches
- `pytest.mark.parametrize` captures the full input space without copy-paste tests; idempotency test catches double-encode bugs
- Health checks run concurrently via `asyncio.gather` — total latency is max(individual), not sum
- `ServiceStatus.DEGRADED` triggers HTTP 503 so load balancers route away from the broken replica automatically
- All external I/O (HTTP, socket) is mocked in tests — the suite runs offline and fast
