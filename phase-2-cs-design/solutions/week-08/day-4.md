# Solutions — Week 08 · Day 4

## Exercise 1

**For each scenario, choose sync REST, gRPC, or async messaging and justify.**

### (a) User service calling payment service to process checkout
**Choice: Synchronous REST (or gRPC)**

Justification: Checkout requires an immediate, authoritative result. The user is waiting; the UI must show "success" or "failure" before the flow continues. Payment state must be known before releasing inventory or confirming the order. Async messaging would require polling or webhooks and introduces a window where the user doesn't know if they were charged.

gRPC is preferable over REST if both services are internal and latency matters (binary framing, HTTP/2 multiplexing), but REST is acceptable for simpler stacks.

### (b) Order service notifying email service after order placed
**Choice: Async messaging (message broker — Kafka, RabbitMQ, SQS)**

Justification: Email delivery is a side effect; the order is already committed. The order service should not block on email. Failures in email sending must not roll back the order. Async decouples the two services and allows the email service to retry independently. This is a classic fire-and-forget notification pattern.

### (c) Real-time chat messages between users
**Choice: Async messaging (with WebSocket delivery at the edge)**

Justification: Chat messages are inherently async — the sender should not block waiting for the recipient to "receive" the message. A message broker (Kafka, Redis Streams, or NATS) stores and fans out messages; an edge service delivers them over persistent WebSocket connections. gRPC bidirectional streaming is viable for service-to-service hops but the fan-out pattern still needs a broker.

### (d) Inventory service querying product catalog for current stock
**Choice: Synchronous REST or gRPC**

Justification: Stock availability is a point-in-time read used to build a response (e.g., "add to cart" button state). The caller needs the answer now. gRPC is preferable if the catalog service is internal and the query is frequent — schema stability via protobuf and lower overhead. REST is fine for lower-frequency reads or when HTTP caching (ETags, Cache-Control) on stock data is useful.

---

## Exercise 2

```python
"""
inter_service_client.py

Demonstrates sync HTTP client with timeout, exponential-backoff retry,
and a simple in-process circuit breaker.
"""
import logging
import time
from functools import wraps
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker (minimal in-process implementation)
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Simple closed/open circuit breaker.

    States:
      - CLOSED   : requests flow through normally
      - OPEN     : requests are rejected immediately after threshold failures
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._consecutive_failures: int = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._recovery_timeout:
            # Enter HALF-OPEN: allow one probe
            logger.info("CircuitBreaker: recovery timeout elapsed, resetting to CLOSED")
            self._consecutive_failures = 0
            self._opened_at = None
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        logger.warning(
            "CircuitBreaker: consecutive failures=%d threshold=%d",
            self._consecutive_failures,
            self._failure_threshold,
        )
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = time.monotonic()
            logger.error(
                "CircuitBreaker: OPEN — skipping calls for %.0f seconds",
                self._recovery_timeout,
            )


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(
    fn,
    *args,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    **kwargs,
) -> Any:
    """
    Call fn(*args, **kwargs) up to max_attempts times.
    Backoff: base_delay * 2^attempt (0.5s, 1s, 2s).
    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Service client
# ---------------------------------------------------------------------------

CATALOG_BASE_URL = "http://catalog-service:8001"
_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)


def _fetch_product(product_id: int) -> dict:
    """Single HTTP call — used inside call_with_retry."""
    with httpx.Client(timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)) as client:
        response = client.get(f"{CATALOG_BASE_URL}/api/v1/products/{product_id}")
        response.raise_for_status()
        return response.json()


def get_product(product_id: int) -> dict:
    """
    Fetch a product from the catalog service.

    Applies circuit breaker check first, then retries with backoff.
    """
    if _circuit_breaker.is_open:
        logger.error("CircuitBreaker OPEN: skipping call to catalog-service")
        raise HTTPException(status_code=503, detail="Catalog service unavailable (circuit open)")

    try:
        result = call_with_retry(_fetch_product, product_id, max_attempts=3, base_delay=0.5)
        _circuit_breaker.record_success()
        return result
    except Exception as exc:
        _circuit_breaker.record_failure()
        logger.error("get_product failed after retries: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Order Service")


@app.get("/api/v1/orders/{order_id}/product")
async def get_order_product(order_id: int, product_id: int) -> dict:
    """
    Demo endpoint: fetch product details for an order from the catalog service.
    """
    product = get_product(product_id)
    return {"order_id": order_id, "product": product}
```

---

## Exercise 3

```python
"""
gateway_config.py

Pydantic v2 models for an API gateway configuration covering routing,
per-client / per-route rate limiting, and auth passthrough.
"""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    ANY = "*"


class AuthScheme(StrEnum):
    BEARER = "bearer"
    API_KEY = "api_key"
    NONE = "none"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimit(BaseModel):
    """Sliding-window rate limit bucket."""

    requests_per_minute: Annotated[int, Field(gt=0, description="Max requests per 60-second window")]
    burst: Annotated[int, Field(gt=0, description="Instantaneous burst allowance")] = 10

    @model_validator(mode="after")
    def burst_must_not_exceed_rpm(self) -> RateLimit:
        if self.burst > self.requests_per_minute:
            raise ValueError("burst cannot exceed requests_per_minute")
        return self


class PerClientRateLimit(BaseModel):
    """Rate limits keyed by client-id (e.g., API key prefix or JWT sub claim)."""

    client_id: str
    limit: RateLimit


class PerRouteRateLimit(BaseModel):
    """Rate limit applied to a specific route path regardless of caller."""

    path_pattern: str  # regex
    limit: RateLimit

    @field_validator("path_pattern")
    @classmethod
    def validate_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex '{v}': {exc}") from exc
        return v


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class AuthConfig(BaseModel):
    """Auth passthrough: gateway validates token, then forwards identity header downstream."""

    scheme: AuthScheme = AuthScheme.BEARER
    jwks_uri: str | None = None              # for bearer JWT validation
    api_key_header: str | None = "X-Api-Key" # for api_key scheme
    passthrough_header: str = "X-Authenticated-User"
    skip_paths: list[str] = Field(default_factory=list, description="Paths that bypass auth")

    @model_validator(mode="after")
    def validate_scheme_config(self) -> AuthConfig:
        if self.scheme == AuthScheme.BEARER and not self.jwks_uri:
            raise ValueError("jwks_uri required for bearer scheme")
        return self


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class Upstream(BaseModel):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)]
    scheme: str = "http"
    health_check_path: str = "/health"
    timeout_seconds: float = 10.0
    strip_prefix: bool = False  # strip the gateway prefix before forwarding

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


class Route(BaseModel):
    name: str
    path_prefix: str          # e.g. "/api/v1/catalog"
    methods: list[HttpMethod] = [HttpMethod.ANY]
    upstream: Upstream
    per_route_rate_limit: PerRouteRateLimit | None = None
    auth_required: bool = True


# ---------------------------------------------------------------------------
# Top-level gateway config
# ---------------------------------------------------------------------------

class GatewayConfig(BaseModel):
    """
    Full API gateway configuration for a 3-service system.

    Services:
      - catalog-service  (products)
      - order-service    (orders)
      - user-service     (accounts / identity)
    """

    listen_port: Annotated[int, Field(ge=1, le=65535)] = 8080
    auth: AuthConfig
    global_rate_limit: RateLimit = RateLimit(requests_per_minute=1000, burst=50)
    per_client_limits: list[PerClientRateLimit] = Field(default_factory=list)
    routes: list[Route]

    @field_validator("routes")
    @classmethod
    def no_duplicate_prefixes(cls, routes: list[Route]) -> list[Route]:
        prefixes = [r.path_prefix for r in routes]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("Duplicate path prefixes in routes")
        return routes


# ---------------------------------------------------------------------------
# Example instantiation — 3 services
# ---------------------------------------------------------------------------

EXAMPLE_GATEWAY_CONFIG = GatewayConfig(
    listen_port=8080,
    auth=AuthConfig(
        scheme=AuthScheme.BEARER,
        jwks_uri="https://auth.example.com/.well-known/jwks.json",
        passthrough_header="X-Authenticated-User",
        skip_paths=["/health", "/metrics"],
    ),
    global_rate_limit=RateLimit(requests_per_minute=2000, burst=100),
    per_client_limits=[
        PerClientRateLimit(
            client_id="partner-abc",
            limit=RateLimit(requests_per_minute=500, burst=20),
        ),
        PerClientRateLimit(
            client_id="internal-dashboard",
            limit=RateLimit(requests_per_minute=5000, burst=200),
        ),
    ],
    routes=[
        Route(
            name="catalog",
            path_prefix="/api/v1/catalog",
            methods=[HttpMethod.GET],
            upstream=Upstream(host="catalog-service", port=8001, strip_prefix=True),
            per_route_rate_limit=PerRouteRateLimit(
                path_pattern=r"^/api/v1/catalog/.*",
                limit=RateLimit(requests_per_minute=3000, burst=150),
            ),
            auth_required=True,
        ),
        Route(
            name="orders",
            path_prefix="/api/v1/orders",
            methods=[HttpMethod.GET, HttpMethod.POST],
            upstream=Upstream(host="order-service", port=8002, strip_prefix=True),
            per_route_rate_limit=PerRouteRateLimit(
                path_pattern=r"^/api/v1/orders/.*",
                limit=RateLimit(requests_per_minute=500, burst=30),
            ),
            auth_required=True,
        ),
        Route(
            name="users",
            path_prefix="/api/v1/users",
            methods=[HttpMethod.GET, HttpMethod.PUT, HttpMethod.PATCH],
            upstream=Upstream(host="user-service", port=8003, strip_prefix=True),
            auth_required=True,
        ),
    ],
)
```

---

## Exercise 4

### At-Least-Once vs Exactly-Once Delivery

**At-least-once:** The broker guarantees every message will eventually be delivered to the consumer, but the same message may arrive more than once (on retry after a timeout, network blip, or consumer crash before ack).

**Exactly-once:** Each message is processed precisely one time — no duplicates, no gaps. Requires coordination between producer, broker, and consumer (typically via transactional APIs or two-phase commit). Higher cost and complexity; supported by Kafka transactions and some SQS+DLQ patterns.

---

### Concrete bug from at-least-once

**Scenario:** A payment service consumes a "charge customer" event from a queue. After it charges the card and before it acknowledges the message, the consumer crashes. The broker retries — the customer is charged twice.

```
[Broker] -- "charge $99" --> [Payment Service]
                              ... charges card (success) ...
                              ... crashes before ack ...
[Broker] -- "charge $99" --> [Payment Service]  <-- second delivery
                              ... charges card AGAIN (duplicate charge!) ...
```

**Fix: Idempotency key**

```python
import logging
import os
from datetime import datetime

import psycopg2

logger = logging.getLogger(__name__)

DB_DSN = os.getenv("DB_DSN")


def process_charge_event(event: dict) -> None:
    """
    Idempotent charge handler.

    The event carries a stable idempotency_key (e.g., order_id + attempt_id).
    We INSERT ... ON CONFLICT DO NOTHING into a processed_events table before
    touching the payment provider. If the row already exists, we know this is
    a duplicate and skip.
    """
    idempotency_key: str = event["idempotency_key"]
    amount_cents: int = event["amount_cents"]
    customer_id: str = event["customer_id"]

    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processed_events (idempotency_key, processed_at)
                VALUES (%s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (idempotency_key, datetime.utcnow()),
            )
            rows_inserted = cur.rowcount

        if rows_inserted == 0:
            logger.info(
                "Duplicate event detected, skipping charge. key=%s", idempotency_key
            )
            return

        # Only reaches here on first delivery
        _charge_customer(customer_id, amount_cents)
        conn.commit()
        logger.info("Charge processed. key=%s customer=%s", idempotency_key, customer_id)


def _charge_customer(customer_id: str, amount_cents: int) -> None:
    """Stub — calls payment provider SDK."""
    logger.info("Charging customer %s for %d cents", customer_id, amount_cents)
```

**Key rules for idempotency:**
1. The idempotency key must be stable across retries (generated by the producer, not the consumer).
2. The deduplication check and the side effect must be in the same transaction.
3. Idempotency store TTL should exceed the maximum message retention window.
