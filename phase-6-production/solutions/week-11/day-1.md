# Week 11 · Day 1 Solutions

## Exercise 1

Add log sampling to `CorrelationMiddleware` — log only 10% of successful requests (status < 400) but always log errors.

```python
import logging
import random
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import FastAPI, Request, Response

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

logger = logging.getLogger(__name__)


class CorrelationMiddleware:
    """ASGI middleware that attaches a correlation ID to every request
    and applies log sampling: 10% of successful requests, 100% of errors."""

    SAMPLE_RATE: float = 0.10  # 10% of successful requests

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(
        self, scope: dict, receive: Callable, send: Callable
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)

        start = time.perf_counter()
        status_code = 500  # default in case of unhandled exception

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._emit_log(
                request=request,
                status_code=status_code,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            )

    def _emit_log(
        self,
        request: Request,
        status_code: int,
        duration_ms: float,
        correlation_id: str,
    ) -> None:
        is_error = status_code >= 400

        # Always log errors; sample 10% of successes
        if not is_error and random.random() >= self.SAMPLE_RATE:
            return

        level = logging.ERROR if is_error else logging.INFO
        logger.log(
            level,
            "request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "sampled": not is_error,  # True means this was a sampled success
            },
        )


# --- minimal smoke-test (run directly) ---
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = FastAPI()
    app.add_middleware(CorrelationMiddleware)

    @app.get("/ok")
    async def ok() -> dict:
        return {"status": "ok"}

    @app.get("/fail")
    async def fail() -> Response:
        return Response(status_code=500)

    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

## Exercise 2

Propagate `correlation_id` to a downstream HTTP call using `httpx`, adding it as `X-Correlation-ID` header.

```python
import logging
import uuid
from contextvars import ContextVar
from typing import Any

import httpx
from fastapi import FastAPI, Request

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

logger = logging.getLogger(__name__)


async def call_downstream(
    url: str,
    *,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> httpx.Response:
    """Make an outbound HTTP request, forwarding the active correlation ID."""
    correlation_id = correlation_id_var.get()
    headers: dict[str, str] = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            json=json,
            headers=headers,
        )

    logger.info(
        "downstream call completed",
        extra={
            "correlation_id": correlation_id,
            "downstream_url": url,
            "downstream_status": response.status_code,
        },
    )
    return response


# --- FastAPI integration example ---
app = FastAPI()


class CorrelationMiddleware:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            request = Request(scope, receive)
            cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
            correlation_id_var.set(cid)
        await self.app(scope, receive, send)


app.add_middleware(CorrelationMiddleware)


@app.get("/proxy")
async def proxy_handler() -> dict:
    # correlation_id automatically injected into downstream request
    response = await call_downstream("https://httpbin.org/get")
    return {"upstream_status": response.status_code}
```

---

## Exercise 3

Read `LOG_LEVEL` from environment with fallback to `"INFO"`, validate it is a recognised Python level.

```python
import logging
import os
import sys


_VALID_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


def get_log_level() -> int:
    """Return a validated logging level integer from the LOG_LEVEL env var.

    Defaults to INFO.  Exits with a clear message if the value is invalid.
    """
    raw = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    if raw not in _VALID_LEVELS:
        print(  # noqa: T201 — intentional startup error before logging is configured
            f"Invalid LOG_LEVEL={raw!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_LEVELS))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return getattr(logging, raw)


def configure_logging() -> None:
    """Configure root logger using the environment-controlled level."""
    level = get_log_level()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger(__name__).debug(
        "logging configured", extra={"level": logging.getLevelName(level)}
    )


# Usage: call configure_logging() once at application startup.
if __name__ == "__main__":
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.debug("debug message — only visible when LOG_LEVEL=DEBUG")
    logger.info("info message")
    logger.warning("warning message")
```

---

## Exercise 4

Log a `WARNING` when request duration exceeds `SLOW_REQUEST_THRESHOLD_MS` (default 500 ms).

```python
import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable

from fastapi import FastAPI, Request

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

logger = logging.getLogger(__name__)


def _get_slow_threshold() -> float:
    """Read SLOW_REQUEST_THRESHOLD_MS from env, default 500 ms."""
    raw = os.getenv("SLOW_REQUEST_THRESHOLD_MS", "500")
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError("threshold must be positive")
        return value
    except ValueError:
        logger.warning(
            "Invalid SLOW_REQUEST_THRESHOLD_MS=%r, using default 500 ms", raw
        )
        return 500.0


# Read once at import time; re-read at each request if you need live reload.
_SLOW_THRESHOLD_MS: float = _get_slow_threshold()


class CorrelationMiddleware:
    """Middleware that tracks duration and warns on slow requests."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(
        self, scope: dict, receive: Callable, send: Callable
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            extra: dict[str, Any] = {
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            }

            if duration_ms > _SLOW_THRESHOLD_MS:
                logger.warning(
                    "slow request detected: %.1f ms > threshold %.0f ms",
                    duration_ms,
                    _SLOW_THRESHOLD_MS,
                    extra=extra,
                )
            else:
                logger.info("request completed", extra=extra)
```
