# Week 11 · Day 2 Solutions

## Exercise 1

Add request body logging to `LoggingMiddleware`: POST/PUT only, truncate at 1000 chars, sanitize sensitive keys.

```python
import json
import logging
from typing import Any

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {"password", "token", "secret", "api_key"}
)
_BODY_LOG_METHODS: frozenset[str] = frozenset({"POST", "PUT"})
_BODY_MAX_LEN: int = 1000


def _sanitize(data: Any, sensitive_keys: frozenset[str] = _SENSITIVE_KEYS) -> Any:
    """Recursively replace sensitive key values with '[REDACTED]'."""
    if isinstance(data, dict):
        return {
            k: "[REDACTED]" if k.lower() in sensitive_keys else _sanitize(v, sensitive_keys)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_sanitize(item, sensitive_keys) for item in data]
    return data


def _truncate(text: str, max_len: int = _BODY_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated {len(text) - max_len} chars]"


class LoggingMiddleware:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        if request.method in _BODY_LOG_METHODS:
            await self._log_request_body(request)

        await self.app(scope, receive, send)

    async def _log_request_body(self, request: Request) -> None:
        try:
            raw_body = await request.body()
            if not raw_body:
                return

            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                parsed: Any = json.loads(raw_body)
                sanitized = _sanitize(parsed)
                body_str = _truncate(json.dumps(sanitized))
            else:
                body_str = _truncate(raw_body.decode("utf-8", errors="replace"))

            logger.info(
                "request body",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "body": body_str,
                },
            )
        except Exception:
            logger.debug("could not read request body", exc_info=True)


# --- unit tests ---
import pytest


def test_sanitize_removes_sensitive_keys() -> None:
    data = {"username": "alice", "password": "hunter2", "token": "abc123"}
    result = _sanitize(data)
    assert result["username"] == "alice"
    assert result["password"] == "[REDACTED]"
    assert result["token"] == "[REDACTED]"


def test_sanitize_nested() -> None:
    data = {"user": {"api_key": "key", "name": "bob"}}
    result = _sanitize(data)
    assert result["user"]["api_key"] == "[REDACTED]"
    assert result["user"]["name"] == "bob"


def test_truncate() -> None:
    long_str = "x" * 1200
    result = _truncate(long_str, max_len=1000)
    assert len(result) < 1200
    assert "truncated" in result


def test_truncate_short_string_unchanged() -> None:
    s = "hello"
    assert _truncate(s) == s
```

---

## Exercise 2

Rate-limited log filter: max N records/second per logger name; drop excess with a counter.

```python
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar


@dataclass
class _BucketState:
    count: int = 0
    dropped: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimitedFilter(logging.Filter):
    """Drop log records that exceed `max_per_second` per logger name.

    A token-bucket style 1-second sliding window.  Dropped records are
    counted; on the next accepted record a summary is appended to `msg`.
    """

    _DEFAULT_MAX: ClassVar[int] = 10

    def __init__(self, max_per_second: int = _DEFAULT_MAX) -> None:
        super().__init__()
        if max_per_second < 1:
            raise ValueError("max_per_second must be >= 1")
        self._max = max_per_second
        self._buckets: dict[str, _BucketState] = defaultdict(_BucketState)
        self._lock = Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets[record.name]
            if now - bucket.window_start >= 1.0:
                # New window — reset counters
                dropped = bucket.dropped
                bucket.count = 0
                bucket.dropped = 0
                bucket.window_start = now
                if dropped:
                    record.msg = f"{record.msg} [rate limiter dropped {dropped} records in previous window]"

            if bucket.count < self._max:
                bucket.count += 1
                return True

            bucket.dropped += 1
            return False


# --- demonstration ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    limiter = RateLimitedFilter(max_per_second=3)
    root = logging.getLogger()
    root.addFilter(limiter)

    logger = logging.getLogger("demo")
    for i in range(10):
        logger.info("message %d", i)
    # Only first 3 pass; remaining 7 are dropped.


# --- unit tests ---
import pytest


def test_allows_up_to_max() -> None:
    f = RateLimitedFilter(max_per_second=3)
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    results = [f.filter(record) for _ in range(5)]
    assert results[:3] == [True, True, True]
    assert results[3:] == [False, False]


def test_rejects_invalid_max() -> None:
    with pytest.raises(ValueError):
        RateLimitedFilter(max_per_second=0)


def test_different_loggers_have_separate_buckets() -> None:
    f = RateLimitedFilter(max_per_second=1)
    r1 = logging.LogRecord("logger_a", logging.INFO, "", 0, "msg", (), None)
    r2 = logging.LogRecord("logger_b", logging.INFO, "", 0, "msg", (), None)
    assert f.filter(r1) is True
    assert f.filter(r2) is True  # separate bucket — not dropped
    assert f.filter(r1) is False  # same logger, limit reached
```

---

## Exercise 3

Child loggers inherit `correlation_id` from `contextvars` without explicit passing — three nested callers each use `getLogger(__name__)`.

```python
"""
Demonstrates that contextvars.ContextVar values are visible to all code in
the same execution context, so nested callers can each use their own
module-level logger and still emit the correlation ID set by the entry point.
"""
import logging
import uuid
from contextvars import ContextVar

# Shared ContextVar — typically lives in a dedicated `context.py` module.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationFilter(logging.Filter):
    """Inject the active correlation_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.correlation_id = correlation_id_var.get() or "-"
        return True


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s %(message)s"
        )
    )
    handler.addFilter(CorrelationFilter())
    logging.root.setLevel(logging.DEBUG)
    logging.root.handlers = [handler]


# --- three separate "modules" each with their own logger ---

# module_a.py
_logger_a = logging.getLogger("app.module_a")


def process_a() -> str:
    _logger_a.info("process_a: starting")
    result = process_b()
    _logger_a.info("process_a: finished, got %r", result)
    return f"a({result})"


# module_b.py
_logger_b = logging.getLogger("app.module_b")


def process_b() -> str:
    _logger_b.info("process_b: doing work")
    inner = process_c()
    return f"b({inner})"


# module_c.py
_logger_c = logging.getLogger("app.module_c")


def process_c() -> str:
    _logger_c.info("process_c: leaf call")
    return "c"


# entry point
if __name__ == "__main__":
    _setup_logging()

    cid = str(uuid.uuid4())
    correlation_id_var.set(cid)

    root_logger = logging.getLogger(__name__)
    root_logger.info("request started, correlation_id=%s", cid)

    output = process_a()
    root_logger.info("request completed, output=%s", output)
    # All log lines — from __main__, module_a, module_b, module_c — share the
    # same correlation_id because they all read the same ContextVar.
```

---

## Exercise 4

PII sanitization filter: redact email addresses and phone numbers from log messages using regex.

```python
import logging
import re


# Patterns — deliberately conservative to avoid false positives in prod.
_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Covers: +1-800-555-0199, (800) 555-0199, 800.555.0199, 8005550199 etc.
_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}"
)

_EMAIL_REPLACEMENT = "[EMAIL REDACTED]"
_PHONE_REPLACEMENT = "[PHONE REDACTED]"


class PIISanitizationFilter(logging.Filter):
    """Redact email addresses and phone numbers from log record messages.

    Works on `record.msg` (before %-formatting) and also on `record.getMessage()`
    output stored back into `record.msg` to handle pre-formatted strings.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            # Sanitize the already-formatted message string
            message = record.getMessage()
            sanitized = _EMAIL_PATTERN.sub(_EMAIL_REPLACEMENT, message)
            sanitized = _PHONE_PATTERN.sub(_PHONE_REPLACEMENT, sanitized)

            # Replace msg with the sanitized version and clear args so
            # LogRecord.getMessage() won't re-apply % formatting.
            record.msg = sanitized
            record.args = ()
        except Exception:  # noqa: BLE001
            pass  # never suppress a log record because of a filter bug
        return True


# --- unit tests ---
import pytest


@pytest.fixture()
def pii_filter() -> PIISanitizationFilter:
    return PIISanitizationFilter()


def _make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, "", 0, msg, (), None)


def test_email_redacted(pii_filter: PIISanitizationFilter) -> None:
    record = _make_record("User alice@example.com logged in")
    pii_filter.filter(record)
    assert "alice@example.com" not in record.msg
    assert _EMAIL_REPLACEMENT in record.msg
    assert "User" in record.msg
    assert "logged in" in record.msg


def test_phone_redacted(pii_filter: PIISanitizationFilter) -> None:
    record = _make_record("Contact us at 800-555-0199 for support")
    pii_filter.filter(record)
    assert "800-555-0199" not in record.msg
    assert _PHONE_REPLACEMENT in record.msg


def test_multiple_emails_all_redacted(pii_filter: PIISanitizationFilter) -> None:
    record = _make_record("Sent to alice@a.com and bob@b.org")
    pii_filter.filter(record)
    assert "alice@a.com" not in record.msg
    assert "bob@b.org" not in record.msg
    assert record.msg.count(_EMAIL_REPLACEMENT) == 2


def test_no_pii_unchanged(pii_filter: PIISanitizationFilter) -> None:
    msg = "Normal log message without PII"
    record = _make_record(msg)
    pii_filter.filter(record)
    assert record.msg == msg


def test_filter_always_returns_true(pii_filter: PIISanitizationFilter) -> None:
    record = _make_record("any message")
    assert pii_filter.filter(record) is True
```
