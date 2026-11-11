# Solutions — Week 08 · Day 5

## Exercise 1 — read_mcp_resource + test

```python
# In mcp_agent.py

async def read_mcp_resource(session: ClientSession, uri: str) -> str:
    """Read an MCP resource by URI and return its text content."""
    try:
        result = await session.read_resource(uri)
        texts = []
        for content_item in result.contents:
            if hasattr(content_item, "text") and content_item.text:
                texts.append(content_item.text)
            elif hasattr(content_item, "blob"):
                texts.append(f"[binary content, {len(content_item.blob)} bytes]")
        return "\n".join(texts) if texts else "[empty resource]"
    except Exception as exc:
        logger.warning("Failed to read resource %s: %s", uri, exc)
        return f"Error reading resource {uri!r}: {exc}"
```

```python
# tests/test_read_resource.py
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_read_resource_returns_text():
    from mcp_agent import read_mcp_resource

    content = MagicMock()
    content.text = "Resource text content"
    result_mock = MagicMock()
    result_mock.contents = [content]

    session = AsyncMock()
    session.read_resource.return_value = result_mock

    text = await read_mcp_resource(session, "kb://articles/mcp-intro")
    assert "Resource text content" in text
    session.read_resource.assert_called_once_with("kb://articles/mcp-intro")


@pytest.mark.asyncio
async def test_read_resource_handles_error():
    from mcp_agent import read_mcp_resource

    session = AsyncMock()
    session.read_resource.side_effect = Exception("URI not found")

    text = await read_mcp_resource(session, "kb://articles/nonexistent")
    assert "Error" in text
    assert "nonexistent" in text
```

## Exercise 2 — safe_file_path

```python
# security_utils.py
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_file_path(requested: str, allowed_root: str) -> str:
    """Resolve a requested path and ensure it stays within allowed_root.

    Args:
        requested: The path requested (may be relative or contain ..)
        allowed_root: The directory the resolved path must be inside.

    Returns:
        The resolved absolute path string.

    Raises:
        ValueError: If the resolved path escapes allowed_root.
    """
    root = Path(allowed_root).resolve()
    resolved = (root / requested).resolve()

    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{requested}' resolves to '{resolved}', "
            f"which is outside allowed root '{root}'."
        )

    logger.debug("Safe path: %s → %s", requested, resolved)
    return str(resolved)
```

```python
# tests/test_security_utils.py
import pytest
import tempfile
import os
from pathlib import Path
from security_utils import safe_file_path


@pytest.fixture
def tmp_root(tmp_path):
    return str(tmp_path)


def test_normal_path_allowed(tmp_root):
    result = safe_file_path("subdir/file.txt", tmp_root)
    assert result.startswith(tmp_root)
    assert "file.txt" in result


def test_traversal_blocked(tmp_root):
    with pytest.raises(ValueError, match="traversal"):
        safe_file_path("../../etc/passwd", tmp_root)


def test_absolute_path_inside_root(tmp_root):
    # Absolute path that IS inside the root should be allowed
    allowed = os.path.join(tmp_root, "file.txt")
    result = safe_file_path(allowed, tmp_root)
    assert result == allowed


def test_dot_notation_within_root(tmp_root):
    result = safe_file_path("./subdir/../other.txt", tmp_root)
    assert result.startswith(tmp_root)


def test_path_to_root_itself_allowed(tmp_root):
    result = safe_file_path(".", tmp_root)
    assert result == str(Path(tmp_root).resolve())
```

## Exercise 3 — @audit_log decorator

```python
# audit_log.py
import asyncio
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("audit")

_SENSITIVE_KEYWORDS = {"key", "token", "secret", "password", "credential"}


def _sanitize_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace values of sensitive-named keys with '<redacted>'."""
    return {
        k: "<redacted>" if any(kw in k.lower() for kw in _SENSITIVE_KEYWORDS) else v
        for k, v in arguments.items()
    }


def audit_log(func: Callable) -> Callable:
    """Decorator: log tool_name, sanitized args, result_is_error, duration_ms."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        tool_name = func.__name__
        sanitized = _sanitize_args(kwargs)
        start = time.monotonic()
        is_error = False
        try:
            result = await func(*args, **kwargs)
            # Check if result is an MCP-style dict with isError
            if isinstance(result, dict) and result.get("isError"):
                is_error = True
            return result
        except Exception:
            is_error = True
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            logger.info(
                "tool=%s args=%s is_error=%s duration_ms=%.1f",
                tool_name,
                sanitized,
                is_error,
                duration_ms,
            )

    return wrapper
```

```python
# tests/test_audit_log.py
import pytest
import logging
from unittest.mock import AsyncMock
from audit_log import audit_log, _sanitize_args


def test_sanitize_hides_token():
    args = {"query": "hello", "api_token": "secret123", "user": "alice"}
    result = _sanitize_args(args)
    assert result["query"] == "hello"
    assert result["api_token"] == "<redacted>"
    assert result["user"] == "alice"


def test_sanitize_hides_key():
    args = {"api_key": "sk-abc", "data": "value"}
    result = _sanitize_args(args)
    assert result["api_key"] == "<redacted>"
    assert result["data"] == "value"


@pytest.mark.asyncio
async def test_audit_log_calls_wrapped_function():
    calls = []

    @audit_log
    async def my_tool(query: str) -> str:
        calls.append(query)
        return f"result:{query}"

    result = await my_tool(query="hello")
    assert result == "result:hello"
    assert "hello" in calls


@pytest.mark.asyncio
async def test_audit_log_captures_error(caplog):
    @audit_log
    async def failing_tool(data: str) -> dict:
        return {"isError": True, "content": []}

    with caplog.at_level(logging.INFO, logger="audit"):
        await failing_tool(data="test")

    assert "is_error=True" in caplog.text


@pytest.mark.asyncio
async def test_audit_log_does_not_log_secret(caplog):
    @audit_log
    async def secure_tool(query: str, api_key: str) -> str:
        return "ok"

    with caplog.at_level(logging.INFO, logger="audit"):
        await secure_tool(query="search", api_key="sk-real-key")

    assert "sk-real-key" not in caplog.text
    assert "<redacted>" in caplog.text
```

## Exercise 4 — RateLimiter

```python
# rate_limiter.py
import asyncio
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter per client_id."""

    def __init__(self, calls_per_minute: int) -> None:
        self._calls_per_minute = calls_per_minute
        self._window_seconds = 60.0
        self._history: dict[str, deque[float]] = defaultdict(deque)

    async def check(self, client_id: str) -> bool:
        """Return True if the call is allowed; raise PermissionError if rate exceeded."""
        now = time.monotonic()
        window_start = now - self._window_seconds
        history = self._history[client_id]

        # Remove expired entries
        while history and history[0] < window_start:
            history.popleft()

        if len(history) >= self._calls_per_minute:
            raise PermissionError(
                f"Rate limit exceeded for client '{client_id}': "
                f"{self._calls_per_minute} calls/minute."
            )

        history.append(now)
        logger.debug("client=%s calls_in_window=%d", client_id, len(history))
        return True
```

```python
# tests/test_rate_limiter.py
import pytest
import time
from unittest.mock import patch
from rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_under_limit_passes():
    limiter = RateLimiter(calls_per_minute=5)
    for _ in range(5):
        assert await limiter.check("client-a")


@pytest.mark.asyncio
async def test_over_limit_raises():
    limiter = RateLimiter(calls_per_minute=3)
    for _ in range(3):
        await limiter.check("client-a")
    with pytest.raises(PermissionError, match="Rate limit"):
        await limiter.check("client-a")


@pytest.mark.asyncio
async def test_different_clients_are_independent():
    limiter = RateLimiter(calls_per_minute=2)
    await limiter.check("client-a")
    await limiter.check("client-a")
    # client-a is now at limit; client-b should still be allowed
    assert await limiter.check("client-b")


@pytest.mark.asyncio
async def test_window_expires_allows_new_calls():
    limiter = RateLimiter(calls_per_minute=2)

    # Fill the window with calls at t=0
    with patch("time.monotonic", return_value=0.0):
        await limiter.check("client-a")
        await limiter.check("client-a")

    # 61 seconds later — window has expired
    with patch("time.monotonic", return_value=61.0):
        # Should be allowed again
        assert await limiter.check("client-a")
```
