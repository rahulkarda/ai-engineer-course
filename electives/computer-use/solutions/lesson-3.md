# Solutions: Lesson 3 — Browser Automation with Playwright

## Exercise 1 — Accessibility tree tool

```python
async def get_accessibility_snapshot(self) -> list[dict]:
    """Return the accessibility tree as a flat list of node dicts."""
    # Playwright returns a nested dict; we pass it through as-is
    # In production: process/flatten as needed for the LLM context
    snapshot = await self._page.accessibility.snapshot()
    if snapshot is None:
        return []
    return [snapshot]  # or flatten recursively


# Test
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_get_accessibility_snapshot():
    page = MagicMock()
    sample_tree = {
        "role": "WebArea",
        "name": "Example",
        "children": [{"role": "button", "name": "Submit"}],
    }
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=sample_tree)

    tools = BrowserTools(page)
    result = await tools.get_accessibility_snapshot()
    assert result == [sample_tree]

@pytest.mark.asyncio
async def test_get_accessibility_snapshot_none():
    page = MagicMock()
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=None)
    tools = BrowserTools(page)
    result = await tools.get_accessibility_snapshot()
    assert result == []
```

---

## Exercise 2 — Network interception

```python
async def block_external_requests(self, allowed_hosts: list[str]) -> None:
    """
    Block all requests to hosts not in allowed_hosts.
    This is a key sandboxing technique — prevents data exfiltration and
    reduces the attack surface of a computer-use agent.
    """
    import re

    async def route_handler(route, request) -> None:
        from urllib.parse import urlparse
        host = urlparse(request.url).hostname or ""
        if any(host == allowed or host.endswith(f".{allowed}") for allowed in allowed_hosts):
            await route.continue_()
        else:
            await route.abort("blockedbyclient")

    await self._page.route("**/*", route_handler)


# Test
@pytest.mark.asyncio
async def test_block_external_requests_registers_route():
    page = MagicMock()
    page.route = AsyncMock()
    tools = BrowserTools(page)
    await tools.block_external_requests(["example.com"])
    page.route.assert_called_once()
    args = page.route.call_args
    assert args[0][0] == "**/*"  # pattern covers all requests
```

---

## Exercise 3 — Page state snapshot

```python
async def page_snapshot(self) -> dict:
    """
    Return current URL, title, and screenshot in a single dict.
    Reduces the number of async calls in the agent loop.
    """
    import base64
    raw = await self._page.screenshot(full_page=False)
    data_b64 = base64.standard_b64encode(raw).decode("ascii")
    title = await self._page.title()
    return {
        "url": self._page.url,
        "title": title,
        "screenshot_b64": data_b64,
    }


# Test
@pytest.mark.asyncio
async def test_page_snapshot():
    page = make_mock_page("https://example.com")
    tools = BrowserTools(page)
    snap = await tools.page_snapshot()
    assert snap["url"] == "https://example.com"
    assert snap["title"] == "Example Domain"
    assert len(snap["screenshot_b64"]) > 0
```

---

## Exercise 4 — Type safety for navigate

```python
async def navigate(self, url: str, timeout_ms: int = 15_000) -> NavigateResult:
    """Navigate to URL; rejects non-http/https schemes."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"Only http/https URLs are allowed; got scheme: {url.split(':')[0]!r}"
        )
    import time
    t0 = time.monotonic()
    await self._page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    elapsed_ms = (time.monotonic() - t0) * 1000
    title = await self._page.title()
    return NavigateResult(url=self._page.url, title=title, load_time_ms=elapsed_ms)


# Tests
@pytest.mark.asyncio
async def test_navigate_rejects_javascript_url():
    page = make_mock_page()
    tools = BrowserTools(page)
    with pytest.raises(ValueError, match="http"):
        await tools.navigate("javascript:alert(1)")

@pytest.mark.asyncio
async def test_navigate_rejects_file_url():
    page = make_mock_page()
    tools = BrowserTools(page)
    with pytest.raises(ValueError):
        await tools.navigate("file:///etc/passwd")

@pytest.mark.asyncio
async def test_navigate_accepts_https():
    page = make_mock_page("https://example.com")
    tools = BrowserTools(page)
    result = await tools.navigate("https://example.com")
    assert result.url == "https://example.com"
```

**Why URL scheme validation in BrowserTools (not just actions.py)?**
Defense in depth. The `NavigateAction` validator blocks bad URLs at parse time,
but `BrowserTools.navigate` may be called from other code paths (e.g., direct
programmatic calls, future refactors). Both layers protect independently.
