# Elective: Computer Use · Lesson 3 — Browser Automation with Playwright

> Objective: Build typed, async Playwright browser tools that an agent can call safely.
> Time: ~60 min · Prereqs: [[lesson-2.md]], `playwright` package added via `uv add playwright`

## Concept

**Playwright** is a headless browser automation library that controls Chromium, Firefox,
and WebKit via the Chrome DevTools Protocol. It gives you a programmable browser with a
Python async API — exactly what a computer-use agent needs as its "hands."

Why Playwright over Selenium or raw WebDriver?
- Async-native: plays well with async agent loops.
- Auto-waits: waits for elements to be ready before acting (reduces timing errors).
- Network interception: lets you block ads, mock responses, and capture traffic.
- Persistent browser contexts: share cookies/storage between page loads.
- Screenshot + accessibility tree: both vision and DOM paths in one library.

For a computer-use agent, Playwright serves as the **action executor** layer — the code
that actually moves pixels after the LLM decides what to do.

## How it works internally

**Architecture layers:**

```
Agent loop (Python)
    └─ BrowserTools (this lesson) — typed wrappers around Playwright
          └─ Playwright Page (async)
                └─ Browser process (Chromium/Firefox)
```

**Browser context vs. page.** A `BrowserContext` is like a fresh user profile
(cookies, storage, permissions). A `Page` is a browser tab. Always launch with an
isolated context per agent session — never reuse contexts across tasks.

**Auto-waiting.** `page.click(selector)` automatically waits for the element to be
visible, enabled, and stable before clicking. This eliminates a whole class of timing
bugs that plague Selenium.

**Playwright locators.** The modern API uses `page.locator(selector)` which returns a
`Locator` object with auto-retry. Prefer locators over `page.query_selector` (which
returns a snapshot and doesn't auto-retry).

```python
# Old (fragile): element might not exist yet
element = await page.query_selector("#submit-btn")
await element.click()

# Modern (auto-retry, auto-wait):
await page.locator("#submit-btn").click()
```

**Screenshots.** `await page.screenshot(full_page=False)` returns `bytes`. Encode with
`base64.standard_b64encode(raw).decode("ascii")` for the Anthropic API.

## Worked example

```python
# lesson_3_demo.py
"""
Typed Playwright browser tools for a computer-use agent.
Tests use a MockPage — no real browser is launched.
"""
import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class ScreenshotResult:
    data_b64: str  # base64 PNG
    width: int
    height: int
    url: str


@dataclass
class ClickResult:
    success: bool
    x: int
    y: int
    url_after: str


@dataclass
class NavigateResult:
    url: str
    title: str
    load_time_ms: float


@dataclass
class TypeResult:
    text_typed: str
    url: str


@dataclass
class ElementInfo:
    selector: str
    visible: bool
    text: str | None
    bounding_box: dict[str, float] | None  # {x, y, width, height}


# ---------------------------------------------------------------------------
# BrowserTools — typed wrapper around Playwright Page
# ---------------------------------------------------------------------------

class BrowserTools:
    """
    High-level browser tool layer for a computer-use agent.

    Each method is a discrete tool the agent may call.
    Use an isolated BrowserContext per agent session.
    """

    def __init__(self, page: Any) -> None:
        self._page = page

    async def screenshot(self) -> ScreenshotResult:
        """Capture full viewport screenshot."""
        raw: bytes = await self._page.screenshot(full_page=False)
        data_b64 = base64.standard_b64encode(raw).decode("ascii")
        viewport = self._page.viewport_size or {"width": 1280, "height": 800}
        url = self._page.url
        log.info("screenshot captured  url=%s", url)
        return ScreenshotResult(
            data_b64=data_b64,
            width=viewport.get("width", 1280),
            height=viewport.get("height", 800),
            url=url,
        )

    async def navigate(self, url: str, timeout_ms: int = 15_000) -> NavigateResult:
        """Navigate to URL; wait for network idle."""
        import time
        t0 = time.monotonic()
        await self._page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        elapsed_ms = (time.monotonic() - t0) * 1000
        title: str = await self._page.title()
        current_url: str = self._page.url
        log.info("navigated  url=%s  title=%r  ms=%.0f", current_url, title, elapsed_ms)
        return NavigateResult(url=current_url, title=title, load_time_ms=elapsed_ms)

    async def click_coordinate(self, x: int, y: int) -> ClickResult:
        """Click at absolute screen coordinates (vision-grounded action)."""
        await self._page.mouse.click(x, y)
        url_after: str = self._page.url
        log.info("click_coordinate  x=%d  y=%d  url_after=%s", x, y, url_after)
        return ClickResult(success=True, x=x, y=y, url_after=url_after)

    async def click_selector(self, selector: str, timeout_ms: int = 5_000) -> ClickResult:
        """Click element by CSS/XPath/text selector (DOM-grounded action)."""
        locator = self._page.locator(selector)
        await locator.click(timeout=timeout_ms)
        url_after: str = self._page.url
        log.info("click_selector  selector=%r  url_after=%s", selector, url_after)
        return ClickResult(success=True, x=0, y=0, url_after=url_after)

    async def type_text(
        self, selector: str, text: str, clear_first: bool = True
    ) -> TypeResult:
        """Type text into a form field, optionally clearing existing content."""
        locator = self._page.locator(selector)
        if clear_first:
            await locator.clear()
        await locator.fill(text)
        url: str = self._page.url
        log.info("type_text  selector=%r  text=%r", selector, text)
        return TypeResult(text_typed=text, url=url)

    async def scroll(
        self,
        x: int,
        y: int,
        direction: str = "down",
        clicks: int = 3,
    ) -> None:
        """Scroll at a given coordinate in the given direction."""
        delta_y = clicks * 100 if direction == "down" else -(clicks * 100)
        delta_x = 0
        if direction == "right":
            delta_x, delta_y = clicks * 100, 0
        elif direction == "left":
            delta_x, delta_y = -(clicks * 100), 0
        await self._page.mouse.wheel(delta_x, delta_y)
        log.info("scroll  direction=%s  clicks=%d", direction, clicks)

    async def get_element_info(self, selector: str) -> ElementInfo:
        """Inspect an element: visibility, text content, bounding box."""
        locator = self._page.locator(selector)
        visible: bool = await locator.is_visible()
        text: str | None = await locator.text_content() if visible else None
        bbox_raw = await locator.bounding_box() if visible else None
        bbox: dict[str, float] | None = (
            {k: bbox_raw[k] for k in ("x", "y", "width", "height")}
            if bbox_raw else None
        )
        return ElementInfo(
            selector=selector,
            visible=visible,
            text=text,
            bounding_box=bbox,
        )

    async def wait_for_selector(self, selector: str, timeout_ms: int = 10_000) -> bool:
        """Wait for selector to appear; returns True if found, False if timeout."""
        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            return True
        except Exception:  # PlaywrightTimeoutError
            log.warning("wait_for_selector timeout  selector=%r", selector)
            return False


# ---------------------------------------------------------------------------
# Context manager for isolated browser sessions
# ---------------------------------------------------------------------------

class IsolatedBrowserSession:
    """
    Creates an isolated BrowserContext + Page for one agent session.
    Cleans up automatically on __aexit__.

    Usage (production, requires playwright install chromium):
        async with IsolatedBrowserSession(playwright_instance) as tools:
            result = await tools.navigate("https://example.com")

    In tests: pass a mock page to BrowserTools directly.
    """

    def __init__(self, playwright: Any, headless: bool = True) -> None:
        self._playwright = playwright
        self._headless = headless
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    async def __aenter__(self) -> BrowserTools:
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
        )
        self._page = await self._context.new_page()
        log.info("browser session started  headless=%s", self._headless)
        return BrowserTools(self._page)

    async def __aexit__(self, *exc: Any) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        log.info("browser session closed")


# ---------------------------------------------------------------------------
# Demo (uses mock; no real browser)
# ---------------------------------------------------------------------------

def make_mock_page(url: str = "https://example.com") -> MagicMock:
    """Build a minimal async mock Page for testing."""
    page = MagicMock()
    page.url = url
    page.viewport_size = {"width": 1280, "height": 800}

    # Minimal PNG bytes (1×1 white pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    page.screenshot = AsyncMock(return_value=png_bytes)
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Example Domain")
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.wait_for_selector = AsyncMock()

    # Locator mock
    locator_mock = MagicMock()
    locator_mock.click = AsyncMock()
    locator_mock.fill = AsyncMock()
    locator_mock.clear = AsyncMock()
    locator_mock.is_visible = AsyncMock(return_value=True)
    locator_mock.text_content = AsyncMock(return_value="Submit")
    locator_mock.bounding_box = AsyncMock(
        return_value={"x": 100.0, "y": 200.0, "width": 80.0, "height": 30.0}
    )
    page.locator = MagicMock(return_value=locator_mock)

    return page


async def demo() -> None:
    page = make_mock_page()
    tools = BrowserTools(page)

    shot = await tools.screenshot()
    log.info("screenshot  width=%d  height=%d", shot.width, shot.height)

    nav = await tools.navigate("https://example.com")
    log.info("navigate  title=%r  ms=%.0f", nav.title, nav.load_time_ms)

    click = await tools.click_coordinate(320, 240)
    log.info("click  success=%s", click.success)

    typed = await tools.type_text("input#search", "browser agents")
    log.info("typed  text=%r", typed.text_typed)

    info = await tools.get_element_info("button[type=submit]")
    log.info("element  visible=%s  text=%r", info.visible, info.text)


if __name__ == "__main__":
    asyncio.run(demo())
```

Key observations:
- Every tool method returns a **typed dataclass** — no raw dicts from browser calls.
- `click_coordinate` (vision grounding) and `click_selector` (DOM grounding) are separate
  tools. The agent chooses which to use.
- `IsolatedBrowserSession` ensures cleanup even on exceptions — critical for resource
  management in a long-running agent.
- Mock page uses `AsyncMock` for async methods — required for `pytest-asyncio`.

## Exercises

1. **Accessibility tree tool.** Add a method `get_accessibility_snapshot() -> list[dict]`
   to `BrowserTools` that calls `await self._page.accessibility.snapshot()` and returns
   the tree. Write a test with a mock that returns a sample tree and verify the method
   passes it through unchanged.

2. **Network interception.** Add a method `block_external_requests(allowed_hosts: list[str])`
   that registers a route handler (`page.route(...)`) to block requests to any host not
   in `allowed_hosts`. This is a key sandboxing technique. Write a test that verifies
   the route handler is registered with the correct pattern.

3. **Page state snapshot.** Add a method `page_snapshot() -> dict` that returns
   `{"url": ..., "title": ..., "screenshot_b64": ...}` in a single async call.
   This reduces round-trips in the agent loop. Test it with the mock page.

4. **Type safety.** Extend `BrowserTools` so that `navigate` raises a `ValueError`
   if the URL scheme is not `http` or `https` (reject `file://`, `javascript:`, etc.).
   Write tests for both the rejection and the happy path.

(Solutions: solutions/lesson-3.md)

## Self-check

- [ ] I can explain the difference between `page.query_selector` and `page.locator`.
- [ ] I can write an `IsolatedBrowserSession` context manager that closes cleanly on error.
- [ ] I understand why `AsyncMock` is required for Playwright methods in tests.
- [ ] I can mock a Playwright Page without launching a real browser.
- [ ] I can describe two ways to ground a click (coordinate vs. selector) and when each is appropriate.

## Supplementary links

- [Playwright Python async API](https://playwright.dev/python/docs/api/class-page)
- [Playwright locators guide](https://playwright.dev/python/docs/locators)
- [Playwright auto-waiting](https://playwright.dev/python/docs/actionability)
- [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/)
