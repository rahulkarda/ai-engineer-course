# Solutions: Capstone Project — Sandboxed Browser Agent

This solution provides complete, production-quality implementations for all project modules.
Tests use mocked LLM and browser — no real API calls.

---

## `src/sandboxed_agent/config.py`

```python
# src/sandboxed_agent/config.py
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = "claude-sonnet-5-5"
    display_width: int = 1280
    display_height: int = 800
    max_turns: int = 20
    max_history_images: int = 3
    sandbox_base_url: str = "https://sandbox.example.com"
    approval_timeout_s: float = 30.0

    model_config = {"populate_by_name": True}
```

---

## `src/sandboxed_agent/actions.py`

```python
# src/sandboxed_agent/actions.py
from __future__ import annotations

from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, TypeAdapter, field_validator


class ClickCoordinateAction(BaseModel):
    type: Literal["click_coordinate"] = "click_coordinate"
    x: int = Field(ge=0, le=4096)
    y: int = Field(ge=0, le=4096)


class ClickElementAction(BaseModel):
    type: Literal["click_element"] = "click_element"
    element_id: str


class TypeAction(BaseModel):
    type: Literal["type"] = "type"
    text: str = Field(min_length=1, max_length=4096)
    target_selector: str | None = None

    @field_validator("text")
    @classmethod
    def no_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("text must not contain null bytes")
        return v


class ScrollAction(BaseModel):
    type: Literal["scroll"] = "scroll"
    x: int = Field(default=640, ge=0, le=4096)
    y: int = Field(default=400, ge=0, le=4096)
    direction: Literal["up", "down", "left", "right"] = "down"
    clicks: int = Field(default=3, ge=1, le=20)


class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    url: str

    @field_validator("url")
    @classmethod
    def require_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs allowed, got: {v!r}")
        return v


class ScreenshotAction(BaseModel):
    type: Literal["screenshot"] = "screenshot"


class DoneAction(BaseModel):
    type: Literal["done"] = "done"
    result: str = ""


AgentAction = Annotated[
    ClickCoordinateAction
    | ClickElementAction
    | TypeAction
    | ScrollAction
    | NavigateAction
    | ScreenshotAction
    | DoneAction,
    Field(discriminator="type"),
]

_adapter: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def parse_action(raw: dict[str, Any]) -> AgentAction:
    return _adapter.validate_python(raw)


def scale_coordinate(
    x: int,
    y: int,
    model_width: int,
    model_height: int,
    browser_width: int,
    browser_height: int,
) -> tuple[int, int]:
    return (
        round(x * browser_width / model_width),
        round(y * browser_height / model_height),
    )
```

---

## `src/sandboxed_agent/browser_tools.py`

```python
# src/sandboxed_agent/browser_tools.py
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass
class ScreenshotResult:
    data_b64: str
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
    bounding_box: dict[str, float] | None


class BrowserTools:
    def __init__(self, page: Any) -> None:
        self._page = page

    async def screenshot(self) -> ScreenshotResult:
        raw: bytes = await self._page.screenshot(full_page=False)
        viewport = self._page.viewport_size or {"width": 1280, "height": 800}
        return ScreenshotResult(
            data_b64=base64.standard_b64encode(raw).decode("ascii"),
            width=viewport.get("width", 1280),
            height=viewport.get("height", 800),
            url=self._page.url,
        )

    async def navigate(self, url: str, timeout_ms: int = 15_000) -> NavigateResult:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs allowed: {url!r}")
        t0 = time.monotonic()
        await self._page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        title: str = await self._page.title()
        return NavigateResult(
            url=self._page.url,
            title=title,
            load_time_ms=(time.monotonic() - t0) * 1000,
        )

    async def click_coordinate(self, x: int, y: int) -> ClickResult:
        await self._page.mouse.click(x, y)
        return ClickResult(success=True, x=x, y=y, url_after=self._page.url)

    async def click_selector(self, selector: str, timeout_ms: int = 5_000) -> ClickResult:
        await self._page.locator(selector).click(timeout=timeout_ms)
        return ClickResult(success=True, x=0, y=0, url_after=self._page.url)

    async def type_text(
        self, selector: str, text: str, clear_first: bool = True
    ) -> TypeResult:
        locator = self._page.locator(selector)
        if clear_first:
            await locator.clear()
        await locator.fill(text)
        return TypeResult(text_typed=text, url=self._page.url)

    async def scroll(
        self, x: int, y: int, direction: str = "down", clicks: int = 3
    ) -> None:
        dy = clicks * 100 * (1 if direction == "down" else -1 if direction == "up" else 0)
        dx = clicks * 100 * (1 if direction == "right" else -1 if direction == "left" else 0)
        await self._page.mouse.wheel(dx, dy)

    async def get_element_info(self, selector: str) -> ElementInfo:
        locator = self._page.locator(selector)
        visible = await locator.is_visible()
        text = await locator.text_content() if visible else None
        bbox_raw = await locator.bounding_box() if visible else None
        bbox = (
            {k: bbox_raw[k] for k in ("x", "y", "width", "height")} if bbox_raw else None
        )
        return ElementInfo(selector=selector, visible=visible, text=text, bounding_box=bbox)

    async def wait_for_selector(self, selector: str, timeout_ms: int = 10_000) -> bool:
        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            return True
        except Exception:
            return False
```

---

## `src/sandboxed_agent/safety.py`

```python
# src/sandboxed_agent/safety.py
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

_SENSITIVE_URL_FRAGMENTS = [
    "delete", "remove", "destroy", "payment", "checkout",
    "purchase", "order", "confirm", "submit", "send",
    "password", "credentials", "transfer", "withdraw",
]
_PII_PATTERNS = [
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]


@dataclass
class AuditEntry:
    timestamp: float
    action_type: str
    action_params: dict[str, Any]
    approved: bool
    executed: bool
    screenshot_before_b64: str | None = None
    rejection_reason: str | None = None


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        log.info(
            "audit  action=%s  approved=%s  executed=%s",
            entry.action_type, entry.approved, entry.executed,
        )

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def violation_count(self) -> int:
        return sum(1 for e in self._entries if not e.approved)

    def summary(self) -> dict[str, int]:
        total = len(self._entries)
        executed = sum(1 for e in self._entries if e.executed)
        not_approved = [e for e in self._entries if not e.approved]
        return {
            "total": total,
            "executed": executed,
            "blocked": len(not_approved),
            "rejected": len(not_approved),
        }


@dataclass
class AllowlistRule:
    action_type: str
    url_pattern: re.Pattern | None = None


class ActionAllowlist:
    def __init__(self, rules: list[AllowlistRule]) -> None:
        self._rules = rules
        self._always_allowed = {"screenshot", "done", "error"}

    def is_allowed(self, action_type: str, url: str | None = None) -> bool:
        if action_type in self._always_allowed:
            return True
        for rule in self._rules:
            if rule.action_type != action_type:
                continue
            if rule.url_pattern is None:
                return True
            if url and rule.url_pattern.match(url):
                return True
        return False

    @classmethod
    def sandbox_only(cls, base_url: str) -> "ActionAllowlist":
        pattern = re.compile(re.escape(base_url))
        return cls([
            AllowlistRule("navigate", pattern),
            AllowlistRule("click_coordinate"),
            AllowlistRule("click_element"),
            AllowlistRule("type"),
            AllowlistRule("scroll"),
        ])


def is_sensitive_action(
    action_type: str,
    action_params: dict[str, Any],
    current_url: str = "",
) -> tuple[bool, str]:
    if action_type == "navigate":
        url = action_params.get("url", "").lower()
        for frag in _SENSITIVE_URL_FRAGMENTS:
            if frag in url:
                return True, f"URL contains sensitive fragment: '{frag}'"
    if action_type == "type":
        text = action_params.get("text", "")
        for pat in _PII_PATTERNS:
            if pat.search(text):
                return True, f"Text matches PII pattern"
    if action_type in ("click_coordinate", "click_element"):
        for frag in _SENSITIVE_URL_FRAGMENTS:
            if frag in current_url.lower():
                return True, f"Clicking on sensitive page ('{frag}' in URL)"
    return False, ""


class ApprovalGate(Protocol):
    def request_approval(
        self,
        action_type: str,
        action_params: dict[str, Any],
        reason: str,
        screenshot_b64: str | None,
    ) -> tuple[bool, str]: ...


class AutoApproveGate:
    def request_approval(self, *_: Any, **__: Any) -> tuple[bool, str]:
        return True, "auto-approved (test)"


class AutoRejectGate:
    def request_approval(self, *_: Any, **__: Any) -> tuple[bool, str]:
        return False, "auto-rejected (test)"


class CLIApprovalGate:
    def request_approval(
        self,
        action_type: str,
        action_params: dict[str, Any],
        reason: str,
        screenshot_b64: str | None,
    ) -> tuple[bool, str]:
        print(f"\nSENSITIVE ACTION: {action_type} — {reason}")
        answer = input("Approve? [y/N] ").strip().lower()
        approved = answer == "y"
        return approved, "approved by operator" if approved else "rejected by operator"


class SafetyLayer:
    def __init__(
        self,
        allowlist: ActionAllowlist,
        approval_gate: ApprovalGate,
        audit_log: AuditLog,
    ) -> None:
        self._allowlist = allowlist
        self._gate = approval_gate
        self._audit = audit_log

    def check_and_execute(
        self,
        action_type: str,
        action_params: dict[str, Any],
        current_url: str,
        execute_fn: Any,
        screenshot_before_b64: str | None = None,
    ) -> dict[str, Any]:
        navigate_url = (
            action_params.get("url") if action_type == "navigate" else current_url
        )
        if not self._allowlist.is_allowed(action_type, navigate_url):
            reason = f"'{action_type}' to '{navigate_url}' not in allowlist"
            log.warning("BLOCKED  %s", reason)
            self._audit.record(AuditEntry(
                timestamp=time.time(), action_type=action_type,
                action_params=action_params, approved=False, executed=False,
                screenshot_before_b64=screenshot_before_b64, rejection_reason=reason,
            ))
            return {"status": "blocked", "reason": reason}

        sensitive, reason = is_sensitive_action(action_type, action_params, current_url)
        if sensitive:
            log.warning("SENSITIVE ACTION: %s", reason)
            approved, notes = self._gate.request_approval(
                action_type, action_params, reason, screenshot_before_b64
            )
            if not approved:
                self._audit.record(AuditEntry(
                    timestamp=time.time(), action_type=action_type,
                    action_params=action_params, approved=False, executed=False,
                    screenshot_before_b64=screenshot_before_b64, rejection_reason=notes,
                ))
                return {"status": "rejected", "reason": notes}

        result = execute_fn()
        self._audit.record(AuditEntry(
            timestamp=time.time(), action_type=action_type,
            action_params=action_params, approved=True, executed=True,
            screenshot_before_b64=screenshot_before_b64,
        ))
        return {"status": "executed", "result": result}
```

---

## `src/sandboxed_agent/verification.py`

```python
# src/sandboxed_agent/verification.py
import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

log = logging.getLogger(__name__)


class VerificationError(Exception):
    pass


@dataclass
class VerificationCondition:
    description: str
    check: Callable[[], Awaitable[bool]]


async def verify_with_retry(
    conditions: list[VerificationCondition],
    max_attempts: int = 3,
    base_wait_s: float = 0.5,
) -> None:
    for condition in conditions:
        for attempt in range(max_attempts):
            try:
                passed = await condition.check()
                if passed:
                    log.info("verified  condition=%r", condition.description)
                    break
                wait = base_wait_s * (2 ** attempt) + random.uniform(0, 0.1)
                if attempt < max_attempts - 1:
                    log.warning(
                        "verification failed  attempt=%d  retrying in %.2fs  condition=%r",
                        attempt + 1, wait, condition.description,
                    )
                    await asyncio.sleep(wait)
            except Exception as exc:
                log.error("verification error  condition=%r  exc=%s", condition.description, exc)
        else:
            raise VerificationError(
                f"Condition '{condition.description}' failed after {max_attempts} attempts"
            )


async def with_retry(
    action_fn: Callable[[], Awaitable[Any]],
    label: str,
    max_attempts: int = 3,
    base_wait_s: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await action_fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = base_wait_s * (2 ** attempt) + random.uniform(0, 0.2)
                log.warning(
                    "retry  label=%s  attempt=%d  wait=%.2fs  exc=%s",
                    label, attempt + 1, wait, exc,
                )
                await asyncio.sleep(wait)
    raise RuntimeError(f"'{label}' failed after {max_attempts} attempts") from last_exc


def build_navigation_conditions(page: Any, expected_url_fragment: str) -> list[VerificationCondition]:
    async def check_url() -> bool:
        return expected_url_fragment in page.url
    return [VerificationCondition(
        f"URL contains '{expected_url_fragment}'", check_url
    )]


def build_form_submit_conditions(page: Any, success_selector: str) -> list[VerificationCondition]:
    async def check_visible() -> bool:
        try:
            return await page.locator(success_selector).is_visible()
        except Exception:
            return False
    return [VerificationCondition(
        f"success element '{success_selector}' visible", check_visible
    )]
```

---

## `src/sandboxed_agent/agent_loop.py`

```python
# src/sandboxed_agent/agent_loop.py
import base64
import logging
from typing import Any, Literal

from pydantic import BaseModel

from .actions import (
    AgentAction, ClickCoordinateAction, ClickElementAction, DoneAction,
    NavigateAction, ScreenshotAction, ScrollAction, TypeAction,
    parse_action, scale_coordinate,
)
from .browser_tools import BrowserTools
from .config import AgentConfig
from .safety import SafetyLayer
from .verification import VerificationCondition, verify_with_retry, VerificationError

log = logging.getLogger(__name__)

COMPUTER_TOOL = {
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": 1280,
    "display_height_px": 800,
}


class AgentRunResult(BaseModel):
    status: Literal["done", "blocked", "timeout", "error"]
    turns_used: int
    safety_violations: int
    verification_failures: int
    final_message: str
    audit_entries: list[dict]


class SandboxedWebAgent:
    def __init__(
        self,
        config: AgentConfig,
        browser_tools: BrowserTools,
        safety: SafetyLayer,
        llm_client: Any,
    ) -> None:
        self._config = config
        self._tools = browser_tools
        self._safety = safety
        self._llm = llm_client

        # Update tool definition from config
        COMPUTER_TOOL["display_width_px"] = config.display_width
        COMPUTER_TOOL["display_height_px"] = config.display_height

    async def run(self, task: str) -> AgentRunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        violations = 0
        verif_failures = 0

        for turn in range(self._config.max_turns):
            log.info("--- turn %d ---", turn)

            response = self._llm.create(
                model=self._config.model,
                max_tokens=4096,
                tools=[COMPUTER_TOOL],
                messages=messages,
                betas=["computer-use-2025-01-01"],
                system=(
                    "You are a sandboxed computer-use agent. "
                    "Use the computer tool to complete the given task. "
                    "Always take a screenshot first to see the current state."
                ),
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return AgentRunResult(
                    status="done",
                    turns_used=turn + 1,
                    safety_violations=violations,
                    verification_failures=verif_failures,
                    final_message=final_text,
                    audit_entries=[
                        vars(e) for e in self._safety._audit.entries
                    ],
                )

            if response.stop_reason != "tool_use":
                log.error("unexpected stop_reason: %s", response.stop_reason)
                return AgentRunResult(
                    status="error",
                    turns_used=turn + 1,
                    safety_violations=violations,
                    verification_failures=verif_failures,
                    final_message=f"unexpected stop_reason: {response.stop_reason}",
                    audit_entries=[vars(e) for e in self._safety._audit.entries],
                )

            tool_results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Take screenshot before action for audit
                shot = await self._tools.screenshot()
                action_type = block.input.get("action", "unknown")

                # Safety check + execute
                safety_result = self._safety.check_and_execute(
                    action_type=action_type,
                    action_params=block.input,
                    current_url=self._tools._page.url,
                    execute_fn=lambda: None,  # replaced below
                    screenshot_before_b64=shot.data_b64,
                )

                if safety_result["status"] in ("blocked", "rejected"):
                    violations += 1
                    # Return an error tool_result so the agent knows
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [{
                            "type": "text",
                            "text": f"Action blocked: {safety_result['reason']}",
                        }],
                        "is_error": True,
                    })
                    continue

                # Execute the actual action
                result_data = await self._execute_action(block.input)

                # Post-action verification for state-changing actions
                if action_type not in ("screenshot",):
                    try:
                        new_shot = await self._tools.screenshot()
                        async def check_changed(
                            before: str = shot.data_b64,
                            after_shot: Any = new_shot,
                        ) -> bool:
                            return before != after_shot.data_b64
                        await verify_with_retry(
                            [VerificationCondition("state changed", check_changed)],
                            max_attempts=1,
                            base_wait_s=0.1,
                        )
                    except VerificationError:
                        verif_failures += 1
                        log.warning("verification failed for action %s", action_type)

                # Build tool_result
                after_shot = await self._tools.screenshot()
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": after_shot.data_b64,
                        },
                    }],
                })

            messages.append({"role": "user", "content": tool_results})
            messages = _trim_screenshot_history(messages, self._config.max_history_images)

        return AgentRunResult(
            status="timeout",
            turns_used=self._config.max_turns,
            safety_violations=violations,
            verification_failures=verif_failures,
            final_message="max_turns exceeded",
            audit_entries=[vars(e) for e in self._safety._audit.entries],
        )

    async def _execute_action(self, action_input: dict[str, Any]) -> Any:
        action = action_input.get("action")
        if action == "screenshot":
            return await self._tools.screenshot()
        if action == "left_click":
            coord = action_input["coordinate"]
            bw = self._tools._page.viewport_size["width"]
            bh = self._tools._page.viewport_size["height"]
            sx, sy = scale_coordinate(
                coord[0], coord[1],
                self._config.display_width, self._config.display_height, bw, bh,
            )
            return await self._tools.click_coordinate(sx, sy)
        if action == "type":
            return await self._tools.type_text(
                action_input.get("selector", "body"),
                action_input["text"],
            )
        if action in ("navigate",):
            return await self._tools.navigate(action_input["url"])
        if action == "scroll":
            coord = action_input.get("coordinate", [640, 400])
            await self._tools.scroll(
                coord[0], coord[1],
                action_input.get("direction", "down"),
                action_input.get("amount", 3),
            )
        return None


def _trim_screenshot_history(
    messages: list[dict], max_images: int
) -> list[dict]:
    """Keep only the last max_images screenshots in message history."""
    image_locations: list[tuple[int, int, int]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for ci, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                for ii, inner in enumerate(block.get("content", [])):
                    if isinstance(inner, dict) and inner.get("type") == "image":
                        image_locations.append((mi, ci, ii))

    to_drop = image_locations[:-max_images] if len(image_locations) > max_images else []
    for mi, ci, _ in to_drop:
        messages[mi]["content"][ci]["content"] = []
    return messages
```

---

## `tests/test_actions.py`

```python
# tests/test_actions.py
import pytest
from pydantic import ValidationError
from sandboxed_agent.actions import (
    ClickCoordinateAction, ClickElementAction, DoneAction, NavigateAction,
    ScreenshotAction, ScrollAction, TypeAction, parse_action, scale_coordinate,
)


def test_click_coordinate_valid():
    a = parse_action({"type": "click_coordinate", "x": 100, "y": 200})
    assert isinstance(a, ClickCoordinateAction)
    assert a.x == 100


def test_click_coordinate_invalid_x_string():
    with pytest.raises(ValidationError):
        parse_action({"type": "click_coordinate", "x": "not-int", "y": 0})


def test_navigate_rejects_javascript():
    with pytest.raises(ValidationError, match="http"):
        parse_action({"type": "navigate", "url": "javascript:alert(1)"})


def test_navigate_rejects_file():
    with pytest.raises(ValidationError):
        parse_action({"type": "navigate", "url": "file:///etc/passwd"})


def test_navigate_accepts_https():
    a = parse_action({"type": "navigate", "url": "https://example.com"})
    assert isinstance(a, NavigateAction)


def test_type_rejects_null_bytes():
    with pytest.raises(ValidationError, match="null"):
        parse_action({"type": "type", "text": "hello\x00world"})


def test_type_valid():
    a = parse_action({"type": "type", "text": "hello world"})
    assert isinstance(a, TypeAction)


def test_done_action():
    a = parse_action({"type": "done", "result": "completed"})
    assert isinstance(a, DoneAction)
    assert a.result == "completed"


def test_screenshot_action():
    a = parse_action({"type": "screenshot"})
    assert isinstance(a, ScreenshotAction)


def test_scale_coordinate_same_size():
    assert scale_coordinate(640, 400, 1280, 800, 1280, 800) == (640, 400)


def test_scale_coordinate_double_size():
    assert scale_coordinate(640, 400, 1280, 800, 2560, 1600) == (1280, 800)


def test_scale_coordinate_half_size():
    assert scale_coordinate(640, 400, 1280, 800, 640, 400) == (320, 200)
```

---

## `tests/test_safety.py`

```python
# tests/test_safety.py
import pytest
from sandboxed_agent.safety import (
    ActionAllowlist, AllowlistRule, AuditEntry, AuditLog,
    AutoApproveGate, AutoRejectGate, SafetyLayer, is_sensitive_action,
)
import re, time


def test_allowlist_blocks_non_matching():
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    assert not allowlist.is_allowed("navigate", "https://production.example.com")


def test_allowlist_permits_sandbox():
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    assert allowlist.is_allowed("navigate", "https://sandbox.example.com/search")


def test_allowlist_always_permits_screenshot():
    allowlist = ActionAllowlist(rules=[])
    assert allowlist.is_allowed("screenshot")


def test_sensitive_delete_url():
    sensitive, reason = is_sensitive_action(
        "navigate", {"url": "https://example.com/delete/1"}, ""
    )
    assert sensitive
    assert "delete" in reason


def test_sensitive_pii_email():
    sensitive, reason = is_sensitive_action(
        "type", {"text": "user@example.com"}, ""
    )
    assert sensitive


def test_not_sensitive_plain_text():
    sensitive, _ = is_sensitive_action("type", {"text": "hello world"}, "")
    assert not sensitive


def make_safety(approve: bool = True) -> tuple[SafetyLayer, AuditLog]:
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    gate = AutoApproveGate() if approve else AutoRejectGate()
    return SafetyLayer(allowlist, gate, audit), audit


def test_execute_allowed_action():
    safety, audit = make_safety(approve=True)
    called = []
    result = safety.check_and_execute(
        "click_coordinate", {"x": 10, "y": 10},
        current_url="https://sandbox.example.com",
        execute_fn=lambda: called.append(True) or "ok",
    )
    assert result["status"] == "executed"
    assert called

def test_blocked_action_not_executed():
    safety, audit = make_safety(approve=True)
    called = []
    result = safety.check_and_execute(
        "navigate", {"url": "https://production.example.com"},
        current_url="",
        execute_fn=lambda: called.append(True),
    )
    assert result["status"] == "blocked"
    assert not called
    assert audit.violation_count() == 1


def test_rejected_sensitive_action():
    safety, audit = make_safety(approve=False)
    called = []
    result = safety.check_and_execute(
        "navigate", {"url": "https://sandbox.example.com/delete/item"},
        current_url="",
        execute_fn=lambda: called.append(True),
    )
    assert result["status"] == "rejected"
    assert not called
    assert audit.violation_count() == 1


def test_audit_records_all_actions():
    safety, audit = make_safety(approve=True)
    safety.check_and_execute("screenshot", {}, "", lambda: None)
    safety.check_and_execute("click_coordinate", {"x": 0, "y": 0},
                             "https://sandbox.example.com", lambda: None)
    assert len(audit.entries) == 2
```

---

## `tests/test_verification.py`

```python
# tests/test_verification.py
import asyncio
import pytest
from sandboxed_agent.verification import (
    VerificationCondition, VerificationError, verify_with_retry, with_retry,
)


@pytest.mark.asyncio
async def test_verify_passes_immediately():
    cond = VerificationCondition("always true", lambda: asyncio.coroutine(lambda: True)())

    async def always_true() -> bool:
        return True

    cond = VerificationCondition("always true", always_true)
    await verify_with_retry([cond], max_attempts=3, base_wait_s=0.01)


@pytest.mark.asyncio
async def test_verify_retries_flaky():
    call_count = [0]

    async def flaky() -> bool:
        call_count[0] += 1
        return call_count[0] >= 2

    cond = VerificationCondition("flaky check", flaky)
    await verify_with_retry([cond], max_attempts=3, base_wait_s=0.01)
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_verify_raises_on_permanent_fail():
    async def always_false() -> bool:
        return False

    cond = VerificationCondition("never true", always_false)
    with pytest.raises(VerificationError):
        await verify_with_retry([cond], max_attempts=2, base_wait_s=0.01)


@pytest.mark.asyncio
async def test_with_retry_succeeds_eventually():
    attempt = [0]

    async def flaky_action() -> str:
        attempt[0] += 1
        if attempt[0] < 3:
            raise ConnectionError("transient")
        return "success"

    result = await with_retry(
        flaky_action, "test", max_attempts=3, base_wait_s=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "success"
    assert attempt[0] == 3
```

---

## `tests/test_agent_loop.py`

```python
# tests/test_agent_loop.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from sandboxed_agent.agent_loop import SandboxedWebAgent
from sandboxed_agent.browser_tools import BrowserTools, MINIMAL_PNG
from sandboxed_agent.config import AgentConfig
from sandboxed_agent.safety import (
    ActionAllowlist, AuditLog, AutoApproveGate, AutoRejectGate, SafetyLayer,
)


def make_agent(approve: bool = True, llm_responses: list | None = None):
    config = AgentConfig(
        anthropic_api_key="test",
        sandbox_base_url="https://sandbox.example.com",
        max_turns=10,
        max_history_images=2,
    )
    page = MagicMock()
    page.url = "https://sandbox.example.com"
    page.viewport_size = {"width": 1280, "height": 800}
    page.screenshot = AsyncMock(return_value=MINIMAL_PNG)
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Test")
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.type = AsyncMock()
    page.locator = MagicMock(return_value=MagicMock(
        click=AsyncMock(), fill=AsyncMock(), clear=AsyncMock(),
        is_visible=AsyncMock(return_value=True),
        text_content=AsyncMock(return_value=""),
        bounding_box=AsyncMock(return_value=None),
    ))
    page.wait_for_selector = AsyncMock()

    tools = BrowserTools(page)
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    gate = AutoApproveGate() if approve else AutoRejectGate()
    safety = SafetyLayer(allowlist, gate, audit)

    # Default: 2 screenshot tool_use turns, then end_turn
    if llm_responses is None:
        def make_tool_block(tid: str) -> MagicMock:
            b = MagicMock()
            b.type = "tool_use"
            b.id = tid
            b.input = {"action": "screenshot"}
            b.name = "computer"
            return b

        def make_end_turn() -> MagicMock:
            r = MagicMock()
            r.stop_reason = "end_turn"
            tb = MagicMock()
            tb.type = "text"
            tb.text = "Task complete."
            r.content = [tb]
            return r

        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                r = MagicMock()
                r.stop_reason = "tool_use"
                r.content = [make_tool_block(f"t{call_count[0]}")]
                return r
            return make_end_turn()

        llm_client = MagicMock()
        llm_client.create = mock_create
    else:
        llm_client = MagicMock()
        it = iter(llm_responses)
        llm_client.create = MagicMock(side_effect=lambda **kw: next(it))

    return SandboxedWebAgent(config, tools, safety, llm_client), audit


@pytest.mark.asyncio
async def test_happy_path_returns_done():
    agent, audit = make_agent()
    result = await agent.run("Search for 'test'")
    assert result.status == "done"
    assert result.turns_used > 0
    assert "Task complete" in result.final_message


@pytest.mark.asyncio
async def test_blocked_action_increments_violations():
    # Agent tries to navigate to a non-sandboxed domain
    def make_nav_block(tid: str, url: str) -> MagicMock:
        b = MagicMock()
        b.type = "tool_use"
        b.id = tid
        b.input = {"action": "navigate", "url": url}
        b.name = "computer"
        return b

    def make_end() -> MagicMock:
        r = MagicMock()
        r.stop_reason = "end_turn"
        tb = MagicMock()
        tb.type = "text"
        tb.text = "Done."
        r.content = [tb]
        return r

    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            r = MagicMock()
            r.stop_reason = "tool_use"
            r.content = [make_nav_block("t1", "https://evil.com/admin")]
            return r
        return make_end()

    config = AgentConfig(anthropic_api_key="test", max_turns=5)
    page = MagicMock()
    page.url = "https://sandbox.example.com"
    page.viewport_size = {"width": 1280, "height": 800}
    page.screenshot = AsyncMock(return_value=MINIMAL_PNG)
    tools = BrowserTools(page)
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    safety = SafetyLayer(allowlist, AutoApproveGate(), audit)
    llm = MagicMock()
    llm.create = mock_create
    agent = SandboxedWebAgent(config, tools, safety, llm)
    result = await agent.run("test")
    assert result.safety_violations >= 1


@pytest.mark.asyncio
async def test_sensitive_action_rejected_with_auto_reject_gate():
    def make_del_nav(tid: str) -> MagicMock:
        b = MagicMock()
        b.type = "tool_use"
        b.id = tid
        b.input = {"action": "navigate", "url": "https://sandbox.example.com/delete/1"}
        b.name = "computer"
        return b

    def make_end() -> MagicMock:
        r = MagicMock()
        r.stop_reason = "end_turn"
        tb = MagicMock()
        tb.type = "text"
        tb.text = "Done."
        r.content = [tb]
        return r

    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            r = MagicMock()
            r.stop_reason = "tool_use"
            r.content = [make_del_nav("t1")]
            return r
        return make_end()

    config = AgentConfig(anthropic_api_key="test", max_turns=5)
    page = MagicMock()
    page.url = "https://sandbox.example.com"
    page.viewport_size = {"width": 1280, "height": 800}
    page.screenshot = AsyncMock(return_value=MINIMAL_PNG)
    tools = BrowserTools(page)
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    # Use AUTO-REJECT gate
    safety = SafetyLayer(allowlist, AutoRejectGate(), audit)
    llm = MagicMock()
    llm.create = mock_create
    agent = SandboxedWebAgent(config, tools, safety, llm)
    result = await agent.run("test")
    # Rejection is counted as a violation
    assert result.safety_violations >= 1
    assert audit.violation_count() >= 1


@pytest.mark.asyncio
async def test_max_turns_returns_timeout():
    # LLM never stops — always returns tool_use
    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        b = MagicMock()
        b.type = "tool_use"
        b.id = f"t{call_count[0]}"
        b.input = {"action": "screenshot"}
        b.name = "computer"
        r = MagicMock()
        r.stop_reason = "tool_use"
        r.content = [b]
        return r

    config = AgentConfig(anthropic_api_key="test", max_turns=3)
    page = MagicMock()
    page.url = "https://sandbox.example.com"
    page.viewport_size = {"width": 1280, "height": 800}
    page.screenshot = AsyncMock(return_value=MINIMAL_PNG)
    tools = BrowserTools(page)
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    safety = SafetyLayer(allowlist, AutoApproveGate(), audit)
    llm = MagicMock()
    llm.create = mock_create
    agent = SandboxedWebAgent(config, tools, safety, llm)
    result = await agent.run("never-ending task")
    assert result.status == "timeout"
    assert result.turns_used == 3
```

---

## `.env.example`

```
ANTHROPIC_API_KEY=your-key-here
```

## `pyproject.toml` (minimal)

```toml
[project]
name = "sandboxed-agent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "playwright>=1.40.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```
