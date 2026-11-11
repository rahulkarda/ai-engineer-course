# Elective: Computer Use · Lesson 5 — Safety, Sandboxing & Human-Approval Gates

> Objective: Design a layered safety architecture that prevents irreversible and unauthorized actions.
> Time: ~70 min · Prereqs: [[lesson-4.md]]

## Concept

**Computer-use agents can cause real damage.** A single hallucinated `click` on a
"Delete Account" button, a mistaken form submission, or a navigate to a phishing page
in the context of a logged-in session can have consequences that are hard or impossible
to reverse.

Safety for computer-use agents is not optional and not "good to add later." It must be
designed in from the start, as a structural constraint the agent cannot bypass.

The layered model:

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Sandboxed environment                      │
│   (isolated browser context, network allowlist,     │
│    no access to real accounts)                       │
├─────────────────────────────────────────────────────┤
│  Layer 2: Action allowlist                           │
│   (only permitted action types and target URLs)     │
├─────────────────────────────────────────────────────┤
│  Layer 3: Sensitive-action detection                 │
│   (flag irreversible actions before execution)      │
├─────────────────────────────────────────────────────┤
│  Layer 4: Human-approval gate                        │
│   (require explicit approval for flagged actions)   │
├─────────────────────────────────────────────────────┤
│  Layer 5: Audit log                                  │
│   (every action recorded with context)              │
└─────────────────────────────────────────────────────┘
```

Every layer is an independent safety net. A bug that bypasses one layer is caught by the
next. **Defense in depth.**

## How it works internally

**Layer 1 — Sandboxed environment.** Use an isolated `BrowserContext` per session with:
- No persistent storage (no cookies from real accounts).
- Network route filtering to block non-allowlisted hosts.
- No file system access from the browser (block `file://`).
- If possible, run in a container with network egress restricted by firewall rules.

**Layer 2 — Action allowlist.** A set of (action_type, pattern) rules. Example:

```python
allowlist = [
    ("navigate", r"^https://sandbox\.example\.com/.*"),
    ("click_coordinate", None),   # always allowed
    ("screenshot", None),          # always allowed
    ("type", None),                # allowed; content checked separately
]
```

An action that doesn't match any rule is **blocked** — not deferred, not logged and
continued, **blocked**.

**Layer 3 — Sensitive-action detection.** Before executing, classify the action:
- Does it target a URL containing `/delete`, `/payment`, `/confirm`, `/submit`?
- Does the text contain credit card patterns, PII, credentials?
- Does the LLM's stated plan include words like "purchase", "delete", "send"?

Flag these for Layer 4.

**Layer 4 — Human-approval gate.** When a flagged action arrives:
1. Pause the agent loop.
2. Present the action and context to the human (log, UI, Slack message, etc.).
3. Wait for explicit `approved` / `rejected` response.
4. Resume or abort accordingly.

In automated testing: the approval gate is a callable that you inject — tests can inject
a mock that auto-approves or auto-rejects to test both paths.

**Layer 5 — Audit log.** Every action, its approval status, the screenshot before
execution, and the screenshot after. This is your forensic record if something goes wrong.

## Worked example

```python
# lesson_5_demo.py
"""
Layered safety architecture for a computer-use agent.
ApprovalGate is injectable — tests mock it; production wires real human approval.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    timestamp: float
    action_type: str
    action_params: dict[str, Any]
    approved: bool
    executed: bool
    screenshot_before_b64: str | None = None
    screenshot_after_b64: str | None = None
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


# ---------------------------------------------------------------------------
# Action allowlist
# ---------------------------------------------------------------------------

@dataclass
class AllowlistRule:
    action_type: str                  # matches action type string exactly
    url_pattern: re.Pattern | None    # None = no URL restriction


class ActionAllowlist:
    """
    Layer 2: Only allow explicitly permitted (action_type, url) combinations.
    Any action not matching a rule is blocked unconditionally.
    """

    def __init__(self, rules: list[AllowlistRule]) -> None:
        self._rules = rules
        # Always-allowed housekeeping actions
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
        """Convenience: allow all actions only on the sandbox domain."""
        pattern = re.compile(re.escape(base_url))
        return cls([
            AllowlistRule("navigate", pattern),
            AllowlistRule("click_coordinate", None),
            AllowlistRule("click_element", None),
            AllowlistRule("type", None),
            AllowlistRule("scroll", None),
        ])


# ---------------------------------------------------------------------------
# Sensitive action detection
# ---------------------------------------------------------------------------

_SENSITIVE_URL_FRAGMENTS = [
    "delete", "remove", "destroy",
    "payment", "checkout", "purchase", "order",
    "confirm", "submit", "send",
    "password", "credentials",
    "transfer", "withdraw",
]

_PII_PATTERNS = [
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),  # credit card
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]


def is_sensitive_action(
    action_type: str,
    action_params: dict[str, Any],
    current_url: str = "",
) -> tuple[bool, str]:
    """
    Returns (is_sensitive, reason).
    Checks URL fragments and text content for sensitive patterns.
    """
    # Navigate to sensitive URL
    if action_type == "navigate":
        url = action_params.get("url", "").lower()
        for fragment in _SENSITIVE_URL_FRAGMENTS:
            if fragment in url:
                return True, f"URL contains sensitive fragment: '{fragment}'"

    # Type containing PII
    if action_type == "type":
        text = action_params.get("text", "")
        for pattern in _PII_PATTERNS:
            if pattern.search(text):
                return True, f"Text matches PII pattern: {pattern.pattern[:40]}"

    # Click on current sensitive URL
    if action_type in ("click_coordinate", "click_element"):
        for fragment in _SENSITIVE_URL_FRAGMENTS:
            if fragment in current_url.lower():
                return True, f"Clicking on sensitive page: '{fragment}' in URL"

    return False, ""


# ---------------------------------------------------------------------------
# Human-approval gate (injectable Protocol)
# ---------------------------------------------------------------------------

class ApprovalGate(Protocol):
    """
    Injectable approval interface.
    Production: prompt human via CLI/Slack/UI.
    Tests: inject AutoApproveGate or AutoRejectGate.
    """

    def request_approval(
        self,
        action_type: str,
        action_params: dict[str, Any],
        reason: str,
        screenshot_b64: str | None,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, notes: str).
        This call MAY block waiting for human input.
        """
        ...


class CLIApprovalGate:
    """
    Production gate: prints action details to stdout, reads yes/no from stdin.
    Use this when running interactively. Never use in automated tests.
    """

    def request_approval(
        self,
        action_type: str,
        action_params: dict[str, Any],
        reason: str,
        screenshot_b64: str | None,
    ) -> tuple[bool, str]:
        print("\n" + "=" * 60)
        print("SENSITIVE ACTION REQUIRES APPROVAL")
        print(f"  Action:  {action_type}")
        print(f"  Params:  {action_params}")
        print(f"  Reason:  {reason}")
        print("=" * 60)
        answer = input("Approve? [y/N] ").strip().lower()
        approved = answer == "y"
        notes = "approved by operator" if approved else "rejected by operator"
        return approved, notes


class AutoApproveGate:
    """Test stub: always approves. Never use in production."""

    def request_approval(self, *_: Any, **__: Any) -> tuple[bool, str]:
        return True, "auto-approved (test)"


class AutoRejectGate:
    """Test stub: always rejects. Never use in production."""

    def request_approval(self, *_: Any, **__: Any) -> tuple[bool, str]:
        return False, "auto-rejected (test)"


# ---------------------------------------------------------------------------
# SafetyLayer — wraps execution with all five layers
# ---------------------------------------------------------------------------

class SafetyLayer:
    """
    Composes all five safety layers around action execution.
    """

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
        execute_fn: Any,  # Callable[[], Any]
        screenshot_before_b64: str | None = None,
    ) -> dict[str, Any]:
        """
        Run all safety checks, then execute if cleared.

        Returns {"status": "executed"|"blocked"|"rejected", "result": ...}
        """
        # Layer 2: allowlist check
        navigate_url = action_params.get("url") if action_type == "navigate" else current_url
        if not self._allowlist.is_allowed(action_type, navigate_url):
            reason = f"Action '{action_type}' to '{navigate_url}' not in allowlist"
            log.warning("BLOCKED  %s", reason)
            self._audit.record(AuditEntry(
                timestamp=time.time(),
                action_type=action_type,
                action_params=action_params,
                approved=False,
                executed=False,
                screenshot_before_b64=screenshot_before_b64,
                rejection_reason=reason,
            ))
            return {"status": "blocked", "reason": reason}

        # Layer 3: sensitive action detection
        sensitive, reason = is_sensitive_action(action_type, action_params, current_url)

        if sensitive:
            # Layer 4: human approval
            log.warning("SENSITIVE ACTION detected: %s", reason)
            approved, notes = self._gate.request_approval(
                action_type, action_params, reason, screenshot_before_b64
            )
            if not approved:
                self._audit.record(AuditEntry(
                    timestamp=time.time(),
                    action_type=action_type,
                    action_params=action_params,
                    approved=False,
                    executed=False,
                    screenshot_before_b64=screenshot_before_b64,
                    rejection_reason=notes,
                ))
                return {"status": "rejected", "reason": notes}

        # Execute
        result = execute_fn()

        self._audit.record(AuditEntry(
            timestamp=time.time(),
            action_type=action_type,
            action_params=action_params,
            approved=True,
            executed=True,
            screenshot_before_b64=screenshot_before_b64,
        ))
        return {"status": "executed", "result": result}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    allowlist = ActionAllowlist.sandbox_only("https://sandbox.example.com")
    audit = AuditLog()
    safety = SafetyLayer(allowlist, AutoApproveGate(), audit)

    # Allowed action
    r = safety.check_and_execute(
        "navigate", {"url": "https://sandbox.example.com/search"},
        current_url="", execute_fn=lambda: "navigated",
    )
    log.info("result: %s", r)

    # Blocked: URL not in allowlist
    r = safety.check_and_execute(
        "navigate", {"url": "https://production.example.com/admin"},
        current_url="", execute_fn=lambda: "should not run",
    )
    log.info("result: %s", r)

    # Sensitive: navigate to /delete — auto-approved by test gate
    r = safety.check_and_execute(
        "navigate", {"url": "https://sandbox.example.com/delete/item/42"},
        current_url="", execute_fn=lambda: "deleted",
    )
    log.info("result: %s", r)

    # Sensitive: auto-reject gate
    reject_safety = SafetyLayer(allowlist, AutoRejectGate(), audit)
    r = reject_safety.check_and_execute(
        "navigate", {"url": "https://sandbox.example.com/payment"},
        current_url="", execute_fn=lambda: "paid",
    )
    log.info("result: %s", r)

    log.info("audit entries: %d  violations: %d", len(audit.entries), audit.violation_count())


if __name__ == "__main__":
    demo()
```

Key observations:
- `ApprovalGate` is a **Protocol** — inject `CLIApprovalGate` in production,
  `AutoApproveGate` / `AutoRejectGate` in tests. Never hardcode the gate.
- `ActionAllowlist` blocks actions that never matched any rule — **default deny**, not
  default allow. This is the correct security posture.
- `is_sensitive_action` returns `(bool, reason)` — the reason goes to both the audit
  log and the human reviewer. Always explain the flag.
- The five layers are independent Python objects — each can be tested in isolation.

## Exercises

1. **Content-based type filter.** Extend `is_sensitive_action` to also flag `type`
   actions where the text contains a sequence that looks like a password
   (`length >= 8, at least one digit, at least one special char`).
   Write tests with both matching and non-matching strings.

2. **URL-based allowlist with wildcards.** Enhance `AllowlistRule` to support
   glob-style patterns for URL matching (`fnmatch.fnmatch`) in addition to regex,
   e.g. `"https://sandbox.example.com/search/*"`.
   Write tests that verify the pattern matching behaves correctly.

3. **Approval timeout.** Extend `CLIApprovalGate` to accept a `timeout_seconds`
   parameter. If no input is received within the timeout, auto-reject.
   Use `select` or `threading` — do not block the loop indefinitely.
   Write a test that simulates timeout by mocking stdin.

4. **Audit report.** Add a method `AuditLog.summary() -> dict` that returns:
   `{"total": int, "executed": int, "blocked": int, "rejected": int, "sensitive": int}`.
   Write a test that records a mix of audit entries and verifies the counts.

(Solutions: solutions/lesson-5.md)

## Self-check

- [ ] I can draw the five-layer safety stack from memory.
- [ ] I can implement an allowlist with default-deny semantics.
- [ ] I can write a Protocol-based approval gate and inject mocks in tests.
- [ ] I can list five URL fragments and two text patterns that should trigger a review.
- [ ] I understand why sandboxed browser contexts are an essential first defense.

## Supplementary links

- [Anthropic computer use safety guidelines](https://docs.anthropic.com/en/docs/build-with-claude/computer-use#computer-use-safety)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Playwright browser contexts](https://playwright.dev/python/docs/browser-contexts)
- [Playwright network routing](https://playwright.dev/python/docs/network#handle-requests)
