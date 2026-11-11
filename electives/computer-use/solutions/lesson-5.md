# Solutions: Lesson 5 — Safety, Sandboxing & Human-Approval Gates

## Exercise 1 — Content-based type filter

```python
import re

# Add to _PII_PATTERNS or create a separate check:
_PASSWORD_PATTERN = re.compile(
    r"""
    (?=.*\d)           # at least one digit
    (?=.*[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~]) # at least one special char
    .{8,}              # at least 8 chars total
    """,
    re.VERBOSE,
)


def is_sensitive_action_extended(
    action_type: str,
    action_params: dict,
    current_url: str = "",
) -> tuple[bool, str]:
    """Extended sensitivity check including password-like text."""
    # Base checks (from lesson)
    sensitive, reason = is_sensitive_action(action_type, action_params, current_url)
    if sensitive:
        return sensitive, reason

    # Password-like text check
    if action_type == "type":
        text = action_params.get("text", "")
        if _PASSWORD_PATTERN.search(text):
            return True, "Text looks like a password or credential"

    return False, ""


# Tests
def test_password_like_text_flagged():
    sensitive, reason = is_sensitive_action_extended(
        "type", {"text": "Secr3t!Pass"}, ""
    )
    assert sensitive
    assert "password" in reason.lower() or "credential" in reason.lower()

def test_short_text_not_flagged():
    sensitive, _ = is_sensitive_action_extended(
        "type", {"text": "hello world"}, ""
    )
    assert not sensitive

def test_text_without_special_not_flagged():
    # No special character — not password-like
    sensitive, _ = is_sensitive_action_extended(
        "type", {"text": "Hello12345678"}, ""
    )
    assert not sensitive

def test_text_without_digit_not_flagged():
    # No digit
    sensitive, _ = is_sensitive_action_extended(
        "type", {"text": "Hello!World"}, ""
    )
    assert not sensitive
```

---

## Exercise 2 — URL allowlist with glob wildcards

```python
import fnmatch
from dataclasses import dataclass

@dataclass
class AllowlistRule:
    action_type: str
    url_pattern: re.Pattern | None = None  # regex match
    url_glob: str | None = None            # fnmatch glob

    def url_matches(self, url: str | None) -> bool:
        if url is None:
            return self.url_pattern is None and self.url_glob is None
        if self.url_pattern is not None:
            if self.url_pattern.match(url):
                return True
        if self.url_glob is not None:
            if fnmatch.fnmatch(url, self.url_glob):
                return True
        # If neither pattern is set, any URL is OK for this rule
        if self.url_pattern is None and self.url_glob is None:
            return True
        return False


class ActionAllowlist:
    def __init__(self, rules: list[AllowlistRule]) -> None:
        self._rules = rules
        self._always_allowed = {"screenshot", "done", "error"}

    def is_allowed(self, action_type: str, url: str | None = None) -> bool:
        if action_type in self._always_allowed:
            return True
        for rule in self._rules:
            if rule.action_type == action_type and rule.url_matches(url):
                return True
        return False


# Tests
def test_glob_pattern_matches():
    rule = AllowlistRule(
        "navigate",
        url_glob="https://sandbox.example.com/search/*",
    )
    allowlist = ActionAllowlist([rule])
    assert allowlist.is_allowed("navigate", "https://sandbox.example.com/search/results")
    assert allowlist.is_allowed("navigate", "https://sandbox.example.com/search/")
    assert not allowlist.is_allowed("navigate", "https://sandbox.example.com/delete/1")

def test_glob_does_not_match_different_domain():
    rule = AllowlistRule("navigate", url_glob="https://sandbox.example.com/*")
    allowlist = ActionAllowlist([rule])
    assert not allowlist.is_allowed("navigate", "https://evil.example.com/path")

def test_regex_and_glob_both_work():
    rules = [
        AllowlistRule("navigate", url_pattern=re.compile(r"https://api\.example\.com/.*")),
        AllowlistRule("navigate", url_glob="https://sandbox.example.com/*"),
    ]
    allowlist = ActionAllowlist(rules)
    assert allowlist.is_allowed("navigate", "https://api.example.com/v1/resource")
    assert allowlist.is_allowed("navigate", "https://sandbox.example.com/page")
    assert not allowlist.is_allowed("navigate", "https://other.com/page")
```

---

## Exercise 3 — Approval timeout

```python
import threading
import sys

class TimedCLIApprovalGate:
    """
    CLI approval gate with timeout. Auto-rejects if no response within timeout_s.
    Uses a thread to read stdin without blocking the main loop indefinitely.
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    def request_approval(
        self,
        action_type: str,
        action_params: dict,
        reason: str,
        screenshot_b64: str | None,
    ) -> tuple[bool, str]:
        print(f"\nSENSITIVE ACTION: {action_type} — {reason}")
        print(f"Approve? [y/N] (auto-rejects in {self._timeout_s}s): ", end="", flush=True)

        result_holder: list[str] = []
        event = threading.Event()

        def read_input() -> None:
            try:
                line = sys.stdin.readline().strip().lower()
                result_holder.append(line)
            except Exception:
                result_holder.append("")
            finally:
                event.set()

        t = threading.Thread(target=read_input, daemon=True)
        t.start()
        got_input = event.wait(timeout=self._timeout_s)

        if not got_input or not result_holder or result_holder[0] != "y":
            reason_out = "timeout — auto-rejected" if not got_input else "rejected by operator"
            return False, reason_out
        return True, "approved by operator"


# Test (mock stdin)
import io

def test_approval_timeout(monkeypatch):
    import time

    gate = TimedCLIApprovalGate(timeout_s=0.05)
    # Patch stdin to never produce input (block forever)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # EOF immediately
    approved, notes = gate.request_approval("navigate", {}, "test", None)
    assert not approved
    assert "reject" in notes.lower() or "timeout" in notes.lower()
```

---

## Exercise 4 — Audit report

```python
def test_audit_summary():
    audit = AuditLog()
    import time

    # 3 executed, 1 blocked, 1 rejected
    audit.record(AuditEntry(time.time(), "click_coordinate", {}, True, True))
    audit.record(AuditEntry(time.time(), "click_coordinate", {}, True, True))
    audit.record(AuditEntry(time.time(), "navigate", {"url": "..."}, True, True))
    audit.record(AuditEntry(time.time(), "navigate", {"url": "blocked"}, False, False,
                            rejection_reason="not in allowlist"))
    audit.record(AuditEntry(time.time(), "navigate", {"url": "payment"}, False, False,
                            rejection_reason="rejected by gate"))

    # Add summary method to AuditLog:
    def summary(self) -> dict:
        total = len(self._entries)
        executed = sum(1 for e in self._entries if e.executed)
        blocked = sum(
            1 for e in self._entries
            if not e.approved and e.rejection_reason and "allowlist" in e.rejection_reason
        )
        rejected = sum(
            1 for e in self._entries
            if not e.approved and e.rejection_reason and "gate" in e.rejection_reason
        )
        sensitive = sum(1 for e in self._entries if not e.approved)
        return {
            "total": total, "executed": executed,
            "blocked": blocked, "rejected": rejected, "sensitive": sensitive,
        }

    AuditLog.summary = summary
    result = audit.summary()

    assert result["total"] == 5
    assert result["executed"] == 3
    assert result["blocked"] == 1
    assert result["rejected"] == 1
    assert result["sensitive"] == 2
```
