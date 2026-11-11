# Solutions — Week 05 · Day 5: Supervisor Pattern & Guardrail Checks

## Exercise 1: Extended PII patterns

```python
import re
from day5_supervisor import GuardrailResult, GuardrailAction

EXTENDED_PII_PATTERNS = {
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "email":       r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "phone":       r"\b\d{3}[.\-\s]\d{3}[.\-\s]\d{4}\b",
}


def check_pii_extended(text: str) -> GuardrailResult:
    found = {}
    for name, pattern in EXTENDED_PII_PATTERNS.items():
        if re.search(pattern, text):
            found[name] = pattern
    if found:
        return GuardrailResult(
            action=GuardrailAction.WARN,
            rule="pii_extended",
            detail=f"PII types found: {list(found.keys())}",
        )
    return GuardrailResult(action=GuardrailAction.PASS, rule="pii_extended")
```

Parametrised tests:

```python
import pytest


@pytest.mark.parametrize("text, expected_action", [
    ("Call me at 555-123-4567", "WARN"),          # phone
    ("SSN: 123-45-6789", "WARN"),                 # SSN
    ("Pay with 4111 1111 1111 1111", "WARN"),     # credit card
    ("Email: alice@example.com", "WARN"),          # email
    ("No PII here at all", "PASS"),               # clean
    ("The price is $1234", "PASS"),               # false positive: not credit card (not 16 digits)
])
def test_pii_extended(text, expected_action):
    result = check_pii_extended(text)
    assert result.action.value == expected_action
```

---

## Exercise 2: Token budget guardrail

```python
from dataclasses import dataclass, field


@dataclass
class BudgetTracker:
    max_tokens: int
    _used: int = field(default=0, init=False)

    def add(self, tokens: int) -> None:
        self._used += tokens

    @property
    def remaining(self) -> int:
        return self.max_tokens - self._used

    @property
    def exceeded(self) -> bool:
        return self._used > self.max_tokens


# In SupervisorAgent.run — after each worker call:
# tracker.add(resp.usage.input_tokens + resp.usage.output_tokens)
# if tracker.exceeded:
#     raise EscalationError(f"Token budget exceeded: {tracker._used}/{tracker.max_tokens}")
```

Test with mocked usage:

```python
def test_budget_exceeded_raises():
    from unittest.mock import MagicMock, patch
    from day5_supervisor import SupervisorAgent, WorkerAgent, EscalationError

    mock_client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Some output")]
    response.usage = MagicMock(input_tokens=600, output_tokens=600)  # 1200 total
    mock_client.messages.create.return_value = response

    with patch("day5_supervisor.anthropic.Anthropic", return_value=mock_client):
        worker = WorkerAgent()
        supervisor = SupervisorAgent(worker=worker, max_tokens_budget=500)  # very tight budget
        with pytest.raises(EscalationError, match="budget"):
            supervisor.run("Summarise the French Revolution.")
```

---

## Exercise 3: Multi-worker routing

```python
from dataclasses import dataclass, field
import anthropic, os


@dataclass
class RoutingSupervisor:
    workers: dict[str, WorkerAgent]
    default_worker: str
    client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    )

    def route(self, task: str) -> str:
        worker_keys = list(self.workers.keys())
        prompt = (
            f"Task: {task}\n\nAvailable workers: {worker_keys}\n"
            f"Which worker should handle this task? Reply with only the worker name."
        )
        resp = self.client.messages.create(
            model="claude-haiku-5-5", max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        key = resp.content[0].text.strip().lower()
        if key not in self.workers:
            log.warning("Router returned unknown key '%s', using default '%s'", key, self.default_worker)
            return self.default_worker
        return key

    def run(self, task: str) -> str:
        key = self.route(task)
        log.info("Routed to worker: %s", key)
        return self.workers[key].run(task)
```

Test:

```python
def test_routing_selects_correct_worker(monkeypatch):
    from unittest.mock import MagicMock
    worker_a = MagicMock()
    worker_a.run.return_value = "code answer"
    worker_b = MagicMock()
    worker_b.run.return_value = "writing answer"

    supervisor = RoutingSupervisor(
        workers={"code": worker_a, "writing": worker_b},
        default_worker="code",
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="code")]
    )
    supervisor.client = mock_client

    result = supervisor.run("Write a Python function to sort a list.")
    worker_a.run.assert_called_once()
    worker_b.run.assert_not_called()
    assert result == "code answer"
```

---

## Exercise 4: Structured audit log

```python
from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class EventType(str, Enum):
    INPUT_CHECK = "input_check"
    WORKER_CALL = "worker_call"
    OUTPUT_CHECK = "output_check"
    SUPERVISOR_REVIEW = "supervisor_review"
    ESCALATION = "escalation"
    SUCCESS = "success"


class SupervisorEvent(BaseModel):
    event_type: EventType
    timestamp: str
    detail: str
    action: str | None = None


# In SupervisorAgent — add to __init__:
self._audit_log: list[SupervisorEvent] = []

def _emit(self, event_type: EventType, detail: str, action: str | None = None) -> None:
    event = SupervisorEvent(
        event_type=event_type,
        timestamp=datetime.utcnow().isoformat(),
        detail=detail,
        action=action,
    )
    self._audit_log.append(event)
    log.info("[%s] %s", event_type.value, detail)

def get_audit_log(self) -> list[SupervisorEvent]:
    return list(self._audit_log)
```

Test:

```python
def test_audit_log_sequence_on_success(monkeypatch):
    # Mock: clean input, passing worker, passing output
    ...
    supervisor.run("Explain X")
    log = supervisor.get_audit_log()
    types = [e.event_type for e in log]
    assert EventType.INPUT_CHECK in types
    assert EventType.WORKER_CALL in types
    assert EventType.OUTPUT_CHECK in types
    assert EventType.SUCCESS in types
    # Input check must come before worker call
    assert types.index(EventType.INPUT_CHECK) < types.index(EventType.WORKER_CALL)
```
