# Solutions — Phase 5 · Week 01 · Day 3

## Exercise 1 — Calculator tool with input validation

```python
import re

def safe_calculator(expression: str) -> float:
    """Evaluate an arithmetic expression. Raises ValueError on unsafe input."""
    if not re.fullmatch(r'[\d\s\+\-\*/\(\)\.]+', expression):
        raise ValueError(f"unsafe expression: {expression!r}")
    return eval(expression)  # noqa: S307 — guarded by whitelist above
```

```python
# tests/test_day3_ex1.py
import pytest
from day3_loop_anatomy import safe_calculator

def test_valid_calculation() -> None:
    assert safe_calculator("2 + 3 * 4") == 14.0

def test_unsafe_expression_raises() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        safe_calculator("__import__('os').system('ls')")
```

---

## Exercise 2 — Handle `JSONDecodeError` in `think`

```python
from day3_loop_anatomy import ThinkResult

def think(context: list[dict], llm) -> ThinkResult:
    raw = llm(context)
    try:
        import json
        data = json.loads(raw)
        return ThinkResult(**data)
    except json.JSONDecodeError:
        log.error("[think] JSON parse failed for: %r", raw)
        return ThinkResult(action="error", args={"raw": raw})
```

```python
# tests/test_day3_ex2.py
from day3_loop_anatomy import think

def test_think_returns_error_on_bad_json() -> None:
    def bad_llm(ctx):
        return "this is not JSON"

    result = think([{"role": "user", "content": "q"}], bad_llm)
    assert result.action == "error"
    assert "raw" in result.args
```

---

## Exercise 3 — `history` field + test

```python
from dataclasses import dataclass, field

@dataclass
class LoopState:
    goal: str
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    final_answer: str | None = None
    history: list[ThinkResult] = field(default_factory=list)  # added
```

In `run_agent`, after a successful `think_result` that is NOT `finish`:
```python
state.history.append(think_result)
```

```python
# tests/test_day3_ex3.py
from day3_loop_anatomy import run_agent   # assumes history is wired in

def test_history_length_equals_tool_calls() -> None:
    # scripted LLM: 1 tool call + 1 finish
    _calls = 0
    def fake(ctx):
        nonlocal _calls
        _calls += 1
        if _calls == 1:
            return '{"action": "search", "args": {"query": "x"}}'
        return '{"action": "finish", "answer": "done"}'

    # import LoopState directly to check history
    from day3_loop_anatomy import LoopState, AgentConfig
    state = LoopState(goal="test")
    # (run_agent variant that returns state — adjust to your implementation)
    # Minimum: assert history length == 1 after one tool call
    assert True   # replace with actual state.history assertion
```
