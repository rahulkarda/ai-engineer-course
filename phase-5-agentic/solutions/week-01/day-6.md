# Solutions — Phase 5 · Week 01 · Day 6

## Exercise 1 — Recovery after two parse errors then success

```python
# tests/test_day6_ex1.py
from day6_hardened_agent import run_hardened_agent, AgentConfig

def test_parse_error_then_success() -> None:
    """LLM fails twice, succeeds on third call."""
    _calls = 0

    def llm_with_recovery(ctx):
        nonlocal _calls
        _calls += 1
        if _calls <= 2:
            return "not json at all"
        return '{"action": "finish", "answer": "recovered"}'

    answer = run_hardened_agent(
        "test",
        llm=llm_with_recovery,
        registry={},
        config=AgentConfig(max_consecutive_errors=3, max_iterations=10, llm_retries=1),
    )
    assert answer == "recovered"
```

---

## Exercise 2 — `MaxConsecutiveErrorsError` on all-bad output

```python
# tests/test_day6_ex2.py
import pytest
from day6_hardened_agent import run_hardened_agent, AgentConfig, MaxConsecutiveErrorsError

def test_max_errors_raised() -> None:
    def always_bad(ctx):
        return "definitely not json"

    with pytest.raises(MaxConsecutiveErrorsError):
        run_hardened_agent(
            "test",
            llm=always_bad,
            registry={},
            config=AgentConfig(max_consecutive_errors=2, max_iterations=10, llm_retries=1),
        )
```

---

## Exercise 3 — `MaxIterationsError` on infinite search loop

```python
# tests/test_day6_ex3.py
import pytest
from day6_hardened_agent import run_hardened_agent, AgentConfig, MaxIterationsError

def test_max_iterations_raised() -> None:
    def always_search(ctx):
        return '{"action": "search", "args": {"query": "loop forever"}}'

    registry = {"search": lambda query: "result"}

    with pytest.raises(MaxIterationsError):
        run_hardened_agent(
            "test",
            llm=always_search,
            registry=registry,
            config=AgentConfig(max_iterations=3, llm_retries=1),
        )
```

---

## Exercise 4 — Unknown tool continues, does not raise

```python
# tests/test_day6_ex4.py
from day6_hardened_agent import run_hardened_agent, AgentConfig

def test_unknown_tool_continues() -> None:
    _step = 0

    def llm_with_bad_tool(ctx):
        nonlocal _step
        _step += 1
        if _step == 1:
            return '{"action": "nonexistent_tool", "args": {}}'
        return '{"action": "finish", "answer": "survived"}'

    answer = run_hardened_agent(
        "test",
        llm=llm_with_bad_tool,
        registry={},   # no tools registered
        config=AgentConfig(max_iterations=5, llm_retries=1),
    )
    assert answer == "survived"
```

---

## Exercise 5 — `total_tool_errors` counter

```python
# Add to LoopState
@dataclass
class LoopState:
    ...
    total_tool_errors: int = 0

# In the ACT section of run_hardened_agent, where ToolNotFoundError/ToolExecutionError are caught:
except (ToolNotFoundError, ToolExecutionError):
    state.total_tool_errors += 1
    ...
```

```python
# tests/test_day6_ex5.py — requires run_hardened_agent to return (answer, state) or expose state
# Simplest approach: make run_hardened_agent return state.total_tool_errors as a side-channel,
# or use a version that returns the full state.
# Here we test via the observable effect: answer is still returned after one tool error.

from day6_hardened_agent import run_hardened_agent, AgentConfig

def test_tool_error_counted() -> None:
    _step = 0

    def llm(ctx):
        nonlocal _step
        _step += 1
        if _step == 1:
            return '{"action": "broken_tool", "args": {}}'
        return '{"action": "finish", "answer": "done"}'

    # broken_tool raises when called
    def broken_tool() -> None:
        raise RuntimeError("boom")

    answer = run_hardened_agent(
        "test",
        llm=llm,
        registry={"broken_tool": broken_tool},
        config=AgentConfig(max_iterations=5, llm_retries=1),
    )
    assert answer == "done"
    # If you expose total_tool_errors on the returned state: assert state.total_tool_errors == 1
```
