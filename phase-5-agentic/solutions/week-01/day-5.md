# Solutions — Phase 5 · Week 01 · Day 5

## Exercise 1 — Line-prefix parser

```python
import re

def parse_line_prefix(text: str) -> dict | None:
    """
    Parses:
      Action: search
      Args: {"query": "x"}
    Returns dict or None if format not matched.
    """
    action_match = re.search(r'^Action:\s*(\w+)', text, re.MULTILINE)
    args_match = re.search(r'^Args:\s*(\{.*\})', text, re.MULTILINE | re.DOTALL)
    if not action_match:
        return None
    action = action_match.group(1)
    args: dict = {}
    if args_match:
        try:
            args = json.loads(args_match.group(1))
        except json.JSONDecodeError:
            return None
    return {"action": action, "args": args}
```

Extend `parse_llm_output` to try this before the error fallback:

```python
def parse_llm_output(text: str) -> ParsedAction:
    # Try 1: clean JSON
    # Try 2: embedded JSON
    # Try 3: line-prefix
    result = parse_line_prefix(text)
    if result:
        log.warning("line-prefix format detected")
        return ParsedAction(**result)
    # Fallback
    return ParsedAction(action=ActionType.PARSE_ERROR, raw=text)
```

```python
# tests/test_day5_ex1.py
from day5_parser import parse_llm_output, ActionType

def test_line_prefix_valid() -> None:
    text = 'Action: search\nArgs: {"query": "agent"}'
    result = parse_llm_output(text)
    assert result.action == ActionType.SEARCH
    assert result.args["query"] == "agent"

def test_line_prefix_malformed_args() -> None:
    text = 'Action: search\nArgs: {bad json}'
    result = parse_llm_output(text)
    assert result.action == ActionType.PARSE_ERROR
```

---

## Exercise 2 — `nudge_message`

```python
def nudge_message(raw: str) -> str:
    return (
        "Your previous response could not be parsed. "
        "Respond only with a JSON object. "
        f"Your output was: {raw}"
    )
```

```python
# tests/test_day5_ex2.py
from day5_parser import nudge_message

def test_nudge_contains_raw() -> None:
    raw = "some unparseable text"
    nudge = nudge_message(raw)
    assert raw in nudge
    assert "JSON" in nudge
```

---

## Exercise 3 — Wiring `consecutive_errors` into the loop

```python
# In LoopState (day3):
@dataclass
class LoopState:
    ...
    consecutive_errors: int = 0

# In run_agent:
think_result = think(ctx, llm)
if think_result.action == "error":
    state.consecutive_errors += 1
    nudge = nudge_message(think_result.args.get("raw", ""))
    state.messages.append(Message(role="user", content=nudge))
    stop, reason = stopper.check(state.iteration, state.consecutive_errors, state.done)
    if stop:
        raise AgentError(f"stopped: {reason}")
    continue
else:
    state.consecutive_errors = 0
```

```python
# tests/test_day5_ex3.py
import pytest
from day5_parser import StopCondition

def test_consecutive_error_stop() -> None:
    stopper = StopCondition(max_iterations=10, max_consecutive_errors=3)
    # 3 consecutive errors → should stop
    stop, reason = stopper.check(iteration=0, consecutive_errors=3, done=False)
    assert stop
    assert "errors" in reason
```
