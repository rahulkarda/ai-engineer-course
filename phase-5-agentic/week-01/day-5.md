# Phase 5 · Week 01 · Day 5 — Parsing Model Output into Actions + Stop Conditions

> Objective: Implement a robust output parser and define explicit stop conditions so the agent loop terminates deterministically.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Between THINK and ACT there is a critical step that is easy to get wrong: **parsing the model's text output into a structured action**.

LLMs produce text. Your code needs a Python object. The parser is the bridge. When it fails, the agent stalls or behaves unpredictably — so it must be robust.

### Parsing strategies (from strictest to most forgiving)

| Strategy | When to use | Risk |
|----------|-------------|------|
| JSON-only (`json.loads`) | Model is instructed to always output JSON | Fails on any extra text |
| JSON extraction (regex) | Model sometimes adds prose before/after JSON | Fragile on nested braces |
| Structured output (Anthropic `tool_use`) | Production agents | Adds API complexity |
| Line-prefix parsing (`Action: search`) | Simple ReAct-style prompts | Brittle on formatting |

For this week: **JSON-only** in a `try/except` with fallback to a `ParseError` action.

### Stop conditions

An agent without explicit stop conditions will loop forever (or until `max_tokens`).
Define all of these **before** you start the loop:

| Condition | Trigger | Handling |
|-----------|---------|----------|
| Finish action | LLM outputs `{"action": "finish", ...}` | Return the answer |
| Max iterations | `iteration >= max_iterations` | Return partial answer or raise |
| Parse error threshold | N consecutive parse failures | Raise `AgentError` |
| Empty tool result | Tool returns nothing for N turns | Raise or fallback |

Never rely on the LLM alone to stop the loop. Always have a numeric ceiling.

## How it works internally

### Prompt engineering for parseable output

Your system prompt must instruct the model explicitly:

```
Always respond with a single JSON object.
For tool calls: {"action": "<tool_name>", "args": {<kwargs>}}
To give a final answer: {"action": "finish", "answer": "<text>"}
Do not include any text outside the JSON object.
```

Even with this instruction, models occasionally add "Sure! Here is my response:" before the JSON. Your parser must handle this.

### Regex extraction pattern

```python
import re, json

def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON found in: {text!r}")
    return json.loads(match.group())
```

This finds the outermost `{...}` in the text. It fails on nested JSON with mismatched braces — sufficient for this week.

### Why parse errors must be handled, not ignored

If parsing fails silently and you pass `None` to the action dispatcher, you get an `AttributeError` ten lines later with a confusing traceback. Explicit `ParseError` actions let you:
- Retry with a "please format your response as JSON" nudge
- Count consecutive failures and abort
- Log the raw output for debugging

## Worked example

```python
"""
day5_parser.py — output parser + stop-condition manager.
"""

import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------- Models ----------

class ActionType(str, Enum):
    SEARCH = "search"
    CALCULATE = "calculate"
    FINISH = "finish"
    PARSE_ERROR = "parse_error"


class ParsedAction(BaseModel):
    action: ActionType
    args: dict[str, Any] = {}
    answer: str | None = None
    raw: str | None = None          # preserved on error for debugging


# ---------- Parser ----------

def parse_llm_output(text: str) -> ParsedAction:
    """
    Try to extract a JSON action from the model's text output.
    Returns a ParseError action instead of raising, so the loop can react.
    """
    try:
        # First try: the entire text is JSON
        data = json.loads(text.strip())
        return ParsedAction(**data)
    except json.JSONDecodeError:
        pass

    # Second try: find JSON embedded in prose
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            log.warning("embedded JSON extracted; model produced prose prefix")
            return ParsedAction(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.error("JSON found but invalid: %s", exc)

    log.error("could not parse LLM output: %r", text)
    return ParsedAction(action=ActionType.PARSE_ERROR, raw=text)


# ---------- Stop-condition manager ----------

class StopCondition(BaseModel):
    max_iterations: int = 10
    max_consecutive_errors: int = 3

    def check(
        self,
        iteration: int,
        consecutive_errors: int,
        done: bool,
    ) -> tuple[bool, str]:
        """
        Returns (should_stop, reason).
        Call after every iteration.
        """
        if done:
            return True, "finish"
        if iteration >= self.max_iterations:
            return True, f"max_iterations ({self.max_iterations})"
        if consecutive_errors >= self.max_consecutive_errors:
            return True, f"too_many_errors ({consecutive_errors} consecutive)"
        return False, ""


# ---------- Demo ----------

GOOD_JSON = '{"action": "search", "args": {"query": "agent stop conditions"}}'
PROSE_JSON = 'Sure! Here is my response:\n{"action": "finish", "answer": "Stop when done."}'
BAD_OUTPUT = "I think we should search for things."


def demo() -> None:
    for sample in (GOOD_JSON, PROSE_JSON, BAD_OUTPUT):
        result = parse_llm_output(sample)
        log.info("input=%r  →  action=%s", sample[:40], result.action)

    stopper = StopCondition(max_iterations=5, max_consecutive_errors=2)
    cases = [
        (0, 0, False),   # continue
        (4, 0, False),   # still under limit
        (5, 0, False),   # at limit → stop
        (0, 2, False),   # error threshold → stop
        (0, 0, True),    # done → stop
    ]
    for it, errs, done in cases:
        stop, reason = stopper.check(it, errs, done)
        log.info("it=%d errs=%d done=%s → stop=%s reason=%r", it, errs, done, stop, reason)


if __name__ == "__main__":
    demo()
```

### Unit tests

```python
"""tests/test_parser.py"""

import pytest
from day5_parser import ActionType, ParsedAction, StopCondition, parse_llm_output


def test_parse_clean_json() -> None:
    result = parse_llm_output('{"action": "search", "args": {"q": "x"}}')
    assert result.action == ActionType.SEARCH
    assert result.args == {"q": "x"}


def test_parse_embedded_json() -> None:
    result = parse_llm_output('Here you go: {"action": "finish", "answer": "done"}')
    assert result.action == ActionType.FINISH
    assert result.answer == "done"


def test_parse_failure_returns_parse_error() -> None:
    result = parse_llm_output("no json here at all")
    assert result.action == ActionType.PARSE_ERROR
    assert result.raw is not None


def test_stop_at_max_iterations() -> None:
    stopper = StopCondition(max_iterations=3)
    should_stop, reason = stopper.check(iteration=3, consecutive_errors=0, done=False)
    assert should_stop
    assert "max_iterations" in reason


def test_stop_at_error_threshold() -> None:
    stopper = StopCondition(max_consecutive_errors=2)
    should_stop, _ = stopper.check(iteration=0, consecutive_errors=2, done=False)
    assert should_stop


def test_no_stop_within_limits() -> None:
    stopper = StopCondition(max_iterations=10, max_consecutive_errors=3)
    should_stop, _ = stopper.check(iteration=5, consecutive_errors=1, done=False)
    assert not should_stop
```

## Exercises

1. Extend `parse_llm_output` to handle a third format: `Action: search\nArgs: {"query": "x"}` (line-prefix style). Add tests for both valid and malformed line-prefix input.

2. Add a `nudge_message(raw: str) -> str` function that returns a string like: `"Your previous response could not be parsed. Respond only with a JSON object. Your output was: <raw>"`. This will be appended to the context before the next LLM call. Write a test asserting the raw text appears in the nudge.

3. Add a `consecutive_errors` counter to `LoopState` from day-3. Increment it on `PARSE_ERROR`, reset to 0 on any successful parse. Wire `StopCondition.check` into the loop. Write a test that feeds three consecutive bad outputs and asserts the loop stops with the error-threshold reason.

(Solutions: ../solutions/week-01/day-5.md)

## Self-check

- [ ] I can implement `parse_llm_output` from scratch with fallback to embedded JSON extraction.
- [ ] I can list four stop conditions every production agent should have.
- [ ] I can explain why `max_iterations` must be a hard numeric cap, not just a request to the LLM.
- [ ] I can wire a `StopCondition` object into the loop state so every iteration checks it.

## Supplementary links

- Anthropic structured output / tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
- ReAct prompting: https://www.promptingguide.ai/techniques/react
