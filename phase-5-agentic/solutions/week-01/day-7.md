# Solutions — Phase 5 · Week 01 · Day 7

This is a project/review day. Solutions fall into two parts:

## Part 1 — `week1_review.py` implementation

```python
"""week1_review.py — complete agent loop from memory (reference implementation)."""

import json
import re
from day4_llm_client import FakeLLMClient


def build_tool_registry() -> dict:
    def search(query: str) -> str:
        return f"search result for: {query}"

    def calculator(expression: str) -> float:
        if not re.fullmatch(r'[\d\s\+\-\*/\(\)\.]+', expression):
            raise ValueError(f"unsafe: {expression!r}")
        return eval(expression)  # noqa: S307

    return {"search": search, "calculator": calculator}


def run_review_agent(goal: str, llm, registry: dict, max_iterations: int = 5) -> str:
    messages = [{"role": "user", "content": goal}]
    consecutive_errors = 0

    for _i in range(max_iterations):
        raw = llm(messages)

        # Parse
        try:
            data = json.loads(raw.strip())
            consecutive_errors = 0
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    consecutive_errors = 0
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is None:
            consecutive_errors += 1
            if consecutive_errors >= 2:
                raise RuntimeError("too many parse errors")
            messages.append({"role": "user", "content": "Respond only with JSON."})
            continue

        if data.get("action") == "finish":
            return data.get("answer", "")

        action = data.get("action", "")
        args = data.get("args", {})
        if action in registry:
            try:
                result = str(registry[action](**args))
            except Exception as exc:
                result = f"ERROR: {exc}"
        else:
            result = f"ERROR: unknown tool '{action}'"

        messages.append({"role": "assistant", "content": f"called {action}"})
        messages.append({"role": "tool", "content": result})

    raise RuntimeError("max_iterations")


def test_review_agent() -> None:
    script = [
        '{"action": "search", "args": {"query": "agent"}}',
        '{"action": "finish", "answer": "An agent is LLM + tools + loop."}',
    ]
    llm = FakeLLMClient(script)
    registry = build_tool_registry()
    answer = run_review_agent("What is an agent?", llm, registry)
    assert answer == "An agent is LLM + tools + loop.", f"got: {answer}"
    print("PASS")


if __name__ == "__main__":
    test_review_agent()
```

---

## Part 2 — Conceptual review answers

1. **Four components**: LLM, tools, loop, memory. Removing the loop turns the agent into a chain (single inference).

2. **ReAct** = Reasoning + Acting. Two-turn trace:
   - User asks → *Observe* (assemble context)
   - LLM produces `{"action": "search"}` → *Think* (LLM call) then *Act* (dispatch tool)
   - Tool result appended → *Observe* again
   - LLM produces `{"action": "finish", "answer": "..."}` → *Think* → loop exits

3. **Workflow vs agent**: workflow developer defines every edge; agent LLM decides next step. Choose workflow when steps are known at design time and auditability is required.

4. **Tool registry**: dict mapping action name → callable. Tool execution lives in host code (not LLM) because the LLM cannot actually invoke functions — it only produces text. The host interprets that text and calls real code.

5. `RateLimitError` → exponential backoff with jitter, up to `llm_retries` attempts. If all retries exhausted: re-raise as `LLMError`.

6. Three stop conditions: `max_iterations` (hard numeric ceiling), `max_consecutive_errors` (parse-error cascade), `action == "finish"` (goal reached). Missing `max_iterations`: infinite loop. Missing `max_consecutive_errors`: cascade of bad nudges fills context. Missing `finish` check: agent ignores the answer and keeps looping.

7. `stop_reason == "max_tokens"`: response was truncated mid-token; likely causes a parse error. Log a warning, return the truncated text to the parser; if it fails, treat as a parse error.

8. `FakeLLMClient.calls` records exactly what was sent to the LLM. This lets tests assert on the content of messages (e.g., was the nudge appended? was the system prompt correct?) not just on the response.

9. Prose + embedded JSON: use `re.search(r'\{.*\}', text, re.DOTALL)` to extract the first JSON block, then `json.loads` it.
