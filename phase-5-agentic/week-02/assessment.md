# Week 02 Assessment — Tool & Function Calling

> Pass mark: 75%  ·  Estimated time: ~2 hrs  ·  Prereqs: [[day-5]], [[day-6]], [[day-7]]

---

## Conceptual (interview-style)

Answer each question in 3–6 sentences. Bullet points are acceptable.

**1. Tool schema contract**

Describe the three required top-level fields in an Anthropic tool definition. For each field, explain what happens at runtime if it is missing or malformed.

---

**2. The message cycle**

Walk through the complete HTTP message exchange when Claude makes a single tool call. Include: what the assistant response looks like, what your next API request must contain, and why passing `response.content` (not just the tool_use block) back in the assistant turn matters.

---

**3. Parallel tool calls — detection and dispatch**

How do you detect that Claude has requested parallel tool calls rather than a single one? What is the correct way to return all results to Claude, and why is sending them in separate messages an anti-pattern?

---

**4. Input validation**

You have registered a tool that expects `quantity: int`. Claude sends `"quantity": "five"`. Describe what should happen at each layer: the Pydantic validation step, the dispatcher, and the message Claude receives. Should the loop abort?

---

**5. Tool design — split vs. combine**

Give two concrete examples where splitting one tool into two is the right choice, and one example where keeping them combined is correct. For each, justify your decision using the principles covered in the course.

---

**6. Idempotency and retry**

Why is idempotency important for tools in an agentic loop? Describe a scenario where a non-idempotent tool combined with retry logic causes a data integrity problem. How would you fix it?

---

## Coding Challenge

### Task: Minimal typed tool-calling framework

Implement a minimal but complete tool-calling framework. You may use the day 6/7 code as reference but must implement it yourself without copy-pasting.

### Specification

Build a Python package `mini_framework` with three public classes: `Registry`, `Dispatcher`, and `Loop`.

**Registry**
- `register` decorator: accepts any Python function, extracts its name, first line of docstring as description, and builds a JSON Schema from its type annotations.
- `get_schemas() -> list[dict]`: returns Anthropic-compatible tool schema list.
- `get(name: str)`: returns the registered function (raise `KeyError` if not found).

**Dispatcher**
- `dispatch(tool_id: str, name: str, inputs: dict) -> dict`: validates inputs with a dynamically created Pydantic model, calls the function, returns a `tool_result` block. Must not raise — errors go into `is_error` results.

**Loop**
- `run(user_message: str, client, model: str) -> str`: drives the tool_use loop until `end_turn` or 10 iterations. Accepts any client with a `.messages.create()` method.

### Constraints

- Python 3.12+
- Type hints on every function and method
- `logging` not `print`
- `os.getenv("ANTHROPIC_API_KEY")` for any real client construction (not in the framework itself — only in your demo script)
- No hardcoded API keys anywhere
- Model IDs must be from the `claude-*-4-5` family only

### Acceptance criteria

- [ ] `Registry.register` works as a decorator with zero configuration
- [ ] `Registry.get_schemas()` returns valid Anthropic tool schema dicts
- [ ] `Dispatcher.dispatch` returns `is_error: True` for unknown tools
- [ ] `Dispatcher.dispatch` returns `is_error: True` when Pydantic validation fails
- [ ] `Loop.run` returns the final assistant text on `end_turn`
- [ ] `Loop.run` returns `""` (not raises) when 10 iterations are exhausted
- [ ] All three classes are covered by pytest tests
- [ ] All LLM calls in tests are mocked (no real API calls)
- [ ] Tests use `pytest.fixture` for the registry and mock client
- [ ] `ruff check` passes on all files

### Deliverable

A git-committed directory `submissions/week-02/` containing:
```
mini_framework/
    __init__.py
    registry.py
    dispatcher.py
    loop.py
tests/
    test_mini_framework.py
demo.py          ← shows the framework working with a mocked client
```

---

## Rubric (pass >= 75%)

| Dimension | Weight | Full marks criteria |
|---|---|---|
| Independent implementation | 40% | All three classes implemented correctly without copying day-6/7 solutions verbatim; logic is sound |
| Conceptual understanding | 25% | Conceptual answers are accurate, specific, and use correct terminology (tool_use, tool_result, stop_reason, etc.) |
| Debugging & problem-solving | 20% | Error paths handled; loop does not crash on bad tool calls; edge cases (unknown tool, wrong type, exhausted loop) covered |
| Code quality & testing | 15% | Type hints everywhere, logging not print, tests pass, ruff clean, no hardcoded keys |

### Scoring guide

| Score | Grade |
|---|---|
| 90–100% | Distinction |
| 75–89% | Pass |
| 60–74% | Near pass — retry with fixes |
| < 60% | Fail — revisit day 6 and 7 before resubmitting |

---

Answer key: ../solutions/week-02/assessment.md
