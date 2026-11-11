# Week 01 Assessment — Agent Fundamentals & the Execution Loop

> Phase 5 · Week 01  ·  Pass gate: 75%  ·  Estimated time: 2 hr

---

## Part 1 — Conceptual (interview-style)  [25%]

Answer each question in 2–5 sentences. Bullet points are fine.

1. Define an AI agent. Name its four mandatory components and explain what happens if you remove the loop.

2. Describe the ReAct pattern. What does "ReAct" stand for? Walk through a two-turn trace: user asks a question, agent calls a tool, agent produces a final answer. Name the phase at each step.

3. Explain the key architectural difference between a workflow and an agent, then give one scenario where you would deliberately choose a workflow over an agent.

4. What is a "tool registry"? Why should tool execution live in the host program rather than inside the LLM?

5. Your LLM wrapper's `__call__` raises `anthropic.RateLimitError`. What strategy do you use? What parameter(s) govern it? What happens if all retries are exhausted?

6. Name three stop conditions that must be enforced as code (not just as instructions to the LLM). For each, describe what happens if it is missing.

---

## Part 2 — Coding challenge  [40%]

Implement a minimal, fully-tested agent from scratch.

### Spec

Build a module `agent_challenge.py` with:

**`class FakeLLMClient`** (identical protocol to day-4):
- Constructor takes `responses: list[str]`
- `__call__(messages, *, system="") -> str` pops and returns the next response
- Records all calls in `self.calls`

**`def parse_action(text: str) -> dict`**:
- Returns the parsed JSON dict
- Handles both clean JSON and embedded-in-prose JSON
- On failure, returns `{"action": "parse_error", "raw": text}`

**`def run_agent(goal, llm, registry, *, max_iterations=10, max_errors=3) -> str`**:
- Full observe→think→act→observe loop
- Stops on `action=="finish"`, max_iterations, or max_errors consecutive parse errors
- Raises `RuntimeError("max_iterations")` or `RuntimeError("max_errors")` when those limits fire
- Tool exceptions are caught and returned as `"ERROR: <msg>"` strings, not re-raised

**`def build_registry() -> dict`**:
- Returns a registry with two tools:
  - `search(query: str) -> str` — returns `f"result:{query}"`
  - `add(a: float, b: float) -> float` — returns `a + b`

### Acceptance criteria

Write a `tests/test_challenge.py` that **passes all of the following** without any real API calls:

```python
# TC-01: two-step agent finishes successfully
def test_two_step_finish(): ...

# TC-02: parse error → nudge → next call succeeds
def test_parse_error_recovery(): ...

# TC-03: max_iterations raises RuntimeError
def test_max_iterations_raises(): ...

# TC-04: max_errors raises RuntimeError
def test_max_errors_raises(): ...

# TC-05: tool exception returns error string, agent continues
def test_tool_exception_continues(): ...

# TC-06: FakeLLMClient records all calls
def test_fake_records_calls(): ...
```

---

## Part 3 — Debugging challenge  [20%]

The following `run_buggy_agent` function has **three bugs**. Identify each bug, explain why it causes incorrect behaviour, and give the fix (one line per fix is enough).

```python
import json

def run_buggy_agent(goal, llm, registry, max_iterations=10):
    messages = [{"role": "user", "content": goal}]
    for i in range(max_iterations):
        response = llm(messages)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            continue                                      # line A
        if data["action"] == "finish":
            return data["answer"]
        tool = registry[data["action"]]                   # line B
        result = tool(**data["args"])
        messages.append({"role": "tool", "content": result})  # line C
    return None
```

Hints: think about what happens when (a) the model returns a key that's not in the registry, (b) the tool result is not a string, (c) the parser silently skips bad output without informing the model.

---

## Part 4 — Code quality & testing  [15%]

Graded on your submission for Part 2. The grader checks:

- [ ] All functions have complete type annotations
- [ ] `logging` used (not `print`)
- [ ] Pydantic model used for at least one data boundary (e.g., parsed action)
- [ ] No secrets, no hardcoded API keys
- [ ] All test cases in `tests/test_challenge.py` pass with `pytest`
- [ ] No test makes a real HTTP or LLM call (verified with `pytest --tb=short`)

---

## Rubric

| Dimension | Weight | Markers |
|-----------|--------|---------|
| Independent implementation (Part 2) | 40% | All 6 TCs pass; code compiles; no imports of reference solutions |
| Conceptual understanding (Part 1) | 25% | Accurate definitions; correct trade-offs; no significant factual errors |
| Debugging & problem-solving (Part 3) | 20% | All 3 bugs correctly identified with explanations |
| Code quality & testing (Part 4) | 15% | Checklist above |

**Pass gate: 75%.** Scores below 75% require re-work before moving to Week 02.

---

Answer key: ../solutions/week-01/assessment.md
