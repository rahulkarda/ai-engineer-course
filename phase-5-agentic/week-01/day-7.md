# Phase 5 · Week 01 · Day 7 — Project Day & Review

> Objective: Complete and polish the week's minimal agent project; review all week concepts.  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Review: What you built this week

| Day | Concept | Deliverable |
|-----|---------|-------------|
| 1 | Agent definition, components | `AgentState` data model |
| 2 | Agent vs workflow vs chain | Decision framework |
| 3 | Loop anatomy (observe/think/act) | Four-phase loop in functions |
| 4 | LLM client wrapper | `AnthropicLLMClient` + `FakeLLMClient` |
| 5 | Output parser + stop conditions | `parse_llm_output`, `StopCondition` |
| 6 | Error handling + guards | `run_hardened_agent` with exception hierarchy |
| 7 | Project integration | Full minimal agent |

## Project work

Spend the first 3 hours completing the project described in `project.md`.
If you finished it during the Saturday session, spend this time:

1. **Polish** — are all type hints complete? `logging` everywhere `print` was used? All Pydantic models validated?
2. **Test coverage** — run `pytest --tb=short -v`. Every module should have tests. Aim for ≥ 80% coverage on `loop.py` and `parser.py`.
3. **Refactor** — is `loop.py` under 150 lines? If not, split it.

## Architecture review

Draw the final architecture of your agent on paper (or in a text diagram):

```
┌─────────────────────────────────────────────────────┐
│                  run_agent(goal)                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Client  │  │  Parser  │  │  Tool Registry   │  │
│  │(Protocol)│  │ (JSON+  │  │  search()        │  │
│  │          │  │  regex) │  │  calculator()    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│        │              │               │              │
│        └──────────────┴───────────────┘              │
│                       │                              │
│              ┌────────────────┐                      │
│              │  LoopState     │                      │
│              │  + StopCond    │                      │
│              └────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

Verify your implementation matches this.

## Week 1 integration exercise

Without looking at your notes, implement the following from scratch in ~45 minutes:

```python
"""
week1_review.py — complete agent loop from memory.
Implement all three functions below. LLM client is already provided.
"""

from day4_llm_client import FakeLLMClient

def build_tool_registry() -> dict:
    """Return a registry with 'search' and 'calculator' tools."""
    ...

def run_review_agent(goal: str, llm, registry: dict, max_iterations: int = 5) -> str:
    """
    Minimal agent loop:
    - observe: build context list
    - think: call llm
    - parse: extract action/args/answer
    - act: dispatch tool or return answer
    - guard: max_iterations, consecutive parse errors ≤ 2
    """
    ...

def test_review_agent() -> None:
    """Quick smoke test — no pytest needed."""
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

Check your solution against `../solutions/week-01/day-7.md`.

## Conceptual review questions

Answer these without looking at your notes. Then check `../solutions/week-01/day-7.md`.

1. Name the four components of an agent. Which one can be removed to produce a chain?
2. What is the decisive difference between a workflow and an agent in terms of control flow ownership?
3. In the ReAct loop, which phase handles the LLM call and which handles tool dispatch? Why are they separate?
4. What are three stop conditions every production agent must have?
5. Why should tool exceptions be caught and converted to strings rather than re-raised?
6. What does `stop_reason == "max_tokens"` mean and how should the agent handle it?
7. What is the purpose of `FakeLLMClient.calls` beyond just returning scripted responses?
8. If the LLM returns prose with embedded JSON, what is the correct parsing strategy?

## Assessment preparation

Tomorrow's assessment covers everything from this week. Revise:
- The four agent components and the control-flow diagram
- Your implementation of `parse_llm_output` — can you write it from memory?
- The exception hierarchy from day-6
- When to choose agent vs workflow vs chain

## Self-check

- [ ] My project passes all tests without any real API calls.
- [ ] Every module has type hints and uses `logging`.
- [ ] `loop.py` is under 150 lines or cleanly split.
- [ ] I answered all 8 conceptual review questions correctly.
- [ ] I can implement a minimal agent loop (observe/think/act/stop) from scratch in under 30 minutes.

## Supplementary links

- Anthropic agents guide: https://docs.anthropic.com/en/docs/build-with-claude/agents
- pytest docs: https://docs.pytest.org/en/stable/
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
