# Solutions — Phase 5 · Week 01 · Day 2

## Exercise 1 — Classification decisions

| Scenario | Pattern | Justification |
|----------|---------|---------------|
| Sentiment analysis on 10k reviews | **Chain** | Single classification call per review; no loop needed; deterministic |
| Customer support bot (order lookup, escalation decision) | **Agent** | Unknown number of tool calls; LLM decides when to escalate; open-ended |
| Extract → knowledge graph → summary pipeline | **Workflow** | Three fixed steps, known branching at design time |

---

## Exercise 2 — State machine diagram for `workflow_answer`

```
         ┌─────────┐
  start  │ classify │
─────────►          ├──── intent == "question" ────►  ┌─────────┐
         └─────────┘                                  │ retrieve │
               │                                      └────┬─────┘
               │ intent != "question"                      │
               ▼                                           ▼
         ┌──────────────────────┐                   ┌──────────┐
         │ "I can only answer   │                   │ generate │
         │  questions." (stop) │                   └────┬─────┘
         └──────────────────────┘                        │
                                                         ▼
                                                      (return)
```

States: `classify`, `retrieve`, `generate`, `fallback_return`
Transitions: `intent == "question"` / `intent != "question"` (from classify)

---

## Exercise 3 — Action trace + test

```python
import json
from typing import Any

def agent_answer(user_input: str, max_iter: int = 5) -> tuple[str, list[str]]:
    context = user_input
    trace: list[str] = []
    for i in range(max_iter):
        raw = fake_llm(f"search: {context}" if i == 0 else f"answer: {context}")
        if '"action": "finish"' in raw:
            data = json.loads(raw)
            return data["answer"], trace
        trace.append("search")
        context += "  result: Agents combine LLM + tools + loop."
    return "max iterations reached", trace
```

```python
# tests/test_day2_ex3.py
from day2_pattern_comparison import agent_answer

def test_trace_starts_with_search() -> None:
    answer, trace = agent_answer("What is an agent?")
    assert trace[0] == "search"

def test_answer_is_returned() -> None:
    answer, trace = agent_answer("What is an agent?")
    assert "agent" in answer.lower()
```
