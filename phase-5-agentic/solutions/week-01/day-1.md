# Solutions — Phase 5 · Week 01 · Day 1

## Exercise 1 — `tool_calls` field

```python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    final_answer: str | None = None
    tool_calls: list[str] = field(default_factory=list)  # added

    def add(self, msg: Message) -> None:
        self.messages.append(msg)
        log.info("context now has %d messages", len(self.messages))


def simulate_trace() -> None:
    state = AgentState()
    state.add(Message(role="user", content="What is the capital of France?"))
    state.add(Message(role="assistant", content='{"action": "search", "query": "capital France"}'))
    state.iteration += 1
    state.tool_calls.append("search")   # record tool
    state.add(Message(role="tool_result", content="Paris", tool_name="search"))
    state.add(Message(role="assistant", content="The capital of France is Paris."))
    state.done = True
    state.final_answer = "The capital of France is Paris."
    log.info("tools used: %s", state.tool_calls)
```

---

## Exercise 2 — `is_stuck` + test

```python
@dataclass
class AgentState:
    ...
    max_iterations: int = 10

    def is_stuck(self) -> bool:
        return self.iteration >= self.max_iterations
```

```python
# tests/test_day1_ex2.py
from day1_agent_concept import AgentState

def test_is_stuck_when_at_limit() -> None:
    state = AgentState(max_iterations=5)
    state.iteration = 5
    assert state.is_stuck() is True

def test_not_stuck_below_limit() -> None:
    state = AgentState(max_iterations=5)
    state.iteration = 4
    assert state.is_stuck() is False
```

---

## Exercise 3 — `classify_system` + tests

```python
def classify_system(description: str) -> str:
    lower = description.lower()
    if "loop" in lower or "agent" in lower or "decide" in lower:
        return "agent"
    if "step" in lower or "pipeline" in lower or "workflow" in lower:
        return "workflow"
    return "chain"
```

```python
# tests/test_day1_ex3.py
from day1_agent_concept import classify_system

def test_loop_keyword_returns_agent() -> None:
    assert classify_system("a system with a loop that iterates") == "agent"

def test_step_keyword_returns_workflow() -> None:
    assert classify_system("a pipeline with three steps") == "workflow"

def test_no_keyword_returns_chain() -> None:
    assert classify_system("single prompt to get a summary") == "chain"
```
