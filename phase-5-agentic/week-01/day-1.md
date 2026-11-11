# Phase 5 · Week 01 · Day 1 — What is an Agent?

> Objective: Define an agent precisely, distinguish it from chains and workflows, and understand the design space.  ·  Time: ~1 hr  ·  Prereqs: [[phase-4-rag/week-08/day-7]]

## Concept

An **AI agent** is a system that perceives its environment, decides which action to take, executes that action, observes the outcome, and repeats — autonomously — until a goal is reached or a stop condition fires.

The minimal ingredients are:

| Component | Role |
|-----------|------|
| LLM | Reasoning engine — decides what to do next |
| Tools | Actuators — let the model affect the world |
| Loop | Control flow — run until done |
| Memory | Context — what has happened so far |

Remove any one of these and you no longer have an agent in the full sense.

### Agent vs chain vs workflow

```
Chain:     prompt → LLM → output              (single pass, no loop)
Workflow:  step1 → step2 → step3              (multi-step but predetermined)
Agent:     loop{ observe → think → act }      (open-ended, self-directed)
```

The decisive difference is **who controls the flow**:

- **Chain/workflow** — the developer hardcodes every transition.
- **Agent** — the LLM decides what to do next, which tool to call, and when to stop.

This gives agents emergent capability (they can handle cases the developer never anticipated) at the cost of predictability.

### Where agents beat workflows

- Number of steps unknown at design time ("keep searching until you find a definitive answer")
- Conditional branching that depends on content ("if the retrieved doc is ambiguous, query a second source")
- Recovery from partial failures ("tool returned an error → try a different approach")

### Where workflows beat agents

- Safety-critical pipelines where every transition must be auditable
- High-throughput tasks with known structure (batch classification, extraction)
- Any time latency or cost must be tightly bounded

## How it works internally

When an LLM "decides" to call a tool it is producing structured text. The loop then:

1. Parses that text to extract an action name and arguments
2. Dispatches the function call
3. Appends the result to the context window
4. Sends the extended context back to the LLM

The LLM never "runs" code. The **host program** (your loop) does all execution. The LLM only produces tokens. This distinction matters for security, testing, and debugging.

Memory lives in one of four places:

- **In-context** — the message list (cheapest, limited by window size)
- **External retrieval** — vector store, database (unbounded, adds latency)
- **Model weights** — pretrained knowledge (fixed, cannot be updated at runtime)
- **Cache** — prompt caching to reduce cost on repeated prefixes

For this week you will use only in-context memory.

## Worked example

```python
"""
day1_agent_concept.py — illustrating agent components without running an LLM.
Shows the data structures that underpin a real agent loop.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Message:
    """A single turn in the agent's context window."""
    role: str          # "user" | "assistant" | "tool_result"
    content: str
    tool_name: str | None = None


@dataclass
class AgentState:
    """Everything the agent knows at one point in time."""
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    final_answer: str | None = None

    def add(self, msg: Message) -> None:
        self.messages.append(msg)
        log.info("context now has %d messages", len(self.messages))


# Illustrate a two-turn trace without any real LLM
def simulate_trace() -> None:
    state = AgentState()

    # Turn 0: user asks a question
    state.add(Message(role="user", content="What is the capital of France?"))

    # Turn 1: model decides to call a tool
    state.add(Message(role="assistant", content='{"action": "search", "query": "capital France"}'))
    state.iteration += 1

    # Turn 2: tool returns a result
    state.add(Message(role="tool_result", content="Paris", tool_name="search"))

    # Turn 3: model produces final answer
    state.add(Message(role="assistant", content="The capital of France is Paris."))
    state.done = True
    state.final_answer = "The capital of France is Paris."

    log.info("final answer: %s  (iterations: %d)", state.final_answer, state.iteration)


if __name__ == "__main__":
    simulate_trace()
```

Run it:
```bash
python day1_agent_concept.py
```

Expected output:
```
INFO context now has 1 messages
INFO context now has 2 messages
INFO context now has 3 messages
INFO context now has 4 messages
INFO final answer: The capital of France is Paris.  (iterations: 1)
```

## Exercises

1. Extend `AgentState` with a `tool_calls: list[str]` field that records the name of every tool invoked. Update `simulate_trace` to append to it.

2. Add a `max_iterations: int = 10` field to `AgentState` and a method `is_stuck(self) -> bool` that returns `True` when `iteration >= max_iterations`. Write a pytest unit test that verifies the method.

3. Write a function `classify_system(description: str) -> str` that returns `"chain"`, `"workflow"`, or `"agent"` based on keyword heuristics (e.g., contains "loop" → `"agent"`). Write two pytest tests for it.

(Solutions: ../solutions/week-01/day-1.md)

## Self-check

- [ ] I can define an agent in one sentence, naming all four components.
- [ ] I can explain why LLM + prompt alone is not an agent.
- [ ] I can give a concrete scenario where an agent outperforms a workflow, and one where it does not.
- [ ] I can draw (on paper) the data flow between LLM, tool, and loop.

## Supplementary links

- Anthropic model overview: https://docs.anthropic.com/en/docs/about-claude/models
- "What is an agent?" (Anthropic): https://docs.anthropic.com/en/docs/build-with-claude/agents
- LangChain conceptual intro to agents (framework-agnostic reading): https://python.langchain.com/docs/concepts/agents/
