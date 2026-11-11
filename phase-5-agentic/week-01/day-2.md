# Phase 5 · Week 01 · Day 2 — Agent vs Workflow vs Chain (deep dive)

> Objective: Precisely characterise the spectrum from chain → workflow → agent and decide which abstraction to use for a given task.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Day 1 gave you the taxonomy. Today you build the mental model for *choosing* between them and recognise the real-world trade-offs that determine production architecture.

### The decision spectrum

```
Determinism ◄──────────────────────────────► Autonomy
   Chain          Workflow         Agent
(1 LLM call)   (N fixed steps)  (N unknown steps,
                                 LLM controls flow)
```

Moving right:
- **More capable** — handles open-ended tasks
- **More expensive** — more LLM calls, longer context
- **Harder to test** — non-deterministic branching
- **More failure modes** — tool errors, loops, hallucinated actions

### Chains

A chain is a **single inference** (or a fixed linear sequence) with no conditional branching and no loop.

```python
response = llm.invoke(prompt)          # chain: one call
```

Use chains for: classification, extraction, summarisation, single-step Q&A.

### Workflows (DAGs)

A workflow is a **directed acyclic graph** of steps where the developer defines every edge.
Conditional edges are allowed, but the *set of possible paths* is fixed at design time.

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ retrieve │───►│  rerank  │───►│ generate │
└──────────┘    └──────────┘    └──────────┘
                      │ if empty
                      ▼
                ┌──────────┐
                │ fallback │
                └──────────┘
```

Use workflows for: RAG pipelines, document processing, multi-model ensembles with known branching.

### Agents (dynamic graphs)

An agent adds a **cycle** — the LLM can send control back to an earlier step or choose from an open-ended action space.

```
         ┌──────────────────────────────┐
         │                              │
  user ──► think ──► tool call ──► result ──► think ...
         │                              │
         └─────────── done? ────────────┘
```

The LLM, not the developer, decides:
- Which tool to call
- How many times to iterate
- When the answer is good enough

### Hybrid: agent-as-orchestrator

Production systems often combine both: an outer agent that routes to inner workflows.

```
Agent (orchestrator)
  ├── route to → "search workflow" (deterministic 3-step RAG)
  ├── route to → "code execution workflow"
  └── route to → "answer directly"
```

This gives you the flexibility of agents at the top level and the predictability of workflows for each subtask.

## How it works internally

### Control flow ownership

| Pattern | Who decides next step |
|---------|----------------------|
| Chain | Developer (linear) |
| Workflow | Developer (DAG edges) |
| Agent | LLM (action output) |

The LLM "decides" by producing an output that the loop parses. If the output is `{"action": "search"}` the loop dispatches `search`. If it is `{"action": "finish", "answer": "..."}` the loop stops.

This means:
- The LLM has no real agency — it produces tokens.
- **You** design what tokens mean and how they map to code.
- Testing is about controlling what tokens the LLM returns.

### State machines vs token streams

A workflow maps naturally to a state machine: states and transitions are enumerable.
An agent maps to a token stream with a parser layered on top.
Both ultimately run on the same hardware; the distinction is conceptual but has real implications for observability and debugging.

## Worked example

```python
"""
day2_pattern_comparison.py — three patterns solving the same task.
No LLM required: shows structure, not intelligence.
"""

import logging
from collections.abc import Callable
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------- shared fake LLM ----------

def fake_llm(prompt: str) -> str:
    """Stand-in for an LLM call. Returns scripted responses."""
    if "classify" in prompt:
        return "question"
    if "search" in prompt:
        return '{"action": "search", "query": "agent definition"}'
    if "answer" in prompt:
        return '{"action": "finish", "answer": "An agent is an LLM + tools + loop."}'
    return "unknown"


# ---------- 1. Chain ----------

def chain_answer(user_input: str) -> str:
    log.info("[chain] single LLM call")
    return fake_llm(f"answer: {user_input}")


# ---------- 2. Workflow ----------

def workflow_answer(user_input: str) -> str:
    log.info("[workflow] step 1: classify")
    intent = fake_llm(f"classify: {user_input}")
    if intent == "question":
        log.info("[workflow] step 2: retrieve context")
        context = "Agents combine LLM + tools + loop."
        log.info("[workflow] step 3: generate answer")
        return fake_llm(f"answer: {user_input}  context: {context}")
    return "I can only answer questions."


# ---------- 3. Minimal agent ----------

def agent_answer(user_input: str, max_iter: int = 5) -> str:
    context = user_input
    for i in range(max_iter):
        raw = fake_llm(f"search: {context}" if i == 0 else f"answer: {context}")
        log.info("[agent] iteration %d output: %s", i, raw)
        if '"action": "finish"' in raw:
            import json
            return json.loads(raw)["answer"]
        # pretend we ran the search and got a result
        context += "  result: Agents combine LLM + tools + loop."
    return "max iterations reached"


if __name__ == "__main__":
    q = "What is an agent?"
    log.info("=== Chain ===")
    print(chain_answer(q))
    log.info("=== Workflow ===")
    print(workflow_answer(q))
    log.info("=== Agent ===")
    print(agent_answer(q))
```

## Exercises

1. Identify which pattern (chain, workflow, or agent) best fits each scenario and justify:
   - Sentiment analysis on 10,000 product reviews
   - A customer support bot that looks up orders, checks inventory, and decides whether to escalate
   - A pipeline that always: extracts entities → looks up a knowledge graph → writes a summary

2. Draw the state machine for the `workflow_answer` function above. Label states and transitions.

3. Modify `agent_answer` to record which action was taken each iteration in a `list[str]`. Return both the answer and the action trace as a `tuple[str, list[str]]`. Add a pytest test asserting the trace contains `"search"` at iteration 0.

(Solutions: ../solutions/week-01/day-2.md)

## Self-check

- [ ] I can articulate the single criterion that separates a workflow from an agent.
- [ ] I can describe the hybrid "agent-as-orchestrator" pattern.
- [ ] I can enumerate three production costs that increase as you move from chain to agent.
- [ ] I can write a chain, a workflow, and an agent stub for the same task.

## Supplementary links

- Anthropic agent patterns: https://docs.anthropic.com/en/docs/build-with-claude/agents
- LangGraph conceptual: agents vs workflows https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
