# Phase 5 · Week 07 · Day 1 — Multi-Agent Architectures Overview

> Objective: Understand the four canonical multi-agent topologies and when to choose each.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-7]]

## Concept

A **multi-agent system (MAS)** is a collection of autonomous agents that collaborate, compete, or coordinate to solve problems too large or complex for a single agent. The key insight: decompose the problem so that each agent's scope is tractable, then wire them together.

### Why multi-agent beats single-agent

| Scenario | Single-agent problem | Multi-agent solution |
|---|---|---|
| Long context | Token limits exceed capacity | Split into specialist subproblems |
| Parallel subtasks | Sequential bottleneck | Concurrent specialist agents |
| Error isolation | One bad step corrupts everything | Agents catch each other's errors |
| Domain expertise | One model can't master all domains | Specialist agents per domain |
| Verification | Self-checking is unreliable | Independent reviewer agent |

### The four canonical topologies

**1. Hierarchical / Supervisor**  
A supervisor agent decomposes the task and delegates to worker agents. Workers report back; supervisor synthesises. Classic command-and-control. Deterministic routing — the supervisor decides who does what.

```
        ┌──────────┐
        │ Supervisor│
        └─────┬────┘
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Worker A  Worker B  Worker C
```

**2. Network / Peer-to-Peer**  
Every agent can message any other. No central authority. Emergent coordination. Higher communication overhead; harder to reason about globally. Good for discovery-heavy tasks.

```
  A ◄──► B
  ▲  ╲ ╱  ▲
  │   ╳   │
  ▼  ╱ ╲  ▼
  C ◄──► D
```

**3. Sequential (Pipeline)**  
Output of agent N is input to agent N+1. Simple, debuggable, predictable latency. No parallelism. Good for document processing pipelines.

```
  A → B → C → D → result
```

**4. Concurrent (Scatter-Gather)**  
Orchestrator fans out the same or split task to multiple workers in parallel, then gathers results. Maximum throughput. Requires result merging strategy.

```
         ┌→ Worker A ─┐
Scatter ─┼→ Worker B ─┼─ Gather → result
         └→ Worker C ─┘
```

Real systems combine these: a supervisor might scatter to a pool of concurrent workers whose results flow through a sequential post-processor.

## How it works internally

**Agent identity**: each agent is a function (or class) with its own LLM context, tool set, memory, and role prompt. Agents do not share context windows — they communicate via explicit messages.

**Routing**: in hierarchical systems the supervisor maintains a **task queue** and a **worker registry**. It selects workers based on capability tags, current load, or LLM reasoning. The routing decision is itself a reasoning step.

**Message envelope**: every inter-agent message should carry:
- `sender` / `receiver` identifiers
- `message_type` (request / response / broadcast / error)
- `payload` (typed Pydantic model)
- `correlation_id` (to match requests to responses)
- `timestamp`

**State isolation**: each agent owns its local state. Shared state (global blackboard) is mediated by a separate store — never direct memory access between agents.

## Worked example

```python
"""
day1_architectures.py — Demonstrate all four topologies with mock LLM calls.
"""
import logging
import os
from typing import Any
from pydantic import BaseModel
import anthropic

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Message envelope ──────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    sender: str
    receiver: str
    message_type: str  # "request" | "response" | "broadcast"
    payload: dict[str, Any]
    correlation_id: str = ""


# ── Minimal agent abstraction ─────────────────────────────────────────────────

class Agent:
    def __init__(self, name: str, role: str, client: anthropic.Anthropic) -> None:
        self.name = name
        self.role = role
        self._client = client
        self._log = logging.getLogger(f"agent.{name}")

    def run(self, task: str) -> str:
        """Send task to LLM and return response."""
        self._log.info("running task: %s", task[:60])
        response = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=self.role,
            messages=[{"role": "user", "content": task}],
        )
        result = response.content[0].text
        self._log.info("result: %s", result[:60])
        return result


# ── 1. Sequential pipeline ────────────────────────────────────────────────────

def sequential_pipeline(topic: str, client: anthropic.Anthropic) -> str:
    """Research → Draft → Review pipeline."""
    logger.info("=== SEQUENTIAL PIPELINE ===")
    researcher = Agent("researcher", "You are a research agent. Return 3 bullet facts.", client)
    drafter    = Agent("drafter",   "You are a writing agent. Turn facts into a paragraph.", client)
    reviewer   = Agent("reviewer",  "You are a review agent. Return 'APPROVED' or one fix.", client)

    facts     = researcher.run(f"Facts about: {topic}")
    draft     = drafter.run(f"Write paragraph from:\n{facts}")
    verdict   = reviewer.run(f"Review:\n{draft}")
    logger.info("Pipeline verdict: %s", verdict[:40])
    return verdict


# ── 2. Concurrent scatter-gather ──────────────────────────────────────────────

def scatter_gather(questions: list[str], client: anthropic.Anthropic) -> list[str]:
    """Answer multiple questions in 'parallel' (simulated sequentially here)."""
    logger.info("=== SCATTER-GATHER ===")
    worker = Agent("worker", "You are a concise Q&A agent. Answer in one sentence.", client)
    results = [worker.run(q) for q in questions]
    logger.info("Gathered %d answers", len(results))
    return results


# ── 3. Hierarchical supervisor ────────────────────────────────────────────────

def hierarchical(task: str, client: anthropic.Anthropic) -> str:
    """Supervisor decomposes, delegates, synthesises."""
    logger.info("=== HIERARCHICAL ===")
    supervisor = Agent(
        "supervisor",
        "You are a supervisor. Given a task, output exactly two sub-tasks as JSON: "
        '{"sub1": "...", "sub2": "..."}',
        client,
    )
    worker_a = Agent("worker_a", "You are specialist A. Complete the sub-task given.", client)
    worker_b = Agent("worker_b", "You are specialist B. Complete the sub-task given.", client)
    synth    = Agent("synth",    "You synthesise two results into a final answer.", client)

    import json
    raw = supervisor.run(task)
    try:
        subs = json.loads(raw)
    except json.JSONDecodeError:
        subs = {"sub1": task, "sub2": task}  # fallback

    r1 = worker_a.run(subs.get("sub1", task))
    r2 = worker_b.run(subs.get("sub2", task))
    final = synth.run(f"Result A: {r1}\nResult B: {r2}")
    logger.info("Hierarchical final: %s", final[:60])
    return final


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    sequential_pipeline("the Python GIL", client)
    scatter_gather(
        ["What is a coroutine?", "What is a generator?", "What is a decorator?"],
        client,
    )
    hierarchical("Explain how async/await works in Python", client)
```

## Exercises

1. Draw (ASCII art or description) a topology for a code-review system that: (a) analyses security, (b) analyses style, (c) checks tests, then (d) produces a final verdict. Which topology best fits?
2. Modify `sequential_pipeline` to add a fourth agent that translates the reviewed text to Spanish.
3. The `scatter_gather` above is sequential under the hood. Rewrite it using `concurrent.futures.ThreadPoolExecutor` to run workers truly in parallel.
4. Add a `correlation_id` to `AgentMessage` and show how a supervisor tracks which response matches which request.
5. Identify one failure mode unique to each of the four topologies.

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can name the four topologies and give a one-sentence description of each
- [ ] I can explain why multi-agent outperforms single-agent for long-context and parallel tasks
- [ ] I can implement a minimal 3-agent sequential pipeline
- [ ] I can explain what a message envelope carries and why each field matters

## Supplementary links

- [LangGraph Multi-Agent Docs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Anthropic Agent patterns overview](https://docs.anthropic.com/en/docs/agents-and-tools/agents-overview)
- [AutoGen Multi-Agent Framework](https://microsoft.github.io/autogen/stable/)
