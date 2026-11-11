# Phase 5 · Week 09 · Day 1 — The Agent Interoperability Problem

> Objective: Understand why isolated agents fail at scale and what interoperability requires.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-7]]

## Concept

Single agents hit hard ceilings: knowledge limits, capability gaps, compute constraints, and organizational boundaries. The interoperability problem emerges when you need multiple agents—possibly built by different teams, in different languages, running on different infrastructure—to cooperate on a task.

**Three failure modes of siloed agents:**

1. **Capability siloing** — Agent A can do NLP, Agent B can write SQL. Neither can do an end-to-end data analysis without the other. With no protocol for cooperation, a human must relay outputs.
2. **Duplication** — Five teams independently build a "summarization agent." Without discovery, each reinvents the wheel; updates don't propagate.
3. **Tight coupling** — You wire agents together with bespoke HTTP calls and custom JSON schemas. When Agent B changes its API, every caller breaks.

**The core insight:** Agents need what microservices needed in 2012—a contract layer that decouples caller from implementation. The difference is that agent "calls" are richer than RPC: they carry goals, context, and partial results, not just function arguments.

## How it works internally

Why can't agents just call each other's REST APIs directly?

They can, but three things go wrong immediately:

**Semantic mismatch.** Agent A sends `{"action": "analyze", "text": "..."}`. Agent B expects `{"task": "summarization", "content": "..."}`. Both are correct internally; neither understands the other without a translation layer.

**Trust and scope ambiguity.** When Agent A asks Agent B to "complete the user's request," how much authority is being delegated? Can B make external API calls? Write to a database? Spawn sub-agents? Without an explicit capability manifest, B doesn't know what it's allowed to do, and A doesn't know what B can do.

**Lifecycle opacity.** REST calls are synchronous, but agent tasks may run for seconds to hours. The caller needs to know: is the task running, done, failed, or abandoned? Standard HTTP 200/404/500 doesn't cover these states.

The interoperability layer must solve all three: **semantic contracts**, **capability advertisement**, and **task lifecycle management**.

## Worked example

```python
"""
day_1_interop_problem.py
Demonstrates the three failure modes with a minimal simulation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Failure mode 1: Semantic mismatch
# ──────────────────────────────────────────────

@dataclass
class AgentAMessage:
    action: str
    text: str

@dataclass
class AgentBExpects:
    task: str
    content: str


def agent_b_handler(msg: dict[str, Any]) -> str:
    """Agent B only knows its own schema."""
    if "task" not in msg or "content" not in msg:
        raise KeyError(f"Schema mismatch — expected 'task'+'content', got: {list(msg.keys())}")
    return f"Processed: {msg['content'][:30]}..."


def demo_semantic_mismatch() -> None:
    logger.info("=== Failure 1: Semantic mismatch ===")
    msg = AgentAMessage(action="analyze", text="quarterly earnings report")
    try:
        result = agent_b_handler({"action": msg.action, "text": msg.text})
        logger.info("Result: %s", result)
    except KeyError as exc:
        logger.error("Mismatch: %s", exc)


# ──────────────────────────────────────────────
# Failure mode 2: No capability discovery
# ──────────────────────────────────────────────

AGENT_REGISTRY: dict[str, list[str]] = {}  # Empty — no discovery


def find_agent_for_task(task: str) -> str | None:
    """Without a registry, discovery is impossible."""
    for agent_id, capabilities in AGENT_REGISTRY.items():
        if task in capabilities:
            return agent_id
    return None


def demo_no_discovery() -> None:
    logger.info("=== Failure 2: No capability discovery ===")
    agent = find_agent_for_task("sql-generation")
    if agent is None:
        logger.error(
            "No agent found for 'sql-generation'. "
            "Three teams built one — none registered it."
        )


# ──────────────────────────────────────────────
# Failure mode 3: No lifecycle management
# ──────────────────────────────────────────────

def call_agent_synchronously(task: str) -> str:
    """
    Pretends to call a long-running agent synchronously.
    In reality this would block or time out for multi-minute tasks.
    """
    import time
    logger.info("Calling agent for: %s", task)
    # Simulating a task that takes 3 minutes — would timeout here
    # time.sleep(180)
    raise TimeoutError("Agent did not respond within 30s — task lifecycle unknown.")


def demo_lifecycle_opacity() -> None:
    logger.info("=== Failure 3: Lifecycle opacity ===")
    try:
        result = call_agent_synchronously("generate 500-row synthetic dataset")
        logger.info("Result: %s", result)
    except TimeoutError as exc:
        logger.error(
            "%s\nIs the task still running? Completed? Failed? We don't know.", exc
        )


# ──────────────────────────────────────────────
# What interoperability must provide
# ──────────────────────────────────────────────

def show_requirements() -> None:
    logger.info("=== Interoperability requirements ===")
    requirements = {
        "Semantic contracts": "Shared schema / capability manifest so agents understand each other",
        "Capability advertisement": "Registry where agents publish what they can do",
        "Task lifecycle": "States: submitted → running → completed / failed / cancelled",
        "Trust scoping": "Explicit delegation — what the delegatee is allowed to do",
        "Transport agnosticism": "HTTP, message queue, in-process — caller shouldn't care",
    }
    for name, desc in requirements.items():
        logger.info("  [%s] %s", name, desc)


if __name__ == "__main__":
    demo_semantic_mismatch()
    demo_no_discovery()
    demo_lifecycle_opacity()
    show_requirements()
```

Run it:
```bash
python day_1_interop_problem.py
```

Expected output shows three error scenarios followed by the five requirements. No external dependencies needed.

## Exercises

1. Extend `AGENT_REGISTRY` with three fake agents (each with 2–3 capabilities) and rewrite `find_agent_for_task` to return all matching agents sorted by a mock "reliability score." Add a unit test.

2. Design (in code, as a dataclass) a `TaskStatus` enum and a `TaskHandle` that lets a caller poll for completion without blocking. The handle should expose: `task_id`, `status`, `result` (optional), `error` (optional). No LLM needed.

3. Write a `translate_message` function that maps `AgentAMessage` → `AgentBExpects` using a field mapping dict. Unit-test with at least two mappings.

4. **Research exercise:** Look up the Google A2A protocol specification (a2a-protocol.com or GitHub). Write three bullet points in your own words describing how it addresses the three failure modes above.

(Solutions: ../solutions/week-09/day-1.md)

## Self-check
- [ ] I can name three failure modes of siloed agents and why each breaks in practice
- [ ] I can explain what "semantic contract" means between agents
- [ ] I can sketch the five requirements for an interoperability layer from memory
- [ ] I can implement a basic task status model without reference

## Supplementary links
- [Google A2A Protocol](https://github.com/google-deepmind/a2a)
- [Agent-to-Agent specification — a2a-protocol.com](https://a2aproject.github.io/A2A/)
- [Microservices contract patterns (Martin Fowler)](https://martinfowler.com/articles/microservices.html)
