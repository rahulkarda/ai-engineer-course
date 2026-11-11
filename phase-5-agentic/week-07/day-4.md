# Phase 5 · Week 07 · Day 4 — Role Specialization & Coordination

> Objective: Design role-based specialist agents and implement coordinator patterns that route tasks by capability.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

**Role specialization** is the practice of giving each agent a narrow, well-defined responsibility. A specialist outperforms a generalist on its domain because:
- Its system prompt is focused — no conflicting instructions
- Its tool set matches its domain
- Its context window is spent on relevant history only
- It can be fine-tuned, prompted, or swapped independently

### Designing roles

A good role has:
1. **Single responsibility** — one clear primary function
2. **Bounded interface** — known input schema, known output schema
3. **Capability declaration** — what tasks it can accept
4. **Graceful refusal** — it says "not my domain" rather than silently failing

### Capability registry

A **coordinator** (not necessarily a supervisor — it may be a pure routing layer) maintains a registry mapping capabilities to agents:

```python
registry = {
    "code_review":    [code_reviewer_agent],
    "summarization":  [summarizer_agent],
    "data_analysis":  [analyst_agent, analyst_agent_2],  # pool
    "translation":    [translator_agent],
}
```

When a task arrives with `required_capability="code_review"`, the coordinator picks the best available agent from the pool (round-robin, least-loaded, etc.).

### Coordination patterns

**1. Capability-based routing (today's focus)**  
Coordinator inspects task metadata → routes to matching specialist.

**2. Auction / bidding**  
Coordinator broadcasts task → agents bid (e.g., "I can do this with confidence 0.9") → coordinator awards to highest bidder.

**3. Contract net protocol**  
Formalised version of auction: announce → bid → award → execute → report. Classic MAS protocol (FIPA standard).

**4. Hierarchical delegation**  
Supervisor decomposes into sub-tasks and recursively delegates each sub-task to a sub-supervisor or specialist.

### Conflict resolution preview

When two specialists produce conflicting results (e.g., security agent says "block this", safety agent says "allow this"):
- **Priority rules**: one role always overrides another (security > usability)
- **Voting**: majority wins
- **Escalation**: escalate to a higher-level arbitrator
- **Debate**: agents argue for their position (covered in Day 5)

## How it works internally

A `Coordinator` is a routing function: `(task) → agent`. It doesn't hold business logic — it just matches. The key data structures:

```python
@dataclass
class AgentSpec:
    name: str
    capabilities: frozenset[str]
    max_concurrent: int
    current_load: int
```

The coordinator scores available agents by `(capability_match, load)` and selects the best. For pools of identical specialists, round-robin or load-based selection suffices.

**Health-checking**: before routing, coordinator checks if agent is alive (last heartbeat within threshold). If not, route to next agent in pool.

## Worked example

```python
"""
day4_specialization.py — Role registry, capability routing, load balancing.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Task and result models ─────────────────────────────────────────────────────

class Task(BaseModel):
    id: str
    required_capability: str
    payload: dict[str, Any]
    priority: int = 0


class TaskResult(BaseModel):
    task_id: str
    agent_name: str
    output: str
    success: bool = True
    error: str = ""


# ── Agent spec ────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    name: str
    capabilities: frozenset[str]
    system_prompt: str
    max_concurrent: int = 1
    current_load: int = field(default=0, compare=False)
    last_heartbeat: float = field(default_factory=time.time, compare=False)

    def is_healthy(self, threshold: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < threshold

    def is_available(self) -> bool:
        return self.current_load < self.max_concurrent and self.is_healthy()


# ── Coordinator ────────────────────────────────────────────────────────────────

class Coordinator:
    def __init__(self) -> None:
        self._agents: list[AgentSpec] = []
        self._round_robin: dict[str, int] = {}

    def register(self, spec: AgentSpec) -> None:
        self._agents.append(spec)
        logger.info("coordinator | registered %s caps=%s", spec.name, spec.capabilities)

    def route(self, task: Task) -> AgentSpec | None:
        candidates = [
            a for a in self._agents
            if task.required_capability in a.capabilities and a.is_available()
        ]
        if not candidates:
            logger.warning("coordinator | no available agent for capability: %s", task.required_capability)
            return None

        # Round-robin within capability pool
        cap = task.required_capability
        idx = self._round_robin.get(cap, 0) % len(candidates)
        self._round_robin[cap] = idx + 1
        chosen = candidates[idx]
        logger.info("coordinator | routing %s → %s", task.id, chosen.name)
        return chosen

    def list_capabilities(self) -> set[str]:
        caps: set[str] = set()
        for a in self._agents:
            caps.update(a.capabilities)
        return caps


# ── Specialist runner ──────────────────────────────────────────────────────────

class SpecialistRunner:
    """Executes a task using the given AgentSpec + Anthropic client."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def execute(self, spec: AgentSpec, task: Task) -> TaskResult:
        spec.current_load += 1
        try:
            user_content = str(task.payload)
            response = self._client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=256,
                system=spec.system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            output = response.content[0].text
            spec.last_heartbeat = time.time()
            return TaskResult(task_id=task.id, agent_name=spec.name, output=output)
        except Exception as exc:
            logger.exception("specialist %s failed task %s", spec.name, task.id)
            return TaskResult(
                task_id=task.id, agent_name=spec.name,
                output="", success=False, error=str(exc),
            )
        finally:
            spec.current_load -= 1


# ── Driver ─────────────────────────────────────────────────────────────────────

def main() -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    runner = SpecialistRunner(client)

    coordinator = Coordinator()
    coordinator.register(AgentSpec(
        name="code-reviewer",
        capabilities=frozenset({"code_review", "code_analysis"}),
        system_prompt="You are a senior code reviewer. Find bugs and style issues. Be concise.",
        max_concurrent=2,
    ))
    coordinator.register(AgentSpec(
        name="summarizer-1",
        capabilities=frozenset({"summarization"}),
        system_prompt="You summarise text in 2 sentences max.",
        max_concurrent=1,
    ))
    coordinator.register(AgentSpec(
        name="summarizer-2",
        capabilities=frozenset({"summarization"}),
        system_prompt="You summarise text in 2 sentences max.",
        max_concurrent=1,
    ))
    coordinator.register(AgentSpec(
        name="translator",
        capabilities=frozenset({"translation"}),
        system_prompt="You translate text to the target language specified in the task.",
        max_concurrent=1,
    ))

    tasks = [
        Task(id="t1", required_capability="code_review",
             payload={"code": "def f(x): return x*x\n# no type hints"}),
        Task(id="t2", required_capability="summarization",
             payload={"text": "Python 3.12 introduces new syntax for type parameter declarations."}),
        Task(id="t3", required_capability="summarization",
             payload={"text": "LangGraph enables stateful multi-step LLM workflows."}),
        Task(id="t4", required_capability="translation",
             payload={"text": "Hello, world!", "target_language": "French"}),
        Task(id="t5", required_capability="unknown_cap",
             payload={}),
    ]

    for task in tasks:
        spec = coordinator.route(task)
        if spec is None:
            logger.warning("no route for task %s", task.id)
            continue
        result = runner.execute(spec, task)
        logger.info("task %s | success=%s | output: %s", result.task_id, result.success, result.output[:60])


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `bid` method to `AgentSpec`: it returns a confidence float (0–1) for a given capability. Modify the coordinator to use max-bid routing instead of round-robin.
2. Implement a `HealthMonitor` thread that periodically sends a heartbeat task to each registered agent and updates `last_heartbeat`. If an agent misses 3 consecutive heartbeats, mark it unhealthy.
3. Add a `fallback_capability` to `AgentSpec`: if the primary capability produces `success=False`, automatically re-route to an agent with the fallback capability.
4. Build a `ContractNetCoordinator` that: (a) broadcasts task to all agents, (b) collects bids for 100ms, (c) awards to highest bidder.
5. Write a pytest test that registers 3 summarizer agents and verifies that 6 consecutive tasks are distributed in round-robin fashion (2 per agent).

(Solutions: ../solutions/week-07/day-4.md)

## Self-check

- [ ] I can define what makes a good specialist agent role
- [ ] I can implement a capability registry and route tasks by capability
- [ ] I can implement round-robin and load-aware routing
- [ ] I understand the contract net protocol at a conceptual level

## Supplementary links

- [Contract Net Protocol (FIPA)](http://www.fipa.org/specs/fipa00029/)
- [LangGraph Supervisor tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [CrewAI Role-Based Agents](https://docs.crewai.com/concepts/agents)
