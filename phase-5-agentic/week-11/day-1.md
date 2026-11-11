# Phase 5 · Week 11 · Day 1 — Pydantic AI: Typed Agents & Dependency Injection

> Objective: Build typed agents with Pydantic AI, understand the agent model, and wire dependencies cleanly.  ·  Time: ~1 hr  ·  Prereqs: [[week-10/day-7]]

## Concept

Pydantic AI is a production-grade agent framework built around three ideas:

1. **Type safety end-to-end** — agent inputs, outputs, tool parameters, and system prompts are all typed with Pydantic models.
2. **Dependency injection** — agents declare what they need (DB connections, config, API clients) via a `deps_type`; the runtime injects them at call time, keeping agents pure and testable.
3. **Model-agnostic** — swap `anthropic:claude-sonnet-5-5` for any supported model without changing agent logic.

The core primitive is `Agent[DepsType, ResultType]`. The generic parameters enforce: (a) what the agent receives as context, and (b) what structured type it must return.

### Why this matters over raw API calls

Raw API calls give you untyped dicts. Pydantic AI gives you:
- Compile-time errors if you access a field that doesn't exist on the result
- Validated, coerced output — the LLM's JSON is parsed through your Pydantic model
- Testable agents — inject mock deps, swap model to `TestModel`

## How it works internally

```
Agent.run(user_prompt, deps=deps_instance)
       │
       ├─ resolves system_prompt (static str OR fn(ctx) → str)
       ├─ injects deps into RunContext
       ├─ calls model with messages + tool schemas
       │       │
       │       └─ model returns text / tool call
       │               │
       │               ├─ if tool call → execute tool(ctx) → append result → re-call model
       │               └─ if text → parse into ResultType via Pydantic
       └─ returns RunResult[ResultType]
```

`RunContext[DepsType]` is threaded through every tool and dynamic prompt function — it carries `deps`, `usage` (tokens), `messages`, and `retry` count. Deps are never global state; they're scoped to a single `run()` call.

## Worked example

```python
# src/agents/research_agent.py
"""Typed research agent with dependency injection."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)


# ── 1. Result type ────────────────────────────────────────────────────────────
class ResearchResult(BaseModel):
    """Structured output the agent must produce."""
    summary: str
    key_points: list[str]
    confidence: float  # 0-1


# ── 2. Dependency type ────────────────────────────────────────────────────────
@dataclass
class ResearchDeps:
    """Injected at call time — never hard-coded inside the agent."""
    knowledge_base: dict[str, str]  # topic → text snippet
    max_points: int = 5


# ── 3. Agent definition ───────────────────────────────────────────────────────
research_agent: Agent[ResearchDeps, ResearchResult] = Agent(
    model="anthropic:claude-sonnet-5-5",
    deps_type=ResearchDeps,
    result_type=ResearchResult,
    system_prompt=(
        "You are a research assistant. Summarise the provided knowledge and "
        "return structured JSON matching the required schema."
    ),
)


# ── 4. Dynamic system prompt (uses deps) ─────────────────────────────────────
@research_agent.system_prompt
def add_kb_context(ctx: RunContext[ResearchDeps]) -> str:
    """Append relevant KB entries to the system prompt."""
    entries = "\n".join(
        f"- {k}: {v}" for k, v in ctx.deps.knowledge_base.items()
    )
    return f"Available knowledge:\n{entries}"


# ── 5. Run the agent ─────────────────────────────────────────────────────────
async def research(topic: str, deps: ResearchDeps) -> ResearchResult:
    logger.info("Running research agent for topic=%s", topic)
    result = await research_agent.run(
        f"Research the topic: {topic}. Return at most {deps.max_points} key points.",
        deps=deps,
    )
    logger.info("Tokens used: %s", result.usage())
    return result.data  # type: ResearchResult (validated)


# ── 6. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    deps = ResearchDeps(
        knowledge_base={
            "LLMs": "Large language models are trained on vast text corpora.",
            "Agents": "Agents use LLMs to take actions in an environment.",
        },
        max_points=3,
    )

    result = asyncio.run(research("agentic AI", deps))
    print(result.model_dump_json(indent=2))
```

**Key observations:**
- `Agent[ResearchDeps, ResearchResult]` — generics catch type errors before runtime.
- `@research_agent.system_prompt` decorator — dynamic prompt has access to injected deps.
- `result.data` is already a `ResearchResult` — no manual JSON parsing.
- `result.usage()` — token counts are always available.

### Installing (uv)

```bash
uv add pydantic-ai
# or with logfire tracing:
uv add "pydantic-ai[logfire]"
```

Never `pip install` when the project uses `uv`.

## Exercises

1. Add a `@dataclass` dep `TranslationDeps` with a `target_language: str` field. Build a `translation_agent` that accepts a text string and returns a `TranslationResult(translated_text: str, detected_source_lang: str)`. Wire a dynamic system prompt that reads the target language from deps.

2. Change the model in the worked example to `anthropic:claude-haiku-5-5`. Run it with `TestModel` (from `pydantic_ai.models.test`) to verify the agent completes without a real API key. Print the raw messages from `result.all_messages()`.

3. Add a `max_retries: int` field to `ResearchDeps`. Pass it to `Agent(retries=ctx.deps.max_retries)` — wait, you can't pass dynamic values to `Agent()`; explain why and find the correct way to set per-run retries.

4. Write a pytest test for `research()` that injects `TestModel` and asserts `result.confidence` is between 0 and 1. Mock the model so it returns a fixed `ResearchResult`.

(Solutions: ../solutions/week-11/day-1.md)

## Self-check

- [ ] I can explain what `RunContext[DepsType]` is and why it's generic.
- [ ] I can build an `Agent` with a `deps_type` and `result_type` and run it with injected deps.
- [ ] I can add a dynamic system prompt function that reads from deps.
- [ ] I can swap in `TestModel` for unit tests without any real API calls.
- [ ] I can retrieve token usage from `RunResult.usage()`.

## Supplementary links

- [Pydantic AI docs — Agents](https://ai.pydantic.dev/agents/)
- [Pydantic AI docs — Dependencies](https://ai.pydantic.dev/dependencies/)
- [Pydantic AI docs — Testing](https://ai.pydantic.dev/testing-evals/)
