# Solutions — Week 11 · Day 1: Pydantic AI Typed Agents & Dependency Injection

## Exercise 1 — Translation agent with deps

```python
# src/agents/translation_agent.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)


class TranslationResult(BaseModel):
    translated_text: str
    detected_source_lang: str


@dataclass
class TranslationDeps:
    target_language: str


translation_agent: Agent[TranslationDeps, TranslationResult] = Agent(
    model="anthropic:claude-haiku-5-5",
    deps_type=TranslationDeps,
    result_type=TranslationResult,
    system_prompt="You are a professional translator. Return structured JSON.",
)


@translation_agent.system_prompt
def add_target_language(ctx: RunContext[TranslationDeps]) -> str:
    return f"Translate into: {ctx.deps.target_language}"


async def translate(text: str, deps: TranslationDeps) -> TranslationResult:
    logger.info("Translating to %s", deps.target_language)
    result = await translation_agent.run(f"Translate: {text}", deps=deps)
    return result.data
```

**Explanation:** `@translation_agent.system_prompt` is a dynamic prompt decorator. It receives `RunContext[TranslationDeps]` so it can read `ctx.deps.target_language`. This keeps the target language out of the static system prompt string and allows per-call customisation.

---

## Exercise 2 — Swap model, use TestModel, inspect messages

```python
# tests/unit/test_research_agent.py
import asyncio
import pytest
from pydantic_ai.models.test import TestModel
from src.agents.research_agent import ResearchDeps, ResearchResult, research_agent


@pytest.mark.asyncio
async def test_research_with_test_model():
    deps = ResearchDeps(
        knowledge_base={"test": "test value"},
        max_points=2,
    )
    # Override model to TestModel — no API key needed
    with research_agent.override(model=TestModel()):
        result = await research_agent.run("test topic", deps=deps)

    messages = result.all_messages()
    # TestModel returns a fixed default result; the agent runs without errors
    assert result.data is not None
    # Messages include at least a SystemPrompt and UserPrompt
    assert len(messages) >= 1
    print("Messages:", messages)
```

`TestModel` returns `None`/default values without calling any real API. It exercises the agent's plumbing (prompt assembly, result parsing) but not the LLM's reasoning.

---

## Exercise 3 — Per-run retries (why Agent() doesn't accept dynamic values)

`Agent(retries=...)` accepts only a static integer at construction time because the agent schema (including retry policy) is compiled once when the `Agent` object is created — before any `run()` call. You cannot pass `ctx.deps.max_retries` there because deps don't exist yet.

**Correct approach:** Use `agent.run(..., model_settings=ModelSettings(max_retries=deps.max_retries))` at call time:

```python
from pydantic_ai.settings import ModelSettings

async def research(topic: str, deps: ResearchDeps) -> ResearchResult:
    result = await research_agent.run(
        f"Research: {topic}",
        deps=deps,
        model_settings=ModelSettings(max_retries=deps.max_retries),
    )
    return result.data
```

Alternatively, set `retries` on the `Agent` at a reasonable static default and document that it's not per-call configurable.

---

## Exercise 4 — Pytest test with mocked result

```python
# tests/unit/test_research_agent_mock.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.research_agent import ResearchDeps, ResearchResult, research


@pytest.mark.asyncio
async def test_research_returns_valid_result():
    fixed_result = ResearchResult(
        summary="Test summary",
        key_points=["point 1", "point 2"],
        confidence=0.85,
    )
    mock_run_result = MagicMock()
    mock_run_result.data = fixed_result
    mock_run_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)

    with patch("src.agents.research_agent.research_agent.run", new_callable=AsyncMock) as mock:
        mock.return_value = mock_run_result
        result = await research("test topic", ResearchDeps(knowledge_base={}, max_points=3))

    assert 0.0 <= result.confidence <= 1.0
    assert result.summary == "Test summary"
    mock.assert_awaited_once()
```
