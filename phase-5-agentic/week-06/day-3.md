# Phase 5 · Week 06 · Day 3 — Dynamic Agent/Tool Selection

> Objective: Build a runtime tool-selection system that uses an LLM to pick the best tool from a registry given a task description.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Static tool lists break down when agent capabilities grow. Dynamic selection lets an agent inspect a registry at runtime, score each tool against the current task, and dispatch to the winner — without hard-coded if/else logic.

Three layers:

1. **Registry** — a catalogue of tools with structured metadata (name, description, capability tags).
2. **Selector** — uses an LLM to rank candidates, returns a scored selection with reasoning.
3. **Dispatcher** — invokes the chosen tool, handles fallback when no tool meets the confidence threshold.

The "agent-as-tool" pattern extends this: an entire sub-agent can be registered as a tool entry, making the selector the orchestration layer for multi-agent systems.

## How it works internally

**Registration** happens at import time via a decorator that wraps the callable and appends a `ToolMetadata` entry to the registry's internal dict.

**Capability matching** at selection time:

1. Registry serialises all `ToolMetadata` objects to a JSON summary.
2. `DynamicSelector` sends the task + candidate list to the LLM with a structured-output prompt.
3. LLM returns a `SelectionResult` (tool name, confidence 0–1, reasoning).
4. If `confidence < threshold` (default 0.6) the selector raises `NoSuitableToolError` — the caller decides whether to abort or escalate.

**Why LLM scoring outperforms rule-based matching**: free-text task descriptions rarely align with keyword tags; the LLM bridges semantic distance. Confidence score lets callers set acceptable risk.

**Caching**: the registry is a singleton; repeated tool lookups are O(1) dict access.

## Worked example

```python
"""
Dynamic tool selection via a ToolRegistry and LLM-powered DynamicSelector.

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python day3_dynamic_tool_selection.py
"""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class ToolMetadata(BaseModel):
    """Metadata registered alongside every tool."""

    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    callable_ref: Any = Field(exclude=True)  # excluded from serialisation


class SelectionResult(BaseModel):
    """LLM selection output."""

    tool_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class NoSuitableToolError(RuntimeError):
    """Raised when selector confidence falls below threshold."""


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Singleton catalogue of registered tools."""

    _instance: "ToolRegistry | None" = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, ToolMetadata] = {}
        return cls._instance

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
    ) -> Callable:
        """Decorator factory — registers the wrapped function and returns it unchanged."""

        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolMetadata(
                name=name,
                description=description,
                tags=tags or [],
                callable_ref=fn,
            )
            logger.info("Registered tool: %s (tags=%s)", name, tags)
            return fn

        return decorator

    def all_tools(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolMetadata | None:
        return self._tools.get(name)

    def summary_json(self) -> str:
        """Serialise registry to JSON for the LLM prompt."""
        return json.dumps(
            [
                {"name": t.name, "description": t.description, "tags": t.tags}
                for t in self._tools.values()
            ],
            indent=2,
        )


# ── Global registry instance ──────────────────────────────────────────────────

registry = ToolRegistry()


# ── Example tools (registered via decorator) ──────────────────────────────────

@registry.register(
    name="web_search",
    description="Search the internet for current information or news.",
    tags=["search", "web", "retrieval"],
)
def web_search(query: str) -> str:
    logger.info("web_search called with query=%r", query)
    return f"[mock] Web results for: {query}"


@registry.register(
    name="code_executor",
    description="Execute Python code snippets and return stdout.",
    tags=["code", "python", "execution"],
)
def code_executor(code: str) -> str:
    logger.info("code_executor called")
    return f"[mock] Executed code, output: 42"


@registry.register(
    name="database_query",
    description="Run SQL queries against a relational database.",
    tags=["sql", "database", "data"],
)
def database_query(sql: str) -> str:
    logger.info("database_query called with sql=%r", sql)
    return f"[mock] Query result for: {sql}"


@registry.register(
    name="email_sender",
    description="Compose and send an email to one or more recipients.",
    tags=["email", "communication", "notification"],
)
def email_sender(to: str, subject: str, body: str) -> str:
    logger.info("email_sender called to=%r", to)
    return f"[mock] Email sent to {to}"


# ── Selector ──────────────────────────────────────────────────────────────────

SELECTION_PROMPT = """You are a tool-selection expert.

Available tools:
{tools_json}

Task:
{task}

Pick the single best tool for this task. Respond with valid JSON only:
{{
  "tool_name": "<exact name from the list>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}"""


class DynamicSelector:
    """Uses an LLM to pick the best tool from a ToolRegistry for a given task."""

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "claude-sonnet-5-5",
        confidence_threshold: float = 0.6,
    ) -> None:
        self._registry = registry
        self._model = model
        self._threshold = confidence_threshold
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def select(self, task: str) -> SelectionResult:
        """Ask the LLM to pick a tool; raise NoSuitableToolError if confidence is low."""
        prompt = SELECTION_PROMPT.format(
            tools_json=self._registry.summary_json(),
            task=task,
        )
        logger.info("Selecting tool for task: %r", task)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.debug("LLM raw selection: %s", raw)

        result = SelectionResult.model_validate_json(raw)

        if result.confidence < self._threshold:
            raise NoSuitableToolError(
                f"Best candidate '{result.tool_name}' has confidence "
                f"{result.confidence:.2f} < threshold {self._threshold}"
            )
        return result

    def dispatch(self, task: str, **kwargs: Any) -> Any:
        """Select a tool, then call it with supplied kwargs."""
        selection = self.select(task)
        tool_meta = self._registry.get(selection.tool_name)
        if tool_meta is None:
            raise KeyError(f"Tool '{selection.tool_name}' not found in registry")
        logger.info(
            "Dispatching to '%s' (confidence=%.2f): %s",
            selection.tool_name,
            selection.confidence,
            selection.reasoning,
        )
        return tool_meta.callable_ref(**kwargs)


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    selector = DynamicSelector(registry)

    tasks = [
        ("Find the latest Python release notes", {"query": "Python latest release notes"}),
        ("Run a quick sort in Python", {"code": "sorted([3,1,2])"}),
        ("Get all users from the users table", {"sql": "SELECT * FROM users"}),
    ]

    for task_description, call_kwargs in tasks:
        print(f"\nTask: {task_description}")
        try:
            output = selector.dispatch(task_description, **call_kwargs)
            print(f"Result: {output}")
        except NoSuitableToolError as exc:
            logger.warning("No tool selected: %s", exc)
            print("Fallback: escalating to human operator.")


if __name__ == "__main__":
    main()
```

## Exercises

1. Add a `max_candidates: int` parameter to `DynamicSelector.select` that limits the registry to the top-N tools by tag overlap before sending to the LLM (pre-filter to reduce prompt size).
2. Extend `ToolMetadata` with a `cost_estimate: float` field and modify the selection prompt to instruct the LLM to balance capability with cost. Observe how selections change.
3. Register a sub-agent function (one that calls another `DynamicSelector` with its own sub-registry) and verify the outer selector can dispatch to it correctly — the agent-as-tool pattern.
4. Write a `ToolRegistry.unregister(name)` method and add a test verifying that calling `select` after unregistering the only suitable tool raises `NoSuitableToolError`.

(Solutions: ../solutions/week-06/day-3.md)

## Self-check

- [ ] I can explain why a static if/else dispatch breaks at scale and what a registry pattern solves.
- [ ] I can implement `ToolRegistry` with decorator-based registration from memory.
- [ ] I can describe the role of `confidence_threshold` and what happens when it is not met.
- [ ] I can implement `DynamicSelector.dispatch` end-to-end without reference.
- [ ] I can extend the pattern to treat a sub-agent as a first-class tool entry.
