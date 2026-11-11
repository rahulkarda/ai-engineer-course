# Solutions — Week 06 · Day 3 — Dynamic Agent/Tool Selection

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Add a scoring function that ranks tools by tag overlap with the query, falling back to LLM selection only when the top score is below a threshold.**

Design: `_tag_score(query, tool)` counts how many of the tool's tags appear as words in the lower-cased query. If the top-scoring tool exceeds `tag_threshold` (default `1`), dispatch directly; only call the LLM when the best tag score is below the threshold.

```python
"""
day3_ex1.py — DynamicSelector with tag-overlap pre-filter and LLM fallback.

Run: python day3_ex1.py
Env: ANTHROPIC_API_KEY
"""
import json
import logging
import os
import re
import unittest.mock
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ToolMetadata(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    callable_ref: Any = Field(exclude=True)


class SelectionResult(BaseModel):
    tool_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    source: str = "llm"  # "tag_match" | "llm"


class NoSuitableToolError(RuntimeError):
    """Raised when no tool meets the selection threshold."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> Any:
        def decorator(fn: Any) -> Any:
            self._tools[name] = ToolMetadata(
                name=name,
                description=description,
                tags=tags or [],
                version=version,
                callable_ref=fn,
            )
            logger.info("Registered tool: %s tags=%s", name, tags)
            return fn
        return decorator

    def all_tools(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolMetadata | None:
        return self._tools.get(name)

    def summary_json(self) -> str:
        return json.dumps(
            [{"name": t.name, "description": t.description, "tags": t.tags}
             for t in self._tools.values()],
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tag-overlap scoring
# ---------------------------------------------------------------------------

def _tag_score(query: str, tool: ToolMetadata) -> int:
    """
    Count how many of the tool's tags appear as whole words in the query.

    Case-insensitive. Each tag counts at most once.
    """
    query_lower = query.lower()
    return sum(
        1
        for tag in tool.tags
        if re.search(rf"\b{re.escape(tag.lower())}\b", query_lower)
    )


def _select_by_tags(
    query: str,
    tools: list[ToolMetadata],
    threshold: int = 1,
) -> SelectionResult | None:
    """
    Return a SelectionResult via tag matching when top score >= threshold.
    Returns None when no tool meets the threshold.
    """
    scored = [(t, _tag_score(query, t)) for t in tools]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_tool, best_score = scored[0] if scored else (None, 0)

    if best_score >= threshold and best_tool is not None:
        logger.info(
            "Tag match: tool=%s score=%d (threshold=%d)",
            best_tool.name,
            best_score,
            threshold,
        )
        return SelectionResult(
            tool_name=best_tool.name,
            confidence=min(1.0, best_score / max(len(best_tool.tags), 1)),
            reasoning=f"Tag overlap score {best_score} >= threshold {threshold}",
            source="tag_match",
        )
    return None


# ---------------------------------------------------------------------------
# Dynamic selector with tag-first, LLM-fallback
# ---------------------------------------------------------------------------

SELECTION_PROMPT = """You are a tool-selection expert.

Available tools:
{tools_json}

Task:
{task}

Pick the single best tool. Respond with valid JSON only:
{{
  "tool_name": "<exact name from the list>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""


class DynamicSelector:
    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "claude-sonnet-5-5",
        confidence_threshold: float = 0.6,
        tag_threshold: int = 1,
    ) -> None:
        self._registry = registry
        self._model = model
        self._threshold = confidence_threshold
        self._tag_threshold = tag_threshold
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def select(self, task: str) -> SelectionResult:
        """
        1. Try tag-overlap scoring first.
        2. Fall back to LLM only when tag score < tag_threshold.
        """
        tools = self._registry.all_tools()
        if not tools:
            raise NoSuitableToolError("Registry is empty")

        # Fast path: tag match
        tag_result = _select_by_tags(task, tools, threshold=self._tag_threshold)
        if tag_result is not None:
            return tag_result

        # Slow path: LLM selection
        logger.info("No tag match — using LLM for task: %r", task)
        prompt = SELECTION_PROMPT.format(
            tools_json=self._registry.summary_json(),
            task=task,
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)
        result = SelectionResult(**{**data, "source": "llm"})

        if result.confidence < self._threshold:
            raise NoSuitableToolError(
                f"Best LLM candidate '{result.tool_name}' confidence "
                f"{result.confidence:.2f} < {self._threshold}"
            )
        return result

    def dispatch(self, task: str, **kwargs: Any) -> Any:
        selection = self.select(task)
        tool = self._registry.get(selection.tool_name)
        if tool is None:
            raise KeyError(f"Tool '{selection.tool_name}' not in registry")
        logger.info(
            "Dispatching to %s (source=%s confidence=%.2f)",
            selection.tool_name,
            selection.source,
            selection.confidence,
        )
        return tool.callable_ref(**kwargs)


# ---------------------------------------------------------------------------
# Demo tools
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.register("web_search", "Search the internet.", tags=["search", "web"])
def web_search(query: str) -> str:
    return f"[mock] Web results for: {query}"


@registry.register("code_executor", "Execute Python code.", tags=["code", "python"])
def code_executor(code: str) -> str:
    return "[mock] Executed"


@registry.register("database_query", "Run SQL queries.", tags=["sql", "database"])
def database_query(sql: str) -> str:
    return f"[mock] Query result: {sql}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    selector = DynamicSelector(registry, tag_threshold=1)
    cases = [
        ("Search the web for Python docs", {"query": "Python docs"}),
        ("Execute some python code", {"code": "print(1)"}),
        ("Run a sql query on the database", {"sql": "SELECT 1"}),
    ]
    for task, kwargs in cases:
        print(f"\nTask: {task}")
        try:
            out = selector.dispatch(task, **kwargs)
            print(f"Result: {out}")
        except NoSuitableToolError as exc:
            print(f"No tool: {exc}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_registry(*tools: tuple[str, list[str]]) -> ToolRegistry:
    """Helper: build a fresh registry with stub tools."""
    reg = ToolRegistry()
    for name, tags in tools:
        @reg.register(name, f"Description of {name}", tags=tags)
        def stub(**kwargs: Any) -> str:
            return f"result from {name}"
    return reg


def test_tag_score_counts_overlapping_tags() -> None:
    tool = ToolMetadata(
        name="web_search",
        description="search web",
        tags=["search", "web", "retrieval"],
        callable_ref=None,
    )
    assert _tag_score("search the web for docs", tool) == 2
    assert _tag_score("execute code", tool) == 0


def test_select_by_tags_returns_best_match() -> None:
    tools = [
        ToolMetadata(name="a", description="", tags=["search", "web"], callable_ref=None),
        ToolMetadata(name="b", description="", tags=["code", "python"], callable_ref=None),
    ]
    result = _select_by_tags("search the web", tools, threshold=1)
    assert result is not None
    assert result.tool_name == "a"
    assert result.source == "tag_match"


def test_select_by_tags_returns_none_below_threshold() -> None:
    tools = [
        ToolMetadata(name="a", description="", tags=["sql"], callable_ref=None),
    ]
    result = _select_by_tags("something completely unrelated", tools, threshold=1)
    assert result is None


def test_selector_uses_tag_path_without_llm() -> None:
    """Tag match should not call the LLM."""
    reg = _make_registry(("web_search", ["search", "web"]))
    selector = DynamicSelector(reg, tag_threshold=1)
    selector._client = unittest.mock.MagicMock()  # LLM must not be called

    result = selector.select("search the web for news")

    selector._client.messages.create.assert_not_called()
    assert result.tool_name == "web_search"
    assert result.source == "tag_match"


def test_selector_falls_back_to_llm_on_no_tag_match() -> None:
    """No tag match → LLM is called."""
    reg = _make_registry(("email_sender", ["email", "notification"]))
    selector = DynamicSelector(reg, tag_threshold=1)

    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    llm_resp = unittest.mock.MagicMock()
    llm_resp.content = [
        unittest.mock.MagicMock(
            text='{"tool_name": "email_sender", "confidence": 0.8, "reasoning": "best fit"}'
        )
    ]
    mock_client.messages.create.return_value = llm_resp
    selector._client = mock_client

    result = selector.select("I need to send a notification to the team")

    mock_client.messages.create.assert_called_once()
    assert result.tool_name == "email_sender"
    assert result.source == "llm"


def test_selector_raises_no_suitable_tool_on_empty_registry() -> None:
    reg = ToolRegistry()
    selector = DynamicSelector(reg)
    try:
        selector.select("anything")
        assert False, "Should raise"
    except NoSuitableToolError:
        pass
```


## Exercise 2

**Implement tool versioning: each tool has a `version` field; the selector always picks the latest version when multiple tools share the same name.**

Design: `ToolRegistry` stores tools in a `dict[str, list[ToolMetadata]]` keyed by name. `get(name)` returns the entry with the highest semantic version. A helper `_latest_version(tools)` uses `packaging.version.Version` for comparison (falls back to string sort when unavailable).

```python
"""
day3_ex2.py — ToolRegistry with versioning; latest version always wins.

Run: python day3_ex2.py
Env: ANTHROPIC_API_KEY
"""
import json
import logging
import os
import unittest.mock
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version comparison helper
# ---------------------------------------------------------------------------

def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a semver string into an integer tuple for comparison."""
    try:
        parts = v.lstrip("v").split(".")
        return tuple(int(p) for p in parts)
    except ValueError:
        return (0,)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ToolMetadata(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    callable_ref: Any = Field(exclude=True)


class SelectionResult(BaseModel):
    tool_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    version_selected: str = "1.0.0"


class NoSuitableToolError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Versioned registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Supports multiple versions of the same tool name.
    `get(name)` returns the highest-version entry.
    `all_tools()` returns only the latest version per name (for the LLM prompt).
    """

    def __init__(self) -> None:
        # name → list of all registered versions
        self._versions: dict[str, list[ToolMetadata]] = {}

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> Any:
        def decorator(fn: Any) -> Any:
            meta = ToolMetadata(
                name=name,
                description=description,
                tags=tags or [],
                version=version,
                callable_ref=fn,
            )
            self._versions.setdefault(name, []).append(meta)
            logger.info("Registered tool: %s v%s", name, version)
            return fn
        return decorator

    def get(self, name: str) -> ToolMetadata | None:
        """Return the latest version of the named tool, or None."""
        versions = self._versions.get(name)
        if not versions:
            return None
        return max(versions, key=lambda t: _parse_version(t.version))

    def all_tools(self) -> list[ToolMetadata]:
        """Return one entry per tool name — the latest version only."""
        return [self.get(name) for name in self._versions]  # type: ignore[misc]

    def summary_json(self) -> str:
        return json.dumps(
            [
                {"name": t.name, "description": t.description, "tags": t.tags, "version": t.version}
                for t in self.all_tools()
            ],
            indent=2,
        )


# ---------------------------------------------------------------------------
# Selector (unchanged logic, uses versioned registry)
# ---------------------------------------------------------------------------

SELECTION_PROMPT = """You are a tool-selection expert.

Available tools (latest versions only):
{tools_json}

Task:
{task}

Pick the single best tool. Respond with valid JSON only:
{{
  "tool_name": "<exact name>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""


class DynamicSelector:
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
        prompt = SELECTION_PROMPT.format(
            tools_json=self._registry.summary_json(),
            task=task,
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)

        if data["confidence"] < self._threshold:
            raise NoSuitableToolError(
                f"Confidence {data['confidence']:.2f} < {self._threshold}"
            )

        tool = self._registry.get(data["tool_name"])
        version = tool.version if tool else "unknown"
        return SelectionResult(
            tool_name=data["tool_name"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            version_selected=version,
        )

    def dispatch(self, task: str, **kwargs: Any) -> Any:
        selection = self.select(task)
        tool = self._registry.get(selection.tool_name)
        if tool is None:
            raise KeyError(f"Tool '{selection.tool_name}' not found")
        logger.info(
            "Dispatching to %s v%s", selection.tool_name, selection.version_selected
        )
        return tool.callable_ref(**kwargs)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    reg = ToolRegistry()

    @reg.register("web_search", "Search the internet", tags=["search"], version="1.0.0")
    def web_search_v1(query: str) -> str:
        return f"[v1] Web results: {query}"

    @reg.register("web_search", "Search the internet (improved)", tags=["search", "web"], version="2.1.0")
    def web_search_v2(query: str) -> str:
        return f"[v2] Web results: {query}"

    latest = reg.get("web_search")
    print(f"Latest web_search: v{latest.version if latest else 'N/A'}")

    summary = reg.summary_json()
    print(f"\nRegistry summary:\n{summary}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_returns_latest_version() -> None:
    reg = ToolRegistry()

    @reg.register("tool_a", "desc", version="1.0.0")
    def v1(x: str) -> str:
        return "v1"

    @reg.register("tool_a", "desc", version="2.0.0")
    def v2(x: str) -> str:
        return "v2"

    @reg.register("tool_a", "desc", version="1.5.0")
    def v15(x: str) -> str:
        return "v1.5"

    latest = reg.get("tool_a")
    assert latest is not None
    assert latest.version == "2.0.0"
    assert latest.callable_ref() == "v2"


def test_registry_all_tools_deduplicated() -> None:
    """all_tools() returns one entry per name (the latest)."""
    reg = ToolRegistry()

    @reg.register("tool_a", "desc", version="1.0.0")
    def a1(x: str) -> str:
        return ""

    @reg.register("tool_a", "desc", version="3.0.0")
    def a3(x: str) -> str:
        return ""

    @reg.register("tool_b", "desc", version="1.0.0")
    def b1(x: str) -> str:
        return ""

    tools = reg.all_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"tool_a", "tool_b"}
    a_tool = next(t for t in tools if t.name == "tool_a")
    assert a_tool.version == "3.0.0"


def test_parse_version_ordering() -> None:
    assert _parse_version("2.0.0") > _parse_version("1.9.9")
    assert _parse_version("1.10.0") > _parse_version("1.9.0")
    assert _parse_version("v1.2.3") == (1, 2, 3)


def test_selector_picks_latest_version_when_dispatching() -> None:
    """Selector dispatch calls the latest version's callable."""
    reg = ToolRegistry()
    v1_called = []
    v2_called = []

    @reg.register("search", "search tool", tags=["search"], version="1.0.0")
    def search_v1(query: str) -> str:
        v1_called.append(True)
        return "v1"

    @reg.register("search", "search tool improved", tags=["search"], version="2.0.0")
    def search_v2(query: str) -> str:
        v2_called.append(True)
        return "v2"

    selector = DynamicSelector(reg)
    mock_client = unittest.mock.MagicMock(spec=anthropic.Anthropic)
    llm_resp = unittest.mock.MagicMock()
    llm_resp.content = [
        unittest.mock.MagicMock(
            text='{"tool_name": "search", "confidence": 0.9, "reasoning": "best"}'
        )
    ]
    mock_client.messages.create.return_value = llm_resp
    selector._client = mock_client

    result = selector.dispatch("search for news", query="news")
    assert result == "v2"
    assert v2_called == [True]
    assert v1_called == []


def test_registry_get_none_for_unknown_tool() -> None:
    reg = ToolRegistry()
    assert reg.get("nonexistent") is None


def test_registry_summary_json_shows_latest_version_only() -> None:
    reg = ToolRegistry()

    @reg.register("t", "desc", version="1.0.0")
    def t1() -> None:
        pass

    @reg.register("t", "desc", version="9.0.0")
    def t9() -> None:
        pass

    summary = json.loads(reg.summary_json())
    assert len(summary) == 1
    assert summary[0]["version"] == "9.0.0"
```


## Exercise 3

**Build a `ToolComposer` that chains two selected tools in sequence, passing the output of the first as input to the second.**

Design: `ToolComposer` takes two task descriptions. It calls `DynamicSelector.select()` for each, resolves the tools from the registry, then calls `tool1(input) → intermediate` and `tool2(intermediate) → final`. The second tool always receives a single `input` keyword argument containing the first tool's output string.

```python
"""
day3_ex3.py — ToolComposer: chain two dynamically selected tools in sequence.

Run: python day3_ex3.py
Env: ANTHROPIC_API_KEY
"""
import json
import logging
import os
import unittest.mock
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ToolMetadata(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    callable_ref: Any = Field(exclude=True)


class SelectionResult(BaseModel):
    tool_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class NoSuitableToolError(RuntimeError):
    pass


@dataclass
class ComposedResult:
    """Result of a two-tool chain."""
    first_tool: str
    second_tool: str
    intermediate: str  # output of first tool
    final: str         # output of second tool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> Any:
        def decorator(fn: Any) -> Any:
            self._tools[name] = ToolMetadata(
                name=name,
                description=description,
                tags=tags or [],
                version=version,
                callable_ref=fn,
            )
            logger.info("Registered tool: %s", name)
            return fn
        return decorator

    def all_tools(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolMetadata | None:
        return self._tools.get(name)

    def summary_json(self) -> str:
        return json.dumps(
            [{"name": t.name, "description": t.description, "tags": t.tags}
             for t in self._tools.values()],
            indent=2,
        )


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

SELECTION_PROMPT = """Available tools:
{tools_json}

Task: {task}

Pick the single best tool. Respond with valid JSON only:
{{"tool_name": "<exact name>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


class DynamicSelector:
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
        prompt = SELECTION_PROMPT.format(
            tools_json=self._registry.summary_json(),
            task=task,
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text.strip())
        result = SelectionResult(**data)
        if result.confidence < self._threshold:
            raise NoSuitableToolError(
                f"Confidence {result.confidence:.2f} < {self._threshold}"
            )
        return result


# ---------------------------------------------------------------------------
# ToolComposer
# ---------------------------------------------------------------------------

class ToolComposer:
    """
    Chain two dynamically selected tools in sequence.

    `compose(task1, task2, first_input)`:
      1. Selects and calls the best tool for task1 with first_input.
      2. Passes the output of step 1 as `input=` to the best tool for task2.
      3. Returns a ComposedResult with both outputs.
    """

    def __init__(self, selector: DynamicSelector) -> None:
        self._selector = selector

    def compose(
        self,
        task1: str,
        task2: str,
        first_input: str,
        **first_kwargs: Any,
    ) -> ComposedResult:
        """
        Args:
            task1: Description of what the first tool should do.
            task2: Description of what the second tool should do with the result.
            first_input: The initial input string passed to the first tool.
            **first_kwargs: Additional keyword arguments for the first tool callable.
        """
        logger.info("Composing: [%s] → [%s]", task1, task2)

        # Step 1: select and invoke first tool
        sel1 = self._selector.select(task1)
        tool1 = self._selector._registry.get(sel1.tool_name)
        if tool1 is None:
            raise KeyError(f"Tool '{sel1.tool_name}' not found in registry")

        logger.info("Step 1: calling %s", tool1.name)
        intermediate = tool1.callable_ref(first_input, **first_kwargs)

        # Step 2: select and invoke second tool — receives intermediate as `input`
        sel2 = self._selector.select(task2)
        tool2 = self._selector._registry.get(sel2.tool_name)
        if tool2 is None:
            raise KeyError(f"Tool '{sel2.tool_name}' not found in registry")

        logger.info("Step 2: calling %s with intermediate output", tool2.name)
        final = tool2.callable_ref(input=intermediate)

        return ComposedResult(
            first_tool=tool1.name,
            second_tool=tool2.name,
            intermediate=str(intermediate),
            final=str(final),
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.register("web_search", "Search the internet for information.", tags=["search", "web"])
def web_search(query: str) -> str:
    return f"[mock search results for: {query}]"


@registry.register(
    "text_summarizer",
    "Summarise a block of text into key bullet points.",
    tags=["summarize", "text", "nlp"],
)
def text_summarizer(input: str) -> str:
    return f"[mock summary of: {input[:60]}...]"


@registry.register(
    "sentiment_analyzer",
    "Analyse the sentiment of a given text.",
    tags=["sentiment", "analysis", "nlp"],
)
def sentiment_analyzer(input: str) -> str:
    return f"[mock sentiment: positive (input length {len(input)})]"


def main() -> None:
    selector = DynamicSelector(registry)
    composer = ToolComposer(selector)

    # NOTE: in a real run this calls the LLM; in tests we mock it.
    # For demo we patch manually to avoid requiring a live key.
    print("ToolComposer demo — use real ANTHROPIC_API_KEY to run live.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _mock_selector_for_tools(
    selector: DynamicSelector,
    tool_sequence: list[str],
) -> None:
    """
    Patch selector.select to return tools from tool_sequence in order.
    """
    call_count = {"n": 0}

    def fake_select(task: str) -> SelectionResult:
        name = tool_sequence[call_count["n"] % len(tool_sequence)]
        call_count["n"] += 1
        return SelectionResult(
            tool_name=name, confidence=0.95, reasoning="mocked"
        )

    selector.select = fake_select  # type: ignore[method-assign]


def test_composer_chains_two_tools_in_order() -> None:
    """First tool output is passed as input to the second tool."""
    reg = ToolRegistry()

    received_by_tool2: list[str] = []

    @reg.register("fetcher", "Fetch data", tags=["fetch"])
    def fetcher(query: str) -> str:
        return f"fetched:{query}"

    @reg.register("processor", "Process data", tags=["process"])
    def processor(input: str) -> str:
        received_by_tool2.append(input)
        return f"processed:{input}"

    selector = DynamicSelector(reg)
    _mock_selector_for_tools(selector, ["fetcher", "processor"])
    composer = ToolComposer(selector)

    result = composer.compose("fetch some data", "process the fetched data", "my_query")

    assert result.first_tool == "fetcher"
    assert result.second_tool == "processor"
    assert result.intermediate == "fetched:my_query"
    assert result.final == "processed:fetched:my_query"
    assert received_by_tool2 == ["fetched:my_query"]


def test_composer_raises_when_tool_not_in_registry() -> None:
    reg = ToolRegistry()

    @reg.register("tool_a", "Tool A", tags=["a"])
    def tool_a(query: str) -> str:
        return "a"

    selector = DynamicSelector(reg)
    _mock_selector_for_tools(selector, ["tool_a", "nonexistent_tool"])
    composer = ToolComposer(selector)

    try:
        composer.compose("task1", "task2", "input")
        assert False, "Should raise KeyError"
    except KeyError:
        pass


def test_composer_intermediate_output_not_empty() -> None:
    reg = ToolRegistry()

    @reg.register("search", "Search", tags=["search"])
    def search(query: str) -> str:
        return "some search results"

    @reg.register("summarize", "Summarize", tags=["summarize"])
    def summarize(input: str) -> str:
        return f"Summary: {input}"

    selector = DynamicSelector(reg)
    _mock_selector_for_tools(selector, ["search", "summarize"])
    composer = ToolComposer(selector)

    result = composer.compose("search web", "summarize results", "test query")

    assert result.intermediate != ""
    assert result.final.startswith("Summary:")


def test_composed_result_fields_populated() -> None:
    result = ComposedResult(
        first_tool="tool_a",
        second_tool="tool_b",
        intermediate="intermediate output",
        final="final output",
    )
    assert result.first_tool == "tool_a"
    assert result.second_tool == "tool_b"
    assert result.intermediate == "intermediate output"
    assert result.final == "final output"


def test_selector_called_twice_in_compose() -> None:
    """compose() calls selector.select exactly twice."""
    reg = ToolRegistry()

    @reg.register("a", "Tool A", tags=["a"])
    def tool_a(query: str) -> str:
        return "a_result"

    @reg.register("b", "Tool B", tags=["b"])
    def tool_b(input: str) -> str:
        return "b_result"

    selector = DynamicSelector(reg)
    select_calls: list[str] = []

    def tracking_select(task: str) -> SelectionResult:
        select_calls.append(task)
        name = "a" if len(select_calls) == 1 else "b"
        return SelectionResult(tool_name=name, confidence=0.9, reasoning="mock")

    selector.select = tracking_select  # type: ignore[method-assign]
    composer = ToolComposer(selector)
    composer.compose("task1", "task2", "input_data")

    assert len(select_calls) == 2
    assert select_calls[0] == "task1"
    assert select_calls[1] == "task2"
```
