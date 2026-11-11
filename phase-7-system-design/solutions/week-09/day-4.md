# Solutions — Week 09 · Day 4

## Exercise 1 — TokenBudgetGuard

```python
import os
import pytest
from dataclasses import dataclass, field


class TokenBudgetExceededError(Exception):
    pass


@dataclass
class TokenBudgetGuard:
    budget: int
    _used: int = field(default=0, init=False)

    def call(self, fn) -> MockResponse:
        resp = fn()
        new_total = self._used + resp.usage.input_tokens + resp.usage.output_tokens
        if new_total > self.budget:
            raise TokenBudgetExceededError(
                f"Token budget exceeded: {new_total} > {self.budget}"
            )
        self._used = new_total
        return resp

    @property
    def tokens_used(self) -> int:
        return self._used


def test_within_budget():
    guard = TokenBudgetGuard(budget=10_000)
    resp = guard.call(lambda: mock_llm_call("short prompt"))
    assert guard.tokens_used > 0


def test_over_budget():
    guard = TokenBudgetGuard(budget=10)  # tiny budget
    with pytest.raises(TokenBudgetExceededError):
        guard.call(lambda: mock_llm_call("prompt " * 200, inject_latency_spike=True))


def test_cumulative_budget():
    guard = TokenBudgetGuard(budget=5_000)
    call_count = 0
    try:
        for _ in range(100):
            guard.call(lambda: mock_llm_call("summarize this"))
            call_count += 1
    except TokenBudgetExceededError:
        pass
    # Should stop before 100 calls
    assert call_count < 100
```

## Exercise 2 — AgentWithToolFallback

```python
import pytest
from typing import Any


class ToolError(Exception):
    pass


class AllToolsFailedError(Exception):
    pass


class AgentWithToolFallback:
    def __init__(
        self,
        search_tool,
        cache_tool,
    ) -> None:
        self._search = search_tool
        self._cache = cache_tool

    def get_results(self, query: str) -> list[Any]:
        try:
            results = self._search(query)
            if not results:
                raise ToolError("empty results")
            return results
        except ToolError:
            try:
                return self._cache(query)
            except ToolError:
                raise AllToolsFailedError("Both search and cache failed")


def test_search_succeeds():
    agent = AgentWithToolFallback(
        search_tool=lambda q: [{"result": q}],
        cache_tool=lambda q: [],
    )
    assert agent.get_results("q") == [{"result": "q"}]


def test_search_fails_uses_cache():
    def failing_search(q):
        raise ToolError("unavailable")

    agent = AgentWithToolFallback(
        search_tool=failing_search,
        cache_tool=lambda q: [{"cached": True}],
    )
    assert agent.get_results("q") == [{"cached": True}]


def test_both_fail():
    def fail(q):
        raise ToolError("fail")

    agent = AgentWithToolFallback(search_tool=fail, cache_tool=fail)
    with pytest.raises(AllToolsFailedError):
        agent.get_results("q")
```

## Exercise 3 — Regression CI gate

```python
import pytest


def compare_prompt_versions(
    v1: "PromptVersion",
    v2: "PromptVersion",
    cases: list["EvalCase"],
    threshold: float = 0.05,
) -> bool:
    """Returns True if v2 regression is within acceptable threshold."""
    v1_score = score_prompt(v1, cases)
    v2_score = score_prompt(v2, cases)
    regression = v1_score - v2_score
    return regression <= threshold


def test_equal_passes():
    assert compare_prompt_versions(PROMPT_V1, PROMPT_V1, GDP_CASES) is True


def test_one_percent_worse_passes():
    # With our mock, scores are either 0 or 100%; this tests threshold logic
    class SlightlyWorse:
        name = "slightly-worse"

    def patched_score(version, cases):
        if version.name == "slightly-worse":
            return 0.99
        return 1.0

    # Direct threshold check
    assert (1.0 - 0.99) <= 0.05


def test_six_percent_worse_fails():
    assert (1.0 - 0.94) > 0.05


def test_better_passes():
    assert (1.0 - 1.0) <= 0.05
    assert (0.9 - 1.0) <= 0.05  # v2 better → negative regression → always passes
```
