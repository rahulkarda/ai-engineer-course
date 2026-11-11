# Solutions — Week 09 · Day 3 — Agent Cards & Capability Advertisement

## Exercise 1: reliability_score routing

```python
from typing import Literal
from pydantic import BaseModel, Field


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)

    def has_skill(self, skill_id: str) -> bool:
        return any(s.id == skill_id for s in self.skills)


class AgentRegistry:
    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card

    def find_by_skill(self, skill_id: str) -> list[AgentCard]:
        return [c for c in self._cards.values() if c.has_skill(skill_id)]


def route_task(registry: AgentRegistry, required_skill: str) -> AgentCard | None:
    candidates = registry.find_by_skill(required_skill)
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.reliability_score)


# Tests
import pytest


def _make_card(name: str, skills: list[str], reliability: float) -> AgentCard:
    return AgentCard(
        name=name, description="test", version="1.0.0", url=f"http://{name}",
        skills=[AgentSkill(id=s, name=s, description=s) for s in skills],
        reliability_score=reliability,
    )


def test_route_returns_highest_reliability():
    reg = AgentRegistry()
    reg.register(_make_card("low", ["summarize"], 0.7))
    reg.register(_make_card("mid", ["summarize"], 0.85))
    reg.register(_make_card("high", ["summarize"], 0.95))
    best = route_task(reg, "summarize")
    assert best is not None
    assert best.name == "high"


def test_route_no_match():
    reg = AgentRegistry()
    result = route_task(reg, "unknown")
    assert result is None
```

## Exercise 2: find_by_capabilities

```python
def find_by_capabilities(
    registry: "AgentRegistry",
    streaming: bool = False,
    push_notifications: bool = False,
) -> list[AgentCard]:
    return [
        c for c in registry._cards.values()
        if (not streaming or c.capabilities.streaming)
        and (not push_notifications or c.capabilities.push_notifications)
    ]


def test_find_by_capabilities():
    reg = AgentRegistry()
    c1 = _make_card("s", ["x"], 1.0)
    c1.capabilities = AgentCapabilities(streaming=True, push_notifications=False)
    c2 = _make_card("sp", ["y"], 1.0)
    c2.capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    c3 = _make_card("none", ["z"], 1.0)
    for c in [c1, c2, c3]:
        reg.register(c)

    streaming_only = find_by_capabilities(reg, streaming=True)
    assert len(streaming_only) == 2

    both = find_by_capabilities(reg, streaming=True, push_notifications=True)
    assert len(both) == 1
    assert both[0].name == "sp"
```

## Exercise 3: CardCache with TTL

```python
import time
from unittest.mock import MagicMock


class CardCache:
    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[AgentCard, float]] = {}  # url → (card, expiry)
        self._fetch_count = 0

    def _fetch(self, url: str) -> AgentCard:
        """Override in subclass or inject a fetch function for testing."""
        raise NotImplementedError

    def get(self, url: str) -> AgentCard:
        if url in self._cache:
            card, expiry = self._cache[url]
            if time.monotonic() < expiry:
                return card
        # Cache miss or expired
        card = self._fetch(url)
        self._cache[url] = (card, time.monotonic() + self._ttl)
        self._fetch_count += 1
        return card


class TestableCardCache(CardCache):
    def __init__(self, ttl_seconds: float, mock_card: AgentCard) -> None:
        super().__init__(ttl_seconds)
        self._mock_card = mock_card

    def _fetch(self, url: str) -> AgentCard:
        return self._mock_card


def test_card_cache_hit():
    card = _make_card("test", ["s"], 1.0)
    cache = TestableCardCache(ttl_seconds=60.0, mock_card=card)
    cache.get("http://agent")
    cache.get("http://agent")
    assert cache._fetch_count == 1  # Only fetched once


def test_card_cache_expiry():
    card = _make_card("test", ["s"], 1.0)
    cache = TestableCardCache(ttl_seconds=0.01, mock_card=card)
    cache.get("http://agent")
    time.sleep(0.05)  # Wait for TTL to expire
    cache.get("http://agent")
    assert cache._fetch_count == 2  # Re-fetched after expiry
```

## Exercise 4: min_version filtering

```python
def _semver_tuple(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def route_task_with_version(
    registry: AgentRegistry,
    required_skill: str,
    min_version: str = "0.0.0",
) -> AgentCard | None:
    min_v = _semver_tuple(min_version)
    candidates = [
        c for c in registry.find_by_skill(required_skill)
        if _semver_tuple(c.version) >= min_v
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.reliability_score)


def test_route_with_min_version():
    reg = AgentRegistry()
    reg.register(_make_card("old", ["s"], 0.9) if False else
        AgentCard(name="old", description="", version="1.0.0", url="http://old",
                  skills=[AgentSkill(id="s", name="s", description="s")], reliability_score=0.9))
    reg.register(AgentCard(name="new", description="", version="3.1.0", url="http://new",
                  skills=[AgentSkill(id="s", name="s", description="s")], reliability_score=0.8))

    # min 2.0.0 — only new qualifies
    result = route_task_with_version(reg, "s", min_version="2.0.0")
    assert result is not None
    assert result.name == "new"

    # min 1.0.0 — both qualify; reliability wins
    result2 = route_task_with_version(reg, "s", min_version="1.0.0")
    assert result2 is not None
    assert result2.name == "old"  # higher reliability
```
