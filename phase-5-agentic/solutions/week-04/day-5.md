# Day 5 Solutions — Episodic vs Semantic Memory

## Exercise 1: EpisodicMemory.get_recent(n)

```python
# src/memory/episodic.py
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Episode(BaseModel):
    """A single episodic memory record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)


class EpisodicMemory:
    """Stores and retrieves time-ordered episodes."""

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def add(self, content: str, tags: Optional[list[str]] = None) -> Episode:
        episode = Episode(content=content, tags=tags or [])
        self._episodes.append(episode)
        logger.debug("Added episode id=%s", episode.id)
        return episode

    def get_recent(self, n: int) -> list[Episode]:
        """Return the N most recent episodes (newest first)."""
        if n < 0:
            raise ValueError("n must be >= 0")
        # Episodes are stored in insertion order; reverse to get newest first.
        return list(reversed(self._episodes))[:n]

    def all(self) -> list[Episode]:
        return list(self._episodes)
```

```python
# tests/unit/test_episodic_get_recent.py
import pytest
from memory.episodic import EpisodicMemory


def test_get_recent_empty():
    mem = EpisodicMemory()
    assert mem.get_recent(5) == []


def test_get_recent_fewer_than_n():
    mem = EpisodicMemory()
    mem.add("first")
    mem.add("second")
    result = mem.get_recent(10)
    assert len(result) == 2
    # Newest first
    assert result[0].content == "second"
    assert result[1].content == "first"


def test_get_recent_exact_n():
    mem = EpisodicMemory()
    for i in range(5):
        mem.add(f"episode {i}")
    result = mem.get_recent(3)
    assert len(result) == 3
    assert result[0].content == "episode 4"
    assert result[1].content == "episode 3"
    assert result[2].content == "episode 2"


def test_get_recent_zero():
    mem = EpisodicMemory()
    mem.add("something")
    assert mem.get_recent(0) == []


def test_get_recent_negative_raises():
    mem = EpisodicMemory()
    with pytest.raises(ValueError):
        mem.get_recent(-1)
```

---

## Exercise 2: EpisodicMemory.search_by_tag(tag)

```python
# Append to src/memory/episodic.py

    def search_by_tag(self, tag: str) -> list[Episode]:
        """Return all episodes that contain the given tag (case-insensitive)."""
        tag_lower = tag.lower()
        matches = [ep for ep in self._episodes if tag_lower in (t.lower() for t in ep.tags)]
        logger.debug("search_by_tag tag=%s found=%d", tag, len(matches))
        return matches
```

```python
# tests/unit/test_episodic_search_by_tag.py
from memory.episodic import EpisodicMemory


def test_search_by_tag_no_matches():
    mem = EpisodicMemory()
    mem.add("no tags here")
    assert mem.search_by_tag("python") == []


def test_search_by_tag_single_match():
    mem = EpisodicMemory()
    mem.add("wrote unit tests", tags=["testing", "python"])
    mem.add("deployed service", tags=["devops"])
    result = mem.search_by_tag("testing")
    assert len(result) == 1
    assert result[0].content == "wrote unit tests"


def test_search_by_tag_multiple_matches():
    mem = EpisodicMemory()
    mem.add("wrote unit tests", tags=["testing"])
    mem.add("wrote integration tests", tags=["testing", "integration"])
    mem.add("deployed", tags=["devops"])
    result = mem.search_by_tag("testing")
    assert len(result) == 2


def test_search_by_tag_case_insensitive():
    mem = EpisodicMemory()
    mem.add("something", tags=["Python"])
    result = mem.search_by_tag("python")
    assert len(result) == 1


def test_search_by_tag_no_tags_field():
    mem = EpisodicMemory()
    mem.add("bare episode")  # tags defaults to []
    assert mem.search_by_tag("anything") == []
```

---

## Exercise 3: SemanticMemory.update_fact(fact_id, new_text)

```python
# src/memory/semantic.py
from __future__ import annotations

import logging
import uuid
from typing import Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: list[float]


class SemanticMemory:
    """Stores facts as embedding vectors for similarity retrieval."""

    def __init__(self, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        self._facts: dict[str, Fact] = {}

    def add_fact(self, text: str) -> Fact:
        embedding = self._embed_fn(text)
        fact = Fact(text=text, embedding=embedding)
        self._facts[fact.id] = fact
        logger.debug("Added fact id=%s", fact.id)
        return fact

    def update_fact(self, fact_id: str, new_text: str) -> Fact:
        """Replace text and re-embed an existing fact."""
        if fact_id not in self._facts:
            raise KeyError(f"Fact {fact_id!r} not found")
        new_embedding = self._embed_fn(new_text)
        updated = self._facts[fact_id].model_copy(
            update={"text": new_text, "embedding": new_embedding}
        )
        self._facts[fact_id] = updated
        logger.debug("Updated fact id=%s", fact_id)
        return updated

    def get(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def all_facts(self) -> list[Fact]:
        return list(self._facts.values())
```

```python
# tests/unit/test_semantic_update_fact.py
import pytest
from memory.semantic import SemanticMemory


def _mock_embed(text: str) -> list[float]:
    """Deterministic embedding: hash-based fixed-length vector."""
    h = hash(text) % (10 ** 8)
    dim = 4
    return [(h >> i) % 256 / 255.0 for i in range(dim)]


def test_update_fact_changes_text():
    mem = SemanticMemory(embed_fn=_mock_embed)
    fact = mem.add_fact("original text")
    updated = mem.update_fact(fact.id, "updated text")
    assert updated.text == "updated text"
    assert updated.id == fact.id  # Same ID


def test_update_fact_re_embeds():
    mem = SemanticMemory(embed_fn=_mock_embed)
    fact = mem.add_fact("original text")
    original_embedding = list(fact.embedding)
    updated = mem.update_fact(fact.id, "completely different content")
    # Embedding must have been recomputed (different text → different hash)
    assert updated.embedding != original_embedding


def test_update_fact_not_found_raises():
    mem = SemanticMemory(embed_fn=_mock_embed)
    with pytest.raises(KeyError, match="nonexistent"):
        mem.update_fact("nonexistent", "text")


def test_update_fact_persists_in_store():
    mem = SemanticMemory(embed_fn=_mock_embed)
    fact = mem.add_fact("old")
    mem.update_fact(fact.id, "new")
    retrieved = mem.get(fact.id)
    assert retrieved.text == "new"
```

---

## Exercise 4: AgentMemorySystem.consolidate()

```python
# src/memory/agent_memory_system.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from memory.episodic import Episode, EpisodicMemory
from memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[str], str]
EmbedFn = Callable[[str], list[float]]


class AgentMemorySystem:
    """Combines episodic and semantic memory with consolidation support."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        summarize_fn: Optional[SummarizeFn] = None,
    ) -> None:
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(embed_fn=embed_fn)
        self._summarize_fn: SummarizeFn = summarize_fn or (lambda text: f"Summary: {text[:80]}")

    def consolidate(self, age_threshold_hours: float = 24.0) -> int:
        """
        Move episodic episodes older than age_threshold_hours to semantic memory.

        Each old episode is summarized via _summarize_fn and stored as a semantic
        fact. The old episodes are removed from the episodic store.

        Returns the number of episodes consolidated.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=age_threshold_hours)
        old: list[Episode] = [ep for ep in self.episodic.all() if ep.timestamp < cutoff]

        for ep in old:
            summary_text = self._summarize_fn(ep.content)
            self.semantic.add_fact(summary_text)
            logger.debug("Consolidated episode id=%s as fact", ep.id)

        # Remove consolidated episodes from episodic store
        old_ids = {ep.id for ep in old}
        self.episodic._episodes = [
            ep for ep in self.episodic.all() if ep.id not in old_ids
        ]

        logger.info("Consolidated %d episodes to semantic memory", len(old))
        return len(old)
```

```python
# tests/unit/test_consolidate.py
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from memory.agent_memory_system import AgentMemorySystem
from memory.episodic import Episode


def _mock_embed(text: str) -> list[float]:
    h = hash(text) % (10 ** 8)
    return [(h >> i) % 256 / 255.0 for i in range(4)]


def _make_old_episode(content: str, hours_ago: float = 30.0) -> Episode:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return Episode(content=content, timestamp=ts)


def test_consolidate_moves_old_episodes():
    system = AgentMemorySystem(embed_fn=_mock_embed)
    # Manually inject an old episode
    old_ep = _make_old_episode("old meeting notes", hours_ago=30)
    system.episodic._episodes.append(old_ep)
    system.episodic.add("recent action")  # within 24 h

    count = system.consolidate(age_threshold_hours=24.0)

    assert count == 1
    # Old episode removed from episodic
    assert len(system.episodic.all()) == 1
    assert system.episodic.all()[0].content == "recent action"
    # Fact added to semantic
    assert len(system.semantic.all_facts()) == 1


def test_consolidate_no_old_episodes():
    system = AgentMemorySystem(embed_fn=_mock_embed)
    system.episodic.add("very recent")
    count = system.consolidate(age_threshold_hours=24.0)
    assert count == 0
    assert len(system.episodic.all()) == 1


def test_consolidate_uses_summarize_fn():
    calls: list[str] = []

    def recording_summarize(text: str) -> str:
        calls.append(text)
        return f"[SUMMARY] {text}"

    system = AgentMemorySystem(embed_fn=_mock_embed, summarize_fn=recording_summarize)
    old_ep = _make_old_episode("important meeting", hours_ago=48)
    system.episodic._episodes.append(old_ep)

    system.consolidate()

    assert calls == ["important meeting"]
    fact_texts = [f.text for f in system.semantic.all_facts()]
    assert fact_texts == ["[SUMMARY] important meeting"]


def test_consolidate_all_old():
    system = AgentMemorySystem(embed_fn=_mock_embed)
    for i in range(3):
        ep = _make_old_episode(f"old {i}", hours_ago=25 + i)
        system.episodic._episodes.append(ep)

    count = system.consolidate()
    assert count == 3
    assert system.episodic.all() == []
    assert len(system.semantic.all_facts()) == 3


def test_consolidate_with_mocked_datetime():
    """Demonstrate patching datetime.now to control cutoff."""
    system = AgentMemorySystem(embed_fn=_mock_embed)

    # Add episode with a real recent timestamp
    system.episodic.add("seems recent but will be old under shifted time")

    # Shift now() 48 hours into the future
    future = datetime.now(timezone.utc) + timedelta(hours=48)
    with patch("memory.agent_memory_system.datetime") as mock_dt:
        mock_dt.now.return_value = future
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        count = system.consolidate(age_threshold_hours=24.0)

    assert count == 1
```

---

## Exercise 5: MemoryRouter

```python
# src/memory/router.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from memory.agent_memory_system import AgentMemorySystem

logger = logging.getLogger(__name__)

_TIME_PATTERN = re.compile(
    r"\b(yesterday|last|when|ago|before|earlier|previously|recently|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)


class MemoryBackend(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class RouterResult:
    backend: MemoryBackend
    query: str


class MemoryRouter:
    """Routes a natural-language query to the appropriate memory backend."""

    def route(self, query: str) -> RouterResult:
        """Return the backend best suited for the query."""
        if _TIME_PATTERN.search(query):
            backend = MemoryBackend.EPISODIC
        else:
            backend = MemoryBackend.SEMANTIC
        logger.debug("Routed query=%r to backend=%s", query, backend)
        return RouterResult(backend=backend, query=query)
```

```python
# tests/unit/test_memory_router.py
import pytest
from memory.router import MemoryBackend, MemoryRouter


@pytest.fixture
def router() -> MemoryRouter:
    return MemoryRouter()


@pytest.mark.parametrize(
    "query",
    [
        "What did we discuss yesterday?",
        "What happened last week?",
        "When did we deploy?",
        "I forgot what we said earlier",
        "Tell me what happened previously",
        "I remember something from Monday",
    ],
)
def test_routes_to_episodic(router: MemoryRouter, query: str) -> None:
    result = router.route(query)
    assert result.backend == MemoryBackend.EPISODIC


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "Explain gradient descent",
        "What does our API endpoint do?",
        "How does the authentication system work?",
    ],
)
def test_routes_to_semantic(router: MemoryRouter, query: str) -> None:
    result = router.route(query)
    assert result.backend == MemoryBackend.SEMANTIC


def test_router_returns_original_query(router: MemoryRouter) -> None:
    q = "What did we do last Tuesday?"
    result = router.route(q)
    assert result.query == q


def test_router_case_insensitive(router: MemoryRouter) -> None:
    result = router.route("YESTERDAY we talked about this")
    assert result.backend == MemoryBackend.EPISODIC
```
