# Day 7 Solutions — Pluggable Memory System (Saturday Deep Practice)

## Setup: Shared Fixtures

```python
# tests/conftest.py  (or inline in each test file)
import math
from typing import Callable

def _mock_embed(text: str) -> list[float]:
    """Deterministic hash-based 4-dim unit embedding."""
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / magnitude for v in vec]


def _mock_summarize(text: str) -> str:
    return f"[SUMMARY] {text[:60]}"
```

---

## Base PluggableMemorySystem (referenced by all exercises)

```python
# src/memory/pluggable.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]
SummarizeFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Protocols                                                                     #
# --------------------------------------------------------------------------- #

@runtime_checkable
class MemoryBackend(Protocol):
    def add(self, text: str) -> None: ...
    def retrieve(self, query: str, k: int = 5) -> list[str]: ...


# --------------------------------------------------------------------------- #
# Buffer backend                                                                #
# --------------------------------------------------------------------------- #

class BufferMemory:
    def __init__(self, max_size: int = 20) -> None:
        self._buffer: list[str] = []
        self._max_size = max_size

    def add(self, text: str) -> None:
        self._buffer.append(text)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return list(reversed(self._buffer))[:k]

    def size(self) -> int:
        return len(self._buffer)


# --------------------------------------------------------------------------- #
# Summary backend                                                               #
# --------------------------------------------------------------------------- #

class SummaryMemory:
    def __init__(self, summarize_fn: SummarizeFn, trigger: int = 5) -> None:
        self._summarize_fn = summarize_fn
        self._trigger = trigger
        self._pending: list[str] = []
        self._summaries: list[str] = []

    def add(self, text: str) -> None:
        self._pending.append(text)
        if len(self._pending) >= self._trigger:
            combined = " | ".join(self._pending)
            self._summaries.append(self._summarize_fn(combined))
            self._pending.clear()

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return self._summaries[-k:]

    def summary_count(self) -> int:
        return len(self._summaries)


# --------------------------------------------------------------------------- #
# Vector backend                                                                #
# --------------------------------------------------------------------------- #

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


class VectorMemory:
    def __init__(self, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        self._store: list[tuple[str, list[float]]] = []

    def add(self, text: str) -> None:
        self._store.append((text, self._embed_fn(text)))

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q_emb = self._embed_fn(query)
        scored = sorted(
            self._store,
            key=lambda item: _cosine(q_emb, item[1]),
            reverse=True,
        )
        return [text for text, _ in scored[:k]]

    def vector_count(self) -> int:
        return len(self._store)

    def remove_containing(self, keyword: str) -> int:
        kw = keyword.lower()
        before = len(self._store)
        self._store = [(t, e) for t, e in self._store if kw not in t.lower()]
        return before - len(self._store)


# --------------------------------------------------------------------------- #
# Pluggable orchestrator                                                        #
# --------------------------------------------------------------------------- #

class PluggableMemorySystem:
    def __init__(
        self,
        buffer: BufferMemory,
        summary: SummaryMemory,
        vector: VectorMemory,
    ) -> None:
        self._buffer = buffer
        self._summary = summary
        self._vector = vector
        self._retrieval_count = 0

    def add(self, text: str, priority: int = 0) -> None:
        if priority > 0:
            # High priority → bypass buffer, go straight to vector
            self._vector.add(text)
            logger.debug("High-priority add to vector: %r", text[:40])
        else:
            self._buffer.add(text)
            self._summary.add(text)
            self._vector.add(text)
            logger.debug("Normal add to all backends: %r", text[:40])

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        self._retrieval_count += 1
        return self._vector.retrieve(query, k=k)
```

---

## Exercise 1: MemoryStats

```python
# src/memory/stats.py
from __future__ import annotations

from pydantic import BaseModel


class MemoryStats(BaseModel):
    buffer_size: int
    summary_count: int
    vector_count: int
    total_retrievals: int
```

```python
# Append to PluggableMemorySystem:

    def stats(self) -> "MemoryStats":
        from memory.stats import MemoryStats
        return MemoryStats(
            buffer_size=self._buffer.size(),
            summary_count=self._summary.summary_count(),
            vector_count=self._vector.vector_count(),
            total_retrievals=self._retrieval_count,
        )
```

```python
# tests/unit/test_memory_stats.py
import math
from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory, PluggableMemorySystem


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_system() -> PluggableMemorySystem:
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=10),
        summary=SummaryMemory(summarize_fn=lambda t: f"s:{t[:20]}", trigger=3),
        vector=VectorMemory(embed_fn=_embed),
    )


def test_stats_initial():
    sys = _make_system()
    s = sys.stats()
    assert s.buffer_size == 0
    assert s.summary_count == 0
    assert s.vector_count == 0
    assert s.total_retrievals == 0


def test_stats_after_adds():
    sys = _make_system()
    sys.add("alpha")
    sys.add("beta")
    s = sys.stats()
    assert s.buffer_size == 2
    assert s.vector_count == 2


def test_stats_retrieval_count():
    sys = _make_system()
    sys.add("something")
    sys.retrieve("query")
    sys.retrieve("query2")
    assert sys.stats().total_retrievals == 2


def test_stats_summary_triggers():
    sys = _make_system()
    # trigger=3, so after 3 adds summary_count becomes 1
    for i in range(3):
        sys.add(f"item {i}")
    assert sys.stats().summary_count == 1
```

---

## Exercise 2: Priority parameter

The `add(text, priority: int = 0)` is already implemented in the base class above.

```python
# tests/unit/test_priority_add.py
import math
from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory, PluggableMemorySystem


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_system() -> PluggableMemorySystem:
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=10),
        summary=SummaryMemory(summarize_fn=lambda t: f"s:{t[:20]}", trigger=10),
        vector=VectorMemory(embed_fn=_embed),
    )


def test_high_priority_skips_buffer():
    sys = _make_system()
    sys.add("urgent fact", priority=1)
    # Buffer should be empty
    assert sys.stats().buffer_size == 0
    # Vector should have it
    assert sys.stats().vector_count == 1


def test_normal_priority_goes_to_all_backends():
    sys = _make_system()
    sys.add("normal fact", priority=0)
    assert sys.stats().buffer_size == 1
    assert sys.stats().vector_count == 1


def test_high_priority_is_retrievable():
    sys = _make_system()
    sys.add("top secret data", priority=1)
    results = sys.retrieve("top secret")
    assert "top secret data" in results


def test_mixed_priority_routing():
    sys = _make_system()
    sys.add("normal", priority=0)
    sys.add("urgent", priority=5)
    sys.add("regular", priority=0)
    # buffer only has the two normal ones
    assert sys.stats().buffer_size == 2
    assert sys.stats().vector_count == 3
```

---

## Exercise 3: PluggableMemorySystem.forget(keyword)

```python
# Append to PluggableMemorySystem and VectorMemory (remove_containing already defined above)

    def forget(self, keyword: str) -> int:
        """
        Remove all memories containing keyword from all backends.

        Returns total count of items removed across all stores.
        """
        kw = keyword.lower()

        # Buffer
        buf_before = len(self._buffer._buffer)
        self._buffer._buffer = [t for t in self._buffer._buffer if kw not in t.lower()]
        buf_removed = buf_before - len(self._buffer._buffer)

        # Summary (remove matching summaries)
        sum_before = len(self._summary._summaries)
        self._summary._summaries = [
            s for s in self._summary._summaries if kw not in s.lower()
        ]
        sum_removed = sum_before - len(self._summary._summaries)

        # Vector
        vec_removed = self._vector.remove_containing(keyword)

        total = buf_removed + sum_removed + vec_removed
        logger.info("forget(%r) removed %d items total", keyword, total)
        return total
```

```python
# tests/unit/test_forget.py
import math
from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory, PluggableMemorySystem


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_system() -> PluggableMemorySystem:
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=20),
        summary=SummaryMemory(summarize_fn=lambda t: f"s:{t[:20]}", trigger=100),
        vector=VectorMemory(embed_fn=_embed),
    )


def test_forget_removes_from_buffer():
    sys = _make_system()
    sys.add("delete this secret")
    sys.add("keep this safe")
    removed = sys.forget("secret")
    assert removed >= 1
    assert all("secret" not in t for t in sys._buffer._buffer)


def test_forget_removes_from_vector():
    sys = _make_system()
    sys.add("top secret data")
    sys.add("public info")
    sys.forget("secret")
    results = sys.retrieve("top secret", k=10)
    assert "top secret data" not in results


def test_forget_no_match_returns_zero():
    sys = _make_system()
    sys.add("harmless memory")
    count = sys.forget("nuclear codes")
    assert count == 0


def test_forget_case_insensitive():
    sys = _make_system()
    sys.add("SECRET password here")
    removed = sys.forget("secret")
    assert removed >= 1
    assert sys.stats().buffer_size == 0
```

---

## Exercise 4: MemoryMiddleware and LoggingMiddleware

```python
# src/memory/middleware.py
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryMiddleware(Protocol):
    def before_add(self, text: str, priority: int) -> None: ...
    def after_add(self, text: str, priority: int) -> None: ...
    def before_retrieve(self, query: str, k: int) -> None: ...
    def after_retrieve(self, query: str, k: int, results: list[str]) -> None: ...


class LoggingMiddleware:
    """Logs every add() and retrieve() call via the standard logging module."""

    def __init__(self, name: str = "memory") -> None:
        self._log = logging.getLogger(name)

    def before_add(self, text: str, priority: int) -> None:
        self._log.info("ADD text=%r priority=%d", text[:60], priority)

    def after_add(self, text: str, priority: int) -> None:
        self._log.debug("ADD complete text=%r", text[:40])

    def before_retrieve(self, query: str, k: int) -> None:
        self._log.info("RETRIEVE query=%r k=%d", query[:60], k)

    def after_retrieve(self, query: str, k: int, results: list[str]) -> None:
        self._log.debug("RETRIEVE returned %d results", len(results))
```

Wire middleware into PluggableMemorySystem:

```python
# Updated PluggableMemorySystem.__init__ (add middlewares param):

    def __init__(self, buffer, summary, vector, middlewares=None):
        self._buffer = buffer
        self._summary = summary
        self._vector = vector
        self._retrieval_count = 0
        self._middlewares: list[MemoryMiddleware] = list(middlewares or [])

    def add(self, text: str, priority: int = 0) -> None:
        for mw in self._middlewares:
            mw.before_add(text, priority)
        if priority > 0:
            self._vector.add(text)
        else:
            self._buffer.add(text)
            self._summary.add(text)
            self._vector.add(text)
        for mw in self._middlewares:
            mw.after_add(text, priority)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        self._retrieval_count += 1
        for mw in self._middlewares:
            mw.before_retrieve(query, k)
        results = self._vector.retrieve(query, k=k)
        for mw in self._middlewares:
            mw.after_retrieve(query, k, results)
        return results
```

```python
# tests/unit/test_middleware.py
import logging
import math
from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory, PluggableMemorySystem
from memory.middleware import LoggingMiddleware


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_system_with_logger(caplog) -> PluggableMemorySystem:
    mw = LoggingMiddleware(name="memory")
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=10),
        summary=SummaryMemory(summarize_fn=lambda t: t, trigger=100),
        vector=VectorMemory(embed_fn=_embed),
        middlewares=[mw],
    )


def test_logging_middleware_logs_add(caplog):
    with caplog.at_level(logging.INFO, logger="memory"):
        sys = _make_system_with_logger(caplog)
        sys.add("test memory")
    assert any("ADD" in record.message for record in caplog.records)
    assert any("test memory" in record.message for record in caplog.records)


def test_logging_middleware_logs_retrieve(caplog):
    with caplog.at_level(logging.INFO, logger="memory"):
        sys = _make_system_with_logger(caplog)
        sys.add("something to find")
        sys.retrieve("find me")
    assert any("RETRIEVE" in record.message for record in caplog.records)


def test_middleware_does_not_affect_results(caplog):
    with caplog.at_level(logging.DEBUG, logger="memory"):
        sys = _make_system_with_logger(caplog)
        sys.add("findable fact")
        results = sys.retrieve("findable fact")
    assert "findable fact" in results
```

---

## Exercise 5: CompositeRetriever with deduplication

```python
# src/memory/composite.py
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


class CompositeRetriever:
    """
    Queries both a buffer and a vector store, deduplicates by cosine similarity,
    and returns the merged top-k results.
    """

    def __init__(
        self,
        buffer,           # BufferMemory
        vector,           # VectorMemory
        embed_fn,         # EmbedFn
        sim_threshold: float = 0.9,
    ) -> None:
        self._buffer = buffer
        self._vector = vector
        self._embed_fn = embed_fn
        self._sim_threshold = sim_threshold

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        buf_results = self._buffer.retrieve(query, k=k)
        vec_results = self._vector.retrieve(query, k=k)

        merged: list[str] = []
        merged_embeddings: list[list[float]] = []

        for text in buf_results + vec_results:
            emb = self._embed_fn(text)
            is_dup = any(
                _cosine(emb, existing) > self._sim_threshold
                for existing in merged_embeddings
            )
            if not is_dup:
                merged.append(text)
                merged_embeddings.append(emb)

        logger.debug(
            "CompositeRetriever merged %d results into %d after dedup",
            len(buf_results) + len(vec_results),
            len(merged),
        )
        return merged[:k]
```

```python
# tests/unit/test_composite_retriever.py
import math
from memory.pluggable import BufferMemory, VectorMemory
from memory.composite import CompositeRetriever


def _embed(text: str) -> list[float]:
    """Two distinct groups: 'cat' texts → near [1,0,0,0], others → near [0,1,0,0]."""
    if "cat" in text.lower():
        return [1.0, 0.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0, 0.0]


def test_composite_deduplicates_identical_text():
    buf = BufferMemory(max_size=10)
    vec = VectorMemory(embed_fn=_embed)
    # Same text in both backends
    buf._buffer.append("cat food")
    vec._store.append(("cat food", _embed("cat food")))

    retriever = CompositeRetriever(buf, vec, _embed, sim_threshold=0.9)
    results = retriever.retrieve("cat", k=5)

    assert results.count("cat food") == 1


def test_composite_merges_distinct_results():
    buf = BufferMemory(max_size=10)
    vec = VectorMemory(embed_fn=_embed)
    buf._buffer.append("cat food")
    vec._store.append(("dog food", _embed("dog food")))

    retriever = CompositeRetriever(buf, vec, _embed, sim_threshold=0.9)
    results = retriever.retrieve("food", k=5)

    assert "cat food" in results
    assert "dog food" in results


def test_composite_respects_top_k():
    buf = BufferMemory(max_size=20)
    vec = VectorMemory(embed_fn=_embed)
    for i in range(10):
        tag = "cat" if i % 2 == 0 else "dog"
        buf._buffer.append(f"{tag} item {i}")
        vec._store.append((f"{tag} extra {i}", _embed(f"{tag} extra {i}")))

    retriever = CompositeRetriever(buf, vec, _embed, sim_threshold=0.9)
    results = retriever.retrieve("cat", k=3)
    assert len(results) <= 3
```

---

## Exercise 6: Async variants with pytest-asyncio

```python
# src/memory/async_memory.py
from __future__ import annotations

import asyncio
import logging

from memory.pluggable import PluggableMemorySystem

logger = logging.getLogger(__name__)


class AsyncPluggableMemorySystem(PluggableMemorySystem):
    """Async-capable wrapper around PluggableMemorySystem."""

    async def add_async(self, text: str, priority: int = 0) -> None:
        """Non-blocking add: delegates CPU-bound work to a thread pool."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.add, text, priority)
        logger.debug("async add complete: %r", text[:40])

    async def retrieve_async(self, query: str, k: int = 5) -> list[str]:
        """Non-blocking retrieve."""
        loop = asyncio.get_event_loop()
        results: list[str] = await loop.run_in_executor(
            None, self.retrieve, query, k
        )
        logger.debug("async retrieve returned %d results", len(results))
        return results
```

```python
# tests/unit/test_async_memory.py
# Requires: pip install pytest-asyncio
import math
import pytest
import pytest_asyncio

from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory
from memory.async_memory import AsyncPluggableMemorySystem


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_async_system() -> AsyncPluggableMemorySystem:
    return AsyncPluggableMemorySystem(
        buffer=BufferMemory(max_size=10),
        summary=SummaryMemory(summarize_fn=lambda t: t, trigger=100),
        vector=VectorMemory(embed_fn=_embed),
    )


@pytest.mark.asyncio
async def test_add_async_stores_memory():
    sys = _make_async_system()
    await sys.add_async("async fact")
    assert sys.stats().vector_count == 1


@pytest.mark.asyncio
async def test_retrieve_async_finds_added_memory():
    sys = _make_async_system()
    await sys.add_async("machine learning basics")
    results = await sys.retrieve_async("machine learning")
    assert "machine learning basics" in results


@pytest.mark.asyncio
async def test_concurrent_adds():
    sys = _make_async_system()
    texts = [f"fact {i}" for i in range(10)]
    await asyncio.gather(*[sys.add_async(t) for t in texts])
    assert sys.stats().vector_count == 10


@pytest.mark.asyncio
async def test_async_retrieve_returns_list():
    sys = _make_async_system()
    await sys.add_async("something")
    results = await sys.retrieve_async("something")
    assert isinstance(results, list)
```

---

## Exercise 7: export_snapshot / import_snapshot

```python
# Append to PluggableMemorySystem:

    def export_snapshot(self) -> dict:
        """
        Serialize entire system state to a plain dict.

        Includes buffer contents, summaries, and vector store (text + embedding).
        """
        return {
            "buffer": list(self._buffer._buffer),
            "summaries": list(self._summary._summaries),
            "pending_summary": list(self._summary._pending),
            "vector": [
                {"text": text, "embedding": emb}
                for text, emb in self._vector._store
            ],
            "retrieval_count": self._retrieval_count,
        }

    def import_snapshot(self, data: dict) -> None:
        """
        Restore system state from a snapshot dict produced by export_snapshot.

        Clears existing state before loading.
        """
        self._buffer._buffer = list(data.get("buffer", []))
        self._summary._summaries = list(data.get("summaries", []))
        self._summary._pending = list(data.get("pending_summary", []))
        self._vector._store = [
            (entry["text"], entry["embedding"])
            for entry in data.get("vector", [])
        ]
        self._retrieval_count = data.get("retrieval_count", 0)
```

```python
# tests/unit/test_snapshot.py
import math
from memory.pluggable import BufferMemory, SummaryMemory, VectorMemory, PluggableMemorySystem


def _embed(text: str) -> list[float]:
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _make_system() -> PluggableMemorySystem:
    return PluggableMemorySystem(
        buffer=BufferMemory(max_size=10),
        summary=SummaryMemory(summarize_fn=lambda t: f"s:{t[:20]}", trigger=100),
        vector=VectorMemory(embed_fn=_embed),
    )


def test_export_snapshot_produces_dict():
    sys = _make_system()
    sys.add("hello world")
    snap = sys.export_snapshot()
    assert isinstance(snap, dict)
    assert "buffer" in snap
    assert "vector" in snap


def test_import_snapshot_round_trip():
    sys_a = _make_system()
    for text in ["alpha", "beta", "gamma"]:
        sys_a.add(text)
    sys_a.retrieve("alpha")  # bump retrieval_count
    snap = sys_a.export_snapshot()

    sys_b = _make_system()
    sys_b.import_snapshot(snap)

    assert sys_b.stats().buffer_size == sys_a.stats().buffer_size
    assert sys_b.stats().vector_count == sys_a.stats().vector_count
    assert sys_b.stats().total_retrievals == sys_a.stats().total_retrievals


def test_import_snapshot_preserves_retrievability():
    sys_a = _make_system()
    sys_a.add("unique knowledge")
    snap = sys_a.export_snapshot()

    sys_b = _make_system()
    sys_b.import_snapshot(snap)
    results = sys_b.retrieve("unique knowledge")
    assert "unique knowledge" in results


def test_import_snapshot_clears_previous_state():
    sys = _make_system()
    sys.add("old data")
    snap = {"buffer": [], "summaries": [], "pending_summary": [], "vector": [], "retrieval_count": 0}
    sys.import_snapshot(snap)
    assert sys.stats().buffer_size == 0
    assert sys.stats().vector_count == 0


def test_snapshot_includes_summaries():
    sys = _make_system()
    # Manually add a summary
    sys._summary._summaries.append("manual summary")
    snap = sys.export_snapshot()
    assert "manual summary" in snap["summaries"]
```
