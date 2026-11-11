# Phase 5 · Week 07 · Day 3 — Shared vs Isolated State & the Blackboard Pattern

> Objective: Choose the right state sharing strategy; implement a typed blackboard for multi-agent collaboration.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

How agents share (or don't share) information is the most consequential design decision in multi-agent systems. Get it wrong and you get race conditions, stale reads, or deadlocks.

### Three strategies

**1. Fully isolated state**  
Each agent owns its memory. Communication only via messages. No shared mutable object.  
Pros: safe by construction, trivially distributable.  
Cons: every fact that matters to multiple agents must be explicitly transmitted.

**2. Shared mutable state (naive)**  
Agents hold a reference to the same Python dict/object.  
Pros: zero serialisation overhead.  
Cons: race conditions — two agents write the same key simultaneously, one overwrites the other silently. Never do this in production.

**3. Blackboard pattern**  
A **structured, concurrency-safe, central knowledge store** that all agents can read from and write to, but through a typed interface with locking and versioning. This is the sweet spot for collaborative agents.

### The blackboard pattern in depth

Originated in AI planning systems (HEARSAY speech recognition, 1970s). Three components:

```
┌──────────────────────────────────────────────┐
│                  BLACKBOARD                   │
│  Namespaced, versioned, observable store      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ facts   │  │ plans   │  │ results │       │
│  └─────────┘  └─────────┘  └─────────┘       │
└──────────────────────────────────────────────┘
         ▲ read/write ▲           ▲
         │            │           │
    Agent A       Agent B      Agent C
         ▲
    ┌────┴────────┐
    │   CONTROL   │  ← decides who acts next
    └─────────────┘
```

**Blackboard** — the shared store. Namespaced (sections), versioned (optimistic locking).  
**Knowledge sources (KS)** — the agents, each watching for triggers in the blackboard.  
**Control** — scheduler/supervisor that decides which KS activates next based on blackboard state.

### Optimistic vs pessimistic concurrency

**Pessimistic (lock-based)**: agent acquires a write lock before touching a key. Safe, but can deadlock if agents hold multiple locks.

**Optimistic (version-based)**: agent reads a key along with its version number. Write succeeds only if the version hasn't changed since the read. On conflict, agent retries. Better for low-contention scenarios.

## How it works internally

A `Blackboard` implementation typically wraps a `threading.RLock` (reentrant) or uses a `read/write lock` pattern:

- **Reads** can be concurrent — no lock needed if you only read.
- **Writes** are exclusive — acquire the write lock, check version, update, release.

For distributed systems you'd back this with Redis (`WATCH` + `MULTI/EXEC` for optimistic locking) or a persistent store. For in-process multi-threading, a `threading.Lock` suffices.

**Namespacing** prevents agents from accidentally clobbering unrelated sections. Key format: `{section}/{key}`.

**Observation / triggers**: agents register callbacks (or poll) for keys matching a pattern. When a key is written, registered watchers are notified. This replaces polling.

## Worked example

```python
"""
day3_blackboard.py — Thread-safe blackboard with namespaces, versioning, watchers.
"""
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Entry model ────────────────────────────────────────────────────────────────

class BlackboardEntry(BaseModel):
    key: str
    value: Any
    version: int = 1
    author: str = "unknown"


# ── Blackboard ─────────────────────────────────────────────────────────────────

class Blackboard:
    """Thread-safe blackboard with optimistic concurrency control."""

    def __init__(self) -> None:
        self._store: dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()
        self._watchers: dict[str, list[Callable[[BlackboardEntry], None]]] = {}

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, key: str) -> BlackboardEntry | None:
        with self._lock:
            return self._store.get(key)

    def read_section(self, section: str) -> dict[str, Any]:
        prefix = f"{section}/"
        with self._lock:
            return {
                k[len(prefix):]: v.value
                for k, v in self._store.items()
                if k.startswith(prefix)
            }

    # ── Write (optimistic CAS) ────────────────────────────────────────────────

    def write(
        self,
        key: str,
        value: Any,
        author: str,
        expected_version: int | None = None,
    ) -> bool:
        """
        Write key=value. If expected_version is given, fails (returns False)
        if the stored version differs — caller should retry.
        """
        with self._lock:
            existing = self._store.get(key)
            if expected_version is not None:
                stored_version = existing.version if existing else 0
                if stored_version != expected_version:
                    logger.warning(
                        "blackboard | CAS conflict on %s (expected v%d, got v%d)",
                        key, expected_version, stored_version,
                    )
                    return False
            new_version = (existing.version + 1) if existing else 1
            entry = BlackboardEntry(
                key=key, value=value, version=new_version, author=author
            )
            self._store[key] = entry
            logger.debug("blackboard | wrote %s v%d by %s", key, new_version, author)

        self._notify(key, entry)
        return True

    # ── Watchers ──────────────────────────────────────────────────────────────

    def watch(self, key_prefix: str, callback: Callable[[BlackboardEntry], None]) -> None:
        with self._lock:
            self._watchers.setdefault(key_prefix, []).append(callback)

    def _notify(self, key: str, entry: BlackboardEntry) -> None:
        # Run watchers outside the lock to avoid deadlock
        handlers: list[Callable[[BlackboardEntry], None]] = []
        with self._lock:
            for prefix, cbs in self._watchers.items():
                if key.startswith(prefix):
                    handlers.extend(cbs)
        for cb in handlers:
            threading.Thread(target=cb, args=(entry,), daemon=True).start()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: v.value for k, v in self._store.items()}


# ── Agents using the blackboard ────────────────────────────────────────────────

class ResearchAgent(threading.Thread):
    def __init__(self, board: Blackboard) -> None:
        super().__init__(name="researcher", daemon=True)
        self._board = board

    def run(self) -> None:
        logger.info("researcher | gathering facts")
        time.sleep(0.2)  # simulate LLM call
        self._board.write("facts/topic",    "Python async",   author="researcher")
        self._board.write("facts/year",     2024,             author="researcher")
        self._board.write("facts/keywords", ["asyncio", "coroutine"], author="researcher")
        logger.info("researcher | done")


class WriterAgent(threading.Thread):
    def __init__(self, board: Blackboard) -> None:
        super().__init__(name="writer", daemon=True)
        self._board = board
        self._ready = threading.Event()
        # Watch for facts to be available
        board.watch("facts/", lambda _: self._ready.set())

    def run(self) -> None:
        logger.info("writer | waiting for facts")
        self._ready.wait(timeout=5.0)
        facts = self._board.read_section("facts")
        draft = f"Article about {facts.get('topic')} ({facts.get('year')})"
        self._board.write("drafts/article", draft, author="writer")
        logger.info("writer | draft posted: %s", draft)


class ReviewerAgent(threading.Thread):
    def __init__(self, board: Blackboard) -> None:
        super().__init__(name="reviewer", daemon=True)
        self._board = board
        self._ready = threading.Event()
        board.watch("drafts/", lambda _: self._ready.set())

    def run(self) -> None:
        logger.info("reviewer | waiting for draft")
        self._ready.wait(timeout=5.0)
        draft_entry = self._board.read("drafts/article")
        if draft_entry is None:
            logger.error("reviewer | no draft found")
            return

        # Optimistic update: read version, write with expected version
        ok = self._board.write(
            "results/verdict",
            f"APPROVED: {draft_entry.value}",
            author="reviewer",
            expected_version=None,
        )
        logger.info("reviewer | verdict written: %s", ok)


# ── Driver ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    board    = Blackboard()
    research = ResearchAgent(board)
    writer   = WriterAgent(board)
    reviewer = ReviewerAgent(board)

    for a in (research, writer, reviewer):
        a.start()
    for a in (research, writer, reviewer):
        a.join(timeout=6.0)

    logger.info("Final blackboard: %s", board.snapshot())
```

## Exercises

1. Add a `delete(key, author)` method to `Blackboard` that marks an entry as deleted (set value to `None`, bump version) rather than removing it — this preserves history.
2. Implement a `history(key)` method that tracks every write to a key. Each entry in history has value, version, author, timestamp.
3. Add a `lock_key(key)` / `unlock_key(key)` for pessimistic locking. Show how `WriterAgent` acquires a lock before writing its draft.
4. Demonstrate a CAS conflict: start two `WriterAgent` threads simultaneously, both reading `facts/topic` at version 1, both trying to write `drafts/article` with `expected_version=0`. Which one wins?
5. Replace the in-process `Blackboard` backend with Redis (use `fakeredis` in tests). What interface stays the same?

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can explain the three state-sharing strategies and their tradeoffs
- [ ] I can implement a thread-safe blackboard with optimistic CAS
- [ ] I can register watchers and trigger agents reactively on state changes
- [ ] I understand when to use blackboard vs isolated state

## Supplementary links

- [Blackboard system (Wikipedia)](https://en.wikipedia.org/wiki/Blackboard_system)
- [Threading — RLock docs](https://docs.python.org/3/library/threading.html#rlock-objects)
- [Redis WATCH/MULTI/EXEC](https://redis.io/docs/manual/transactions/)
