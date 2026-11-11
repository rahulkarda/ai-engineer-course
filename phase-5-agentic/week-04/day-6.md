# Phase 5 · Week 04 · Day 6 — Persisting Agent State

> Objective: Implement durable agent state using JSON file persistence with Pydantic serialization and session management.  ·  Time: ~1 hr  ·  Prereqs: [[day-5.md]]

## Concept

An in-memory agent loses all context the moment the process exits. Persistence solves three problems:

1. **Crash recovery** — resume from the last checkpoint instead of starting over.
2. **Multi-session continuity** — a user returns the next day; the agent remembers them.
3. **Agent restarts** — deploy a new version without wiping user state.

Persistence backends trade off complexity vs. capability:

| Backend | Complexity | Concurrent writers | Query flexibility |
|---|---|---|---|
| JSON file | Low | No | None (load all) |
| SQLite | Medium | Yes (WAL mode) | SQL |
| Redis | Medium-High | Yes | Key/value + TTL |
| PostgreSQL | High | Yes | Full SQL |

For Week 04, the JSON file backend is the right level. It is simple enough to implement completely in one session and illustrates every concept (serialization, session IDs, load/save, error handling) without infrastructure dependencies.

**Session IDs** are the key to multi-user / multi-session storage. Each saved state lives under a unique `session_id` string (typically a UUID or a user-scoped key). The persistence layer maps `session_id → file path`.

## How it works internally

**Serialization with Pydantic**

Pydantic v2 provides `model.model_dump()` → `dict` and `Model.model_validate(dict)` → `Model`. For persistence:

- `save`: `state.model_dump(mode="json")` converts the model to a JSON-serializable dict (handles `datetime`, `UUID`, nested models). Write that dict to disk with `json.dump`.
- `load`: `json.load` → raw dict → `AgentMemoryState.model_validate(raw_dict)` reconstructs the typed model.

**Checkpointing patterns**

Two main approaches:
- *Save-on-write*: persist after every state mutation. Safe but potentially expensive.
- *Save-on-demand*: persist explicitly (e.g., at conversation end, on timeout). Efficient but loses un-persisted work on crash.

For most agent use cases, save-on-demand at natural boundaries (end of turn, end of session) is the pragmatic choice.

**File layout**

```
.agent_state/
  <session_id>.json
  <session_id>.json
  ...
```

One file per session. `pathlib.Path` handles cross-platform path construction. Atomic writes (write to `<file>.tmp`, then rename) prevent corrupt state files on crash mid-write.

## Worked example

```python
"""
day6_persistence.py

JSON-file-based agent state persistence.
Run standalone or import StatePersistenceManager for use in other modules.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------

class EpisodeRecord(BaseModel):
    timestamp: str  # ISO 8601 string — serializes cleanly to JSON
    event: str
    metadata: dict = Field(default_factory=dict)


class SemanticFactRecord(BaseModel):
    fact: str
    source: str = "unknown"


class AgentMemoryState(BaseModel):
    """
    Complete, serializable snapshot of an agent's memory.
    Pairs with StatePersistenceManager for save/load.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    episodic: list[EpisodeRecord] = Field(default_factory=list)
    semantic: list[SemanticFactRecord] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def bump_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_episode(self, event: str, **meta) -> None:
        self.episodic.append(
            EpisodeRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event=event,
                metadata=meta,
            )
        )
        self.bump_updated()

    def add_fact(self, fact: str, source: str = "unknown") -> None:
        self.semantic.append(SemanticFactRecord(fact=fact, source=source))
        self.bump_updated()


# ---------------------------------------------------------------------------
# Persistence manager
# ---------------------------------------------------------------------------

class StatePersistenceManager:
    """
    Saves and loads AgentMemoryState to/from JSON files.

    Each session gets its own file: <storage_dir>/<session_id>.json
    Writes are atomic (write to .tmp, rename).
    """

    def __init__(self, storage_dir: Path | str = ".agent_state") -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info("Storage directory: %s", self._dir.resolve())

    def _path(self, session_id: str) -> Path:
        # Sanitize to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe_id}.json"

    def save_state(self, session_id: str, state: AgentMemoryState) -> Path:
        """
        Persist state to disk atomically.
        Returns the path written.
        """
        state.bump_updated()
        target = self._path(session_id)
        tmp = target.with_suffix(".tmp")
        payload = state.model_dump(mode="json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.rename(target)
        self._log.info(
            "Saved session '%s' (%d episodes, %d facts)",
            session_id,
            len(state.episodic),
            len(state.semantic),
        )
        return target

    def load_state(self, session_id: str) -> Optional[AgentMemoryState]:
        """
        Load state for a session. Returns None if not found.
        """
        path = self._path(session_id)
        if not path.exists():
            self._log.warning("No saved state for session '%s'", session_id)
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = AgentMemoryState.model_validate(raw)
        self._log.info(
            "Loaded session '%s' (%d episodes, %d facts)",
            session_id,
            len(state.episodic),
            len(state.semantic),
        )
        return state

    def list_sessions(self) -> list[str]:
        """Return all known session IDs."""
        return [p.stem for p in sorted(self._dir.glob("*.json"))]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's state file. Returns True if deleted."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            self._log.info("Deleted session '%s'", session_id)
            return True
        return False

    def get_or_create(self, session_id: str) -> AgentMemoryState:
        """Load existing state, or create a fresh one for the session."""
        existing = self.load_state(session_id)
        if existing is not None:
            return existing
        state = AgentMemoryState(session_id=session_id)
        self._log.info("Created new state for session '%s'", session_id)
        return state


# ---------------------------------------------------------------------------
# Minimal agent loop demonstrating persistence
# ---------------------------------------------------------------------------

def run_agent_session(
    session_id: str,
    manager: StatePersistenceManager,
    user_messages: list[str],
) -> AgentMemoryState:
    """
    Mock agent loop: loads state, processes messages, saves state.
    In production, replace mock_llm_response with a real LLM call.
    """
    state = manager.get_or_create(session_id)

    def mock_llm_response(user_msg: str) -> str:
        return f"[mock] Acknowledged: {user_msg[:30]}"

    for msg in user_messages:
        state.add_episode(f"User: {msg}", role="user")
        response = mock_llm_response(msg)
        state.add_episode(f"Agent: {response}", role="agent")

    manager.save_state(session_id, state)
    return state


def main() -> None:
    storage = Path("/tmp/demo_agent_state")
    mgr = StatePersistenceManager(storage_dir=storage)

    session_a = "user-alice-session-1"

    # First run — creates new state
    state = run_agent_session(session_a, mgr, ["Hello", "What can you do?"])
    print(f"After first run: {len(state.episodic)} episodes")

    # Simulate restart — load from disk
    loaded = mgr.load_state(session_a)
    assert loaded is not None
    print(f"After reload: {len(loaded.episodic)} episodes, updated_at={loaded.updated_at}")

    # Second run — continues from loaded state
    state2 = run_agent_session(session_a, mgr, ["Remember our first chat?"])
    print(f"After second run: {len(state2.episodic)} episodes")

    # List sessions
    print("Sessions on disk:", mgr.list_sessions())

    # Store a semantic fact and persist
    state2.add_fact("Alice prefers concise answers.", source="user-preference")
    mgr.save_state(session_a, state2)

    # Clean up demo
    mgr.delete_session(session_a)
    print("Demo complete — session deleted.")


if __name__ == "__main__":
    main()
```

### Example test pattern (uses pytest `tmp_path` fixture)

```python
# tests/unit/test_day6_persistence.py
import pytest
from pathlib import Path
from day6_persistence import AgentMemoryState, StatePersistenceManager


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(storage_dir=tmp_path)
    state = AgentMemoryState(session_id="test-session")
    state.add_episode("User said hello", role="user")
    state.add_fact("User is testing persistence.")
    mgr.save_state("test-session", state)

    loaded = mgr.load_state("test-session")
    assert loaded is not None
    assert len(loaded.episodic) == 1
    assert len(loaded.semantic) == 1
    assert loaded.session_id == "test-session"


def test_load_missing_session_returns_none(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(storage_dir=tmp_path)
    result = mgr.load_state("does-not-exist")
    assert result is None


def test_list_sessions(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(storage_dir=tmp_path)
    for sid in ["alpha", "beta", "gamma"]:
        mgr.save_state(sid, AgentMemoryState(session_id=sid))
    sessions = mgr.list_sessions()
    assert set(sessions) == {"alpha", "beta", "gamma"}


def test_delete_session(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(storage_dir=tmp_path)
    state = AgentMemoryState(session_id="to-delete")
    mgr.save_state("to-delete", state)
    assert mgr.delete_session("to-delete") is True
    assert mgr.load_state("to-delete") is None


def test_get_or_create_returns_existing(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(storage_dir=tmp_path)
    state = AgentMemoryState(session_id="existing")
    state.add_fact("Fact A")
    mgr.save_state("existing", state)

    loaded = mgr.get_or_create("existing")
    assert len(loaded.semantic) == 1
```

## Exercises

1. Add a `backup_session(session_id: str, backup_dir: Path)` method that copies the session file to a backup directory, appending a timestamp suffix (e.g., `<session_id>_20240101T120000.json`). Test it with `tmp_path`.

2. Extend `AgentMemoryState` with a `turn_count: int = 0` field. Increment it in `add_episode` only when `role == "user"`. Verify that a round-trip through `save_state` / `load_state` preserves the correct count.

3. The current atomic write uses `.rename()`. On some systems, rename across filesystems fails. Add a fallback: catch `OSError` on rename and fall back to a direct write. Write a test that patches `Path.rename` to raise `OSError` and verifies the fallback succeeds.

4. Add a `prune_old_episodes(keep_recent: int = 20)` method to `AgentMemoryState` that drops all but the most recent `keep_recent` episodes in-place. After pruning, call `bump_updated()`. Test with a state that has 30 episodes.

5. Implement `StatePersistenceManager.export_all(export_path: Path)` that writes a single JSON file containing a dict mapping session_id → state dict for all sessions. Implement `import_all(export_path: Path)` to restore from that file. Test the round-trip.

(Solutions: ../solutions/week-04/day-6.md)

## Self-check

- [ ] I can explain why JSON persistence is appropriate for single-process agents and where it breaks down
- [ ] I can use `model_dump(mode="json")` and `model_validate()` for state round-trips
- [ ] I can implement an atomic write (write-to-tmp, rename) and explain why it prevents corruption
- [ ] I can implement `save_state`, `load_state`, `list_sessions`, `delete_session` without reference
- [ ] I can write pytest tests for persistence using the `tmp_path` fixture
