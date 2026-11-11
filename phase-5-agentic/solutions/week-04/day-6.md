# Day 6 Solutions — Persisting Agent State

## Exercise 1: StatePersistenceManager.export_all(output_path)

```python
# src/persistence/manager.py
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentMemoryState(BaseModel):
    """Typed state model for a single agent session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    messages: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatePersistenceManager:
    """Saves and loads AgentMemoryState to/from a directory of JSON files."""

    def __init__(self, storage_dir: Path) -> None:
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers

    def _session_path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    # ------------------------------------------------------------------ core API

    def save(self, state: AgentMemoryState) -> Path:
        state = state.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        path = self._session_path(state.session_id)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved session %s → %s", state.session_id, path)
        self._update_index(state.session_id, state.updated_at)
        return path

    def load(self, session_id: str) -> AgentMemoryState:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id!r} not found at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentMemoryState.model_validate(data)

    def delete(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.debug("Deleted session %s", session_id)
        self._remove_from_index(session_id)

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json") if p.stem != "_index"]

    # ------------------------------------------------------------------ export_all

    def export_all(self, output_path: Path) -> int:
        """
        Write every stored session to a single JSON file as a list.

        Returns the count of sessions written.
        """
        sessions = []
        for session_id in self.list_sessions():
            try:
                state = self.load(session_id)
                sessions.append(json.loads(state.model_dump_json()))
            except Exception:
                logger.warning("Skipping unreadable session %s", session_id, exc_info=True)

        output_path.write_text(
            json.dumps(sessions, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Exported %d sessions to %s", len(sessions), output_path)
        return len(sessions)

    # ------------------------------------------------------------------ index helpers (used by later exercises)

    @property
    def _index_path(self) -> Path:
        return self._dir / "_index.json"

    def _update_index(self, session_id: str, updated_at: datetime) -> None:
        index = self._read_index()
        index[session_id] = updated_at.isoformat()
        self._write_index(index)

    def _remove_from_index(self, session_id: str) -> None:
        index = self._read_index()
        index.pop(session_id, None)
        self._write_index(index)

    def _read_index(self) -> dict[str, str]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict[str, str]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
```

```python
# tests/unit/test_export_all.py
import json
from pathlib import Path

import pytest
from persistence.manager import AgentMemoryState, StatePersistenceManager


def test_export_all_empty(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    out = tmp_path / "export.json"
    count = mgr.export_all(out)
    assert count == 0
    data = json.loads(out.read_text())
    assert data == []


def test_export_all_multiple_sessions(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    states = [
        AgentMemoryState(messages=[{"role": "user", "content": f"msg {i}"}])
        for i in range(3)
    ]
    for s in states:
        mgr.save(s)

    out = tmp_path / "export.json"
    count = mgr.export_all(out)

    assert count == 3
    loaded = json.loads(out.read_text())
    assert len(loaded) == 3
    ids = {item["session_id"] for item in loaded}
    assert ids == {s.session_id for s in states}


def test_export_all_output_is_valid_json(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    mgr.save(AgentMemoryState(messages=[{"role": "user", "content": "hello"}]))
    out = tmp_path / "export.json"
    mgr.export_all(out)
    # Must parse without error
    data = json.loads(out.read_text())
    assert isinstance(data, list)
```

---

## Exercise 2: StatePersistenceManager.import_session(data)

```python
# Append to src/persistence/manager.py

    def import_session(self, data: dict[str, Any]) -> str:
        """
        Load a session from a plain dict and persist it to disk.

        Validates via AgentMemoryState so schema errors surface early.
        Returns the session_id.
        """
        state = AgentMemoryState.model_validate(data)
        self.save(state)
        logger.info("Imported session %s", state.session_id)
        return state.session_id
```

```python
# tests/unit/test_import_session.py
import json
from pathlib import Path

import pytest
from persistence.manager import AgentMemoryState, StatePersistenceManager


def test_import_session_round_trip(tmp_path: Path) -> None:
    mgr_a = StatePersistenceManager(tmp_path / "store_a")
    original = AgentMemoryState(
        messages=[{"role": "assistant", "content": "hello"}],
        metadata={"source": "test"},
    )
    mgr_a.save(original)

    # Export to dict
    raw = json.loads(original.model_dump_json())

    # Import into a different store
    mgr_b = StatePersistenceManager(tmp_path / "store_b")
    imported_id = mgr_b.import_session(raw)

    assert imported_id == original.session_id
    loaded = mgr_b.load(imported_id)
    assert loaded.messages == original.messages
    assert loaded.metadata == original.metadata


def test_import_session_invalid_data_raises(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        mgr.import_session({"messages": "not-a-list"})
```

---

## Exercise 3: Schema versioning and migrate_v1_to_v2

```python
# src/persistence/migrations.py
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade a v1 AgentMemoryState dict to v2.

    Migration contract (example):
    - v2 adds a 'tags' field (list[str]) to each message.
    - v2 bumps schema_version to 2.
    """
    if data.get("schema_version", 1) != 1:
        raise ValueError(
            f"migrate_v1_to_v2 expects schema_version=1, got {data.get('schema_version')}"
        )
    migrated = dict(data)
    migrated["schema_version"] = 2
    migrated["messages"] = [
        {**msg, "tags": []} for msg in data.get("messages", [])
    ]
    logger.info("Migrated session %s from v1 to v2", data.get("session_id"))
    return migrated


def auto_migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Apply all outstanding migrations in order."""
    version = data.get("schema_version", 1)
    if version < 2:
        data = migrate_v1_to_v2(data)
    return data
```

AgentMemoryState updated with `schema_version`:

```python
class AgentMemoryState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1          # <-- new field (already included above)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

```python
# tests/unit/test_migrations.py
import pytest
from persistence.migrations import migrate_v1_to_v2, auto_migrate


def test_migrate_v1_to_v2_bumps_version():
    data = {"schema_version": 1, "session_id": "abc", "messages": []}
    result = migrate_v1_to_v2(data)
    assert result["schema_version"] == 2


def test_migrate_v1_to_v2_adds_tags():
    data = {
        "schema_version": 1,
        "session_id": "abc",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = migrate_v1_to_v2(data)
    assert result["messages"][0]["tags"] == []
    # Original fields preserved
    assert result["messages"][0]["role"] == "user"


def test_migrate_v1_to_v2_wrong_version_raises():
    with pytest.raises(ValueError):
        migrate_v1_to_v2({"schema_version": 2, "messages": []})


def test_auto_migrate_no_op_on_current():
    """If already v2, auto_migrate should not re-apply v1→v2."""
    data = {"schema_version": 2, "messages": [{"role": "user", "content": "x", "tags": []}]}
    result = auto_migrate(data)
    assert result["schema_version"] == 2
    assert result["messages"][0]["tags"] == []  # unchanged


def test_auto_migrate_chains_from_v1():
    data = {"schema_version": 1, "messages": [{"role": "user", "content": "x"}]}
    result = auto_migrate(data)
    assert result["schema_version"] == 2
```

---

## Exercise 4: atomic_save using tmp file + rename

```python
# Append to src/persistence/manager.py

    def atomic_save(self, state: AgentMemoryState) -> Path:
        """
        Write state to a .tmp file then atomically rename to final path.

        Guarantees that readers never see a partially-written file.
        """
        state = state.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        final = self._session_path(state.session_id)
        tmp = final.with_suffix(".tmp")
        try:
            tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(final)  # atomic on POSIX; best-effort on Windows
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.debug("Atomic save session %s", state.session_id)
        self._update_index(state.session_id, state.updated_at)
        return final
```

```python
# tests/unit/test_atomic_save.py
from pathlib import Path

import pytest
from persistence.manager import AgentMemoryState, StatePersistenceManager


def test_atomic_save_no_tmp_file_after_success(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    state = AgentMemoryState(messages=[{"role": "user", "content": "test"}])
    final = mgr.atomic_save(state)

    assert final.exists()
    tmp = final.with_suffix(".tmp")
    assert not tmp.exists(), ".tmp file should be cleaned up after rename"


def test_atomic_save_final_file_readable(tmp_path: Path) -> None:
    mgr = StatePersistenceManager(tmp_path / "store")
    state = AgentMemoryState(messages=[{"role": "user", "content": "atomic"}])
    mgr.atomic_save(state)
    loaded = mgr.load(state.session_id)
    assert loaded.session_id == state.session_id
    assert loaded.messages[0]["content"] == "atomic"


def test_atomic_save_tmp_removed_on_error(tmp_path: Path, monkeypatch) -> None:
    """If replace() raises, the .tmp file is cleaned up."""
    mgr = StatePersistenceManager(tmp_path / "store")
    state = AgentMemoryState()

    original_replace = Path.replace

    def failing_replace(self, target):
        raise OSError("simulated disk full")

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError):
        mgr.atomic_save(state)

    tmp = mgr._session_path(state.session_id).with_suffix(".tmp")
    assert not tmp.exists()
```

---

## Exercise 5: SessionIndex

```python
# src/persistence/session_index.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SessionIndexEntry(BaseModel):
    session_id: str
    last_modified: datetime


class SessionIndex:
    """
    Lightweight JSON registry of session IDs and their last_modified timestamps.

    Stays in sync with the persistence manager via on_save() / on_delete().
    """

    def __init__(self, index_path: Path) -> None:
        self._path = index_path
        if not self._path.exists():
            self._write({})

    # ------------------------------------------------------------------ public

    def on_save(self, session_id: str, last_modified: datetime) -> None:
        index = self._read()
        index[session_id] = last_modified.isoformat()
        self._write(index)
        logger.debug("Index updated: %s at %s", session_id, last_modified.isoformat())

    def on_delete(self, session_id: str) -> None:
        index = self._read()
        removed = index.pop(session_id, None)
        if removed:
            self._write(index)
            logger.debug("Index removed: %s", session_id)

    def entries(self) -> list[SessionIndexEntry]:
        raw = self._read()
        return [
            SessionIndexEntry(
                session_id=sid,
                last_modified=datetime.fromisoformat(ts),
            )
            for sid, ts in raw.items()
        ]

    def session_ids(self) -> list[str]:
        return list(self._read().keys())

    # ------------------------------------------------------------------ private

    def _read(self) -> dict[str, str]:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, str]) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

Integration with StatePersistenceManager (wire-up):

```python
# In StatePersistenceManager.__init__, add:
#   self._index = SessionIndex(self._dir / "_session_index.json")
# In save():  self._index.on_save(state.session_id, state.updated_at)
# In delete(): self._index.on_delete(session_id)
```

```python
# tests/unit/test_session_index.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
from persistence.session_index import SessionIndex


@pytest.fixture
def index(tmp_path: Path) -> SessionIndex:
    return SessionIndex(tmp_path / "index.json")


def test_index_empty_on_init(index: SessionIndex) -> None:
    assert index.entries() == []
    assert index.session_ids() == []


def test_index_on_save_adds_entry(index: SessionIndex) -> None:
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    index.on_save("session-1", ts)
    assert "session-1" in index.session_ids()


def test_index_on_save_updates_existing(index: SessionIndex) -> None:
    ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    index.on_save("session-1", ts1)
    index.on_save("session-1", ts2)
    entries = index.entries()
    assert len(entries) == 1
    assert entries[0].last_modified == ts2


def test_index_on_delete_removes_entry(index: SessionIndex) -> None:
    ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
    index.on_save("session-1", ts)
    index.on_save("session-2", ts)
    index.on_delete("session-1")
    assert "session-1" not in index.session_ids()
    assert "session-2" in index.session_ids()


def test_index_on_delete_nonexistent_noop(index: SessionIndex) -> None:
    """Deleting a non-existent session must not raise."""
    index.on_delete("ghost-session")  # should not raise
    assert index.entries() == []


def test_index_persists_across_instances(tmp_path: Path) -> None:
    """Index survives a new SessionIndex instance pointing at the same file."""
    path = tmp_path / "index.json"
    ts = datetime(2024, 6, 1, tzinfo=timezone.utc)

    idx1 = SessionIndex(path)
    idx1.on_save("s-abc", ts)

    idx2 = SessionIndex(path)  # new object, same file
    assert "s-abc" in idx2.session_ids()


def test_index_consistency_with_manager(tmp_path: Path) -> None:
    """
    Simulate save + delete lifecycle and verify the index mirrors disk state.
    """
    from persistence.manager import AgentMemoryState, StatePersistenceManager

    mgr = StatePersistenceManager(tmp_path / "store")
    s1 = AgentMemoryState()
    s2 = AgentMemoryState()

    mgr.save(s1)
    mgr.save(s2)

    # Both sessions in index
    index_ids = set(mgr._read_index().keys())
    assert s1.session_id in index_ids
    assert s2.session_id in index_ids

    mgr.delete(s1.session_id)
    index_ids_after = set(mgr._read_index().keys())
    assert s1.session_id not in index_ids_after
    assert s2.session_id in index_ids_after
```
