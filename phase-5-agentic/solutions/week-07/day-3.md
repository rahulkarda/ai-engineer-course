# Solutions — Week 07 · Day 3: Shared vs Isolated State & Blackboard Pattern

## Exercise 1: Soft-delete

```python
def delete(self, key: str, author: str) -> bool:
    with self._lock:
        existing = self._store.get(key)
        if existing is None:
            return False
        new_version = existing.version + 1
        tombstone = BlackboardEntry(
            key=key, value=None, version=new_version, author=author
        )
        self._store[key] = tombstone
    self._notify(key, tombstone)
    return True

# Read should check for tombstone:
def read(self, key: str) -> BlackboardEntry | None:
    with self._lock:
        entry = self._store.get(key)
    if entry is None or entry.value is None:
        return None
    return entry
```

## Exercise 2: History tracking

```python
from collections import defaultdict

class BlackboardWithHistory(Blackboard):
    def __init__(self) -> None:
        super().__init__()
        self._history: dict[str, list[BlackboardEntry]] = defaultdict(list)

    def write(self, key: str, value, author: str, expected_version: int | None = None) -> bool:
        success = super().write(key, value, author, expected_version)
        if success:
            with self._lock:
                entry = self._store[key]
                self._history[key].append(entry.model_copy())
        return success

    def history(self, key: str) -> list[BlackboardEntry]:
        with self._lock:
            return list(self._history.get(key, []))
```

## Exercise 3: Pessimistic locking

```python
import threading

class BlackboardWithLocks(Blackboard):
    def __init__(self) -> None:
        super().__init__()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_meta = threading.Lock()

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._key_locks_meta:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def lock_key(self, key: str) -> None:
        self._get_key_lock(key).acquire()

    def unlock_key(self, key: str) -> None:
        lock = self._key_locks.get(key)
        if lock:
            lock.release()


# Writer agent using pessimistic lock
class SafeWriterAgent(threading.Thread):
    def __init__(self, board: BlackboardWithLocks, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._board = board

    def run(self) -> None:
        self._board.lock_key("drafts/article")
        try:
            import time; time.sleep(0.1)  # simulate work
            self._board.write("drafts/article", f"Draft by {self.name}", author=self.name)
        finally:
            self._board.unlock_key("drafts/article")
```

## Exercise 4: CAS conflict demonstration

```python
import threading

def demonstrate_cas_conflict() -> None:
    board = Blackboard()
    # Pre-seed facts at version 1
    board.write("facts/topic", "Python async", author="seeder")

    entry = board.read("facts/topic")
    assert entry is not None
    seen_version = entry.version  # both writers see version 1

    results: list[bool] = []
    barrier = threading.Barrier(2)  # synchronize start

    def writer(name: str) -> None:
        barrier.wait()  # both start at the same time
        ok = board.write(
            "drafts/article",
            f"Draft by {name}",
            author=name,
            expected_version=0,  # both expect "not yet written" (version 0)
        )
        results.append(ok)
        logger.info("%s: CAS result = %s", name, ok)

    t1 = threading.Thread(target=writer, args=("writer-1",))
    t2 = threading.Thread(target=writer, args=("writer-2",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Exactly one writer wins
    assert results.count(True) == 1
    assert results.count(False) == 1
    logger.info("Winner wrote version 1; loser got CAS failure")
```

## Exercise 5: Redis-backed blackboard

```python
# In tests: pip install fakeredis
import fakeredis
import json
import redis

class RedisBlackboard:
    """Same interface as Blackboard, backed by Redis for distribution."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._r = redis_client

    def write(self, key: str, value, author: str,
              expected_version: int | None = None) -> bool:
        pipe = self._r.pipeline(True)  # WATCH + MULTI/EXEC
        try:
            pipe.watch(key)
            raw = pipe.get(key)
            existing = json.loads(raw) if raw else None
            if expected_version is not None:
                stored = existing["version"] if existing else 0
                if stored != expected_version:
                    return False
            new_version = (existing["version"] + 1) if existing else 1
            entry = {"value": value, "version": new_version, "author": author}
            pipe.multi()
            pipe.set(key, json.dumps(entry))
            pipe.execute()
            return True
        except redis.WatchError:
            return False  # CAS conflict

    def read(self, key: str):
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)


# Test
def test_redis_blackboard_write_read() -> None:
    client = fakeredis.FakeRedis()
    board = RedisBlackboard(client)
    assert board.write("test/key", "hello", "author") is True
    entry = board.read("test/key")
    assert entry["value"] == "hello"
    assert entry["version"] == 1
```
