# Solutions — Week 06 · Day 3 — Concurrency Primitives & Deadlock

## Exercise 1 — Race condition non-determinism

The counter rarely reaches 200,000 without the lock because `counter += 1` compiles to:
1. `LOAD_FAST counter`
2. `LOAD_CONST 1`
3. `BINARY_ADD`
4. `STORE_FAST counter`

The GIL can switch threads between any two bytecodes. If two threads both load the same value, add 1, and both store — one increment is lost. The result depends on exact scheduling timing, so it varies run to run and machine to machine.

## Exercise 2 — BoundedQueue with single Condition

```python
import threading
from collections import deque
from dataclasses import dataclass, field

@dataclass
class BoundedQueueSingle:
    maxsize: int
    _q: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cond: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self._cond = threading.Condition(self._lock)

    def put(self, item: int) -> None:
        with self._cond:
            while len(self._q) >= self.maxsize:
                self._cond.wait()
            self._q.append(item)
            self._cond.notify_all()  # wake both producers and consumers

    def get(self) -> int:
        with self._cond:
            while not self._q:
                self._cond.wait()
            item = self._q.popleft()
            self._cond.notify_all()
            return item
```

**Trade-off**: `notify_all()` wakes every waiting thread — producers woken when a slot opens, consumers woken when an item appears, but also consumers woken on `put` wake up only to re-check and sleep again if the item was taken. This causes spurious wakeups and extra lock acquisitions. Two-condition design (`_not_empty`, `_not_full`) eliminates spurious wakeups: `put` notifies only consumers, `get` notifies only producers.

## Exercise 3 — A→{1,2}, B→{2,3}, C→{3,1} resource allocation graph

```
Resource-allocation graph:

Thread A ──holds──► Lock 1   Thread A ──waits──► Lock 2
Thread B ──holds──► Lock 2   Thread B ──waits──► Lock 3
Thread C ──holds──► Lock 3   Thread C ──waits──► Lock 1

Cycle: A waits for 2 (held by B) → B waits for 3 (held by C) → C waits for 1 (held by A)
```

**Deadlock is possible.** There exists a circular wait: A→B→C→A. If all three start simultaneously and each acquires its first lock before the other's second lock is released, all three will block forever.

## Exercise 4 — RWLock implementation

```python
import threading
from dataclasses import dataclass, field

@dataclass
class RWLock:
    """Multiple readers OR one exclusive writer."""
    _readers: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cond: threading.Condition = field(init=False)
    _write_locked: bool = False

    def __post_init__(self) -> None:
        self._cond = threading.Condition(self._lock)

    def acquire_read(self) -> None:
        with self._cond:
            while self._write_locked:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._write_locked or self._readers > 0:
                self._cond.wait()
            self._write_locked = True

    def release_write(self) -> None:
        with self._cond:
            self._write_locked = False
            self._cond.notify_all()
```

```python
# test_rwlock.py
import threading
import pytest
from solution_day3 import RWLock

def test_multiple_readers():
    rw = RWLock()
    results = []
    def reader():
        rw.acquire_read()
        results.append("reading")
        rw.release_read()
    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(results) == 5

def test_exclusive_write_blocks_readers():
    rw = RWLock()
    order = []
    rw.acquire_write()
    def reader():
        rw.acquire_read()
        order.append("read")
        rw.release_read()
    t = threading.Thread(target=reader)
    t.start()
    import time; time.sleep(0.05)
    order.append("write_releasing")
    rw.release_write()
    t.join()
    assert order == ["write_releasing", "read"]
```

## Exercise 5 — Livelock vs Deadlock

**Livelock**: threads are not blocked — they are actively running — but keep reacting to each other in a way that prevents progress. No thread is waiting, but none advances.

**Real-world analogy**: two people walking toward each other in a corridor. Each steps aside to let the other pass — but both step to the same side simultaneously, then correct again, repeatedly. They are moving (not frozen), but never get past each other.

Deadlock = frozen. Livelock = spinning uselessly.
