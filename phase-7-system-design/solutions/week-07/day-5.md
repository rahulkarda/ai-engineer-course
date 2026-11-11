# Solutions — Phase 7 · Week 07 · Day 5

## Exercise 1 — Elevator: `DoorController` State Machine

The lesson exercise asks for a `DoorController` class with states `CLOSED`, `OPENING`, `OPEN`,
`CLOSING` and transitions `open()`, `close()`, `obstruct()`, `clear()`.

```python
"""
door_controller.py — Exercise 1 solution.

DoorController implements the four-state elevator door state machine.
Illegal transitions raise InvalidTransition instead of silently no-oping.
"""

import logging
from enum import Enum, auto

log = logging.getLogger(__name__)


class DoorState(Enum):
    CLOSED = auto()
    OPENING = auto()
    OPEN = auto()
    CLOSING = auto()


class InvalidTransition(Exception):
    """Raised when a transition is not valid from the current state."""


# Allowed transitions: {current_state -> {event -> next_state}}
_TRANSITIONS: dict[DoorState, dict[str, DoorState]] = {
    DoorState.CLOSED:  {"open":     DoorState.OPENING},
    DoorState.OPENING: {"arrived":  DoorState.OPEN},
    DoorState.OPEN:    {"close":    DoorState.CLOSING,
                        "obstruct": DoorState.OPEN},     # stays OPEN on obstruction
    DoorState.CLOSING: {"closed":   DoorState.CLOSED,
                        "obstruct": DoorState.OPENING},  # re-open if obstructed while closing
}


class DoorController:
    """
    State machine for an elevator door.

    States:   CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED
    Events:
      open()      — begin opening (valid from CLOSED)
      arrived()   — door fully open (valid from OPENING)
      close()     — begin closing (valid from OPEN)
      obstruct()  — sensor detects obstruction (valid from OPEN or CLOSING)
      clear()     — obstruction cleared; re-triggers close (valid from OPEN)
      closed()    — door fully closed (valid from CLOSING)
    """

    def __init__(self) -> None:
        self._state: DoorState = DoorState.CLOSED

    @property
    def state(self) -> DoorState:
        return self._state

    def _transition(self, event: str) -> None:
        allowed = _TRANSITIONS.get(self._state, {})
        if event not in allowed:
            raise InvalidTransition(
                f"Event '{event}' is not valid from state {self._state.name}. "
                f"Allowed events: {list(allowed)}"
            )
        next_state = allowed[event]
        log.debug("door: %s --[%s]--> %s", self._state.name, event, next_state.name)
        self._state = next_state

    # --- public events ---

    def open(self) -> None:
        """Begin opening the door.  Valid from CLOSED."""
        self._transition("open")

    def arrived(self) -> None:
        """Door has finished opening.  Valid from OPENING."""
        self._transition("arrived")

    def close(self) -> None:
        """Begin closing the door.  Valid from OPEN."""
        self._transition("close")

    def obstruct(self) -> None:
        """
        Obstruction sensor fired.
        - From OPEN: stay OPEN.
        - From CLOSING: reverse to OPENING.
        """
        self._transition("obstruct")
        if self._state is DoorState.OPENING:
            # Automatically advance to OPEN (obstruction during close re-opens fully)
            self._transition("arrived")

    def clear(self) -> None:
        """Obstruction cleared; begin closing again.  Valid from OPEN."""
        self.close()

    def closed(self) -> None:
        """Door has finished closing.  Valid from CLOSING."""
        self._transition("closed")


# --- quick smoke test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    d = DoorController()

    d.open()      # CLOSED -> OPENING
    d.arrived()   # OPENING -> OPEN
    d.obstruct()  # OPEN -> OPEN (via CLOSING -> OPENING -> OPEN if mid-close)
    d.close()     # OPEN -> CLOSING
    d.closed()    # CLOSING -> CLOSED

    assert d.state == DoorState.CLOSED

    # Illegal transition
    try:
        d.close()   # CLOSED -> ??? should raise
        assert False, "Expected InvalidTransition"
    except InvalidTransition as exc:
        log.info("Correctly caught: %s", exc)

    log.info("DoorController smoke test passed")
```

---

## Exercise 2 — Vending Machine: `CoinInventory` with Change-Making

```python
"""
coin_inventory.py — Exercise 2 solution.

CoinInventory holds a supply of coins (value in cents -> count).
make_change(amount_cents) returns the breakdown or None if impossible.
"""

import logging
from copy import deepcopy

log = logging.getLogger(__name__)


class CoinInventory:
    """
    Tracks a supply of coins indexed by denomination in cents.

    make_change uses a greedy algorithm (largest denomination first).
    This is optimal when denominations are standard (1, 5, 10, 25, 50, 100).
    """

    def __init__(self, initial: dict[int, int] | None = None) -> None:
        # denomination_cents -> count
        self._coins: dict[int, int] = dict(initial) if initial else {}

    def add(self, denomination: int, count: int = 1) -> None:
        """Add coins to inventory."""
        if denomination <= 0:
            raise ValueError("denomination must be > 0")
        self._coins[denomination] = self._coins.get(denomination, 0) + count

    def remove(self, denomination: int, count: int) -> None:
        """Remove coins from inventory.  Raises ValueError if insufficient."""
        available = self._coins.get(denomination, 0)
        if available < count:
            raise ValueError(
                f"Insufficient {denomination}¢ coins: have {available}, need {count}"
            )
        self._coins[denomination] = available - count
        if self._coins[denomination] == 0:
            del self._coins[denomination]

    def make_change(self, amount_cents: int) -> dict[int, int] | None:
        """
        Return the minimum-coin breakdown for amount_cents, or None if impossible.

        Uses a greedy algorithm on denominations sorted largest-first.
        Does NOT modify the inventory — call remove() explicitly after confirming.
        """
        if amount_cents < 0:
            raise ValueError("amount_cents must be >= 0")
        if amount_cents == 0:
            return {}

        remaining = amount_cents
        result: dict[int, int] = {}
        # Work largest denomination first
        for denom in sorted(self._coins, reverse=True):
            available = self._coins[denom]
            needed = min(available, remaining // denom)
            if needed > 0:
                result[denom] = needed
                remaining -= denom * needed
            if remaining == 0:
                break

        if remaining != 0:
            log.debug(
                "make_change: cannot make %d¢ with available coins %s",
                amount_cents,
                self._coins,
            )
            return None

        return result

    def apply_change(self, change: dict[int, int]) -> None:
        """Deduct coins returned as change from inventory."""
        for denom, count in change.items():
            self.remove(denom, count)

    def total_value(self) -> int:
        """Total value held in inventory, in cents."""
        return sum(d * c for d, c in self._coins.items())

    def __repr__(self) -> str:
        sorted_coins = sorted(self._coins.items(), reverse=True)
        parts = [f"{d}¢x{c}" for d, c in sorted_coins]
        return f"CoinInventory({', '.join(parts)})"


# --- smoke test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    inv = CoinInventory({25: 4, 10: 5, 5: 2, 1: 10})

    change = inv.make_change(41)
    assert change == {25: 1, 10: 1, 5: 1, 1: 1}, f"Got: {change}"
    log.info("make_change(41) = %s", change)

    change2 = inv.make_change(3)
    assert change2 == {1: 3}, f"Got: {change2}"

    inv2 = CoinInventory({25: 1})
    assert inv2.make_change(30) is None, "Should be impossible"

    log.info("CoinInventory smoke test passed")
```

---

## Exercise 3 — Library: `search_by_author` and `overdue_loans`

The lesson already provides a `LibrarySystem` skeleton.  This solution adds the two requested
methods to that skeleton.

```python
"""
library_system.py — Exercise 3 solution.

Adds search_by_author() and overdue_loans() to the lesson's LibrarySystem.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

log = logging.getLogger(__name__)

FINE_PER_DAY: Decimal = Decimal("0.25")
MAX_LOANS: int = 3


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    total_copies: int
    available: int = field(init=False)

    def __post_init__(self) -> None:
        self.available = self.total_copies


@dataclass
class Loan:
    member_id: str
    isbn: str
    due_date: date
    returned_at: date | None = None

    def is_active(self) -> bool:
        return self.returned_at is None

    def fine(self, as_of: date | None = None) -> Decimal:
        ref = as_of or self.returned_at or date.today()
        overdue_days = max(0, (ref - self.due_date).days)
        return FINE_PER_DAY * overdue_days


class LibrarySystem:
    def __init__(self) -> None:
        self._books: dict[str, Book] = {}
        self._loans: list[Loan] = []

    def add_book(self, book: Book) -> None:
        self._books[book.isbn] = book

    # ---- original methods (from lesson) ----

    def checkout(self, member_id: str, isbn: str, due: date) -> Loan:
        book = self._books[isbn]
        active = [l for l in self._loans if l.member_id == member_id and l.is_active()]
        if len(active) >= MAX_LOANS:
            raise ValueError("Loan limit reached")
        if book.available == 0:
            raise ValueError("No copies available")
        book.available -= 1
        loan = Loan(member_id=member_id, isbn=isbn, due_date=due)
        self._loans.append(loan)
        log.info("checkout: member=%s isbn=%s due=%s", member_id, isbn, due)
        return loan

    def checkin(self, member_id: str, isbn: str) -> Decimal:
        for loan in self._loans:
            if loan.member_id == member_id and loan.isbn == isbn and loan.is_active():
                loan.returned_at = date.today()
                self._books[isbn].available += 1
                fine = loan.fine()
                log.info("checkin: member=%s isbn=%s fine=%.2f", member_id, isbn, fine)
                return fine
        raise ValueError("No active loan found")

    def search(self, query: str) -> list[Book]:
        q = query.lower()
        return [
            b for b in self._books.values()
            if q in b.title.lower() or q in b.author.lower() or q == b.isbn
        ]

    # ---- Exercise 3 additions ----

    def search_by_author(self, author: str) -> list[Book]:
        """
        Return all books whose author contains `author` (case-insensitive),
        sorted alphabetically by title.
        """
        a = author.lower()
        results = [b for b in self._books.values() if a in b.author.lower()]
        results.sort(key=lambda b: b.title.lower())
        log.debug("search_by_author(%r): %d results", author, len(results))
        return results

    def overdue_loans(self, as_of: date | None = None) -> list[tuple[Loan, Decimal]]:
        """
        Return (loan, fine) pairs for all active loans that are overdue as of `as_of`
        (defaults to today), sorted by fine amount descending.
        """
        ref = as_of or date.today()
        overdue = [
            (loan, loan.fine(as_of=ref))
            for loan in self._loans
            if loan.is_active() and ref > loan.due_date
        ]
        overdue.sort(key=lambda pair: pair[1], reverse=True)
        log.debug("overdue_loans as_of=%s: %d overdue", ref, len(overdue))
        return overdue


# --- smoke test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    lib = LibrarySystem()
    lib.add_book(Book(isbn="111", title="Zelda's Garden", author="Alice Smith", total_copies=2))
    lib.add_book(Book(isbn="222", title="Ada's Code",     author="Alice Smith", total_copies=1))
    lib.add_book(Book(isbn="333", title="Bob's Journey",  author="Bob Jones",   total_copies=1))

    by_alice = lib.search_by_author("alice")
    assert [b.title for b in by_alice] == ["Ada's Code", "Zelda's Garden"], by_alice

    past_due = date(2024, 1, 1)
    lib.checkout("m1", "111", due=past_due)
    lib.checkout("m2", "222", due=date(2025, 1, 1))

    overdue = lib.overdue_loans(as_of=date(2025, 6, 1))
    assert len(overdue) == 1
    assert overdue[0][0].member_id == "m1"
    log.info("Library exercise smoke test passed")
```

---

## Exercise 4 — Rate Limiter: `PerEndpointLimiter`

```python
"""
per_endpoint_limiter.py — Exercise 4 solution.

PerEndpointLimiter wraps a RateLimiter and namespaces keys as
"user_id:endpoint" so that user A's calls to /foo don't consume
user B's quota for /foo.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock

log = logging.getLogger(__name__)


class RateLimiter(ABC):
    @abstractmethod
    def allow(self, key: str) -> bool: ...


class SlidingWindowLimiter(RateLimiter):
    def __init__(self, window_sec: int, max_requests: int) -> None:
        self._window = window_sec
        self._max = max_requests
        self._windows: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            dq = self._windows.setdefault(key, deque())
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) < self._max:
                dq.append(now)
                return True
            return False


class PerEndpointLimiter(RateLimiter):
    """
    Wraps a RateLimiter and prefixes keys with "user_id:endpoint".

    This ensures quota is tracked independently per (user, endpoint) pair.
    User A hitting /foo does NOT reduce User B's quota for /foo.
    """

    def __init__(self, inner: RateLimiter) -> None:
        self._inner = inner

    def allow(self, key: str) -> bool:
        """
        key must be "user_id:endpoint", e.g. "alice:/search".
        The namespace is passed as-is to the inner limiter because the inner
        limiter already keys by the full string.
        """
        return self._inner.allow(key)

    def allow_for(self, user_id: str, endpoint: str) -> bool:
        """Convenience method: build the namespaced key and delegate."""
        namespaced_key = f"{user_id}:{endpoint}"
        allowed = self._inner.allow(namespaced_key)
        log.debug("allow_for user=%s endpoint=%s -> %s", user_id, endpoint, allowed)
        return allowed


# --- unit tests (runnable directly) ---

def _run_tests() -> None:
    """
    Verify that:
    1. User A hitting /foo does not consume User B's quota.
    2. A single user exhausts their own quota independently.
    """
    base_limiter = SlidingWindowLimiter(window_sec=60, max_requests=3)
    limiter = PerEndpointLimiter(inner=base_limiter)

    # User A makes 3 requests to /search — should all succeed
    for i in range(3):
        assert limiter.allow_for("alice", "/search"), f"alice /search request {i} should be allowed"

    # User A's 4th request to /search should be denied
    assert not limiter.allow_for("alice", "/search"), "alice 4th /search should be denied"

    # User B's first request to /search should STILL succeed (independent quota)
    assert limiter.allow_for("bob", "/search"), "bob /search should be allowed (independent quota)"

    # Alice hitting a different endpoint /upload should still be allowed
    assert limiter.allow_for("alice", "/upload"), "alice /upload should be allowed (different endpoint)"

    log.info("PerEndpointLimiter tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _run_tests()
```

**Key design notes:**
- The namespace key `f"{user_id}:{endpoint}"` is the simplest approach; it requires no extra
  wrapping logic in the base limiter.
- `allow_for(user_id, endpoint)` is a convenience layer; `allow(key)` with a pre-built key
  keeps the interface compatible with `RateLimiter`.
- If the endpoint itself could contain `:`, use a separator that cannot appear in either part
  (e.g. `"\x00"`) or URL-encode before joining.
