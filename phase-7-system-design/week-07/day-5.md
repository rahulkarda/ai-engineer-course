# Phase 7 · Week 07 · Day 5 — Classic LLD Problems Walkthrough

> Objective: Walk four canonical LLD interview problems at interview pace: requirements, class diagram, key decisions  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-4]]

---

## Concept

In a 45-minute LLD interview you have roughly **10 minutes per problem** to:

1. Clarify requirements (2 min)
2. Sketch entities and relationships — class diagram (4 min)
3. Walk the interviewer through key design decisions (3 min)
4. Write skeleton code for the most interesting class (rest)

This lesson walks four problems at that pace.  Use it as a rehearsal script.

---

## Problem 1 — Elevator System

### Requirements (clarify these aloud)

- Building has N floors, M elevators.
- Users press UP/DOWN on each floor (hall call) or a floor button inside the car (car call).
- Scheduler assigns requests to elevators to minimise wait time.
- Doors open, wait, then close; re-open on obstruction.

### Class diagram

```
+------------------+       +---------------------+
|    Building      |       |    ElevatorCar       |
+------------------+       +---------------------+
| elevators: list  |<>---->| id: int             |
| floors: int      |       | current_floor: int  |
| call(floor, dir) |       | state: ElevatorState|
+------------------+       | queue: SortedSet    |
                           | move_to(floor)      |
+------------------+       | open_doors()        |
|  HallButton      |       | close_doors()       |
+------------------+       +---------------------+
| floor: int       |              |
| direction: Dir   |              |
+------------------+     +-------------------+
                         |   ElevatorState   | (enum)
+------------------+     +-------------------+
|   Scheduler      |     | IDLE              |
+------------------+     | MOVING_UP         |
| assign(req)      |     | MOVING_DOWN       |
| nearest_car(req) |     | DOOR_OPEN         |
+------------------+     +-------------------+
```

### Key design decisions

1. **Scheduler algorithm** — SCAN (elevator sweeps direction, picks up same-direction requests)
   is simpler than LOOK and fair under load.  For interviews: name it, say why.

2. **Request queue per elevator** — a `SortedSet` (or `sortedcontainers.SortedList`) of
   floor numbers keeps the car's stops in traversal order.  O(log n) insert, O(1) next stop.

3. **Door state machine** — separate `DoorController` class prevents the car from moving
   while doors are open.  State: `CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED`.
   Transitions gated by sensors, not timers alone.

### Key class skeleton

```python
from enum import Enum, auto
from sortedcontainers import SortedList  # pip install sortedcontainers

class Direction(Enum):
    UP = auto()
    DOWN = auto()

class ElevatorState(Enum):
    IDLE = auto()
    MOVING_UP = auto()
    MOVING_DOWN = auto()
    DOOR_OPEN = auto()

class ElevatorCar:
    def __init__(self, car_id: int) -> None:
        self.id = car_id
        self.current_floor: int = 0
        self.state: ElevatorState = ElevatorState.IDLE
        self._up_stops: SortedList[int] = SortedList()
        self._down_stops: SortedList[int] = SortedList()
        self._direction: Direction = Direction.UP

    def add_stop(self, floor: int) -> None:
        if floor > self.current_floor:
            self._up_stops.add(floor)
        else:
            self._down_stops.add(floor)

    def next_stop(self) -> int | None:
        if self._direction is Direction.UP and self._up_stops:
            return self._up_stops[0]
        if self._direction is Direction.DOWN and self._down_stops:
            return self._down_stops[-1]
        return None

class Scheduler:
    def __init__(self, cars: list[ElevatorCar]) -> None:
        self._cars = cars

    def assign(self, floor: int, direction: Direction) -> ElevatorCar:
        # Nearest-car heuristic; replace with SCAN for production
        return min(
            self._cars,
            key=lambda c: abs(c.current_floor - floor),
        )
```

---

## Problem 2 — Vending Machine

### Requirements

- Machine holds multiple product slots, each with a price and count.
- User inserts coins/notes, selects product, machine dispenses or returns change.
- Out-of-stock and insufficient-funds are explicit error states.

### Class diagram

```
+-------------------+       +------------------+
|   VendingMachine  |       |   ProductSlot    |
+-------------------+       +------------------+
| slots: dict       |<>---->| product: Product |
| state: VMState    |       | price: Decimal   |
| balance: Decimal  |       | count: int       |
| insert(amount)    |       | dispense() bool  |
| select(slot_id)   |       +------------------+
| cancel()          |
| dispense()        |       +-------------------+
+-------------------+       |    VMState        | (enum)
                            +-------------------+
+-------------------+       | IDLE              |
|    Product        |       | HAS_MONEY         |
+-------------------+       | DISPENSING        |
| id: str           |       | OUT_OF_SERVICE    |
| name: str         |       +-------------------+
+-------------------+
```

### Key design decisions

1. **State machine** controls which operations are valid.
   `insert()` only valid in `IDLE`/`HAS_MONEY`; `select()` only in `HAS_MONEY`.
   Encode transitions as a dict to avoid scattered `if state == ...` checks.

2. **`Decimal` for money** — never `float` for currency.  Two decimal places, no rounding
   surprises.

3. **Change-making** — keep a `CoinInventory` object; return change greedily from largest
   denomination.  If exact change unavailable: refund and stay in `HAS_MONEY`.

### Key class skeleton

```python
from decimal import Decimal
from enum import Enum, auto
from dataclasses import dataclass, field

class VMState(Enum):
    IDLE = auto()
    HAS_MONEY = auto()
    DISPENSING = auto()

@dataclass
class ProductSlot:
    product_name: str
    price: Decimal
    count: int

    def dispense(self) -> bool:
        if self.count <= 0:
            return False
        self.count -= 1
        return True

class VendingMachine:
    def __init__(self) -> None:
        self._state = VMState.IDLE
        self._balance = Decimal("0.00")
        self._slots: dict[str, ProductSlot] = {}

    def insert(self, amount: Decimal) -> None:
        if self._state not in (VMState.IDLE, VMState.HAS_MONEY):
            raise RuntimeError("Cannot insert money now")
        self._balance += amount
        self._state = VMState.HAS_MONEY

    def select(self, slot_id: str) -> str:
        if self._state is not VMState.HAS_MONEY:
            raise RuntimeError("Insert money first")
        slot = self._slots.get(slot_id)
        if slot is None:
            raise KeyError(f"Unknown slot {slot_id}")
        if slot.count == 0:
            raise ValueError("Out of stock")
        if self._balance < slot.price:
            raise ValueError(f"Insufficient funds: need {slot.price}, have {self._balance}")
        self._state = VMState.DISPENSING
        change = self._balance - slot.price
        slot.dispense()
        self._balance = Decimal("0.00")
        self._state = VMState.IDLE
        return f"Dispensed {slot.product_name}. Change: {change}"
```

---

## Problem 3 — Library Management System

### Requirements

- Catalog of books (title, author, ISBN, copies).
- Members can borrow/return books; max N books at a time.
- Search by title, author, or ISBN.
- Track overdue loans; compute fines.

### Class diagram

```
+---------------+       +-------------------+
|    Catalog    |       |       Book        |
+---------------+       +-------------------+
| add_book()    |<>---->| isbn: str         |
| search()      |       | title: str        |
| get_copy()    |       | author: str       |
+---------------+       | total_copies: int |
                        | available: int    |
+---------------+       +-------------------+
|    Member     |
+---------------+       +-------------------+
| id: str       |       |      Loan         |
| name: str     |<>---->+-------------------+
| active_loans  |       | member: Member    |
| borrow()      |       | book: Book        |
| return_book() |       | due_date: date    |
+---------------+       | returned_at: date |
                        | fine() Decimal    |
+---------------+       +-------------------+
| LibrarySystem |
+---------------+
| catalog       |
| members       |
| checkout()    |
| checkin()     |
| overdue_loans()|
+---------------+
```

### Key design decisions

1. **`Loan` as a first-class entity** — not just a list on `Member`.  It carries state
   (due date, returned date) and behaviour (`fine()`).  This makes overdue queries trivial.

2. **Search** — for an interview, a linear scan over a list is acceptable.  Mention that in
   production you'd use an inverted index or full-text search (e.g. Elasticsearch).

3. **Borrow limit** — enforced in `LibrarySystem.checkout()`, not in `Member`.  Keeps
   `Member` a data object; policy lives in the service layer.

### Key class skeleton

```python
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

FINE_PER_DAY = Decimal("0.25")
MAX_LOANS = 3

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

    def fine(self) -> Decimal:
        ref = self.returned_at or date.today()
        overdue_days = max(0, (ref - self.due_date).days)
        return FINE_PER_DAY * overdue_days

class LibrarySystem:
    def __init__(self) -> None:
        self._books: dict[str, Book] = {}
        self._loans: list[Loan] = []

    def add_book(self, book: Book) -> None:
        self._books[book.isbn] = book

    def checkout(self, member_id: str, isbn: str, due: date) -> Loan:
        book = self._books[isbn]
        active = [l for l in self._loans if l.member_id == member_id and l.returned_at is None]
        if len(active) >= MAX_LOANS:
            raise ValueError("Loan limit reached")
        if book.available == 0:
            raise ValueError("No copies available")
        book.available -= 1
        loan = Loan(member_id=member_id, isbn=isbn, due_date=due)
        self._loans.append(loan)
        return loan

    def checkin(self, member_id: str, isbn: str) -> Decimal:
        for loan in self._loans:
            if loan.member_id == member_id and loan.isbn == isbn and loan.returned_at is None:
                loan.returned_at = date.today()
                self._books[isbn].available += 1
                return loan.fine()
        raise ValueError("No active loan found")

    def search(self, query: str) -> list[Book]:
        q = query.lower()
        return [
            b for b in self._books.values()
            if q in b.title.lower() or q in b.author.lower() or q == b.isbn
        ]
```

---

## Problem 4 — Rate Limiter (object model)

### Requirements

- Multiple algorithms: token bucket (burst-friendly), sliding window (precise).
- Per-user and per-API-endpoint limits.
- Thread-safe; in-memory (Redis integration out of scope for interview).

### Class diagram

```
+------------------+         +----------------------+
|  RateLimiter     |         |  TokenBucketLimiter  |
+------------------+         +----------------------+
| allow(key) bool  |<|------>| capacity: int        |
+------------------+         | refill_rate: float   |
        ^                    | tokens: float        |
        |                    | last_refill: float   |
+------------------+         | allow(key) bool      |
| SlidingWindow    |         +----------------------+
| Limiter          |
+------------------+         +----------------------+
| window_sec: int  |         | SlidingWindowLimiter |
| max_requests: int|         +----------------------+
| allow(key) bool  |         | window_sec: int      |
+------------------+         | max_requests: int    |
                             | timestamps: deque    |
+------------------+         | allow(key) bool      |
| CompositeLimit   |         +----------------------+
+------------------+
| limiters: list   |
| allow(key) bool  |  -- AND of all limiters
+------------------+
```

### Key design decisions

1. **Strategy pattern** — `RateLimiter` is an abstract interface.  `allow(key: str) -> bool`
   is the only public method.  Swap algorithms without changing callers.

2. **Per-user state** — each limiter holds a `dict[str, ...]` keyed by user/endpoint.
   Avoids a single shared lock; lock per key is better under high concurrency.

3. **Composite limiter** — chain a token bucket (burst) AND a sliding window (daily cap)
   via `CompositeRateLimiter`.  Both must return `True` for the request to proceed.

### Key class skeleton

```python
import time
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock

class RateLimiter(ABC):
    @abstractmethod
    def allow(self, key: str) -> bool: ...

class TokenBucketLimiter(RateLimiter):
    def __init__(self, capacity: int, refill_rate: float) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate  # tokens per second
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            tokens, last = self._buckets.get(key, (float(self._capacity), now))
            elapsed = now - last
            tokens = min(self._capacity, tokens + elapsed * self._refill_rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            self._buckets[key] = (tokens, now)
            return False

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

class CompositeRateLimiter(RateLimiter):
    def __init__(self, *limiters: RateLimiter) -> None:
        self._limiters = limiters

    def allow(self, key: str) -> bool:
        return all(lim.allow(key) for lim in self._limiters)
```

---

## Exercises

1. **Elevator.** Implement the `DoorController` state machine as a class with transitions
   `open()`, `close()`, `obstruct()`, `clear()`.  States: `CLOSED`, `OPENING`, `OPEN`,
   `CLOSING`.  Raise `InvalidTransition` for illegal moves.

2. **Vending Machine.** Implement `CoinInventory` that holds a `dict[int, int]` (coin value
   in cents -> count).  Add `make_change(amount_cents: int) -> dict[int, int] | None` that
   returns the coin breakdown or `None` if exact change is impossible.

3. **Library.** Add a `search_by_author(author: str) -> list[Book]` that returns books sorted
   by title.  Add a `overdue_loans(as_of: date) -> list[tuple[Loan, Decimal]]` method.

4. **Rate Limiter.** Write a `PerEndpointLimiter` that wraps a `RateLimiter` and namespaces
   keys as `f"{user_id}:{endpoint}"`.  Verify with unit tests that user A hitting `/foo`
   does not consume user B's quota for `/foo`.

(Solutions: ../solutions/week-07/day-5.md)

---

## Self-check

- [ ] I can sketch the Elevator class diagram from memory in under 4 minutes
- [ ] I know why `Decimal` is required for money and can explain the alternative's failure mode
- [ ] I can explain why `Loan` is a first-class entity (not just a list on `Member`)
- [ ] I can compare token bucket vs sliding window: burst behaviour, memory, precision
- [ ] I know how the Strategy pattern applies to `RateLimiter`
- [ ] I can name the state machine states for both the Elevator door and the Vending Machine

---

## Supplementary

- [Refactoring Guru — Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Refactoring Guru — State Pattern](https://refactoring.guru/design-patterns/state)
- [Rate Limiting Algorithms — Cloudflare blog](https://blog.cloudflare.com/counting-things-a-lot-of-different-things/)
- [System Design Interview Vol 1, Chapter 4 — Rate Limiter (Alex Xu)](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF)
- [sortedcontainers](https://grantjenks.com/docs/sortedcontainers/)
