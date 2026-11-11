# Solutions — Week 07 Assessment

## Conceptual answers

**1. SRP vs. ISP**

SRP says a class should have only one reason to change — it's about the class's responsibility. ISP says a client should not be forced to depend on methods it doesn't use — it's about the interface the client sees.

Violation example:
```python
class UserService:
    def create_user(self, name: str) -> None: ...
    def send_email(self, email: str) -> None: ...   # SRP: second responsibility
    def export_csv(self) -> None: ...               # SRP: third responsibility

class IUserService(ABC):                            # fat interface
    @abstractmethod
    def create_user(self, name: str) -> None: ...
    @abstractmethod
    def send_email(self, email: str) -> None: ...
    @abstractmethod
    def export_csv(self) -> None: ...               # ISP: clients needing only create_user
                                                    # are forced to depend on export_csv
```

Fix: split into `UserCreator`, `Emailer`, `CsvExporter` classes; split the interface into `IUserCreator`, `IEmailer`, `ICsvExporter`.

---

**2. Strategy vs. Conditional**

The colleague is right when: the algorithm is trivial (2 cases, never changes), the code is a throwaway script, or adding Strategy would mean three new files for a 5-line `if/else`. Simplicity wins there.

Reach for Strategy when: you're adding a 4th or 5th branch, when behaviour changes at runtime (promo codes, A/B testing), or when each branch is complex enough to warrant independent testing. The `DiscountStrategy` example (Day 4) starts as `if type == "percent": ...` but quickly explodes when you add seasonal, loyalty, and B2G1 discounts — Strategy pays off when the set of algorithms is open-ended.

---

**3. Decorator (GoF) vs. Python `@decorator`**

Python's `@decorator` syntax is syntactic sugar for higher-order functions: `f = decorator(f)`. It wraps *callables* (functions), not objects with rich interfaces.

The GoF Decorator pattern wraps an *object* that implements an interface, adding behaviour while preserving the interface. The wrapped object and all decorators share the same type (same Protocol/ABC).

The GoF pattern is necessary when you need to decorate an *instance* at runtime (not at definition time), stack multiple decorators in any order, and have each decorator be independently tested against the interface contract — e.g., `TimingDecorator(RetryDecorator(LegacyAlertAdapter(legacy)))`.

---

**4. Repository pattern and testability**

Repository hides persistence behind a collection-like interface. Domain logic depends on the abstract `Repository`, not on a concrete ORM. In tests, you swap in an `InMemoryRepository` (a Fake) — no real DB needed.

A **Fake** is a real (but simplified) implementation: `InMemoryRepository` uses a `dict`. Tests exercise actual domain logic. A **Mock** is a recorder: you assert it was called with specific arguments. Use a Fake when you want to test the *result* of a sequence of operations (e.g., "after three adds, list_all returns 3 items"). Use a Mock when you want to assert the *interaction* (e.g., "save() was called exactly once with this payload").

---

**5. Dependency Inversion**

What is inverted: traditionally, high-level modules import low-level modules (control flows down). DIP inverts this by making *both* high and low levels depend on an abstraction — neither depends on the other concretely.

Why it matters: if `OrderService` depends on `PostgresRepository`, changing the DB requires changing `OrderService`. With DIP, `OrderService` depends on `AbstractRepository`; `PostgresRepository` implements it. Changing DB requires only a new implementation — `OrderService` is untouched. At scale (hundreds of modules), this eliminates ripple-change cascades.

---

**6. Singleton pitfalls**

**Pitfall 1 — Test isolation.** A Singleton survives between tests, carrying state. Tests become order-dependent. Mitigation: add a `reset()` classmethod used in test fixtures, or use `pydantic-settings` with dependency injection instead of global Singletons.

**Pitfall 2 — Hidden coupling.** Code that calls `AppConfig()` directly is secretly coupled to that global. It's difficult to use different configs in different parts of the system (e.g., test DB vs. prod DB). Mitigation: inject config objects rather than accessing the Singleton directly; the Singleton becomes a convenience at the composition root only.

---

**7. Builder vs. `@dataclass`**

`@dataclass(kw_only=True)` handles simple optional parameters well. Builder adds value when:

1. **Cross-field validation** is required during construction. `HttpRequestBuilder.build()` can check "POST requires body" — this is impossible to express cleanly in `__post_init__` without exposing a partially-valid intermediate state.
2. **Fluent interface for complex sequences.** A `QueryBuilder` might call `.where().order_by().limit()` in any order, and `build()` assembles the final SQL — the order matters for correctness, and the builder enforces it.

---

**8. Observer weak references**

If the bus holds strong references and the subscriber has no other owner, the GC cannot collect it. The subscriber stays alive indefinitely, running callbacks that may access deallocated state or keep large objects (ML models, DB connections) from being released.

`weakref.WeakSet` holds weak references: Python's GC ignores them when deciding whether to collect an object. When the last strong reference disappears, the object is collected and silently removed from the `WeakSet`. No explicit `unsubscribe()` is needed.

---

## Debugging challenge — answers

**Bug 1 — Mutable default argument in `@dataclass`:**
```python
created_at: datetime = datetime.now()   # evaluated ONCE at class definition
```
All `Reservation` instances share the same `created_at`. Fix:
```python
from dataclasses import field
import datetime as dt

created_at: datetime = field(default_factory=dt.datetime.now)
```
SOLID violation: hidden dependency on `datetime.now` — violates DIP (should inject a clock). Also testability issue (non-deterministic).

**Bug 2 — Mutable class variable:**
```python
class ReservationService:
    reservations = []   # shared across ALL instances
```
Fix:
```python
def __init__(self) -> None:
    self.reservations: list[Reservation] = []
```
SOLID violation: SRP (also) — `ReservationService` owns both the collection and the email sending. Mutable class variable is a testability violation.

**Bug 3 — `send_confirmation` embeds SMTP call directly:**
```python
def send_confirmation(self, r: Reservation):
    import smtplib
    # real SMTP
```
`ReservationService` has two responsibilities: booking and emailing. SMTP setup in `__init__` (or method) makes the service untestable without a live server.

Fix: inject a `Notifier` abstraction:
```python
class ReservationService:
    def __init__(self, notifier: Notifier) -> None:
        self._reservations: list[Reservation] = []
        self._notifier = notifier

    def book(self, guest: str, room: int, nights: int) -> Reservation:
        for r in self._reservations:
            if r.room == room:
                raise ValueError("Room occupied")
        r = Reservation(guest, room, nights)
        self._reservations.append(r)
        self._notifier.send(f"Reservation confirmed for {guest}")
        return r
```
SOLID violation fixed: SRP (email separated), DIP (depends on `Notifier` abstraction).

---

## Coding challenge — Library Management System (reference implementation)

```python
# library/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


class LibraryError(Exception):
    """Base domain exception."""


class BookUnavailableError(LibraryError):
    def __init__(self, isbn: str) -> None:
        super().__init__(f"No copy available for ISBN {isbn}")


class BorrowLimitExceededError(LibraryError):
    def __init__(self, member_id: str) -> None:
        super().__init__(f"Member {member_id} already has 3 books")


class UnknownTicketError(LibraryError):
    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"Unknown loan ticket: {ticket_id}")


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    total_copies: int
    available_copies: int = field(init=False)

    def __post_init__(self) -> None:
        self.available_copies = self.total_copies


@dataclass
class Member:
    member_id: str
    name: str


@dataclass
class LoanTicket:
    ticket_id: UUID
    member_id: str
    isbn: str
    checked_out_at: datetime
    due_date: datetime
    returned_at: datetime | None = None
```

```python
# library/repository.py
from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID
from .models import Book, LoanTicket, Member


class BookRepository(ABC):
    @abstractmethod
    def save(self, book: Book) -> None: ...
    @abstractmethod
    def find_by_isbn(self, isbn: str) -> Book | None: ...
    @abstractmethod
    def search_by_title(self, fragment: str) -> list[Book]: ...


class LoanRepository(ABC):
    @abstractmethod
    def save(self, ticket: LoanTicket) -> None: ...
    @abstractmethod
    def find_by_id(self, ticket_id: UUID) -> LoanTicket | None: ...
    @abstractmethod
    def active_loans_for_member(self, member_id: str) -> list[LoanTicket]: ...


class InMemoryBookRepository(BookRepository):
    def __init__(self) -> None:
        self._books: dict[str, Book] = {}

    def save(self, book: Book) -> None:
        self._books[book.isbn] = book

    def find_by_isbn(self, isbn: str) -> Book | None:
        return self._books.get(isbn)

    def search_by_title(self, fragment: str) -> list[Book]:
        f = fragment.lower()
        return [b for b in self._books.values() if f in b.title.lower()]


class InMemoryLoanRepository(LoanRepository):
    def __init__(self) -> None:
        self._loans: dict[UUID, LoanTicket] = {}

    def save(self, ticket: LoanTicket) -> None:
        self._loans[ticket.ticket_id] = ticket

    def find_by_id(self, ticket_id: UUID) -> LoanTicket | None:
        return self._loans.get(ticket_id)

    def active_loans_for_member(self, member_id: str) -> list[LoanTicket]:
        return [
            t for t in self._loans.values()
            if t.member_id == member_id and t.returned_at is None
        ]
```

```python
# library/config.py
import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class LibrarySettings(BaseSettings):
    fine_rate_per_day: float = Field(default=0.25, gt=0)
    loan_period_days: int = Field(default=14, ge=1)
    max_books_per_member: int = Field(default=3, ge=1)
    model_config = {"env_prefix": "LIBRARY_"}


@lru_cache(maxsize=1)
def get_settings() -> LibrarySettings:
    return LibrarySettings()
```

```python
# library/service.py  (Strategy pattern for fine calculation)
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Callable
from uuid import uuid4

from .config import get_settings
from .models import (
    Book, BookUnavailableError, BorrowLimitExceededError,
    LoanTicket, UnknownTicketError,
)
from .repository import BookRepository, LoanRepository

log = logging.getLogger(__name__)


# Strategy pattern: fine calculation algorithm
class FineStrategy(ABC):
    @abstractmethod
    def compute(self, due: datetime, returned: datetime, rate: float) -> float: ...


class DailyFineStrategy(FineStrategy):
    """$rate per day overdue; 0 if returned on time."""
    def compute(self, due: datetime, returned: datetime, rate: float) -> float:
        overdue_days = max(0, (returned - due).days)
        return round(overdue_days * rate, 2)


class LendingService:
    def __init__(
        self,
        book_repo: BookRepository,
        loan_repo: LoanRepository,
        fine_strategy: FineStrategy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._books = book_repo
        self._loans = loan_repo
        self._fine = fine_strategy or DailyFineStrategy()
        self._clock = clock or datetime.now
        self._cfg = get_settings()

    def checkout(self, member_id: str, isbn: str) -> LoanTicket:
        active = self._loans.active_loans_for_member(member_id)
        if len(active) >= self._cfg.max_books_per_member:
            raise BorrowLimitExceededError(member_id)

        book = self._books.find_by_isbn(isbn)
        if book is None or book.available_copies < 1:
            raise BookUnavailableError(isbn)

        book.available_copies -= 1
        self._books.save(book)

        now = self._clock()
        ticket = LoanTicket(
            ticket_id=uuid4(),
            member_id=member_id,
            isbn=isbn,
            checked_out_at=now,
            due_date=now + timedelta(days=self._cfg.loan_period_days),
        )
        self._loans.save(ticket)
        log.info("Checked out %s to %s", isbn, member_id)
        return ticket

    def return_book(self, ticket_id: str) -> float:
        from uuid import UUID
        tid = UUID(ticket_id)
        ticket = self._loans.find_by_id(tid)
        if ticket is None:
            raise UnknownTicketError(ticket_id)

        returned_at = self._clock()
        ticket.returned_at = returned_at
        self._loans.save(ticket)

        book = self._books.find_by_isbn(ticket.isbn)
        if book:
            book.available_copies += 1
            self._books.save(book)

        fine = self._fine.compute(ticket.due_date, returned_at, self._cfg.fine_rate_per_day)
        log.info("Returned %s. Fine: $%.2f", ticket.isbn, fine)
        return fine

    def search(self, title_fragment: str) -> list[Book]:
        return self._books.search_by_title(title_fragment)

    def member_loans(self, member_id: str) -> list[LoanTicket]:
        return self._loans.active_loans_for_member(member_id)
```

```python
# tests/test_lending_service.py
from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from library.models import Book, BookUnavailableError, BorrowLimitExceededError
from library.repository import InMemoryBookRepository, InMemoryLoanRepository
from library.service import LendingService


CHECKOUT = datetime(2024, 6, 1, 10, 0, 0)
ON_TIME  = CHECKOUT + timedelta(days=14)
LATE_2D  = ON_TIME + timedelta(days=2)


def book(isbn="ISBN-001", copies=2) -> Book:
    return Book(isbn=isbn, title="Clean Code", author="Martin", total_copies=copies)


@pytest.fixture
def repos():
    br = InMemoryBookRepository()
    lr = InMemoryLoanRepository()
    br.save(book())
    return br, lr


@pytest.fixture
def service(repos):
    br, lr = repos
    calls = iter([CHECKOUT, ON_TIME])
    return LendingService(br, lr, clock=lambda: next(calls))


def test_checkout_success(service, repos):
    br, _ = repos
    ticket = service.checkout("M1", "ISBN-001")
    assert ticket.isbn == "ISBN-001"
    assert br.find_by_isbn("ISBN-001").available_copies == 1


def test_return_on_time_no_fine(repos):
    br, lr = repos
    calls = iter([CHECKOUT, ON_TIME])
    svc = LendingService(br, lr, clock=lambda: next(calls))
    ticket = svc.checkout("M1", "ISBN-001")
    fine = svc.return_book(str(ticket.ticket_id))
    assert fine == 0.0


def test_return_late_fine(repos):
    br, lr = repos
    calls = iter([CHECKOUT, LATE_2D])
    svc = LendingService(br, lr, clock=lambda: next(calls))
    ticket = svc.checkout("M1", "ISBN-001")
    fine = svc.return_book(str(ticket.ticket_id))
    assert fine == pytest.approx(0.50)   # 2 days * $0.25/day


def test_borrow_limit(repos):
    br, lr = repos
    # Add 3 more books
    for i in range(2, 5):
        br.save(Book(isbn=f"ISBN-00{i}", title=f"Book {i}", author="X", total_copies=1))
    calls = iter([CHECKOUT] * 10)
    svc = LendingService(br, lr, clock=lambda: next(calls))
    svc.checkout("M1", "ISBN-001")
    svc.checkout("M1", "ISBN-002")
    svc.checkout("M1", "ISBN-003")
    with pytest.raises(BorrowLimitExceededError):
        svc.checkout("M1", "ISBN-004")


def test_unavailable_book(repos):
    br, lr = repos
    br.save(Book(isbn="ISBN-999", title="Rare", author="X", total_copies=1))
    calls = iter([CHECKOUT, CHECKOUT])
    svc = LendingService(br, lr, clock=lambda: next(calls))
    svc.checkout("M1", "ISBN-999")
    with pytest.raises(BookUnavailableError):
        svc.checkout("M2", "ISBN-999")


def test_search_case_insensitive(repos):
    br, lr = repos
    results = LendingService(br, lr).search("clean")
    assert any("Clean Code" in b.title for b in results)

    results2 = LendingService(br, lr).search("CLEAN")
    assert len(results) == len(results2)
```
