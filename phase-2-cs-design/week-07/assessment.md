# Week 07 Assessment — Low-Level Design & Design Patterns

> Duration: ~1 hr  ·  Pass gate: 75%  ·  Answer key: ../solutions/week-07/assessment.md

---

## Conceptual (interview-style) — 25%

Answer each question in 3–6 sentences. Precision counts more than length.

1. **SOLID — SRP vs. ISP.** Explain the difference between the Single Responsibility Principle and the Interface Segregation Principle. Give a concrete Python example where they are both violated and show how to fix each violation.

2. **Strategy vs. Conditional.** A colleague argues that `if/elif` chains are simpler than the Strategy pattern. When is the colleague right, and when should you reach for Strategy instead? Use the `DiscountStrategy` from Day 4 as a concrete reference.

3. **Decorator (GoF) vs. Python `@decorator`.** Explain the relationship and difference between the GoF Decorator pattern and Python's `@decorator` syntax. Give an example where only the GoF pattern is appropriate.

4. **Repository pattern.** Why does the Repository pattern improve testability? What is the difference between a Fake and a Mock in this context, and when would you choose each?

5. **Dependency Inversion.** What is the "Dependency Inversion" in DIP? What exactly is inverted, and why does it matter for large systems?

6. **Singleton pitfalls.** Name two risks of the Singleton pattern in a production Python service, and describe a mitigation for each.

7. **Builder vs. `@dataclass`.** When does a Builder class add value over a `@dataclass(kw_only=True)`? Give a scenario where the Builder's validation capability is necessary.

8. **Observer weak references.** A long-lived `EventBus` holds strong references to all subscribed observers. Describe the problem this causes and how `weakref.WeakSet` addresses it.

---

## Coding challenge — 40%

### Spec: Library Management System (LLD)

Design and implement a **library book-lending system** with the following requirements:

**Functional requirements:**
- A library has multiple **copies** of each **book** (identified by ISBN).
- A **member** can borrow up to 3 books at a time.
- `checkout(member_id, isbn)` → returns a `LoanTicket` or raises an appropriate exception.
- `return_book(ticket_id)` → marks the copy available, computes a fine if overdue (configurable rate per day, loaded from environment `LIBRARY_FINE_RATE`).
- `search(title_fragment)` → returns all books whose title contains the fragment (case-insensitive).
- `member_loans(member_id)` → returns all active loans for a member.

**Technical requirements:**
- Package layout: `library/{models.py, repository.py, service.py, config.py}`.
- Abstract `BookRepository` + `InMemoryBookRepository`.
- Abstract `LoanRepository` + `InMemoryLoanRepository`.
- `LendingService` with all dependencies injected.
- Domain exception hierarchy (`LibraryError` base).
- At least one GoF pattern explicitly applied (name it in a comment).
- Full `pytest` suite: unit tests for domain logic, integration tests for `LendingService`.
- Config via `pydantic-settings`; no hardcoded values.
- Type hints throughout.

**Acceptance criteria:**
- [ ] `checkout` enforces the 3-book limit.
- [ ] `checkout` raises `BookUnavailableError` when all copies are on loan.
- [ ] `return_book` correctly computes a fine for a 2-day overdue return.
- [ ] `search` is case-insensitive and returns partial matches.
- [ ] All tests pass; coverage ≥ 80%.

---

## Debugging challenge — 20%

The following implementation has **three bugs**. Find and fix each one. For each fix, state which SOLID principle or pattern is also being violated.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Reservation:
    guest: str
    room: int
    nights: int
    created_at: datetime = datetime.now()   # Bug 1

class ReservationService:
    reservations = []                        # Bug 2

    def book(self, guest: str, room: int, nights: int):
        for r in self.reservations:
            if r.room == room:
                raise ValueError("Room occupied")
        r = Reservation(guest, room, nights)
        self.reservations.append(r)
        self.send_confirmation(r)            # Bug 3 (design issue)
        return r

    def send_confirmation(self, r: Reservation):
        import smtplib
        # ... real SMTP call ...
        pass
```

---

## Code quality & testing — 15%

For the code you write in the Coding Challenge:
- [ ] All public functions/methods have type hints.
- [ ] No bare `except:` or catching `Exception` without re-raising or logging.
- [ ] Logging uses `logging.getLogger(__name__)`, not `print`.
- [ ] Tests use fixtures (not setup/teardown methods).
- [ ] No hardcoded secrets or config values — all via `os.getenv` / `pydantic-settings`.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Full marks criteria |
|-----------|--------|---------------------|
| Independent implementation (Coding Challenge) | 40% | All acceptance criteria met; patterns named and applied correctly |
| Conceptual understanding | 25% | 6+/8 conceptual questions answered accurately |
| Debugging & problem-solving | 20% | All 3 bugs found and fixed with SOLID violation named |
| Code quality & testing | 15% | All 5 quality checklist items satisfied; coverage ≥ 80% |

Answer key: ../solutions/week-07/assessment.md
