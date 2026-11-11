# Phase 2 · Week 07 · Day 1 — OOD Process, UML Basics & SOLID Recap

> Objective: Apply the OOD design process end-to-end, read/write class diagrams, and articulate all five SOLID principles with code examples.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-7]]

---

## Concept

### The OOD Design Process

Object-Oriented Design is not "draw boxes and arrows." It is a structured thinking process:

1. **Gather requirements** — clarify actors, use-cases, constraints (time, memory, concurrency).
2. **Identify nouns → candidate classes** — scan the spec for entities and concepts.
3. **Identify verbs → responsibilities** — what does each object *do*? Assign single responsibilities.
4. **Define relationships** — inheritance vs. composition vs. dependency; prefer composition.
5. **Apply patterns** — recognise recurring problems; reach for the canonical solution.
6. **Iterate** — LLD is not one-shot; feedback from coding exposes bad abstractions early.

The key insight: **objects model behaviour, not data**. A class is a contract (interface) backed by an implementation, not a struct with methods bolted on.

### UML Class Diagram Basics

You need only six notations for LLD interviews and design docs:

| Symbol | Meaning |
|--------|---------|
| `+` prefix | public |
| `-` prefix | private |
| `#` prefix | protected |
| Solid arrow (`→`) | association / dependency |
| Open-triangle arrow (`▷`) | inheritance / realisation |
| Diamond (`◇`) | aggregation / composition |

A minimal class box:

```
┌───────────────────┐
│     ClassName     │
├───────────────────┤
│ - _attr: Type     │
├───────────────────┤
│ + method(): Type  │
└───────────────────┘
```

Composition (filled diamond) means the child cannot exist without the parent. Aggregation (open diamond) means shared ownership.

### SOLID Principles

| Letter | Principle | One-line summary |
|--------|-----------|-----------------|
| S | Single Responsibility | One reason to change |
| O | Open/Closed | Open for extension, closed for modification |
| L | Liskov Substitution | Subtypes must be substitutable for base types |
| I | Interface Segregation | Clients should not depend on interfaces they don't use |
| D | Dependency Inversion | Depend on abstractions, not concretions |

---

## How it works internally

**SRP at the module level**: Python has no formal `interface` keyword, but `abc.ABC` + `abstractmethod` gives you contracts. The interpreter enforces that concrete subclasses implement every abstract method — instantiating an abstract class raises `TypeError`.

**LSP in Python's type system**: `mypy` / `pyright` enforces LSP statically. At runtime Python is duck-typed — violations silently pass until the wrong method is called.

**DIP wiring**: Python's default argument trick (`def __init__(self, repo: AbstractRepo = None`) is poor man's DI. Production code uses an IoC container or explicit constructor injection.

---

## Worked example

```python
"""
day1_solid_demo.py
Illustrates all five SOLID principles with a minimal notification system.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── S: Single Responsibility ────────────────────────────────────────────────
# OrderProcessor only handles order logic; notification is a separate concern.

@dataclass
class Order:
    order_id: str
    amount: float


class OrderProcessor:
    """Validates and records orders — nothing else."""

    def process(self, order: Order) -> bool:
        if order.amount <= 0:
            log.warning("Invalid order %s", order.order_id)
            return False
        log.info("Order %s processed", order.order_id)
        return True


# ─── O: Open/Closed ──────────────────────────────────────────────────────────
# Add new notification channels without touching existing code.

class Notifier(ABC):
    @abstractmethod
    def send(self, message: str) -> None: ...


class EmailNotifier(Notifier):
    def send(self, message: str) -> None:
        log.info("[EMAIL] %s", message)


class SlackNotifier(Notifier):
    def send(self, message: str) -> None:
        log.info("[SLACK] %s", message)


# ─── L: Liskov Substitution ──────────────────────────────────────────────────
# Any Notifier subtype can replace the base without breaking callers.

def notify_all(notifiers: list[Notifier], msg: str) -> None:
    for n in notifiers:
        n.send(msg)  # works for Email or Slack or any future type


# ─── I: Interface Segregation ────────────────────────────────────────────────
# Split fat interface into focused ones; classes implement only what they need.

class Persistable(ABC):
    @abstractmethod
    def save(self) -> None: ...


class Auditable(ABC):
    @abstractmethod
    def audit_log(self) -> str: ...


@dataclass
class AuditableOrder(Order, Auditable):
    """An order that can produce an audit trail — without forcing Persistable."""

    def audit_log(self) -> str:
        return f"Order {self.order_id} amount={self.amount}"


# ─── D: Dependency Inversion ─────────────────────────────────────────────────
# OrderService depends on abstract Notifier, injected at construction time.

class OrderService:
    def __init__(self, processor: OrderProcessor, notifier: Notifier) -> None:
        self._processor = processor
        self._notifier = notifier

    def submit(self, order: Order) -> None:
        if self._processor.process(order):
            self._notifier.send(f"Order {order.order_id} confirmed")


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    service = OrderService(
        processor=OrderProcessor(),
        notifier=EmailNotifier(),   # swap to SlackNotifier() with zero code change
    )
    service.submit(Order("ORD-1", 99.99))
    service.submit(Order("ORD-2", -5.00))
```

Run it:
```bash
python day1_solid_demo.py
```

Expected output (INFO level):
```
INFO Order ORD-1 processed
INFO [EMAIL] Order ORD-1 confirmed
WARNING Invalid order ORD-2
```

---

## Exercises

1. **SRP violation hunt.** The class below violates SRP. Name the two responsibilities and refactor it into two classes:
   ```python
   class UserManager:
       def create_user(self, name: str, email: str) -> dict: ...
       def send_welcome_email(self, email: str) -> None: ...
       def export_users_to_csv(self, path: str) -> None: ...
   ```

2. **OCP extension.** Add a `PushNotifier` channel to the worked example above without modifying any existing class. Write the class and show how to wire it into `OrderService`.

3. **LSP check.** Explain why the following subclass violates LSP and fix it:
   ```python
   class ReadOnlyNotifier(Notifier):
       def send(self, message: str) -> None:
           raise NotImplementedError("This notifier is read-only")
   ```

4. **UML sketch.** Draw (ASCII or describe) a class diagram for `OrderService`, `OrderProcessor`, `Notifier`, `EmailNotifier`, and `SlackNotifier`, showing relationships and cardinalities.

5. **SOLID audit.** Which SOLID principles does Python's `abc.ABC` mechanism directly support? Explain each.

(Solutions: ../solutions/week-07/day-1.md)

---

## Self-check

- [ ] I can describe the six-step OOD process from memory.
- [ ] I can read and write a UML class diagram with associations and inheritance.
- [ ] I can give a concrete code example for each of the five SOLID principles.
- [ ] I can identify SOLID violations in existing code and propose fixes.

---

## Supplementary links

- *Clean Architecture* — Robert C. Martin, Chapter 7 (SRP)
- [Python `abc` module docs](https://docs.python.org/3/library/abc.html)
- [SOLID explained with Python](https://realpython.com/solid-principles-python/) — Real Python
- *Head First Design Patterns* — Freeman & Robson, Chapter 1 (strategy motivates OCP)
