# Solutions — Week 07 · Day 1

## Exercise 1 — SRP refactor

```python
from __future__ import annotations
import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class UserStore:
    """Responsibility 1: manage user records."""

    def __init__(self) -> None:
        self._users: list[dict] = []

    def create_user(self, name: str, email: str) -> dict:
        user = {"name": name, "email": email}
        self._users.append(user)
        log.info("User created: %s", email)
        return user

    def all_users(self) -> list[dict]:
        return list(self._users)


class UserEmailer:
    """Responsibility 2: send emails."""

    def send_welcome_email(self, email: str) -> None:
        log.info("Welcome email → %s", email)


class UserCsvExporter:
    """Responsibility 3: export data."""

    def export_users_to_csv(self, users: list[dict], path: Path) -> None:
        path.write_text("\n".join(f"{u['name']},{u['email']}" for u in users))
        log.info("Exported %d users to %s", len(users), path)
```

The original `UserManager` had three reasons to change: user storage changes, email vendor changes, and export format changes. Each is now an independent class.

---

## Exercise 2 — OCP extension (PushNotifier)

```python
from day1_solid_demo import Notifier, OrderService, OrderProcessor
import logging

log = logging.getLogger(__name__)


class PushNotifier(Notifier):
    """New channel added without touching existing code."""

    def send(self, message: str) -> None:
        log.info("[PUSH] %s", message)


# Wire it in — zero modification to OrderService or OrderProcessor
service = OrderService(
    processor=OrderProcessor(),
    notifier=PushNotifier(),
)
```

---

## Exercise 3 — LSP fix

The violation: `ReadOnlyNotifier` cannot be substituted for `Notifier` because it raises on `send()`, breaking any caller that depends on `send()` actually sending.

**Fix option A** — Remove `ReadOnlyNotifier` entirely. An object that can never perform its primary responsibility shouldn't exist as a subtype.

**Fix option B** — Introduce a `NullNotifier` that does nothing (valid no-op), and `ReadOnlyNotifier` becomes a `LoggingNotifier` that logs without sending.

```python
class NullNotifier(Notifier):
    """Valid LSP-compliant implementation: no-op send."""
    def send(self, message: str) -> None:
        pass  # intentional no-op
```

---

## Exercise 4 — UML sketch

```
┌─────────────────────┐          ┌──────────────────────┐
│   OrderService      │────────▷ │   OrderProcessor     │
├─────────────────────┤  depends ├──────────────────────┤
│ - _processor        │          │ + process(Order):bool│
│ - _notifier         │          └──────────────────────┘
├─────────────────────┤
│ + submit(Order):void│          ┌──────────────┐
└──────────┬──────────┘          │  «abstract»  │
           │ depends on          │  Notifier    │
           └────────────────────▷├──────────────┤
                                 │ + send(str)  │
                                 └──────┬───────┘
                              ┌─────────┴──────────┐
                  ┌───────────┴────┐     ┌──────────┴──────────┐
                  │  EmailNotifier │     │   SlackNotifier     │
                  │ + send(str)    │     │ + send(str)         │
                  └────────────────┘     └─────────────────────┘
```

Cardinality: `OrderService` has exactly one `Notifier` (1..1) and one `OrderProcessor` (1..1). `Notifier` has two concrete realisations.

---

## Exercise 5 — SOLID principles supported by `abc.ABC`

- **S (SRP):** `ABC` encourages thin, focused interfaces — each abstract class defines one concept. (Indirect support.)
- **O (OCP):** You extend by subclassing and implementing abstract methods without modifying the base. (Direct support.)
- **L (LSP):** The interpreter enforces that subclasses implement all abstract methods, making substitution contractually sound. (Direct support.)
- **I (ISP):** You can define multiple small ABCs and have a concrete class inherit only the ones it needs. (Direct support.)
- **D (DIP):** Depending on an `ABC` rather than a concrete class is the textbook form of DIP. (Direct support.)
