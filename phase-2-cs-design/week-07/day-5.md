# Phase 2 · Week 07 · Day 5 — Dependency Injection & Testable Object-Oriented Design

> Objective: Apply constructor injection, understand IoC containers, and structure OO code so every class is testable in isolation.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

### What is Dependency Injection?

**Dependency Injection (DI)** is a technique where an object receives its dependencies from the outside rather than creating them internally.

Without DI:
```python
class OrderService:
    def __init__(self):
        self._repo = PostgresRepository()   # hard-wired — untestable
        self._mailer = SmtpMailer()         # hard-wired
```

With DI:
```python
class OrderService:
    def __init__(self, repo: OrderRepository, mailer: Mailer) -> None:
        self._repo = repo     # injected — can be mocked
        self._mailer = mailer # injected
```

The caller (or a container) decides what concrete objects to pass. The class expresses only what *kind* of collaborator it needs (the abstraction).

### Three forms of injection

| Form | Mechanism | Use when |
|------|-----------|----------|
| Constructor injection | `__init__` parameters | Required dependencies; prefer this |
| Property injection | `obj.repo = ...` | Optional overrides; rare in Python |
| Method injection | pass dep as argument | Dependency varies per call |

### Inversion of Control (IoC) containers

A container automates DI: you register types (or factories) and the container resolves the dependency graph at startup. Python options: `dependency-injector`, `punq`, `lagom`. For this course we use manual constructor injection (simpler, explicit, testable).

### Designing for testability

A class is testable in isolation when:
1. All collaborators are injected (not created internally).
2. No hidden global state or singleton access.
3. Side effects (I/O, time, randomness) are behind interfaces that can be replaced by test doubles.

**Test double taxonomy:**
- **Stub**: returns canned data, no assertions.
- **Mock**: records calls; test asserts what was called.
- **Fake**: lightweight real implementation (e.g., `InMemoryRepository`).
- **Spy**: real object that records calls too.

---

## How it works internally

**`unittest.mock.MagicMock`**: Python's mock library creates objects that accept any method call and record it. Combine with `spec=SomeClass` to restrict to real interface methods — prevents tests from passing due to typos.

**`@pytest.fixture` and DI**: A pytest fixture is a function that *constructs* a dependency. Fixtures compose — a high-level fixture can receive a lower-level fixture. This mirrors the IoC container pattern.

**`@dataclass` + `__post_init__` for validation**: Validate injected dependencies in `__post_init__` to fail fast rather than at first use.

---

## Worked example

```python
"""
day5_di_testable.py
Manual DI with constructor injection + pytest tests.
Run tests: pytest day5_di_testable.py -v
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Domain ──────────────────────────────────────────────────────────────────

@dataclass
class User:
    username: str
    email: str
    user_id: UUID = field(default_factory=uuid4)


# ─── Abstractions (ports) ────────────────────────────────────────────────────

class UserRepository(Protocol):
    def save(self, user: User) -> None: ...
    def find_by_email(self, email: str) -> User | None: ...


class EmailService(Protocol):
    def send_welcome(self, email: str, username: str) -> None: ...


class IdGenerator(Protocol):
    def new_id(self) -> UUID: ...


# ─── Concrete implementations ─────────────────────────────────────────────────

class InMemoryUserRepository:
    """Fake — suitable for both tests and early prototypes."""

    def __init__(self) -> None:
        self._users: dict[UUID, User] = {}

    def save(self, user: User) -> None:
        self._users[user.user_id] = user

    def find_by_email(self, email: str) -> User | None:
        return next(
            (u for u in self._users.values() if u.email == email), None
        )


class LoggingEmailService:
    """Stub implementation — logs rather than sending real email."""

    def send_welcome(self, email: str, username: str) -> None:
        log.info("[EMAIL] Welcome email to %s (%s)", username, email)


class UUIDGenerator:
    def new_id(self) -> UUID:
        return uuid4()


# ─── Application service (all deps injected) ─────────────────────────────────

class UserRegistrationService:
    """
    Registers new users.
    All collaborators are injected — fully testable without real I/O.
    """

    def __init__(
        self,
        repo: UserRepository,
        email_service: EmailService,
        id_gen: IdGenerator,
    ) -> None:
        self._repo = repo
        self._email_service = email_service
        self._id_gen = id_gen

    def register(self, username: str, email: str) -> User:
        if not username.strip():
            raise ValueError("Username cannot be blank")
        if "@" not in email:
            raise ValueError(f"Invalid email: {email}")

        existing = self._repo.find_by_email(email)
        if existing is not None:
            raise ValueError(f"Email already registered: {email}")

        user = User(username=username, email=email, user_id=self._id_gen.new_id())
        self._repo.save(user)
        self._email_service.send_welcome(email, username)
        log.info("Registered user %s", user.user_id)
        return user


# ─── Wiring (production) ─────────────────────────────────────────────────────

def build_service() -> UserRegistrationService:
    """Composition root: assemble the object graph."""
    return UserRegistrationService(
        repo=InMemoryUserRepository(),
        email_service=LoggingEmailService(),
        id_gen=UUIDGenerator(),
    )


# ─── Tests ───────────────────────────────────────────────────────────────────
# Run with: pytest day5_di_testable.py -v

import pytest
from unittest.mock import MagicMock, call
from uuid import UUID


@pytest.fixture()
def fixed_id() -> UUID:
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture()
def repo() -> InMemoryUserRepository:
    return InMemoryUserRepository()


@pytest.fixture()
def email_mock() -> MagicMock:
    return MagicMock(spec=["send_welcome"])


@pytest.fixture()
def id_gen_mock(fixed_id: UUID) -> MagicMock:
    mock = MagicMock(spec=["new_id"])
    mock.new_id.return_value = fixed_id
    return mock


@pytest.fixture()
def service(repo, email_mock, id_gen_mock) -> UserRegistrationService:
    return UserRegistrationService(repo=repo, email_service=email_mock, id_gen=id_gen_mock)


class TestUserRegistration:
    def test_successful_registration(self, service, repo, email_mock, fixed_id):
        user = service.register("alice", "alice@example.com")

        assert user.username == "alice"
        assert user.user_id == fixed_id
        assert repo.find_by_email("alice@example.com") is not None
        email_mock.send_welcome.assert_called_once_with("alice@example.com", "alice")

    def test_duplicate_email_raises(self, service):
        service.register("alice", "alice@example.com")
        with pytest.raises(ValueError, match="already registered"):
            service.register("alice2", "alice@example.com")

    def test_blank_username_raises(self, service):
        with pytest.raises(ValueError, match="blank"):
            service.register("  ", "alice@example.com")

    def test_invalid_email_raises(self, service):
        with pytest.raises(ValueError, match="Invalid email"):
            service.register("bob", "not-an-email")

    def test_email_service_not_called_on_validation_failure(self, service, email_mock):
        with pytest.raises(ValueError):
            service.register("", "bad")
        email_mock.send_welcome.assert_not_called()


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    svc = build_service()
    u = svc.register("bob", "bob@example.com")
    log.info("Created: %s (%s)", u.username, u.user_id)
```

Run tests:
```bash
pytest day5_di_testable.py -v
```

---

## Exercises

1. **Property injection.** Add an optional `logger: logging.Logger` dependency to `UserRegistrationService` via a property setter (not constructor). Default to `logging.getLogger(__name__)`. Write a test that injects a mock logger and asserts `logger.info` was called after successful registration.

2. **Composition root.** Create a `build_service_from_env()` function that reads `SMTP_HOST`, `SMTP_PORT` from environment variables (use `os.getenv`) and instantiates a `SmtpEmailService` stub (class signature only, no real SMTP). The function returns a `UserRegistrationService`. Never hardcode credentials.

3. **Test the duplicate check.** Write a test that pre-populates the `InMemoryUserRepository` with a user, then calls `register()` and asserts the `email_mock.send_welcome` was never called.

4. **IoC container.** Install `dependency-injector` in a scratch venv and wire `UserRegistrationService` using a `containers.DeclarativeContainer`. Show the `providers.Factory` declarations for each dependency.

5. **Testability audit.** Review the following class for testability issues and list every problem:
   ```python
   import smtplib, time, random

   class WelcomeMailer:
       def __init__(self):
           self._conn = smtplib.SMTP("smtp.gmail.com", 587)
           self._token = random.randint(1000, 9999)
           self._sent_at = time.time()

       def send(self, to: str) -> None:
           self._conn.sendmail("noreply@co.com", to, f"Token: {self._token}")
   ```

(Solutions: ../solutions/week-07/day-5.md)

---

## Self-check

- [ ] I can explain the three forms of DI and choose the right one.
- [ ] I can structure a class with all dependencies injected (constructor injection).
- [ ] I can write a composition root that assembles the object graph.
- [ ] I can write pytest tests using MagicMock for collaborators.
- [ ] I can identify testability anti-patterns: hidden state, global access, side effects in constructors.

---

## Supplementary links

- [Dependency Injection — Martin Fowler](https://martinfowler.com/articles/injection.html)
- [Python `dependency-injector` library](https://python-dependency-injector.ets-labs.org/)
- [Python `unittest.mock` docs](https://docs.python.org/3/library/unittest.mock.html)
- *Growing Object-Oriented Software, Guided by Tests* — Freeman & Pryce (foundational DI + testing book)
