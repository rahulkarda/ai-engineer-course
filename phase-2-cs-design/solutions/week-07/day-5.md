# Solutions — Week 07 · Day 5

## Exercise 1 — Property injection for logger

```python
import logging
from day5_di_testable import UserRegistrationService, InMemoryUserRepository, UUIDGenerator


class LoggableUserRegistrationService(UserRegistrationService):
    """Extends service with optional logger property injection."""

    def __init__(self, repo, email_service, id_gen) -> None:
        super().__init__(repo, email_service, id_gen)
        self._logger = logging.getLogger(__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, log: logging.Logger) -> None:
        self._logger = log

    def register(self, username, email):
        user = super().register(username, email)
        self._logger.info("Registered via LoggableService: %s", user.username)
        return user


# Test
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4


def test_logger_called_on_success():
    svc = LoggableUserRegistrationService(
        repo=InMemoryUserRepository(),
        email_service=MagicMock(spec=["send_welcome"]),
        id_gen=MagicMock(spec=["new_id"], **{"new_id.return_value": uuid4()}),
    )
    mock_log = MagicMock(spec=logging.Logger)
    svc.logger = mock_log

    svc.register("alice", "alice@example.com")
    mock_log.info.assert_called_once()
```

---

## Exercise 2 — Composition root from env

```python
import os
import logging
from day5_di_testable import UserRegistrationService, InMemoryUserRepository, UUIDGenerator

log = logging.getLogger(__name__)


class SmtpEmailService:
    """Stub — structure only. Real SMTP implementation omitted."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        log.info("SmtpEmailService configured: %s:%d", host, port)

    def send_welcome(self, email: str, username: str) -> None:
        log.info("[SMTP] Would send welcome to %s via %s:%d", email, self._host, self._port)


def build_service_from_env() -> UserRegistrationService:
    """
    Reads SMTP_HOST and SMTP_PORT from environment.
    Falls back to safe defaults for local dev.
    Never hardcodes credentials.
    """
    host = os.getenv("SMTP_HOST", "localhost")
    port = int(os.getenv("SMTP_PORT", "587"))
    return UserRegistrationService(
        repo=InMemoryUserRepository(),
        email_service=SmtpEmailService(host=host, port=port),
        id_gen=UUIDGenerator(),
    )
```

---

## Exercise 3 — Duplicate check with no email sent

```python
import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from day5_di_testable import (
    InMemoryUserRepository, UserRegistrationService, User
)
from uuid import UUID


def make_service(email_mock):
    id_gen = MagicMock(spec=["new_id"])
    id_gen.new_id.return_value = uuid4()
    repo = InMemoryUserRepository()
    # Pre-populate with an existing user
    existing = User(username="carol", email="carol@example.com")
    repo.save(existing)
    return UserRegistrationService(repo=repo, email_service=email_mock, id_gen=id_gen)


def test_duplicate_does_not_call_email():
    email_mock = MagicMock(spec=["send_welcome"])
    svc = make_service(email_mock)

    with pytest.raises(ValueError, match="already registered"):
        svc.register("carol2", "carol@example.com")

    email_mock.send_welcome.assert_not_called()
```

---

## Exercise 4 — `dependency-injector` wiring

```python
# containers.py
from dependency_injector import containers, providers
from day5_di_testable import (
    InMemoryUserRepository,
    LoggingEmailService,
    UUIDGenerator,
    UserRegistrationService,
)


class ApplicationContainer(containers.DeclarativeContainer):
    repo = providers.Factory(InMemoryUserRepository)
    email_service = providers.Factory(LoggingEmailService)
    id_gen = providers.Factory(UUIDGenerator)

    user_service = providers.Factory(
        UserRegistrationService,
        repo=repo,
        email_service=email_service,
        id_gen=id_gen,
    )


# Usage
container = ApplicationContainer()
svc = container.user_service()
svc.register("dan", "dan@example.com")
```

---

## Exercise 5 — Testability audit of `WelcomeMailer`

**Bug / Testability issue 1 — SMTP connection in `__init__`:**
`smtplib.SMTP(...)` opens a real network connection during construction. Every `WelcomeMailer()` call in tests requires a live SMTP server or will raise. Fix: accept an injectable `smtp_factory: Callable[[], smtplib.SMTP]` parameter.

**Bug / Testability issue 2 — `random.randint` in `__init__`:**
`self._token` is random — tests cannot predict or assert its value. Also, tokens generated at construction time can expire or be replayed. Fix: inject an `IdGenerator` or call `random.randint` inside `send()`, and abstract it behind a `TokenGenerator` protocol.

**Bug / Testability issue 3 — `time.time()` in `__init__`:**
`self._sent_at` is frozen at construction time, not send time. Fix: call `time.time()` (or an injected clock) inside `send()`.

**SOLID violations:**
- **SRP:** `WelcomeMailer` does connection management, token generation, time tracking, and sending — four responsibilities.
- **DIP:** Depends on `smtplib.SMTP` concretion, `random`, and `time` — all built-in concretions rather than injected abstractions.

**Fixed skeleton:**
```python
import logging
from typing import Callable, Protocol
from datetime import datetime

log = logging.getLogger(__name__)


class SmtpConnection(Protocol):
    def sendmail(self, from_addr: str, to_addrs: str, msg: str) -> None: ...


class TokenGenerator(Protocol):
    def generate(self) -> int: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class WelcomeMailer:
    def __init__(
        self,
        conn: SmtpConnection,
        token_gen: TokenGenerator,
        clock: Clock,
    ) -> None:
        self._conn = conn
        self._token_gen = token_gen
        self._clock = clock

    def send(self, to: str) -> None:
        token = self._token_gen.generate()
        sent_at = self._clock.now()
        log.info("Sending to %s at %s token=%d", to, sent_at, token)
        self._conn.sendmail("noreply@co.com", to, f"Token: {token}")
```
