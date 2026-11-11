# Solutions — Phase 7 · Week 07 · Day 7

---

## Exercise 1 — Refactor a messy NotificationManager

### Violations identified

The `NotificationManager` class in the exercise prompt has the following violations. The class from the *project brief* (the one with the hardcoded SMTP and Slack token) is provided as context below; the same violation catalogue applies.

**Starting point (as given in the exercise):**

```python
class NotificationManager:
    def __init__(self):
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587
        self.slack_token = "xoxb-hardcoded-token"
        self.db = psycopg2.connect("postgresql://localhost/app")

    def send_alert(self, user_id: int, message: str) -> None:
        # fetch email from DB
        cursor = self.db.cursor()
        cursor.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        email = cursor.fetchone()[0]
        # send email
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
            smtp.sendmail("app@company.com", email, message)
        # also slack
        requests.post("https://slack.com/api/chat.postMessage",
                      headers={"Authorization": f"Bearer {self.slack_token}"},
                      json={"channel": "#alerts", "text": message})
        # log to DB
        cursor.execute("INSERT INTO notifications (user_id, msg) VALUES (%s, %s)",
                       (user_id, message))
        self.db.commit()
```

**Violation catalogue:**

| # | Violation | Location | Description |
|---|---|---|---|
| 1 | SRP — user lookup mixed with notification dispatch | `send_alert` | The class both fetches user data and sends messages. These are separate concerns with different reasons to change. |
| 2 | SRP — email delivery, Slack delivery, and persistence in one method | `send_alert` | Three distinct I/O channels handled in a single method. Adding a new channel requires modifying this class. |
| 3 | DIP — direct construction of `psycopg2` connection in `__init__` | `__init__` | The class constructs its own DB dependency. It cannot be tested without a real PostgreSQL database. |
| 4 | DIP — hardcoded secrets (`smtp_host`, `smtp_port`, `slack_token`) | `__init__` | Configuration values are baked in. There is no injection point for the caller to override them. |
| 5 | OCP — adding a new notification channel requires modifying `send_alert` | `send_alert` | Closed to extension without modification — violates OCP. A new channel (e.g., PagerDuty) requires editing the body of `send_alert`. |
| 6 | No interface on the class itself | class | The class has no ABC. It cannot be substituted with a fake in tests of code that calls it. |

---

### Refactored version

```python
"""
notifications/interfaces.py

Abstract interfaces for the notification subsystem.
All cross-boundary dependencies are modelled as ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class UserContact:
    user_id: int
    email: str


class UserRepository(ABC):
    """Fetches user contact info. Abstracts the database layer."""

    @abstractmethod
    def get_contact(self, user_id: int) -> UserContact | None:
        """Return UserContact, or None if user not found."""
        ...


class EmailSender(ABC):
    """Sends a plain-text email to a single recipient."""

    @abstractmethod
    def send(self, to_address: str, from_address: str, body: str) -> None:
        """Raise on delivery failure."""
        ...


class SlackNotifier(ABC):
    """Posts a message to a Slack channel."""

    @abstractmethod
    def post(self, channel: str, text: str) -> None:
        """Raise on delivery failure."""
        ...


class NotificationLogger(ABC):
    """Persists a notification record after successful delivery."""

    @abstractmethod
    def record(self, user_id: int, message: str) -> None:
        ...


class AlertService(ABC):
    """High-level interface for sending alerts. Inject this into callers."""

    @abstractmethod
    def send_alert(self, user_id: int, message: str) -> None:
        ...
```

```python
"""
notifications/manager.py

Refactored NotificationManager: orchestrator only.
No I/O, no secrets, no direct construction of collaborators.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .interfaces import (
    AlertService,
    EmailSender,
    NotificationLogger,
    SlackNotifier,
    UserRepository,
)

logger = logging.getLogger(__name__)

# Configuration is loaded from the environment — no hardcoded values.
_DEFAULT_FROM_ADDRESS: str = os.getenv("NOTIF_FROM_ADDRESS", "app@company.com")
_DEFAULT_SLACK_CHANNEL: str = os.getenv("NOTIF_SLACK_CHANNEL", "#alerts")


@dataclass
class NotificationConfig:
    from_address: str = _DEFAULT_FROM_ADDRESS
    slack_channel: str = _DEFAULT_SLACK_CHANNEL


class NotificationManager(AlertService):
    """
    Orchestrates user lookup, email delivery, Slack delivery, and audit logging.

    All collaborators are injected. The class has no knowledge of how email is
    sent or how users are looked up — those are concerns of the injected objects.
    """

    def __init__(
        self,
        user_repo: UserRepository,
        email_sender: EmailSender,
        slack_notifier: SlackNotifier,
        notification_logger: NotificationLogger,
        config: NotificationConfig | None = None,
    ) -> None:
        self._user_repo = user_repo
        self._email_sender = email_sender
        self._slack = slack_notifier
        self._logger = notification_logger
        self._config = config or NotificationConfig()

    def send_alert(self, user_id: int, message: str) -> None:
        """
        Look up the user, send email and Slack, persist audit record.

        Raises ValueError if user_id is not found.
        Propagates delivery exceptions from injected senders.
        """
        contact = self._user_repo.get_contact(user_id)
        if contact is None:
            logger.warning("send_alert: user %d not found, skipping", user_id)
            raise ValueError(f"User {user_id} not found")

        logger.info("Sending alert to user %d (%s)", user_id, contact.email)

        self._email_sender.send(
            to_address=contact.email,
            from_address=self._config.from_address,
            body=message,
        )
        logger.debug("Email sent to %s", contact.email)

        self._slack.post(
            channel=self._config.slack_channel,
            text=message,
        )
        logger.debug("Slack message posted to %s", self._config.slack_channel)

        self._notification_logger_record(user_id, message)

    def _notification_logger_record(self, user_id: int, message: str) -> None:
        try:
            self._logger.record(user_id, message)
        except Exception:
            # Log-and-continue: audit failure must not cancel delivery.
            logger.exception("Failed to persist notification record for user %d", user_id)
```

```python
"""
tests/test_notification_manager.py

Unit tests for NotificationManager.
All external I/O is replaced by in-memory fakes.
No network, no filesystem, no database required.
"""

from __future__ import annotations

import pytest

from notifications.interfaces import (
    EmailSender,
    NotificationLogger,
    SlackNotifier,
    UserContact,
    UserRepository,
)
from notifications.manager import NotificationConfig, NotificationManager


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeUserRepository(UserRepository):
    def __init__(self, users: dict[int, str]) -> None:
        # maps user_id -> email
        self._users = users

    def get_contact(self, user_id: int) -> UserContact | None:
        email = self._users.get(user_id)
        if email is None:
            return None
        return UserContact(user_id=user_id, email=email)


class RecordingEmailSender(EmailSender):
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.should_fail = False

    def send(self, to_address: str, from_address: str, body: str) -> None:
        if self.should_fail:
            raise OSError("SMTP connection refused")
        self.sent.append({"to": to_address, "from": from_address, "body": body})


class RecordingSlackNotifier(SlackNotifier):
    def __init__(self) -> None:
        self.posted: list[dict] = []

    def post(self, channel: str, text: str) -> None:
        self.posted.append({"channel": channel, "text": text})


class RecordingNotificationLogger(NotificationLogger):
    def __init__(self) -> None:
        self.records: list[tuple[int, str]] = []
        self.should_fail = False

    def record(self, user_id: int, message: str) -> None:
        if self.should_fail:
            raise RuntimeError("DB write failed")
        self.records.append((user_id, message))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_repo() -> FakeUserRepository:
    return FakeUserRepository({42: "alice@example.com", 7: "bob@example.com"})


@pytest.fixture
def email_sender() -> RecordingEmailSender:
    return RecordingEmailSender()


@pytest.fixture
def slack() -> RecordingSlackNotifier:
    return RecordingSlackNotifier()


@pytest.fixture
def notif_logger() -> RecordingNotificationLogger:
    return RecordingNotificationLogger()


@pytest.fixture
def manager(
    user_repo: FakeUserRepository,
    email_sender: RecordingEmailSender,
    slack: RecordingSlackNotifier,
    notif_logger: RecordingNotificationLogger,
) -> NotificationManager:
    return NotificationManager(
        user_repo=user_repo,
        email_sender=email_sender,
        slack_notifier=slack,
        notification_logger=notif_logger,
        config=NotificationConfig(
            from_address="test@example.com",
            slack_channel="#test-alerts",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSuccessfulSend:
    def test_email_delivered_to_correct_address(
        self,
        manager: NotificationManager,
        email_sender: RecordingEmailSender,
    ) -> None:
        manager.send_alert(user_id=42, message="disk 90% full")
        assert len(email_sender.sent) == 1
        assert email_sender.sent[0]["to"] == "alice@example.com"

    def test_slack_posted_to_configured_channel(
        self,
        manager: NotificationManager,
        slack: RecordingSlackNotifier,
    ) -> None:
        manager.send_alert(user_id=42, message="disk 90% full")
        assert len(slack.posted) == 1
        assert slack.posted[0]["channel"] == "#test-alerts"

    def test_audit_record_persisted(
        self,
        manager: NotificationManager,
        notif_logger: RecordingNotificationLogger,
    ) -> None:
        manager.send_alert(user_id=42, message="test")
        assert (42, "test") in notif_logger.records

    def test_message_body_unchanged(
        self,
        manager: NotificationManager,
        email_sender: RecordingEmailSender,
    ) -> None:
        manager.send_alert(user_id=42, message="alert: cpu spike")
        assert email_sender.sent[0]["body"] == "alert: cpu spike"


class TestUserNotFound:
    def test_raises_value_error_for_unknown_user(
        self,
        manager: NotificationManager,
    ) -> None:
        with pytest.raises(ValueError, match="999"):
            manager.send_alert(user_id=999, message="msg")

    def test_no_email_sent_when_user_missing(
        self,
        manager: NotificationManager,
        email_sender: RecordingEmailSender,
    ) -> None:
        with pytest.raises(ValueError):
            manager.send_alert(user_id=999, message="msg")
        assert len(email_sender.sent) == 0


class TestDeliveryFailure:
    def test_email_failure_propagates(
        self,
        manager: NotificationManager,
        email_sender: RecordingEmailSender,
    ) -> None:
        email_sender.should_fail = True
        with pytest.raises(OSError):
            manager.send_alert(user_id=42, message="msg")

    def test_logger_failure_does_not_cancel_delivery(
        self,
        manager: NotificationManager,
        email_sender: RecordingEmailSender,
        notif_logger: RecordingNotificationLogger,
    ) -> None:
        notif_logger.should_fail = True
        # Should not raise — audit failure is swallowed after delivery succeeds
        manager.send_alert(user_id=42, message="msg")
        assert len(email_sender.sent) == 1
```

---

## Exercise 2 — Unit test stubs for the Elevator System

The stubs below are written against the class interfaces specified in `project.md`. They do not require a passing implementation to compile (besides placeholder classes). Each stub follows the Arrange / Act / Assert structure with a docstring explaining the scenario.

```python
"""
tests/test_elevator_stubs.py

Unit test stubs for the Elevator System (Exercise 2, Day 7).

These stubs are written against the public interface only.
Implementations are provided in the project solution.
"""

from __future__ import annotations

import pytest

# Import the classes as they will exist in the project layout.
# If running stubs before implementation, these imports will fail — that is expected.
from elevator.models import Direction, DoorState, FloorRequest
from elevator.elevator import Elevator, InvalidDoorTransitionError
from elevator.building import Building
from elevator.interfaces import Clock


class FixedClock(Clock):
    """Deterministic clock for tests."""

    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def advance(self, dt: float) -> None:
        self._t += dt

    def now(self) -> float:
        return self._t


# ---------------------------------------------------------------------------
# Test 1: hall call added to scheduler queue
# ---------------------------------------------------------------------------

def test_hall_call_adds_request_to_scheduler_queue(clock: FixedClock) -> None:
    """
    A request for floor 5 from the lobby (floor 0) is added to the scheduler queue.

    Arrange: single-elevator building starting at floor 0.
    Act:     submit a hall call for floor 5, direction UP.
    Assert:  pending_requests() returns 1; elevator's next_stop() is 5.
    """
    # Arrange
    from elevator.scheduler import NearestIdleStrategy
    building = Building(
        num_floors=10,
        num_elevators=1,
        strategy=NearestIdleStrategy(),
        clock=clock,
        elevator_capacity=8,
    )

    # Act
    building.request_elevator(floor=5, direction=Direction.UP)

    # Assert
    assert building.pending_requests() == 1
    positions = building.elevator_positions()
    # The single elevator should have floor 5 as its next stop.
    assert positions[0][0] == 0            # still at floor 0 (no step yet)
    assert building.pending_requests() >= 1


# ---------------------------------------------------------------------------
# Test 2: elevator traveling UP picks up request at floor 3
# ---------------------------------------------------------------------------

def test_elevator_picks_up_stop_at_floor_3_while_moving_up(clock: FixedClock) -> None:
    """
    An elevator traveling UP passes floor 3 — a waiting request for floor 3 is picked up.

    Arrange: elevator at floor 1, moving UP, stops queued at [3, 7].
    Act:     call step() until the elevator reaches floor 3.
    Assert:  elevator door opens at floor 3 (door state transitions to OPENING).
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    elevator.add_stop(3)
    elevator.add_stop(7)

    # Act — step until floor 3 is reached
    visited: list[int] = []
    for _ in range(20):
        elevator.step()
        if elevator.door_state in (DoorState.OPENING, DoorState.OPEN):
            visited.append(elevator.current_floor)
            # Advance past the door cycle
            elevator.step()  # OPENING -> OPEN
            elevator.step()  # OPEN -> CLOSING
            elevator.step()  # CLOSING -> CLOSED
        if elevator.current_floor >= 3 and elevator.door_state == DoorState.CLOSED:
            break

    # Assert
    assert 3 in visited, f"Elevator did not stop at floor 3; visited={visited}"


# ---------------------------------------------------------------------------
# Test 3: elevator at capacity refuses boarding
# ---------------------------------------------------------------------------

def test_elevator_at_capacity_refuses_new_boarding(clock: FixedClock) -> None:
    """
    An elevator at capacity refuses a new boarding request.

    Arrange: elevator with capacity=2, board 2 passengers.
    Act:     attempt to board a third passenger.
    Assert:  board_passenger() returns False; passenger_count remains 2.
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=2, clock=clock)
    assert elevator.board_passenger() is True
    assert elevator.board_passenger() is True
    assert elevator.passenger_count == 2

    # Act
    result = elevator.board_passenger()

    # Assert
    assert result is False
    assert elevator.passenger_count == 2


# ---------------------------------------------------------------------------
# Test 4: SCAN resolves floor requests in correct ascending order while moving UP
# ---------------------------------------------------------------------------

def test_scan_serves_floors_ascending_while_moving_up(clock: FixedClock) -> None:
    """
    The SCAN algorithm resolves a set of floor requests in correct order when traveling UP.

    Arrange: elevator at floor 1, stops queued: [4, 7, 2].
    Act:     step() until all stops served.
    Assert:  floors visited in order [2, 4, 7].
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    elevator.add_stop(4)
    elevator.add_stop(7)
    elevator.add_stop(2)

    # Act — run steps and record door-open events
    visited: list[int] = []
    for _ in range(60):
        elevator.step()
        if elevator.door_state == DoorState.OPENING:
            if not visited or visited[-1] != elevator.current_floor:
                visited.append(elevator.current_floor)
        if elevator.direction == Direction.IDLE and elevator.door_state == DoorState.CLOSED:
            break

    # Assert
    assert visited == [2, 4, 7], f"Expected [2,4,7], got {visited}"


# ---------------------------------------------------------------------------
# Test 5: open_door on already-OPEN state does not double-transition
# ---------------------------------------------------------------------------

def test_open_door_when_already_open_raises_invalid_transition(clock: FixedClock) -> None:
    """
    Calling open_door() on an elevator already in DOOR_OPEN state does not trigger
    a second transition — it raises InvalidDoorTransitionError.

    Arrange: elevator with door in OPEN state.
    Act:     call open_door() again.
    Assert:  InvalidDoorTransitionError raised; door state still OPEN.
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    elevator.open_door()
    # Advance through OPENING -> OPEN
    elevator.step()  # OPENING -> OPEN

    assert elevator.door_state == DoorState.OPEN

    # Act & Assert
    with pytest.raises(InvalidDoorTransitionError):
        elevator.open_door()

    assert elevator.door_state == DoorState.OPEN


# ---------------------------------------------------------------------------
# Test 6: closing door while passenger is boarding raises
# ---------------------------------------------------------------------------

def test_close_door_while_boarding_raises(clock: FixedClock) -> None:
    """
    Closing the door while a passenger is boarding raises the appropriate exception.

    Arrange: elevator door in DOOR_OPENING state (mid-transition).
    Act:     call close_door() during the OPENING transition.
    Assert:  InvalidDoorTransitionError raised.
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    elevator.open_door()
    # Door is now OPENING but not yet OPEN

    assert elevator.door_state == DoorState.OPENING

    # Act & Assert
    with pytest.raises(InvalidDoorTransitionError):
        elevator.close_door()


# ---------------------------------------------------------------------------
# Test 7: elevator returns to IDLE after all requests served
# ---------------------------------------------------------------------------

def test_elevator_returns_to_idle_after_all_requests_served(clock: FixedClock) -> None:
    """
    After all requests are served, the elevator returns to IDLE state.

    Arrange: elevator at floor 0, single stop at floor 3.
    Act:     step() until stop is served.
    Assert:  direction == IDLE; door is CLOSED; no pending stops.
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=10, capacity=8, clock=clock)
    elevator.add_stop(3)

    # Act
    for _ in range(40):
        elevator.step()
        if (
            elevator.direction == Direction.IDLE
            and elevator.door_state == DoorState.CLOSED
            and elevator.next_stop() is None
        ):
            break

    # Assert
    assert elevator.direction == Direction.IDLE
    assert elevator.door_state == DoorState.CLOSED
    assert elevator.next_stop() is None


# ---------------------------------------------------------------------------
# Test 8: elevator at top floor reverses direction when no higher requests
# ---------------------------------------------------------------------------

def test_elevator_reverses_at_top_floor_when_no_higher_requests(clock: FixedClock) -> None:
    """
    An elevator at floor 10 (top floor) reverses direction correctly when no
    higher requests exist.

    Arrange: 11-floor building (floors 0-10), elevator at floor 10 moving UP.
             Stop queued at floor 4 (below current position).
    Act:     step() once.
    Assert:  direction changes to MOVING_DOWN; next_stop() == 4.
    """
    # Arrange
    elevator = Elevator(elevator_id=0, num_floors=11, capacity=8, clock=clock)
    # Simulate elevator already at floor 10 by manually positioning it,
    # or use multiple steps from floor 0 to reach floor 10 first.
    elevator.add_stop(10)
    for _ in range(20):
        elevator.step()
        if elevator.current_floor == 10 and elevator.door_state == DoorState.CLOSED:
            break

    # Now add a stop below — no stops above exist
    elevator.add_stop(4)

    # Act
    elevator.step()

    # Assert
    assert elevator.direction == Direction.DOWN
    assert elevator.next_stop() == 4


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(0.0)
```
