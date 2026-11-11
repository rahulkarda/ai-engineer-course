# Phase 7 · Week 07 · Day 3 — SOLID + Design Patterns Under Time Pressure

> Objective: Apply SOLID principles and the 5 most interview-critical design patterns quickly and correctly under time pressure, recognizing which pattern a problem needs from requirement clues.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### SOLID in one paragraph each

**S — Single Responsibility Principle (SRP)**
A class should have one reason to change. "One reason" means one actor: one stakeholder group whose requirements drive changes to that class. Violation: a `User` class that handles authentication, serialization, AND email sending — three different actors driving changes. Fix: split into `User`, `AuthService`, `UserSerializer`, `EmailNotifier`.

**O — Open/Closed Principle (OCP)**
Classes should be open for extension, closed for modification. Adding new behavior should not require editing existing code. Violation: an `if/elif` chain that grows every time a new type appears. Fix: define an interface; new types add new implementations, existing code untouched.

**L — Liskov Substitution Principle (LSP)**
Subtypes must be substitutable for their base types without breaking callers. Violation: a `Square` subclassing `Rectangle` that overrides `set_width` to also change height — breaks any code that sets width and height independently. Fix: do not use inheritance when the IS-A relationship does not hold for all operations.

**I — Interface Segregation Principle (ISP)**
Clients should not be forced to depend on methods they do not use. Violation: a fat `IWorker` interface with `work()`, `eat()`, and `sleep()` methods — robots implement `IWorker` but must stub `eat()` and `sleep()`. Fix: split into `IWorkable`, `IEatable`, `ISleepable`.

**D — Dependency Inversion Principle (DIP)**
High-level modules should not depend on low-level modules. Both should depend on abstractions. Violation: `OrderService` directly instantiates `MySQLDatabase`. Fix: `OrderService` depends on `DatabaseInterface`; the concrete `MySQLDatabase` is injected.

### Interview-critical design patterns

| Pattern | Category | One-line purpose | Clue words in requirements |
|---------|----------|------------------|---------------------------|
| Strategy | Behavioural | Swap algorithms at runtime | "different ways to", "configurable", "pluggable" |
| Observer | Behavioural | Notify dependents on state change | "notify", "subscribe", "event-driven", "listeners" |
| Factory (Method / Abstract) | Creational | Decouple creation from usage | "create based on type", "produces", "builds" |
| Singleton | Creational | Single shared instance | "global", "shared config", "connection pool" |
| Command | Behavioural | Encapsulate requests as objects | "undo/redo", "queue operations", "audit log" |

## How it works internally

### Recognizing patterns from requirement clues

Train yourself to map phrases to patterns:

```
"Users can choose different sorting algorithms" -> Strategy
"Dashboard updates when new data arrives"       -> Observer
"Create vehicles based on a type string"        -> Factory Method
"The config must be loaded exactly once"        -> Singleton
"Support undo of the last 10 actions"           -> Command
"New log destinations can be added without
 changing the logger"                           -> OCP + Strategy
"Multiple components need the same logger"      -> Singleton
```

### Time-pressure checklist (20-min problem)

```
 0:00 — Read the problem. Underline nouns, circle verbs.
 2:00 — Clarify: scale? variants? undo? notifications?
 5:00 — Write entities + relationships on whiteboard/paper.
 8:00 — Write interfaces first (no bodies). Talk through each.
12:00 — Name the patterns you're applying and why.
15:00 — Implement 2-3 key classes. Skip boilerplate setters.
20:00 — Review: any SOLID violation? Missing edge case?
```

**Never start with implementation.** Interviewers care more about the design than whether you finished coding. An unfinished implementation with a clean design beats finished code with a messy one.

### SOLID violations as code smells

| Smell | Likely violation |
|-------|-----------------|
| Method with `isinstance` checks | OCP / LSP |
| Class with >2 collaborators injected AND used | SRP |
| Method in interface you always stub with `pass` | ISP |
| `new ConcreteClass()` inside a high-level class | DIP |
| Subclass that narrows a method's postcondition | LSP |

## Worked example

**Problem: Configurable Logger System**

Requirements: "Build a logging system that writes log entries to one or more destinations (console, file, remote HTTP endpoint). New destinations should be addable without changing existing code. Only one logger instance should exist in the application. Each destination can format messages differently. Support log levels: DEBUG, INFO, WARNING, ERROR."

Pattern identification:
- "one logger instance" -> **Singleton**
- "one or more destinations, new ones addable without changing existing code" -> **Strategy** + OCP
- "each destination can format messages differently" -> **Strategy** (formatting as a nested strategy)

SOLID analysis:
- SRP: `Logger` handles dispatch, not formatting or I/O.
- OCP: adding `SlackHandler` requires no change to `Logger`.
- DIP: `Logger` depends on `LogHandler` abstraction, not concrete classes.

```python
"""
logger_system.py — LLD: Configurable Logger using Strategy + Singleton
Python 3.12+ | pydantic v2 | logging (meta: we build our own on top)
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional
import logging as stdlib_logging

stdlib_logging.basicConfig(
    level=stdlib_logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)
_meta_log = stdlib_logging.getLogger("logger_system.meta")


# ---------------------------------------------------------------------------
# Log level
# ---------------------------------------------------------------------------

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

    def label(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Log record (value object)
# ---------------------------------------------------------------------------

class LogRecord:
    __slots__ = ("level", "message", "timestamp", "context")

    def __init__(
        self,
        level: LogLevel,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        self.level = level
        self.message = message
        self.timestamp = datetime.now()
        self.context: dict[str, object] = context or {}


# ---------------------------------------------------------------------------
# Strategy: Formatter
# ---------------------------------------------------------------------------

class Formatter(ABC):
    @abstractmethod
    def format(self, record: LogRecord) -> str:
        ...


class SimpleFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        ts = record.timestamp.strftime("%H:%M:%S")
        return f"[{ts}] {record.level.label():<7} {record.message}"


class JsonFormatter(Formatter):
    """Outputs a compact JSON line."""
    def format(self, record: LogRecord) -> str:
        import json
        payload = {
            "ts": record.timestamp.isoformat(),
            "level": record.level.label(),
            "msg": record.message,
        }
        if record.context:
            payload["ctx"] = {k: str(v) for k, v in record.context.items()}
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Strategy: LogHandler (SRP + OCP — each handler is its own class)
# ---------------------------------------------------------------------------

class LogHandler(ABC):
    def __init__(self, formatter: Formatter, min_level: LogLevel = LogLevel.DEBUG) -> None:
        self._formatter = formatter
        self._min_level = min_level

    def handle(self, record: LogRecord) -> None:
        if record.level >= self._min_level:
            formatted = self._formatter.format(record)
            self._write(formatted)

    @abstractmethod
    def _write(self, formatted: str) -> None:
        ...


class ConsoleHandler(LogHandler):
    """Writes to stdout (errors to stderr)."""

    def _write(self, formatted: str) -> None:
        stream = sys.stderr if "ERROR" in formatted else sys.stdout
        print(formatted, file=stream)


class FileHandler(LogHandler):
    """Appends to a file. DIP: path is injected, not hardcoded."""

    def __init__(
        self,
        path: Path,
        formatter: Formatter,
        min_level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        super().__init__(formatter, min_level)
        self._path = path

    def _write(self, formatted: str) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(formatted + "\n")


class InMemoryHandler(LogHandler):
    """Stores records in a list — useful for testing."""

    def __init__(
        self,
        formatter: Formatter,
        min_level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        super().__init__(formatter, min_level)
        self._records: list[str] = []

    def _write(self, formatted: str) -> None:
        self._records.append(formatted)

    def get_records(self) -> list[str]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Singleton: Logger
# ---------------------------------------------------------------------------

class Logger:
    """
    Singleton logger that dispatches to multiple handlers.

    Thread note: not thread-safe for concurrent _instance creation.
    In production, use a threading.Lock() around get_instance().
    """
    _instance: Optional[Logger] = None

    def __init__(self) -> None:
        self._handlers: list[LogHandler] = []
        self._min_level = LogLevel.DEBUG
        _meta_log.debug("Logger singleton created")

    @classmethod
    def get_instance(cls) -> Logger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Testing helper."""
        cls._instance = None

    def add_handler(self, handler: LogHandler) -> Logger:
        """Fluent interface for setup."""
        self._handlers.append(handler)
        return self

    def set_min_level(self, level: LogLevel) -> Logger:
        self._min_level = level
        return self

    def _dispatch(self, level: LogLevel, message: str, context: dict | None) -> None:
        if level < self._min_level:
            return
        record = LogRecord(level=level, message=message, context=context)
        for handler in self._handlers:
            try:
                handler.handle(record)
            except Exception as exc:
                _meta_log.error("Handler %r failed: %s", handler, exc)

    def debug(self, msg: str, **ctx: object) -> None:
        self._dispatch(LogLevel.DEBUG, msg, ctx or None)

    def info(self, msg: str, **ctx: object) -> None:
        self._dispatch(LogLevel.INFO, msg, ctx or None)

    def warning(self, msg: str, **ctx: object) -> None:
        self._dispatch(LogLevel.WARNING, msg, ctx or None)

    def error(self, msg: str, **ctx: object) -> None:
        self._dispatch(LogLevel.ERROR, msg, ctx or None)


# ---------------------------------------------------------------------------
# SOLID violation demonstrations (commented out — for study)
# ---------------------------------------------------------------------------

# SRP VIOLATION (don't do this):
# class GodLogger:
#     def log(self, msg): ...         # logging
#     def format_json(self, r): ...   # formatting
#     def write_file(self, path): ... # I/O
#     def send_http(self, url): ...   # networking
#     # 4 reasons to change = SRP violation

# OCP VIOLATION (don't do this):
# def dispatch(record, dest_type):
#     if dest_type == "console":      # adding "slack" requires editing here
#         print(record.message)
#     elif dest_type == "file":
#         write_file(record)
#     # Every new destination -> modify this function

# DIP VIOLATION (don't do this):
# class Logger:
#     def __init__(self):
#         self._file_handler = FileHandler(Path("app.log"), SimpleFormatter())
#         # Logger creates its own concrete handler -> tightly coupled


# ---------------------------------------------------------------------------
# Observer pattern bonus: log event subscribers
# ---------------------------------------------------------------------------

class LogObserver(ABC):
    @abstractmethod
    def on_log(self, record: LogRecord) -> None:
        ...


class AlertObserver(LogObserver):
    """Example: trigger alerts on ERROR logs."""

    def on_log(self, record: LogRecord) -> None:
        if record.level >= LogLevel.ERROR:
            # In production: send PagerDuty/Slack alert
            _meta_log.warning("ALERT triggered: %s", record.message)


class ObservableLogger(Logger):
    """
    Extends Logger with Observer pattern.
    Demonstrates how patterns compose.
    """
    def __init__(self) -> None:
        super().__init__()
        self._observers: list[LogObserver] = []

    def subscribe(self, observer: LogObserver) -> None:
        self._observers.append(observer)

    def _dispatch(self, level: LogLevel, message: str, context: dict | None) -> None:
        super()._dispatch(level, message, context)
        if level >= self._min_level:
            record = LogRecord(level=level, message=message, context=context)
            for obs in self._observers:
                obs.on_log(record)


# ---------------------------------------------------------------------------
# Command pattern: buffered/deferred log commands
# ---------------------------------------------------------------------------

class LogCommand(ABC):
    """Each log call is an object — enables queuing, undo, replay."""

    @abstractmethod
    def execute(self, logger: Logger) -> None:
        ...


class LogMessageCommand(LogCommand):
    def __init__(self, level: LogLevel, message: str) -> None:
        self.level = level
        self.message = message

    def execute(self, logger: Logger) -> None:
        method = {
            LogLevel.DEBUG: logger.debug,
            LogLevel.INFO: logger.info,
            LogLevel.WARNING: logger.warning,
            LogLevel.ERROR: logger.error,
        }[self.level]
        method(self.message)


class LogBuffer:
    """Collects commands, flushes all at once (e.g., on request end)."""

    def __init__(self) -> None:
        self._queue: list[LogCommand] = []

    def enqueue(self, cmd: LogCommand) -> None:
        self._queue.append(cmd)

    def flush(self, logger: Logger) -> None:
        while self._queue:
            self._queue.pop(0).execute(logger)


# ---------------------------------------------------------------------------
# Factory: handler creation from config
# ---------------------------------------------------------------------------

class HandlerFactory:
    """
    Creates handlers from a config dict.
    OCP: adding a new type requires adding a branch here once,
    not changing every caller.
    """

    @staticmethod
    def create(config: dict[str, object]) -> LogHandler:
        fmt_name = config.get("formatter", "simple")
        formatter: Formatter = (
            JsonFormatter() if fmt_name == "json" else SimpleFormatter()
        )
        min_level = LogLevel[str(config.get("min_level", "DEBUG")).upper()]
        handler_type = config.get("type")

        match handler_type:
            case "console":
                return ConsoleHandler(formatter=formatter, min_level=min_level)
            case "file":
                path = Path(str(config["path"]))
                return FileHandler(path=path, formatter=formatter, min_level=min_level)
            case "memory":
                return InMemoryHandler(formatter=formatter, min_level=min_level)
            case _:
                raise ValueError(f"Unknown handler type: {handler_type!r}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Reset singleton for clean demo
    Logger.reset()

    memory_handler = InMemoryHandler(formatter=SimpleFormatter(), min_level=LogLevel.DEBUG)

    logger = Logger.get_instance()
    logger.add_handler(ConsoleHandler(formatter=SimpleFormatter()))
    logger.add_handler(memory_handler)
    logger.set_min_level(LogLevel.DEBUG)

    logger.debug("System starting", component="main")
    logger.info("User logged in", user_id="u42")
    logger.warning("Disk usage high", percent=85)
    logger.error("Database timeout", host="db-1", retry=3)

    _meta_log.info("--- Captured by InMemoryHandler ---")
    for line in memory_handler.get_records():
        _meta_log.info("  %s", line)

    # Factory demo
    _meta_log.info("--- Factory-created handler ---")
    handler = HandlerFactory.create({
        "type": "memory",
        "formatter": "json",
        "min_level": "WARNING",
    })
    Logger.reset()
    logger2 = Logger.get_instance()
    logger2.add_handler(handler)
    logger2.info("This is below WARNING — should not appear")
    logger2.warning("This should appear in JSON")

    assert isinstance(handler, InMemoryHandler)
    _meta_log.info("JSON records: %s", handler.get_records())

    # Command buffer demo
    _meta_log.info("--- Command buffer ---")
    buffer = LogBuffer()
    buffer.enqueue(LogMessageCommand(LogLevel.INFO, "Buffered info"))
    buffer.enqueue(LogMessageCommand(LogLevel.ERROR, "Buffered error"))
    _meta_log.info("Nothing logged yet. Flushing...")
    buffer.flush(logger2)


if __name__ == "__main__":
    main()
```

## Quick-reference: SOLID violation checklist

```
Symptom                                       | Principle violated
----------------------------------------------|-------------------
Class has >1 unrelated reasons to change      | SRP
if/elif grows with each new type added        | OCP
Subclass breaks caller that uses base class   | LSP
Interface method always stubbed as pass/...   | ISP
High-level class does `new ConcreteClass()`   | DIP
```

## Exercises

1. **Identify SOLID violations** — Find and name every SOLID violation in the following code. For each violation, write the fix (one paragraph + corrected code snippet):

```python
class ReportService:
    def generate(self, data: list[dict]) -> str:
        # Format
        lines = [f"{d['name']}: {d['value']}" for d in data]
        report = "\n".join(lines)
        # Save
        with open("report.txt", "w") as f:
            f.write(report)
        # Email
        import smtplib
        smtp = smtplib.SMTP("mail.example.com")
        smtp.sendmail("from@x.com", "to@x.com", report)
        return report

class PDFReport(ReportService):
    def generate(self, data: list[dict]) -> str:
        # Generates PDF but never saves or emails
        return "<pdf bytes>"
```

2. **Pattern identification drill** — For each requirement snippet, name the pattern and justify in one sentence:
   - "The checkout process can use PayPal, Stripe, or bank transfer interchangeably."
   - "When an order status changes, the warehouse, customer, and analytics system all need to know."
   - "The application has one database connection pool shared across all services."
   - "Users can undo up to 10 text edits."
   - "The system creates notification objects (email, SMS, push) based on user preferences."

3. **Time-pressure design** — Set a 25-minute timer. Design a ride-sharing matching system (drivers, riders, rides, pricing). Requirements: multiple pricing strategies (surge, flat, per-km), a driver-rider matching algorithm that can be swapped, notify driver and rider when a match is made, log all ride events. Apply: Strategy (pricing + matching), Observer (notifications), Singleton (ride log). Write interfaces first (10 min), then implement `RideMatchingService` (15 min).

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can state each SOLID principle in one sentence without looking
- [ ] I can identify which principle is violated from a code symptom
- [ ] I can name 5 design patterns and their intent from memory
- [ ] I can map requirement phrases to patterns (see the table above)
- [ ] I can implement Strategy, Observer, Singleton, and Factory without reference
- [ ] I can compose multiple patterns in a single design (as in the logger example)
- [ ] I can complete a clean interface specification in 10 minutes under time pressure

## Supplementary

- [Refactoring Guru — SOLID](https://refactoring.guru/solid-principles)
- [Refactoring Guru — Design Patterns](https://refactoring.guru/design-patterns)
- [Python Design Patterns — Brandon Rhodes](https://python-patterns.guide/)
- [Head First Design Patterns (book) — Freeman & Robson](https://www.oreilly.com/library/view/head-first-design/0596007124/)
- [OOP Design Interview Patterns — Educative](https://www.educative.io/courses/grokking-the-object-oriented-design-interview)
