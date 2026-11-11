# Phase 1 · Week 05 · Day 4 — Logging & Debugging

> Objective: Use Python's `logging` module properly, set up the VS Code debugger, read tracebacks, and apply a systematic strategy for debugging unfamiliar code.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-3]] (testing), [[week-01/day-3]] (functions)

## Concept

### Why `logging`, not `print`

`print` has no levels, no destinations, no timestamps, and no on/off switch. The `logging` module gives you:
- **Levels**: DEBUG < INFO < WARNING < ERROR < CRITICAL — filter noise by environment.
- **Handlers**: write to console, file, syslog, HTTP endpoint simultaneously.
- **Formatters**: structured JSON or human-readable with timestamps and module names.
- **Logger hierarchy**: loggers inherit from parent loggers, so configuration cascades.

Rule: use `logging` from the first line of any production or long-lived code.

### Logger hierarchy

Every logger has a name, typically the module: `logging.getLogger(__name__)`. Names are dotted paths forming a tree:

```
root
├── src
│   ├── src.ledger
│   └── src.notifications
└── tests
    └── tests.test_ledger
```

A log record propagates up the tree until a handler processes it. Configure the root logger or a specific ancestor to control all descendants.

### Log levels

```python
import logging
log = logging.getLogger(__name__)

log.debug("state=%s", state)          # verbose diagnostic; off in prod
log.info("deposit completed: %s", amount)   # normal operation
log.warning("retry attempt %d/3", n)  # unexpected but recoverable
log.error("payment failed: %s", exc)  # operation failed
log.critical("database unreachable")  # system cannot function
```

Use `%s` formatting in log calls — the message is only formatted if the level is enabled (avoids building the string needlessly).

### Basic configuration

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

For applications, set up once at entry point (`main.py`). Libraries must **never** call `basicConfig` — they should only create loggers and leave handler setup to the application.

### Structured logging (JSON)

JSON logs are machine-parseable by log aggregators (Datadog, Splunk, CloudWatch):

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)
```

### Reading tracebacks

```
Traceback (most recent call last):
  File "src/ledger.py", line 28, in withdraw     ← where the error propagated from
    raise InsufficientFundsError(...)
  File "src/service.py", line 14, in process_payment  ← call above
    ledger.withdraw(amount)
src.ledger.InsufficientFundsError: cannot withdraw 500: balance is 100
```

Read from **bottom to top**: the bottom line is the actual error; each frame above shows the call chain. The culprit is usually the innermost frame in *your* code (not a library frame).

## How it works internally

### Logger propagation

When you call `log.info("msg")`:
1. The logger checks if `INFO >= effective_level`. If not, returns immediately.
2. A `LogRecord` object is created with all metadata.
3. Each handler attached to this logger processes the record.
4. If `propagate=True` (default), the record is passed to the parent logger, which repeats steps 3–4.

Setting `propagate=False` on a library logger stops records from reaching the root — useful when a library's logs are noisy.

### VS Code debugger internals

VS Code's Python debugger uses **debugpy**, which instruments Python bytecode via `sys.settrace()`. This hook fires on every line execution, function call, and exception. Breakpoints inject a `Breakpoint` instruction at the target line; when the interpreter hits it, debugpy suspends the frame and hands control to VS Code's debug adapter protocol (DAP) client.

This means debugging has overhead — the `settrace` callback is called thousands of times per second. It's fine for debugging but not for production profiling.

## Worked example

```python
# src/ledger.py — with proper logging
"""Ledger with structured logging on all state changes."""
from __future__ import annotations
import logging
from decimal import Decimal

log = logging.getLogger(__name__)


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the current balance."""


class Ledger:
    def __init__(self, owner: str, initial_balance: Decimal = Decimal("0")) -> None:
        if initial_balance < 0:
            raise ValueError("initial_balance must be non-negative")
        self.owner = owner
        self._balance = initial_balance
        log.debug("Ledger created: owner=%s initial_balance=%s", owner, initial_balance)

    @property
    def balance(self) -> Decimal:
        return self._balance

    def deposit(self, amount: Decimal) -> None:
        log.debug("deposit attempt: owner=%s amount=%s", self.owner, amount)
        if amount <= 0:
            log.warning("rejected deposit: amount=%s is not positive", amount)
            raise ValueError(f"deposit amount must be positive, got {amount}")
        self._balance += amount
        log.info("deposit: owner=%s amount=%s new_balance=%s", self.owner, amount, self._balance)

    def withdraw(self, amount: Decimal) -> None:
        log.debug("withdraw attempt: owner=%s amount=%s", self.owner, amount)
        if amount <= 0:
            log.warning("rejected withdrawal: amount=%s is not positive", amount)
            raise ValueError(f"withdrawal amount must be positive, got {amount}")
        if amount > self._balance:
            log.error(
                "insufficient funds: owner=%s requested=%s balance=%s",
                self.owner, amount, self._balance,
            )
            raise InsufficientFundsError(
                f"cannot withdraw {amount}: balance is {self._balance}"
            )
        self._balance -= amount
        log.info("withdrawal: owner=%s amount=%s new_balance=%s", self.owner, amount, self._balance)

    def transfer_to(self, other: "Ledger", amount: Decimal) -> None:
        log.info("transfer start: from=%s to=%s amount=%s", self.owner, other.owner, amount)
        self.withdraw(amount)
        other.deposit(amount)
        log.info("transfer complete: from=%s to=%s amount=%s", self.owner, other.owner, amount)
```

```python
# main.py — configure logging at entry point
"""Application entry point — sets up logging before any other imports."""
from __future__ import annotations
import logging
import os
from decimal import Decimal

# Read level from env — INFO in prod, DEBUG in dev
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.ledger import Ledger  # noqa: E402  (import after logging setup)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("application starting")
    alice = Ledger("alice", Decimal("100"))
    try:
        alice.withdraw(Decimal("200"))
    except Exception as exc:
        log.error("transaction failed: %s", exc, exc_info=True)
    log.info("application done")
```

### VS Code launch configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug main.py",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "env": {"LOG_LEVEL": "DEBUG"},
      "justMyCode": true
    },
    {
      "name": "Debug pytest",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v", "--no-header"],
      "justMyCode": true
    }
  ]
}
```

### Debugging strategy for unfamiliar code

1. **Read the traceback bottom-up.** Find the exact file/line.
2. **Reproduce the failure** with the smallest possible input.
3. **Set a breakpoint** one frame above the error. Inspect local variables.
4. **Walk call stack upward** — where did the bad value originate?
5. **Add a targeted log line** at the suspect location if you need to trace across time.
6. **Form a hypothesis, test it** — change one thing, re-run. Avoid shotgun changes.
7. **Write a regression test** once you find the root cause.

## Exercises

1. Add a `FileHandler` to the ledger app that writes all `WARNING` and above to `logs/errors.log`. Do this without modifying `src/ledger.py`. Use `pathlib.Path` to create the `logs/` directory if it doesn't exist.

2. The following code has a bug that only appears at runtime. Set a breakpoint in VS Code and step through it to find the issue:
   ```python
   from decimal import Decimal
   from src.ledger import Ledger

   accounts = [Ledger(f"user_{i}", Decimal(str(i * 10))) for i in range(5)]
   total = sum(a.balance for a in accounts)
   # Expected: 0+10+20+30+40 = 100 but we get the wrong answer
   # (Intentionally off-by-one: range(5) gives 0..4, check your arithmetic)
   print(f"Total: {total}")
   ```
   Set a breakpoint inside the list comprehension. What is each account's balance? Is the total correct?

3. A logger named `src.ledger` logs at DEBUG by default. Add code to `main.py` (without editing `ledger.py`) that raises `src.ledger`'s level to `WARNING` — suppressing all debug/info output from the ledger module while leaving the root logger unchanged.

4. Write a `JsonFormatter` (as shown in the Concept section) and attach it to a `StreamHandler`. Run the app and verify the output is valid JSON (pipe to `python -m json.tool`).

5. Explain: why should library code never call `logging.basicConfig()`? What happens if it does?

(Solutions: ../solutions/week-05/day-4.md)

## Self-check
- [ ] I can create a module-level logger with `logging.getLogger(__name__)`.
- [ ] I know the five log levels and which to use for each scenario.
- [ ] I can configure logging at the application entry point and respect library conventions.
- [ ] I can read a Python traceback and identify the root cause frame.
- [ ] I can set a breakpoint in VS Code, step over/into calls, and inspect locals.
- [ ] I can add a `FileHandler` with a custom level independently of the root logger.

## Supplementary links
- [Python docs — logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Python docs — logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [VS Code — Python debugging](https://code.visualstudio.com/docs/python/debugging)
- [Real Python — Logging in Python](https://realpython.com/python-logging/)
