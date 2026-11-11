# Solutions — Week 05 · Day 4 (Logging & Debugging)

## Exercise 1 — FileHandler for WARNING+ to logs/errors.log

```python
# Add to main.py (after basicConfig)
import logging
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

file_handler = logging.FileHandler(log_dir / "errors.log")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

# Attach to root logger — captures WARNING+ from all loggers
logging.getLogger().addHandler(file_handler)
```

This does not require editing `src/ledger.py` at all. The root logger receives propagated records from all child loggers; the `FileHandler` filters to `WARNING` and above.

## Exercise 2 — Breakpoint in list comprehension

```python
# Set a breakpoint on the next line, then step into the comprehension
accounts = [Ledger(f"user_{i}", Decimal(str(i * 10))) for i in range(5)]
```

Stepping through with the VS Code debugger (or `breakpoint()` in code):

| `i` | Balance |
|-----|---------|
| 0   | 0       |
| 1   | 10      |
| 2   | 20      |
| 3   | 30      |
| 4   | 40      |

`range(5)` yields 0,1,2,3,4 so the balances are 0,10,20,30,40. Total = 100. The code is actually correct — the comment in the exercise was the misdirection. The total IS 100 and the computation IS right. The debugging exercise teaches that stepping through confirms the actual values rather than assuming.

## Exercise 3 — Suppress src.ledger below WARNING

```python
# In main.py, after basicConfig:
logging.getLogger("src.ledger").setLevel(logging.WARNING)
```

This sets the effective level on the `src.ledger` logger specifically. Records at DEBUG or INFO originating from `src/ledger.py` will not pass the level check and will not be forwarded to any handler. The root logger and all other loggers are unaffected.

## Exercise 4 — JsonFormatter

```python
import logging
import json
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# Attach to a handler
json_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(JsonFormatter())
logging.getLogger().addHandler(json_handler)
```

Verify: `python main.py | python -m json.tool` — each line should parse without error.

## Exercise 5 — Why libraries must not call basicConfig() (written answer)

`logging.basicConfig()` configures the **root logger** — it adds a `StreamHandler` and sets the root level. It is a no-op if the root logger already has handlers, but when called by a library before the application configures logging, it installs a default handler that the application cannot easily remove.

The consequence: the application loses control of log format, level, and destination. If two libraries both call `basicConfig`, only the first call has any effect, and the application developer has no way to know which library "won." The correct behaviour for a library is to call only `logging.getLogger(__name__)` and emit records — leaving all handler and formatter setup to the application's entry point.
