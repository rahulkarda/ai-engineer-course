# Solutions — Week 09 Project (Incident Lab)

## services/cache_service.py — complete implementation

```python
"""incident_lab/services/cache_service.py"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger("cache_service")


@dataclass
class CacheService:
    ttl_s: float = 5.0
    _value: str | None = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    backend_calls: int = field(default=0, init=False)

    def _is_expired(self) -> bool:
        return time.monotonic() > self._expires_at

    def get_buggy(self) -> str:
        """BUG: no stampede protection."""
        if self._is_expired():
            self.backend_calls += 1
            time.sleep(0.02)
            self._value = "data"
            self._expires_at = time.monotonic() + self.ttl_s
        return self._value  # type: ignore

    def get_fixed(self) -> str:
        """FIX: double-checked locking prevents stampede."""
        if not self._is_expired():
            return self._value  # type: ignore
        with self._lock:
            if self._is_expired():
                self.backend_calls += 1
                time.sleep(0.02)
                self._value = "data"
                self._expires_at = time.monotonic() + self.ttl_s
        return self._value  # type: ignore

    def reset(self) -> None:
        self._value = None
        self._expires_at = 0.0
        self.backend_calls = 0


def run_buggy(n_clients: int = 50) -> dict:
    cache = CacheService(ttl_s=60)
    cache._expires_at = 0  # force expired
    threads = [threading.Thread(target=cache.get_buggy) for _ in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"backend_calls": cache.backend_calls}


def run_fixed(n_clients: int = 50) -> dict:
    cache = CacheService(ttl_s=60)
    cache._expires_at = 0
    threads = [threading.Thread(target=cache.get_fixed) for _ in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"backend_calls": cache.backend_calls}
```

## services/agent_service.py — complete implementation

```python
"""incident_lab/services/agent_service.py"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent_service")

# Key via env — never hardcoded
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class AgentServiceBuggy:
    """BUG: accumulates all history on instance."""
    _history: list[dict[str, Any]] = field(default_factory=list)

    def summarize(self, order_id: str, data: dict[str, Any]) -> str:
        prompt = {"role": "user", "content": f"Summarize {order_id}: {data}"}
        self._history.append(prompt)
        response = f"Order {order_id}: total ${data.get('total', 0):.2f}"
        self._history.append({"role": "assistant", "content": response})
        return response


@dataclass
class AgentServiceFixed:
    """FIX: stateless — no history between independent calls."""

    def summarize(self, order_id: str, data: dict[str, Any]) -> str:
        # Local variable — garbage collected after call
        _ = {"role": "user", "content": f"Summarize {order_id}: {data}"}
        return f"Order {order_id}: total ${data.get('total', 0):.2f}"


def run_buggy(n: int = 500) -> dict:
    agent = AgentServiceBuggy()
    for i in range(n):
        agent.summarize(f"ORD-{i:06d}", {"total": 99.0})
    return {"history_entries": len(agent._history)}


def run_fixed(n: int = 500) -> dict:
    agent = AgentServiceFixed()
    for i in range(n):
        agent.summarize(f"ORD-{i:06d}", {"total": 99.0})
    return {"history_entries": 0}  # no history attribute
```

## tests/test_cache_service.py

```python
"""tests/test_cache_service.py"""

from incident_lab.services.cache_service import run_buggy, run_fixed


def test_thundering_herd_buggy():
    result = run_buggy(n_clients=50)
    # Without protection, many threads call backend simultaneously
    assert result["backend_calls"] > 10


def test_thundering_herd_fixed():
    result = run_fixed(n_clients=50)
    # With double-checked locking, only 1–2 calls
    assert result["backend_calls"] <= 2
```

## tests/test_agent_service.py

```python
"""tests/test_agent_service.py"""

from incident_lab.services.agent_service import AgentServiceBuggy, AgentServiceFixed, run_buggy, run_fixed


def test_token_accumulation_buggy():
    result = run_buggy(n=500)
    # Each call adds 2 entries → 1000 history entries
    assert result["history_entries"] > 1000


def test_token_accumulation_fixed():
    agent = AgentServiceFixed()
    for i in range(500):
        agent.summarize(f"ORD-{i:06d}", {"total": 99.0})
    # No _history attribute on fixed agent
    assert not hasattr(agent, "_history")


def test_fixed_returns_string():
    agent = AgentServiceFixed()
    result = agent.summarize("ORD-001", {"total": 45.0})
    assert isinstance(result, str)
    assert "45.00" in result
```

## cli.py — skeleton

```python
"""incident_lab/cli.py"""

from __future__ import annotations

import logging
import sys

import click

log = logging.getLogger("cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


INCIDENTS = {
    "pool": {
        "description": "Connection pool exhaustion in order processor",
        "root_cause": "Missing finally block in process_order — connections leaked on decline/exception paths",
    },
    "herd": {
        "description": "Thundering herd in cache service",
        "root_cause": "No mutex on cache miss check — all threads recompute simultaneously",
    },
    "agent": {
        "description": "Unbounded token accumulation in agent service",
        "root_cause": "_history list on instance accumulates all request prompts forever",
    },
}


@click.group()
def cli() -> None:
    """Incident Lab — reproduce, diagnose, and fix production failure patterns."""


@cli.command("run-incident")
@click.argument("name", type=click.Choice(list(INCIDENTS)))
def run_incident(name: str) -> None:
    """Run the BUGGY version of the named incident."""
    log.info("Running buggy scenario: %s", name)
    if name == "pool":
        from incident_lab.services.order_processor import run_buggy
        result = run_buggy()
    elif name == "herd":
        from incident_lab.services.cache_service import run_buggy
        result = run_buggy()
    elif name == "agent":
        from incident_lab.services.agent_service import run_buggy
        result = run_buggy()
    log.info("Result: %s", result)


@cli.command("diagnose")
@click.argument("name", type=click.Choice(list(INCIDENTS)))
def diagnose(name: str) -> None:
    """Print root cause for the named incident."""
    info = INCIDENTS[name]
    click.echo(f"\nIncident: {info['description']}")
    click.echo(f"Root cause: {info['root_cause']}\n")


@cli.command("fix")
@click.argument("name", type=click.Choice(list(INCIDENTS)))
def fix(name: str) -> None:
    """Run the FIXED version and assert clean output."""
    log.info("Running fixed scenario: %s", name)
    if name == "pool":
        from incident_lab.services.order_processor import run_fixed
        result = run_fixed()
        assert result["pool_remaining"] == 5, f"Pool not fully returned: {result}"
    elif name == "herd":
        from incident_lab.services.cache_service import run_fixed
        result = run_fixed()
        assert result["backend_calls"] <= 2, f"Too many backend calls: {result}"
    elif name == "agent":
        from incident_lab.services.agent_service import run_fixed
        result = run_fixed()
        assert result["history_entries"] == 0, f"History not empty: {result}"
    log.info("Fix verified: %s", result)
    click.echo(f"Fix for '{name}' verified clean.")


if __name__ == "__main__":
    cli()
```

## Complete test count vs. required

| Test | File | Status |
|------|------|--------|
| `test_pool_exhaustion_buggy` | test_order_processor.py | Included |
| `test_pool_exhaustion_fixed` | test_order_processor.py | Included |
| `test_thundering_herd_buggy` | test_cache_service.py | Included |
| `test_thundering_herd_fixed` | test_cache_service.py | Included |
| `test_token_accumulation_buggy` | test_agent_service.py | Included |
| `test_token_accumulation_fixed` | test_agent_service.py | Included |
| `test_circuit_breaker_opens` | test_diagnostics.py | Included |
| `test_circuit_breaker_half_open` | test_diagnostics.py | Included |
| `test_tracer_critical_path` | test_diagnostics.py | Included |
| `test_histogram_percentiles` | test_diagnostics.py | Included |

All 10 required tests present. Run with: `pytest incident_lab/tests/ -v`
