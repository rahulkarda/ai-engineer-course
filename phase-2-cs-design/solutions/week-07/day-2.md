# Solutions — Week 07 · Day 2

## Exercise 1 — Add Bus to the registry

```python
from day2_creational import Vehicle, VehicleType, _VEHICLE_REGISTRY


class Bus(Vehicle):
    def describe(self) -> str:
        return "City transit bus (12 m)"


# Extend the registry — no modification to VehicleType enum or create_vehicle body
_VEHICLE_REGISTRY["bus"] = Bus  # type: ignore[assignment]

# Or add BUS = "bus" to VehicleType enum and register:
# _VEHICLE_REGISTRY[VehicleType.BUS] = Bus

bus = _VEHICLE_REGISTRY["bus"]()
print(bus.describe())  # City transit bus (12 m)
```

---

## Exercise 2 — TVUIFactory

```python
from day2_creational import Button, Dropdown, UIFactory, render_form
import logging

log = logging.getLogger(__name__)


class TVButton(Button):
    def render(self) -> str:
        return "[D-PAD SELECT]"


class TVDropdown(Dropdown):
    def render(self) -> str:
        return "[SCROLL LIST]"


class TVUIFactory(UIFactory):
    def create_button(self) -> Button:
        return TVButton()

    def create_dropdown(self) -> Dropdown:
        return TVDropdown()


render_form(TVUIFactory())  # works; render_form() not modified
```

---

## Exercise 3 — Builder validation

```python
from day2_creational import HttpRequestBuilder, HttpRequest


class ValidatingHttpRequestBuilder(HttpRequestBuilder):
    def build(self) -> HttpRequest:
        if self._method.upper() == "POST" and self._body is None:
            raise ValueError("POST requests must include a body")
        return super().build()


# Usage
import pytest

def test_post_without_body_raises():
    with pytest.raises(ValueError, match="body"):
        ValidatingHttpRequestBuilder("POST", "https://api.example.com/").build()

def test_post_with_body_ok():
    req = (
        ValidatingHttpRequestBuilder("POST", "https://api.example.com/")
        .body('{"key": "value"}')
        .build()
    )
    assert req.body is not None
```

---

## Exercise 4 — Singleton reset for tests

```python
from day2_creational import AppConfig


class TestableAppConfig(AppConfig):
    @classmethod
    def reset(cls) -> None:
        """Reset singleton for test isolation. Never call in production."""
        cls._instance = None


# In tests:
import pytest

@pytest.fixture(autouse=True)
def reset_config():
    yield
    TestableAppConfig.reset()


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    TestableAppConfig.reset()
    cfg = TestableAppConfig()
    assert cfg.db_url == "postgresql://localhost/test"
```

Why needed: Without reset, the first test that creates `AppConfig` fixes its state for all subsequent tests, causing order-dependent failures.

---

## Exercise 5 — Module singleton

```python
# config_module.py
import os
from typing import Any

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = {
            "db_url": os.getenv("DATABASE_URL", "sqlite:///:memory:"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        }
    return _config
```

**Trade-offs vs. class-based singleton:**

| | Module singleton | Class-based singleton |
|---|---|---|
| Simplicity | Simpler — no `__new__` override | More boilerplate |
| Thread safety | Safe for simple attribute reads (GIL) | Needs explicit lock |
| Test reset | Replace `_config = None` easily | Needs `reset()` classmethod |
| Subclassing | Cannot subclass a module | Singleton can be subclassed (carefully) |
| Typing | Just `dict` or a typed dataclass | Full type with IDE support |

**Prefer module singleton** for stateless config. **Prefer class-based** when you need subclassing, lifecycle hooks, or a typed interface.
