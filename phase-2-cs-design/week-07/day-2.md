# Phase 2 · Week 07 · Day 2 — Creational Patterns: Factory, Builder & Singleton

> Objective: Implement Factory Method, Abstract Factory, Builder, and Singleton; understand when each is appropriate and their Python-specific trade-offs.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

Creational patterns deal with **how objects are constructed** — separating instantiation logic from the rest of the code.

### Factory Method

Defines an interface for creating an object but lets subclasses decide which class to instantiate. The creator knows *that* an object must be created, not *what* concrete class.

**Use when:** A class cannot anticipate the type of objects it creates, or subclasses should control creation.

### Abstract Factory

Produces families of related objects without specifying their concrete classes. Think "themes" — a UI factory produces buttons *and* dropdowns, all in the same visual family.

**Use when:** A system must be independent of how its products are created, composed, and represented.

### Builder

Separates the construction of a complex object from its representation, so the same construction process can create different representations.

**Use when:** Object construction requires many optional parameters, or construction is multi-step (and partial objects should not be exposed).

### Singleton

Ensures a class has only one instance and provides a global access point.

**Use when:** Exactly one object is needed — logger, config, connection pool. In Python, the module itself is often a better singleton than the class pattern.

---

## How it works internally

**Factory in Python**: Because functions are first-class, a factory can be as simple as `dict[str, type]` + a lookup. The pattern scales to a method on a creator class when you need polymorphic behaviour.

**Builder + `__init__` vs dataclass**: Python's `@dataclass(kw_only=True)` partially replaces Builder for simple objects. Builder shines for objects with mandatory validation *between* fields, or when construction spans multiple calls.

**Singleton and thread safety**: Python's GIL protects simple attribute assignment, but double-checked locking (`threading.Lock`) is still needed if the constructor itself is slow (e.g., opening a DB connection).

**Module-level singleton**: Any `import module` returns the same object; module-level variables are initialized once. This is the Pythonic singleton for stateless config — no class pattern needed.

---

## Worked example

```python
"""
day2_creational.py
Factory Method, Abstract Factory, Builder, Singleton — vehicle domain.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Factory Method ──────────────────────────────────────────────────────────

class VehicleType(StrEnum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"


class Vehicle(ABC):
    @abstractmethod
    def describe(self) -> str: ...


class Car(Vehicle):
    def describe(self) -> str:
        return "4-wheel passenger car"


class Truck(Vehicle):
    def describe(self) -> str:
        return "Heavy-duty cargo truck"


class Motorcycle(Vehicle):
    def describe(self) -> str:
        return "2-wheel motorcycle"


# Factory: a registry-based pattern (more Pythonic than subclassing)
_VEHICLE_REGISTRY: dict[VehicleType, type[Vehicle]] = {
    VehicleType.CAR: Car,
    VehicleType.TRUCK: Truck,
    VehicleType.MOTORCYCLE: Motorcycle,
}


def create_vehicle(vehicle_type: VehicleType) -> Vehicle:
    """Factory function — hides concrete classes from callers."""
    cls = _VEHICLE_REGISTRY.get(vehicle_type)
    if cls is None:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    return cls()


# ─── Abstract Factory ────────────────────────────────────────────────────────
# Two "themes" of UI widgets: Web and Mobile.

class Button(ABC):
    @abstractmethod
    def render(self) -> str: ...


class Dropdown(ABC):
    @abstractmethod
    def render(self) -> str: ...


class WebButton(Button):
    def render(self) -> str:
        return "<button>Click</button>"


class WebDropdown(Dropdown):
    def render(self) -> str:
        return "<select>...</select>"


class MobileButton(Button):
    def render(self) -> str:
        return "[TAP_BUTTON]"


class MobileDropdown(Dropdown):
    def render(self) -> str:
        return "[SWIPE_PICKER]"


class UIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button: ...
    @abstractmethod
    def create_dropdown(self) -> Dropdown: ...


class WebUIFactory(UIFactory):
    def create_button(self) -> Button:
        return WebButton()

    def create_dropdown(self) -> Dropdown:
        return WebDropdown()


class MobileUIFactory(UIFactory):
    def create_button(self) -> Button:
        return MobileButton()

    def create_dropdown(self) -> Dropdown:
        return MobileDropdown()


def render_form(factory: UIFactory) -> None:
    btn = factory.create_button()
    dd = factory.create_dropdown()
    log.info("Form: %s  %s", btn.render(), dd.render())


# ─── Builder ─────────────────────────────────────────────────────────────────

@dataclass
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    timeout: int = 30


class HttpRequestBuilder:
    """Fluent builder for HttpRequest."""

    def __init__(self, method: str, url: str) -> None:
        self._method = method
        self._url = url
        self._headers: dict[str, str] = {}
        self._body: str | None = None
        self._timeout: int = 30

    def header(self, key: str, value: str) -> "HttpRequestBuilder":
        self._headers[key] = value
        return self

    def body(self, payload: str) -> "HttpRequestBuilder":
        self._body = payload
        return self

    def timeout(self, seconds: int) -> "HttpRequestBuilder":
        if seconds <= 0:
            raise ValueError("Timeout must be positive")
        self._timeout = seconds
        return self

    def build(self) -> HttpRequest:
        return HttpRequest(
            method=self._method,
            url=self._url,
            headers=self._headers,
            body=self._body,
            timeout=self._timeout,
        )


# ─── Singleton ───────────────────────────────────────────────────────────────

class AppConfig:
    """Thread-safe singleton configuration store."""

    _instance: "AppConfig | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[attr-defined]
            return
        import os
        self.db_url: str = os.getenv("DATABASE_URL", "sqlite:///:memory:")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self._initialized = True
        log.info("AppConfig initialised: db=%s debug=%s", self.db_url, self.debug)


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Factory Method
    for vt in VehicleType:
        v = create_vehicle(vt)
        log.info(v.describe())

    # Abstract Factory
    render_form(WebUIFactory())
    render_form(MobileUIFactory())

    # Builder
    req = (
        HttpRequestBuilder("POST", "https://api.example.com/orders")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer TOKEN")
        .body('{"item": "widget"}')
        .timeout(10)
        .build()
    )
    log.info("Request: %s %s headers=%s", req.method, req.url, req.headers)

    # Singleton
    cfg1 = AppConfig()
    cfg2 = AppConfig()
    assert cfg1 is cfg2, "Must be same instance"
    log.info("Singleton confirmed: %s", cfg1.db_url)
```

---

## Exercises

1. **Factory extension.** Add a `Bus` vehicle type to the registry-based factory without touching any existing class or the `create_vehicle` function body.

2. **Abstract Factory — add a theme.** Create a `TVUIFactory` that produces TV-appropriate widgets. Show that `render_form(TVUIFactory())` works without modifying `render_form`.

3. **Builder validation.** Extend `HttpRequestBuilder` so `build()` raises `ValueError` if `method` is `"POST"` and `body` is `None`.

4. **Singleton pitfall.** In tests, singletons can leak state between test cases. Write a `reset()` classmethod on `AppConfig` that resets `_instance` to `None`, and explain why you'd need it.

5. **Module singleton.** Refactor `AppConfig` to a module-level approach (plain module with `_config: dict` populated on first import) and discuss trade-offs vs. the class-based singleton.

(Solutions: ../solutions/week-07/day-2.md)

---

## Self-check

- [ ] I can distinguish Factory Method from Abstract Factory.
- [ ] I can write a fluent Builder with validation in `build()`.
- [ ] I can implement a thread-safe Singleton using double-checked locking.
- [ ] I understand why the module singleton is often preferable in Python.

---

## Supplementary links

- *Design Patterns* — GoF, Chapters 3.1 (Abstract Factory), 3.2 (Builder), 3.5 (Singleton)
- [Python Patterns — Factory](https://python-patterns.guide/gang-of-four/factory-method/) — python-patterns.guide
- [Python Patterns — Builder](https://python-patterns.guide/gang-of-four/builder/)
- [Thread-safe Singleton in Python](https://refactoring.guru/design-patterns/singleton/python/example) — Refactoring.guru
