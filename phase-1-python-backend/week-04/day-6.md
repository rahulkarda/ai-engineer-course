# Phase 1 · Week 04 · Day 6 — Type Hints, Generics & TypedDict

> Objective: Master Python's type annotation system — from basic syntax through generics, Protocols, and TypeGuard — so your code is statically verifiable and self-documenting.  ·  Time: ~4 hrs  ·  Prereqs: [[day-5]]

> This is Saturday: ~4 hours. More exercises, deeper content.

## Concept

Python is dynamically typed: the interpreter does not enforce types at runtime. Type hints, introduced in PEP 484, allow you to annotate variables, parameters, and return values with type information that static analysers (mypy, pyright, Pylance) can verify without running the code.

Type hints serve two audiences:
1. **Static tools** — find bugs before runtime by checking that values flow consistently through annotated call sites.
2. **Human readers** — document what a function expects and returns, reducing the need to read the body.

Runtime behaviour is unchanged: adding `x: int = 5` does not prevent you from later doing `x = "hello"`. The enforcement is entirely in the tooling.

## How it works internally

### Annotation storage

At module level and in class bodies, annotations are collected into `__annotations__`, a dict mapping name to type expression. Function annotations are stored in `__annotations__` on the function object. Accessing these at runtime gives you the raw expression — either a type object or (with `from __future__ import annotations`) a string (lazy evaluation).

`from __future__ import annotations` (PEP 563) makes all annotations strings at runtime, deferring evaluation. This avoids `NameError` for forward references and speeds import time. Python 3.14+ will change the default, but for 3.12 this pragma is recommended in larger codebases.

`typing.get_type_hints(obj)` evaluates the stringified annotations against the correct namespace, resolving forward references.

### Type hint syntax evolution

**Python 3.9+**: Built-in generics — use `list[int]`, `dict[str, int]`, `tuple[int, ...]` directly. No need to import `List`, `Dict` from `typing`.

**Python 3.10+**: Union with `|` operator — `int | str | None` replaces `Union[int, str, Optional[int]]`. `match`/`case` statements for structural pattern matching.

**Python 3.12**: `type` statement for type aliases:
```python
type Vector = list[float]
type Matrix = list[Vector]
```
This replaces `Vector = list[float]` and the older `TypeAlias` from `typing_extensions`.

### typing module essentials

- `Optional[X]` — equivalent to `X | None`; use `X | None` in 3.10+ code
- `Union[X, Y]` — equivalent to `X | Y`; prefer `|` in 3.10+
- `Any` — opts out of type checking; a value typed `Any` is compatible with everything
- `Callable[[ArgType, ...], ReturnType]` — for function types
- `TypeVar("T")` — generic type variable, used to express that input and output types are related
- `Generic[T]` — base class for generic classes
- `Protocol` — structural subtyping ("duck typing" with static verification)
- `Final` — marks a variable as constant (cannot be reassigned)
- `Literal` — restricts a type to specific values: `Literal["GET", "POST"]`
- `Annotated[X, metadata]` — attaches extra metadata to a type; used by Pydantic, FastAPI, etc.
- `TypedDict` — a `dict` with a known set of typed string keys

### TypedDict

`TypedDict` creates a dict type with specific keys and value types. Unlike dataclasses, the underlying runtime value is a plain `dict` — no class overhead, JSON-friendly.

```python
from typing import TypedDict, Required, NotRequired

class Config(TypedDict):
    host: str                    # required by default (total=True)
    port: int
    debug: NotRequired[bool]     # explicitly optional (3.11+)
```

`total=False` makes all keys optional. Combining `total=True` and `total=False` via inheritance:

```python
class _RequiredConfig(TypedDict):
    host: str

class Config(_RequiredConfig, total=False):
    port: int
    debug: bool
```

### TypeVar constraints and bounds

`TypeVar("T", bound=SomeClass)` means T must be `SomeClass` or a subclass.
`TypeVar("T", int, str)` means T must be exactly `int` or exactly `str` (constrained TypeVar).

### ParamSpec and Concatenate

`ParamSpec` captures the parameter signature of a callable, enabling correct typing of decorators that wrap functions without changing their signature.

```python
from typing import ParamSpec, Callable, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def logged(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

### Protocol

`Protocol` enables structural subtyping: a class satisfies a Protocol if it has the required methods/attributes, regardless of inheritance. This is PEP 544 and is how Python formalises duck typing.

```python
from typing import Protocol

class Serializable(Protocol):
    def to_json(self) -> str: ...

def save(obj: Serializable) -> None:
    print(obj.to_json())
```

Any class with a `to_json(self) -> str` method satisfies `Serializable` without inheriting from it.

### Type narrowing

Type narrowing is when a static analyser infers a more specific type after a check:

```python
def process(value: int | str) -> None:
    if isinstance(value, int):
        # here value: int
        print(value + 1)
    else:
        # here value: str
        print(value.upper())
```

`TypeGuard[T]` lets you write custom narrowing functions:

```python
from typing import TypeGuard

def is_list_of_str(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)
```

## Worked example

The example builds a typed generic cache, a typed pipeline, TypedDict config, and a Protocol for duck typing.

```python
"""
day6_types.py
Comprehensive demonstration of Python 3.12+ type system features.
"""
from __future__ import annotations

import logging
import time
from typing import (
    Annotated,
    Any,
    Callable,
    Final,
    Generic,
    Literal,
    NotRequired,
    ParamSpec,
    Protocol,
    Required,
    TypedDict,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Type aliases (Python 3.12 syntax)
# ---------------------------------------------------------------------------

type Vector = list[float]
type Matrix = list[Vector]
type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


# ---------------------------------------------------------------------------
# 2. TypedDict — typed config dicts
# ---------------------------------------------------------------------------

class _DBRequired(TypedDict):
    host: str
    port: int
    database: str


class DBConfig(_DBRequired, total=False):
    """Database connection config. host/port/database are required; rest optional."""
    username: str
    password: str
    pool_size: int


class PipelineConfig(TypedDict):
    name: str
    steps: list[str]
    config: NotRequired[DBConfig]
    mode: Required[Literal["batch", "stream"]]


def load_config() -> PipelineConfig:
    return {
        "name": "etl-pipeline",
        "steps": ["extract", "transform", "load"],
        "mode": "batch",
        "config": {
            "host": "localhost",
            "port": 5432,
            "database": "analytics",
        },
    }


# ---------------------------------------------------------------------------
# 3. Generic class — typed LRU-like cache
# ---------------------------------------------------------------------------

KT = TypeVar("KT")
VT = TypeVar("VT")


class TypedCache(Generic[KT, VT]):
    """A generic in-memory cache with a configurable max size."""

    def __init__(self, max_size: int = 128) -> None:
        self._store: dict[KT, VT] = {}
        self._max_size: Final[int] = max_size

    def get(self, key: KT) -> VT | None:
        return self._store.get(key)

    def put(self, key: KT, value: VT) -> None:
        if len(self._store) >= self._max_size:
            # Evict oldest key (insertion order guaranteed in 3.7+)
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store


def demo_cache() -> None:
    log.info("=== Generic TypedCache ===")

    # str -> float cache for computed values
    cache: TypedCache[str, float] = TypedCache(max_size=3)
    cache.put("pi", 3.14159)
    cache.put("e", 2.71828)
    log.info("pi in cache: %s | value: %s", "pi" in cache, cache.get("pi"))
    log.info("tau in cache: %s", cache.get("tau"))

    # int -> list[str] cache
    int_cache: TypedCache[int, list[str]] = TypedCache()
    int_cache.put(1, ["a", "b"])
    log.info("int_cache[1]: %s", int_cache.get(1))


# ---------------------------------------------------------------------------
# 4. Protocol — structural subtyping
# ---------------------------------------------------------------------------

@runtime_checkable
class Transformer(Protocol):
    """Anything with a transform(data) -> list[float] method."""

    def transform(self, data: list[Any]) -> list[float]: ...


class NormalisationStep:
    """Implements Transformer structurally (no explicit inheritance)."""

    def transform(self, data: list[Any]) -> list[float]:
        floats: list[float] = [float(x) for x in data]
        if not floats:
            return floats
        min_val = min(floats)
        max_val = max(floats)
        if max_val == min_val:
            return [0.0] * len(floats)
        return [(v - min_val) / (max_val - min_val) for v in floats]


class LoggingStep:
    """Another Transformer that just logs and passes through."""

    def transform(self, data: list[Any]) -> list[float]:
        floats = [float(x) for x in data]
        log.info("LoggingStep received %d items", len(floats))
        return floats


def run_pipeline(steps: list[Transformer], data: list[Any]) -> list[float]:
    result: list[Any] = data
    for step in steps:
        result = step.transform(result)
    return result


def demo_protocol() -> None:
    log.info("=== Protocol (structural subtyping) ===")

    # Both satisfy Transformer without inheriting it
    pipeline_steps: list[Transformer] = [LoggingStep(), NormalisationStep()]
    output = run_pipeline(pipeline_steps, [10, 20, 30, 40, 50])
    log.info("Pipeline output: %s", output)

    # runtime_checkable enables isinstance checks
    step = NormalisationStep()
    log.info("isinstance(step, Transformer): %s", isinstance(step, Transformer))


# ---------------------------------------------------------------------------
# 5. ParamSpec — typed decorator
# ---------------------------------------------------------------------------

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs execution time, preserving the original signature."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.info("%s took %.4fs", func.__name__, elapsed)
        return result

    return wrapper


@timed
def expensive_computation(n: int, multiplier: float = 1.0) -> float:
    return sum(i * multiplier for i in range(n))


# ---------------------------------------------------------------------------
# 6. Literal types and Final
# ---------------------------------------------------------------------------

HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

MAX_RETRIES: Final[int] = 3


def make_request(method: HTTPMethod, url: str) -> None:
    log.info("Request: %s %s (max retries: %d)", method, url, MAX_RETRIES)


# ---------------------------------------------------------------------------
# 7. Annotated with metadata
# ---------------------------------------------------------------------------

# Annotated is used by Pydantic, FastAPI, etc. to attach validation metadata
PositiveFloat = Annotated[float, "must be > 0"]
Port = Annotated[int, "valid TCP port", "range: 1-65535"]


def create_server(host: str, port: Port) -> dict[str, str | int]:
    return {"host": host, "port": port}


# ---------------------------------------------------------------------------
# 8. TypeGuard for custom narrowing
# ---------------------------------------------------------------------------

def is_list_of_str(val: list[object]) -> TypeGuard[list[str]]:
    """Return True iff every element of val is a str."""
    return all(isinstance(x, str) for x in val)


def process_tags(tags: list[object]) -> None:
    if is_list_of_str(tags):
        # type narrowed to list[str] here
        log.info("Tags (upper): %s", [t.upper() for t in tags])
    else:
        log.info("Tags contain non-strings: %s", tags)


# ---------------------------------------------------------------------------
# 9. TypeVar bounds
# ---------------------------------------------------------------------------

from pathlib import Path


class FileResource:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __repr__(self) -> str:
        return f"FileResource({self.path})"


class DirectoryResource(FileResource):
    pass


ResourceT = TypeVar("ResourceT", bound=FileResource)


def clone_resource(resource: ResourceT) -> ResourceT:
    """Return a copy preserving the concrete subtype."""
    return resource.__class__(resource.path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Config
    config = load_config()
    log.info("Config: %s", config)

    # Cache
    demo_cache()

    # Protocol
    demo_protocol()

    # Timed decorator (ParamSpec in action)
    result = expensive_computation(1_000_000, multiplier=2.5)
    log.info("Computation result: %.2f", result)

    # Literal + Final
    make_request("GET", "https://api.example.com/data")

    # Annotated
    server = create_server("0.0.0.0", 8080)
    log.info("Server: %s", server)

    # TypeGuard
    process_tags(["alpha", "beta", "gamma"])
    process_tags(["alpha", 42, "gamma"])

    # TypeVar bound
    dir_res = DirectoryResource(Path("/tmp/data"))
    cloned = clone_resource(dir_res)
    log.info("Cloned resource type preserved: %s | %s", type(cloned).__name__, cloned)
```

Run:
```bash
python day6_types.py
```

## Exercises

1. Define a `TypedDict` called `AgentConfig` with required fields `name: str`, `model: str`, and optional fields `temperature: float`, `max_tokens: int`, `system_prompt: str`. Write a function `validate_agent_config(cfg: AgentConfig) -> bool` that checks temperature is in `[0.0, 2.0]` if present.

2. Write a generic `Result[T]` class (similar to Rust's `Result`) with two class methods: `ok(value: T) -> Result[T]` and `err(message: str) -> Result[T]`. Add a property `is_ok: bool` and a method `unwrap() -> T` that raises `ValueError` if it is an error. Ensure the whole class is correctly typed.

3. Write a `Protocol` called `Serializable` with methods `to_dict() -> dict[str, Any]` and `from_dict(cls, data: dict[str, Any]) -> Self`. Mark it `@runtime_checkable`. Write two classes that satisfy it without inheriting from it, and verify with `isinstance`.

4. Write a decorator `retry(max_attempts: int)` using `ParamSpec` and `TypeVar` that retries a function up to `max_attempts` times if it raises an exception. The decorator must preserve the original function's type signature. Test it with a function that fails the first two calls and succeeds on the third.

5. Create a `TypeVar` constrained to `int | float | complex`. Write a function `add_all(values: list[T]) -> T` that uses it. Verify with mypy or pyright that passing `list[str]` is rejected.

6. Write a `TypeGuard` function `is_non_empty_list(val: list[T]) -> TypeGuard[list[T]]` that returns `True` iff `val` is non-empty. Use it in a function that processes only non-empty lists and verify type narrowing works.

7. Using `Annotated`, create type aliases `Email = Annotated[str, "RFC 5321 email address"]` and `URL = Annotated[str, "HTTP/HTTPS URL"]`. Write a `ContactInfo(TypedDict)` that uses these aliases for its fields. Extract and log the metadata from `typing.get_type_hints(ContactInfo, include_extras=True)`.

8. Write a `Pipeline` generic class `Pipeline[InputT, OutputT]` that holds a list of processing steps. Each step is a `Callable[[InputT], OutputT]`. Add an `add_step` method and a `run(data: InputT) -> OutputT` method. Instantiate it with `str -> list[str] -> int` (split -> count words).

(Solutions: ../solutions/week-04/day-6.md)

## Self-check

- [ ] I can write `list[int]`, `dict[str, float]`, `int | None` without importing from `typing` (3.10+ syntax).
- [ ] I can define a `TypedDict` with both required and optional keys using `total=False` inheritance or `NotRequired`.
- [ ] I can write a generic class `Container[T]` and instantiate it as `Container[str]`.
- [ ] I can explain the difference between `TypeVar("T")`, `TypeVar("T", int, str)`, and `TypeVar("T", bound=MyClass)`.
- [ ] I can write a `Protocol` for duck typing and mark it `@runtime_checkable`.
- [ ] I can use `ParamSpec` to write a decorator that preserves the original function's signature.
- [ ] I can write a `TypeGuard` function and explain how the type narrowing works.
- [ ] I know when to use `Any` and why overusing it defeats the purpose of type hints.
- [ ] I understand that `from __future__ import annotations` makes all annotations lazy strings.

## Supplementary links

- https://docs.python.org/3/library/typing.html — full typing module reference
- https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html — mypy cheat sheet
- https://peps.python.org/pep-0484/ — PEP 484: type hints
- https://peps.python.org/pep-0544/ — PEP 544: Protocols
- https://peps.python.org/pep-0612/ — PEP 612: ParamSpec
- https://peps.python.org/pep-0695/ — PEP 695: type statement (Python 3.12)
- https://peps.python.org/pep-0655/ — PEP 655: Required/NotRequired in TypedDict
- https://mypy.readthedocs.io/en/stable/protocols.html — using Protocols with mypy
