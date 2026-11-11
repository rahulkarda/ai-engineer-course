# Solutions — Phase 5 · Week 01 · Day 4

## Exercise 1 — `call_count` property

```python
class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(self, messages: list[dict], *, system: str = "") -> str:
        self.calls.append(messages)
        return next(self._responses)
```

```python
# tests/test_day4_ex1.py
from day4_llm_client import FakeLLMClient

def test_call_count() -> None:
    client = FakeLLMClient(["r1", "r2", "r3"])
    client([{"role": "user", "content": "a"}])
    client([{"role": "user", "content": "b"}])
    assert client.call_count == 2
```

---

## Exercise 2 — `system_prompts` field

```python
class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []
        self.system_prompts: list[str] = []   # added

    def __call__(self, messages: list[dict], *, system: str = "") -> str:
        self.calls.append(messages)
        self.system_prompts.append(system)    # record
        return next(self._responses)
```

```python
# tests/test_day4_ex2.py
from day4_llm_client import FakeLLMClient

def test_system_prompts_recorded() -> None:
    client = FakeLLMClient(["r1", "r2"])
    client([{"role": "user", "content": "q1"}], system="You are a search agent")
    client([{"role": "user", "content": "q2"}], system="You are a search agent")
    assert len(client.system_prompts) == 2
    assert client.system_prompts[0] == "You are a search agent"
    assert client.system_prompts[1] == "You are a search agent"
```

---

## Exercise 3 — `delay_ms` + timing test

```python
import time

class FakeLLMClient:
    def __init__(self, responses: list[str], delay_ms: int = 0) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []
        self.system_prompts: list[str] = []
        self.delay_ms = delay_ms

    def __call__(self, messages: list[dict], *, system: str = "") -> str:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)
        self.calls.append(messages)
        self.system_prompts.append(system)
        return next(self._responses)
```

```python
# tests/test_day4_ex3.py
import time
from day4_llm_client import FakeLLMClient

def test_delay_is_observed() -> None:
    client = FakeLLMClient(["r"], delay_ms=50)
    start = time.time()
    client([{"role": "user", "content": "q"}])
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms >= 50, f"expected >=50ms but got {elapsed_ms:.1f}ms"
```
