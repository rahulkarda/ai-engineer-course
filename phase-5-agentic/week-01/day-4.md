# Phase 5 · Week 01 · Day 4 — LLM Client Wrapper (mockable)

> Objective: Build a thin, mockable LLM client that wraps the Anthropic Messages API and supports easy test substitution.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Your agent loop calls the LLM on every iteration. For the loop to be testable without real API calls, the LLM client must be:

1. **Abstracted** behind a protocol/interface so tests can swap in a fake.
2. **Deterministic under mocking** — given the same inputs, the fake always returns the same output.
3. **Minimal** — it should do one thing: take a list of messages and return a string.

### Design choices

| Option | Pros | Cons |
|--------|------|------|
| Raw `anthropic.Anthropic()` everywhere | Simple | Impossible to unit test without real API |
| Thin wrapper class with `__call__` | Clean boundary, mockable | Small overhead |
| Protocol/ABC + real + fake implementations | Type-safe | More files |

You will use option 2 (wrapper class) with a `Protocol` for type safety — the real implementation calls Anthropic, the fake returns scripted responses.

### The Anthropic Messages API essentials

```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-haiku-5-5",          # always Claude 5 family
    max_tokens=1024,
    system="You are a helpful agent.",
    messages=[
        {"role": "user", "content": "Hello"},
    ],
)
text = response.content[0].text
```

Key parameters:
- `model` — always use `claude-haiku-5-5`, `claude-sonnet-5-5`, or `claude-opus-5-5`
- `max_tokens` — required; set a sensible ceiling
- `system` — the agent's persona / instructions (keep out of message list)
- `messages` — the conversation history as `[{"role": ..., "content": ...}]`

## How it works internally

### Response object structure

```python
response.content        # list of ContentBlock
response.content[0]     # first block (usually the text block)
response.content[0].text  # the actual string
response.stop_reason    # "end_turn" | "tool_use" | "max_tokens"
response.usage          # InputTokens + OutputTokens
```

`stop_reason` is important for agents:
- `"end_turn"` — model finished naturally
- `"max_tokens"` — truncated, handle gracefully
- `"tool_use"` — model wants to call a tool (when tool use is configured)

### Why wrap at all?

The wrapper is your seam. Without it:
- Every test hits the network (slow, costs money, flaky)
- You can't test failure scenarios (rate limits, bad JSON)
- You can't assert what was sent to the LLM

With the wrapper, tests instantiate `FakeLLMClient` with a list of scripted responses. The agent loop sees the same interface.

## Worked example

```python
"""
day4_llm_client.py — real and fake LLM client implementations.
Tests import FakeLLMClient; production code imports AnthropicLLMClient.
"""

import logging
import os
from collections.abc import Iterator
from typing import Protocol

import anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": ..., "content": ...}


# ---------- Protocol (the seam) ----------

class LLMClient(Protocol):
    """Anything that implements __call__ with this signature is a valid LLM client."""

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        ...


# ---------- Real client ----------

class AnthropicLLMClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        log.info("calling LLM model=%s messages=%d", self.model, len(messages))
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        if response.stop_reason == "max_tokens":
            log.warning("LLM response was truncated (max_tokens reached)")

        text = response.content[0].text
        log.info("LLM response length=%d stop_reason=%s", len(text), response.stop_reason)
        return text


# ---------- Fake client (for tests) ----------

class FakeLLMClient:
    """
    Scripted LLM client for tests.
    Pass a list of strings; each __call__ pops and returns the next one.
    Raises StopIteration when the script is exhausted — tests that hit this
    have called the LLM more times than expected.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.calls: list[list[Message]] = []   # record every call for assertions

    def __call__(self, messages: list[Message], *, system: str = "") -> str:
        self.calls.append(messages)
        return next(self._responses)


# ---------- Demo (no API key needed) ----------

def demo_fake() -> None:
    script = [
        '{"action": "search", "args": {"query": "agent loop"}}',
        '{"action": "finish", "answer": "An agent loop is observe-think-act."}',
    ]
    client: LLMClient = FakeLLMClient(script)

    resp1 = client([{"role": "user", "content": "What is an agent loop?"}])
    log.info("fake response 1: %s", resp1)

    resp2 = client([{"role": "user", "content": "follow up"}])
    log.info("fake response 2: %s", resp2)


if __name__ == "__main__":
    demo_fake()
```

### Unit tests

```python
"""tests/test_llm_client.py"""

import pytest
from day4_llm_client import FakeLLMClient


def test_fake_returns_scripted_responses() -> None:
    client = FakeLLMClient(["response_a", "response_b"])
    assert client([{"role": "user", "content": "q1"}]) == "response_a"
    assert client([{"role": "user", "content": "q2"}]) == "response_b"


def test_fake_records_calls() -> None:
    client = FakeLLMClient(["r1"])
    msgs = [{"role": "user", "content": "hello"}]
    client(msgs)
    assert len(client.calls) == 1
    assert client.calls[0] == msgs


def test_fake_exhausted_raises() -> None:
    client = FakeLLMClient(["only_one"])
    client([{"role": "user", "content": "first"}])
    with pytest.raises(StopIteration):
        client([{"role": "user", "content": "second"}])
```

## Exercises

1. Add a `call_count` property to `FakeLLMClient` that returns `len(self.calls)`. Write a test that creates a fake with 3 scripted responses, makes 2 calls, and asserts `call_count == 2`.

2. Add a `system_prompts: list[str]` field to `FakeLLMClient` that records the `system` kwarg for each call (empty string if not provided). Write a test that passes `system="You are a search agent"` on two calls and asserts both are recorded.

3. Add a `delay_ms: int = 0` parameter to `AnthropicLLMClient.__init__`. When set, sleep for `delay_ms / 1000` seconds before making the API call. This simulates latency in tests without hitting the real API — implement this using `FakeLLMClient` with `delay_ms` too. Write a test that verifies a `FakeLLMClient(["r"], delay_ms=50)` takes at least 50 ms for one call (use `time.time()`).

(Solutions: ../solutions/week-01/day-4.md)

## Self-check

- [ ] I can explain the role of a Protocol in making code testable.
- [ ] I can implement `FakeLLMClient` from scratch without reference.
- [ ] I can call the Anthropic Messages API with the correct parameters using a Claude 5-family model.
- [ ] I can write a unit test that asserts both the return value and the recorded call history of a fake.

## Supplementary links

- Anthropic Python SDK: https://github.com/anthropics/anthropic-sdk-python
- Anthropic models reference: https://docs.anthropic.com/en/docs/about-claude/models
- Python `typing.Protocol` docs: https://docs.python.org/3/library/typing.html#typing.Protocol
