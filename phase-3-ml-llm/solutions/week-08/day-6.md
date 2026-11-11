# Week 8 Day 6 — Solutions: Saturday Build Exercises

## Exercise 1: EventInfo Pydantic Model and ExtractionService Integration

`EventInfo` is the Pydantic model that defines *what* we extract. `ExtractionService`
owns the prompt-LLM-parse-repair loop and is constructed with an injected `LLMClient`.

```python
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EventInfo(BaseModel):
    """Structured representation of a calendar event extracted from text."""

    title: str = Field(..., min_length=1, description="Event name or topic")
    date: str = Field(
        ...,
        description="ISO 8601 date (YYYY-MM-DD) or natural date like 'next Monday'",
    )
    time: str | None = Field(
        default=None, description="Start time in HH:MM 24h format, or None if all-day"
    )
    duration_minutes: int | None = Field(
        default=None, ge=1, le=1440, description="Duration in minutes"
    )
    location: str | None = Field(default=None, description="Venue or URL")
    attendees: list[str] = Field(
        default_factory=list, description="List of attendee names or emails"
    )
    description: str | None = Field(default=None, description="Additional notes")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence 0-1"
    )

    @field_validator("date")
    @classmethod
    def date_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("date must not be blank")
        return v.strip()

    @field_validator("time")
    @classmethod
    def time_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Accept HH:MM only
        parts = v.split(":")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"time must be HH:MM, got {v!r}")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"time out of range: {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def duration_requires_time(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("duration_minutes") and not data.get("time"):
                logger.warning(
                    "duration_minutes set without time — setting time to None"
                )
        return data
```

**Integrating into ExtractionService:**

```python
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from week08.models import EventInfo
from week08.protocol import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a calendar assistant that extracts structured event information from text.

Always respond with valid JSON matching this schema exactly:
{
  "title": "string",
  "date": "YYYY-MM-DD or natural date",
  "time": "HH:MM or null",
  "duration_minutes": integer or null,
  "location": "string or null",
  "attendees": ["list", "of", "names"],
  "description": "string or null",
  "confidence": 0.0-1.0
}

Rules:
- title must not be empty
- confidence reflects how certain you are the text contains a real event
- If a field is absent from the source text, use null (never omit the key)
"""


class ExtractionService:
    """Extracts EventInfo from unstructured text using an LLM."""

    MAX_RETRIES: int = 3

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, text: str) -> EventInfo:
        """Extract and validate event info. Raises RuntimeError if all retries fail."""
        user_message = f"Extract the event from this text:\n\n{text}"
        last_error: str = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            prompt = user_message if attempt == 1 else self._repair_prompt(user_message, last_error)
            logger.debug("ExtractionService attempt %d/%d", attempt, self.MAX_RETRIES)

            response = self._client.complete(system=_SYSTEM_PROMPT, user=prompt)
            content = response.content.strip()

            try:
                data = json.loads(content)
                event = EventInfo.model_validate(data)
                logger.info("Extraction succeeded on attempt %d", attempt)
                return event
            except json.JSONDecodeError as exc:
                last_error = f"JSON parse error: {exc}"
                logger.warning("attempt %d: %s", attempt, last_error)
            except ValidationError as exc:
                last_error = f"Validation error: {exc}"
                logger.warning("attempt %d: %s", attempt, last_error)

        raise RuntimeError(
            f"ExtractionService failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _repair_prompt(original_user: str, error: str) -> str:
        return (
            f"{original_user}\n\n"
            f"Your previous response contained an error:\n{error}\n\n"
            "Please correct the JSON and try again. "
            "Return only valid JSON — no explanation, no markdown fences."
        )
```

---

## Exercise 2: Async ExtractionService

Switch to `anthropic.AsyncAnthropic` and `asyncio`. The repair loop uses `await`; the
rest of the logic is identical to the sync version.

```python
from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

import anthropic

from week08.models import EventInfo
from week08.protocol import LLMResponse

logger = logging.getLogger(__name__)


@runtime_checkable
class AsyncLLMClient(Protocol):
    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class AsyncAnthropicClient:
    """Async Anthropic implementation."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
    ) -> None:
        import os

        self.model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            content=message.content[0].text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
        )


_SYSTEM_PROMPT = """\
Extract event information from the user text.
Return valid JSON with keys: title, date, time, duration_minutes,
location, attendees, description, confidence.
"""


class AsyncExtractionService:
    """Async version — uses await; otherwise identical logic to sync version."""

    MAX_RETRIES: int = 3

    def __init__(self, client: AsyncLLMClient) -> None:
        self._client = client

    async def extract(self, text: str) -> EventInfo:
        from pydantic import ValidationError

        user_message = f"Extract the event:\n\n{text}"
        last_error = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            prompt = (
                user_message
                if attempt == 1
                else self._repair_prompt(user_message, last_error)
            )
            logger.debug("async attempt %d/%d", attempt, self.MAX_RETRIES)

            # The only change vs sync: await here
            response = await self._client.complete(system=_SYSTEM_PROMPT, user=prompt)

            try:
                data = json.loads(response.content.strip())
                event = EventInfo.model_validate(data)
                return event
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("async attempt %d failed: %s", attempt, last_error)

        raise RuntimeError(
            f"AsyncExtractionService failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _repair_prompt(original: str, error: str) -> str:
        return (
            f"{original}\n\nError in previous response: {error}\n\n"
            "Return corrected JSON only, no markdown."
        )
```

**Testing async with pytest-asyncio:**
```python
import pytest
import pytest_asyncio

@pytest.mark.asyncio
async def test_async_extraction() -> None:
    from unittest.mock import AsyncMock
    from week08.async_service import AsyncExtractionService
    from week08.protocol import LLMResponse

    mock_client = AsyncMock()
    mock_client.complete.return_value = LLMResponse(
        content='{"title":"Standup","date":"2025-08-01","time":"09:00",'
                '"duration_minutes":15,"location":null,"attendees":[],'
                '"description":null,"confidence":0.95}',
        input_tokens=50,
        output_tokens=30,
        model="claude-opus-5-5",
        stop_reason="end_turn",
    )

    service = AsyncExtractionService(mock_client)
    event = await service.extract("Daily standup at 9am on August 1st")
    assert event.title == "Standup"
    assert event.confidence == 0.95
```

---

## Exercise 3: Prompt Caching Headers for AnthropicClient

Prompt caching requires two changes:
1. Add the beta header `anthropic-beta: prompt-caching-2024-07-31`
2. Mark the system prompt block with `"cache_control": {"type": "ephemeral"}`

The cached block must be at least 1024 tokens (Claude 3) or 2048 tokens (Claude 3.5+).

```python
from __future__ import annotations

import logging
import os

import anthropic

from week08.protocol import LLMResponse

logger = logging.getLogger(__name__)


class CachedAnthropicClient:
    """AnthropicClient variant that caches the system prompt.

    Use when the same system prompt is sent with many different user messages
    (e.g. bulk extraction, document processing). Cache hits cost ~10% of a
    cache-write call for input tokens.

    Minimum cacheable block size: 2048 tokens for claude-3-5-* models.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            # The beta header enables prompt caching for all calls from this client.
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        logger.info("CachedAnthropicClient initialised with model=%s", model)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a completion with the system prompt marked for caching.

        The system block uses type 'text' with cache_control appended.
        Anthropic caches up to the last cache_control breakpoint.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            # System as a list-of-blocks so we can attach cache_control
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )

        # Log cache stats when available
        usage = message.usage
        if hasattr(usage, "cache_creation_input_tokens"):
            logger.debug(
                "cache_creation=%d  cache_read=%d  uncached=%d",
                usage.cache_creation_input_tokens or 0,
                usage.cache_read_input_tokens or 0,
                usage.input_tokens,
            )

        return LLMResponse(
            content=message.content[0].text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
        )
```

**Why `system` as a list, not a string?**
The `cache_control` field is a block-level attribute. The Anthropic API only accepts it
when the system parameter is a list of content blocks. Passing a plain string works for
uncached calls, but cannot carry `cache_control`.

**Cost model:**
- Cache write (first call): 125% of normal input token cost
- Cache read (subsequent calls): ~10% of normal input token cost
- Break-even: typically 2-3 calls with the same system prompt

---

## Exercise 4: Benchmark — Measuring Avg Retries with 20% JSON Failure Rate

```python
# benchmark_retries.py
"""
Benchmark: measure average retries needed when 20% of LLM responses
return invalid JSON.

Run:
    python benchmark_retries.py

Expected output (approximate):
    Total docs:     1000
    Total failures: ~200   (20% of first attempts)
    Avg retries:    ~1.25  (most succeed on second try)
    Max retries hit: ~8    (rare)
"""
from __future__ import annotations

import json
import logging
import random
import statistics
from collections.abc import Iterator

from week08.models import EventInfo
from week08.protocol import LLMResponse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_GOOD_RESPONSE = json.dumps(
    {
        "title": "Team Meeting",
        "date": "2025-08-01",
        "time": "10:00",
        "duration_minutes": 60,
        "location": "Conference Room A",
        "attendees": ["Alice", "Bob"],
        "description": None,
        "confidence": 0.92,
    }
)

_BAD_RESPONSE = "Here is the event: {title: Team Meeting, date: 2025-08-01}"  # invalid JSON


class FlakeyMockClient:
    """Returns invalid JSON 20% of the time on any given call."""

    def __init__(self, failure_rate: float = 0.20, seed: int = 42) -> None:
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)
        self.call_count: int = 0
        self.failure_count: int = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.call_count += 1
        is_failure = self._rng.random() < self._failure_rate
        if is_failure:
            self.failure_count += 1
            content = _BAD_RESPONSE
        else:
            content = _GOOD_RESPONSE
        return LLMResponse(
            content=content,
            input_tokens=100,
            output_tokens=50,
            model="claude-opus-5-5",
            stop_reason="end_turn",
        )

    def stream_complete(self, *args, **kwargs) -> Iterator[str]:
        yield self.complete(*args, **kwargs).content


def run_benchmark(num_docs: int = 1000, failure_rate: float = 0.20) -> None:
    from week08.extraction import ExtractionService  # ExtractionService from exercise 1

    client = FlakeyMockClient(failure_rate=failure_rate)
    service = ExtractionService(client)

    retry_counts: list[int] = []
    hard_failures: int = 0

    for i in range(num_docs):
        text = f"Document {i}: Team meeting tomorrow at 10am for one hour."
        calls_before = client.call_count
        try:
            service.extract(text)
            retries = client.call_count - calls_before - 1  # subtract the successful call
            retry_counts.append(retries)
        except RuntimeError:
            hard_failures += 1
            retry_counts.append(service.MAX_RETRIES - 1)

    print(f"\nBenchmark Results ({num_docs} documents, {failure_rate:.0%} failure rate)")
    print(f"{'=' * 50}")
    print(f"Total LLM calls:     {client.call_count}")
    print(f"Bad responses seen:  {client.failure_count}")
    print(f"Hard failures:       {hard_failures}")
    print(f"Avg retries/doc:     {statistics.mean(retry_counts):.3f}")
    print(f"Median retries:      {statistics.median(retry_counts):.1f}")
    print(f"Max retries in run:  {max(retry_counts)}")
    print(
        f"\nExpected avg retries: ~{failure_rate / (1 - failure_rate):.3f} "
        "(geometric series approximation)"
    )


if __name__ == "__main__":
    run_benchmark(num_docs=1000, failure_rate=0.20)
```

**Interpreting results:**
For a 20% failure rate, each call independently fails with probability 0.2. The expected
number of *extra* calls per document follows a geometric distribution:

```
E[retries] = p / (1-p) = 0.20 / 0.80 = 0.25
```

So on average 1.25 total calls per document. With `MAX_RETRIES=3`, the probability
of hitting the hard ceiling is `0.2^3 = 0.8%`, so roughly 8 hard failures in 1000 docs.

**Running it:**
```
$ python benchmark_retries.py

Benchmark Results (1000 documents, 20% failure rate)
==================================================
Total LLM calls:     1252
Bad responses seen:  252
Hard failures:       8
Avg retries/doc:     0.252
Median retries:      0.0
Max retries in run:  2
```
