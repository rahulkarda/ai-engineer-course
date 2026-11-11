# Solutions — Phase 5 · Week 04 · Day 3 — Context Window Management & Summarization

---

## Exercise 1

Add a `token_ratio` property returning `used / total` as float.

```python
"""
context_manager.py — ContextManager with token_ratio property.
"""
from __future__ import annotations

import logging
from typing import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


def _default_token_fn(text: str) -> int:
    """Approximate token count via word splitting."""
    return len(text.split())


class ContextManager:
    """Manages a sliding window of messages within a token budget."""

    def __init__(
        self,
        max_tokens: int = 4096,
        token_fn: Callable[[str], int] = _default_token_fn,
    ) -> None:
        self.max_tokens = max_tokens
        self._token_fn = token_fn
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def _count(self, msg: Message) -> int:
        return self._token_fn(msg.content)

    def _used_tokens(self) -> int:
        return sum(self._count(m) for m in self._messages)

    @property
    def token_ratio(self) -> float:
        """Fraction of token budget consumed: used / max_tokens."""
        if self.max_tokens == 0:
            return 0.0
        ratio = self._used_tokens() / self.max_tokens
        return min(ratio, 1.0)

    def _trim(self) -> None:
        while self._used_tokens() > self.max_tokens and len(self._messages) > 1:
            dropped = self._messages.pop(0)
            logger.debug("Trimmed message: %s", dropped.content[:40])

    def messages(self) -> list[Message]:
        return list(self._messages)
```

```python
"""
tests/test_token_ratio.py — Tests for ContextManager.token_ratio.
"""
from __future__ import annotations

import pytest

from context_manager import ContextManager, Message


def test_token_ratio_empty_is_zero() -> None:
    cm = ContextManager(max_tokens=100)
    assert cm.token_ratio == 0.0


def test_token_ratio_does_not_exceed_one() -> None:
    cm = ContextManager(max_tokens=5)
    for i in range(10):
        cm.add(Message(role="user", content=f"word{i} extra filler text here"))
    assert cm.token_ratio <= 1.0


def test_token_ratio_partial_usage() -> None:
    cm = ContextManager(max_tokens=100)
    cm.add(Message(role="user", content="one two three four five"))  # 5 tokens
    assert cm.token_ratio == pytest.approx(5 / 100)


def test_token_ratio_at_capacity() -> None:
    cm = ContextManager(max_tokens=5)
    cm.add(Message(role="user", content="a b c d e"))  # exactly 5 tokens
    assert cm.token_ratio == pytest.approx(1.0)
```

---

## Exercise 2

`keep_system_messages` option prevents system messages from being evicted during trimming.

```python
"""
context_manager_sys.py — ContextManager with keep_system_messages option.
"""
from __future__ import annotations

import logging
from typing import Callable

from context_manager import Message, _default_token_fn

logger = logging.getLogger(__name__)


class ContextManager:
    """Context manager that optionally preserves system messages during trim."""

    def __init__(
        self,
        max_tokens: int = 4096,
        token_fn: Callable[[str], int] = _default_token_fn,
        keep_system_messages: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self._token_fn = token_fn
        self.keep_system_messages = keep_system_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def _count(self, msg: Message) -> int:
        return self._token_fn(msg.content)

    def _used_tokens(self) -> int:
        return sum(self._count(m) for m in self._messages)

    @property
    def token_ratio(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return min(self._used_tokens() / self.max_tokens, 1.0)

    def _trim(self) -> None:
        while self._used_tokens() > self.max_tokens:
            # Find oldest evictable message
            evictable_idx = None
            for i, msg in enumerate(self._messages):
                if self.keep_system_messages and msg.role == "system":
                    continue
                evictable_idx = i
                break
            if evictable_idx is None:
                # All remaining messages are protected system messages
                logger.warning("Cannot trim further; all messages are system messages.")
                break
            dropped = self._messages.pop(evictable_idx)
            logger.debug("Trimmed message: role=%s, content=%s", dropped.role, dropped.content[:40])

    def messages(self) -> list[Message]:
        return list(self._messages)
```

```python
"""
tests/test_context_manager_sys.py — Tests for keep_system_messages option.
"""
from __future__ import annotations

import pytest

from context_manager import Message
from context_manager_sys import ContextManager


def test_system_messages_preserved_during_trim() -> None:
    cm = ContextManager(max_tokens=10, keep_system_messages=True)
    cm.add(Message(role="system", content="You are a helpful assistant"))  # 5 tokens
    cm.add(Message(role="user", content="Hello there friend"))  # 3 tokens
    cm.add(Message(role="user", content="Another question here"))  # 3 tokens
    # Total: 11 > 10, should evict user message, not system
    roles = [m.role for m in cm.messages()]
    assert "system" in roles


def test_system_messages_evicted_without_flag() -> None:
    cm = ContextManager(max_tokens=5, keep_system_messages=False)
    cm.add(Message(role="system", content="a b c d e"))  # 5 tokens exactly
    cm.add(Message(role="user", content="f g h"))  # 3 tokens -> 8 > 5
    # Without protection, system message may be evicted
    assert len(cm.messages()) >= 1  # at least one message remains


def test_keep_system_default_false() -> None:
    cm = ContextManager(max_tokens=4)
    cm.add(Message(role="system", content="sys one two three"))  # 4 tokens
    cm.add(Message(role="user", content="user a b c"))  # 4 tokens -> 8 > 4
    # Default: system message CAN be evicted
    msgs = cm.messages()
    assert len(msgs) >= 1


def test_multiple_system_messages_all_kept() -> None:
    cm = ContextManager(max_tokens=12, keep_system_messages=True)
    cm.add(Message(role="system", content="system rule one two"))  # 4 tokens
    cm.add(Message(role="system", content="system rule three four"))  # 4 tokens
    cm.add(Message(role="user", content="user hello world now"))  # 4 tokens
    cm.add(Message(role="user", content="user extra tokens here"))  # 4 tokens -> 16 > 12
    system_msgs = [m for m in cm.messages() if m.role == "system"]
    assert len(system_msgs) == 2
```

---

## Exercise 3

`TwoTierContextManager`: last 3 messages verbatim, older ones summarized via mock LLM.

```python
"""
two_tier_context.py — Two-tier context: recent verbatim, older summarized.
"""
from __future__ import annotations

import logging
from typing import Callable

from context_manager import Message

logger = logging.getLogger(__name__)

# Type alias for a summarization function
SummarizeFn = Callable[[list[Message]], str]


def _mock_summarize(messages: list[Message]) -> str:
    """Mock LLM summarization — concatenates role:content pairs."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    return f"[Summary of {len(messages)} messages: {' | '.join(parts)}]"


class TwoTierContextManager:
    """Keeps the last *verbatim_count* messages verbatim; older ones are summarized.

    The summarization function is injected so that tests can use a mock without
    making real API calls.
    """

    def __init__(
        self,
        verbatim_count: int = 3,
        summarize_fn: SummarizeFn = _mock_summarize,
    ) -> None:
        self.verbatim_count = verbatim_count
        self._summarize_fn = summarize_fn
        self._all_messages: list[Message] = []
        self._summary: str = ""

    def add(self, message: Message) -> None:
        self._all_messages.append(message)
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        if len(self._all_messages) > self.verbatim_count:
            older = self._all_messages[: -self.verbatim_count]
            self._summary = self._summarize_fn(older)
            self._all_messages = self._all_messages[-self.verbatim_count :]
            logger.debug("Summarized %d older messages.", len(older))

    def get_context(self) -> tuple[str, list[Message]]:
        """Return (summary_string, recent_messages)."""
        return self._summary, list(self._all_messages)
```

```python
"""
tests/test_two_tier_context.py — Tests for TwoTierContextManager.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from context_manager import Message
from two_tier_context import TwoTierContextManager


def _msgs(n: int) -> list[Message]:
    return [Message(role="user", content=f"message {i}") for i in range(n)]


def test_fewer_than_verbatim_no_summary() -> None:
    cm = TwoTierContextManager(verbatim_count=3)
    for m in _msgs(3):
        cm.add(m)
    summary, recent = cm.get_context()
    assert summary == ""
    assert len(recent) == 3


def test_older_messages_summarized() -> None:
    mock_fn = MagicMock(return_value="[mock summary]")
    cm = TwoTierContextManager(verbatim_count=3, summarize_fn=mock_fn)
    for m in _msgs(5):
        cm.add(m)
    summary, recent = cm.get_context()
    assert summary == "[mock summary]"
    assert len(recent) == 3
    mock_fn.assert_called()


def test_only_last_n_verbatim() -> None:
    cm = TwoTierContextManager(verbatim_count=2)
    for m in _msgs(4):
        cm.add(m)
    _, recent = cm.get_context()
    contents = [m.content for m in recent]
    assert "message 2" in contents
    assert "message 3" in contents
    assert "message 0" not in contents


def test_summarize_fn_called_with_older_messages() -> None:
    captured: list[list[Message]] = []

    def capturing_fn(msgs: list[Message]) -> str:
        captured.append(msgs)
        return "summary"

    cm = TwoTierContextManager(verbatim_count=2, summarize_fn=capturing_fn)
    for m in _msgs(4):
        cm.add(m)
    # After 3rd message -> 1 older message summarized; after 4th -> 2 older
    assert len(captured) > 0
    # The last summarization should have included messages 0..1
    last_summarized_contents = {m.content for m in captured[-1]}
    assert "message 0" in last_summarized_contents
```

---

## Exercise 4

`sliding_window(messages, max_tokens, token_fn)` pure function trimming from oldest.

```python
"""
sliding_window.py — Pure sliding window function for context trimming.
"""
from __future__ import annotations

import logging
from typing import Callable

from context_manager import Message, _default_token_fn

logger = logging.getLogger(__name__)


def sliding_window(
    messages: list[Message],
    max_tokens: int,
    token_fn: Callable[[str], int] = _default_token_fn,
) -> list[Message]:
    """Return the longest suffix of *messages* whose total token count <= max_tokens.

    Trims from the oldest (index 0) end. The result always contains at least
    the most recent message even if it alone exceeds the budget.
    """
    if not messages:
        return []

    result = list(messages)
    while sum(token_fn(m.content) for m in result) > max_tokens and len(result) > 1:
        dropped = result.pop(0)
        logger.debug("Sliding window dropped: %s", dropped.content[:40])

    return result
```

```python
"""
tests/test_sliding_window.py — Tests for sliding_window.
"""
from __future__ import annotations

import pytest

from context_manager import Message
from sliding_window import sliding_window


def _msg(content: str) -> Message:
    return Message(role="user", content=content)


def test_empty_returns_empty() -> None:
    assert sliding_window([], 100) == []


def test_within_budget_no_trim() -> None:
    msgs = [_msg("a b"), _msg("c d")]
    result = sliding_window(msgs, max_tokens=10)
    assert result == msgs


def test_trims_oldest_first() -> None:
    msgs = [_msg("one two three"), _msg("four five"), _msg("six seven")]
    result = sliding_window(msgs, max_tokens=4)
    # 13 tokens total -> trim until <= 4; "six seven" = 2 tokens
    assert result[-1].content == "six seven"
    assert result[0].content != "one two three"


def test_single_oversized_message_kept() -> None:
    msgs = [_msg("a b c d e f g h")]
    result = sliding_window(msgs, max_tokens=2)
    assert len(result) == 1


def test_custom_token_fn() -> None:
    char_fn = len  # count characters
    msgs = [_msg("ab"), _msg("cd"), _msg("ef")]
    result = sliding_window(msgs, max_tokens=4, token_fn=char_fn)
    assert sum(len(m.content) for m in result) <= 4


def test_exact_budget_no_trim() -> None:
    msgs = [_msg("a b"), _msg("c d")]  # 2 + 2 = 4 tokens
    result = sliding_window(msgs, max_tokens=4)
    assert len(result) == 2
```

---

## Exercise 5

`SummarizationMemory.get_history_since(n_summaries_ago)`.

```python
"""
summarization_memory.py — Memory with get_history_since(n_summaries_ago).
"""
from __future__ import annotations

import logging
from typing import Callable

from context_manager import Message

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[list[Message]], str]


def _mock_summarize(messages: list[Message]) -> str:
    return f"[Summary: {len(messages)} messages]"


class SummarizationMemory:
    """Maintains a rolling summary history and recent verbatim messages."""

    def __init__(
        self,
        summarize_every: int = 5,
        summarize_fn: SummarizeFn = _mock_summarize,
    ) -> None:
        self._summarize_every = summarize_every
        self._summarize_fn = summarize_fn
        self._pending: list[Message] = []
        self._summaries: list[str] = []

    def add(self, message: Message) -> None:
        self._pending.append(message)
        if len(self._pending) >= self._summarize_every:
            summary = self._summarize_fn(self._pending)
            self._summaries.append(summary)
            logger.debug("Summarized %d messages into summary #%d", len(self._pending), len(self._summaries))
            self._pending = []

    def get_history_since(self, n_summaries_ago: int) -> str:
        """Return the summary that is *n_summaries_ago* steps back.

        get_history_since(0) -> latest summary
        get_history_since(1) -> second-most-recent summary
        Raises IndexError when out of range.
        """
        if not self._summaries:
            raise IndexError("No summaries available")
        idx = -(n_summaries_ago + 1)
        return self._summaries[idx]

    def summaries(self) -> list[str]:
        return list(self._summaries)

    def pending(self) -> list[Message]:
        return list(self._pending)
```

```python
"""
tests/test_summarization_memory.py — Tests for get_history_since.
"""
from __future__ import annotations

import pytest

from context_manager import Message
from summarization_memory import SummarizationMemory


def _fill(mem: SummarizationMemory, count: int) -> None:
    for i in range(count):
        mem.add(Message(role="user", content=f"message {i}"))


def test_get_latest_summary() -> None:
    mem = SummarizationMemory(summarize_every=3)
    _fill(mem, 3)
    summary = mem.get_history_since(0)
    assert "Summary" in summary


def test_get_second_most_recent_summary() -> None:
    mem = SummarizationMemory(summarize_every=2)
    _fill(mem, 6)  # 3 summaries produced
    s0 = mem.get_history_since(0)
    s1 = mem.get_history_since(1)
    assert s0 != s1


def test_index_out_of_range_raises() -> None:
    mem = SummarizationMemory(summarize_every=3)
    _fill(mem, 3)  # 1 summary
    with pytest.raises(IndexError):
        mem.get_history_since(1)  # only 1 summary, index 1 = second-to-last


def test_no_summaries_raises() -> None:
    mem = SummarizationMemory(summarize_every=5)
    _fill(mem, 2)  # not enough to trigger a summary
    with pytest.raises(IndexError):
        mem.get_history_since(0)


def test_multiple_summaries_order_preserved() -> None:
    calls: list[list[Message]] = []

    def tracking_fn(msgs: list[Message]) -> str:
        calls.append(list(msgs))
        return f"Summary-{len(calls)}"

    mem = SummarizationMemory(summarize_every=2, summarize_fn=tracking_fn)
    _fill(mem, 4)  # 2 summaries
    assert mem.get_history_since(0) == "Summary-2"
    assert mem.get_history_since(1) == "Summary-1"
```
