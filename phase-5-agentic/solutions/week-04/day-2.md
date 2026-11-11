# Solutions — Phase 5 · Week 04 · Day 2 — Short-Term Memory: Buffer & Scratchpad

---

## Exercise 1

Extend `ConversationBuffer` with a `search(keyword: str) -> list[Message]` method.

```python
"""
buffer_memory.py — ConversationBuffer with keyword search.
"""
from __future__ import annotations

import logging
from collections import deque

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ConversationBuffer:
    """Fixed-capacity FIFO message buffer with keyword search."""

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._buffer: deque[Message] = deque(maxlen=max_size)

    def add(self, message: Message) -> None:
        self._buffer.append(message)
        logger.debug("Added message from %s. Buffer size: %d", message.role, len(self._buffer))

    def messages(self) -> list[Message]:
        return list(self._buffer)

    def search(self, keyword: str) -> list[Message]:
        """Return all messages whose content contains *keyword* (case-insensitive)."""
        lower = keyword.lower()
        return [m for m in self._buffer if lower in m.content.lower()]
```

```python
"""
tests/test_buffer_search.py — Tests for ConversationBuffer.search.
"""
from __future__ import annotations

import pytest

from buffer_memory import ConversationBuffer, Message


def test_search_finds_matching_message() -> None:
    buf = ConversationBuffer()
    buf.add(Message(role="user", content="Tell me about Python"))
    buf.add(Message(role="assistant", content="Python is a great language"))
    buf.add(Message(role="user", content="What about Java?"))

    results = buf.search("python")
    assert len(results) == 2


def test_search_case_insensitive() -> None:
    buf = ConversationBuffer()
    buf.add(Message(role="user", content="HELLO WORLD"))
    results = buf.search("hello")
    assert len(results) == 1


def test_search_no_match_returns_empty() -> None:
    buf = ConversationBuffer()
    buf.add(Message(role="user", content="Nothing relevant here"))
    assert buf.search("xyz123") == []


def test_search_empty_buffer() -> None:
    buf = ConversationBuffer()
    assert buf.search("anything") == []


def test_search_returns_correct_messages() -> None:
    buf = ConversationBuffer()
    m1 = Message(role="user", content="I like LangGraph")
    m2 = Message(role="user", content="I like FastAPI")
    buf.add(m1)
    buf.add(m2)
    results = buf.search("LangGraph")
    assert len(results) == 1
    assert results[0].content == "I like LangGraph"
```

---

## Exercise 2

Implement `get_last_n(n: int) -> list[Message]`.

```python
"""
buffer_last_n.py — ConversationBuffer extended with get_last_n.
"""
from __future__ import annotations

import logging
from collections import deque

from buffer_memory import Message

logger = logging.getLogger(__name__)


class ConversationBuffer:
    """Fixed-capacity FIFO buffer with last-N retrieval."""

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._buffer: deque[Message] = deque(maxlen=max_size)

    def add(self, message: Message) -> None:
        self._buffer.append(message)

    def messages(self) -> list[Message]:
        return list(self._buffer)

    def search(self, keyword: str) -> list[Message]:
        lower = keyword.lower()
        return [m for m in self._buffer if lower in m.content.lower()]

    def get_last_n(self, n: int) -> list[Message]:
        """Return the *n* most recent messages, oldest first."""
        if n <= 0:
            return []
        all_msgs = list(self._buffer)
        return all_msgs[-n:]
```

```python
"""
tests/test_buffer_last_n.py — Tests for get_last_n.
"""
from __future__ import annotations

import pytest

from buffer_memory import Message
from buffer_last_n import ConversationBuffer


def _fill(buf: ConversationBuffer, count: int) -> None:
    for i in range(count):
        buf.add(Message(role="user", content=f"message-{i}"))


def test_get_last_n_returns_n_messages() -> None:
    buf = ConversationBuffer()
    _fill(buf, 5)
    result = buf.get_last_n(3)
    assert len(result) == 3


def test_get_last_n_returns_most_recent() -> None:
    buf = ConversationBuffer()
    _fill(buf, 5)
    result = buf.get_last_n(2)
    assert result[0].content == "message-3"
    assert result[1].content == "message-4"


def test_get_last_n_larger_than_buffer() -> None:
    buf = ConversationBuffer()
    _fill(buf, 3)
    result = buf.get_last_n(10)
    assert len(result) == 3


def test_get_last_n_zero_returns_empty() -> None:
    buf = ConversationBuffer()
    _fill(buf, 5)
    assert buf.get_last_n(0) == []


def test_get_last_n_on_empty_buffer() -> None:
    buf = ConversationBuffer()
    assert buf.get_last_n(3) == []
```

---

## Exercise 3

Priority queue variant: system messages are never evicted.

```python
"""
priority_buffer.py — Buffer that never evicts system messages.
"""
from __future__ import annotations

import logging
from collections import deque

from buffer_memory import Message

logger = logging.getLogger(__name__)


class PriorityConversationBuffer:
    """FIFO buffer that preserves system messages during eviction.

    When the buffer is full and a non-system message is added, the oldest
    non-system message is dropped. If all messages are system messages the
    buffer grows beyond max_size (system messages cannot be evicted).
    """

    def __init__(self, max_size: int = 5) -> None:
        self.max_size = max_size
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        if len(self._messages) >= self.max_size:
            self._evict_oldest_non_system()
        self._messages.append(message)
        logger.debug(
            "Added %s message. Buffer length: %d", message.role, len(self._messages)
        )

    def _evict_oldest_non_system(self) -> None:
        for i, msg in enumerate(self._messages):
            if msg.role != "system":
                logger.debug("Evicting message at index %d: %s", i, msg.content[:40])
                self._messages.pop(i)
                return
        # All messages are system messages — allow buffer to grow
        logger.warning("All buffered messages are system messages; cannot evict.")

    def messages(self) -> list[Message]:
        return list(self._messages)
```

```python
"""
tests/test_priority_buffer.py — Tests for PriorityConversationBuffer eviction.
"""
from __future__ import annotations

import pytest

from buffer_memory import Message
from priority_buffer import PriorityConversationBuffer


def test_system_messages_not_evicted() -> None:
    buf = PriorityConversationBuffer(max_size=3)
    buf.add(Message(role="system", content="You are a helpful assistant"))
    buf.add(Message(role="user", content="Hello"))
    buf.add(Message(role="assistant", content="Hi there"))
    # Buffer full; next add should evict oldest non-system (user "Hello")
    buf.add(Message(role="user", content="What's 2+2?"))

    roles = [m.role for m in buf.messages()]
    contents = [m.content for m in buf.messages()]
    assert "You are a helpful assistant" in contents
    assert "Hello" not in contents


def test_non_system_evicted_first() -> None:
    buf = PriorityConversationBuffer(max_size=2)
    buf.add(Message(role="system", content="System prompt"))
    buf.add(Message(role="user", content="User msg"))
    buf.add(Message(role="assistant", content="Assistant reply"))

    msgs = buf.messages()
    roles = [m.role for m in msgs]
    assert "system" in roles
    assert len(msgs) == 2


def test_only_system_messages_buffer_grows() -> None:
    buf = PriorityConversationBuffer(max_size=2)
    buf.add(Message(role="system", content="Rule 1"))
    buf.add(Message(role="system", content="Rule 2"))
    buf.add(Message(role="system", content="Rule 3"))
    # All system — cannot evict, buffer allowed to grow
    assert len(buf.messages()) == 3


def test_normal_eviction_without_system() -> None:
    buf = PriorityConversationBuffer(max_size=2)
    buf.add(Message(role="user", content="A"))
    buf.add(Message(role="user", content="B"))
    buf.add(Message(role="user", content="C"))
    contents = [m.content for m in buf.messages()]
    assert "A" not in contents
    assert "B" in contents
    assert "C" in contents
```

---

## Exercise 4

`ScratchpadMemory.clear_step(key)` and `to_summary()`.

```python
"""
scratchpad_memory.py — Scratchpad with step deletion and summary formatting.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ScratchpadMemory:
    """Key-ordered reasoning steps with deletion and summary output."""

    def __init__(self) -> None:
        # Use a list to preserve insertion order for numbering
        self._steps: dict[str, str] = {}

    def add_step(self, key: str, thought: str) -> None:
        self._steps[key] = thought
        logger.debug("Added scratchpad step: %s", key)

    def clear_step(self, key: str) -> None:
        """Delete reasoning step identified by *key*.

        Silently ignores missing keys.
        """
        removed = self._steps.pop(key, None)
        if removed is not None:
            logger.debug("Cleared scratchpad step: %s", key)
        else:
            logger.debug("clear_step called for unknown key: %s", key)

    def to_summary(self) -> str:
        """Format all steps as a numbered list string."""
        if not self._steps:
            return ""
        lines = [
            f"{i + 1}. [{key}] {thought}"
            for i, (key, thought) in enumerate(self._steps.items())
        ]
        return "\n".join(lines)

    def steps(self) -> dict[str, str]:
        return dict(self._steps)
```

```python
"""
tests/test_scratchpad_memory.py — Tests for ScratchpadMemory.
"""
from __future__ import annotations

import pytest

from scratchpad_memory import ScratchpadMemory


def test_clear_step_removes_entry() -> None:
    pad = ScratchpadMemory()
    pad.add_step("step1", "Identify the problem")
    pad.add_step("step2", "Gather data")
    pad.clear_step("step1")
    assert "step1" not in pad.steps()
    assert "step2" in pad.steps()


def test_clear_step_nonexistent_key_silent() -> None:
    pad = ScratchpadMemory()
    pad.clear_step("does_not_exist")  # must not raise


def test_to_summary_numbered_list() -> None:
    pad = ScratchpadMemory()
    pad.add_step("plan", "Form a plan")
    pad.add_step("execute", "Execute the plan")
    summary = pad.to_summary()
    assert "1." in summary
    assert "2." in summary
    assert "plan" in summary
    assert "execute" in summary


def test_to_summary_empty_returns_empty_string() -> None:
    pad = ScratchpadMemory()
    assert pad.to_summary() == ""


def test_to_summary_after_clear() -> None:
    pad = ScratchpadMemory()
    pad.add_step("a", "Alpha")
    pad.add_step("b", "Beta")
    pad.clear_step("a")
    summary = pad.to_summary()
    assert "[a]" not in summary
    assert "[b]" in summary
    assert "1." in summary
```

---

## Exercise 5

`TokenAwareScratchpad` that limits total token budget (word-count proxy), evicting oldest.

```python
"""
token_aware_scratchpad.py — Scratchpad with token budget and eviction.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    """Approximate token count using word splitting as proxy."""
    return len(text.split())


class TokenAwareScratchpad:
    """Scratchpad that evicts oldest entries when token budget is exceeded.

    Token count is approximated as len(text.split()).
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        # OrderedDict preserves insertion order for FIFO eviction
        self._steps: OrderedDict[str, str] = OrderedDict()

    def _total_tokens(self) -> int:
        return sum(_count_tokens(v) for v in self._steps.values())

    def add_step(self, key: str, thought: str) -> None:
        """Add a reasoning step, evicting oldest entries if over budget."""
        # Update existing key first to avoid double-counting
        if key in self._steps:
            self._steps[key] = thought
            self._steps.move_to_end(key)
        else:
            self._steps[key] = thought

        logger.debug(
            "Added step %s (%d tokens). Total: %d / %d",
            key, _count_tokens(thought), self._total_tokens(), self.max_tokens,
        )

        while self._total_tokens() > self.max_tokens and len(self._steps) > 1:
            evicted_key, evicted_val = self._steps.popitem(last=False)
            logger.debug(
                "Evicted step %s (%d tokens) to stay within budget",
                evicted_key, _count_tokens(evicted_val),
            )

    def steps(self) -> dict[str, str]:
        return dict(self._steps)

    def total_tokens(self) -> int:
        return self._total_tokens()
```

```python
"""
tests/test_token_aware_scratchpad.py — Tests for TokenAwareScratchpad eviction.
"""
from __future__ import annotations

import pytest

from token_aware_scratchpad import TokenAwareScratchpad


def test_add_within_budget_no_eviction() -> None:
    pad = TokenAwareScratchpad(max_tokens=20)
    pad.add_step("a", "one two three")  # 3 tokens
    pad.add_step("b", "four five six")  # 3 tokens
    assert len(pad.steps()) == 2


def test_evicts_oldest_when_over_budget() -> None:
    pad = TokenAwareScratchpad(max_tokens=6)
    pad.add_step("step1", "alpha beta gamma")  # 3 tokens
    pad.add_step("step2", "delta epsilon zeta")  # 3 tokens
    # Budget is exactly 6; adding step3 triggers eviction of step1
    pad.add_step("step3", "eta theta iota")  # 3 tokens
    assert "step1" not in pad.steps()
    assert "step2" in pad.steps()
    assert "step3" in pad.steps()


def test_last_step_never_evicted() -> None:
    """Even a single step larger than budget is kept (can't go to zero)."""
    pad = TokenAwareScratchpad(max_tokens=2)
    pad.add_step("big", "one two three four five")
    assert "big" in pad.steps()
    assert len(pad.steps()) == 1


def test_total_tokens_accurate() -> None:
    pad = TokenAwareScratchpad(max_tokens=100)
    pad.add_step("a", "hello world")  # 2 tokens
    pad.add_step("b", "foo bar baz")  # 3 tokens
    assert pad.total_tokens() == 5


def test_update_existing_key_does_not_double_count() -> None:
    pad = TokenAwareScratchpad(max_tokens=10)
    pad.add_step("key", "one two")  # 2 tokens
    pad.add_step("key", "one two three four")  # 4 tokens — replaces old
    assert pad.total_tokens() == 4
```
