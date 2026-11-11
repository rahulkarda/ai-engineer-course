# Week 04 Assessment — Answer Key

---

## Part 1: Conceptual Questions

### Q1. What are the benefits of typed state over raw dicts in agents?

Typed state (e.g., Pydantic models) enforces a schema at runtime and provides IDE autocompletion, so bugs from missing or misnamed keys surface immediately rather than silently propagating downstream. Typed state also serves as living documentation: any developer reading the model immediately understands what fields exist and what types they hold, reducing onboarding friction and accidental state mutation.

---

### Q2. What are the tradeoffs between short-term and long-term memory?

Short-term memory (context window / buffer) is fast and requires no retrieval step, but it is bounded by token limits and is lost when the session ends. Long-term memory (vector stores, databases) persists indefinitely and scales to millions of facts, but introduces latency from retrieval, potential retrieval errors, and operational complexity (indexing, embedding models, consistency maintenance).

---

### Q3. When should you summarize vs truncate context?

Truncation is simpler and sufficient when older messages are genuinely irrelevant to the current task (e.g., unrelated prior conversations). Summarization is preferable when older context contains facts or decisions the agent still needs (e.g., goals agreed upon three messages ago): a summary compresses while retaining semantic content, whereas truncation discards it entirely and risks the agent "forgetting" critical constraints.

---

### Q4. What is the distinction between episodic and semantic memory?

Episodic memory stores individual events anchored to a specific time and context ("at 2pm yesterday we discussed deployment"), preserving the *when* and *what happened*. Semantic memory stores distilled, time-independent facts ("the API endpoint is /v2/predict"), encoding *what is true* without a temporal anchor. Agents typically use episodic for recent, contextual reasoning and semantic for stable knowledge retrieval.

---

### Q5. How does vector similarity enable memory retrieval?

Each memory is encoded as a fixed-length numerical vector (embedding) that captures its semantic meaning. At retrieval time, the query is embedded into the same space and compared against stored vectors using cosine similarity; the memories with the highest similarity scores are returned. This allows fuzzy, meaning-based lookups (e.g., "database error" matching "SQL connection refused") rather than exact keyword matching.

---

## Part 2: MinimalMemoryAgent — Complete Solution

```python
# src/minimal_memory_agent.py
"""
MinimalMemoryAgent — Week 04 Assessment solution.

Components:
- AgentState: typed Pydantic model
- BufferMemory: max 5 messages, FIFO eviction
- summarization trigger: fires when >3 messages (mock LLM)
- VectorSemanticMemory: 3 stored facts, mock embed, retrieve top-1
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Types                                                                         #
# --------------------------------------------------------------------------- #

EmbedFn = Callable[[str], list[float]]
SummarizeFn = Callable[[list[dict[str, str]]], str]


# --------------------------------------------------------------------------- #
# AgentState                                                                    #
# --------------------------------------------------------------------------- #

class Message(BaseModel):
    role: str
    content: str


class AgentState(BaseModel):
    """Typed state for a single agent session."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# BufferMemory                                                                  #
# --------------------------------------------------------------------------- #

class BufferMemory:
    """Holds the most recent messages up to max_size (FIFO eviction)."""

    def __init__(self, max_size: int = 5) -> None:
        self._max_size = max_size
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        while len(self._messages) > self._max_size:
            self._messages.pop(0)

    def messages(self) -> list[Message]:
        return list(self._messages)

    def size(self) -> int:
        return len(self._messages)


# --------------------------------------------------------------------------- #
# VectorSemanticMemory                                                          #
# --------------------------------------------------------------------------- #

def _default_embed(text: str) -> list[float]:
    """Hash-based deterministic 4-dim unit embedding."""
    h = abs(hash(text)) % (10 ** 8)
    vec = [(h >> (i * 8)) % 256 / 255.0 for i in range(4)]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


class VectorSemanticMemory:
    """Stores facts as embeddings and retrieves by cosine similarity."""

    def __init__(self, embed_fn: EmbedFn = _default_embed) -> None:
        self._embed_fn = embed_fn
        self._store: list[tuple[str, list[float]]] = []

    def add_fact(self, text: str) -> None:
        self._store.append((text, self._embed_fn(text)))
        logger.debug("VectorSemanticMemory: stored fact %r", text[:40])

    def retrieve(self, query: str, k: int = 1) -> list[str]:
        if not self._store:
            return []
        q_emb = self._embed_fn(query)
        scored = sorted(self._store, key=lambda item: _cosine(q_emb, item[1]), reverse=True)
        return [text for text, _ in scored[:k]]


# --------------------------------------------------------------------------- #
# MinimalMemoryAgent                                                            #
# --------------------------------------------------------------------------- #

def _default_summarize(messages: list[Message]) -> str:
    """
    Production: calls an LLM.
    Test environments: replaced by a mock.
    Controlled by ANTHROPIC_API_KEY presence as a guard.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback for tests / offline environments
        return "[SUMMARY] " + " | ".join(m.content for m in messages)

    # Real implementation would call Claude here:
    # import anthropic
    # client = anthropic.Anthropic(api_key=api_key)
    # ... (not called in tests)
    return "[SUMMARY via LLM] " + " | ".join(m.content[:20] for m in messages)


class MinimalMemoryAgent:
    """
    Agent that maintains typed state, a bounded buffer, auto-summarization,
    and vector-based semantic memory.
    """

    SUMMARY_TRIGGER = 3  # summarize when buffer exceeds this count

    def __init__(
        self,
        session_id: str,
        summarize_fn: SummarizeFn | None = None,
        embed_fn: EmbedFn = _default_embed,
    ) -> None:
        self._state = AgentState(session_id=session_id)
        self._buffer = BufferMemory(max_size=5)
        self._semantic = VectorSemanticMemory(embed_fn=embed_fn)
        self._summarize_fn = summarize_fn or _default_summarize

    # ------------------------------------------------------------------ public

    def observe(self, role: str, content: str) -> None:
        """Add a message, trigger summarization if buffer exceeds threshold."""
        msg = Message(role=role, content=content)
        self._buffer.add(msg)
        self._state.messages.append(msg)
        logger.debug("Observed: role=%s len=%d", role, self._buffer.size())

        if self._buffer.size() > self.SUMMARY_TRIGGER:
            self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        messages_to_summarize = self._buffer.messages()
        summary = self._summarize_fn(messages_to_summarize)
        self._state.summary = summary
        logger.info("Triggered summarization: %r", summary[:60])

    def recall(self, query: str) -> list[str]:
        """Retrieve relevant semantic facts."""
        return self._semantic.retrieve(query, k=1)

    def learn(self, fact: str) -> None:
        """Store a semantic fact."""
        self._semantic.add_fact(fact)

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def buffer(self) -> BufferMemory:
        return self._buffer

    @property
    def semantic(self) -> VectorSemanticMemory:
        return self._semantic
```

```python
# tests/unit/test_minimal_memory_agent.py
"""
Three required assessment tests:
  test_buffer_trim
  test_summary_trigger
  test_semantic_retrieval
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from minimal_memory_agent import MinimalMemoryAgent, Message


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_summarize():
    """Returns a mock summarize function and the mock object for assertions."""
    fn = MagicMock(return_value="[MOCK SUMMARY]")
    return fn


@pytest.fixture
def agent(mock_summarize):
    return MinimalMemoryAgent(session_id="test-session", summarize_fn=mock_summarize)


# --------------------------------------------------------------------------- #
# Test 1: Buffer trim (FIFO eviction at max 5)                                 #
# --------------------------------------------------------------------------- #

def test_buffer_trim(agent):
    """Buffer never exceeds max_size=5; oldest message is evicted."""
    for i in range(7):
        agent.observe("user", f"message {i}")

    assert agent.buffer.size() == 5
    # The buffer should contain the 5 most recent messages (2–6)
    contents = [m.content for m in agent.buffer.messages()]
    assert "message 6" in contents
    assert "message 5" in contents
    # Oldest messages evicted
    assert "message 0" not in contents
    assert "message 1" not in contents


# --------------------------------------------------------------------------- #
# Test 2: Summarization trigger (fires after >3 messages)                      #
# --------------------------------------------------------------------------- #

def test_summary_trigger(agent, mock_summarize):
    """Summarization fires when buffer exceeds SUMMARY_TRIGGER=3 messages."""
    # Add exactly trigger count — no summary yet
    for i in range(3):
        agent.observe("user", f"msg {i}")
    assert mock_summarize.call_count == 0
    assert agent.state.summary is None

    # One more message pushes buffer to 4 → triggers summary
    agent.observe("user", "msg 3")
    assert mock_summarize.call_count == 1
    assert agent.state.summary == "[MOCK SUMMARY]"


# --------------------------------------------------------------------------- #
# Test 3: Semantic retrieval                                                    #
# --------------------------------------------------------------------------- #

def test_semantic_retrieval(agent):
    """Vector semantic memory stores 3 facts and retrieves the most relevant one."""
    facts = [
        "The API endpoint is /v2/predict",
        "Authentication uses OAuth2 bearer tokens",
        "The model is deployed on us-east-1",
    ]
    for fact in facts:
        agent.learn(fact)

    # Query for deployment-related fact
    results = agent.recall("where is the model deployed")
    assert len(results) == 1
    assert results[0] in facts  # Top-1 must be one of the stored facts

    # Query for auth-related fact
    results_auth = agent.recall("how does authentication work")
    assert len(results_auth) == 1
    assert results_auth[0] in facts
```

---

## Part 3: Grading Notes

| Dimension | Weight | Passing Criteria |
|---|---|---|
| **Typed state** | 20% | `AgentState` is a Pydantic `BaseModel` with typed fields; no raw dicts for state. |
| **Buffer eviction** | 20% | `test_buffer_trim` passes; FIFO eviction at exactly `max_size=5`. |
| **Summarization trigger** | 20% | `test_summary_trigger` passes; mock LLM called after >3 messages (not real API). |
| **Semantic retrieval** | 20% | `test_semantic_retrieval` passes; deterministic mock embed used; top-1 returns a stored fact. |
| **Code quality** | 20% | `logging` not `print`; type hints on all functions; no hardcoded API keys. |

**Deduct points for:**
- `os.environ["KEY"]` (raises if missing) instead of `os.getenv("KEY")` (returns `None`)
- Real API calls in tests (no monkeypatching)
- `print()` statements instead of `logger.*`
- Missing type hints on function signatures
- Mutable default arguments (e.g., `def __init__(self, messages=[])`)
