# Solutions: Lesson 3 — Writing & Consolidating Memories

---

## Exercise 1 — `KeywordFilterSummariser`

```python
from dataclasses import dataclass, field
from lesson_3_write_pipeline import Message


@dataclass
class KeywordFilterSummariser:
    keywords: list[str] = field(default_factory=list)
    discard_keywords: list[str] = field(default_factory=list)

    def summarise(self, messages: list[Message]) -> str | None:
        combined = " ".join(m.content for m in messages).lower()

        # Discard keywords take priority
        if any(kw in combined for kw in self.discard_keywords):
            return None

        # Must match at least one keep keyword
        if any(kw in combined for kw in self.keywords):
            user_msgs = [m.content for m in messages if m.role == "user"]
            return user_msgs[-1] if user_msgs else None

        return None
```

Test:

```python
def test_discard_overrides_keep():
    s = KeywordFilterSummariser(
        keywords=["prefer"],
        discard_keywords=["spam"],
    )
    msgs = [Message("user", "I prefer spam emails")]
    assert s.summarise(msgs) is None  # "spam" triggers discard

def test_keep_without_discard():
    s = KeywordFilterSummariser(keywords=["prefer"])
    msgs = [Message("user", "I prefer Python")]
    assert s.summarise(msgs) == "I prefer Python"

def test_no_match():
    s = KeywordFilterSummariser(keywords=["prefer"])
    msgs = [Message("user", "Hello there")]
    assert s.summarise(msgs) is None
```

---

## Exercise 2 — `_merge_semantic_duplicate`

```python
MERGE_THRESHOLD = 0.96

def process_turn(self, messages: list[Message]) -> MemoryRecord | None:
    summary = self._summariser.summarise(messages)
    if summary is None:
        return None

    candidate = MemoryCandidate(content=summary)
    vector = self._embedder.embed(candidate.content)
    content_hash = hashlib.sha256(candidate.content.encode()).hexdigest()

    if self._is_exact_duplicate(content_hash):
        return None

    # Check for merge-level duplicate (>0.96)
    merged = self._merge_semantic_duplicate(vector, summary, vector, content_hash)
    if merged is not None:
        return merged

    existing_count = self._count_semantic_duplicates(vector, candidate.content)
    importance = score_importance(candidate, existing_count)

    record = MemoryRecord(
        id=uuid4(),
        content=candidate.content,
        vector=vector,
        importance=importance,
        created_at=datetime.now(timezone.utc),
        content_hash=content_hash,
    )
    self._store.append(record)
    return record


def _merge_semantic_duplicate(
    self,
    vector: list[float],
    new_content: str,
    new_vector: list[float],
    new_hash: str,
) -> MemoryRecord | None:
    for existing in self._store:
        sim = cosine_sim(vector, existing.vector)
        if sim >= self.MERGE_THRESHOLD and existing.content != new_content:
            log.info("Merging near-identical record id=%s", existing.id)
            # Overwrite content, keep id and created_at
            existing.content = new_content
            existing.vector = new_vector
            existing.content_hash = new_hash
            return existing
    return None
```

---

## Exercise 3 — Long-message importance boost

```python
def score_importance(candidate: MemoryCandidate, existing_count: int = 0) -> float:
    score = 0.25
    text = candidate.content.lower()
    if any(kw in text for kw in ["prefer", "always", "never", "must", "required"]):
        score += 0.35
    if existing_count > 0:
        score += min(0.25, existing_count * 0.1)
    score += min(0.15, len(candidate.content) / 500)
    # Long message boost
    if len(candidate.content) > 100:
        score += 0.10
    return min(1.0, score)
```

Parametrize test:

```python
import pytest
from lesson_3_write_pipeline import MemoryCandidate, score_importance


@pytest.mark.parametrize("content,expected_boost", [
    ("Short", False),
    ("I prefer " + "x" * 95, True),  # >100 chars
])
def test_long_message_boost(content: str, expected_boost: bool):
    short_candidate = MemoryCandidate(content="Short")
    long_candidate = MemoryCandidate(content=content)

    short_score = score_importance(short_candidate)
    long_score = score_importance(long_candidate)

    if expected_boost:
        assert long_score > short_score
    else:
        assert long_score == short_score
```

---

## Exercise 4 — Replacing `MockSummariser` with a real Anthropic call

Use `claude-haiku-5-5` (cheapest, fastest):

```python
import os
import anthropic

class AnthropicSummariser:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def summarise(self, messages: list[Message]) -> str | None:
        combined = "\n".join(f"{m.role}: {m.content}" for m in messages)
        response = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=100,
            system=(
                "Extract a single concise fact worth remembering from this conversation. "
                "If there is nothing worth storing, reply with exactly: SKIP"
            ),
            messages=[{"role": "user", "content": combined}],
        )
        result = response.content[0].text.strip()
        return None if result == "SKIP" else result
```

Rate limit handling: wrap the call in a retry loop with exponential backoff, or
use the Anthropic SDK's built-in retry (`max_retries` parameter). For high-throughput
scenarios, batch multiple turns into a single call, or use an async client with a
token-bucket rate limiter.
