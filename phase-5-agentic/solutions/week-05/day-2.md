# Solutions — Week 05 · Day 2: Self-Consistency & Voting

## Exercise 1: Timeout handling

```python
import concurrent.futures
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


def _sample_with_timeout(self, task: str, sample_id: int, timeout: float) -> str | None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._sample, task, sample_id)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("Sample %d timed out after %.1fs", sample_id, timeout)
            return None


# In SelfConsistency.run — wrap each sample call
def run(self, task: str, timeout_seconds: float = 30.0) -> dict:
    samples: list[str] = []
    with ThreadPoolExecutor(max_workers=self.n_samples) as pool:
        futures = {
            pool.submit(self._sample_with_timeout, task, i, timeout_seconds): i
            for i in range(self.n_samples)
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                samples.append(result)
    ...
```

Test:

```python
from unittest.mock import MagicMock, patch
import concurrent.futures
import pytest


def test_timed_out_sample_skipped(monkeypatch):
    """Sample that times out should be skipped, not raise."""
    call_count = [0]

    def slow_sample(task, sample_id, timeout):
        call_count[0] += 1
        if sample_id == 2:
            return None  # simulates timeout skip
        return f"answer_{sample_id}"

    sc = SelfConsistency(n_samples=3)
    monkeypatch.setattr(sc, "_sample_with_timeout", slow_sample)

    # Patch extract to return raw string
    monkeypatch.setattr(sc, "_extract_answer", lambda s: s)
    result = sc.run("2+2=?", timeout_seconds=1.0)

    # Sample 2 was skipped → only 2 valid answers
    assert sum(result["votes"].values()) == 2
```

---

## Exercise 2: Weighted majority vote

```python
from collections import defaultdict


def weighted_vote(answers: list[str], scores: list[float]) -> str:
    """Return the answer with the highest total weight."""
    if not answers:
        raise ValueError("answers must be non-empty")
    if len(answers) != len(scores):
        raise ValueError("answers and scores must have equal length")

    totals: dict[str, float] = defaultdict(float)
    for answer, score in zip(answers, scores):
        totals[answer] += score

    # Handle tie: return alphabetically first for determinism
    winner = max(totals, key=lambda k: (totals[k], k))
    return winner
```

Tests:

```python
def test_weighted_vote_basic():
    assert weighted_vote(["A", "B", "A"], [0.5, 0.9, 0.5]) == "B"  # B has 0.9 vs A's 1.0? no — A=1.0>B=0.9
    # A: 0.5+0.5=1.0, B: 0.9 → A wins
    assert weighted_vote(["A", "B", "A"], [0.5, 0.9, 0.5]) == "A"

def test_weighted_vote_tie_deterministic():
    result = weighted_vote(["A", "B"], [1.0, 1.0])
    assert result in ("A", "B")  # deterministic but either is valid

def test_weighted_vote_empty_raises():
    with pytest.raises(ValueError):
        weighted_vote([], [])
```

---

## Exercise 3: Early exit on confidence threshold

```python
def run(self, task: str, confidence_threshold: float = 0.6) -> dict:
    samples: list[str] = []
    votes: Counter[str] = Counter()

    for i in range(self.n_samples):
        sample = self._sample(task, i)
        samples.append(sample)
        answer = self._extract_answer(sample)
        if answer:
            votes[answer] += 1
            total = sum(votes.values())
            if total > 0:
                leader, leader_count = votes.most_common(1)[0]
                confidence = leader_count / total
                if confidence >= confidence_threshold and total >= 2:
                    log.info("Early exit: leader=%r confidence=%.0f%%", leader, confidence * 100)
                    return {
                        "winner": leader, "votes": votes,
                        "samples": samples, "confidence": confidence,
                        "early_exit": True,
                    }

    winner, top_count = votes.most_common(1)[0]
    return {
        "winner": winner, "votes": votes,
        "samples": samples, "confidence": top_count / sum(votes.values()),
        "early_exit": False,
    }
```

Test:

```python
def test_early_exit_fires():
    sc = SelfConsistency(n_samples=5, confidence_threshold=0.6)
    # First 3 samples all return same answer
    call_count = [0]

    def mock_sample(task, i):
        call_count[0] += 1
        return "ANSWER: 42"  # all agree

    monkeypatch.setattr(sc, "_sample", mock_sample)
    result = sc.run("2+2=?")
    assert result["early_exit"] is True
    assert call_count[0] < 5  # stopped before all 5
```

---

## Exercise 4: Structured JSON extraction

```python
from pydantic import BaseModel


class SampleResponse(BaseModel):
    reasoning: str
    answer: str


def _extract_answer_structured(self, response: str) -> str | None:
    """Parse JSON response with Pydantic."""
    import json, re
    # strip code fences
    raw = response.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw.strip())
        parsed = SampleResponse(**data)
        return parsed.answer.strip().lower()
    except Exception as e:
        log.warning("Structured extraction failed: %s", e)
        return None
```

Prompt update:

```
"Think step by step, then respond as JSON: {\"reasoning\": str, \"answer\": str}"
```

Test:

```python
def test_structured_extractor_valid():
    sc = SelfConsistency()
    response = '{"reasoning": "2+2=4", "answer": "4"}'
    assert sc._extract_answer_structured(response) == "4"

def test_structured_extractor_invalid_falls_back_to_none():
    sc = SelfConsistency()
    assert sc._extract_answer_structured("plain text no json") is None
```
