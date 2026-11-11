# Solutions — Phase 4 · Week 07 · Day 6

## Exercise 1: Evaluation Profiles

Load named threshold configs from JSON. A profile holds per-metric pass/fail thresholds so the
same harness can enforce "strict" CI gates and "lenient" exploratory runs without changing code.

### Config file: `eval_profiles.json`

```json
{
  "profiles": {
    "strict": {
      "faithfulness": 0.85,
      "answer_relevance": 0.80,
      "context_precision": 0.75,
      "context_recall": 0.75,
      "context_utilization": 0.60
    },
    "lenient": {
      "faithfulness": 0.60,
      "answer_relevance": 0.55,
      "context_precision": 0.50,
      "context_recall": 0.50,
      "context_utilization": 0.30
    },
    "default": {
      "faithfulness": 0.70,
      "answer_relevance": 0.70,
      "context_precision": 0.60,
      "context_recall": 0.60,
      "context_utilization": 0.40
    }
  }
}
```

### `eval_profile.py`

```python
"""Evaluation profile loading and threshold checking."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class EvalProfile(BaseModel):
    """Named evaluation profile with per-metric thresholds."""

    name: str
    thresholds: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "EvalProfile":
        for metric, value in self.thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Threshold for '{metric}' must be in [0, 1], got {value}"
                )
        return self

    def passes(self, metric: str, score: float) -> bool:
        """Return True if score meets or exceeds this profile's threshold."""
        threshold = self.thresholds.get(metric)
        if threshold is None:
            logger.warning("No threshold defined for metric '%s' in profile '%s'", metric, self.name)
            return True
        result = score >= threshold
        if not result:
            logger.debug(
                "Metric '%s' score %.3f below threshold %.3f (profile=%s)",
                metric, score, threshold, self.name,
            )
        return result

    def check_all(self, scores: dict[str, float]) -> dict[str, bool]:
        """Return pass/fail map for every metric in scores."""
        return {metric: self.passes(metric, score) for metric, score in scores.items()}


class ProfileRegistry:
    """Load and retrieve EvalProfile instances from a JSON config file."""

    def __init__(self, config_path: Path | str) -> None:
        self._config_path = Path(config_path)
        self._profiles: dict[str, EvalProfile] = {}
        self._load()

    def _load(self) -> None:
        logger.info("Loading eval profiles from %s", self._config_path)
        with self._config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for name, thresholds in raw.get("profiles", {}).items():
            self._profiles[name] = EvalProfile(name=name, thresholds=thresholds)
        logger.info("Loaded %d profile(s): %s", len(self._profiles), list(self._profiles))

    def get(self, name: str) -> EvalProfile:
        if name not in self._profiles:
            available = list(self._profiles)
            raise KeyError(f"Profile '{name}' not found. Available: {available}")
        return self._profiles[name]

    def names(self) -> list[str]:
        return list(self._profiles)
```

### Integration into `RAGEvalHarness`

```python
# harness.py  (excerpt — add profile support)
from eval_profile import EvalProfile, ProfileRegistry

class RAGEvalHarness:
    def __init__(
        self,
        rag_system,
        golden_dataset,
        profile: EvalProfile | None = None,
    ) -> None:
        self._rag = rag_system
        self._dataset = golden_dataset
        self._profile = profile

    def run(self, use_llm_judge: bool = True):
        # ... compute scores as usual ...
        scores = {"faithfulness": 0.88, "answer_relevance": 0.72}

        passed: dict[str, bool] = {}
        if self._profile is not None:
            passed = self._profile.check_all(scores)
            logger.info("Profile '%s' gate: %s", self._profile.name, passed)

        return scores, passed


# Usage
registry = ProfileRegistry("eval_profiles.json")
profile = registry.get("strict")

harness = RAGEvalHarness(rag_system=my_rag, golden_dataset=dataset, profile=profile)
scores, passed = harness.run()
```

### Tests (`tests/unit/test_eval_profile.py`)

```python
"""Unit tests for EvalProfile and ProfileRegistry."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from eval_profile import EvalProfile, ProfileRegistry


def test_profile_passes_at_threshold():
    profile = EvalProfile(name="strict", thresholds={"faithfulness": 0.85})
    assert profile.passes("faithfulness", 0.85) is True
    assert profile.passes("faithfulness", 0.84) is False


def test_profile_unknown_metric_passes():
    profile = EvalProfile(name="lenient", thresholds={})
    assert profile.passes("nonexistent_metric", 0.0) is True


def test_profile_threshold_out_of_range():
    with pytest.raises(ValueError, match="must be in"):
        EvalProfile(name="bad", thresholds={"faithfulness": 1.5})


def test_check_all_returns_per_metric_booleans():
    profile = EvalProfile(
        name="default",
        thresholds={"faithfulness": 0.70, "answer_relevance": 0.70},
    )
    result = profile.check_all({"faithfulness": 0.80, "answer_relevance": 0.60})
    assert result == {"faithfulness": True, "answer_relevance": False}


def test_registry_loads_profiles(tmp_path: Path):
    config = {
        "profiles": {
            "strict": {"faithfulness": 0.85},
            "lenient": {"faithfulness": 0.50},
        }
    }
    cfg_file = tmp_path / "profiles.json"
    cfg_file.write_text(json.dumps(config))

    registry = ProfileRegistry(cfg_file)
    assert set(registry.names()) == {"strict", "lenient"}
    assert registry.get("strict").thresholds["faithfulness"] == 0.85


def test_registry_unknown_profile_raises(tmp_path: Path):
    cfg_file = tmp_path / "profiles.json"
    cfg_file.write_text(json.dumps({"profiles": {"default": {}}}))
    registry = ProfileRegistry(cfg_file)
    with pytest.raises(KeyError, match="not found"):
        registry.get("missing")
```

---

## Exercise 2: Context Utilization Metric

Context utilization measures how many of the retrieved documents actually contributed tokens to
the answer, without calling an LLM. A document "contributes" if at least one of its tokens
appears in the final answer (case-insensitive, stop-words excluded).

```python
"""context_utilization.py — token-overlap-based metric."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Common English stop words to exclude from overlap check.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "this", "that", "these",
        "those", "it", "its", "not", "as", "from", "into", "than", "then",
        "so", "if", "about", "which", "who", "what", "when", "where", "how",
    }
)


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase word tokens, stop-words removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def context_utilization(answer: str, retrieved_docs: list[str]) -> float:
    """Return fraction of retrieved docs that contributed >= 1 token to answer.

    Score is 0.0 when no docs are retrieved. A document is considered
    "contributing" if its token set intersects the answer token set after
    stop-word removal.

    Args:
        answer: The generated answer string.
        retrieved_docs: Ordered list of document strings passed to the LLM.

    Returns:
        Float in [0, 1].
    """
    if not retrieved_docs:
        logger.warning("context_utilization called with empty retrieved_docs")
        return 0.0

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        logger.warning("Answer produced no content tokens; utilization=0.0")
        return 0.0

    contributing = 0
    for i, doc in enumerate(retrieved_docs):
        doc_tokens = _tokenize(doc)
        overlap = doc_tokens & answer_tokens
        if overlap:
            contributing += 1
            logger.debug("Doc %d contributes %d token(s): %s", i, len(overlap), overlap)
        else:
            logger.debug("Doc %d contributes no tokens", i)

    score = contributing / len(retrieved_docs)
    logger.info(
        "context_utilization: %d/%d docs contributed -> %.3f",
        contributing, len(retrieved_docs), score,
    )
    return score
```

### Integration into harness

```python
# In RAGEvalHarness.run():
from context_utilization import context_utilization

answer, docs = self._rag(question)
cu_score = context_utilization(answer, docs)
retrieval_scores["context_utilization"] = cu_score
```

### Tests (`tests/unit/test_context_utilization.py`)

```python
"""Tests for context_utilization metric."""
from context_utilization import context_utilization


def test_all_docs_contribute():
    answer = "The capital of France is Paris and it has the Eiffel Tower."
    docs = [
        "Paris is the capital city of France.",
        "The Eiffel Tower is a famous landmark.",
    ]
    score = context_utilization(answer, docs)
    assert score == 1.0


def test_no_docs_contribute():
    answer = "sunny weather today"
    docs = ["Quantum mechanics describes subatomic particles."]
    score = context_utilization(answer, docs)
    assert score == 0.0


def test_partial_contribution():
    answer = "Python is popular for machine learning."
    docs = [
        "Python programming language is widely used.",  # contributes: python, programming, language, widely, used
        "Java enterprise applications are common.",     # no overlap
    ]
    score = context_utilization(answer, docs)
    assert score == 0.5


def test_empty_docs_returns_zero():
    assert context_utilization("some answer", []) == 0.0


def test_empty_answer_returns_zero():
    assert context_utilization("", ["some document text"]) == 0.0


def test_stop_words_not_counted():
    # Answer is only stop words; no content tokens -> 0.0
    answer = "the and or but"
    docs = ["the document is here"]
    assert context_utilization(answer, docs) == 0.0
```

---

## Exercise 3: Base64 Detection

A retrieved document that contains long base64-encoded blobs may be attempting to exfiltrate
data or encode malicious instructions. The regex heuristic matches 40+ character base64 strings.

```python
"""security_scanner.py — extend with base64 detection."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Heuristic: base64 payload of >= 40 chars (padding optional).
_BASE64_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# Existing injection pattern (kept for context).
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"disregard (?:all )?(?:previous|prior) (?:instructions|context)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new persona", re.IGNORECASE),
]


@dataclass
class SecurityAlert:
    doc_index: int
    alert_type: str
    detail: str
    snippet: str = ""


@dataclass
class ScanResult:
    alerts: list[SecurityAlert] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.alerts) == 0


class SecurityScanner:
    """Scan retrieved documents for security issues."""

    def scan(self, docs: list[str]) -> ScanResult:
        result = ScanResult()
        for idx, doc in enumerate(docs):
            result.alerts.extend(self._check_injection(idx, doc))
            result.alerts.extend(self._check_base64(idx, doc))
        if result.alerts:
            logger.warning(
                "SecurityScanner: %d alert(s) across %d doc(s)",
                len(result.alerts), len(docs),
            )
        return result

    def _check_injection(self, idx: int, doc: str) -> list[SecurityAlert]:
        alerts: list[SecurityAlert] = []
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(doc)
            if match:
                alerts.append(
                    SecurityAlert(
                        doc_index=idx,
                        alert_type="PROMPT_INJECTION",
                        detail=f"Pattern matched: '{pattern.pattern}'",
                        snippet=doc[max(0, match.start() - 20): match.end() + 20],
                    )
                )
                logger.warning("Injection pattern in doc %d: %s", idx, pattern.pattern)
        return alerts

    def _check_base64(self, idx: int, doc: str) -> list[SecurityAlert]:
        """Detect base64-encoded content that may indicate data exfiltration."""
        alerts: list[SecurityAlert] = []
        matches = _BASE64_PATTERN.findall(doc)
        if matches:
            # Report the first match as a representative snippet.
            first_match = matches[0]
            alerts.append(
                SecurityAlert(
                    doc_index=idx,
                    alert_type="BASE64_CONTENT",
                    detail=(
                        f"Found {len(matches)} potential base64 blob(s); "
                        f"longest={max(len(m) for m in matches)} chars"
                    ),
                    snippet=first_match[:80],
                )
            )
            logger.warning(
                "Base64 blob(s) detected in doc %d (%d match(es))", idx, len(matches)
            )
        return alerts
```

### Tests (`tests/unit/test_security_scanner.py`)

```python
"""Tests for SecurityScanner base64 detection."""
import pytest
from security_scanner import SecurityAlert, SecurityScanner


SCANNER = SecurityScanner()


def test_clean_document_no_alerts():
    result = SCANNER.scan(["This is a normal document about RAG systems."])
    assert result.is_clean


def test_base64_detected_in_single_doc():
    # 44-char valid base64 string
    blob = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IHN0cmluZw=="
    result = SCANNER.scan([f"Normal text. {blob} more text."])
    assert not result.is_clean
    assert any(a.alert_type == "BASE64_CONTENT" for a in result.alerts)


def test_short_base64_like_string_not_flagged():
    # 39 chars — below threshold
    short = "A" * 39
    result = SCANNER.scan([f"token: {short}"])
    base64_alerts = [a for a in result.alerts if a.alert_type == "BASE64_CONTENT"]
    assert len(base64_alerts) == 0


def test_base64_exactly_at_threshold_flagged():
    # Exactly 40 chars of valid base64 chars
    blob = "A" * 40
    result = SCANNER.scan([f"data: {blob}"])
    assert any(a.alert_type == "BASE64_CONTENT" for a in result.alerts)


def test_base64_alert_has_correct_doc_index():
    clean = "Normal text here."
    malicious = "Encoded: " + "B" * 50
    result = SCANNER.scan([clean, malicious])
    alerts = [a for a in result.alerts if a.alert_type == "BASE64_CONTENT"]
    assert len(alerts) == 1
    assert alerts[0].doc_index == 1


def test_multiple_base64_blobs_reported_once_per_doc():
    blob1 = "C" * 45
    blob2 = "D" * 50
    result = SCANNER.scan([f"{blob1} ... {blob2}"])
    base64_alerts = [a for a in result.alerts if a.alert_type == "BASE64_CONTENT"]
    # One alert per doc (with count in detail), not one per blob
    assert len(base64_alerts) == 1
    assert "2" in base64_alerts[0].detail


def test_injection_and_base64_both_detected():
    blob = "E" * 44
    doc = f"Ignore previous instructions. Also: {blob}"
    result = SCANNER.scan([doc])
    types = {a.alert_type for a in result.alerts}
    assert "PROMPT_INJECTION" in types
    assert "BASE64_CONTENT" in types
```
