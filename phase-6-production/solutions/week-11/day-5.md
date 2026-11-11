# Week 11 · Day 5 Solutions

## LLM and Agent Observability

---

## Exercise 1

**Extend `LLMCallRecord` with a `cost_alert` method that returns a warning string if cost exceeds `threshold_usd`, else `None`.**

```python
from __future__ import annotations

import logging
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, computed_field, model_validator

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD) — as of course snapshot
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 15.00, "output": 75.00},
    "claude-sonnet-5-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-5-5": {"input": 0.80, "output": 4.00},
}


class LLMCallRecord(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: datetime = None  # type: ignore[assignment]
    error: str | None = None

    PRICING: ClassVar[dict[str, dict[str, float]]] = PRICING

    @model_validator(mode="before")
    @classmethod
    def set_timestamp(cls, data: dict) -> dict:
        if data.get("timestamp") is None:
            data["timestamp"] = datetime.utcnow()
        return data

    @computed_field  # type: ignore[misc]
    @property
    def cost_usd(self) -> float:
        """Compute cost in USD based on model pricing."""
        pricing = self.PRICING.get(self.model)
        if pricing is None:
            logger.warning("Unknown model %s — cost computed as 0.0", self.model)
            return 0.0
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    def cost_alert(self, threshold_usd: float) -> str | None:
        """Return a warning string if this call's cost exceeds the threshold.

        Args:
            threshold_usd: Maximum acceptable cost per call in USD.

        Returns:
            Warning string if cost > threshold_usd, None otherwise.
        """
        if self.cost_usd > threshold_usd:
            msg = (
                f"COST ALERT: call to {self.model} cost ${self.cost_usd:.6f} "
                f"(threshold ${threshold_usd:.6f})"
            )
            logger.warning(msg)
            return msg
        return None


# --- Usage example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    record = LLMCallRecord(
        model="claude-opus-5-5",
        input_tokens=5000,
        output_tokens=2000,
        latency_ms=1234.5,
    )
    print(record.cost_usd)           # e.g. 0.225
    print(record.cost_alert(0.10))   # warning string
    print(record.cost_alert(1.00))   # None
```

---

## Exercise 2

**Implement a rolling 1-hour cost window using a `deque`.**

```python
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class RollingCostWindow:
    """Maintains a rolling 1-hour window of LLMCallRecords and tracks cumulative cost."""

    WINDOW = timedelta(hours=1)

    def __init__(self) -> None:
        self._records: deque[LLMCallRecord] = deque()

    def add(self, record: LLMCallRecord) -> None:
        """Append a record and evict stale entries older than the window."""
        self._records.append(record)
        self._evict()

    def _evict(self) -> None:
        """Remove records older than 1 hour from the left of the deque."""
        cutoff = datetime.now(tz=timezone.utc) - self.WINDOW
        # Records are appended in chronological order, so the oldest are on the left.
        while self._records:
            record = self._records[0]
            # Normalise naive timestamps to UTC for comparison
            ts = record.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                self._records.popleft()
            else:
                break

    @property
    def hourly_cost(self) -> float:
        """Total cost of all records in the current 1-hour window."""
        self._evict()
        total = sum(r.cost_usd for r in self._records)
        logger.debug(
            "Rolling hourly cost: $%.6f over %d records", total, len(self._records)
        )
        return round(total, 8)

    @property
    def record_count(self) -> int:
        self._evict()
        return len(self._records)


# --- Usage example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    window = RollingCostWindow()
    for i in range(10):
        window.add(
            LLMCallRecord(
                model="claude-sonnet-5-5",
                input_tokens=1000,
                output_tokens=500,
                latency_ms=300.0,
            )
        )
    print(f"Hourly cost: ${window.hourly_cost:.4f}")   # 10 * cost_per_call
    print(f"Records in window: {window.record_count}")
```

---

## Exercise 3

**Per-model latency percentiles (p50, p90, p99) using `statistics.quantiles`.**

```python
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyPercentiles:
    model: str
    p50_ms: float
    p90_ms: float
    p99_ms: float
    sample_size: int


def compute_latency_percentiles(
    records: list[LLMCallRecord],
    model_name: str,
) -> LatencyPercentiles:
    """Return p50/p90/p99 latencies (ms) for the given model.

    Args:
        records: Full list of LLMCallRecords (may include multiple models).
        model_name: Filter by this model name.

    Returns:
        LatencyPercentiles dataclass.

    Raises:
        ValueError: If fewer than 2 records exist for the model (quantiles require >=2).
    """
    filtered = [r.latency_ms for r in records if r.model == model_name]
    if len(filtered) < 2:
        raise ValueError(
            f"Need at least 2 records for model '{model_name}', got {len(filtered)}"
        )

    # statistics.quantiles returns n-1 cut points dividing data into n equal groups.
    # For p50: quantiles(data, n=2)[0]
    # For p90/p99: quantiles(data, n=100)[88] and [97]  (0-indexed: p90=index 89 of 100)
    sorted_data = sorted(filtered)

    p50 = statistics.quantiles(sorted_data, n=2)[0]

    q100 = statistics.quantiles(sorted_data, n=100)
    p90 = q100[89]   # index 89 → 90th percentile cut point
    p99 = q100[98]   # index 98 → 99th percentile cut point

    result = LatencyPercentiles(
        model=model_name,
        p50_ms=round(p50, 3),
        p90_ms=round(p90, 3),
        p99_ms=round(p99, 3),
        sample_size=len(filtered),
    )
    logger.info(
        "Latency percentiles for %s (n=%d): p50=%.1fms p90=%.1fms p99=%.1fms",
        model_name,
        result.sample_size,
        result.p50_ms,
        result.p90_ms,
        result.p99_ms,
    )
    return result


# --- Usage example ---
if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    records = [
        LLMCallRecord(
            model="claude-haiku-5-5",
            input_tokens=500,
            output_tokens=200,
            latency_ms=random.gauss(150, 30),
        )
        for _ in range(200)
    ]
    perc = compute_latency_percentiles(records, "claude-haiku-5-5")
    print(perc)
```

**Note on index selection:** `statistics.quantiles(data, n=100)` returns 99 cut points
(indices 0–98). Index 49 = p50, index 89 = p90, index 98 = p99.  For `n=2` there is 1
cut point at index 0, which is the median (p50).

---

## Exercise 4

**Pytest test: verify cost calculation for `claude-opus-5-5` with 1000 input tokens and 500 output tokens — LLM mocked, no real API calls.**

```python
# tests/unit/test_llm_call_record.py
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Assuming the module above lives at src/observability/llm_records.py
# Adjust the import path to match your project layout.
from observability.llm_records import LLMCallRecord, PRICING


class TestLLMCallRecordCost:
    """Unit tests for LLMCallRecord cost calculations."""

    def _make_record(
        self,
        model: str = "claude-opus-5-5",
        input_tokens: int = 1000,
        output_tokens: int = 500,
        latency_ms: float = 500.0,
    ) -> LLMCallRecord:
        return LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

    def test_opus_cost_1000_input_500_output(self) -> None:
        """claude-opus-5-5 with 1000 input + 500 output tokens has the correct USD cost.

        Pricing: $15.00 / 1M input,  $75.00 / 1M output
        Expected:
            input_cost  = 1000 / 1_000_000 * 15.00 = 0.015
            output_cost =  500 / 1_000_000 * 75.00 = 0.0375
            total       = 0.0525
        """
        record = self._make_record(
            model="claude-opus-5-5",
            input_tokens=1000,
            output_tokens=500,
        )
        expected_cost = (1000 / 1_000_000 * 15.00) + (500 / 1_000_000 * 75.00)
        assert record.cost_usd == pytest.approx(expected_cost, rel=1e-6)

    def test_cost_alert_below_threshold_returns_none(self) -> None:
        record = self._make_record()
        # threshold above the computed cost
        assert record.cost_alert(1.00) is None

    def test_cost_alert_above_threshold_returns_warning(self) -> None:
        record = self._make_record()
        result = record.cost_alert(0.01)
        assert result is not None
        assert "COST ALERT" in result
        assert "claude-opus-5-5" in result

    def test_cost_alert_at_exact_threshold_returns_none(self) -> None:
        """Cost must EXCEED threshold, not merely equal it."""
        record = self._make_record()
        # Set threshold exactly equal to computed cost — should not alert
        assert record.cost_alert(record.cost_usd) is None

    @patch("observability.llm_records.LLMCallRecord")
    def test_cost_with_mocked_llm_response(self, MockRecord: MagicMock) -> None:
        """Demonstrate mocking approach: LLM response object is mocked; cost is
        computed locally from token counts — no real API call is made."""
        # Simulate what an Anthropic SDK response looks like
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500
        mock_response.model = "claude-opus-5-5"

        # In production code you would call the SDK; here we verify the math
        # using the same formula as LLMCallRecord.cost_usd without touching the API.
        pricing = PRICING["claude-opus-5-5"]
        computed_cost = (
            mock_response.usage.input_tokens / 1_000_000 * pricing["input"]
            + mock_response.usage.output_tokens / 1_000_000 * pricing["output"]
        )
        assert computed_cost == pytest.approx(0.0525, rel=1e-6)

    def test_unknown_model_cost_is_zero(self) -> None:
        record = self._make_record(model="gpt-unknown-99")
        assert record.cost_usd == 0.0

    @pytest.mark.parametrize(
        "model,input_tok,output_tok,expected",
        [
            # claude-haiku-5-5: $0.80 / 1M in, $4.00 / 1M out
            ("claude-haiku-5-5", 1000, 500, 0.0008 + 0.002),
            # claude-sonnet-5-5: $3.00 / 1M in, $15.00 / 1M out
            ("claude-sonnet-5-5", 1000, 500, 0.003 + 0.0075),
            # claude-opus-5-5: $15.00 / 1M in, $75.00 / 1M out
            ("claude-opus-5-5", 1000, 500, 0.015 + 0.0375),
        ],
    )
    def test_cost_parametrized_all_models(
        self, model: str, input_tok: int, output_tok: int, expected: float
    ) -> None:
        record = self._make_record(
            model=model, input_tokens=input_tok, output_tokens=output_tok
        )
        assert record.cost_usd == pytest.approx(expected, rel=1e-6)
```

**Run with:**

```bash
pytest tests/unit/test_llm_call_record.py -v
```

No environment variables, no API key, no network access required — all LLM interaction
is mocked via `unittest.mock.MagicMock`.
