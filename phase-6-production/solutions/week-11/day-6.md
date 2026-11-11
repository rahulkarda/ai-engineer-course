# Week 11 · Day 6 Solutions

## SLOs, SLIs, Error Budgets, and Alerting

---

## Exercise 1

**Multi-window SLO check: compute SLO compliance for 1h, 6h, 24h windows and return which are breaching.**

```python
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, computed_field, field_validator

logger = logging.getLogger(__name__)


class RequestRecord(BaseModel):
    """Represents a single HTTP/API request outcome."""

    timestamp: datetime
    success: bool
    latency_ms: float


class WindowResult(BaseModel):
    """SLO compliance result for a single time window."""

    window_hours: int
    total_requests: int
    successful_requests: int
    error_rate: float          # fraction, e.g. 0.002 = 0.2%
    slo_target: float          # fraction, e.g. 0.999
    is_breaching: bool

    @computed_field  # type: ignore[misc]
    @property
    def availability(self) -> float:
        return round(1.0 - self.error_rate, 6)


class MultiWindowSLOResult(BaseModel):
    """Aggregated results across all requested windows."""

    windows: list[WindowResult]
    breaching_windows: list[int]   # list of window_hours that are breaching

    @computed_field  # type: ignore[misc]
    @property
    def any_breaching(self) -> bool:
        return len(self.breaching_windows) > 0


def check_multi_window_slo(
    records: list[RequestRecord],
    slo_target: float = 0.999,
    window_hours: tuple[int, ...] = (1, 6, 24),
) -> MultiWindowSLOResult:
    """Compute SLO compliance for multiple time windows simultaneously.

    Args:
        records: All available RequestRecords (may span days).
        slo_target: Required availability fraction (e.g. 0.999 = 99.9%).
        window_hours: Tuple of window sizes in hours to evaluate.

    Returns:
        MultiWindowSLOResult with per-window details and list of breaching windows.
    """
    now = datetime.now(tz=timezone.utc)
    window_results: list[WindowResult] = []

    for hours in window_hours:
        cutoff = now - timedelta(hours=hours)
        window_records = [
            r for r in records
            if _to_utc(r.timestamp) >= cutoff
        ]
        total = len(window_records)
        if total == 0:
            logger.warning("No records in %dh window — treating as 100%% available", hours)
            window_results.append(WindowResult(
                window_hours=hours,
                total_requests=0,
                successful_requests=0,
                error_rate=0.0,
                slo_target=slo_target,
                is_breaching=False,
            ))
            continue

        successes = sum(1 for r in window_records if r.success)
        error_rate = (total - successes) / total
        availability = 1.0 - error_rate
        breaching = availability < slo_target

        if breaching:
            logger.warning(
                "SLO BREACH in %dh window: availability=%.4f%% (target=%.4f%%)",
                hours,
                availability * 100,
                slo_target * 100,
            )

        window_results.append(WindowResult(
            window_hours=hours,
            total_requests=total,
            successful_requests=successes,
            error_rate=round(error_rate, 8),
            slo_target=slo_target,
            is_breaching=breaching,
        ))

    breaching = [w.window_hours for w in window_results if w.is_breaching]
    return MultiWindowSLOResult(windows=window_results, breaching_windows=breaching)


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts
```

---

## Exercise 2

**Burn rate calculation.**

Error budget burn rate formula:

```
burn_rate = (1 - current_availability) / (1 - slo_target)
           = current_error_rate / allowed_error_rate
```

A burn rate > 1 means the budget is being consumed faster than it naturally replenishes
over the SLO window (typically 30 days).

```python
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BurnRateResult(BaseModel):
    """Burn rate metrics for a given window."""

    window_hours: int
    error_rate: float
    slo_target: float
    burn_rate: float
    interpretation: str


def compute_burn_rate(
    error_rate: float,
    slo_target: float,
    window_hours: int = 1,
) -> BurnRateResult:
    """Compute the error budget burn rate.

    burn_rate = current_error_rate / allowed_error_rate
              = (1 - availability) / (1 - slo_target)

    Args:
        error_rate: Observed fraction of failures (e.g. 0.005 = 0.5%).
        slo_target: SLO target as a fraction (e.g. 0.999).
        window_hours: Measurement window in hours (informational).

    Returns:
        BurnRateResult with burn_rate and a human-readable interpretation.

    Raises:
        ValueError: If slo_target >= 1.0 (would produce division by zero).
    """
    if slo_target >= 1.0:
        raise ValueError(f"slo_target must be < 1.0, got {slo_target}")
    if error_rate < 0.0 or error_rate > 1.0:
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")

    allowed_error_rate = 1.0 - slo_target
    burn_rate = error_rate / allowed_error_rate if allowed_error_rate > 0 else float("inf")
    burn_rate = round(burn_rate, 4)

    if burn_rate <= 1.0:
        interpretation = "Within budget — consuming error budget at a sustainable rate."
    elif burn_rate <= 5.0:
        interpretation = (
            f"Elevated burn ({burn_rate:.2f}x) — ticket-severity; "
            "alert if sustained for 6h."
        )
    else:
        interpretation = (
            f"Critical burn ({burn_rate:.2f}x) — page-severity; "
            "page on-call immediately."
        )

    logger.info(
        "[%dh window] error_rate=%.4f%% slo=%.4f%% burn_rate=%.2fx",
        window_hours,
        error_rate * 100,
        slo_target * 100,
        burn_rate,
    )
    return BurnRateResult(
        window_hours=window_hours,
        error_rate=error_rate,
        slo_target=slo_target,
        burn_rate=burn_rate,
        interpretation=interpretation,
    )


def compute_burn_rates_multi_window(
    result: MultiWindowSLOResult,
    slo_target: float = 0.999,
) -> list[BurnRateResult]:
    """Convenience: compute burn rates for all windows in a MultiWindowSLOResult."""
    return [
        compute_burn_rate(
            error_rate=w.error_rate,
            slo_target=slo_target,
            window_hours=w.window_hours,
        )
        for w in result.windows
    ]
```

---

## Exercise 3

**Extended `generate_alert_rules()` supporting "ticket" and "page" severity.**

```python
from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    TICKET = "ticket"   # burn rate 1–5x; page after 6h
    PAGE = "page"       # burn rate >5x; page immediately


class AlertRule(BaseModel):
    """Prometheus/Alertmanager-compatible alert rule specification."""

    name: str
    severity: AlertSeverity
    burn_rate_threshold: float
    window_hours: int
    for_duration: str          # e.g. "6h", "0m"
    description: str
    annotations: dict[str, str]
    labels: dict[str, str]


def generate_alert_rules(
    slo_name: str = "api_availability",
    slo_target: float = 0.999,
) -> list[AlertRule]:
    """Generate multi-burn-rate alert rules for a given SLO.

    Produces both ticket-severity (1–5x burn, sustained 6h) and
    page-severity (>5x burn, immediate) alert rules across standard
    1h and 6h measurement windows.

    Args:
        slo_name: Logical name for the SLO (used in rule names/labels).
        slo_target: SLO availability target as a fraction (e.g. 0.999).

    Returns:
        List of AlertRule objects ready for serialisation to YAML/JSON.
    """
    rules: list[AlertRule] = []

    # Page-severity: fast burn on 1h window — act immediately
    rules.append(AlertRule(
        name=f"{slo_name}_burn_rate_critical",
        severity=AlertSeverity.PAGE,
        burn_rate_threshold=5.0,
        window_hours=1,
        for_duration="0m",      # fire immediately, no grace period
        description=(
            f"SLO '{slo_name}' error budget burning at >5x rate in the last 1h. "
            "Page on-call immediately."
        ),
        annotations={
            "summary": f"Critical error budget burn for {slo_name}",
            "runbook": f"https://runbooks.internal/{slo_name}/critical",
        },
        labels={
            "slo": slo_name,
            "severity": AlertSeverity.PAGE.value,
            "burn_rate_multiplier": "5",
        },
    ))

    # Page-severity: slightly slower but still aggressive burn on 6h window
    rules.append(AlertRule(
        name=f"{slo_name}_burn_rate_high",
        severity=AlertSeverity.PAGE,
        burn_rate_threshold=2.0,
        window_hours=6,
        for_duration="0m",
        description=(
            f"SLO '{slo_name}' error budget burning at >2x rate sustained over 6h. "
            "Page on-call immediately."
        ),
        annotations={
            "summary": f"High error budget burn for {slo_name}",
            "runbook": f"https://runbooks.internal/{slo_name}/high",
        },
        labels={
            "slo": slo_name,
            "severity": AlertSeverity.PAGE.value,
            "burn_rate_multiplier": "2",
        },
    ))

    # Ticket-severity: moderate burn on 1h window — create ticket, page after 6h
    rules.append(AlertRule(
        name=f"{slo_name}_burn_rate_elevated",
        severity=AlertSeverity.TICKET,
        burn_rate_threshold=1.0,
        window_hours=1,
        for_duration="6h",      # only fire if sustained for 6h
        description=(
            f"SLO '{slo_name}' error budget burn rate between 1x–5x for >6h. "
            "Create a ticket; escalate to page if unresolved."
        ),
        annotations={
            "summary": f"Elevated error budget burn for {slo_name}",
            "runbook": f"https://runbooks.internal/{slo_name}/elevated",
        },
        labels={
            "slo": slo_name,
            "severity": AlertSeverity.TICKET.value,
            "burn_rate_multiplier": "1",
        },
    ))

    logger.info(
        "Generated %d alert rules for SLO '%s' (target=%.4f%%)",
        len(rules),
        slo_name,
        slo_target * 100,
    )
    return rules


# --- Serialise to Prometheus-style YAML (for illustration) ---
def rules_to_prometheus_yaml(rules: list[AlertRule]) -> str:
    """Render alert rules to a Prometheus alerting rules YAML string."""
    lines = ["groups:", "- name: slo_alerts", "  rules:"]
    for r in rules:
        lines += [
            f"  - alert: {r.name}",
            f"    for: {r.for_duration}",
            "    labels:",
        ]
        for k, v in r.labels.items():
            lines.append(f"      {k}: {v!r}")
        lines.append("    annotations:")
        for k, v in r.annotations.items():
            lines.append(f"      {k}: {v!r}")
    return "\n".join(lines)
```

---

## Exercise 4

**Pytest tests: 1 failure in 1000 requests passes 99.9% SLO; 2 failures breaches it.**

```python
# tests/unit/test_slo.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from observability.slo import (
    RequestRecord,
    check_multi_window_slo,
    compute_burn_rate,
)


def _make_records(
    total: int,
    failures: int,
    base_time: datetime | None = None,
) -> list[RequestRecord]:
    """Generate `total` records with the last `failures` being failures."""
    if base_time is None:
        base_time = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
    records: list[RequestRecord] = []
    for i in range(total):
        records.append(RequestRecord(
            timestamp=base_time + timedelta(seconds=i),
            success=(i >= failures),   # first `failures` records are failures
            latency_ms=50.0,
        ))
    return records


class TestSLOCompliance:
    """Tests for multi-window SLO compliance checking."""

    SLO_TARGET = 0.999  # 99.9%

    def test_one_failure_in_1000_passes_slo(self) -> None:
        """1 failure / 1000 requests = 0.1% error rate = exactly at budget; should pass."""
        records = _make_records(total=1000, failures=1)
        result = check_multi_window_slo(records, slo_target=self.SLO_TARGET)

        # Find the 1h window result (all records are within 30 min)
        one_hour = next(w for w in result.windows if w.window_hours == 1)

        assert one_hour.total_requests == 1000
        assert one_hour.successful_requests == 999
        assert one_hour.error_rate == pytest.approx(0.001, rel=1e-6)
        # 99.9% availability == SLO target → NOT breaching (strict less-than check)
        assert not one_hour.is_breaching, (
            f"Expected no breach with 1/1000 failures, got availability="
            f"{one_hour.availability:.6f}"
        )
        assert 1 not in result.breaching_windows

    def test_two_failures_in_1000_breaches_slo(self) -> None:
        """2 failures / 1000 requests = 0.2% error rate > 0.1% budget; should breach."""
        records = _make_records(total=1000, failures=2)
        result = check_multi_window_slo(records, slo_target=self.SLO_TARGET)

        one_hour = next(w for w in result.windows if w.window_hours == 1)

        assert one_hour.total_requests == 1000
        assert one_hour.successful_requests == 998
        assert one_hour.error_rate == pytest.approx(0.002, rel=1e-6)
        assert one_hour.is_breaching, (
            f"Expected breach with 2/1000 failures, got availability="
            f"{one_hour.availability:.6f}"
        )
        assert 1 in result.breaching_windows

    def test_zero_failures_does_not_breach(self) -> None:
        records = _make_records(total=1000, failures=0)
        result = check_multi_window_slo(records, slo_target=self.SLO_TARGET)
        assert not result.any_breaching

    def test_all_failures_breaches_all_windows(self) -> None:
        records = _make_records(total=100, failures=100)
        result = check_multi_window_slo(records, slo_target=self.SLO_TARGET)
        # All windows covering these records should breach
        assert result.any_breaching

    def test_multi_window_result_has_three_windows(self) -> None:
        records = _make_records(total=100, failures=0)
        result = check_multi_window_slo(records, slo_target=self.SLO_TARGET)
        window_hours = [w.window_hours for w in result.windows]
        assert sorted(window_hours) == [1, 6, 24]


class TestBurnRate:
    """Tests for error budget burn rate calculations."""

    def test_burn_rate_greater_than_one_means_over_budget(self) -> None:
        # 0.2% error rate with 99.9% SLO → 2x burn
        result = compute_burn_rate(error_rate=0.002, slo_target=0.999)
        assert result.burn_rate == pytest.approx(2.0, rel=1e-4)

    def test_burn_rate_at_exactly_slo_boundary(self) -> None:
        # 0.1% error rate with 99.9% SLO → 1x burn (exactly at budget)
        result = compute_burn_rate(error_rate=0.001, slo_target=0.999)
        assert result.burn_rate == pytest.approx(1.0, rel=1e-4)

    def test_invalid_slo_target_raises(self) -> None:
        with pytest.raises(ValueError, match="slo_target must be < 1.0"):
            compute_burn_rate(error_rate=0.001, slo_target=1.0)
```

**Run with:**

```bash
pytest tests/unit/test_slo.py -v
```
