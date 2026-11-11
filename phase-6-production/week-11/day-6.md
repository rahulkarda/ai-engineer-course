# Phase 6 · Week 11 · Day 6 — SLOs, SLIs, Error Budgets, and Alerting Rules

> Objective: Define SLIs and SLOs for an LLM service, compute error budget consumption, and generate Prometheus alerting rule YAML programmatically.  ·  Time: ~1 hr  ·  Prereqs: [[day-5]]

---

## Concept

### SLI — Service Level Indicator

An SLI is a *measured* quantity that describes one aspect of your service's behaviour. It must be:

- **Quantifiable**: a ratio, a latency percentile, or a count.
- **From the user's perspective**: measure what users experience, not internal implementation details.
- **Reliable to measure**: it should be the same whether you measure it or your user does.

Common SLIs for LLM services:

| SLI | Formula | Notes |
|---|---|---|
| Availability | `successful_requests / total_requests` | Exclude client errors (4xx) from denominator |
| Latency | `requests_under_threshold / total_requests` | e.g., P95 < 5 s |
| Token throughput | `output_tokens_per_second` | Average over a time window |
| Error rate | `error_requests / total_requests` | HTTP 5xx + API errors |
| Context limit hit rate | `truncated_responses / total_responses` | LLM-specific |

### SLO — Service Level Objective

An SLO is the *target* for an SLI over a rolling time window:

```
Availability SLO: 99.9% over a rolling 30-day window
Latency SLO:      95% of requests complete in under 5 seconds
```

SLOs are *internal* targets. They should be stricter than the SLA to give you a buffer.

### SLA — Service Level Agreement

An SLA is the *contractual commitment* to customers. It is usually less strict than your SLO:

- SLO: 99.9% availability (internal target)
- SLA: 99.5% availability (contractual — what you promise and pay penalties for if breached)

You need headroom between SLO and SLA so that a brief SLO breach doesn't immediately trigger an SLA violation.

### Error budget

```
Error budget = 100% − SLO target
```

For a 99.9% availability SLO over 30 days:

```
Error budget = 0.1% of 30 days
             = 0.001 × 30 × 24 × 60 min
             = 43.2 minutes of allowable downtime per month
```

Error budgets encode the key insight: **reliability has a cost**. Every minute of planned maintenance, every risky deployment, every experimental feature consumes budget. When the budget is full, you are free to move fast. When it is nearly depleted, you slow down and focus on reliability.

This creates a self-regulating feedback loop:
- Budget healthy → deploy new features, experiment.
- Budget low → freeze risky changes, prioritise reliability work.
- Budget exhausted → incident response only until the window resets.

### Alerting rules

Prometheus alerting rules fire when a PromQL expression is true for a sustained duration (`for`). A good alerting rule:

1. **Fires on user impact, not symptoms**: alert on high error rate, not high CPU.
2. **Has a meaningful `for` duration**: avoids flapping from transient spikes.
3. **Has clear severity labels**: `critical` (page someone now) vs `warning` (investigate tomorrow).
4. **Includes runbook links**: tell the on-call engineer what to do, not just what is wrong.

### Alert fatigue

Too many alerts = ignored alerts = missed incidents. Rules of thumb:

- If an alert fires more than once a week without action, it is not actionable — fix or remove it.
- Every `critical` alert must have a human response playbook.
- Use `warning` for "look at this tomorrow", `critical` for "wake someone up now".
- Aggregate related alerts: one "LLM service degraded" alert is better than five separate threshold alerts firing simultaneously.

---

## How it works internally

The SLO compliance check is a simple ratio computation:

1. Filter requests to the window of interest.
2. Count good events (success) vs. total events.
3. Compute `sli = good / total`.
4. Compare to `slo_target`.
5. Compute remaining budget: `budget_remaining = slo_target - sli` (expressed as fraction of total events).

Prometheus alert rules are YAML evaluated by the Prometheus server's rule evaluator. You can generate them from code to keep rule definitions DRY and type-safe.

---

## Worked example

Install dependencies:

```bash
uv add pydantic pyyaml
```

```python
# slo_engine.py
"""
SLO/SLI computation engine with Prometheus alerting rule generation.

Demonstrates:
- Pydantic v2 models for SLOConfig, RequestRecord, SLOReport
- Computing SLI ratio from raw request records
- Error budget calculation and consumption tracking
- Generating Prometheus alerting rule YAML from SLOConfig
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── 1. Enums ───────────────────────────────────────────────────────────────────

class SLIType(StrEnum):
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


class AlertSeverity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


# ── 2. Pydantic models ─────────────────────────────────────────────────────────

class SLOConfig(BaseModel):
    """
    Configuration for a single SLO.

    An SLO targets one SLI (e.g., availability) with a specific threshold
    over a rolling time window.
    """

    name: str = Field(description="Human-readable name, e.g. 'LLM Gateway Availability'")
    service: str = Field(description="Service identifier for Prometheus label matching")
    sli_type: SLIType
    target: float = Field(
        ge=0.0,
        le=1.0,
        description="SLO target as a fraction, e.g. 0.999 for 99.9%",
    )
    window_days: int = Field(
        ge=1,
        le=365,
        default=30,
        description="Rolling window in days",
    )
    latency_threshold_ms: float | None = Field(
        default=None,
        description="Only used when sli_type=LATENCY: requests under this threshold are 'good'",
    )
    # Alerting configuration
    warning_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Fire warning alert when SLI drops below (target × warning_threshold)",
    )
    critical_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Fire critical alert when SLI drops below (target × critical_threshold)",
    )
    runbook_url: str = Field(default="https://wiki.example.com/runbooks")

    @field_validator("target")
    @classmethod
    def target_must_be_meaningful(cls, v: float) -> float:
        if v < 0.5:
            raise ValueError("SLO target below 50% is not meaningful")
        return v

    @model_validator(mode="after")
    def latency_requires_threshold(self) -> "SLOConfig":
        if self.sli_type == SLIType.LATENCY and self.latency_threshold_ms is None:
            raise ValueError("latency_threshold_ms is required when sli_type=LATENCY")
        return self

    @property
    def error_budget_fraction(self) -> float:
        """Fraction of total requests that may be bad: 1 - target."""
        return round(1.0 - self.target, 8)

    @property
    def allowable_downtime_minutes(self) -> float:
        """
        For availability SLOs: maximum minutes of downtime allowed in the window.
        Not meaningful for latency SLOs (returns 0).
        """
        if self.sli_type != SLIType.AVAILABILITY:
            return 0.0
        total_minutes = self.window_days * 24 * 60
        return round(total_minutes * self.error_budget_fraction, 2)


class RequestRecord(BaseModel):
    """One HTTP or LLM API request event."""

    request_id: str
    timestamp: datetime
    success: bool
    latency_ms: float = Field(ge=0.0)
    status_code: int = Field(ge=100, le=599)
    service: str


class SLOReport(BaseModel):
    """Result of evaluating an SLOConfig against a set of RequestRecords."""

    slo_name: str
    sli_type: SLIType
    window_days: int
    total_requests: int
    good_requests: int
    bad_requests: int
    sli_value: float = Field(description="Measured SLI as a fraction")
    slo_target: float
    slo_met: bool
    error_budget_total: float = Field(description="Total error budget as a fraction (1 - target)")
    error_budget_consumed: float = Field(description="Fraction of error budget consumed so far")
    error_budget_remaining_pct: float = Field(
        description="Remaining error budget as percentage of total budget"
    )
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_budget_exhausted(self) -> bool:
        return self.error_budget_consumed >= 1.0

    @property
    def budget_burn_rate(self) -> float:
        """How fast budget is being consumed relative to target (1.0 = on track)."""
        if self.error_budget_total == 0:
            return 0.0
        bad_fraction = 1.0 - self.sli_value
        return round(bad_fraction / self.error_budget_total, 4)


# ── 3. SLI computation ─────────────────────────────────────────────────────────

def compute_sli(
    config: SLOConfig,
    records: list[RequestRecord],
    as_of: datetime | None = None,
) -> SLOReport:
    """
    Compute the current SLI and error budget from raw request records.

    Args:
        config: The SLO configuration defining target, window, and SLI type.
        records: All request records (will be filtered to the rolling window).
        as_of: Reference time for the rolling window (defaults to now UTC).

    Returns:
        SLOReport with SLI value, compliance status, and budget metrics.
    """
    if as_of is None:
        as_of = datetime.now(UTC)

    window_start = as_of - timedelta(days=config.window_days)

    # Filter to rolling window and matching service
    windowed = [
        r for r in records
        if r.timestamp >= window_start
        and r.service == config.service
    ]

    total = len(windowed)

    if total == 0:
        logger.warning("No records found for service=%s in the last %d days", config.service, config.window_days)
        return SLOReport(
            slo_name=config.name,
            sli_type=config.sli_type,
            window_days=config.window_days,
            total_requests=0,
            good_requests=0,
            bad_requests=0,
            sli_value=1.0,  # vacuously compliant
            slo_target=config.target,
            slo_met=True,
            error_budget_total=config.error_budget_fraction,
            error_budget_consumed=0.0,
            error_budget_remaining_pct=100.0,
        )

    # Classify each request as good or bad according to SLI type
    good = 0
    for r in windowed:
        if config.sli_type == SLIType.AVAILABILITY:
            # Good = request succeeded (not a server-side error)
            if r.success and r.status_code < 500:
                good += 1
        elif config.sli_type == SLIType.ERROR_RATE:
            # Good = not an error
            if r.status_code < 400:
                good += 1
        elif config.sli_type == SLIType.LATENCY:
            # Good = response within latency threshold
            assert config.latency_threshold_ms is not None
            if r.latency_ms <= config.latency_threshold_ms:
                good += 1

    bad = total - good
    sli_value = good / total

    # Error budget accounting
    budget_total = config.error_budget_fraction
    bad_fraction = bad / total

    if budget_total == 0:
        consumed = 1.0 if bad > 0 else 0.0
    else:
        consumed = round(bad_fraction / budget_total, 6)

    remaining_pct = round(max(0.0, (1.0 - consumed)) * 100, 2)

    report = SLOReport(
        slo_name=config.name,
        sli_type=config.sli_type,
        window_days=config.window_days,
        total_requests=total,
        good_requests=good,
        bad_requests=bad,
        sli_value=round(sli_value, 8),
        slo_target=config.target,
        slo_met=sli_value >= config.target,
        error_budget_total=budget_total,
        error_budget_consumed=consumed,
        error_budget_remaining_pct=remaining_pct,
    )

    logger.info(
        "SLO '%s': SLI=%.4f%% target=%.4f%% budget_remaining=%.1f%% met=%s",
        config.name,
        sli_value * 100,
        config.target * 100,
        remaining_pct,
        report.slo_met,
    )
    return report


# ── 4. Alerting rule generation ────────────────────────────────────────────────

def generate_alert_rules(config: SLOConfig) -> str:
    """
    Generate Prometheus alerting rule YAML for an SLOConfig.

    Produces two alert rules:
    - Warning: fires when SLI drops below (target × warning_threshold)
    - Critical: fires when SLI drops below (target × critical_threshold)

    Args:
        config: SLO configuration.

    Returns:
        YAML string ready to be written to a Prometheus rules file.
    """
    warning_target = round(config.target * config.warning_threshold, 6)
    critical_target = round(config.target * config.critical_threshold, 6)

    # PromQL expressions for each SLI type
    if config.sli_type == SLIType.AVAILABILITY:
        metric_expr = (
            f'sum(rate(http_requests_total{{service="{config.service}",status!~"5.."}}[5m])) '
            f'/ sum(rate(http_requests_total{{service="{config.service}"}}[5m]))'
        )
    elif config.sli_type == SLIType.ERROR_RATE:
        metric_expr = (
            f'1 - (sum(rate(http_requests_total{{service="{config.service}",status=~"[45].."}}[5m])) '
            f'/ sum(rate(http_requests_total{{service="{config.service}"}}[5m])))'
        )
    else:  # LATENCY
        threshold_s = (config.latency_threshold_ms or 5000) / 1000
        metric_expr = (
            f'sum(rate(http_request_duration_seconds_bucket{{service="{config.service}",'
            f'le="{threshold_s}"}}[5m])) '
            f'/ sum(rate(http_request_duration_seconds_count{{service="{config.service}"}}[5m]))'
        )

    alert_name_base = config.name.replace(" ", "_").replace("-", "_")

    rules_dict = {
        "groups": [
            {
                "name": f"slo.{config.service}",
                "rules": [
                    {
                        "alert": f"{alert_name_base}_Warning",
                        "expr": f"{metric_expr} < {warning_target}",
                        "for": "5m",
                        "labels": {
                            "severity": AlertSeverity.WARNING,
                            "service": config.service,
                            "slo": config.name,
                        },
                        "annotations": {
                            "summary": f"SLO warning: {config.name} SLI below {warning_target * 100:.2f}%",
                            "description": (
                                f"The {config.sli_type} SLI for {config.service} has been below "
                                f"{warning_target * 100:.3f}% for 5+ minutes. "
                                f"SLO target: {config.target * 100:.3f}%."
                            ),
                            "runbook_url": config.runbook_url,
                        },
                    },
                    {
                        "alert": f"{alert_name_base}_Critical",
                        "expr": f"{metric_expr} < {critical_target}",
                        "for": "2m",
                        "labels": {
                            "severity": AlertSeverity.CRITICAL,
                            "service": config.service,
                            "slo": config.name,
                        },
                        "annotations": {
                            "summary": f"SLO CRITICAL: {config.name} SLI below {critical_target * 100:.2f}%",
                            "description": (
                                f"The {config.sli_type} SLI for {config.service} has critically degraded. "
                                f"Current threshold: {critical_target * 100:.3f}%. "
                                f"SLO target: {config.target * 100:.3f}%."
                            ),
                            "runbook_url": config.runbook_url,
                        },
                    },
                ],
            }
        ]
    }

    yaml_str = yaml.dump(rules_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Generated alert rules for SLO '%s'", config.name)
    return yaml_str


# ── 5. Example usage ───────────────────────────────────────────────────────────

def _make_sample_records(
    service: str,
    total: int,
    failure_rate: float,
    base_time: datetime | None = None,
) -> list[RequestRecord]:
    """Generate synthetic request records for demo/test purposes."""
    if base_time is None:
        base_time = datetime.now(UTC)

    records: list[RequestRecord] = []
    for i in range(total):
        is_success = (i / total) >= failure_rate
        records.append(
            RequestRecord(
                request_id=f"req-{i:05d}",
                timestamp=base_time - timedelta(minutes=i),
                success=is_success,
                latency_ms=200.0 if is_success else 8000.0,
                status_code=200 if is_success else 503,
                service=service,
            )
        )
    return records


if __name__ == "__main__":
    # Define an SLO
    availability_slo = SLOConfig(
        name="LLM Gateway Availability",
        service="llm-gateway",
        sli_type=SLIType.AVAILABILITY,
        target=0.999,  # 99.9%
        window_days=30,
        runbook_url="https://wiki.example.com/runbooks/llm-gateway-availability",
    )

    latency_slo = SLOConfig(
        name="LLM Gateway Latency P95",
        service="llm-gateway",
        sli_type=SLIType.LATENCY,
        target=0.95,  # 95% of requests under threshold
        latency_threshold_ms=5000.0,
        window_days=7,
    )

    logger.info("Allowable downtime: %.1f min/month", availability_slo.allowable_downtime_minutes)

    # Simulate 10,000 requests with 0.05% failure rate (under budget)
    records = _make_sample_records("llm-gateway", total=10_000, failure_rate=0.0005)
    report = compute_sli(availability_slo, records)

    logger.info("SLO met: %s", report.slo_met)
    logger.info("Error budget remaining: %.1f%%", report.error_budget_remaining_pct)
    logger.info("Burn rate: %.2fx", report.budget_burn_rate)

    # Simulate a bad day: 2% failure rate (blows the budget)
    bad_records = _make_sample_records("llm-gateway", total=1_000, failure_rate=0.02)
    bad_report = compute_sli(availability_slo, bad_records)
    logger.info("After bad day — budget exhausted: %s", bad_report.is_budget_exhausted)

    # Generate alert rules YAML
    alert_yaml = generate_alert_rules(availability_slo)
    logger.info("=== Alert rules YAML ===\n%s", alert_yaml)

    # Example alert rules output (for reference):
    EXAMPLE_PROMETHEUS_ALERT_YAML = """
groups:
  - name: slo.llm-gateway
    rules:
      - alert: LLM_Gateway_Availability_Warning
        expr: |
          sum(rate(http_requests_total{service="llm-gateway",status!~"5.."}[5m]))
          / sum(rate(http_requests_total{service="llm-gateway"}[5m])) < 0.94905
        for: 5m
        labels:
          severity: warning
          service: llm-gateway
          slo: LLM Gateway Availability
        annotations:
          summary: "SLO warning: LLM Gateway Availability SLI below 94.91%"
          description: |
            The availability SLI for llm-gateway has been below 94.905% for 5+ minutes.
            SLO target: 99.900%.
          runbook_url: https://wiki.example.com/runbooks/llm-gateway-availability
    """
    logger.info("Example alert YAML structure shown above.")
```

### Test file

```python
# tests/unit/test_slo_engine.py
"""Unit tests for SLO computation and alert rule generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import yaml

from slo_engine import (
    AlertSeverity,
    RequestRecord,
    SLIType,
    SLOConfig,
    SLOReport,
    compute_sli,
    generate_alert_rules,
)


def _record(
    service: str,
    success: bool,
    status_code: int,
    latency_ms: float = 200.0,
    offset_minutes: int = 0,
) -> RequestRecord:
    return RequestRecord(
        request_id=f"r-{offset_minutes}",
        timestamp=datetime.now(UTC) - timedelta(minutes=offset_minutes),
        success=success,
        latency_ms=latency_ms,
        status_code=status_code,
        service=service,
    )


class TestSLOConfig:
    def test_error_budget_fraction(self) -> None:
        cfg = SLOConfig(name="t", service="s", sli_type=SLIType.AVAILABILITY, target=0.999)
        assert abs(cfg.error_budget_fraction - 0.001) < 1e-9

    def test_allowable_downtime(self) -> None:
        cfg = SLOConfig(name="t", service="s", sli_type=SLIType.AVAILABILITY, target=0.999, window_days=30)
        # 0.001 × 30 × 24 × 60 = 43.2 minutes
        assert abs(cfg.allowable_downtime_minutes - 43.2) < 0.01

    def test_latency_slo_requires_threshold(self) -> None:
        with pytest.raises(ValueError, match="latency_threshold_ms"):
            SLOConfig(name="t", service="s", sli_type=SLIType.LATENCY, target=0.95)

    def test_target_below_50_rejected(self) -> None:
        with pytest.raises(ValueError, match="50%"):
            SLOConfig(name="t", service="s", sli_type=SLIType.AVAILABILITY, target=0.4)


class TestComputeSLI:
    def test_all_success_meets_slo(self) -> None:
        cfg = SLOConfig(name="t", service="svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        records = [_record("svc", success=True, status_code=200, offset_minutes=i) for i in range(100)]
        report = compute_sli(cfg, records)
        assert report.slo_met is True
        assert report.sli_value == 1.0
        assert report.error_budget_remaining_pct == 100.0

    def test_high_failure_rate_breaches_slo(self) -> None:
        cfg = SLOConfig(name="t", service="svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        # 10% failures — far exceeds 0.1% budget
        records = (
            [_record("svc", success=False, status_code=503, offset_minutes=i) for i in range(10)]
            + [_record("svc", success=True, status_code=200, offset_minutes=i + 10) for i in range(90)]
        )
        report = compute_sli(cfg, records)
        assert report.slo_met is False
        assert report.is_budget_exhausted is True

    def test_empty_records_vacuously_compliant(self) -> None:
        cfg = SLOConfig(name="t", service="svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        report = compute_sli(cfg, [])
        assert report.total_requests == 0
        assert report.slo_met is True

    def test_records_outside_window_excluded(self) -> None:
        cfg = SLOConfig(name="t", service="svc", sli_type=SLIType.AVAILABILITY, target=0.5, window_days=7)
        now = datetime.now(UTC)
        old_failure = RequestRecord(
            request_id="old",
            timestamp=now - timedelta(days=10),  # outside 7-day window
            success=False,
            latency_ms=500.0,
            status_code=503,
            service="svc",
        )
        recent_success = _record("svc", success=True, status_code=200)
        report = compute_sli(cfg, [old_failure, recent_success])
        # Only the recent success is in window
        assert report.total_requests == 1
        assert report.slo_met is True

    def test_latency_sli_counts_fast_requests(self) -> None:
        cfg = SLOConfig(
            name="t",
            service="svc",
            sli_type=SLIType.LATENCY,
            target=0.95,
            latency_threshold_ms=1000.0,
        )
        fast = [_record("svc", success=True, status_code=200, latency_ms=500.0, offset_minutes=i) for i in range(95)]
        slow = [_record("svc", success=True, status_code=200, latency_ms=2000.0, offset_minutes=i + 95) for i in range(5)]
        report = compute_sli(cfg, fast + slow)
        assert abs(report.sli_value - 0.95) < 0.001
        assert report.slo_met is True

    def test_different_service_ignored(self) -> None:
        cfg = SLOConfig(name="t", service="svc-a", sli_type=SLIType.AVAILABILITY, target=0.999)
        records = [_record("svc-b", success=False, status_code=503)]  # different service
        report = compute_sli(cfg, records)
        assert report.total_requests == 0


class TestGenerateAlertRules:
    def test_yaml_parses(self) -> None:
        cfg = SLOConfig(name="Test SLO", service="my-svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        yaml_str = generate_alert_rules(cfg)
        parsed = yaml.safe_load(yaml_str)
        assert "groups" in parsed
        assert len(parsed["groups"][0]["rules"]) == 2

    def test_warning_before_critical(self) -> None:
        cfg = SLOConfig(name="Test SLO", service="my-svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        yaml_str = generate_alert_rules(cfg)
        parsed = yaml.safe_load(yaml_str)
        rules = parsed["groups"][0]["rules"]
        warning_rule = next(r for r in rules if r["labels"]["severity"] == "warning")
        critical_rule = next(r for r in rules if r["labels"]["severity"] == "critical")
        # Warning threshold should be higher (less strict) than critical
        # Both are embedded in the expr string; check labels instead
        assert warning_rule["for"] == "5m"
        assert critical_rule["for"] == "2m"

    def test_service_label_in_rules(self) -> None:
        cfg = SLOConfig(name="Test SLO", service="my-svc", sli_type=SLIType.AVAILABILITY, target=0.999)
        yaml_str = generate_alert_rules(cfg)
        assert "my-svc" in yaml_str
```

---

## Exercises

1. Add an `ErrorBudgetBurnRateAlert` to `generate_alert_rules` that fires when the budget is being consumed at more than 2× the sustainable rate (burn rate > 2.0). Use a 1-hour PromQL window. This pattern is described in Google's SRE workbook as "multi-window burn rate alerting".

2. Extend `SLOConfig` with a `burn_rate_thresholds: list[tuple[float, str]]` field (e.g., `[(14.4, "critical"), (6.0, "warning")]`) and generate one alert rule per threshold. Write a test verifying that three thresholds generate three rules.

3. Write a function `project_budget_exhaustion(report: SLOReport, window_days: int) -> datetime | None` that, given the current burn rate, estimates the UTC datetime when the error budget will be exhausted. Return `None` if the budget is not being consumed. Write tests covering burn rate < 1, burn rate > 1, and zero consumption.

4. Add a `composite_slo` function that takes a list of `SLOReport` objects and returns a single composite `slo_met` status. The composite SLO is met only if all individual SLOs are met. Write a test for all-pass, one-fail, and all-fail scenarios.

(Solutions: ../solutions/week-11/day-6.md)

---

## Self-check

- [ ] I can define SLI, SLO, and SLA in one sentence each and explain the relationship between them.
- [ ] I can compute the error budget for a 99.9% SLO over 30 days and explain what it means for engineering velocity decisions.
- [ ] I can write a Prometheus alerting rule YAML with `for`, `labels`, and `annotations` fields and explain the role of each.
- [ ] I can explain what "burn rate" means in the context of error budgets and why a burn rate > 1.0 is dangerous.
