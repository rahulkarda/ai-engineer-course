# Solutions — Week 09 · Day 5

## Exercise 1 — time_to_detect

```python
from datetime import datetime, timezone


def time_to_detect(self) -> timedelta | None:
    """Time from actual fault start to first alert-fired event."""
    alert_events = [
        e for e in self.timeline
        if "alert" in e.action.lower() or "fired" in e.action.lower()
    ]
    if not alert_events:
        return None
    first_alert = min(alert_events, key=lambda e: e.ts)
    return first_alert.ts - self.started_at


# Patch onto Incident:
# Incident.time_to_detect = time_to_detect
```

## Exercise 2 — PostMortem dataclass

```python
from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass
class ActionItem:
    description: str
    owner: str
    due_date: date
    status: Literal["open", "in-progress", "done"] = "open"


@dataclass
class PostMortem:
    incident_id: str
    timeline: list[TimelineEvent]
    root_cause: str
    contributing_factors: list[str]
    action_items: list[ActionItem]

    def to_markdown(self) -> str:
        lines = [
            f"# Post-Mortem: {self.incident_id}",
            "",
            "## Root Cause",
            self.root_cause,
            "",
            "## Contributing Factors",
        ]
        for factor in self.contributing_factors:
            lines.append(f"- {factor}")
        lines += ["", "## Action Items", "| # | Description | Owner | Due | Status |", "|---|-------------|-------|-----|--------|"]
        for i, item in enumerate(self.action_items, 1):
            lines.append(
                f"| {i} | {item.description} | {item.owner} | {item.due_date} | {item.status} |"
            )
        lines += ["", "## Timeline"]
        for event in sorted(self.timeline, key=lambda e: e.ts):
            lines.append(f"- `{event.ts.isoformat()}` **{event.actor}**: {event.action} — {event.details}")
        return "\n".join(lines)


def test_postmortem_to_markdown_contains_action_items():
    pm = PostMortem(
        incident_id="INC-001",
        timeline=[],
        root_cause="Missing index",
        contributing_factors=["No migration linter"],
        action_items=[
            ActionItem("Add migration linter", "DBA", date(2026, 12, 1)),
            ActionItem("Add latency alert", "SRE", date(2026, 11, 20)),
        ],
    )
    md = pm.to_markdown()
    assert "Add migration linter" in md
    assert "Add latency alert" in md
    assert "Missing index" in md
```

## Exercise 3 — SLOBurnRateAlert

```python
from datetime import datetime, timezone
from typing import NamedTuple


def slo_burn_rate_alert(
    events: list[tuple[datetime, bool]],  # (timestamp, is_error)
    window_minutes: int = 10,
    error_budget_pct: float = 0.01,  # 1% = 99% SLO
    burn_multiplier: float = 2.0,
) -> bool:
    """Return True if burn rate exceeds burn_multiplier × expected rate."""
    if not events:
        return False

    cutoff = max(e[0] for e in events) - __import__("datetime").timedelta(minutes=window_minutes)
    window_events = [e for e in events if e[0] >= cutoff]
    if not window_events:
        return False

    error_count = sum(1 for _, is_err in window_events if is_err)
    observed_rate = error_count / len(window_events)
    return observed_rate > (error_budget_pct * burn_multiplier)


def test_all_ok():
    now = datetime.now(timezone.utc)
    events = [(now, False)] * 100
    assert slo_burn_rate_alert(events) is False


def test_50_pct_errors():
    now = datetime.now(timezone.utc)
    events = [(now, i % 2 == 0) for i in range(100)]
    assert slo_burn_rate_alert(events, error_budget_pct=0.01, burn_multiplier=2.0) is True


def test_5_pct_errors_below_threshold():
    now = datetime.now(timezone.utc)
    events = [(now, i < 5) for i in range(100)]  # 5% errors
    # 5% > 1% * 2 = 2% → True
    assert slo_burn_rate_alert(events, error_budget_pct=0.1, burn_multiplier=2.0) is False
```
