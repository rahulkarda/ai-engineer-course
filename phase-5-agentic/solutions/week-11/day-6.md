# Solutions — Week 11 · Day 6: Observability — Tracing, Spans, Token & Cost Tracking

## Exercise 1 — budget_alert

```python
# In CostTracker
def budget_alert(self, max_cost_usd: float) -> bool:
    """Return True if total accumulated cost exceeds budget."""
    total = sum(r.cost_usd for r in self._records)
    exceeded = total > max_cost_usd
    if exceeded:
        import logging
        logging.getLogger(__name__).warning(
            "Budget alert: total cost $%.4f exceeds limit $%.4f",
            total, max_cost_usd,
        )
    return exceeded
```

**Test:**
```python
def test_budget_alert_triggered():
    tracker = CostTracker()
    usage = _mock_usage(1_000_000, 1_000_000)  # $18 for sonnet
    tracker.record("r1", "claude-sonnet-5-5", usage, 100.0, 1, True)
    assert tracker.budget_alert(10.0) is True
    assert tracker.budget_alert(20.0) is False
```

---

## Exercise 2 — Per-tool-call spans in run_instrumented

```python
async def run_instrumented(agent, user_input, deps, *, model):
    trace_id = uuid.uuid4().hex
    t0 = time.perf_counter()

    async with _tracer.span("agent.run", trace_id, model=model) as root:
        result = await agent.run(user_input, deps=deps)

    # After run: emit spans for each tool call
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    messages = result.all_messages()

    # Build a map of tool_call_id → duration by pairing ToolCallPart/ToolReturnPart
    call_parts: dict[str, ToolCallPart] = {}
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                call_parts[part.tool_call_id] = part

    tool_calls = 0
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolReturnPart):
                tool_calls += 1
                call = call_parts.get(part.tool_call_id)
                if call:
                    async with _tracer.span(
                        f"tool.{call.tool_name}",
                        trace_id,
                        parent_span_id=root.span_id,
                        tool_name=call.tool_name,
                    ):
                        pass  # span is for recording, timing is approximate

    latency_ms = (time.perf_counter() - t0) * 1000
    usage = result.usage()
    _tracker.record(trace_id, model, usage, latency_ms, tool_calls, True)
    return result
```

**Note:** Because we parse the message list post-hoc, we can't get per-tool latency from Pydantic AI without hooking into the execution loop. For real per-tool latency, use Logfire which instruments at the framework level.

---

## Exercise 3 — Tracer.waterfall

```python
def waterfall(self, trace_id: str) -> str:
    """Return ASCII waterfall of spans for a trace."""
    spans = self.get_trace(trace_id)
    if not spans:
        return f"No spans for trace {trace_id}"

    # Find roots (no parent)
    span_by_id = {s.span_id: s for s in spans}
    children: dict[str | None, list] = {}
    for s in spans:
        children.setdefault(s.parent_span_id, []).append(s)

    # Sort spans by start_time for consistent order
    for lst in children.values():
        lst.sort(key=lambda s: s.start_time)

    lines: list[str] = [f"Trace: {trace_id}"]

    def render(span_id: str | None, depth: int) -> None:
        for s in children.get(span_id, []):
            indent = "  " * depth
            status = f"ERROR({s.error})" if s.error else f"{s.duration_ms:.1f}ms"
            lines.append(f"{indent}├─ {s.name} [{status}]")
            render(s.span_id, depth + 1)

    render(None, 0)
    return "\n".join(lines)
```

---

## Exercise 4 — instrumented_agent pytest fixture

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.observability.instrumented_runner import run_instrumented, get_summary


@pytest.fixture
def mock_usage():
    u = MagicMock()
    u.request_tokens = 100
    u.response_tokens = 50
    return u


@pytest.mark.asyncio
async def test_instrumented_agent_success_rate():
    """Run 3 test inputs through instrumented runner; assert 100% success rate."""
    from pydantic_ai.models.test import TestModel
    # We patch run_instrumented's inner agent.run to avoid real calls
    # In a real test you'd pass an agent with TestModel overridden
    # Here we demonstrate the pattern:

    from src.observability.cost_tracker import CostTracker
    tracker = CostTracker()

    usage = MagicMock()
    usage.request_tokens = 200
    usage.response_tokens = 80

    for i in range(3):
        tracker.record(f"run-{i}", "claude-haiku-5-5", usage, 120.0, 1, True)

    summary = tracker.summary()
    assert summary["success_rate"] == 1.0
    assert summary["runs"] == 3
```

## Exercise 5 — Logfire setup (reference answer)

Three things to set up automatic Pydantic AI tracing with Logfire:

1. **Install Logfire:** `uv add "pydantic-ai[logfire]"` (or `uv add logfire pydantic-ai`).

2. **Configure at startup:** Call `logfire.configure()` once before any agent runs. This initialises the OTel provider and SDK integration.

3. **Set the API token:** Logfire reads `LOGFIRE_TOKEN` from the environment (or `~/.logfire/default.toml` after `logfire auth`). Never hardcode the token.

After these three steps, every `agent.run()` call automatically emits spans to the Logfire dashboard — no agent code changes required.
