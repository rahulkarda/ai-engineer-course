# Solutions — Day 6: LiteLLM / Portkey Conceptual

## Exercise 1: Decision Matrix — Gateway Approach by Scenario

| Scenario | Recommended approach | Reasoning |
|---|---|---|
| **Early-stage startup** | LiteLLM (self-hosted proxy) | Zero infra cost, multi-provider out of the box, fast setup. Build custom only if you have very specific requirements that LiteLLM can't meet. |
| **Enterprise (internal platform)** | Build custom gateway | Full control over authN/authZ, audit logs, cost attribution per business unit, SOC-2 compliance hooks. LiteLLM proxy can be a starting point, but large enterprises typically outgrow it. |
| **Regulated industry (finance, health)** | Build custom + Portkey for observability | Data residency constraints, explainability requirements, and audit trails require custom routing logic. Portkey adds LLM observability without sending data to a third-party AI provider. |
| **Cost-sensitive / high volume** | LiteLLM with budget_manager | LiteLLM's built-in spend tracking, model tiering, and fallbacks cover most cost-control needs without engineering overhead. |

**Build vs. LiteLLM decision heuristic:**
- Need it working in a day → LiteLLM
- Need it to match company security/audit standards → build (or extend LiteLLM proxy with custom middleware)
- Need multi-provider failover with zero code → LiteLLM `fallbacks=`
- Need tenant-level cost attribution to internal cost centers → build (LiteLLM's budget tracking is user-level, not multi-tenant hierarchical)

---

## Exercise 2: Custom Gateway → LiteLLM Component Mapping

| Custom gateway component | LiteLLM equivalent | Notes |
|---|---|---|
| `BaseProvider` + `ProviderRegistry` | `litellm.completion()` unified interface | LiteLLM normalizes provider APIs behind a single `completion(model=..., messages=...)` call. The `model` string encodes the provider: `"anthropic/claude-sonnet-5-5"`, `"openai/gpt-4o"`. |
| `FallbackChain` | `fallbacks=` parameter | Pass a list of fallback model strings: `fallbacks=["anthropic/claude-haiku-5-5", "openai/gpt-4o-mini"]`. LiteLLM retries sequentially on exception. |
| `RateLimiter` | LiteLLM proxy `rate_limit` config | Set per-user or per-key RPM limits in `config.yaml`. Not available in the Python SDK — requires the proxy server. |
| `ExactCache` | `litellm.cache` with `type="local"` | `litellm.cache = Cache(type="local")` enables in-process exact-match caching. Redis backend: `type="redis"`. Semantic cache: `type="redis-semantic"`. |
| `CostAttributor` | `litellm.spend_logs` + `budget_manager` | `litellm.success_callback = ["langfuse"]` streams cost events. `BudgetManager` enforces per-user budgets. For tenant-level multi-org tracking, custom callback hooks are needed. |

**Callback hook pattern for custom cost tracking:**
```python
import litellm

def track_cost(kwargs, completion_response, start_time, end_time):
    tenant_id = kwargs.get("metadata", {}).get("tenant_id", "unknown")
    cost = litellm.completion_cost(completion_response)
    my_cost_attributor.record(tenant_id, cost)

litellm.success_callback = [track_cost]
```

---

## Exercise 3: LiteLLM Fallback Config Pattern

```python
# LiteLLM fallback pattern (conceptual — shows config, does not call the API)
import litellm

# Optional: suppress verbose debug logs
litellm.set_verbose = False

# Configure caching (optional but recommended)
# litellm.cache = Cache(type="local")

async def complete_with_fallback(messages: list[dict]) -> str:
    """
    Attempt claude-sonnet-5-5 first; fall back to claude-haiku-5-5 on any error.

    Note: LiteLLM requires provider-prefixed model IDs:
      "anthropic/claude-sonnet-5-5"  not  "claude-sonnet-5-5"
    The unprefixed form may work if ANTHROPIC_API_KEY is set, but
    explicit prefixes are required when using multiple providers.
    """
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=messages,
        fallbacks=["anthropic/claude-haiku-5-5"],
        # Per-request metadata for cost tracking callbacks
        metadata={"tenant_id": "acme-corp"},
    )
    return response.choices[0].message.content


async def complete_with_budget(
    messages: list[dict],
    user_id: str,
    budget_manager: litellm.BudgetManager,
) -> str:
    """Pattern for per-user budget enforcement before calling."""
    if not budget_manager.is_valid_user(user_id=user_id):
        budget_manager.create_budget(
            total_budget=5.0,
            user=user_id,
            duration="daily",
        )

    if budget_manager.get_current_cost(user=user_id) >= budget_manager.get_total_budget(user=user_id):
        raise ValueError(f"User {user_id} has exceeded daily budget")

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=messages,
        user=user_id,
    )
    budget_manager.update_cost(completion_obj=response, user=user_id)
    return response.choices[0].message.content
```

**When NOT to use LiteLLM fallbacks:**
- Semantic fallback logic (e.g., "use sonnet only for code, haiku for everything else") — implement as routing in your own layer.
- Cross-tenant isolation — `fallbacks=` is request-scoped, not tenant-scoped; build tenant-aware routing on top.

---

## Exercise 4: Observability Signals and Alert Thresholds

Three essential signals to emit from a production gateway:

### Signal 1 — p99 Latency per Provider

**What:** 99th percentile end-to-end response time, broken down by provider and model.

**How to emit:**
```python
import time
start = time.monotonic()
response = await provider.complete(request)
latency_ms = (time.monotonic() - start) * 1000
logger.info("llm.latency provider=%s model=%s p99_sample_ms=%.1f",
            provider.provider_name(), request.model, latency_ms)
# emit to metrics: histogram_observe("gateway.latency_ms", latency_ms, tags={...})
```

**Alert trigger:** p99 > 5000 ms over a 5-minute window.
**Why 5s:** LLM responses vary widely; p99 < 5s is achievable for most models. Above 5s likely indicates provider degradation, not normal variance.

---

### Signal 2 — Error Rate per Provider

**What:** Percentage of requests that return an error (4xx from provider, timeout, network failure), per provider per model.

**How to emit:**
```python
try:
    response = await provider.complete(request)
    metrics.increment("gateway.requests", tags={"status": "ok", "provider": ...})
except Exception as e:
    metrics.increment("gateway.requests", tags={"status": "error", "provider": ...})
    logger.warning("provider error provider=%s error=%s", provider.provider_name(), type(e).__name__)
    raise
```

**Alert trigger:** Error rate > 5% over any rolling 5-minute window for a single provider.
**Why 5%:** A single flaky request is noise; 5% sustained indicates provider instability. Trigger fallback routing or circuit-break the provider.

---

### Signal 3 — Cost per Tenant per Hour

**What:** Rolling hourly spend per tenant, compared against expected rate derived from daily budget.

**How to calculate alert threshold:**
```python
# Alert when a tenant's hourly spend exceeds 2x their "expected" hourly rate.
# Expected hourly rate = daily_budget / 24
# Alert threshold = expected_hourly * 2

daily_budget_usd = 10.0
expected_hourly = daily_budget_usd / 24          # ~$0.42/hr
alert_threshold = expected_hourly * 2            # ~$0.83/hr

if tenant_hourly_cost > alert_threshold:
    logger.warning(
        "cost spike tenant=%s hourly_cost=%.4f threshold=%.4f",
        tenant_id, tenant_hourly_cost, alert_threshold,
    )
```

**Alert trigger:** Tenant's last-hour cost > `(daily_budget / 24) * 2`.
**Why 2x:** Normal usage varies; 2x the expected rate indicates a runaway loop, data pipeline error, or abuse — worth investigating before the tenant blows their daily budget by 10:00 AM.

---

### Summary table

| Signal | Metric type | Alert condition | Response |
|---|---|---|---|
| p99 latency / provider | Histogram | > 5000 ms (5-min p99) | Switch traffic to fallback provider |
| Error rate / provider | Counter ratio | > 5% (5-min window) | Circuit-break provider, alert on-call |
| Cost / tenant / hour | Gauge | > 2× expected hourly | Throttle tenant, notify admin |
