# Phase 6 · Week 09 · Day 6 — LiteLLM and Portkey: Conceptual Gateway Survey

> Objective: Understand the trade-off space between building your own gateway, using LiteLLM, and using Portkey — and map those choices to the gateway you built this week.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

---

## Concept

By Friday you built a working model gateway from scratch. Today steps back to survey the ecosystem: two popular abstractions that tackle the same problem, each with different trade-offs.

**Three options for any team:**

| Option | Description |
|---|---|
| **Roll your own** | What you did in days 1–5: providers, routing, cache, quota, cost |
| **LiteLLM** | OSS Python SDK + optional proxy server; 100+ provider integrations |
| **Portkey** | Managed SaaS gateway; semantic cache, guardrails, observability |

Understanding the ecosystem helps you make informed decisions for real projects — and helps you evaluate new tools by mapping them to primitives you already understand.

---

## How it works internally

### LiteLLM

LiteLLM is an open-source library that provides a unified `completion()` interface across 100+ LLM providers. It has two deployment modes:

**Library mode** — import and call directly in your Python code:
```python
import litellm

# Same call signature regardless of provider
response = litellm.completion(
    model="anthropic/claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**Proxy mode** — run as a standalone server that exposes an OpenAI-compatible REST API. Any client that speaks OpenAI's API (LangChain, OpenAI SDK, curl) can point at it without code changes:

```
LiteLLM Proxy (port 4000)
  POST /v1/chat/completions  →  routes to configured provider
```

**Key LiteLLM features:**

- **Provider abstraction**: change `model="anthropic/..."` to `model="openai/..."` without touching call logic
- **Fallbacks config** (declarative, in config YAML or code):

```python
# Conceptual pattern — do NOT run; requires litellm installed and keys set
import litellm

# Fallback: try sonnet, if it fails try haiku
response = litellm.completion(
    model="anthropic/claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Summarize this"}],
    fallbacks=["anthropic/claude-haiku-5-5"],
    num_retries=2,
)
```

- **Budget limits**: set `max_budget` in proxy config; requests are rejected once exceeded
- **Rate limiting**: RPM/TPM limits per key in the proxy config file
- **Callbacks**: register functions that fire on success/failure for logging

**Proxy config YAML pattern (conceptual):**
```yaml
model_list:
  - model_name: claude-smart
    litellm_params:
      model: anthropic/claude-sonnet-5-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-fast
    litellm_params:
      model: anthropic/claude-haiku-5-5
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  fallbacks:
    - {"claude-smart": ["claude-fast"]}

general_settings:
  max_budget: 100.0  # USD
  budget_duration: 30d
```

### Portkey

Portkey is a **managed SaaS gateway**. Your application calls Portkey's API; Portkey calls the actual LLM provider on your behalf.

**Key Portkey features:**

- **Semantic caching**: embeds request messages, stores responses; on similar (not identical) future requests returns cached response. Reduces cost on paraphrase repetition.
- **Guardrails**: pre/post-call content filters — PII detection, topic restrictions, output validation — configured in the Portkey dashboard.
- **Observability**: full request/response logging, cost dashboards, latency P50/P95/P99 — out of the box, zero instrumentation.
- **A/B testing**: configure traffic splits between models in the dashboard with no code changes.
- **Fallbacks and retries**: declarative config in API request headers or dashboard.

**Call pattern** (conceptual — requires Portkey account):
```python
# Conceptual — shows interface, not runnable without credentials
from portkey_ai import Portkey  # hypothetical import

client = Portkey(
    api_key="PORTKEY_API_KEY",       # os.getenv("PORTKEY_API_KEY")
    virtual_key="ANTHROPIC_VIRTUAL_KEY",
)

response = client.chat.completions.create(
    model="claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Decision matrix

| Dimension | Roll Your Own | LiteLLM (library) | LiteLLM (proxy) | Portkey |
|---|---|---|---|---|
| **Provider coverage** | Only what you build | 100+ | 100+ | 100+ |
| **Ops burden** | High | Low | Medium | None |
| **Data sovereignty** | Full | Full | Full | Data leaves your infra |
| **Semantic caching** | Build it | Not built-in | Not built-in | Included |
| **Guardrails** | Build it | Not built-in | Not built-in | Included |
| **Observability** | Build it | Via callbacks | Via callbacks | Dashboard included |
| **Cost** | Engineering time | Free | Free | SaaS pricing |
| **Customization** | Unlimited | High | Medium (config-driven) | Dashboard only |
| **Vendor lock-in** | None | Low (OSS) | Medium (config format) | High (SaaS dependency) |
| **Time to first request** | Days (as you saw) | Minutes | 30 minutes | Minutes |

### When to choose each

**Roll your own** when:
- You need custom routing logic that config files cannot express
- Data must never leave your infrastructure (regulated industries)
- You're building a product where the gateway itself is a differentiator
- Team has Python/infra expertise and time to maintain

**LiteLLM library** when:
- You want provider flexibility without a separate service
- You already have a FastAPI service and want to add model routing
- Open-source, no external dependencies preferred

**LiteLLM proxy** when:
- Multiple teams/languages need to call LLMs; unified endpoint simplifies auth
- You want OpenAI-compatible interface for drop-in LangChain/SDK compatibility
- Budget/rate-limit enforcement at the infra layer, not per-app

**Portkey** when:
- Speed matters more than control (POC, startup, small team)
- You need semantic caching and guardrails with zero build time
- Observability/analytics dashboards are a requirement
- Data residency is not a concern

---

## Mapping your gateway to LiteLLM equivalents

The gateway you built in days 1–5 directly maps to LiteLLM concepts:

| Your component | LiteLLM equivalent |
|---|---|
| `ProviderBase` abstract class | `litellm.completion()` unified interface |
| `ModelRouter` with policies | `router_settings` + `routing_strategy` in config |
| `FallbackChain` | `fallbacks` list in completion call or config |
| `ExactCache` | `litellm.cache` with `InMemoryCache` backend |
| `RateLimiter` | Proxy `rpm_limit` / `tpm_limit` per model |
| `QuotaTracker` | Proxy `max_budget` + `budget_duration` |
| `CostAttributor` | Proxy `spend` tracking + callbacks |
| `ModelABRouter` | No direct equivalent — LiteLLM's router uses latency/cost, not controlled experiments |

The main gap: LiteLLM's router optimizes for performance/cost dynamically; it does not support controlled A/B experiments with deterministic user assignment. For proper experimentation you still need to build that layer.

---

## Exercises

1. **Architecture decision record**: You're joining a team that currently calls Anthropic directly from each microservice (5 services, 3 languages). They want unified rate limiting, cost visibility, and the ability to swap models without deploying every service. Write a one-page ADR (Architecture Decision Record) recommending LiteLLM proxy, roll-your-own FastAPI gateway, or Portkey. Justify the choice with at least three concrete trade-off points from the decision matrix.

2. **Gap analysis**: Your custom gateway from days 1–5 is missing semantic caching. Design (do not implement) the interface for a `SemanticCache` class. What embedding model would you use? What similarity threshold? What are the correctness risks of serving a cached response to a "similar but not identical" query? Write your design as a docstring + type signature only.

3. **LiteLLM fallback config**: Write a LiteLLM proxy `config.yaml` (conceptual — no running required) that defines three model aliases (`claude-best`, `claude-smart`, `claude-fast`) mapping to `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` respectively, with fallback chain `best → smart → fast`. Include a `max_budget` of $50 per 30 days.

4. **Portkey vs. compliance**: A healthcare startup wants to use Portkey for their patient intake chatbot. List three specific questions their security/compliance team must answer before approving Portkey, and explain why each matters for HIPAA compliance.

(Solutions: ../solutions/week-09/day-6.md)

---

## Supplementary links

- LiteLLM docs: https://docs.litellm.ai/
- Portkey docs: https://portkey.ai/docs/

---

## Self-check

1. LiteLLM's proxy exposes an OpenAI-compatible endpoint. What specific advantage does this give teams that already use LangChain or the OpenAI Python SDK?

2. Portkey provides semantic caching. Explain why semantic caching can return an incorrect answer even when the similarity score is above the threshold — give a concrete example.

3. You built `ModelABRouter` with deterministic hash-based assignment. LiteLLM's router uses latency-based dynamic routing. Why are these two different problems, and when would you need both in the same system?
