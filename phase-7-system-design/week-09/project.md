# Week 09 Project — Incident Lab

## Overview

Build `incident_lab/` — a self-contained Python package with three buggy simulated services, diagnostic tooling, a circuit breaker, and a full test suite. The project consolidates all week's skills into one runnable, testable artefact.

## Context

Production engineers need *reproducible* failure scenarios for onboarding, chaos drills, and testing monitoring setups. This project produces exactly that: a portable "incident lab" that can spin up any of three realistic failure modes on demand.

## Week connection

| Day | Skill | Applied in project |
|-----|-------|-------------------|
| Day 1 | Systematic debugging + traces | `diagnostics/tracer.py` + `find_critical_path` |
| Day 2 | Memory leaks + pool exhaustion | `services/order_processor.py` |
| Day 3 | Circuit breaker + cascading | `diagnostics/circuit_breaker.py` |
| Day 4 | LLM cost/hallucination | `services/agent_service.py` |
| Day 5 | Incident timeline + RCA | `diagnostics/incident.py` |
| Day 6 | Scenario practice | All three services |

## Deliverables

```
incident_lab/
├── __init__.py
├── services/
│   ├── __init__.py
│   ├── order_processor.py   # pool exhaustion bug + fix
│   ├── cache_service.py     # thundering herd bug + fix
│   └── agent_service.py     # unbounded token accumulation + fix
├── diagnostics/
│   ├── __init__.py
│   ├── tracer.py            # Span + Tracer + find_critical_path
│   ├── metrics.py           # Counter + Histogram (p50/p99)
│   ├── circuit_breaker.py   # 3-state circuit breaker
│   └── incident.py          # Incident + WhyChain dataclasses
├── cli.py                   # click CLI
└── tests/
    ├── test_order_processor.py
    ├── test_cache_service.py
    ├── test_agent_service.py
    ├── test_diagnostics.py
    └── test_circuit_breaker.py
```

## Grading criteria

| Criterion | Points |
|-----------|--------|
| All 10 required tests pass | 40 |
| Circuit breaker: all 3 states tested | 15 |
| `find_critical_path` correct | 10 |
| `p99()` within 5% of expected | 10 |
| `run_buggy()` demonstrably reproduces failure | 10 |
| `run_fixed()` demonstrably passes clean | 10 |
| Code quality: types, logging, no secrets | 5 |
| **Total** | **100** |

Pass threshold: ≥ 75 points.

## Submission checklist

- [ ] `pytest tests/ -v` output copied into `NOTES.md`
- [ ] No `# TODO` markers remain in source
- [ ] `ruff check incident_lab/` exits 0
- [ ] All LLM calls mocked — `pytest` runs without `ANTHROPIC_API_KEY`
- [ ] `NOTES.md` contains one-sentence bug description per service

## Extension challenges (optional, not graded)

1. Add a `chaos_monkey.py` that randomly activates one of the three failure modes every 30 seconds and logs which mode is active — useful for monitoring drills.
2. Instrument `diagnostics/metrics.py` to emit Prometheus-format text (`# HELP`, `# TYPE`, metric lines) so a real Prometheus scraper could ingest it.
3. Build a `PostMortem.to_html() -> str` that renders the post-mortem as a self-contained HTML page for sharing.

(Solutions: ../solutions/week-09/project.md)
