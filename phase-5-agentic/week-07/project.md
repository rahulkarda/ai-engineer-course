# Week 07 Project — Multi-Agent Research Assistant

> Time: ~4 hr (Saturday) + polish on Sunday  ·  Builds on: weeks 1–6 primitives

## Project goal

Build a production-quality **multi-agent research assistant** that accepts a complex question, decomposes it, researches sub-topics in parallel, synthesises findings, fact-checks with debate, and produces a structured final report — all with typed inter-agent messaging and shared blackboard state.

## Domain

Software engineering knowledge base. Example question:
> "Compare the trade-offs between PostgreSQL and MongoDB for a high-write, analytical workload requiring complex joins."

The system must handle any knowledge question without domain-specific hardcoding.

## Required architecture

```
User → Supervisor
         │
         ├─ QueryAnalyser        (decomposes question → sub-queries)
         │
         ├─ ResearcherPool [×N]  (concurrent; N = len(sub-queries))
         │
         ├─ Synthesiser          (merges findings → draft + claims)
         │
         ├─ FactChecker          (verifies each claim)
         │   └─ optional DebateRound (Synthesiser ↔ FactChecker)
         │
         └─ Publisher            (formats FinalReport)

State: Blackboard (namespaced sections)
Comms: MessageBus or direct function calls (your choice)
```

## Technical requirements

### Must have (pass gate)

- [ ] **5 distinct agents** with single-responsibility roles
- [ ] **Pydantic models** for all data crossing agent boundaries (minimum: `SubQuery`, `ResearchFinding`, `SynthesisOutput`, `FactCheckResult`, `FinalReport`)
- [ ] **Concurrent research**: sub-queries processed in parallel (`ThreadPoolExecutor` or `asyncio.gather`)
- [ ] **Blackboard** used for at least 3 stages (queries → research → results)
- [ ] **Model usage**: use `claude-haiku-5-5` for fast tasks (query analysis, fact-checking), `claude-sonnet-5-5` for synthesis; never `claude-opus-5-5` without user confirmation (cost)
- [ ] **All LLM keys** via `os.getenv("ANTHROPIC_API_KEY")` — zero hardcoded keys
- [ ] **Tests**: `pytest` suite with all LLM calls mocked — zero real API calls in tests
- [ ] **Logging**: `logging` module at each agent boundary, not `print`
- [ ] **Error handling**: a researcher failing mid-pipeline should not crash the entire run; log and continue with remaining findings

### Should have (good grade)

- [ ] **Type hints** on every function signature
- [ ] **Retry logic** with exponential backoff on `anthropic.APIError`
- [ ] **Debate round** between Synthesiser and FactChecker (≥2 rounds, judge resolves)
- [ ] **Pipeline metrics**: wall-clock time per stage, total LLM call count
- [ ] **Configurable** researcher pool size (env var `RESEARCHER_POOL_SIZE`, default 2)

### Nice to have (excellence)

- [ ] **Async version** using `asyncio` + `anthropic.AsyncAnthropic`
- [ ] **CLI interface** (`python main.py "your question here"`) with structured output
- [ ] **JSON report export** (`report.json`) in addition to console output
- [ ] **Cache layer** on `Blackboard`: if the same sub-query appears in a future run, return cached result

## File structure

```
week07_project/
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── blackboard.py
│   ├── bus.py              # optional if using direct calls
│   ├── supervisor.py
│   └── agents/
│       ├── __init__.py
│       ├── base.py
│       ├── query_analyser.py
│       ├── researcher.py
│       ├── synthesiser.py
│       ├── fact_checker.py
│       └── publisher.py
├── tests/
│   ├── conftest.py         # shared fixtures, mock client factory
│   ├── test_models.py
│   ├── test_blackboard.py
│   ├── test_agents.py      # unit tests per agent
│   └── test_pipeline.py    # integration test (fully mocked)
├── main.py
├── pyproject.toml
└── .env.example
```

`.env.example`:
```
ANTHROPIC_API_KEY=your_key_here
RESEARCHER_POOL_SIZE=2
```

`pyproject.toml` (minimum):
```toml
[project]
name = "week07-project"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

## Assessment criteria

| Criterion | Weight | Description |
|---|---|---|
| Architecture | 25% | Correct topology, separation of concerns, no god classes |
| Functionality | 25% | Pipeline runs end-to-end, produces valid `FinalReport` |
| Testing | 25% | All tests pass; LLM mocked; coverage ≥ 70% |
| Code quality | 15% | Type hints, Pydantic models, logging, ruff passes |
| Documentation | 10% | Inline docstrings, README explains how to run |

**Pass gate: 75%.**

## Submission checklist

- [ ] `pytest tests/ -v` passes (zero failures, zero real API calls)
- [ ] `ruff check src/ tests/` returns no errors
- [ ] `main.py` runs against real API and produces a `FinalReport`
- [ ] `.env` is in `.gitignore`; `.env.example` exists
- [ ] No `print()` statements — use `logging`
- [ ] No hardcoded model strings other than `claude-haiku-5-5` and `claude-sonnet-5-5`

## Extension ideas

If you finish early, explore:

1. **Streaming output**: use `client.messages.stream()` so the synthesis appears token-by-token in the terminal
2. **Tool use**: give `ResearcherAgent` a web search tool (mock it in tests)
3. **Persistent blackboard**: swap in-memory blackboard for SQLite-backed store
4. **Multi-language**: add a `TranslatorAgent` that translates the final report to a user-specified language
5. **Evaluation**: add a `QualityEvaluatorAgent` that scores the final report 0–10 on accuracy, completeness, and clarity

(Reference implementation: ../solutions/week-07/project.md)
