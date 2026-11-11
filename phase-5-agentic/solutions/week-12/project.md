# Week 12 Project — Grading Guide & Reference Implementation

## Grading checklist

### Deliverable 1: Source code (20%)
- All 10 modules exist and are importable: `config`, `core/registry`, `core/agent`, `core/planner`, `memory/store`, `routing/router`, `hitl/gate`, `audit/logger`, `orchestration/coordinator`, `a2a/protocol`
- `from finops_agent import AgentCore, ToolRegistry, HumanApprovalGate` succeeds
- No hardcoded secrets; all keys via `os.getenv`
- Model IDs are only `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`

### Deliverable 2: Tests (25%)
- `pytest tests/` exits 0
- All LLM calls patched with `unittest.mock.patch`
- Minimum 12 tests covering: registry, router, HITL gate (auto-approve + block), audit log, memory, tools, A2A, eval harness
- No test calls a real API

### Deliverable 3: Docker (15%)
- `Dockerfile` present and uses multi-stage build (builder + runtime)
- `docker-compose.yml` uses `${ANTHROPIC_API_KEY}` not hardcoded value
- `docker build .` exits 0 (no LLM calls needed during build)
- Health endpoint reachable at `/health`

### Deliverable 4: HITL gate (15%)
- `HumanApprovalGate.request_approval` returns `PENDING` for `risk_level="critical"` tools
- `gate.resolve(request_id, approved=True, reviewer="admin")` changes status to `APPROVED`
- `approve_payment` in `controlled.py` correctly routes through the gate

### Deliverable 5: Eval harness (15%)
- `eval/dataset.jsonl` has >= 6 cases
- `EvalHarness.run_all()` returns a list of `EvalResult`
- `EvalHarness.report()` returns `pass_rate`, `avg_latency_ms`, `avg_score`
- At least one case tagged `"hitl"` and one tagged `"error"`

### Deliverable 6: README + architecture diagram (10%)
- README has: project description, setup instructions (`pip install -e .`), env var list (`.env.example`), API reference (endpoints table)
- Architecture diagram present (ASCII, Mermaid, or image) showing: API → Router → Coordinator → Agents → Tools → HITL Gate → Audit

## Reference architecture diagram (ASCII)

```
+----------------------------------------------------------+
|                   FastAPI HTTP API                        |
|  POST /agent/run  GET /health  GET /metrics              |
|  POST /hitl/resolve  GET /hitl/pending                   |
+---------------------------+------------------------------+
                            |
                     +------v------+
                     | AgentRouter | (keyword + LLM routing)
                     +------+------+
                            |
             +--------------v--------------+
             |   MultiAgentCoordinator     |
             |  (parallel ThreadPoolExec)  |
             +------+------------------+---+
                    |                  |
           +--------v---+   +----------v--+
           |FinanceAgent|   |  OpsAgent   |
           | ReActPlann.|   |  (stub)     |
           +--------+---+   +-------------+
                    |
           +--------v-----------+
           |   ToolRegistry     |
           | retrieve_invoice   |
           | validate_payment   |
           | check_budget       |
           | approve_payment ---+---> HumanApprovalGate
           +--------+-----------+          |
                    |               +------v------+
           +--------v------+        |Human Reviewer|
           |SQLiteMemory   |        +-------------+
           |Store          |
           +---------------+

Cross-cutting:
  AuditLogger (JSONL) <- every agent event
  MetricsCollector <- every span + counter
  A2AChannel <- supervisor <-> sub-agent messaging
  MCP Server (/mcp/invoke) <- external tool consumers
```

## Common mistakes to flag

1. **Hardcoded model ID** — any `claude-*-4-*` string in code is wrong. Only `-5-5` suffix.
2. **Global instantiation** — `memory = SQLiteMemoryStore()` at module level breaks tests.
3. **print() instead of logging** — deduct from code quality score.
4. **Untyped tool function** — every tool function must have full type hints.
5. **Tests import real anthropic.Anthropic without mock** — tests will fail in CI without a real key.
6. **docker-compose.yml with hardcoded key** — security violation, automatic fail on Deliverable 3.

## Interview talking points (reference answers)

1. **Why supervisor/worker over monolithic agent?** Supervisors decompose complex tasks and dispatch to specialists in parallel, reducing latency and improving accuracy per subtask. Monolithic agents exceed context windows on complex enterprise workflows.

2. **Why HITL gates at the tool level?** The tool registry is the single chokepoint where all agent actions pass. Inserting the gate there ensures no path (direct API call, MCP call, A2A routed call) bypasses human review for critical actions.

3. **Why JSONL for audit logs?** Append-only JSONL is crash-safe (no corruption on partial write), trivially grep-able, streamable to SIEM tools, and readable without a database. Each line is a self-contained Pydantic-validated event.

4. **Why eval harness checks structural properties not strings?** LLMs are stochastic — the same correct answer can be phrased many ways. Checking "was the right tool called?" and "does the answer mention INV-001?" is both more robust and more meaningful as a behavioral test.

5. **Why pydantic-settings for config?** It reads from env vars and `.env` files in priority order, validates types at startup (e.g., rejects non-claude model IDs), and never requires manual `os.getenv` calls scattered across modules. One `AgentSettings()` instance is the single source of truth.
