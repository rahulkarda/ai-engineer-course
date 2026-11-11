# Week 03 Assessment — Reliability & Fault Tolerance

> Phase 6 · Production AI  ·  Time: ~1 hr  ·  Pass gate: 75%

---

## Conceptual (interview-style)

1. Explain the difference between **full jitter** and **decorrelated jitter** backoff strategies. When would you prefer one over the other?

2. An LLM tool call charges the user's payment account. A retry fires after a network timeout on the response. What mechanism prevents a double charge, and where should it be implemented?

3. A circuit breaker is in HALF-OPEN state and receives 5 simultaneous requests. What should happen? What's the risk if you allow all 5 through?

4. Your agent pipeline uses `asyncio.Queue(maxsize=50)` between a prompt-generator and an LLM-caller. The LLM gets rate-limited and slows to 1 req/s. Describe what happens to the queue and to the prompt-generator over the next 60 seconds.

5. Compare choreography sagas and orchestration sagas: give one production scenario where choreography is clearly better and one where orchestration is clearly better.

6. The token bucket allows bursting. For an LLM API with a strict tokens-per-minute (TPM) limit, is bursting beneficial or harmful? Justify.

7. Define steady-state in the context of chaos engineering. Why must you define it before injecting failures?

8. A `CircuitBreaker` wraps your primary LLM. The circuit opens. Your fallback chain calls a backup LLM directly, without its own circuit breaker. What's the failure mode? How do you fix it?

---

## Coding challenge

### Spec: `ReliableAgentNode`

Build a LangGraph-compatible node function that wraps an LLM call with full reliability. The node:

1. Accepts `state: AgentState` (Pydantic model, `prompt: str`, `idempotency_key: str | None`)
2. Calls a mock LLM via `ResilientLLMClient.complete()`
3. Returns updated state with `response: str`, `degraded: bool`, `attempt_count: int`
4. If the circuit breaker is open AND the fallback chain is exhausted, set `response = ""` and `error: str` on state — do not raise

### Acceptance criteria

- [ ] Node is a plain `async def` taking and returning `AgentState`
- [ ] Uses `ResilientLLMClient` from your week-03 toolkit
- [ ] All LLM calls are mocked — no real API calls in tests
- [ ] Three tests:
  - Happy path: `degraded=False`, `attempt_count >= 1`
  - 3-attempt retry success: `attempt_count == 3`
  - Total failure: `response == ""`, `error` is non-empty
- [ ] No print statements; logging only
- [ ] `AgentState` is a Pydantic v2 model

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Indicators |
|---|---|---|
| Independent implementation | 40% | Node works without consulting notes; retry/CB/fallback all exercised |
| Conceptual understanding | 25% | Answers to conceptual questions are precise and cover edge cases |
| Debugging & problem-solving | 20% | Correctly handles the total-failure path; no unhandled exceptions |
| Code quality & testing | 15% | Type hints throughout; tests use mocks; logging not print |

**Score breakdown:**
- Conceptual: 8 questions × 3 points = 24 pts
- Coding challenge: 76 pts (implementation 40, acceptance criteria 20, code quality 16)
- Total: 100 pts; pass ≥ 75

Answer key: ../solutions/week-03/assessment.md
