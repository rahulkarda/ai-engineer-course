# Week 09 Assessment — A2A & Agent Interoperability

## Conceptual (interview-style)

1. An interviewer asks: "Why can't two agents just call each other's REST APIs directly? What additional contract layer does A2A provide?" Answer in under 90 seconds.

2. Explain the difference between an A2A *capability* and an A2A *skill*. Give a concrete example of each for a code-review agent.

3. A task transitions to `input-required`. What does this mean? Who acts next, the client or the server? What happens if the client never responds?

4. Your orchestrator delegates to Agent B. Agent B is down and returns HTTP 503. Describe a complete error-handling strategy: what states does the task go through, when do you retry vs. fall back, and what does the user see?

5. You are designing a multi-agent system where Agent A (Python, uses Claude) delegates to Agent B (Java, uses a rules engine). What does A2A give you that a simple REST wrapper doesn't?

6. A new colleague says: "We already use MCP for everything — why do we need A2A too?" Write a three-point response explaining the architectural difference.

7. How does the `/.well-known/agent.json` endpoint enable *zero-config* discovery? What is the RFC it borrows from?

8. Your week-09 project has three agents: orchestrator, SQL specialist, and writer. A new client wants to call the SQL specialist directly, bypassing the orchestrator. What changes (if any) to the SQL specialist's code are needed, and what risks does direct access introduce?

## Coding challenge

**Build a minimal but complete A2A broker.**

**Context:** You are building an internal developer platform. Engineers submit tasks like "analyze this PR" or "generate test stubs for this file." Multiple specialist agents exist. The broker must: accept tasks, route them to the right specialist, handle failures, and return results.

**Requirements:**

1. `AgentCard` Pydantic model with at minimum: `name`, `version`, `url`, `skills: list[AgentSkill]`, `reliability: float`
2. `AgentRegistry` with `register`, `route(skill_id)` (best reliability), `find_by_skill`
3. `TaskStore` with: `create`, `get`, `transition` (enforcing valid transitions), `pending_tasks`
4. `Transport` ABC with two implementations: `InProcessTransport` (routes to Python callables) and `MockTransport` (pre-configured responses for tests)
5. `A2AClient` with `discover(url, card)` and `async delegate(skill_id, payload) -> str`; uses registry + transport
6. `A2AServer` base class with `abstract async handle(task) -> Task` and `get_card() -> AgentCard`
7. Two concrete specialist agents (e.g., `PRReviewAgent`, `TestStubAgent`) implementing `A2AServer`
8. An orchestrator that calls both specialists (in parallel if independent, sequential if dependent) and aggregates
9. Pytest test suite:
   - Unit tests for `TaskStore` (all state transitions including invalid)
   - Unit tests for `AgentRegistry` (routing, reliability ordering, not-found)
   - Integration test using `MockTransport` — no real network or LLM
   - Integration test using `InProcessTransport` with real specialist agents

**Acceptance criteria:**
- All tests pass with `pytest -v`
- No hardcoded API keys; LLM calls (if any) use `os.getenv("ANTHROPIC_API_KEY")` and are mocked in tests
- `TaskState` machine rejects invalid transitions with a domain exception (not `ValueError`)
- `MockTransport` records all calls and exposes `call_count` and `calls_to(url)` for test assertions
- At least 15 test cases total

**Constraints:** Python 3.12+, Pydantic v2, pytest, `asyncio`. No FastAPI or HTTP required (transport is in-process for the submission).

## Rubric (pass ≥ 75%)

| Dimension | Weight | Pass criteria |
|---|---|---|
| Independent implementation | 40% | Core layer works end-to-end; tests pass without modification |
| Conceptual understanding | 25% | Conceptual answers are precise; difference between MCP/A2A is clear |
| Debugging & problem-solving | 20% | Invalid transitions raise domain exceptions; failed delegates retry or fall back |
| Code quality & testing | 15% | Type hints everywhere; Pydantic v2 models; ≥15 test cases; logging not print |

Answer key: ../solutions/week-09/assessment.md
