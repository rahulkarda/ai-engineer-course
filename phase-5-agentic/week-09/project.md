# Week 09 Project — A2A Interop Layer

> Build time: ~4 hr (Sunday)  ·  Language: Python 3.12+  ·  No real LLM/HTTP required

## Goal

Build a complete **A2A-style interoperability layer** where agents:
1. Advertise their capabilities via agent cards
2. Register in a shared discovery registry
3. Delegate tasks to each other over a mockable transport
4. Handle the full task lifecycle (submitted → working → completed/failed)

The system must be fully testable with no real network calls or LLM keys.

## Project structure

```
week_09_project/
├── a2a_interop/
│   ├── __init__.py
│   ├── models.py          # AgentCard, Task, Message, Part, Artifact (Pydantic v2)
│   ├── registry.py        # AgentRegistry — register, discover, route
│   ├── transport.py       # Transport ABC, InProcessTransport, MockTransport
│   ├── task_store.py      # TaskStore with state machine enforcement
│   ├── server.py          # A2AServer ABC
│   ├── client.py          # A2AClient — discovery + delegation
│   └── exceptions.py      # Domain exceptions
├── agents/
│   ├── __init__.py
│   ├── summarizer.py      # SummarizationAgent (A2AServer subclass)
│   ├── sql_agent.py       # SQLGenerationAgent (A2AServer subclass)
│   └── orchestrator.py    # OrchestratorAgent (A2AClient subclass)
├── tests/
│   ├── conftest.py        # shared fixtures
│   ├── test_models.py
│   ├── test_registry.py
│   ├── test_transport.py
│   ├── test_task_store.py
│   ├── test_server.py
│   ├── test_client.py
│   └── test_integration.py
├── main.py                # Demo: runs end-to-end pipeline
└── pyproject.toml
```

## Functional requirements

### FR-1: Agent cards
- Every agent exposes a `get_card() -> AgentCard` method
- `AgentCard` contains: `name`, `description`, `version`, `url`, `skills`, `capabilities`, `reliability`
- Cards are serializable to JSON (for `/.well-known/agent.json` equivalent)

### FR-2: Registry
- `register(card)` — idempotent; re-registering the same name updates the card
- `route(skill_id, min_reliability=0.0) -> AgentCard` — returns best-reliability agent for skill; raises `AgentNotFoundError` if none
- `find_by_skill(skill_id) -> list[AgentCard]` — all agents with the skill, sorted by reliability
- `find_by_multiple_skills(skill_ids, require_all=True) -> list[AgentCard]`

### FR-3: Task store
- `create(task) -> Task`
- `get(task_id) -> Task` — raises `TaskNotFoundError` if missing
- `transition(task_id, new_state) -> Task` — raises `InvalidTransitionError` for illegal transitions
- `add_message(task_id, message) -> Task`
- `add_artifact(task_id, artifact) -> Task`
- `pending_tasks() -> list[Task]`

### FR-4: Transport
- `Transport` is an ABC with `send(agent_url, task) -> Task` and `register_handler(url, handler)`
- `InProcessTransport` routes to callables in the same process (no serialization)
- `MockTransport` stores preloaded responses; records all calls; raises `TransportError` if no response configured

### FR-5: A2A server
- `A2AServer` ABC with `abstract async handle(task: Task) -> Task` and concrete `get_card() -> AgentCard`
- `SummarizationAgent` — given text in a `TextPart`, returns a summary artifact
- `SQLGenerationAgent` — given a natural-language question in a `TextPart`, returns a SQL query artifact
- Both agents: transition task through `submitted → working → completed` (or `failed`)
- Both agents: raise `input-required` if the input message has no text parts

### FR-6: A2A client
- `A2AClient` holds a registry and transport
- `discover(agent_url, card)` — registers card and binds handler to URL in transport
- `async delegate(skill_id, text) -> str` — route to best agent, create task, send, return result text
- `async delegate_parallel(tasks: list[tuple[str, str]]) -> list[str]` — run multiple delegations concurrently

### FR-7: Orchestrator
- `OrchestratorAgent` uses `A2AClient` to coordinate `SummarizationAgent` and `SQLGenerationAgent`
- Given a compound request like `"summarize this doc AND generate a query"`, it:
  1. Splits into two sub-tasks
  2. Delegates in parallel (they are independent)
  3. Aggregates results into a single output string

### FR-8: Error handling
- If the target agent raises an exception in `handle`, the task transitions to `failed` and `delegate` raises `A2AError` with the error message
- `A2AClient` retries up to 2 times on `TransportError` with 0.1s delay
- After max retries, the task transitions to `failed`

## Test requirements

Minimum 18 test cases across:

| File | Min tests | What to cover |
|---|---|---|
| `test_models.py` | 3 | Part discriminated union, AgentCard serialization, Task defaults |
| `test_registry.py` | 4 | register, route best-reliability, not-found, multi-skill |
| `test_transport.py` | 3 | InProcess happy path, MockTransport preloaded, no-response error |
| `test_task_store.py` | 4 | create/get, valid transition, invalid transition, pending_tasks |
| `test_server.py` | 2 | SummarizationAgent completes, input-required on empty |
| `test_integration.py` | 2 | Full pipeline via InProcessTransport; retry on MockTransport |

All tests: `pytest tests/ -v` must pass with zero errors.

## Non-functional requirements

- Python 3.12+, Pydantic v2, `asyncio`, `pytest`, `pytest-asyncio`
- No hardcoded strings as "API keys" — if you add optional real-LLM path, use `os.getenv("ANTHROPIC_API_KEY")`
- Model: `claude-haiku-5-5` for any real LLM call (optional extension)
- `logging` module throughout (not `print`)
- Type hints on every function signature
- No file exceeds 200 lines (split if needed)

## Optional extensions (do these after baseline passes)

- **FastAPI adapter** — expose `GET /.well-known/agent.json` and `POST /tasks/send` for `SummarizationAgent`; use `httpx.AsyncClient` in tests
- **SSE streaming** — make `SummarizationAgent` yield partial results as `TaskArtifactUpdateEvent` objects
- **Real LLM path** — add `--real-llm` pytest flag; when set, use `anthropic` SDK with `claude-haiku-5-5` instead of mock responses; skip in CI

## Deliverables

1. All source files under `week_09_project/`
2. `pytest tests/ -v` output showing all tests green
3. `python main.py` runs the end-to-end demo without errors

## Evaluation

Self-evaluate against the assessment rubric. The project is the primary artifact for week-09 assessment submission.
