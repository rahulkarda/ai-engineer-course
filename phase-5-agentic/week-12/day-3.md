# Phase 5 · Week 12 · Day 3 — Multi-Agent Orchestration + MCP Tool Servers + A2A Integration

> Objective: Implement supervisor/worker multi-agent dispatch, expose tools as MCP endpoints, and wire agents through A2A message envelopes.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Single-agent systems hit a ceiling: one agent reasoning over all domains, all tools, all context produces bloated prompts, routing confusion, and brittle failure modes. Multi-agent architectures fix this by assigning specialists.

**Three multi-agent patterns:**

| Pattern | Structure | Use when |
|---|---|---|
| Supervisor/Worker | One supervisor decomposes + dispatches; workers execute independently | Task has clear sub-domains (finance, ops, approval) |
| Peer-to-peer | Agents message each other directly via A2A | Tasks require negotiation or iterative refinement |
| Pipeline | Output of agent N is input to agent N+1 | Sequential processing (extract → validate → approve) |

This lesson implements all three infrastructure pieces:
- `MultiAgentCoordinator` — parallel dispatch with `ThreadPoolExecutor`
- `SupervisorAgent` — LLM-driven decomposition using `claude-opus-5-5`
- MCP server (FastAPI) — exposes tools over HTTP for external consumers
- `A2AChannel` — in-memory message bus for peer-to-peer coordination

## How it works internally

**Supervisor decomposition flow:**

1. `SupervisorAgent.run(task)` calls `claude-opus-5-5` with a decomposition prompt
2. LLM returns JSON: `[{"agent_name": "...", "payload": "..."}]`
3. Each item becomes a `SubTask` (Pydantic model with `task_id`, `agent_name`, `payload`)
4. `MultiAgentCoordinator.dispatch(subtasks)` submits all subtasks to a `ThreadPoolExecutor`
5. `as_completed()` collects results as they finish (non-deterministic order)
6. Each result is a `SubTaskResult` with `success`, `result`, and optional `error`
7. Supervisor aggregates all results into a single string summary

**MCP server design:**
MCP (Model Context Protocol) exposes agent tools over HTTP so external systems — other agents, orchestration platforms, Claude desktop — can invoke them without importing Python. The FastAPI transport uses a simple `POST /mcp/invoke` endpoint with a `MCPRequest` / `MCPResponse` envelope. A `GET /mcp/tools` endpoint acts as a discovery service.

**A2A correlation:**
Every `A2AMessage` carries a `message_id` (UUID). When replying, the reply's `correlation_id` is set to the original `message_id`. This allows a sender to match a response to the original request without shared state.

**Threading note:**
`ThreadPoolExecutor` introduces concurrency. Each worker thread calls a different agent function. If an agent function accesses shared mutable state (e.g., a single `MemoryStore` connection), that state must be thread-safe. `SQLiteMemoryStore` opens a new `sqlite3.connect()` per call, which is safe.

## Worked example

Multi-agent coordinator with parallel dispatch:

```python
# src/finops_agent/orchestration/coordinator.py
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SubTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    payload: str
    priority: int = 1

class SubTaskResult(BaseModel):
    task_id: str
    agent_name: str
    success: bool
    result: str
    error: str | None = None

class MultiAgentCoordinator:
    def __init__(self, agents: dict):  # name -> callable(payload) -> str
        self.agents = agents
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("Coordinator initialized with agents: %s", list(agents.keys()))

    def dispatch(self, subtasks: list[SubTask]) -> list[SubTaskResult]:
        futures = {
            self.executor.submit(self._run_subtask, st): st
            for st in subtasks
        }
        results: list[SubTaskResult] = []
        for future in as_completed(futures):
            st = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error("Subtask %s failed: %s", st.task_id, e)
                results.append(SubTaskResult(
                    task_id=st.task_id, agent_name=st.agent_name,
                    success=False, result="", error=str(e)
                ))
        return results

    def _run_subtask(self, st: SubTask) -> SubTaskResult:
        agent_fn = self.agents.get(st.agent_name)
        if not agent_fn:
            return SubTaskResult(task_id=st.task_id, agent_name=st.agent_name,
                                  success=False, result="", error=f"Unknown agent: {st.agent_name}")
        result = agent_fn(st.payload)
        logger.info("Subtask %s completed by %s", st.task_id, st.agent_name)
        return SubTaskResult(task_id=st.task_id, agent_name=st.agent_name, success=True, result=result)
```

Supervisor agent using `claude-opus-5-5` for decomposition:

```python
# src/finops_agent/orchestration/supervisor.py
import logging
import anthropic
from finops_agent.orchestration.coordinator import MultiAgentCoordinator, SubTask

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are a supervisor agent. Decompose the task into subtasks for specialist agents.
Available agents: finance_agent, ops_agent, approval_agent, reporting_agent
Return a JSON list: [{{"agent_name": "...", "payload": "..."}}]
Task: {task}"""

class SupervisorAgent:
    def __init__(self, client: anthropic.Anthropic, coordinator: MultiAgentCoordinator, model: str = "claude-opus-5-5"):
        self.client = client
        self.coordinator = coordinator
        self.model = model

    def run(self, task: str) -> str:
        import json
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}],
        )
        raw = response.content[0].text.strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Supervisor failed to parse decomposition: %s", raw)
            return f"Decomposition error: {raw}"

        subtasks = [SubTask(agent_name=p["agent_name"], payload=p["payload"]) for p in plan]
        results = self.coordinator.dispatch(subtasks)
        summary = "\n".join(f"[{r.agent_name}] {'OK' if r.success else 'ERR'}: {r.result or r.error}" for r in results)
        logger.info("Supervisor completed task with %d subtasks", len(subtasks))
        return summary
```

MCP tool server (FastAPI transport):

```python
# src/finops_agent/mcp/server.py
import logging
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from finops_agent.tools.finance import retrieve_invoice, validate_payment, check_budget

logger = logging.getLogger(__name__)
app = FastAPI(title="FinOps MCP Tool Server", version="1.0.0")

class MCPRequest(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    correlation_id: str | None = None

class MCPResponse(BaseModel):
    tool_name: str
    result: Any
    error: str | None = None
    correlation_id: str | None = None

TOOL_MAP: dict[str, Any] = {
    "retrieve_invoice": retrieve_invoice,
    "validate_payment": validate_payment,
    "check_budget": check_budget,
}

@app.post("/mcp/invoke", response_model=MCPResponse)
def invoke_tool(req: MCPRequest) -> MCPResponse:
    if req.tool_name not in TOOL_MAP:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
    try:
        fn = TOOL_MAP[req.tool_name]
        result = fn(**req.parameters)
        logger.info("MCP tool %s invoked (correlation=%s)", req.tool_name, req.correlation_id)
        # Convert Pydantic model to dict if needed
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return MCPResponse(tool_name=req.tool_name, result=result, correlation_id=req.correlation_id)
    except Exception as e:
        logger.error("MCP tool %s failed: %s", req.tool_name, e)
        return MCPResponse(tool_name=req.tool_name, result=None, error=str(e), correlation_id=req.correlation_id)

@app.get("/mcp/tools")
def list_tools() -> dict:
    return {"tools": list(TOOL_MAP.keys())}
```

A2A message protocol and in-memory channel:

```python
# src/finops_agent/a2a/protocol.py
import uuid
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class A2AMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    recipient: str
    payload: dict
    correlation_id: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_type: str = "task_request"  # task_request | task_response | error

class A2AChannel:
    """In-memory A2A message bus for single-process operation."""
    def __init__(self):
        self._inbox: dict[str, list[A2AMessage]] = {}

    def send(self, msg: A2AMessage) -> None:
        self._inbox.setdefault(msg.recipient, []).append(msg)
        logger.info("A2A: %s -> %s (type=%s)", msg.sender, msg.recipient, msg.message_type)

    def receive(self, agent_name: str) -> list[A2AMessage]:
        msgs = self._inbox.pop(agent_name, [])
        logger.debug("A2A: %s received %d messages", agent_name, len(msgs))
        return msgs

    def reply(self, original: A2AMessage, payload: dict) -> A2AMessage:
        reply_msg = A2AMessage(
            sender=original.recipient,
            recipient=original.sender,
            payload=payload,
            correlation_id=original.message_id,
            message_type="task_response",
        )
        self.send(reply_msg)
        return reply_msg
```

**Wiring coordinator + supervisor together:**

```python
# Example: building the supervisor from scratch
import os
import anthropic
from finops_agent.orchestration.coordinator import MultiAgentCoordinator
from finops_agent.orchestration.supervisor import SupervisorAgent

# Specialist agent stubs — in production these are full AgentCore instances
def finance_agent(payload: str) -> str:
    return f"finance_agent processed: {payload}"

def ops_agent(payload: str) -> str:
    return f"ops_agent processed: {payload}"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
coordinator = MultiAgentCoordinator(agents={
    "finance_agent": finance_agent,
    "ops_agent": ops_agent,
})
supervisor = SupervisorAgent(client=client, coordinator=coordinator)

result = supervisor.run("Check budget for Q4 engineering and open a ticket if over limit")
print(result)
```

## Exercises

1. Extend `MultiAgentCoordinator.dispatch` to accept a `sequential: bool = False` keyword argument. When `True`, run subtasks one-by-one in ascending `priority` order (lower = first) rather than in parallel. The return type stays `list[SubTaskResult]`.

2. Add a `GET /mcp/schema/{tool_name}` endpoint to the MCP server that returns a JSON Schema dict describing the tool's input parameters. Where would you store the parameter schemas to avoid duplication between `TOOL_MAP` and the schema endpoint?

3. Write a unit test for `A2AChannel` that: (a) sends 3 messages to `"finance_agent"` from different senders, (b) verifies `receive("finance_agent")` returns all 3 messages, (c) verifies a second `receive("finance_agent")` returns an empty list. Use `pytest` and no external dependencies.

4. Modify `SupervisorAgent.run` to retry the decomposition call up to 2 additional times (3 total) if `json.JSONDecodeError` is raised. Log each retry with `logger.warning`. After all retries exhausted, return the decomposition error string as before.

## Self-check

- [ ] I can implement a supervisor/worker multi-agent pattern with parallel dispatch
- [ ] I can expose tools as MCP endpoints over FastAPI using `MCPRequest`/`MCPResponse` envelopes
- [ ] I understand how `A2AMessage.correlation_id` links a reply to its original request
- [ ] I can dispatch subtasks in parallel with `ThreadPoolExecutor` and collect results with `as_completed`
