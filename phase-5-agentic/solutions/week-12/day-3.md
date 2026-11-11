# Week 12 · Day 3 Solutions — Multi-Agent + MCP + A2A

## Exercise 1 — Sequential dispatch with priority ordering

```python
# src/finops_agent/orchestration/coordinator.py — extended dispatch
def dispatch(self, subtasks: list[SubTask], sequential: bool = False) -> list[SubTaskResult]:
    if sequential:
        ordered = sorted(subtasks, key=lambda st: st.priority, reverse=True)
        results: list[SubTaskResult] = []
        for st in ordered:
            try:
                res = self._run_subtask(st)
                results.append(res)
            except Exception as e:
                logger.error("Sequential subtask %s failed: %s", st.task_id, e)
                results.append(SubTaskResult(
                    task_id=st.task_id, agent_name=st.agent_name,
                    success=False, result="", error=str(e)
                ))
        return results

    # Parallel path (original)
    futures = {self.executor.submit(self._run_subtask, st): st for st in subtasks}
    results = []
    for future in as_completed(futures):
        st = futures[future]
        try:
            results.append(future.result())
        except Exception as e:
            logger.error("Subtask %s failed: %s", st.task_id, e)
            results.append(SubTaskResult(
                task_id=st.task_id, agent_name=st.agent_name,
                success=False, result="", error=str(e)
            ))
    return results
```

## Exercise 2 — MCP schema endpoint

```python
# src/finops_agent/mcp/server.py — addition
import inspect

@app.get("/mcp/schema/{tool_name}")
def get_tool_schema(tool_name: str) -> dict:
    if tool_name not in TOOL_MAP:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    fn = TOOL_MAP[tool_name]
    sig = inspect.signature(fn)
    params = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            params[name] = {"type": "any"}
        elif annotation == str:
            params[name] = {"type": "string"}
        elif annotation == float:
            params[name] = {"type": "number"}
        elif annotation == int:
            params[name] = {"type": "integer"}
        else:
            params[name] = {"type": str(annotation)}
    return {
        "tool_name": tool_name,
        "json_schema": {
            "type": "object",
            "properties": params,
            "required": list(params.keys()),
        }
    }
```

## Exercise 3 — A2AChannel unit test

```python
# tests/unit/test_a2a.py
import pytest
from finops_agent.a2a.protocol import A2AChannel, A2AMessage

def test_send_and_receive():
    channel = A2AChannel()
    for i in range(3):
        msg = A2AMessage(
            sender="supervisor",
            recipient="finance_agent",
            payload={"task": f"task-{i}"},
        )
        channel.send(msg)

    received = channel.receive("finance_agent")
    assert len(received) == 3
    assert all(m.recipient == "finance_agent" for m in received)

def test_receive_empty_after_drain():
    channel = A2AChannel()
    msg = A2AMessage(sender="a", recipient="b", payload={})
    channel.send(msg)
    channel.receive("b")
    second = channel.receive("b")
    assert second == []

def test_reply_sets_correlation_id():
    channel = A2AChannel()
    original = A2AMessage(sender="a", recipient="b", payload={"q": "1"})
    channel.send(original)
    received = channel.receive("b")
    reply = channel.reply(received[0], payload={"answer": "42"})
    assert reply.correlation_id == original.message_id
    assert reply.sender == "b"
    assert reply.recipient == "a"
```

## Exercise 4 — Supervisor retry on JSONDecodeError

```python
# src/finops_agent/orchestration/supervisor.py — updated run method
def run(self, task: str) -> str:
    import json
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}],
        )
        raw = response.content[0].text.strip()
        try:
            plan = json.loads(raw)
            break
        except json.JSONDecodeError:
            logger.warning(
                "Decomposition attempt %d/%d failed to parse JSON: %.80s",
                attempt + 1, MAX_RETRIES + 1, raw
            )
            if attempt == MAX_RETRIES:
                logger.error("All decomposition attempts failed")
                return f"Decomposition error after {MAX_RETRIES + 1} attempts: {raw}"

    subtasks = [SubTask(agent_name=p["agent_name"], payload=p["payload"]) for p in plan]
    results = self.coordinator.dispatch(subtasks)
    summary = "\n".join(
        f"[{r.agent_name}] {'OK' if r.success else 'ERR'}: {r.result or r.error}"
        for r in results
    )
    logger.info("Supervisor completed task with %d subtasks", len(subtasks))
    return summary
```
