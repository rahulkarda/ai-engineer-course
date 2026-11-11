# Solutions — Week 07 · Day 1: Multi-Agent Architectures Overview

## Exercise 1: Topology for a code-review system

Best topology: **hierarchical/supervisor with concurrent scatter-gather inside**.

```
         ┌──────────────────┐
         │  ReviewSupervisor │
         └─────────┬────────┘
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
SecurityAgent  StyleAgent   TestAgent    (concurrent scatter)
     └─────────────┴──────────────┘
                   │ gather
                   ▼
           VerdictSynthesiser
```

Concurrent scatter for the three specialist reviewers (they are independent of each other). Supervisor gathers results and delegates to `VerdictSynthesiser`.

## Exercise 2: Fourth translation agent

```python
def sequential_pipeline(topic: str, client: anthropic.Anthropic) -> str:
    researcher  = Agent("researcher",  "Return 3 bullet facts.", client)
    drafter     = Agent("drafter",     "Turn facts into a paragraph.", client)
    reviewer    = Agent("reviewer",    "Return 'APPROVED' or one fix.", client)
    translator  = Agent("translator",  "Translate the input text to Spanish. Return only the translation.", client)

    facts      = researcher.run(f"Facts about: {topic}")
    draft      = drafter.run(f"Write paragraph from:\n{facts}")
    verdict    = reviewer.run(f"Review:\n{draft}")
    translated = translator.run(verdict)
    logger.info("Translated: %s", translated[:60])
    return translated
```

## Exercise 3: True parallel scatter-gather

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def scatter_gather(questions: list[str], client: anthropic.Anthropic) -> list[str]:
    worker = Agent("worker", "Answer in one sentence.", client)
    results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        future_to_idx = {
            pool.submit(worker.run, q): i
            for i, q in enumerate(questions)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()

    return [results[i] for i in range(len(questions))]
```

Note: `Agent.run` calls `client.messages.create` which is I/O-bound — `ThreadPoolExecutor` achieves real parallelism for I/O.

## Exercise 4: Tracking correlation_id

```python
import uuid
from collections import defaultdict

pending_requests: dict[str, str] = {}  # correlation_id → task_description

def send_request(bus: MessageBus, sender: str, receiver: str, task: str) -> str:
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        message_type="request",
        payload={"task": task},
        correlation_id=str(uuid.uuid4()),
    )
    pending_requests[msg.correlation_id] = task
    bus.send(msg)
    return msg.correlation_id  # supervisor stores this

def handle_response(msg: AgentMessage) -> None:
    task = pending_requests.pop(msg.correlation_id, "unknown")
    logger.info("response for task '%s': %s", task, msg.payload.get("result", ""))
```

## Exercise 5: Unique failure modes

| Topology | Unique failure mode |
|---|---|
| Sequential | Single point of failure — one broken agent halts the entire pipeline |
| Concurrent | Race condition on shared output collector; result ordering not guaranteed |
| Hierarchical | Supervisor bottleneck — supervisor's reasoning error cascades to all workers |
| Network/P2P | Message storms — agents in a feedback loop; no global view of system state |
