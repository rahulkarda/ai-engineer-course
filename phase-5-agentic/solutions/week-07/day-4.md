# Solutions — Week 07 · Day 4: Role Specialization & Coordination

## Exercise 1: Bid-based routing

```python
from dataclasses import dataclass

@dataclass
class AgentSpec:
    name: str
    capabilities: frozenset[str]
    system_prompt: str
    confidence_by_cap: dict[str, float]  # new
    max_concurrent: int = 1
    current_load: int = 0

    def bid(self, capability: str) -> float:
        """Return confidence for this capability, 0 if not supported."""
        return self.confidence_by_cap.get(capability, 0.0)

    def is_available(self) -> bool:
        return self.current_load < self.max_concurrent


class BidCoordinator(Coordinator):
    def route(self, task: Task) -> AgentSpec | None:
        candidates = [
            a for a in self._agents
            if task.required_capability in a.capabilities and a.is_available()
        ]
        if not candidates:
            return None
        # Award to highest bidder
        winner = max(candidates, key=lambda a: a.bid(task.required_capability))
        logger.info(
            "bid-coordinator | awarded %s → %s (bid=%.2f)",
            task.id, winner.name, winner.bid(task.required_capability),
        )
        return winner
```

## Exercise 2: HealthMonitor

```python
import threading
import time
from dataclasses import dataclass, field

@dataclass
class HealthRecord:
    missed_heartbeats: int = 0
    healthy: bool = True


class HealthMonitor(threading.Thread):
    def __init__(
        self,
        agents: list[AgentSpec],
        runner: "SpecialistRunner",
        interval: float = 5.0,
        max_misses: int = 3,
    ) -> None:
        super().__init__(daemon=True)
        self._agents = agents
        self._runner = runner
        self._interval = interval
        self._max_misses = max_misses
        self._records: dict[str, HealthRecord] = {a.name: HealthRecord() for a in agents}

    def run(self) -> None:
        while True:
            for spec in self._agents:
                hb_task = Task(
                    id=f"hb-{spec.name}-{time.time():.0f}",
                    required_capability=next(iter(spec.capabilities)),
                    payload={"healthcheck": True},
                )
                result = self._runner.execute(spec, hb_task)
                rec = self._records[spec.name]
                if result.success:
                    rec.missed_heartbeats = 0
                    rec.healthy = True
                    spec.last_heartbeat = time.time()
                else:
                    rec.missed_heartbeats += 1
                    if rec.missed_heartbeats >= self._max_misses:
                        rec.healthy = False
                        logger.warning("health | %s marked UNHEALTHY", spec.name)
            time.sleep(self._interval)
```

## Exercise 3: Fallback capability

```python
@dataclass
class AgentSpec:
    name: str
    capabilities: frozenset[str]
    system_prompt: str
    fallback_capability: str | None = None
    max_concurrent: int = 1
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)

    def is_available(self) -> bool:
        return self.current_load < self.max_concurrent


class FallbackCoordinator(Coordinator):
    def route_with_fallback(
        self, task: Task, runner: "SpecialistRunner"
    ) -> TaskResult:
        spec = self.route(task)
        if spec is None:
            return TaskResult(task_id=task.id, agent_name="none",
                              output="", success=False, error="no route")

        result = runner.execute(spec, task)
        if not result.success and spec.fallback_capability:
            fallback_task = task.model_copy(
                update={"required_capability": spec.fallback_capability}
            )
            fallback_spec = self.route(fallback_task)
            if fallback_spec:
                logger.info("fallback | routing %s → %s", task.id, fallback_spec.name)
                return runner.execute(fallback_spec, fallback_task)
        return result
```

## Exercise 4: ContractNetCoordinator

```python
import time
import threading

class ContractNetCoordinator(Coordinator):
    def __init__(self, bid_window: float = 0.1) -> None:
        super().__init__()
        self._bid_window = bid_window

    def contract_net_route(self, task: Task) -> AgentSpec | None:
        """Broadcast → collect bids → award to highest bidder."""
        bids: list[tuple[float, AgentSpec]] = []
        lock = threading.Lock()

        def collect_bid(spec: AgentSpec) -> None:
            if task.required_capability not in spec.capabilities:
                return
            if not spec.is_available():
                return
            confidence = getattr(spec, "confidence_by_cap", {}).get(
                task.required_capability, 0.5
            )
            with lock:
                bids.append((confidence, spec))

        threads = [threading.Thread(target=collect_bid, args=(s,)) for s in self._agents]
        for t in threads: t.start()
        time.sleep(self._bid_window)  # bid collection window

        if not bids:
            return None
        winner = max(bids, key=lambda x: x[0])[1]
        logger.info("contract-net | awarded to %s", winner.name)
        return winner
```

## Exercise 5: Round-robin test

```python
import pytest
from coordinator import Coordinator, AgentSpec, Task, SpecialistRunner
from unittest.mock import MagicMock

def test_round_robin_distribution() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text="ok")]

    coord = Coordinator()
    for i in range(3):
        coord.register(AgentSpec(
            name=f"summarizer-{i}",
            capabilities=frozenset({"summarization"}),
            system_prompt="Summarise.",
        ))

    runner = SpecialistRunner(mock_client)
    assignments: dict[str, int] = {}

    for j in range(6):
        task = Task(id=f"t{j}", required_capability="summarization", payload={"text": "x"})
        spec = coord.route(task)
        assert spec is not None
        assignments[spec.name] = assignments.get(spec.name, 0) + 1
        runner.execute(spec, task)

    # Each summarizer should have handled exactly 2 tasks
    assert all(v == 2 for v in assignments.values()), f"uneven distribution: {assignments}"
```
