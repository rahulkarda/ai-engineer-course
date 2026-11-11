# Solutions — Week 07 · Day 2: Agent Communication & Message Passing

## Exercise 1: HEARTBEAT handler

```python
# In BaseAgent.handle (override or add to run loop)
def handle(self, message: AgentMessage) -> AgentMessage | None:
    if message.message_type == MessageType.HEARTBEAT:
        self._log.info("heartbeat from %s at %s", message.sender, message.timestamp)
        return None  # heartbeats are fire-and-forget
    # ... existing handling
    raise NotImplementedError

# Broadcaster utility
def broadcast_heartbeat(bus: MessageBus, sender: str) -> None:
    msg = AgentMessage(
        sender=sender,
        receiver="broadcast",
        message_type=MessageType.HEARTBEAT,
        payload={"ts": datetime.now(timezone.utc).isoformat()},
    )
    bus.send(msg)
```

## Exercise 2: DeadLetterQueue

```python
from collections import deque

class MessageBus:
    def __init__(self) -> None:
        self._mailboxes: dict[str, queue.PriorityQueue] = {}
        self._dead_letters: deque[AgentMessage] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def send(self, message: AgentMessage) -> None:
        if message.ttl <= 0:
            self._dead_letters.append(message)
            logger.warning("bus | TTL expired → DLQ: %s", message.id)
            return
        message = message.model_copy(update={"ttl": message.ttl - 1})

        if message.receiver == "broadcast":
            # ... same as before
            pass
        else:
            with self._lock:
                box = self._mailboxes.get(message.receiver)
            if box is None:
                self._dead_letters.append(message)
                logger.warning("bus | unknown receiver → DLQ: %s", message.receiver)
                return
            box.put((-message.priority, message))

    def drain_dead_letters(self) -> list[AgentMessage]:
        result = list(self._dead_letters)
        self._dead_letters.clear()
        return result
```

## Exercise 3: Priority ordering test

```python
import pytest
from bus import MessageBus, AgentMessage, MessageType

def test_priority_ordering() -> None:
    bus = MessageBus()
    bus.register("worker")
    bus.register("sender")

    for priority, label in [(0, "low"), (9, "high"), (5, "medium")]:
        bus.send(AgentMessage(
            sender="sender",
            receiver="worker",
            message_type=MessageType.REQUEST,
            payload={"label": label},
            priority=priority,
        ))

    received = []
    for _ in range(3):
        msg = bus.receive("worker", timeout=0.1)
        assert msg is not None
        received.append(msg.payload["label"])

    assert received[0] == "high"    # priority 9 first
    assert received[1] == "medium"  # priority 5 second
    assert received[2] == "low"     # priority 0 last
```

## Exercise 4: Error handling and retry in OrchestratorAgent

```python
def handle(self, message: AgentMessage) -> AgentMessage | None:
    cid = message.correlation_id
    stage = self._pending.get(cid)

    if message.message_type == MessageType.ERROR:
        error_msg = message.payload.get("error", "unknown")
        retries = self._retry_counts.get(cid, 0)
        if retries < 1:
            self._log.warning("error from %s, retrying (attempt %d): %s",
                              message.sender, retries + 1, error_msg)
            self._retry_counts[cid] = retries + 1
            # re-send original request
            original_task = self._original_tasks.get(cid, "retry task")
            return AgentMessage(
                sender=self.name, receiver="summariser",
                message_type=MessageType.REQUEST,
                payload={"text": original_task},
                correlation_id=cid,
            )
        else:
            self._log.error("max retries exceeded for %s: %s", cid, error_msg)
            del self._pending[cid]
    # ... rest of handler
    return None
```

## Exercise 5: asyncio version

```python
import asyncio
from collections.abc import AsyncIterator

class AsyncMessageBus:
    def __init__(self) -> None:
        self._mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}

    def register(self, name: str) -> None:
        self._mailboxes[name] = asyncio.Queue()

    async def send(self, message: AgentMessage) -> None:
        if message.receiver == "broadcast":
            for name, box in self._mailboxes.items():
                if name != message.sender:
                    await box.put(message)
        else:
            box = self._mailboxes.get(message.receiver)
            if box is None:
                raise ValueError(f"Unknown: {message.receiver}")
            await box.put(message)

    async def receive(self, agent_name: str, timeout: float = 1.0) -> AgentMessage | None:
        box = self._mailboxes.get(agent_name)
        if box is None:
            raise ValueError(f"Unknown: {agent_name}")
        try:
            return await asyncio.wait_for(box.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class AsyncBaseAgent:
    def __init__(self, name: str, bus: AsyncMessageBus) -> None:
        self.name = name
        self._bus = bus
        self._running = True
        bus.register(name)

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        raise NotImplementedError

    async def run(self) -> None:
        while self._running:
            msg = await self._bus.receive(self.name, timeout=0.5)
            if msg is None:
                continue
            reply = await self.handle(msg)
            if reply is not None:
                await self._bus.send(reply)
```

Key change: `threading.Thread.run` → `async def run()` coroutine scheduled with `asyncio.create_task()`.
