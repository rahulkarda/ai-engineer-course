# Solutions — Week 07 · Day 6: Saturday Deep Practice Reference

The day-6 lesson file contains the full worked solution (scaffold, agents, supervisor, tests).

This solutions file contains:
1. `conftest.py` fixture
2. Full `test_blackboard.py`
3. Full `test_bus.py`
4. Additional test patterns

---

## conftest.py

```python
# tests/conftest.py
from unittest.mock import MagicMock
import pytest


def make_mock_client(responses: list[str]) -> MagicMock:
    """Factory for a mock Anthropic client that cycles responses."""
    client = MagicMock()
    state = {"n": 0}

    def _create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text=responses[state["n"] % len(responses)])]
        state["n"] += 1
        return resp

    client.messages.create.side_effect = _create
    return client


@pytest.fixture
def mock_client():
    return make_mock_client([
        '{"sub_queries": ["sub-q 1", "sub-q 2"]}',
        "Finding for sub-q 1.",
        "Finding for sub-q 2.",
        "DRAFT: Findings merged.\nCLAIMS:\n- claim A\n- claim B",
        "CLAIM: claim A\nVERDICT: VERIFIED\nREASON: It is.\nCLAIM: claim B\nVERDICT: VERIFIED\nREASON: It is.",
    ])
```

---

## test_blackboard.py

```python
# tests/test_blackboard.py
import threading
import pytest
from src.blackboard import Blackboard


def test_write_and_read() -> None:
    board = Blackboard()
    assert board.write("k", "v", author="a") is True
    entry = board.read("k")
    assert entry is not None
    assert entry.value == "v"
    assert entry.version == 1
    assert entry.author == "a"


def test_second_write_bumps_version() -> None:
    board = Blackboard()
    board.write("k", "v1", author="a")
    board.write("k", "v2", author="a")
    entry = board.read("k")
    assert entry is not None
    assert entry.version == 2
    assert entry.value == "v2"


def test_cas_conflict() -> None:
    board = Blackboard()
    board.write("k", "v1", author="a")  # version becomes 1

    # write with correct expected version succeeds
    assert board.write("k", "v2", author="b", expected_version=1) is True

    # write with wrong expected version fails
    assert board.write("k", "v3", author="c", expected_version=1) is False
    assert board.read("k").value == "v2"  # unchanged


def test_read_section() -> None:
    board = Blackboard()
    board.write("facts/a", 1, author="x")
    board.write("facts/b", 2, author="x")
    board.write("other/c", 3, author="x")

    section = board.read_section("facts")
    assert section == {"a": 1, "b": 2}
    assert "c" not in section


def test_watcher_called() -> None:
    board = Blackboard()
    seen = []
    board.watch("notify/", lambda entry: seen.append(entry.value))
    board.write("notify/x", "hello", author="a")

    import time; time.sleep(0.1)  # watcher runs in thread
    assert "hello" in seen


def test_concurrent_writes_no_data_loss() -> None:
    board = Blackboard()
    errors = []

    def write_many(author: str) -> None:
        for i in range(50):
            try:
                board.write(f"k{i}", f"{author}-{i}", author=author)
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=write_many, args=(f"a{j}",)) for j in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors
```

---

## test_bus.py

```python
# tests/test_bus.py
import pytest
from src.bus import MessageBus
from src.models import AgentMessage, MessageType


def make_msg(sender: str, receiver: str, payload: dict | None = None) -> AgentMessage:
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        message_type=MessageType.REQUEST,
        payload=payload or {},
    )


def test_register_and_receive() -> None:
    bus = MessageBus()
    bus.register("agent-a")
    bus.register("agent-b")
    msg = make_msg("agent-a", "agent-b", {"x": 1})
    bus.send(msg)
    received = bus.receive("agent-b", timeout=0.5)
    assert received is not None
    assert received.payload == {"x": 1}


def test_unknown_receiver_raises() -> None:
    bus = MessageBus()
    bus.register("sender")
    with pytest.raises(ValueError, match="Unknown receiver"):
        bus.send(make_msg("sender", "nobody"))


def test_broadcast_delivered_to_all() -> None:
    bus = MessageBus()
    for name in ("a", "b", "c", "sender"):
        bus.register(name)
    msg = AgentMessage(
        sender="sender", receiver="broadcast",
        message_type=MessageType.BROADCAST, payload={},
    )
    bus.send(msg)
    for name in ("a", "b", "c"):
        received = bus.receive(name, timeout=0.5)
        assert received is not None, f"{name} did not receive broadcast"


def test_ttl_expiry_drops_message() -> None:
    bus = MessageBus()
    bus.register("target")
    msg = AgentMessage(
        sender="s", receiver="target",
        message_type=MessageType.REQUEST, payload={}, ttl=1
    )
    # TTL=1 → decremented to 0 on first send attempt
    # Send a message that has already used its hop
    msg_zero_ttl = msg.model_copy(update={"ttl": 0})
    bus.send(msg_zero_ttl)  # should be dropped silently
    received = bus.receive("target", timeout=0.2)
    assert received is None
```

---

## Additional: test coverage tip

Run with coverage to confirm no LLM calls reach the network:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

If `anthropic.Anthropic` is ever instantiated without mocking, the test will fail immediately because there is no real key in CI. Always mock at the `client.messages.create` level, not at the module import level.
