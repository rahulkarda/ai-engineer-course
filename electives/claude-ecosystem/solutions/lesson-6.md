# Solution: Lesson 6 — Token Counting & Usage Tracking

## Exercise 1: Per-model cost comparison

```python
MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"input": 4.00,  "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00,  "output": 10.00},
    "claude-haiku-5-5":  {"input": 0.10,  "output": 0.50},
}


def cost_for_request(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES[model]
    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


def compare_model_costs(input_tokens: int, output_tokens: int) -> dict[str, float]:
    return {
        model: cost_for_request(model, input_tokens, output_tokens)
        for model in MODEL_PRICES
    }


# tests
import pytest

@pytest.mark.parametrize("model,inp,out,expected", [
    ("claude-haiku-5-5",  200, 100,  200*0.10/1e6 + 100*0.50/1e6),
    ("claude-sonnet-5-5", 200, 100,  200*2.00/1e6 + 100*10.0/1e6),
    ("claude-opus-5-5",   200, 100,  200*4.00/1e6 + 100*20.0/1e6),
])
def test_cost_for_request(model, inp, out, expected):
    assert abs(cost_for_request(model, inp, out) - expected) < 1e-12


def test_compare_all_models():
    result = compare_model_costs(200, 100)
    assert set(result.keys()) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}
    # Haiku must be cheapest
    assert result["claude-haiku-5-5"] < result["claude-sonnet-5-5"] < result["claude-opus-5-5"]
```

## Exercise 2: Budget enforcement

```python
from dataclasses import dataclass, field
from unittest.mock import MagicMock


@dataclass
class CostTracker:
    model: str
    total_input_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    _call_costs: list[float] = field(default_factory=list)

    def record(self, usage) -> float:
        self.total_input_tokens += usage.input_tokens
        self.total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.total_output_tokens += usage.output_tokens
        self.call_count += 1
        cost = cost_for_request(self.model, usage.input_tokens, usage.output_tokens)
        self._call_costs.append(cost)
        return cost

    @property
    def total_cost(self) -> float:
        return sum(self._call_costs)

    def summary(self) -> dict:
        return {
            "calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }


def budget_pipeline(
    texts: list[str],
    client,
    model: str = "claude-haiku-5-5",
    total_budget_usd: float = 0.001,
) -> tuple[list[str], dict]:
    """
    Process texts; stop when accumulated cost exceeds total_budget_usd.
    """
    tracker = CostTracker(model=model)
    answers: list[str] = []

    for text in texts:
        if tracker.total_cost >= total_budget_usd:
            answers.append("[BUDGET_EXCEEDED]")
            continue

        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": text}],
        )
        tracker.record(response.usage)
        answer = next(b.text for b in response.content if b.type == "text")
        answers.append(answer)

    return answers, tracker.summary()


def make_response(input_tokens: int = 100, output_tokens: int = 50, text: str = "ok"):
    block = MagicMock(); block.type = "text"; block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return resp


def test_budget_stops_processing():
    """With a tiny budget, processing stops after the first item."""
    client = MagicMock()
    # Each call: 10_000 input tokens on Haiku → 0.001 USD → meets budget after 1 call
    client.messages.create.return_value = make_response(input_tokens=10_000, output_tokens=0)

    answers, summary = budget_pipeline(
        ["text1", "text2", "text3"],
        client,
        total_budget_usd=0.0005,  # half a cent — exceeded after first call
    )

    assert answers[0] == "ok"
    assert "[BUDGET_EXCEEDED]" in answers
    assert client.messages.create.call_count == 1
```

## Exercise 3: CostTracker serialisation

```python
import json

def tracker_to_json(tracker: CostTracker) -> str:
    return json.dumps(tracker.summary())


def test_to_json_round_trip():
    tracker = CostTracker(model="claude-haiku-5-5")
    usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    tracker.record(usage)

    json_str = tracker_to_json(tracker)
    data = json.loads(json_str)
    assert data["calls"] == 1
    assert data["total_input_tokens"] == 100
    assert "total_cost_usd" in data
```

## Exercise 4: Token count vs actual (mocked integration)

```python
def test_safe_to_send_rejects_large_request():
    client = MagicMock()
    # Mock count_tokens to return 60_000 tokens
    count_result = MagicMock(); count_result.input_tokens = 60_000
    client.messages.count_tokens.return_value = count_result

    result = safe_to_send_mocked(
        client,
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "very large text..."}],
        max_input_tokens=50_000,
    )
    assert result is False


def test_tracker_accumulates_correctly():
    tracker = CostTracker(model="claude-haiku-5-5")
    for _ in range(3):
        usage = MagicMock(
            input_tokens=1000, output_tokens=200,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        tracker.record(usage)

    assert tracker.total_input_tokens == 3000
    assert tracker.call_count == 3


def safe_to_send_mocked(client, model, messages, max_input_tokens):
    result = client.messages.count_tokens(model=model, messages=messages)
    return result.input_tokens <= max_input_tokens
```
