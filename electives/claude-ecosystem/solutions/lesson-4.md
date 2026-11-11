# Solution: Lesson 4 — Message Batches API

## Exercise 1: Batch status tracker

```python
def get_batch_status(client, batch_id: str) -> dict:
    """Return a dict with request counts for a batch."""
    batch = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "succeeded": counts.succeeded,
        "errored":   counts.errored,
        "expired":   counts.expired,
        "processing": counts.processing,
        "total": counts.succeeded + counts.errored + counts.expired + counts.processing,
    }


# tests/test_lesson4.py
from unittest.mock import MagicMock

def test_get_batch_status():
    client = MagicMock()
    counts = MagicMock()
    counts.succeeded  = 90
    counts.errored    = 5
    counts.expired    = 2
    counts.processing = 3

    batch = MagicMock()
    batch.request_counts = counts
    client.messages.batches.retrieve.return_value = batch

    status = get_batch_status(client, "batch_abc")

    assert status["succeeded"]  == 90
    assert status["errored"]    == 5
    assert status["expired"]    == 2
    assert status["processing"] == 3
    assert status["total"]      == 100
```

## Exercise 2: Retry on error

```python
from dataclasses import dataclass
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


@dataclass
class ProcessedResult:
    custom_id: str
    text: str


def collect_with_split(
    batch_id: str, client
) -> tuple[list[ProcessedResult], list[str]]:
    """
    Split results into (succeeded, failed_custom_ids).
    """
    succeeded: list[ProcessedResult] = []
    failed: list[str] = []

    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            msg = item.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            succeeded.append(ProcessedResult(custom_id=item.custom_id, text=text))
        else:
            failed.append(item.custom_id)

    return succeeded, failed


def resubmit_failed(
    failed_custom_ids: list[str],
    original_requests: dict[str, MessageCreateParamsNonStreaming],
    client,
) -> str:
    """
    Create a new batch containing only the failed items.
    original_requests: {custom_id: params}
    Returns new batch_id.
    """
    retry_requests = [
        Request(custom_id=cid, params=original_requests[cid])
        for cid in failed_custom_ids
        if cid in original_requests
    ]
    if not retry_requests:
        raise ValueError("No valid failed requests to retry")

    new_batch = client.messages.batches.create(requests=retry_requests)
    return new_batch.id


# Tests
def make_batch_result(custom_id: str, result_type: str = "succeeded", text: str = "ok"):
    item = MagicMock()
    item.custom_id = custom_id
    item.result = MagicMock()
    item.result.type = result_type

    if result_type == "succeeded":
        block = MagicMock(); block.type = "text"; block.text = text
        item.result.message = MagicMock()
        item.result.message.content = [block]

    return item


def test_collect_with_split():
    client = MagicMock()
    client.messages.batches.results.return_value = [
        make_batch_result("item-0", "succeeded", "positive"),
        make_batch_result("item-1", "errored"),
        make_batch_result("item-2", "succeeded", "negative"),
    ]

    ok, failed = collect_with_split("batch_abc", client)
    assert len(ok) == 2
    assert len(failed) == 1
    assert failed[0] == "item-1"


def test_resubmit_failed():
    client = MagicMock()
    new_batch = MagicMock(); new_batch.id = "batch_retry"
    client.messages.batches.create.return_value = new_batch

    params = MessageCreateParamsNonStreaming(
        model="claude-haiku-5-5",
        max_tokens=16,
        messages=[{"role": "user", "content": "test"}],
    )
    new_id = resubmit_failed(["item-1"], {"item-1": params}, client)
    assert new_id == "batch_retry"
    assert client.messages.batches.create.call_count == 1
```

## Exercise 3: Cost comparison calculation

```python
def batch_cost_comparison(
    n_texts: int = 1000,
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 10,
    system_tokens: int = 150,
    model: str = "claude-haiku-5-5",
) -> dict:
    """
    Compare costs:
    (a) No batching, no caching
    (b) Batching only (50% discount)
    (c) Batching + caching for system prompt
    """
    PRICES = {
        "claude-haiku-5-5": (0.10, 0.50),
    }
    input_price, output_price = PRICES[model]
    m = 1_000_000
    batch_discount = 0.50

    total_input = n_texts * (avg_input_tokens + system_tokens)
    total_output = n_texts * avg_output_tokens

    # (a) No batching, no caching
    cost_a = (total_input * input_price + total_output * output_price) / m

    # (b) Batching only — 50% off all tokens
    cost_b = cost_a * batch_discount

    # (c) Batching + caching system prompt
    # First request: cache write for system_tokens (1.25x)
    # Remaining requests: cache read for system_tokens (0.10x)
    cached_input_per_req = system_tokens
    uncached_per_req = avg_input_tokens

    cost_c_per_req_first = (
        (uncached_per_req + cached_input_per_req * 1.25) * input_price / m
        + avg_output_tokens * output_price / m
    ) * batch_discount
    cost_c_per_req_rest = (
        (uncached_per_req + cached_input_per_req * 0.10) * input_price / m
        + avg_output_tokens * output_price / m
    ) * batch_discount
    cost_c = cost_c_per_req_first + (n_texts - 1) * cost_c_per_req_rest

    return {
        "cost_no_batch_usd": round(cost_a, 4),
        "cost_batch_only_usd": round(cost_b, 4),
        "cost_batch_cached_usd": round(cost_c, 4),
        "savings_batch_vs_none_pct": round((1 - cost_b / cost_a) * 100, 1),
        "savings_batched_cached_vs_none_pct": round((1 - cost_c / cost_a) * 100, 1),
    }


import pytest

def test_cost_comparison_savings():
    result = batch_cost_comparison()
    # Batch-only should be 50% cheaper than no-batch
    assert abs(result["savings_batch_vs_none_pct"] - 50.0) < 0.1
    # Batch + caching should save more than batch alone
    assert result["cost_batch_cached_usd"] < result["cost_batch_only_usd"]


## Exercise 4: Large batch chunker

def chunk_batch(items: list, max_size: int = 100_000):
    """Yield successive chunks of up to max_size items."""
    for i in range(0, len(items), max_size):
        yield items[i:i + max_size]


def test_chunk_empty():
    assert list(chunk_batch([], max_size=3)) == []

def test_chunk_exact_size():
    items = list(range(6))
    chunks = list(chunk_batch(items, max_size=3))
    assert chunks == [[0, 1, 2], [3, 4, 5]]

def test_chunk_over_max():
    items = list(range(7))
    chunks = list(chunk_batch(items, max_size=3))
    assert len(chunks) == 3
    assert chunks[-1] == [6]

def test_chunk_single():
    chunks = list(chunk_batch([42], max_size=100_000))
    assert chunks == [[42]]
```
