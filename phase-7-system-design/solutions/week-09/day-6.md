# Solutions — Week 09 · Day 6 (Saturday Scenarios)

## Scenario A — "The Leaking Agent" — Solution

### Bug identification

Lines causing the leak:
- `self._history.append(prompt)` — every `summarize()` call appends the user prompt
- `self._history.append(response_msg)` — every call also appends the assistant response
- `_history` lives on the instance, so it accumulates across all calls forever
- `gc_collection: collected=0` means no cyclic garbage — the list itself holds strong references to live dicts; GC cannot collect them

### Fixed `OrderSummaryAgent`

```python
@dataclass
class OrderSummaryAgent:
    model: str = "claude-opus-5-5"

    def summarize(self, order_id: str, order_data: dict[str, Any]) -> str:
        """Summarize a single order — stateless, no history retained."""
        # No instance-level state — prompt is local to this call
        _ = {
            "role": "user",
            "content": f"Summarize order {order_id}: {order_data}",
        }
        # Mocked LLM response
        return (
            f"Order {order_id}: {len(order_data.get('items', []))} items, "
            f"total ${order_data.get('total', 0):.2f}"
        )
```

### Tests

```python
import pytest
import tracemalloc


def test_memory_does_not_grow():
    agent = OrderSummaryAgent()
    for i in range(1_000):
        agent.summarize(f"ORD-{i:06d}", {"items": [], "total": 0.0})
    # Agent has no _history attribute in the fixed version
    assert not hasattr(agent, "_history")


def test_summarize_returns_string():
    agent = OrderSummaryAgent()
    result = agent.summarize("ORD-001", {"items": [{"sku": "A"}], "total": 45.0})
    assert isinstance(result, str)
    assert len(result) > 0
```

### Post-mortem (Scenario A)

**What happened:** The `OrderSummaryAgent` accumulated every request's prompt and response in a `_history` list on the instance. After 4 hours at 120 RPM (~28 800 requests), each with ~1 KB of data, the list held ~55 MB of entries; with Python object overhead the actual RSS reached 6 GB. The `gc_collection: collected=0` log confirmed the GC could not free anything because the list held live references. Root cause: the `_history` attribute was added during development to prototype multi-turn conversation support but was never scoped to a session — it accumulated across all independent requests sharing the same agent instance.

**Action items:** (1) Add a `tracemalloc`-based fixture to the CI test suite that asserts net allocation < 10 MB after 1 000 calls for all long-lived service objects. (2) Mandate code review check: instance-level mutable containers must have an explicit eviction/clear policy. (3) Add RSS memory alert at 80% of container memory limit.

---

## Scenario B — "Pool Apocalypse" — Solution

### Bug identification

Two code paths do not release the connection:
1. `return {"status": "failed", "reason": "payment"}` — returns before `self._pool.release(conn)`
2. The `except Exception` block logs but re-raises without releasing

### Fixed `CheckoutService`

```python
class CheckoutService:
    def __init__(self, pool: SimplePool) -> None:
        self._pool = pool

    def process_checkout(self, order_id: str, items: list[dict]) -> dict[str, Any]:
        conn = self._pool.acquire()
        try:
            result = self._validate_inventory(conn, items)
            if not result["ok"]:
                return {"status": "failed", "reason": "inventory"}

            payment = self._charge_payment(order_id)
            if not payment["ok"]:
                return {"status": "failed", "reason": "payment"}

            import time
            time.sleep(0.01)
            return {"status": "ok", "order_id": order_id}
        except Exception as exc:
            log.error("Checkout failed for %s: %s", order_id, exc)
            raise
        finally:
            self._pool.release(conn)  # ALWAYS runs
```

### Tests

```python
def test_connection_released_on_success():
    pool = SimplePool(max_size=5)
    service = CheckoutService(pool)
    service.process_checkout("CH-ALWAYS-OK", [])
    assert pool._pool.qsize() == 5


def test_connection_released_on_payment_decline():
    pool = SimplePool(max_size=5)
    service = CheckoutService(pool)
    # hash("CH-DECLINE") % 10 < 3 ensures decline
    result = service.process_checkout("CH-DECLINE", [])
    assert pool._pool.qsize() == 5


def test_connection_released_on_exception():
    import pytest
    pool = SimplePool(max_size=5)

    class BrokenService(CheckoutService):
        def _validate_inventory(self, conn, items):
            raise RuntimeError("inventory system down")

    service = BrokenService(pool)
    with pytest.raises(RuntimeError):
        service.process_checkout("CH-EXPLODE", [])
    assert pool._pool.qsize() == 5  # connection was returned
```

### Post-mortem (Scenario B)

**What happened:** v1.8.2 added a payment-decline early-return path in `process_checkout`. The developer forgot that the connection acquisition was not yet wrapped in a context manager — the `finally` block was missing. Every ~30% of requests (the decline rate) leaked a connection. After 22 minutes at normal load, all 100 connections were exhausted and held in `idle_in_transaction` state. New requests queued for 5 s then timed out. Root cause: the code review for v1.8.2 did not include a check for all return/raise paths following a resource acquisition.

**Action items:** (1) Enforce `with pool.acquire()` context manager — replace all raw `pool.acquire_raw()` calls; add a ruff/pylint rule that flags bare resource acquire outside a `with` block. (2) Add a deploy smoke test that runs 50 concurrent checkouts and asserts `pool.qsize() == pool.max_size` after completion. (3) Add a DB "idle in transaction" metric alert at > 20% of pool size.

---

## Scenario C — "The Hallucinating Analyst" — Solution

### Bug identification

Lines in `SYSTEM_PROMPT_V2_BUGGY` that cause regression:
- `"If you don't have exact data, provide a helpful estimate"` — explicitly permits confabulation
- `"Include relevant context like shipping and taxes when discussing totals"` — encourages adding numbers not in the order data
- `"Always be encouraging and positive!"` — shifts the optimization objective from accuracy to tone

### Fixed prompt

```python
SYSTEM_PROMPT_V2_FIXED = """You are a helpful order management assistant.
Answer questions ONLY using the order data provided in the context below.
Do not infer, estimate, or add information not present in the data.
If the data does not contain the answer, say "I don't have that information."
Format amounts exactly as they appear in the data.
Keep responses concise and friendly."""
```

### Tests

```python
def test_v1_passes_eval():
    analyst = OrderAnalyst(SYSTEM_PROMPT_V1)
    score = run_eval(analyst, "v1")
    assert score == 1.0, f"V1 should pass all eval cases, got {score:.0%}"


def test_v2_fixed_passes_eval():
    analyst = OrderAnalyst(SYSTEM_PROMPT_V2_FIXED)
    score = run_eval(analyst, "v2-fixed")
    assert score == 1.0, f"V2-fixed should pass all eval cases, got {score:.0%}"
```

### Regression gate

```python
def prompt_regression_gate(
    v1_prompt: str,
    v2_prompt: str,
    eval_cases: list[EvalCase],
    threshold: float = 0.0,
) -> bool:
    v1_analyst = OrderAnalyst(v1_prompt)
    v2_analyst = OrderAnalyst(v2_prompt)
    v1_score = run_eval(v1_analyst, "gate-v1")
    v2_score = run_eval(v2_analyst, "gate-v2")
    regression = v1_score - v2_score
    return regression <= threshold


def test_gate_blocks_regression():
    ok = prompt_regression_gate(SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2_BUGGY, EVAL_CASES)
    assert ok is False  # regression detected — should block deploy

def test_gate_passes_fixed():
    ok = prompt_regression_gate(SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2_FIXED, EVAL_CASES)
    assert ok is True
```

### Post-mortem (Scenario C)

**What happened:** A prompt change intended to make the agent "more helpful and conversational" added explicit permission to estimate when exact data is unavailable. The model faithfully followed this instruction and invented plausible-sounding order totals. The change was tested manually on one or two examples where it "felt better" but was never run against the existing eval suite. The regression caused a 3× spike in customer support tickets with wrong amounts. Root cause: the team had an eval suite (`EVAL_CASES`) but it was not wired into the deploy gate — prompt changes could be shipped without running it.

**Action items:** (1) Add `prompt_regression_gate` as a required CI step: any prompt change must achieve ≥ v1 score before merge. (2) Expand `EVAL_CASES` to cover the top 20 question types from production logs. (3) Add a rule to the prompt review checklist: "does this prompt give the model permission to estimate or fabricate?" — if yes, requires an additional 1 000-case eval run.

---

## Wrap-up reflection answers

1. **Shared structural pattern**: All three bugs were missing *resource/state cleanup on non-happy paths* — the happy path worked; errors and early returns bypassed cleanup.

2. **Hardest step under pressure**: Step 2 (Hypothesize) — tunnel vision sets in quickly. The fix is to write down multiple hypotheses before touching anything.

3. **CI gates that would have caught each**:
   - Scenario A: RSS memory assertion over 1 000 calls
   - Scenario B: concurrent request smoke test asserting pool invariant
   - Scenario C: prompt regression eval gate on every PR
