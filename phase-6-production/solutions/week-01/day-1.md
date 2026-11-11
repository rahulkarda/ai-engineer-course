# Solutions — Day 1: Why Distributed + Fallacies

## Exercise 1 — Verify success rate and predict failure count

```python
# Set drop rate to 0 → expect 20/20 success
config_clean = NetworkConfig(packet_loss_rate=0.0, latency_jitter_ms=0.0)

# With drop rate 0.5 → expected failures = 20 * 0.5 = 10
# Plus node_failure_rate=0.05 → ~1 additional failure
# Expected: ~11 failures out of 20 steps
```

With `packet_loss_rate=0.0` and `latency_jitter_ms=0.0`: all 20 steps succeed.
With `packet_loss_rate=0.5`: expected failures ≈ 10 (binomial with n=20, p=0.5).

## Exercise 2 — Retry loop with latency measurement

```python
import time

def simulate_with_retry(
    payload_bytes: int,
    config: NetworkConfig,
    max_retries: int = 3,
) -> ToolCallResult:
    """Retry on failure with simple linear backoff."""
    total_start = time.monotonic()
    for attempt in range(max_retries):
        result = simulate_network_call(payload_bytes, config)
        if result.success:
            return ToolCallResult(
                success=True,
                payload=result.payload,
                latency_ms=(time.monotonic() - total_start) * 1000,
            )
        log.warning("Retry %d/%d after failure: %s", attempt + 1, max_retries, result.error)
    return ToolCallResult(
        success=False,
        payload=None,
        latency_ms=(time.monotonic() - total_start) * 1000,
        error="Max retries exceeded",
    )
```

With 3 retries and 15% drop rate, expected remaining failure rate ≈ 0.15³ ≈ 0.3%. Latency increases by 1–3× the per-call latency.

## Exercise 3 — Missing fallacy

Fallacy 4 (The network is secure) is not modelled. All other fallacies have direct representations: #1 = packet_loss_rate, #2/#3 = latency/bandwidth, #5 = node_failure_rate.

```python
@dataclass
class NetworkConfig:
    # ... existing fields ...
    # Fallacy 4: The network is secure
    # Model: probability a message is intercepted/corrupted in transit
    corruption_rate: float = 0.0


def simulate_network_call(payload_bytes: int, config: NetworkConfig) -> ToolCallResult:
    # ... existing checks ...

    # Fallacy 4: payload corruption
    if random.random() < config.corruption_rate:
        return ToolCallResult(
            success=False,
            payload=None,
            latency_ms=(time.monotonic() - start) * 1000,
            error="DataCorruption: payload failed integrity check (MITM or bit flip)",
        )
    # ... rest of function ...
```

## Exercise 4 — Exponential backoff

```python
import math

def exponential_backoff(attempt: int, base_ms: float = 100.0, max_ms: float = 30_000.0) -> float:
    """
    Return backoff delay in seconds for the given attempt number (0-indexed).
    Doubles each attempt, capped at max_ms, with ±25% jitter.
    """
    delay_ms = min(base_ms * (2 ** attempt), max_ms)
    # Add jitter to prevent thundering herd
    jitter = random.uniform(0.75, 1.25)
    return (delay_ms * jitter) / 1000.0
```

Fixed sleep problems: (1) too long = slow recovery; (2) too short = thundering herd (all retrying at once, overloading a recovering server); (3) no jitter = all clients retry synchronously. Exponential backoff spreads retry load and backs off proportionally to failure severity.
