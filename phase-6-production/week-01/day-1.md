# Phase 6 · Week 01 · Day 1 — Why Distributed Systems + The Fallacies

> Objective: Understand the motivations for distributing computation and internalize the eight classic fallacies that cause production failures.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic]]

## Concept

### Why distribute at all?

A single process on a single machine has hard limits:

| Limit | What breaks |
|---|---|
| CPU/memory ceiling | A model inference server saturates one GPU |
| Storage capacity | A vector store with 1 B embeddings won't fit on one disk |
| Availability | One machine crash = total outage |
| Latency to users | A server in Frankfurt is slow for Tokyo users |
| Blast radius | A bug in one component can take down everything |

Distributing solves these by spreading work, data, and responsibility across multiple machines — but immediately introduces a new class of problems that simply do not exist on a single node: **partial failures, network latency, clock skew, and concurrency across independent state**.

AI systems hit every limit above. A RAG pipeline may spread across an embedding service, a vector database, an LLM inference cluster, and a reranker — each a separate process, potentially on separate machines in different availability zones.

### The eight fallacies of distributed computing

Peter Deutsch (Sun Microsystems, 1994) codified the assumptions engineers routinely make that are always false in a real network. Every distributed system bug traces back to at least one.

| # | Fallacy | Reality |
|---|---|---|
| 1 | The network is reliable | Packets drop, NICs fail, switches reboot |
| 2 | Latency is zero | Even LAN has microseconds; WAN has milliseconds |
| 3 | Bandwidth is infinite | Serialisation + transfer cost real money and time |
| 4 | The network is secure | Default traffic is plaintext; injection attacks exist |
| 5 | Topology doesn't change | IPs change, nodes scale in/out, DNS TTLs expire |
| 6 | There is one administrator | Orgs have teams; config drifts; firewalls conflict |
| 7 | Transport cost is zero | Serialisation CPU, egress fees, compression overhead |
| 8 | The network is homogeneous | Linux + Windows + ARM + x86 + Python 3.9 vs 3.12 |

### Why this matters for AI engineers

An LLM call is a network call. A vector DB query is a network call. An agent invoking a tool is a network call. Every line of agentic code you write operates in an environment where all eight fallacies are lurking.

Concrete failure mode per fallacy in an agentic context:

- **Fallacy 1 (reliable):** The tool-call response never arrives. The agent hangs or retries forever.
- **Fallacy 2 (zero latency):** You assume an embedding lookup takes 1 ms. It takes 40 ms under load. Your p99 SLA breaks.
- **Fallacy 3 (infinite bandwidth):** You send a 200 KB context window to a remote LLM on every agent step. 10 parallel agents → 2 MB/s sustained egress.
- **Fallacy 5 (stable topology):** A vector DB pod restarts mid-query. Your hardcoded IP is stale.
- **Fallacy 8 (homogeneous):** Your orchestrator runs Python 3.12 with Pydantic v2. The tool server runs Python 3.9 with Pydantic v1. Schema validation silently diverges.

## How it works internally

### The vertical scaling wall

Modern CPUs have ~128 physical cores and ~12 TB RAM at the top end. These are not cheap. Beyond that, you must go horizontal — multiple machines coordinating over a network. The moment you do, you cross the **distributed systems boundary**: you can no longer use shared memory, you can no longer rely on atomic operations across state, and you must handle the possibility that messages are delayed or lost.

### The cost model

Distributing is not free. You pay:

1. **Coordination overhead** — nodes must agree on who does what
2. **Serialisation tax** — data must be marshalled across process boundaries
3. **Operational complexity** — more moving parts, more failure surfaces
4. **Eventual consistency** — copies of data diverge and must be reconciled

The decision to distribute should be driven by a genuine capacity or availability need, not by fashion.

### Latency numbers (2024 approximate)

```
L1 cache hit:          ~1 ns
RAM access:            ~100 ns
SSD sequential read:   ~100 µs
LAN round-trip:        ~500 µs
Same-DC TCP round-trip: ~1 ms
Cross-region TCP:      ~70 ms  (US-East → EU-West)
LLM API call (p50):    ~500 ms – 2 s
```

Understanding these numbers is how you decide what to cache locally vs. what to fetch remotely.

## Worked example

The following simulation makes the fallacies tangible. It models an unreliable network channel and shows how a naive RPC-style call fails under the fallacies.

```python
"""
day1_fallacies.py — Simulate network fallacies affecting an agent tool call.

Run: python day1_fallacies.py
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger(__name__)

# Seed for reproducibility; override with env var
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)


@dataclass
class NetworkConfig:
    """Tuneable network conditions — the fallacies made concrete."""

    packet_loss_rate: float = 0.15        # Fallacy 1: unreliable
    base_latency_ms: float = 5.0          # Fallacy 2: non-zero latency
    latency_jitter_ms: float = 20.0       # Fallacy 2: variable latency
    bandwidth_cap_kbps: float = 1000.0    # Fallacy 3: finite bandwidth
    node_failure_rate: float = 0.05       # Fallacy 5: topology changes


@dataclass
class ToolCallResult:
    success: bool
    payload: str | None
    latency_ms: float
    error: str | None = None


def simulate_network_call(
    payload_bytes: int,
    config: NetworkConfig,
) -> ToolCallResult:
    """Simulate one RPC call through an unreliable network."""
    start = time.monotonic()

    # Fallacy 5: node may have gone away
    if random.random() < config.node_failure_rate:
        return ToolCallResult(
            success=False,
            payload=None,
            latency_ms=(time.monotonic() - start) * 1000,
            error="ConnectionRefused: target node is down",
        )

    # Fallacy 1: packet loss
    if random.random() < config.packet_loss_rate:
        # Simulate timeout — caller waits before giving up
        time.sleep(0.03)
        return ToolCallResult(
            success=False,
            payload=None,
            latency_ms=(time.monotonic() - start) * 1000,
            error="Timeout: no response received",
        )

    # Fallacy 2: latency is non-zero and variable
    latency_s = (
        config.base_latency_ms + random.uniform(0, config.latency_jitter_ms)
    ) / 1000.0
    time.sleep(latency_s)

    # Fallacy 3: bandwidth cost
    transfer_time_s = (payload_bytes * 8) / (config.bandwidth_cap_kbps * 1000)
    time.sleep(transfer_time_s)

    elapsed_ms = (time.monotonic() - start) * 1000
    return ToolCallResult(
        success=True,
        payload="result_data",
        latency_ms=elapsed_ms,
    )


def run_agent_steps(n_steps: int = 20) -> None:
    config = NetworkConfig()
    successes = 0
    failures = 0
    total_latency_ms = 0.0

    log.info("Starting %d simulated agent tool calls", n_steps)
    for i in range(n_steps):
        # Each "tool call" sends ~4 KB (e.g. a small context chunk)
        result = simulate_network_call(payload_bytes=4096, config=config)
        total_latency_ms += result.latency_ms

        if result.success:
            successes += 1
            log.info("Step %02d OK  latency=%.1f ms", i, result.latency_ms)
        else:
            failures += 1
            log.warning("Step %02d ERR latency=%.1f ms  %s", i, result.latency_ms, result.error)

    avg_ms = total_latency_ms / n_steps
    log.info(
        "Done: %d/%d succeeded  avg_latency=%.1f ms  failure_rate=%.0f%%",
        successes,
        n_steps,
        avg_ms,
        failures / n_steps * 100,
    )


if __name__ == "__main__":
    run_agent_steps()
```

Run it and observe: even at 15% packet loss, roughly 1 in 6 agent steps fails. A naive agent that assumes success will silently drop results or get stuck.

## Exercises

1. Modify `NetworkConfig` so packet loss is 0% and jitter is 0 ms. Verify that all 20 steps succeed. Now raise `packet_loss_rate` to 0.5. What is the expected number of failed steps?

2. Add a simple **retry loop** (up to 3 attempts) around `simulate_network_call`. Measure how the end-to-end latency changes compared to no-retry.

3. Identify which of the eight fallacies is *not* modelled in `simulate_network_call`. Add a simulation of that fallacy and explain your design choice in a comment.

4. A colleague proposes adding a 10-second `time.sleep` as a "retry backoff". What is wrong with a fixed sleep vs. exponential backoff? Write a helper `exponential_backoff(attempt: int, base_ms: float = 100.0) -> float` that returns the delay in seconds.

(Solutions: ../solutions/week-01/day-1.md)

## Self-check

- [ ] I can name all eight fallacies from memory and give one concrete AI-system failure mode per fallacy
- [ ] I can explain the vertical scaling wall and why horizontal distribution becomes necessary
- [ ] I can read the latency numbers table and use it to estimate whether a given operation should be cached locally
- [ ] I can modify `NetworkConfig` to simulate different network conditions and explain what each parameter models

## Supplementary reading

- Designing Data-Intensive Applications (Kleppmann) — Chapter 1: Reliable, Scalable, Maintainable Applications
- [The Eight Fallacies of Distributed Computing](https://web.archive.org/web/20171107014323/http://blog.fogcreek.com/eight-fallacies-of-distributed-computing-tech-talk/) — Peter Deutsch / James Gosling
- [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html)
