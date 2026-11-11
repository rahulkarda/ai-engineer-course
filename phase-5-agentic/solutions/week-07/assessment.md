# Solutions — Week 07 Assessment

## Conceptual answers

### Q1: Two strategies for 500 docs/hr

**Strategy 1 — Concurrent scatter-gather pool**
Use a pool of N identical agents running in parallel (e.g., `ThreadPoolExecutor(max_workers=N)`). With N=10 agents each at 10 s/doc, throughput = 10 × 360 docs/hr = 3,600 docs/hr.  
Tradeoff: cost scales linearly with N (N × LLM token cost); also higher memory usage per worker.

**Strategy 2 — Hierarchical sub-batching**
A supervisor batches 10 docs and sends each batch to a specialist agent that processes them with a single multi-document prompt. One LLM call handles 10 docs.  
Tradeoff: reduced LLM calls (cost saving) but limited by context window size; quality may drop vs. one-doc-at-a-time.

### Q2: Blackboard pattern

Three components: **Blackboard** (structured central store), **Knowledge Sources/agents** (read/write the blackboard), **Control** (scheduler that decides which agent acts next).

It solves the problem of agents that need to read and write shared intermediate state without tight coupling — agents don't call each other directly, they leave/pick up information on the board.

Choose over direct messaging when: multiple agents need to read the same intermediate result, the order of agent activation is data-driven (reactive), or you need auditability (full state history in one place).

### Q3: Contract Net Protocol

Four phases:
1. **Announce** — manager broadcasts task description to all potential contractors.
2. **Bid** — interested contractors submit proposals (capability, cost, confidence).
3. **Award** — manager selects best proposal and notifies winner (rejects others).
4. **Execute & Report** — contractor performs task and reports result to manager.

Example: a documentation generator broadcasts "write a module docstring for this code" to a pool of writing agents; each bids with confidence; the highest-confidence available agent wins and writes the docstring.

### Q4: Optimistic concurrency (CAS) vs pessimistic locking

CAS (compare-and-swap): agent reads key + version, computes new value, writes only if version unchanged; fails and retries if conflicting write happened.  
Pessimistic: agent acquires a mutex before reading, holds it through write, releases after.

CAS preferred in multi-agent systems because: (a) agents rarely collide on the same key simultaneously so contention is low; (b) no deadlock risk (no lock held); (c) agents can do other work while their write fails (retry asynchronously); (d) pessimistic locking with multiple agents and multiple keys easily deadlocks if lock acquisition order differs.

### Q5: Recovery from worker failure

Three strategies:
1. **Retry to same worker** — re-send the original task. Works if failure was transient (network timeout, rate limit).
2. **Reroute to different worker** — route to a different agent with the same capability. Works if worker crashed.
3. **Graceful degradation** — log the failure, continue with partial results. Use when the failed subtask is non-critical or a fallback exists.

Default choice: retry once (most failures are transient), then reroute, then degrade. Avoids immediate task loss while not hanging indefinitely.

### Q6: Debate and confirmation bias

A single agent reviewing its own output suffers from **confirmation bias**: it already "believes" its initial answer and unconsciously selects evidence supporting it while discounting contradictions. The critique round is effectively a rubber-stamp.

A second adversarial agent is prompted to find weaknesses — it has no stake in the first answer being correct. Its critique surfaces genuine flaws. The judge reads both sides and makes a separate decision, free from commitment to either position. Research (du Plessis et al., 2023) shows debate reduces factual errors by 10–20% on verifiable benchmarks.

### Q7: Detecting and containing bad output from Agent B

**Detection**: schema validation (Pydantic) at the message boundary — if Agent B's output fails to parse into the expected model, it's caught immediately. Additionally, a lightweight confidence score or a secondary `ValidatorAgent` can flag outputs below a quality threshold.

**Prevention from propagating**: gate downstream agents — Agent C only runs when it receives a valid, schema-conforming output from Agent B. On validation failure, the supervisor either retries B, routes to a fallback agent, or injects a sentinel value and logs the incident.

---

## Coding challenge: 3-agent code review system

```python
"""code_review.py — 3-agent code review system."""
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import anthropic
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def max_of(cls, a: "Severity", b: "Severity") -> "Severity":
        order = [cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        return a if order.index(a) >= order.index(b) else b


class Issue(BaseModel):
    line:        int | None = None
    description: str
    suggestion:  str


class ReviewResult(BaseModel):
    reviewer: str
    issues:   list[Issue]
    severity: Severity


class ReviewReport(BaseModel):
    issues:           list[Issue]
    overall_severity: Severity


class Reviewer(ABC):
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    @abstractmethod
    def review(self, code: str) -> ReviewResult:
        ...

    def _call(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _parse(self, raw: str, reviewer_name: str) -> ReviewResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ReviewResult(reviewer=reviewer_name, issues=[], severity=Severity.LOW)
        issues = [Issue(**i) for i in data.get("issues", [])]
        severity = Severity(data.get("severity", "LOW"))
        return ReviewResult(reviewer=reviewer_name, issues=issues, severity=severity)


SECURITY_SYSTEM = """\
You are a security code reviewer. Identify: eval/exec usage, shell=True, hardcoded secrets,
SQL injection risks, unsafe deserialization. Respond ONLY with JSON:
{"issues": [{"line": null, "description": "...", "suggestion": "..."}], "severity": "LOW|MEDIUM|HIGH|CRITICAL"}
"""

STYLE_SYSTEM = """\
You are a style reviewer. Check: missing type hints, bad naming, lines > 100 chars,
missing docstrings, print() instead of logging. Respond ONLY with JSON:
{"issues": [{"line": null, "description": "...", "suggestion": "..."}], "severity": "LOW|MEDIUM|HIGH|CRITICAL"}
"""


class SecurityReviewer(Reviewer):
    def review(self, code: str) -> ReviewResult:
        raw = self._call(SECURITY_SYSTEM, f"Review:\n```python\n{code}\n```")
        return self._parse(raw, "security")


class StyleReviewer(Reviewer):
    def review(self, code: str) -> ReviewResult:
        raw = self._call(STYLE_SYSTEM, f"Review:\n```python\n{code}\n```")
        return self._parse(raw, "style")


class ReviewAggregator:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        self._reviewers = [SecurityReviewer(client), StyleReviewer(client)]

    def review(self, code: str) -> ReviewReport:
        results: list[ReviewResult] = []
        with ThreadPoolExecutor(max_workers=len(self._reviewers)) as pool:
            futures = {pool.submit(r.review, code): r for r in self._reviewers}
            for fut in as_completed(futures):
                results.append(fut.result())

        all_issues = [issue for r in results for issue in r.issues]
        overall = results[0].severity
        for r in results[1:]:
            overall = Severity.max_of(overall, r.severity)

        return ReviewReport(issues=all_issues, overall_severity=overall)


if __name__ == "__main__":
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    agg = ReviewAggregator(client)
    report = agg.review("def f(): eval(input())\n")
    print(f"Severity: {report.overall_severity}")
    for issue in report.issues:
        print(f"  - {issue.description}")
```

---

## Debugging challenge: three bugs

### Bug 1: `defaultdict(dict)` as the store

**What**: `defaultdict(dict)` silently creates an empty dict for any missing key instead of returning `None`. Code calling `self._store.get(key)` may get `{}` (falsy but not `None`) when expecting `None`.

**Runtime failure**: `existing["version"]` on the empty dict raises `KeyError`. Also, code checking `if existing:` sees `{}` as falsy — inconsistent.

**Fix**: Use a plain `dict`:
```python
self._store: dict[str, dict] = {}
```

### Bug 2: `existing["version"]` without None check

**What**: If `key` is not in `_store`, `_store.get(key)` returns `None`. Then `existing["version"]` raises `TypeError: 'NoneType' object is not subscriptable`.

**Runtime failure**: crash on any first-write with `expected_version` set.

**Fix**:
```python
if expected_version is not None:
    stored = existing["version"] if existing else 0
    if stored != expected_version:
        return False
```

### Bug 3: `read_all` returns a shallow copy — not thread-safe

**What**: `dict(self._store)` copies the outer dict but the values are still the same mutable dict objects. Another thread can mutate a value after the copy is returned, causing the caller to see inconsistent state.

**Runtime failure**: TOCTOU (time-of-check-to-time-of-use) data race — caller reads a "snapshot" that changes under it.

**Fix**:
```python
import copy

def read_all(self):
    with self._lock:
        return copy.deepcopy(self._store)
```

**Bonus bug** (not counted): the `watch` loop in `_watch()` iterates `self._store.items()` without a lock — `RuntimeError: dictionary changed size during iteration` if another thread writes concurrently. Fix: acquire lock and copy items first.

---

## Code quality review

Issues in the `Agent` class:

1. **Hardcoded API key** (`"REDACTED-HARDCODED-KEY"`) — critical security violation. Use `os.getenv("ANTHROPIC_API_KEY")` and raise `ValueError` if not set.

2. **No type hints** — `do_task(self, t)` has no parameter or return type. Should be `def do_task(self, t: str) -> str`.

3. **`print()` instead of `logging`** — violates the project's global standards. Replace with `logger.info(...)` and `logger.debug(...)`.

4. **`max_tokens=9999`** — wasteful. Use a reasonable limit (e.g., 1024 or 2048) matched to the expected response length.

5. **`get_results()` returns mutable internal list** — callers can modify `self.results` externally. Should return `list(self.results)` (a copy).

6. **No system prompt** — `messages.create` has no `system=` parameter. Agents need a role prompt; without it, outputs are unpredictable.

7. **`clear_results()` returns `True`** — unnecessary return value, misleading API. Should return `None`.

8. **`self.running` attribute is never used** — dead code; remove or document its purpose.
