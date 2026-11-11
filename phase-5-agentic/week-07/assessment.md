# Week 07 Assessment — Multi-Agent Systems

> Time: ~1 hr  ·  Pass gate: 75%

## Conceptual (interview-style) — 25%

Answer each question in 3–6 sentences.

1. You need to process 500 documents per hour using LLMs. A single-agent sequential pipeline takes 10 seconds per document. Explain two multi-agent strategies to meet the throughput target, and state the tradeoff of each.

2. Describe the blackboard pattern: what are the three components, what problem does it solve, and when would you choose it over direct agent-to-agent messaging?

3. The contract net protocol is a coordination mechanism for multi-agent systems. Describe its four phases and give a concrete example of a task it suits well.

4. Explain optimistic concurrency control (CAS) in the context of a shared blackboard. Why is it generally preferred over pessimistic locking in a multi-agent system?

5. A supervisor agent delegates task T to worker A, which fails. List three specific strategies the supervisor can use to recover, and state which you would choose by default and why.

6. Explain why a debate round with a judge agent produces better answers than a single agent reviewing its own output. Reference the concept of confirmation bias.

7. You have a multi-agent pipeline in production. Agent B, which receives messages from Agent A, occasionally returns garbage output. How do you detect this problem? How do you prevent it from propagating downstream?

## Coding challenge — 40%

Build a **3-agent code review system** in Python 3.12+.

### Specification

**Agents:**
- `SecurityReviewer` — checks for security issues (eval/exec/shell=True/hardcoded secrets)
- `StyleReviewer` — checks for style issues (type hints, naming, line length)
- `ReviewAggregator` — receives both reviews, produces a final `ReviewReport`

**Requirements:**
1. All agents implement a common `Reviewer` abstract base class with `review(code: str) -> ReviewResult`
2. `ReviewResult` is a Pydantic model with: `reviewer: str`, `issues: list[Issue]`, `severity: Severity` (enum: LOW/MEDIUM/HIGH/CRITICAL)
3. `Issue` is a Pydantic model with: `line: int | None`, `description: str`, `suggestion: str`
4. `ReviewAggregator` runs both reviewers concurrently using `ThreadPoolExecutor`, merges issues, and sets overall severity to the max of both reviewers
5. All LLM calls use `model="claude-haiku-5-5"` and `os.getenv("ANTHROPIC_API_KEY")`
6. Tests mock the Anthropic client — no real API calls in tests
7. Pass `ruff check` (or equivalent linting)

### Acceptance criteria

```python
# This test must pass:
from unittest.mock import MagicMock
from code_review import ReviewAggregator, ReviewReport, Severity

def test_aggregator_uses_max_severity():
    client = MagicMock()
    client.messages.create.return_value.content = [
        MagicMock(text='{"issues": [], "severity": "HIGH"}')
    ]
    agg = ReviewAggregator(client)
    report = agg.review("def f(): eval(input())")
    assert isinstance(report, ReviewReport)
    assert report.overall_severity == Severity.HIGH
```

### Bonus (not required for pass):
- Add a third `PerformanceReviewer`
- Implement a debate round between `SecurityReviewer` and `StyleReviewer` when they disagree on severity

## Debugging challenge — 20%

The following code has **three bugs**. Identify each, explain what goes wrong at runtime, and provide the fix.

```python
import threading
from collections import defaultdict

class BuggyBlackboard:
    def __init__(self):
        self._store = defaultdict(dict)  # Bug 1

    def write(self, key: str, value, author: str, expected_version: int = None):
        existing = self._store.get(key)
        if expected_version is not None:
            if existing["version"] != expected_version:  # Bug 2
                return False
        version = existing["version"] + 1 if existing else 1
        self._store[key] = {"value": value, "version": version, "author": author}
        return True

    def read_all(self):
        return dict(self._store)  # Bug 3

    def watch(self, prefix: str, callback):
        # start watcher thread
        def _watch():
            while True:
                for key, entry in self._store.items():
                    if key.startswith(prefix):
                        callback(entry)
                threading.Event().wait(0.1)
        threading.Thread(target=_watch, daemon=True).start()
```

For each bug: (a) name it, (b) describe the runtime failure mode, (c) write the fix.

## Code quality — 15%

Review this agent class and list **at least four** specific improvements with justification:

```python
import anthropic

class Agent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key="REDACTED-HARDCODED-KEY")  # BUG: hardcoded secret
        self.results = []
        self.running = True

    def do_task(self, t):
        print(f"Starting task: {t}")
        r = self.client.messages.create(
            model="claude-opus-5-5",
            max_tokens=9999,
            messages=[{"role": "user", "content": t}]
        )
        self.results.append(r.content[0].text)
        print(f"Done: {r.content[0].text}")
        return r.content[0].text

    def get_results(self):
        return self.results

    def clear_results(self):
        self.results = []
        return True
```

## Rubric

| Dimension | Weight | Passing criteria |
|---|---|---|
| Independent implementation | 40% | Code review system runs, all models typed, concurrent aggregation |
| Conceptual understanding | 25% | Correct answers on topology, blackboard, conflict resolution |
| Debugging & problem-solving | 20% | All 3 bugs identified correctly with accurate runtime description |
| Code quality & testing | 15% | At least 4 valid quality issues, tests mock LLM, no hardcoded keys |

**Pass gate: 75% total.**

Answer key: ../solutions/week-07/assessment.md
