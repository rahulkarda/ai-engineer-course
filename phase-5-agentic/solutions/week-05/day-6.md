# Solutions — Week 05 · Day 6: Saturday Deep Practice

Day 6 is a guided build session. The solution is the complete `week05_wrapper` package. Key implementation notes are provided here; students should have the package fully working after Part 6.

---

## `models.py` — complete

See the lesson file. All models are fully specified.

---

## `guardrails.py` — reference implementation

```python
# week05_wrapper/guardrails.py
import re
from .models import GuardrailAction, GuardrailResult

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"jailbreak",
    r"DAN mode",
    r"act as if you have no restrictions",
    r"disregard your guidelines",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[\w._%+-]+@[\w.-]+\.\w{2,}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
}


def check_prompt_injection(text: str) -> GuardrailResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                rule="prompt_injection",
                detail=f"Matched: {pattern}",
            )
    return GuardrailResult(action=GuardrailAction.PASS, rule="prompt_injection")


def check_pii(text: str) -> GuardrailResult:
    found = [name for name, pat in PII_PATTERNS.items() if re.search(pat, text)]
    if found:
        return GuardrailResult(
            action=GuardrailAction.WARN,
            rule="pii",
            detail=f"PII types: {found}",
        )
    return GuardrailResult(action=GuardrailAction.PASS, rule="pii")


def check_length(text: str, max_chars: int = 2000) -> GuardrailResult:
    if len(text) > max_chars:
        return GuardrailResult(
            action=GuardrailAction.WARN,
            rule="length",
            detail=f"{len(text)} > {max_chars}",
        )
    return GuardrailResult(action=GuardrailAction.PASS, rule="length")


def check_topic_safety(text: str, prohibited: list[str]) -> GuardrailResult:
    for topic in prohibited:
        if re.search(re.escape(topic), text, re.IGNORECASE):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                rule="topic_safety",
                detail=f"Prohibited topic: {topic}",
            )
    return GuardrailResult(action=GuardrailAction.PASS, rule="topic_safety")


def run_guardrails(
    text: str,
    checks: list,
) -> list[GuardrailResult]:
    """Run checks in order; stop at first BLOCK."""
    results: list[GuardrailResult] = []
    for check in checks:
        result = check(text)
        results.append(result)
        if result.action == GuardrailAction.BLOCK:
            break
    return results
```

---

## `reflection.py` — `_reflect` implementation

```python
def _reflect(self, task: str, draft: str, verdict: CriticVerdict) -> str:
    prompt = (
        f"Task: {task}\n\nYour output:\n{draft}\n\n"
        f"Why it failed: {verdict.revision_request or verdict.summary}\n\n"
        "Write ONE sentence starting with 'I should' that captures the key lesson."
    )
    resp = self.client.messages.create(
        model=self.model, max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def run(self, task: str) -> RunResult:
    audit: list[str] = []
    output = ""
    passed = False
    for attempt in range(1, self.max_attempts + 1):
        context = "\n".join(self._reflections[-self.max_reflections:])
        audit.append(f"Attempt {attempt}: worker called")
        output = self.worker.run(task, context=context)
        audit.append(f"Attempt {attempt}: critic called")
        verdict = self.critic.evaluate(task, output)

        if verdict.verdict == Verdict.PASS:
            passed = True
            audit.append(f"Attempt {attempt}: PASS")
            break

        audit.append(f"Attempt {attempt}: FAIL — {verdict.summary}")
        if attempt < self.max_attempts:
            lesson = self._reflect(task, output, verdict)
            self._reflections.append(lesson)
            self._history.append(
                ReflectionEntry(attempt=attempt, draft=output, verdict=verdict, lesson=lesson)
            )

    return RunResult(
        final_output=output,
        passed=passed,
        attempts=attempt,
        reflections=list(self._reflections),
        audit_log=audit,
    )
```

---

## `supervisor.py` — `run` implementation

```python
def run(self, task: str) -> RunResult:
    # 1. Input guardrails
    self._log_event("INPUT CHECK start")
    input_checks = run_guardrails(task, [check_prompt_injection])
    for r in input_checks:
        self._log_event(f"INPUT {r.rule} → {r.action.value}: {r.detail}")
        if r.action == GuardrailAction.BLOCK:
            raise EscalationError(f"Input blocked by {r.rule}: {r.detail}")

    # 2. Reflection loop
    self._log_event("REFLECTION LOOP start")
    try:
        result = self.reflection_loop.run(task)
    except Exception as exc:
        self._log_event(f"REFLECTION LOOP ERROR: {exc}")
        raise

    # 3. Output guardrails
    self._log_event("OUTPUT CHECK start")
    output_checks = run_guardrails(
        result.final_output,
        [check_pii, lambda t: check_length(t, 2000)],
    )
    for r in output_checks:
        self._log_event(f"OUTPUT {r.rule} → {r.action.value}: {r.detail}")
        if r.action == GuardrailAction.BLOCK:
            raise EscalationError(
                f"Output blocked by {r.rule} after all attempts. Audit: {self._audit}"
            )

    self._log_event("SUCCESS")
    return RunResult(
        final_output=result.final_output,
        passed=result.passed,
        attempts=result.attempts,
        reflections=result.reflections,
        audit_log=self._audit + result.audit_log,
    )
```

---

## `conftest.py` — full fixture

The fixture in the lesson file is complete. Key usage pattern:

```python
def test_critic_pass(mock_anthropic):
    verdict_json = json.dumps({
        "verdict": "PASS", "overall_score": 8,
        "dimensions": [{"dimension": "accuracy", "score": 8, "reasoning": "ok"}],
        "summary": "Good", "revision_request": None,
    })
    client = mock_anthropic(["worker output", verdict_json])
    # inject client, call critic.evaluate(...)
    # assert verdict.passed is True
```

---

## Common test failures and fixes

**`json.JSONDecodeError` in CriticAgent**: mock response must be valid JSON — no trailing text. Use `json.dumps(data)` not string literals.

**`CriticVerdict` validation error**: ensure `dimensions` is a list of dicts with all required fields, not just a list of strings.

**Reflection loop runs forever in tests**: ensure mock client cycles correctly — the `mock_anthropic` fixture uses `min(call_count, len(responses) - 1)` to avoid IndexError on repeat calls.
