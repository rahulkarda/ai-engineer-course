# Phase 5 · Week 07 · Day 7 — Project Day & Reflection

> Objective: Complete and polish the week-7 multi-agent project; review architecture decisions; prepare for assessment.  ·  Time: ~3 hr project + 1 hr reflection  ·  Prereqs: [[day-6]]

## Morning: Project completion (2 hr)

Use the first block to finish any incomplete parts of the Saturday build:

### Completion checklist

- [ ] All 5 agents have unit tests in `tests/`
- [ ] `PipelineSupervisor.run()` fully implemented
- [ ] Blackboard populated at each stage
- [ ] `pyproject.toml` configured (pytest, ruff, dependencies)
- [ ] `main.py` runs end-to-end against the real Anthropic API (manually tested)
- [ ] All tests pass: `pytest tests/ -v`

### Common issues and fixes

**Issue: ThreadPoolExecutor swallows exceptions**  
Fix: call `fut.result()` inside a try/except, or use `as_completed` with exception handling:
```python
for fut in as_completed(futures):
    try:
        finding = fut.result()
    except Exception as exc:
        logger.exception("researcher failed: %s", exc)
        findings.append(default_finding)
```

**Issue: JSON parse error from LLM (QueryAnalyser)**  
Fix: add a retry with a more explicit prompt:
```python
for attempt in range(3):
    raw = call_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        prompt += "\nIMPORTANT: respond ONLY with valid JSON, no other text."
raise ValueError("LLM refused to return JSON after 3 attempts")
```

**Issue: Blackboard watcher fires before agent is ready**  
Fix: always set up watchers before starting the agents that will write to those keys.

**Issue: Long FactChecker response breaks regex parsing**  
Fix: parse by splitting on `CLAIM:` pattern rather than line-by-line.

## Midday: Architecture review (1 hr)

Re-examine every design decision made this week. For each, write one sentence justifying it.

### Review prompts

1. **Why a blackboard instead of passing data directly between agents?**  
   When agents need to access different subsets of shared state at different times, a central blackboard decouples producers from consumers better than point-to-point passing.

2. **Why `ThreadPoolExecutor` for concurrent research instead of `asyncio`?**  
   Anthropic's Python SDK uses `httpx` which supports both. For CPU-bound coordination overhead, threads are simpler. For pure I/O, asyncio's `asyncio.gather` would be slightly more efficient. Either works; pick what your codebase already uses.

3. **Why a separate `PublisherAgent` instead of formatting in `PipelineSupervisor`?**  
   Single responsibility: supervisor coordinates flow, publisher formats output. Swap publisher for a different format (HTML, Slack, PDF) without touching supervisor.

4. **Why mock the LLM in tests instead of a smaller model?**  
   Speed and reproducibility. A mock runs in microseconds and always returns the same output. A small model takes seconds and may return different text, making assertions flaky.

5. **Why `Pydantic` for all data that crosses agent boundaries?**  
   Validated schema at the boundary. If a researcher returns a malformed dict, the `ResearchFinding` constructor raises immediately — close to the source rather than mysteriously failing later.

### Architecture anti-patterns to avoid

| Anti-pattern | Why it hurts | Fix |
|---|---|---|
| Agents share a mutable Python dict | Race conditions, silent data loss | Blackboard with locking |
| Supervisor knows agent internals | Tight coupling, untestable | Agents expose only typed input/output |
| LLM calls in tests | Slow, flaky, costly | Mock client |
| Errors swallowed silently | Impossible to debug | Explicit error messages + structured ERROR type |
| God supervisor (>300 lines) | Hard to read, test, extend | Extract per-stage methods or classes |

## Afternoon: Reflection & self-assessment (1 hr)

### What you built this week

| Day | Skill acquired |
|---|---|
| 1 | Four multi-agent topologies |
| 2 | Typed message bus with mailbox pattern |
| 3 | Thread-safe blackboard with CAS |
| 4 | Capability routing and role specialization |
| 5 | Debate loop and conflict resolution |
| 6–7 | End-to-end pipeline integrating all primitives |

### Reflection questions

Write short answers (3–5 sentences each) in your learning journal:

1. Which of the four topologies did your pipeline use? Could you redesign it to use a different one? What would change?
2. The blackboard adds latency (lock contention) compared to direct function calls. When is that overhead worth it?
3. Your FactChecker debated nothing — it just checked claims. How would you modify the pipeline to add a real debate round between Synthesiser and FactChecker?
4. If you had to scale this pipeline to handle 100 questions per minute, what would you change first?
5. What did you find hardest this week? Plan one targeted practice session to address it.

### Preview of Week 8

Week 8 covers **agent memory and long-term context**: episodic memory, semantic memory, procedural memory, retrieval-augmented agent context, and memory consolidation. The multi-agent pipeline you built this week is the base — next week you'll add persistent memory so agents learn across sessions.

## Exercises

1. Add a `retry_with_backoff` decorator to `SpecialistRunner.execute()` that retries on `anthropic.APIError` up to 3 times with exponential backoff.
2. Implement a `PipelineMetrics` class that records: total latency, per-stage latency, LLM call count, token usage. Attach it to `PipelineSupervisor`.
3. Add a `--dry-run` flag to `main.py` that replaces all LLM calls with mock responses and prints the pipeline trace without making API calls.
4. Refactor `FactCheckerAgent.check()` to process claims in batches of 3 (one LLM call per batch) instead of all at once. Measure the token savings.
5. Write a `test_supervisor_resilience` test: one researcher raises an exception mid-run. Verify the pipeline still completes with the remaining researcher's findings.

(Solutions: ../solutions/week-07/day-7.md)

## Self-check

- [ ] My project passes all tests without real LLM calls
- [ ] I can justify each architectural decision in one sentence
- [ ] I can identify the anti-patterns and explain why they're harmful
- [ ] I understand what comes next (Week 8: agent memory)

## Supplementary links

- [LangGraph Conceptual Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [Python concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
