# Week 10 Assessment

> Phase 5 · Agentic AI with LangGraph  
> Time: ~2 hr  ·  Pass threshold: ≥ 75%

---

## Conceptual (Interview-Style)

Answer each question in 3–6 sentences. Be precise. Assume your interviewer is a senior ML engineer who has read the LangGraph source.

**Q1.** What is the difference between `interrupt_before` and `interrupt_after` in `graph.compile()`? Give a concrete example of a workflow where one is clearly preferable to the other.

**Q2.** Explain the role of `thread_id` in LangGraph checkpointing. What happens if two concurrent requests share the same `thread_id`?

**Q3.** A LangGraph graph has a conditional edge that routes back to an earlier node. How does LangGraph prevent infinite loops? What is the developer's responsibility?

**Q4.** Compare `graph.stream()` and `graph.astream_events()`. In which situations would you choose each? What extra information does `astream_events()` expose that `stream()` does not?

**Q5.** You have a typed state field `results: list[str]` and two nodes that both append to it. If you annotate it as `results: list[str]` (no reducer), what happens? How do you fix it?

**Q6.** Explain the plan-execute pattern and map each stage (planner, executor, replanner) to a LangGraph node. What information must the replanner receive to make a meaningful decision?

**Q7.** When should you use `langgraph.prebuilt.create_react_agent` vs building the think/act graph manually? Give one scenario where the built-in is sufficient and one where it forces you to build manually.

**Q8.** A colleague argues: "LCEL chains are simpler and more composable than LangGraph. We should avoid LangGraph unless forced to use it." Write a response that concedes where they are right and explains the three specific capabilities that justify LangGraph's complexity.

---

## Coding Challenge

### Specification

Build a **document-analysis LangGraph agent** that processes a raw text document through a four-stage pipeline with HITL approval.

**Pipeline stages:**

1. **`classify` node** — Classify the document type (e.g., `"technical_spec"`, `"meeting_notes"`, `"legal_contract"`, `"other"`). Output must be a typed Pydantic model `ClassificationResult(doc_type: str, confidence: float)` stored in state.

2. **`extract` node** — Extract key information appropriate to the document type. Output must be a typed Pydantic model `ExtractionResult(title: str, summary: str, key_points: list[str], entities: list[str])` stored in state.

3. **`grade` node** — Grade the extraction quality on a scale of 1–10 using an LLM as judge. Output must be a typed Pydantic model `GradeResult(score: int, reasoning: str)`. Add a **conditional retry edge**: if `score < 7`, route back to `extract` (up to 2 retries); if `score >= 7`, route to `await_approval`.

4. **`await_approval` node** — This node is a pass-through (does nothing). The graph is compiled with `interrupt_before=["await_approval"]` so a human can review `extraction_result` and either approve or override it before the final `output` node runs.

5. **`output` node** — Log the final approved extraction result and set `complete: bool = True` in state.

**Requirements:**

- `AgentState` is a `TypedDict` with typed fields: `document: str`, `classification: ClassificationResult | None`, `extraction: ExtractionResult | None`, `grade: GradeResult | None`, `retry_count: int`, `approved: bool`, `complete: bool`.
- All three LLM calls use `claude-sonnet-5-5`.
- `ANTHROPIC_API_KEY` loaded via `os.getenv`.
- `MemorySaver` checkpointer with a fixed `thread_id`.
- At least **2 pytest tests** with mocked LLM:
  - Test 1: happy path — score >= 7 on first attempt, human approves, `complete == True`.
  - Test 2: retry path — score < 7 on first grade, score >= 7 on second, human approves.
- All LLM calls mocked; no real API calls in tests.
- `logging` not `print`.

**Acceptance criteria checklist:**

- [ ] All five nodes implemented as pure functions (no side effects except logging).
- [ ] `ClassificationResult`, `ExtractionResult`, `GradeResult` are Pydantic v2 `BaseModel` subclasses with field types.
- [ ] Conditional retry edge correctly limits retries to 2 (does not loop forever).
- [ ] `interrupt_before=["await_approval"]` is set on `compile()`.
- [ ] `update_state()` is used in the resume path to inject `approved=True`.
- [ ] Both tests pass with `pytest -q`.
- [ ] No hardcoded API keys. No `print` statements.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Indicators |
|---|---|---|
| Independent implementation | 40% | Code runs end-to-end without scaffolding; state flows correctly through all nodes; retry logic terminates; HITL resume works |
| Conceptual understanding | 25% | Conceptual answers are precise; correct terminology (reducer, checkpointer, interrupt, conditional edge); no confused analogies |
| Debugging & problem-solving | 20% | Evidence of systematic diagnosis (e.g., logging at each node, `get_state()` inspection); retry logic handles edge cases; mocks are correct |
| Code quality & testing | 15% | Typed state; Pydantic v2 models; `logging`; `ruff`-clean; both tests pass; mocks isolate LLM calls |

> Answer key: ../solutions/week-10/assessment.md

---

## Supplementary Links

- [LangGraph state reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)
- [Pydantic v2 `BaseModel`](https://docs.pydantic.dev/latest/concepts/models/)
- [LangGraph HITL guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [pytest mock with `unittest.mock.patch`](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
- [LangChain `FakeListChatModel`](https://python.langchain.com/docs/integrations/chat/fake/)
