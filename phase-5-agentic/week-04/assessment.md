# Week 04 Assessment — Memory & State Management

## Conceptual (interview-style)

1. **Typed state benefits.** You are designing a LangGraph agent. A colleague suggests storing agent state as a plain `dict[str, Any]`. You propose using a Pydantic model instead. What concrete problems does typed state solve that a plain dict does not? Name at least three.

2. **Short-term vs long-term memory tradeoffs.** Your agent has a buffer (last 10 messages) and a vector store (1000 embedded facts). A user asks a question that requires both very recent context and a fact from three weeks ago. Describe the retrieval strategy. What goes wrong if you only have STM? What goes wrong if you only have LTM?

3. **Summarize vs truncate.** Your buffer is full. You have two options: truncate (drop the oldest N messages) or summarize (compress them into a shorter representation). Under what conditions do you choose each? Give a concrete scenario where truncation is acceptable and one where it causes a failure.

4. **Episodic vs semantic memory.** Distinguish episodic from semantic memory with one concrete example each from an agent that assists software engineers. How does retrieval differ between the two? Why can one not replace the other?

5. **Vector similarity for memory retrieval.** Explain why cosine similarity is used for semantic memory retrieval rather than exact string matching. What is the key property of embedding space that makes similarity search useful? What is a failure mode of this approach (i.e., when does it return the wrong facts)?

---

## Coding Challenge

### MinimalMemoryAgent

Build a `MinimalMemoryAgent` that demonstrates all three core memory mechanisms from this week working together. No real LLM is needed — use a mock.

**Specification**

```python
# Implement in minimal_memory_agent.py

class AgentState(BaseModel):
    """Typed Pydantic state — all fields required."""
    session_id: str
    turn_count: int
    buffer: list[str]          # raw recent messages, max 5
    summary: Optional[str]     # compressed history; None until first summarization
    semantic_facts: list[str]  # 3 pre-seeded facts (see below)

class MinimalMemoryAgent:
    def __init__(self, session_id: str) -> None: ...

    def chat(self, user_message: str) -> str:
        """
        Process one user message.
        1. Append user message to buffer. If buffer exceeds 5, trim to 5 (drop oldest).
        2. If buffer length was > 3 BEFORE this message, trigger summarization:
             - Join all buffer entries and store as self.state.summary
             - Use mock_summarize(texts: list[str]) -> str (provided below)
        3. Query semantic_facts for the most relevant fact (cosine similarity
           with fake embedder from week examples).
        4. Return a mock response: f"[turn {self.state.turn_count}] {retrieved_fact}"
        """
        ...

    def get_state(self) -> AgentState: ...
```

**Pre-seeded semantic facts** (hardcode these in `__init__`):
```python
SEED_FACTS = [
    "Python lists are mutable and ordered.",
    "Pydantic v2 validates data at model instantiation.",
    "LangGraph nodes are pure functions.",
]
```

**Provided helpers** (implement these; tests will call them directly):
```python
def mock_summarize(texts: list[str]) -> str:
    return f"[summary of {len(texts)} messages]"

def mock_embed(text: str) -> list[float]:
    # deterministic hash-based unit vector (same pattern as week examples)
    ...
```

**Acceptance criteria — must pass all three tests:**

```python
# tests/test_minimal_memory_agent.py

def test_buffer_trim():
    """Buffer never exceeds 5 messages."""
    agent = MinimalMemoryAgent("test-session")
    for i in range(7):
        agent.chat(f"message {i}")
    assert len(agent.get_state().buffer) <= 5

def test_summary_triggers():
    """Summary is populated after the buffer exceeds 3."""
    agent = MinimalMemoryAgent("test-session")
    for i in range(4):
        agent.chat(f"message {i}")
    assert agent.get_state().summary is not None
    assert "summary" in agent.get_state().summary.lower()

def test_semantic_retrieval():
    """chat() returns a response containing text from a relevant seeded fact."""
    agent = MinimalMemoryAgent("test-session")
    response = agent.chat("Tell me about Python lists")
    assert "Python" in response or "mutable" in response or "ordered" in response
```

**Additional constraints:**
- Use Python 3.12+ type hints throughout.
- `AgentState` must be a Pydantic `BaseModel`.
- No real LLM calls. No `os.getenv` in tests.
- `logging` not `print` for any internal diagnostics.
- All three tests must pass with `pytest minimal_memory_agent.py` or `pytest tests/test_minimal_memory_agent.py`.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation (typed state, buffer trim, summary trigger, semantic retrieval all working without external help) | 40% |
| Conceptual understanding (5 conceptual questions answered with precision and concrete examples) | 25% |
| Debugging & problem-solving (tests fail initially and are fixed systematically; evidence of root-cause analysis) | 20% |
| Code quality & testing (type hints throughout, Pydantic models, logging not print, all 3 tests pass, no hardcoded secrets) | 15% |

Answer key: ../solutions/week-04/assessment.md
