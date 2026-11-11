# Phase 4 · Week 06 · Day 3 — Multi-hop & Iterative Retrieval

> Objective: Answer questions that require chaining multiple retrievals by implementing a configurable multi-hop retrieval loop with LLM-based sufficiency checking.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-2]] (HyDE + contextual retrieval)

## Concept

Single-pass retrieval assumes the answer exists in one chunk. Many real questions do not:

> "What version of Python is required by the package that SAP's LangGraph agent framework depends on for its type checking?"

Answering this requires:
1. Retrieve: find which package SAP's framework depends on for type checking → "Pydantic"
2. Retrieve: find Pydantic's Python version requirements → "Python ≥ 3.8"
3. Synthesise: combine both pieces → answer

This is multi-hop retrieval. Each hop uses the partial answer from the previous hop to construct a more specific query for the next hop.

**When single-pass suffices vs. when multi-hop is needed**

| Pattern | Use single-pass | Use multi-hop |
|---|---|---|
| Self-contained factual queries | "What is RAG?" | — |
| Entity + relationship chains | — | "Who founded the company that made product X?" |
| Aggregation over retrieved info | — | "What do all documents say about X? Now compare." |
| Hierarchical concept lookups | — | "What standard does feature Y implement? What does that standard require?" |
| Simple keyword lookup | "What is BM25?" | — |

**ReAct-style retrieval reasoning**

Multi-hop works best when each retrieval step is preceded by explicit reasoning:

```
Thought: I need to find which package handles type checking in the framework.
Action: retrieve("LangGraph SAP type checking dependency")
Observation: [chunk] "...Pydantic is used for all state type validation..."
Thought: I now know the package is Pydantic. I need its Python version requirement.
Action: retrieve("Pydantic Python version requirements minimum")
Observation: [chunk] "...Pydantic v2 requires Python 3.8 or higher..."
Thought: I have both facts. Final answer: Python ≥ 3.8.
```

This Thought/Action/Observation loop is the ReAct pattern applied to retrieval.

## How it works internally

**The iterative loop**

```
query
  │
  ▼
┌──────────────────────────────────────────┐
│  hop = 0                                 │
│  context = []                            │
│  current_query = original_query          │
│                                          │
│  while hop < max_hops:                   │
│    chunks = retrieve(current_query)      │
│    context.extend(chunks)                │
│    sufficient, next_query = check(       │
│      original_query, context             │
│    )                                     │
│    if sufficient: break                  │
│    current_query = next_query            │
│    hop += 1                              │
└──────────────────────────────────────────┘
  │
  ▼
synthesise(original_query, context)
```

**Sufficiency checking** is the critical inner step. The sufficiency checker (an LLM call) receives:
- The original question
- All chunks retrieved so far
- A prompt that asks: "Do the provided context chunks contain enough information to fully answer the question? If yes, say SUFFICIENT. If no, provide a refined query to retrieve the missing piece."

This produces two outputs: a boolean and a next query (if not sufficient). Structured output (Pydantic model) makes parsing reliable.

**Stopping criteria** — three guards, first to trigger wins:
1. `max_hops` exceeded — hard budget cap
2. LLM says SUFFICIENT — semantic sufficiency detected
3. Confidence threshold — optional: if the sufficiency checker returns a confidence score, stop when it exceeds a threshold

**Context accumulation** grows with each hop. For very long chains, trim context to the most recent N chunks or use a sliding window — the sufficiency check prompt must stay within the model's context limit.

**Query refinement quality** is the key variable. A weak refinement query loops without progress ("still asking the same thing"). A good sufficiency checker prompt explicitly asks for *what specific fact is missing* and *what query would retrieve it*.

## Worked example

```python
"""
day3_multihop_retrieval.py

Multi-hop iterative retrieval with ReAct-style reasoning loop.
Uses Anthropic SDK for sufficiency checking and query refinement.
LLM calls must be mocked in tests — see exercises.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum

import anthropic
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"  # or claude-opus-5-5 for better reasoning chains


def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class SufficiencyVerdict(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class SufficiencyCheck(BaseModel):
    verdict: SufficiencyVerdict
    reasoning: str = Field(description="One sentence explaining the verdict.")
    next_query: str | None = Field(
        default=None,
        description="If INSUFFICIENT: the refined query to retrieve the missing fact. "
        "If SUFFICIENT: null.",
    )


class HopRecord(BaseModel):
    hop_number: int
    query_used: str
    chunks_retrieved: int
    sufficiency: SufficiencyCheck


class MultiHopResult(BaseModel):
    original_question: str
    hops: list[HopRecord]
    final_context_chunks: list[str]  # text of all accumulated chunks
    stopped_by: str  # "sufficient" | "max_hops"
    final_answer: str | None = None


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# Retriever from week 5
# ---------------------------------------------------------------------------
from week_05.retriever import VectorRetriever  # type: ignore[import]


# ---------------------------------------------------------------------------
# SufficiencyChecker — LLM-based, structured output via JSON mode
# ---------------------------------------------------------------------------
class SufficiencyChecker:
    """
    Given the original question and accumulated context, determines:
    - Whether the context is sufficient to answer the question
    - If not, what refined query would retrieve the missing fact
    """

    SYSTEM_PROMPT = """You are a retrieval sufficiency evaluator.

Given an original question and the context chunks retrieved so far, determine:
1. Whether the chunks contain SUFFICIENT information to fully answer the question.
2. If INSUFFICIENT, what SINGLE specific query would retrieve the missing piece.

Respond with valid JSON matching this schema:
{
  "verdict": "SUFFICIENT" or "INSUFFICIENT",
  "reasoning": "<one sentence>",
  "next_query": "<refined retrieval query>" or null
}

Rules:
- Be strict: SUFFICIENT only if the answer can be stated with confidence from the context.
- next_query must be maximally specific — include entity names, version numbers, etc. found in the context so far.
- next_query must NOT be a restatement of the original question.
"""

    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._client = _client()

    def check(
        self,
        original_question: str,
        context_chunks: list[Chunk],
    ) -> SufficiencyCheck:
        context_text = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{c.text}" for i, c in enumerate(context_chunks)
        )
        user_message = (
            f"ORIGINAL QUESTION:\n{original_question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}"
        )
        log.info(
            "Sufficiency check: %d chunks accumulated", len(context_chunks)
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        log.debug("Sufficiency raw response: %s", raw)
        # Parse JSON — Pydantic validates structure
        import json
        parsed = json.loads(raw)
        result = SufficiencyCheck(**parsed)
        log.info(
            "Verdict: %s | Reasoning: %s", result.verdict, result.reasoning
        )
        return result


# ---------------------------------------------------------------------------
# MultiHopRetriever — main loop
# ---------------------------------------------------------------------------
class MultiHopRetriever:
    """
    Iterative retrieval loop: retrieve → check sufficiency → refine query → repeat.

    Parameters
    ----------
    retriever : VectorRetriever
        The underlying vector store retriever.
    sufficiency_checker : SufficiencyChecker
        LLM-based sufficiency evaluator.
    max_hops : int
        Hard cap on retrieval iterations. Default 4.
    top_k_per_hop : int
        Chunks to retrieve per hop. Default 5.
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        sufficiency_checker: SufficiencyChecker,
        max_hops: int = 4,
        top_k_per_hop: int = 5,
    ) -> None:
        self._retriever = retriever
        self._checker = sufficiency_checker
        self._max_hops = max_hops
        self._top_k = top_k_per_hop

    def retrieve(self, question: str) -> MultiHopResult:
        """
        Execute the multi-hop retrieval loop.
        Returns a MultiHopResult with all hop records and accumulated context.
        """
        current_query = question
        accumulated_chunks: list[Chunk] = []
        hop_records: list[HopRecord] = []
        stopped_by = "max_hops"

        for hop in range(self._max_hops):
            log.info("=== Hop %d | Query: %s ===", hop + 1, current_query)

            # Retrieve
            new_chunks = self._retriever.retrieve(current_query, top_k=self._top_k)
            # Deduplicate by ID before adding
            existing_ids = {c.id for c in accumulated_chunks}
            unique_new = [c for c in new_chunks if c.id not in existing_ids]
            accumulated_chunks.extend(unique_new)
            log.info(
                "Hop %d: retrieved %d new chunks (%d total accumulated)",
                hop + 1,
                len(unique_new),
                len(accumulated_chunks),
            )

            # Sufficiency check
            check = self._checker.check(question, accumulated_chunks)
            hop_records.append(
                HopRecord(
                    hop_number=hop + 1,
                    query_used=current_query,
                    chunks_retrieved=len(unique_new),
                    sufficiency=check,
                )
            )

            if check.verdict == SufficiencyVerdict.SUFFICIENT:
                log.info("Sufficient after %d hop(s). Stopping.", hop + 1)
                stopped_by = "sufficient"
                break

            if check.next_query is None:
                log.warning("Checker returned INSUFFICIENT but no next_query — stopping.")
                break

            current_query = check.next_query

        return MultiHopResult(
            original_question=question,
            hops=hop_records,
            final_context_chunks=[c.text for c in accumulated_chunks],
            stopped_by=stopped_by,
        )

    def retrieve_and_answer(
        self,
        question: str,
        answer_model: str = MODEL,
    ) -> MultiHopResult:
        """
        Full pipeline: multi-hop retrieval + final synthesis.
        Adds a final LLM call to produce a grounded answer from accumulated context.
        """
        result = self.retrieve(question)
        context_text = "\n\n---\n\n".join(
            f"[{i + 1}] {t}" for i, t in enumerate(result.final_context_chunks)
        )
        client = _client()
        synthesis_response = client.messages.create(
            model=answer_model,
            max_tokens=1024,
            system=(
                "You are a precise question-answering assistant. "
                "Answer the question using ONLY the provided context. "
                "If the context does not contain the answer, say so explicitly."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"CONTEXT:\n{context_text}"
                    ),
                }
            ],
        )
        result.final_answer = synthesis_response.content[0].text.strip()
        log.info("Final answer: %s", result.final_answer[:120])
        return result


# ---------------------------------------------------------------------------
# ReAct-style retriever (logging-focused variant for teaching)
# ---------------------------------------------------------------------------
class ReActRetriever:
    """
    Explicit Thought/Action/Observation loop for pedagogical clarity.
    Logs each step at INFO level so the loop structure is visible.
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        model: str = MODEL,
        max_hops: int = 3,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._max_hops = max_hops
        self._client = _client()

    def _think_and_act(
        self,
        original_question: str,
        history: list[dict[str, str]],
    ) -> dict[str, str]:
        """
        Returns {"thought": ..., "action": ..., "query": ...} or {"done": ..., "answer": ...}
        """
        system = (
            "You are a retrieval reasoning agent. "
            "You have access to one action: retrieve(query). "
            "At each step, write:\n"
            "Thought: <your reasoning>\n"
            "Action: retrieve(<query>) OR Action: done(<final answer>)\n\n"
            "Use retrieve() to fetch more information. "
            "Use done() only when you have enough context to answer confidently."
        )
        messages = [{"role": "user", "content": f"Question: {original_question}"}]
        for h in history:
            messages.append({"role": "assistant", "content": h["assistant"]})
            messages.append({"role": "user", "content": h["observation"]})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        log.info("Thought/Action:\n%s", raw)
        return {"raw": raw}

    def retrieve(self, question: str) -> MultiHopResult:
        history: list[dict[str, str]] = []
        accumulated: list[Chunk] = []
        hop_records: list[HopRecord] = []
        stopped_by = "max_hops"

        for hop in range(self._max_hops):
            step = self._think_and_act(question, history)
            raw = step["raw"]

            # Parse Action line
            if "done(" in raw.lower():
                stopped_by = "sufficient"
                log.info("ReAct: done() called at hop %d", hop + 1)
                break

            # Extract query from "Action: retrieve(<query>)"
            import re
            match = re.search(r'retrieve\(([^)]+)\)', raw, re.IGNORECASE)
            if not match:
                log.warning("Could not parse retrieve() call — stopping.")
                break
            query = match.group(1).strip().strip('"\'')

            chunks = self._retriever.retrieve(query, top_k=5)
            existing_ids = {c.id for c in accumulated}
            new_chunks = [c for c in chunks if c.id not in existing_ids]
            accumulated.extend(new_chunks)

            observation = (
                "Observation: Retrieved {} chunks:\n".format(len(new_chunks))
                + "\n".join(f"  - {c.text[:100]}" for c in new_chunks[:3])
            )
            log.info(observation)

            history.append({"assistant": raw, "observation": observation})
            # Create a placeholder sufficiency check for the hop record
            hop_records.append(
                HopRecord(
                    hop_number=hop + 1,
                    query_used=query,
                    chunks_retrieved=len(new_chunks),
                    sufficiency=SufficiencyCheck(
                        verdict=SufficiencyVerdict.INSUFFICIENT,
                        reasoning="ReAct loop continuing",
                        next_query=None,
                    ),
                )
            )

        return MultiHopResult(
            original_question=question,
            hops=hop_records,
            final_context_chunks=[c.text for c in accumulated],
            stopped_by=stopped_by,
        )


# ---------------------------------------------------------------------------
# Demo (requires ANTHROPIC_API_KEY + a wired VectorRetriever)
# ---------------------------------------------------------------------------
def demo_sufficiency_checker() -> None:
    """
    Standalone demo: show the sufficiency checker on synthetic chunks.
    Does not require a real VectorRetriever.
    """
    log.info("=== SufficiencyChecker standalone demo ===")
    checker = SufficiencyChecker()

    question = "What Python version does Pydantic v2 require, and what SAP framework uses it?"

    # Simulate hop 1: only partial info retrieved
    hop1_chunks = [
        Chunk(
            id="chunk-001",
            text="SAP's LangGraph agent framework uses Pydantic v2 for all state type validation.",
        )
    ]
    check1 = checker.check(question, hop1_chunks)
    log.info("After hop 1 — verdict: %s", check1.verdict)
    log.info("Next query: %s", check1.next_query)

    # Simulate hop 2: full info retrieved
    hop2_chunks = hop1_chunks + [
        Chunk(
            id="chunk-002",
            text="Pydantic v2 requires Python 3.8 or higher. It is not compatible with Python 3.7.",
        )
    ]
    check2 = checker.check(question, hop2_chunks)
    log.info("After hop 2 — verdict: %s", check2.verdict)
    log.info("Reasoning: %s", check2.reasoning)


if __name__ == "__main__":
    demo_sufficiency_checker()
```

**Key observations:**

- `MultiHopRetriever.retrieve` accumulates chunks across hops and deduplicates by ID so the same chunk is not counted twice toward the context budget.
- `SufficiencyChecker` uses strict JSON output — Pydantic validates the schema. If the LLM returns malformed JSON, `json.loads` raises `JSONDecodeError` — add retry logic in production.
- `ReActRetriever` separates the reasoning trace (Thought/Action) from the sufficiency mechanics — better for debugging chain-of-retrieval problems.
- `stopped_by` field in `MultiHopResult` distinguishes soft stop ("sufficient") from hard stop ("max_hops") — useful for monitoring and eval.

## Exercises

1. **Unit test the sufficiency checker.** Write a pytest test for `SufficiencyChecker.check`. Mock `_client()` to return a mock Anthropic client whose `.messages.create` returns a mock response with `content[0].text` set to a known JSON string. Test three cases: (a) SUFFICIENT with null next_query, (b) INSUFFICIENT with a next_query, (c) malformed JSON that should raise an exception. Use `unittest.mock.patch`.

2. **Max-hops safeguard test.** Write a pytest test for `MultiHopRetriever.retrieve` where the mock sufficiency checker always returns INSUFFICIENT. Assert that: the loop runs exactly `max_hops` times, `result.stopped_by == "max_hops"`, and `len(result.hops) == max_hops`. Mock both the retriever and the sufficiency checker.

3. **Hop logging analysis.** Add a method `MultiHopRetriever.explain(result: MultiHopResult) -> str` that returns a human-readable string showing each hop's query, number of new chunks, and verdict. Write a unit test that calls `explain` on a pre-built `MultiHopResult` and asserts the output contains each hop number and verdict string.

4. **Confidence threshold stopping.** Extend `SufficiencyCheck` with an optional `confidence: float` field (0.0–1.0). Extend `SufficiencyChecker`'s JSON schema and prompt to include it. Add a `confidence_threshold: float = 0.9` parameter to `MultiHopRetriever` and stop the loop when confidence exceeds it (even if verdict is INSUFFICIENT). Write a unit test for this new stopping criterion.

5. **Compare single-hop vs. multi-hop.** Build a mini evaluation harness: define 3 multi-hop questions and a toy corpus where answers genuinely require chaining. For each question, run both a single-pass retrieve (top-10) and the `MultiHopRetriever`. Compare: (a) whether the final context contains both required facts, (b) number of LLM calls, (c) total retrieved chunks. Log a summary. This is the foundation for multi-hop RAG evaluation.

(Solutions: ../solutions/week-06/day-3.md)

## Self-check

- [ ] I can describe when single-pass retrieval fails and multi-hop is required.
- [ ] I can sketch the multi-hop loop (retrieve → check → refine → repeat) from memory.
- [ ] I can explain the three stopping criteria and which takes priority.
- [ ] I can implement `SufficiencyChecker.check` using the Anthropic SDK with JSON output.
- [ ] I can write a pytest test that mocks both retriever and sufficiency checker, asserting the loop runs exactly max_hops times.

## Supplementary links

- [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [IRCoT: Interleaving Retrieval with Chain-of-Thought (Trivedi et al., 2022)](https://arxiv.org/abs/2212.10509)
- [FLARE: Forward-Looking Active Retrieval (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)
- [Anthropic structured output / tool use for reliable JSON](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
