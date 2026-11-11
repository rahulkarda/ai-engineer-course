# Week 06 Assessment — Advanced RAG & Agentic RAG

## Conceptual (interview-style)

**1. Explain query rewriting: why does it help, when does it hurt?**

Query rewriting uses an LLM to rephrase the user's query before retrieval, aligning vocabulary to the corpus rather than to conversational language. It helps when there is vocabulary mismatch: a user asks "how to stop my app crashing on startup" but docs say "handle application initialisation exceptions". The rewriter bridges this gap by producing terminology more likely to match indexed text. It hurts when: (a) the corpus is short and directly written in user-facing language (rewriting adds latency with no recall gain); (b) the original query is already precise and technical (the rewriter may paraphrase away exact terms, reducing precision); (c) the LLM rewrites toward a different domain than the corpus (domain mismatch amplified). Mitigation: A/B test rewriting against raw queries on your specific corpus before enabling it in production.

**2. What is HyDE? Why does embedding a hypothetical answer outperform embedding the query for some query types?**

HyDE (Hypothetical Document Embeddings) generates a short hypothetical answer to the query using an LLM, then embeds that answer text instead of the original question. The geometric argument: in most embedding spaces, questions and answers occupy different regions. A question embedding `q_vec` lies in "question space"; document chunks lie in "answer space". Cosine similarity between a question and its correct answer is typically lower than similarity between two answers on the same topic. By embedding a generated answer, HyDE places the search vector in the same region as real answer chunks, improving ANN recall. It outperforms raw query embedding especially for factual Q&A corpora, technical documentation, and FAQ-style content where the LLM can produce a plausible answer even without seeing the corpus. It underperforms when the LLM hallucinates a wildly incorrect answer (the embedding points toward irrelevant chunks) — hence the `retrieve_with_fallback` pattern: if HyDE returns fewer results than expected, merge with a raw query retrieval.

**3. Explain Anthropic's contextual retrieval approach. What problem does it solve that standard chunking misses?**

Standard chunking splits a document into fixed-size or sentence-boundary segments and embeds each chunk in isolation. This fails when a chunk's meaning depends on surrounding context: "He was fired the following week" is ambiguous without knowing the document and section it came from. The chunk embedding carries no signal about the subject or event. Contextual retrieval solves this at index time: for each chunk, an LLM call receives the full document (up to the context window) plus the chunk text and generates a 1-2 sentence situating prefix — "In Chapter 3 of the Enron Files, discussing the post-scandal HR actions: He was fired the following week." The full `prefix + chunk` text is embedded. The prefix adds retrieval signal that the raw chunk lacks, allowing queries referencing context ("what happened to employees after the accounting scandal") to match this chunk. The cost is one LLM call per chunk at index time — a one-time cost that improves every subsequent query. Anthropic's 2024 benchmarks show a 49% reduction in retrieval failure rate versus standard chunking.

**4. When does multi-hop retrieval help? Give a concrete example requiring at least 2 hops.**

Multi-hop retrieval helps when the answer requires chaining facts across multiple documents or sections — no single chunk contains the full answer. Example: "What Python version is required by the type-validation library used in SAP's LangGraph agent framework?" Hop 1: retrieve on "SAP LangGraph agent framework type validation dependency" → finds "SAP's LangGraph agent uses Pydantic v2 for all state type validation." Hop 2: use the extracted entity "Pydantic v2" as the refined query → finds "Pydantic v2 requires Python 3.8 or higher." Only after both hops can the final answer be synthesised. Single-pass retrieval on the original query typically fails because no chunk says "Pydantic v2 requires Python 3.8 and is used by SAP's LangGraph framework" — the two facts live in different chunks. Multi-hop works by using partial answers to construct increasingly specific queries.

**5. Describe Self-RAG's four decision tokens. What does each gate decide?**

Self-RAG trains a single LLM to emit special reflection tokens at inference time, interleaved with generation:

- **IsRAGRequired** (also called Retrieve): emitted before generation begins. Decides whether to retrieve at all for this query. Avoids unnecessary retrieval for queries the model can answer from parametric knowledge alone (e.g., "What is 2+2?").
- **IsChunkRelevant** (also called ISREL): emitted after each retrieved chunk is examined. Decides whether this specific chunk is relevant to the query. Irrelevant chunks are discarded before generation.
- **IsAnswerGrounded** (also called ISSUP): emitted after generating a response segment. Decides whether the generated claim is fully supported by the provided chunks, partially supported, or unsupported. Triggers correction or retrieval of additional context if unsupported.
- **IsAnswerUseful** (also called ISUSE): emitted after the full response. Rates the overall utility of the answer on a scale. Used to rank multiple candidate responses when beam search or sampling produces several candidates.

Together these four tokens allow the model to self-critique its retrieval and generation at every stage without requiring a separate evaluator.

**6. What is Corrective RAG? How does the three-branch routing (correct/incorrect/ambiguous) differ from Self-RAG?**

Corrective RAG (CRAG) introduces a lightweight evaluator that scores retrieved chunks before generation and routes the pipeline into one of three branches: (1) **Correct** — the retrieved chunks are relevant and sufficient; proceed to generation using them. (2) **Incorrect** — the retrieved chunks are irrelevant or low-confidence; discard them and issue a web search query instead, using the web results for generation. (3) **Ambiguous** — the chunks have mixed relevance; blend both the original retrieved chunks and web-retrieved information. The key difference from Self-RAG: CRAG is a pipeline modification applied to an existing generator, not a specially trained model. Self-RAG requires fine-tuning the base LLM to emit reflection tokens; CRAG adds a separate evaluator model and a conditional routing step. CRAG is therefore easier to apply to existing deployments. However, Self-RAG's inline token mechanism is more granular — it gates at the individual claim level, whereas CRAG's routing operates at the whole-response level.

**7. What makes RAG "agentic"? How does retrieval-as-a-tool differ from retrieval-in-pipeline?**

In a standard RAG pipeline, retrieval is a fixed step: every query triggers retrieval, and the retrieved chunks are always passed to the generator. The pipeline is a static sequence: query → retrieve → generate. Agentic RAG treats retrieval as an optional, LLM-dispatched tool. The LLM decides: (a) whether to retrieve at all; (b) what query to use for retrieval; (c) which retrieval source to use (internal knowledge base vs. web); (d) whether to retrieve again after seeing initial results. The LLM drives the loop rather than executing a fixed sequence. With the Anthropic `tool_use` API, the agent emits `tool_use` content blocks naming the tool and its arguments; the application executes the tool and returns `tool_result` blocks; the agent continues reasoning. This enables multi-source routing, adaptive iteration depth, and query reformulation mid-loop — none of which are possible in a fixed pipeline. The cost is higher latency and more LLM tokens per query.

**8. What are faithfulness and relevance in citation grounding? Why must they be measured separately?**

In citation grounding: **Faithfulness** measures whether each claim in the generated answer is actually supported by the cited source chunks — i.e., can the claim be inferred from the cited text? A faithfulness failure is hallucination: the answer asserts something not present in (or contradicted by) the cited chunks, even if citations are present. **Relevance** measures whether the cited chunks are topically related to the question — i.e., do the sources address the question that was asked? A relevance failure is citation stuffing: chunks are cited that contain vaguely related topic keywords but do not actually support the specific claim. These must be measured separately because a model can score high on one and low on the other. A model might cite highly relevant chunks but paraphrase them inaccurately (high relevance, low faithfulness) — e.g., a source says "Pydantic v2 requires Python ≥ 3.8" but the answer says "Pydantic v2 requires Python ≥ 3.10". Conversely, a model might faithfully reproduce text from irrelevant chunks (low relevance, high faithfulness) — e.g., correctly quoting a chunk about database indexing when the question was about query optimisation. A complete grounding evaluation needs both metrics.

---

## Coding challenge

**Spec:** Build a minimal agentic RAG system that:
- Accepts a user query string
- Decides whether retrieval is needed (mock LLM decision: always returns True in tests)
- If retrieval needed: rewrites query using LLM, retrieves top-3 chunks from a mock store
- Filters chunks by relevance score ≥ 0.7 (mock scorer)
- Generates an answer with inline citations [1], [2], [3]
- Returns a `CitedAnswer(text: str, sources: list[str])` Pydantic model

**Acceptance criteria:**

1. `CitedAnswer` is a Pydantic model with `text: str` and `sources: list[str]`
2. `AgenticRAGSystem.run(query)` returns a `CitedAnswer`
3. Relevant chunks (score ≥ 0.7) are included; irrelevant ones (score < 0.7) are filtered out
4. At least one inline citation `[N]` appears in the answer text
5. All LLM calls are mocked in tests using `unittest.mock.patch`
6. Tests pass with `pytest` without any real API calls
7. No hardcoded API keys; `os.getenv("ANTHROPIC_API_KEY")` used throughout
8. Type hints on all functions and classes

**Starter skeleton:**

```python
import os
from pydantic import BaseModel


class CitedAnswer(BaseModel):
    text: str
    sources: list[str]


class AgenticRAGSystem:
    def run(self, query: str) -> CitedAnswer:
        ...
```

Implement the full system. Tests must mock `anthropic.Anthropic` and assert `CitedAnswer` fields.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-06/assessment.md
