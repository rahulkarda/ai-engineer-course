# Phase 4 · Week 06 · Day 6 — Citations, Grounding & Graph RAG Intro

> Objective: Extract inline citations from LLM answers, verify factual grounding per sentence, and understand the Graph RAG architecture for corpus-wide questions.  ·  Time: ~4 hr  ·  Prereqs: [[week-06/day-5]]

---

*Saturday deep-practice format. Exercises are more involved than weekday sessions — expect each to take 30–60 minutes. Work through them sequentially; each builds on the last.*

---

## Concept

### Why citations and grounding matter

RAG pipelines retrieve supporting context, but the LLM can still:
- Hallucinate facts not present in any retrieved chunk.
- Blend information from multiple chunks in ways that misattribute claims.
- Generate plausible-sounding statements that contradict the retrieved sources.

**Citations** let users verify answers. **Grounding verification** lets the system automatically detect unfaithful claims before they reach the user.

These are two separate concerns:
- **Faithfulness**: does each sentence in the answer follow from the retrieved chunks?
- **Relevance**: do the retrieved chunks address the query at all?

A RAG system can score high on relevance (right topic retrieved) but low on faithfulness (model ignored the context). You need both metrics.

### Citation extraction

Two approaches:

**Post-hoc**: generate the answer first, then ask the LLM to map each sentence to a source chunk. Simple to implement; works with any generation prompt. Risk: the mapping LLM may hallucinate attributions.

**Inline**: instruct the model to emit citation markers as it generates. E.g., `"Python 3.12 introduced PEP 695 [1]."` More reliable; requires prompt engineering.

Inline citation format used in this lesson:
```
Answer text [N]. More text [N, M].

Sources:
[1] chunk_id_a — first 80 chars of text…
[2] chunk_id_b — first 80 chars of text…
```

### Grounding verification

For each sentence in the answer, ask the LLM a binary question:

> "Does this claim appear in, or follow directly from, any of the following passages?"

Return `{grounded: bool, supporting_chunk_ids: list[str]}`. Sentences with `grounded=False` are potential hallucinations.

This is a lightweight but effective heuristic. More rigorous alternatives (NLI models, RAGAS, TruLens) exist but require additional dependencies.

### Faithfulness vs. relevance scores

```
faithfulness  = grounded_sentences / total_sentences
relevance     = (relevant chunks retrieved) / total_chunks_retrieved
```

A well-tuned pipeline targets faithfulness ≥ 0.9 and relevance ≥ 0.7. Track both separately — they degrade for different reasons.

---

## Graph RAG intro

### The problem with vector-only RAG

Vector similarity retrieval works well for **specific, local questions** ("What does function X do?"). It struggles with **global questions** that require synthesising information across many documents:

- "What are the main themes in this corpus?"
- "How do the three competing approaches relate to each other?"
- "Summarise the evolution of this topic over all documents."

A single embedding query cannot capture cross-document relationships. The model would need to read all chunks simultaneously — infeasible at scale.

### Graph RAG architecture (Microsoft, 2024)

Graph RAG (Edge et al., 2024) represents the document corpus as a **knowledge graph** and uses community summaries for global questions.

**Five-stage pipeline:**

```
1. Entity extraction
   LLM reads each chunk → extracts (entity, entity_type) pairs
   e.g., ("Python", "language"), ("PEP 695", "proposal")

2. Relation extraction
   LLM identifies edges: (source_entity, relation, target_entity)
   e.g., ("PEP 695", "introduced_in", "Python 3.12")

3. Graph construction
   Nodes = entities  |  Edges = relations + co-occurrence weights
   Stored in a graph DB or in-memory (networkx for small corpora)

4. Community detection
   Leiden algorithm partitions the graph into clusters of related entities
   Each cluster = a "community" (e.g., all Python typing-related entities)

5. Hierarchical summarisation
   LLM summarises each community at multiple levels (leaf → root)
   Community summaries are stored for retrieval
```

**Query-time routing:**
- **Local query** (specific entity): vector search over entity descriptions.
- **Global query** (themes, comparisons): retrieve relevant community summaries, then synthesise.

### Why community summaries?

A community summary captures what a cluster of entities is about without requiring the model to read every document. Global queries are answered by reading a handful of summaries rather than thousands of chunks.

---

## How it works internally

### Entity extraction prompt pattern

```
System: Extract all named entities from the passage. Return JSON:
        {"entities": [{"name": str, "type": str, "description": str}],
         "relations": [{"source": str, "relation": str, "target": str}]}

User: <chunk text>
```

Run this for every chunk. Deduplicate entities by normalising names. Edge weights are co-occurrence counts across chunks.

### Community detection

The Leiden algorithm (a refinement of Louvain) partitions graph nodes into communities that maximise modularity. For small graphs (< 10k nodes), `python-igraph` or `networkx` + `community` package works. At scale, use a graph database.

### GraphRAG vs. naive vector RAG

| Capability | Vector RAG | Graph RAG |
|---|---|---|
| Specific fact lookup | Excellent | Good |
| Cross-document synthesis | Poor | Excellent |
| Entity relationships | Not captured | Explicit |
| Corpus-level themes | Not supported | Community summaries |
| Latency | Low | Higher (graph traversal) |
| Build cost | Low (embed once) | High (LLM extraction per chunk) |

Graph RAG is appropriate when the corpus is stable (built once, queried many times) and global questions are frequent.

---

## Worked example

```python
"""
day6_citations_grounding_graphrag.py

Three components:
1. CitationExtractor  — instructs LLM to produce inline citations
2. GroundingChecker   — verifies each sentence against retrieved chunks
3. GraphRetriever     — stub showing Graph RAG entity + community structure

Install:
    uv add anthropic pydantic networkx

Run:
    python day6_citations_grounding_graphrag.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────

class Chunk(BaseModel):
    id: str
    text: str
    source: str


class CitedAnswer(BaseModel):
    answer_text: str
    citation_map: dict[str, str]   # citation number → chunk_id
    footnotes: list[str]           # rendered source lines


class GroundingResult(BaseModel):
    sentence: str
    grounded: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class GroundingReport(BaseModel):
    total_sentences: int
    grounded_sentences: int
    faithfulness_score: float
    results: list[GroundingResult]


# ── Part 1: Citation extraction ───────────────────────────────────────────

class CitationExtractor:
    """
    Instructs the LLM to generate an answer with inline citation markers [N]
    and a Sources section.

    Parses the output into a CitedAnswer with a citation_map.
    """

    SYSTEM = """\
You are a precise research assistant. Answer the question using ONLY the provided context.

Rules:
- Add inline citation markers [1], [2], etc. immediately after each claim.
- Multiple citations for one sentence: [1, 2].
- Do NOT cite a source unless the claim directly comes from that chunk.
- After the answer, write a "Sources:" section listing each citation number,
  its chunk_id, and the first 80 characters of the chunk text.

Format:
<your answer with inline [N] markers>

Sources:
[1] chunk_id — first 80 chars…
[2] chunk_id — first 80 chars…
"""

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def _build_context(self, chunks: list[Chunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(f"[{i}] (id={c.id}) {c.text}")
        return "\n\n".join(lines)

    def _parse_response(self, raw: str, chunks: list[Chunk]) -> CitedAnswer:
        """Split on 'Sources:' and parse the footnote block."""
        parts = re.split(r"\nSources:\n", raw, maxsplit=1)
        answer_text = parts[0].strip()

        citation_map: dict[str, str] = {}
        footnotes: list[str] = []

        if len(parts) > 1:
            for line in parts[1].strip().splitlines():
                # Match: [N] chunk_id — text…
                m = re.match(r"\[(\d+)\]\s+(\S+)\s+—\s+(.*)", line)
                if m:
                    num, chunk_id, snippet = m.group(1), m.group(2), m.group(3)
                    citation_map[num] = chunk_id
                    footnotes.append(f"[{num}] {chunk_id} — {snippet}")

        return CitedAnswer(
            answer_text=answer_text,
            citation_map=citation_map,
            footnotes=footnotes,
        )

    def extract(self, query: str, chunks: list[Chunk]) -> CitedAnswer:
        context = self._build_context(chunks)
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text
        log.info("Raw cited answer:\n%s", raw)
        return self._parse_response(raw, chunks)


# ── Part 2: Grounding checker ─────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — splits on '. ', '! ', '? '."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class GroundingChecker:
    """
    Verifies each sentence in an answer against retrieved chunks.

    For each sentence, makes one LLM call:
      "Does this claim appear in or follow directly from any passage?"

    Returns a GroundingReport with faithfulness_score.
    """

    SYSTEM = """\
You are a fact-verification assistant. Given a claim and a set of passages,
determine if the claim is directly supported by at least one passage.

Respond with JSON only:
{
  "grounded": <bool>,
  "supporting_chunk_ids": [<list of chunk ids that support the claim>],
  "confidence": <0.0-1.0>
}

grounded=true  → the claim follows directly from the text of at least one passage.
grounded=false → the claim is not supported, contradicted, or goes beyond the passages.
"""

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def _check_sentence(self, sentence: str, chunks: list[Chunk]) -> GroundingResult:
        passages = "\n\n".join(
            f"[{c.id}] {c.text}" for c in chunks
        )
        user_msg = f"Claim: {sentence}\n\nPassages:\n{passages}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            return GroundingResult(
                sentence=sentence,
                grounded=bool(data["grounded"]),
                supporting_chunk_ids=data.get("supporting_chunk_ids", []),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Grounding parse failed: %s | raw=%r", exc, raw)
            return GroundingResult(sentence=sentence, grounded=False)

    def check(self, answer_text: str, chunks: list[Chunk]) -> GroundingReport:
        sentences = _split_sentences(answer_text)
        results: list[GroundingResult] = []

        for sent in sentences:
            result = self._check_sentence(sent, chunks)
            log.info(
                "  grounded=%s  conf=%.2f  | %s",
                result.grounded,
                result.confidence,
                sent[:80],
            )
            results.append(result)

        grounded_count = sum(1 for r in results if r.grounded)
        total = len(results)
        faithfulness = grounded_count / total if total > 0 else 0.0

        return GroundingReport(
            total_sentences=total,
            grounded_sentences=grounded_count,
            faithfulness_score=faithfulness,
            results=results,
        )


# ── Part 3: Graph RAG stub ────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    entity_type: str
    description: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    weight: float = 1.0


@dataclass
class Community:
    community_id: str
    entities: list[str]
    summary: str


class GraphRetriever:
    """
    Minimal Graph RAG stub demonstrating the data structures.

    In production this would use networkx + Leiden community detection.
    This stub hardcodes a small graph for demonstration and testing.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._communities: list[Community] = []

    def add_entity(self, entity: Entity) -> None:
        key = entity.name.lower()
        if key in self._entities:
            # Merge chunk references
            self._entities[key].chunk_ids.extend(entity.chunk_ids)
        else:
            self._entities[key] = entity

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)

    def add_community(self, community: Community) -> None:
        self._communities.append(community)

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        return len(self._relations)

    def local_query(self, entity_name: str) -> list[Entity]:
        """Return entities matching the name (substring)."""
        key = entity_name.lower()
        return [e for k, e in self._entities.items() if key in k]

    def global_query(self, topic_hint: str) -> list[Community]:
        """Return communities whose summaries mention the topic hint."""
        hint = topic_hint.lower()
        return [c for c in self._communities if hint in c.summary.lower()]

    def neighbours(self, entity_name: str) -> list[str]:
        """Return names of entities directly connected to entity_name."""
        key = entity_name.lower()
        result: list[str] = []
        for r in self._relations:
            if r.source.lower() == key:
                result.append(r.target)
            elif r.target.lower() == key:
                result.append(r.source)
        return list(set(result))


def build_demo_graph() -> GraphRetriever:
    """Populate a small demo graph for the Python typing domain."""
    g = GraphRetriever()

    # Entities
    g.add_entity(Entity("Python 3.12", "language_version", "Python release adding PEP 695", ["vs-1"]))
    g.add_entity(Entity("PEP 695", "proposal", "Type parameter syntax proposal", ["vs-1", "vs-2"]))
    g.add_entity(Entity("Generic", "class", "Base class for generic types pre-3.12", ["vs-2"]))
    g.add_entity(Entity("TypeVar", "construct", "Type variable used in generics", ["vs-3"]))
    g.add_entity(Entity("Pydantic", "library", "Data validation library using type hints", ["vs-4"]))

    # Relations
    g.add_relation(Relation("PEP 695", "introduced_in", "Python 3.12", 1.0))
    g.add_relation(Relation("PEP 695", "replaces", "Generic", 0.8))
    g.add_relation(Relation("PEP 695", "replaces", "TypeVar", 0.8))
    g.add_relation(Relation("Pydantic", "uses", "TypeVar", 0.6))

    # Communities (result of hypothetical Leiden run)
    g.add_community(Community(
        community_id="c0",
        entities=["Python 3.12", "PEP 695", "Generic", "TypeVar"],
        summary="Python typing system community: PEP 695 modernises generic syntax in Python 3.12, replacing TypeVar and Generic.",
    ))
    g.add_community(Community(
        community_id="c1",
        entities=["Pydantic", "TypeVar"],
        summary="Data validation community: Pydantic uses TypeVar-based generics for model type safety.",
    ))

    return g


# ── Pipeline orchestration ────────────────────────────────────────────────

def demo_citations_and_grounding(
    client: anthropic.Anthropic,
    model: str,
) -> None:
    log.info("=" * 60)
    log.info("Demo: Citations + Grounding")

    chunks = [
        Chunk(id="vs-1", text="Python 3.12 introduced PEP 695 type parameter syntax.", source="docs"),
        Chunk(id="vs-2", text="PEP 695 replaces Generic[T] with the new class Stack[T]: syntax.", source="docs"),
        Chunk(id="vs-3", text="TypeVar was the pre-3.12 mechanism for declaring type variables.", source="docs"),
    ]
    query = "What changed in Python 3.12 for generic types?"

    extractor = CitationExtractor(client, model)
    cited = extractor.extract(query, chunks)
    log.info("Answer: %s", cited.answer_text)
    log.info("Citation map: %s", cited.citation_map)
    for fn in cited.footnotes:
        log.info("  %s", fn)

    checker = GroundingChecker(client, model)
    report = checker.check(cited.answer_text, chunks)
    log.info(
        "Faithfulness: %.2f  (%d/%d grounded)",
        report.faithfulness_score,
        report.grounded_sentences,
        report.total_sentences,
    )


def demo_graph_rag() -> None:
    log.info("=" * 60)
    log.info("Demo: Graph RAG structure")

    g = build_demo_graph()
    log.info("Entities: %d  Relations: %d", g.entity_count(), g.relation_count())

    # Local query
    results = g.local_query("PEP 695")
    for e in results:
        log.info("  Entity: %s (%s) — %s", e.name, e.entity_type, e.description)
        log.info("    Neighbours: %s", g.neighbours(e.name))

    # Global query
    communities = g.global_query("typing")
    for c in communities:
        log.info("  Community %s: %s", c.community_id, c.summary[:100])


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = "claude-sonnet-5-5"

    demo_citations_and_grounding(client, model)
    demo_graph_rag()


if __name__ == "__main__":
    main()
```

---

## Exercises

*Saturday format: each exercise is substantial. Budget 30–60 minutes each.*

1. **Inline citation prompt tuning** *(~30 min)*: The `CitationExtractor` prompt says "Do NOT cite a source unless the claim directly comes from that chunk." Add a new field `uncited_sentences: list[str]` to `CitedAnswer` that collects answer sentences containing no citation marker. Implement `_find_uncited(answer_text: str) -> list[str]` using regex. Write a pytest unit test that parses `"Fact one [1]. Claim two. Fact three [2]."` and asserts `uncited_sentences == ["Claim two."]`.

2. **Batch grounding checker** *(~45 min)*: `GroundingChecker.check` makes one LLM call per sentence. Rewrite it to send all sentences in a single prompt and parse a JSON array response:
   ```json
   [{"sentence": "...", "grounded": true, "supporting_chunk_ids": ["vs-1"], "confidence": 0.9}, ...]
   ```
   Write a pytest test that mocks `client.messages.create` to return a valid JSON array for three sentences. Assert the returned `GroundingReport.faithfulness_score` matches the expected value.

3. **Faithfulness vs. relevance metrics** *(~30 min)*: Implement a `RAGMetrics` dataclass with `faithfulness: float` and `relevance: float`. Write a function:
   ```python
   def compute_metrics(
       query: str,
       chunks: list[Chunk],
       grounding_report: GroundingReport,
       relevance_scores: list[float],
   ) -> RAGMetrics:
   ```
   where `relevance = mean(scores for scores above 0.0)` and `faithfulness` comes from the grounding report. Write a pytest parametrize test covering: all chunks relevant + all grounded → both 1.0; no chunks relevant + half grounded → relevance 0.0, faithfulness 0.5.

4. **Graph RAG entity extraction** *(~60 min)*: Write an `EntityExtractor` class that takes a `Chunk` and a `client: anthropic.Anthropic` and returns `tuple[list[Entity], list[Relation]]`. Use the prompt pattern from the "How it works internally" section. Then write a function `build_graph_from_chunks(chunks: list[Chunk], extractor: EntityExtractor) -> GraphRetriever` that calls `extractor` for each chunk and populates a `GraphRetriever`. Mock the LLM in tests to return a fixed JSON with two entities and one relation, and assert the graph has correct entity and relation counts.

5. **Community summary generation** *(~45 min)*: Add a `Communitysummariser` class that takes a `Community` (its entity list) and the full entity dict, and asks the LLM to write a 2–3 sentence summary of what the community is about. Call it for each community in `build_demo_graph()`. Write a unit test that mocks the LLM call and asserts the returned summary string is non-empty and the `Community.summary` field is updated.

6. **End-to-end grounded pipeline** *(~60 min)*: Wire `CitationExtractor` + `GroundingChecker` + `RAGMetrics` into a single function:
   ```python
   def grounded_rag(
       query: str,
       chunks: list[Chunk],
       client: anthropic.Anthropic,
       model: str,
   ) -> tuple[CitedAnswer, GroundingReport, RAGMetrics]:
   ```
   Use mock relevance scores of `[0.9, 0.85, 0.7]` (one per chunk). Write an integration test (marked `@pytest.mark.integration`, skipped without `ANTHROPIC_API_KEY`) that runs the full pipeline on the demo chunks from the worked example and asserts `metrics.faithfulness >= 0.5` and `cited.answer_text` contains at least one `[1]` citation marker.

(Solutions: ../solutions/week-06/day-6.md)

---

## Self-check

- [ ] I can explain the difference between faithfulness and relevance as separate RAG metrics.
- [ ] I can implement a citation prompt that instructs the LLM to emit `[N]` markers inline.
- [ ] I can write a regex that extracts citation numbers from an answer string.
- [ ] I can implement per-sentence grounding verification using a single structured LLM call.
- [ ] I can define `faithfulness_score = grounded / total` and explain when it's misleading.
- [ ] I can describe the five stages of the Graph RAG pipeline (entity extraction → graph construction → community detection → hierarchical summarisation → query routing).
- [ ] I can explain when Graph RAG is worth its higher build cost over vector RAG.
- [ ] I can implement a minimal `GraphRetriever` with local and global query paths.

---

## Supplementary links

- Graph RAG paper (Edge et al., 2024): https://arxiv.org/abs/2404.16130
- Microsoft GraphRAG open source: https://github.com/microsoft/graphrag
- RAGAS evaluation framework: https://docs.ragas.io/
- TruLens RAG triad (faithfulness, relevance, context): https://www.trulens.org/trulens/getting_started/core_concepts/rag_triad/
- Leiden community detection: https://leidenalg.readthedocs.io/
- NLI-based hallucination detection (HHEM): https://huggingface.co/vectara/hallucination_evaluation_model
- Agentic RAG day notes: ./day-5.md
