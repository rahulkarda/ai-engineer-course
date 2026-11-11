# Project Solution Guide — Week 06

See day-1.md through day-6.md for individual component implementations.
This file covers integration, self-check, and key design decisions.

---

## Integration Diagram

```
                        ┌─────────────────────────────────┐
                        │         AgenticRAGSystem         │
                        └────────────────┬────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
   ┌─────────────────┐       ┌─────────────────────┐    ┌──────────────────┐
   │  QueryRewriter  │       │   Tool-Use Loop      │    │ CitationExtractor│
   │  (day-1)        │──────►│   (day-5)            │───►│ (day-6)          │
   │  temperature    │       │   classify_source    │    │ inline [N] refs  │
   └─────────────────┘       └──────────┬──────────┘    └──────────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
        ┌──────────────────┐  ┌─────────────────┐  ┌───────────────┐
        │ HyDERetrieverMulti│  │ web_search_stub  │  │   sql_stub    │
        │ (day-2)           │  │ (day-4)          │  │               │
        │ avg 3 embeddings  │  └─────────────────┘  └───────────────┘
        └──────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ SelfRAGPipeline  │
        │ (day-4)          │
        │ RelevanceScorer  │
        │ threshold filter │
        └──────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ CorrectionRouter │
        │ (day-4)          │
        │ correct / wrong  │
        │ / ambiguous      │
        └──────────────────┘
                   │  (if multi-hop needed)
                   ▼
        ┌──────────────────┐
        │MultiHopRetriever │
        │ (day-3)          │
        │SufficiencyChecker│
        │ conf > 0.85 stop │
        └──────────────────┘
```

---

## Project Self-Check

| Criterion | Status | Notes |
|---|---|---|
| QueryRewriter with temperature param | Done | day-1, Exercise 1 |
| multi_query_expand(query, n) | Done | day-1, Exercise 2 |
| decompose_query returning list | Done | day-1, Exercise 3 |
| step_back generalization | Done | day-1, Exercise 4 |
| HyDERetriever.retrieve | Done | day-2, Exercise 1 |
| Multi-doc HyDE avg embeddings | Done | day-2, Exercise 2 |
| ContextualChunkBuilder.build_context | Done | day-2, Exercise 3 |
| build_context_batch with rate limiting | Done | day-2, Exercise 4 |
| MultiHopRetriever with max_hops | Done | day-3, Exercise 1 |
| SufficiencyChecker Pydantic result | Done | day-3, Exercise 2 |
| query_refiner LLM follow-up | Done | day-3, Exercise 3 |
| Confidence threshold early stop | Done | day-3, Exercise 4 |
| RelevanceScorer LLM judge | Done | day-4, Exercise 1 |
| SelfRAGPipeline threshold filter | Done | day-4, Exercise 2 |
| CorrectionRouter 3-way route | Done | day-4, Exercise 3 |
| Web fallback on incorrect route | Done | day-4, Exercise 4 |
| RETRIEVE_TOOL schema | Done | day-5, Exercise 1 |
| AgenticRAGAgent full tool loop | Done | day-5, Exercise 2 |
| classify_source keyword router | Done | day-5, Exercise 3 |
| Multi-turn max 5 iterations | Done | day-5, Exercise 4 |
| CitationExtractor inline [N] | Done | day-6, Exercise 1 |
| GroundingChecker binary | Done | day-6, Exercise 2 |
| faithfulness_score sentence-level | Done | day-6, Exercise 3 |
| GraphRetriever local_query | Done | day-6, Exercise 4 |
| Citations wired into agent | Done | day-6, Exercise 5 |
| All LLM calls mocked in tests | Done | each day, Exercise 5/6 |
| logging not print everywhere | Done | all files |
| os.getenv("ANTHROPIC_API_KEY") | Done | all files |
| Model IDs: sonnet-5-5 / opus-5-5 | Done | all files |

---

## Key Design Decisions

**Contextual prefixing at index time, not query time**
Cost amortization: an N-chunk corpus queried Q times costs N LLM calls (index time) vs N×Q (query time). The prefix is stable — it describes the chunk's role in the document, which doesn't change per query. Index-time also means the contextualized embedding captures the prefix semantics at retrieval, not just at generation.

**Tool-use over fixed pipeline**
Fixed pipelines hard-code retrieval count and timing (always retrieve once, always use 5 chunks). Tool-use lets the model decide when to retrieve, how many times, and with what query. The model can retrieve nothing (if it knows the answer), retrieve once, or retrieve iteratively. This eliminates unnecessary API calls for simple questions and enables genuine multi-hop reasoning for complex ones.
