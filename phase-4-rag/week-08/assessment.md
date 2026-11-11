# Week 08 Assessment — CAPSTONE: Document Intelligence Platform

> Complete this assessment after finishing the full week build. Time: ~2 hrs. Score >= 75% to pass.

## Conceptual (interview-style)

Answer each question in 3-5 sentences. Be precise — vague answers do not earn full credit.

1. **RAG pipeline integration**: A colleague proposes replacing the entire RAG pipeline with a single prompt that pastes all documents into the context window using a 200K-token model. What are the concrete trade-offs of this approach versus retrieval-based RAG? Under what conditions would each approach be preferable?

2. **Chunking strategies**: You ingest a 300-page legal contract. Explain why fixed-size chunking at 512 tokens would produce poor retrieval results for this document type, and describe two alternative chunking strategies that would perform better. What metadata would you attach to each chunk?

3. **Reranking trade-offs**: Your latency budget for `POST /query` is 500ms. The cross-encoder reranker you are using adds 200ms. Your bi-encoder retrieval step returns 20 candidates. Describe two strategies to stay within the latency budget without removing the reranker entirely. What does each strategy sacrifice?

4. **Citation grounding**: Your platform returns citations that link answers to source chunks. A user reports that the answer is correct but the citations are from irrelevant chunks. Explain two distinct causes for this failure mode and how you would diagnose which is occurring.

5. **Observability in LLM systems**: Explain why traditional APM (average latency, error rate, throughput) is insufficient for monitoring an LLM-backed service. What three additional signals would you instrument, and what anomaly would each detect?

6. **Docker security best practices**: Explain why running the FastAPI container as root is a security risk even if the container is isolated from the host network. Describe three concrete hardening steps you took (or would take) in the Dockerfile and explain the threat each mitigates.

7. **Hybrid retrieval**: Define the BM25 score and the cosine similarity score. In a hybrid search, what does the reciprocal rank fusion (RRF) algorithm do, and why is it preferred over a simple weighted average of the two scores?

8. **Evaluation harness**: Your eval harness reports 82% faithfulness and 71% answer relevance on the golden dataset. Before shipping, what is the minimum additional investigation you would do, and what thresholds would you use to decide whether the system is production-ready?

---

## Coding challenge

### Specification

Extend the Document Intelligence Platform with **multi-tenant namespacing** in the vector store.

Currently, all documents share a single default collection. In a multi-tenant deployment, each tenant must be isolated: tenant A cannot query tenant B's documents, even if they share the same ChromaDB instance.

### Requirements

**Functional:**
- Each `POST /ingest` request accepts an optional `tenant_id: str` field (alphanumeric, max 32 chars)
- Documents are stored in a ChromaDB collection named `{tenant_id}__{collection}` (double underscore separator)
- Each `POST /query` and `DELETE /documents/{doc_id}` must also accept `tenant_id`
- A request without `tenant_id` falls back to `tenant_id = "default"`
- Tenant IDs are validated: only `[a-z0-9_-]`, max 32 chars, no double underscores
- A tenant cannot query or delete documents from another tenant's namespace, even if they know the collection name

**Non-functional:**
- The change must not break any existing tests
- At least 3 new unit tests covering: valid tenant ID, invalid tenant ID (should return 422), cross-tenant access attempt (should return 403)
- Type hints on all new code
- No new environment variables required

### Acceptance criteria

1. `POST /ingest` with `tenant_id: "acme"` and `collection: "reports"` stores chunks in ChromaDB collection `acme__reports`
2. `POST /query` with `tenant_id: "acme"` only retrieves from `acme__*` collections
3. `POST /query` with `tenant_id: "evil"` attempting to query collection `acme__reports` directly returns 403
4. `GET /collections` returns collections namespaced to the requesting tenant only
5. All unit tests pass: `pytest tests/unit/ -v`
6. Ruff reports no violations: `ruff check src/`

### Deliverables

- Modified `src/api/schemas.py` — updated request/response models
- Modified `src/services/ingestion.py` and `src/services/retrieval.py` — namespacing logic
- New `src/api/middleware/tenant.py` — tenant validation middleware
- New `tests/unit/test_tenant.py` — unit tests for the three acceptance criteria above
- Updated OpenAPI schema (can be generated with `curl http://localhost:8080/openapi.json`)

---

## Rubric (pass >= 75%)

| Dimension | Weight | Pass criteria |
|---|---|---|
| Independent implementation | 40% | Tenant namespacing works end-to-end without reference to solutions; no copy-paste from day-6 without adaptation |
| Conceptual understanding | 25% | Conceptual questions answered with specific trade-offs, not generic statements; RAG internals explained correctly |
| Debugging and problem-solving | 20% | Can identify and describe failure modes; troubleshooting approach is systematic (logs, unit test isolation, minimal reproduction) |
| Code quality and testing | 15% | Type hints on all new code; at least 3 new tests; ruff reports 0 violations; no secrets in source |

Partial credit is given for each dimension. A response that covers 3 of 4 dimensions at passing quality earns a pass.

---

Answer key: ../solutions/week-08/assessment.md
