# Phase 4 · Week 08 · Day 7 — Sunday: Final Assessment + Documentation

> Objective: Document the Document Intelligence Platform to production standards and complete the week 8 assessment.  ·  Time: ~3 hr build + 1 hr assessment  ·  Prereqs: [[day-6]]

## Concept

Documentation is not a post-hoc obligation. It is the artifact that answers the question "can someone else (or future you, six months from now) understand, operate, and extend this system without asking you?" A README that passes this test has five properties:

1. **Orientation** — what the system does and who it is for, in two sentences
2. **Architecture** — a diagram that shows components and data flow without requiring code to be read
3. **Quick-start** — working commands that go from zero to a running instance in under five minutes
4. **Reference** — every API endpoint, every environment variable, every config option, documented precisely
5. **Troubleshooting** — the five most common failure modes and how to diagnose them

The Mermaid diagram is a first-class documentation artifact. Render it locally with `mmdc` or in GitHub Markdown. It should be updated whenever the architecture changes.

## How it works internally

**Mermaid flowchart rendering**: GitHub renders `mermaid` code blocks in Markdown automatically. The diagram lives in source alongside the code, diffs correctly in pull requests, and never gets out of sync with a separate Visio or draw.io file. Use `LR` (left-right) orientation for pipeline diagrams and `TB` (top-bottom) for hierarchical component trees.

**API reference generation**: FastAPI generates an OpenAPI 3.1 schema at `/openapi.json`. You can download this and commit it, or redirect humans to the auto-rendered Swagger UI at `/docs`. The manual endpoint table in the README is a quick-reference supplement, not a replacement — the OpenAPI schema is the source of truth.

**Self-assessment vs. unit tests**: The self-assessment checklist in a README tests understanding, not code. The distinction matters: you can have 100% test coverage and still not be able to explain why your chunking strategy uses a 512-token window. Both are required; neither substitutes for the other.

## Worked example

Below is a complete README template for the Document Intelligence Platform. All sections are filled in. Adapt it to your own build — the architecture diagram, API table, and troubleshooting section should reflect the actual system you built.

```markdown
# Document Intelligence Platform

Query your document corpus with grounded, citation-backed answers.

Built on FastAPI, ChromaDB, and Claude — ingests PDFs and plain text, chunks and embeds them,
retrieves relevant context with hybrid search and reranking, and generates answers with source
citations. Includes an evaluation harness against golden datasets.

## Architecture

\`\`\`mermaid
flowchart LR
    subgraph Client
        A[HTTP Client]
    end

    subgraph API["FastAPI (src/api)"]
        B[SecurityMiddleware]
        C[InputValidationMiddleware]
        D[/ingest]
        E[/query]
        F[/eval/run]
        G[/metrics]
    end

    subgraph Services["Domain Services (src/services)"]
        H[IngestionService]
        I[RetrievalService]
        J[GenerationService]
        K[EvalHarness]
    end

    subgraph Infra["Infrastructure Adapters (src/adapters)"]
        L[ChromaAdapter]
        M[AnthropicAdapter]
        N[RedisAdapter]
    end

    subgraph External["External Services"]
        O[(ChromaDB)]
        P[Anthropic API]
        Q[(Redis)]
    end

    A -->|Bearer token| B --> C
    C --> D & E & F & G
    D --> H --> L --> O
    E --> I --> L
    I --> J --> M --> P
    J --> N --> Q
    F --> K --> M
    G --> N
\`\`\`

## Quick-start

Prerequisites: Docker, Docker Compose, an Anthropic API key.

\`\`\`bash
git clone https://github.com/your-org/document-intelligence.git
cd document-intelligence

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and API_SECRET_KEY

docker-compose up -d --build
sleep 15

# Verify the stack is up
curl http://localhost:8080/health

# Ingest a document
curl -X POST http://localhost:8080/ingest \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "Your document text here", "collection": "default"}'

# Query
curl -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here", "collection": "default"}'
\`\`\`

## API reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | No | Liveness + dependency health check |
| POST | /ingest | Bearer | Ingest a document (URL or raw text) |
| POST | /query | Bearer | RAG query with citations |
| GET | /collections | Bearer | List all vector store collections |
| DELETE | /documents/{doc_id} | Bearer | Remove a document and its chunks |
| POST | /eval/run | Bearer | Run eval harness against golden dataset |
| GET | /metrics | Bearer | Aggregated cost and latency metrics |

Full OpenAPI schema: http://localhost:8080/openapi.json
Swagger UI: http://localhost:8080/docs

## Configuration reference

All configuration is via environment variables. See `.env.example` for the full list.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-haiku-5-5` | Model for generation |
| `API_SECRET_KEY` | Yes | — | Bearer token for API auth |
| `CHROMA_HOST` | No | `localhost` | ChromaDB hostname |
| `CHROMA_PORT` | No | `8000` | ChromaDB port |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection URL |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `ENVIRONMENT` | No | `development` | `development` or `production` |
| `MAX_INPUT_LENGTH` | No | `4096` | Max characters in query input |

## Running tests

\`\`\`bash
# Unit tests (no external services)
pytest tests/unit/ -v

# Integration tests (requires running ChromaDB + Redis)
docker-compose up -d chromadb redis
pytest tests/integration/ -v

# Smoke tests (requires full stack)
SMOKE_TEST_BASE_URL=http://localhost:8080 pytest tests/smoke/ -v
\`\`\`

## Troubleshooting

**ChromaDB connection refused on startup**
The app depends on ChromaDB being healthy. If you see `Connection refused` in the logs, check:
\`docker-compose ps chromadb\` — it should be `healthy`. Tail its logs with
\`docker-compose logs chromadb\`. The most common cause is the volume mount having wrong permissions.

**401 Unauthorized on all requests**
Verify `API_SECRET_KEY` in your `.env` matches the `Authorization: Bearer <token>` header you are sending.
Both sides are loaded from env vars — a mismatch usually means one side has trailing whitespace.

**Empty citations array in query response**
If the query returns an answer but no citations, the collection name in the query request does not match
the collection used during ingestion. Collection names are case-sensitive.

**Extremely slow first query**
The embedding model is loaded lazily on first use. If `EMBEDDING_PROVIDER=local`, the first request
downloads the model. Subsequent requests use the in-memory cache and are fast.

**`422 Unprocessable Entity` on /query with valid JSON**
The `InputValidationMiddleware` rejected your request. Enable `DEBUG` logging to see which pattern matched.
Common cause: the question contains a phrase that looks like a prompt injection attempt to the blocklist.
```

## Exercises

1. Fill in the README template above for the platform you built in day-6. Replace every placeholder (`your-org`, `Your document text here`, etc.) with real content from your implementation. Pay particular attention to the Mermaid diagram — update it to reflect the actual component boundaries you built, not the template.

2. Write three interview questions about your system as if you were preparing to defend it to a senior engineer. Then write your own answers. Questions should cover: one design decision you made and why, one thing you would change if you had more time, and one failure mode the current system cannot handle gracefully.

3. Audit your implementation against the day-5 security checklist: API key auth, PII scrubbing, prompt injection defense, no secrets in logs, non-root Docker user. For any item not yet implemented, write a one-paragraph implementation plan with the specific file and function you would change.

(Solutions: ../solutions/week-08/day-7.md)

## Self-check

- [ ] I can explain the five properties of a production-ready README without reference
- [ ] I can draw the Document Intelligence Platform architecture from memory, naming each layer
- [ ] I can answer "why did you choose ChromaDB over Pinecone/Weaviate/pgvector?" with a concrete trade-off argument
- [ ] I can explain what would change architecturally if the platform needed to support 100 concurrent users
- [ ] I can identify the single most likely failure point in the system under production load

## Supplementary links

- [Mermaid diagram syntax reference](https://mermaid.js.org/syntax/flowchart.html)
- [The Art of the README](https://github.com/hackergrrl/art-of-readme)
- [Google's technical writing course](https://developers.google.com/tech-writing)
- [OpenAPI 3.1 specification](https://spec.openapis.org/oas/v3.1.0)
- [Diátaxis documentation framework](https://diataxis.fr/) — tutorials vs how-to guides vs reference vs explanation
