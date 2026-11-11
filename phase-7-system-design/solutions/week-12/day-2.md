# Solutions — Week 12 · Day 2: HLD + AI-System-Design Mock Loop

## Exercise 1 — Real-time leaderboard (HLD, model answer)

### Requirements

**Functional**
- Real-time top-100 global leaderboard (updated every 10 s)
- Per-region top-100 (5 regions)
- Player's own rank + score on demand
- Score update endpoint (game server → leaderboard service)

**Non-functional**
- 10 M players, scores change every 10 s on average
- Leaderboard read p99 < 100 ms; score update p99 < 50 ms
- Availability 99.9%; eventual consistency acceptable for rank display

**Scale math**
- Updates: 10M players / 10 s = 1 M updates/s peak (burst)
- Reads: 10M × 2 reads/min (game lobby refresh) = 333K reads/s
- Redis sorted set (ZSET) ops: O(log N) insert, O(N) range — N=10M → log₂(10M) ≈ 23 ops

### API contract

```
POST /v1/scores              body: {player_id, score, region} → {rank}
GET  /v1/leaderboard/global  ?limit=100 → [{rank, player_id, score}]
GET  /v1/leaderboard/region/{region} ?limit=100 → same
GET  /v1/players/{player_id}/rank → {global_rank, region_rank, score}
```

### Architecture diagram

```
Game Server
    │ POST /v1/scores
    ▼
Score Ingestion Service (async FastAPI)
    │
    ├─ Kafka topic: score-updates (partition by player_id)
    │
Leaderboard Workers (consumer group)
    │  ZADD global_lb score player_id
    │  ZADD region_lb:{region} score player_id
    ▼
Redis Cluster (ZSET per leaderboard, TTL=0 — persistent)
    │
Read Service (cached)
    │  ZREVRANGE global_lb 0 99 WITHSCORES  (O(100) = O(1) practical)
    │  ZRANK global_lb player_id             (own rank)
    ▼
CDN (cache top-100 response, 10s TTL)
    │
Client
```

### Data model

```sql
-- Long-term storage for audit/history (not hot path)
CREATE TABLE score_events (
    event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id   UUID NOT NULL,
    score       BIGINT NOT NULL,
    region      VARCHAR(10) NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON score_events (player_id, recorded_at DESC);
```

### Capacity estimate

- Redis ZSET memory: 10M entries × (8 bytes score + 16 bytes UUID) = ~240 MB — trivial
- Kafka throughput: 1M msgs/s × 100 bytes = 100 MB/s — achievable with 10 partitions
- Read service: 333K reads/s, served 95%+ from CDN → ~17K reads/s hit Redis → well within Redis 200K ops/s per shard

### Trade-offs

- **ZSET vs. Postgres ranked query**: ZSET O(log N) insert vs. Postgres window function O(N log N) — ZSET wins at 10M scale.
- **CDN caching vs. real-time**: 10s cache TTL means top-100 is slightly stale — acceptable for leaderboard (game clients refresh every 10s anyway).
- **Eventual vs. strong consistency**: Kafka async pipeline means a score may appear on leaderboard up to 500ms late — acceptable.

---

## Exercise 2 — Enterprise AI coding assistant (model answer, condensed)

**Model selection**: Self-hosted open-weight model (CodeLlama-70B or DeepSeek-Coder-33B) via vLLM. Rationale: code never leaves VPC; no per-token API cost at scale; can fine-tune on internal codebase.

**Context window strategy**: Sliding window over open files (8K tokens max) + retrieval of relevant internal code snippets via embeddings (top-3 most similar functions from internal repo index). Prioritise: cursor context (4K) > open files (2K) > retrieved snippets (2K).

**RAG vs. fine-tuning**: RAG for internal APIs/docs (frequently changing); fine-tuning for coding style (company conventions, rarely changing). Both together.

**Latency**: vLLM continuous batching, tensor parallelism on 2× A100 80GB. Streaming response to IDE (first token < 300ms target).

**Security**: VPC-isolated inference cluster; no outbound internet; audit log of all prompts/completions; PII scanner on output before display.

---

## Exercise 3 — Critique flaws

**Flaw 1 (Scale + Correctness)**: Embeddings stored as Postgres JSONB array. JSONB cannot be indexed for ANN search → full table scan on every query. At 10M vectors this is O(N) per query. Fix: use pgvector with HNSW index or dedicated vector DB (Qdrant).

**Flaw 2 (Operational + Correctness)**: External OpenAI API with no retry, no fallback. Any transient 429/503 from OpenAI fails the user request immediately. Fix: exponential backoff (3 retries, jitter), circuit breaker, fallback to a smaller cached model.

**Flaw 3 (Scale + Operational)**: Redis stores full prompt history with no TTL. Memory grows unboundedly. A session with 50 turns × 4K tokens = 200K tokens × 4 bytes = 800 KB per session. At 10K concurrent sessions → 8 GB Redis just for history. Fix: TTL (e.g., 24h), compress with zlib, store only last N turns in Redis and archive to S3.

---

## Exercise 4 — Cost math

**Setup**:
- Users: 1,000
- Requests/user/day: 20
- Total requests/day: 20,000
- Input tokens: 500 per request → 10M input tokens/day
- Output tokens: 200 per request → 4M output tokens/day

**Monthly cost** (claude-sonnet-5-5, $3/M input, $15/M output):

```
Input:  10M × 30 = 300M tokens/month × $3/M  = $900/month
Output: 4M × 30  = 120M tokens/month × $15/M = $1,800/month
Total:  $2,700/month
```

**Semantic cache break-even**:

Semantic cache costs $200/month. At what cache hit rate does it pay for itself?

Each cache hit saves: avg (500 input + 200 output) tokens = $3/M × 500 + $15/M × 200 = $0.0015 + $0.003 = $0.0045 per request.

Requests needed to save $200/month: $200 / $0.0045 = 44,444 requests/month.

Total requests/month: 20K × 30 = 600K.

Break-even hit rate: 44,444 / 600,000 ≈ 7.4%.

A semantic cache with 95%+ cosine similarity threshold easily achieves 20–30% hit rate on enterprise users asking similar questions → cache pays for itself ~4× over.
