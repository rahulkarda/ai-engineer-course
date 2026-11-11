# Solutions — Week 02 · Day 7

Day 7 is a project completion and review day. No exercise answers needed.

## Self-check answers

### Q: Which strategy produced the lowest adjacent-chunk similarity?

`MarkdownChunker` typically produces the lowest adjacent-chunk cosine similarity on structured documents. Each chunk corresponds to a distinct heading section — different sections discuss different topics — so the cosine similarity between adjacent chunks is low. `FixedSizeChunker` has the highest adjacent similarity because it overlaps characters, so adjacent chunks share text.

### Q: Which strategy has the most uniform chunk sizes?

`FixedSizeChunker` produces the most uniform sizes — all chunks are exactly `chunk_size` characters except the last. This is its main advantage: predictable storage and retrieval latency. The downside is it ignores sentence and semantic structure.

### Q: For a Markdown-heavy technical knowledge base, which strategy would you recommend?

`MarkdownChunker` with a fallback to `RecursiveChunker` for sections that exceed `max_chunk_chars`. Rationale: the document structure already encodes topic boundaries via headings; using those boundaries produces chunks with high intra-chunk semantic coherence. The `heading` metadata also enables faceted retrieval (e.g., "find only Introduction sections").

## Pre-assessment answers (for self-review)

**Fixed-size overlap:** chunk N+1 starts at `N * step` where `step = chunk_size - overlap`. Example: `chunk_size=100, overlap=20 → step=80`. Chunk 0: chars 0–99. Chunk 1: chars 80–179. Chars 80–99 appear in both.

**Recursive fallback:** if a piece exceeds `chunk_size` and `separators` is exhausted (empty string `""` is the last), the algorithm slices character-by-character into exactly `chunk_size`-length pieces — equivalent to fixed-size chunking as a last resort.

**Semantic chunking failure modes:**
1. Very long documents — the embedding of the entire text becomes expensive.
2. Domain-specific jargon — generic embedding models see all technical paragraphs as "similar" (high cosine similarity everywhere), producing no breakpoints.
Mitigations: use a domain-tuned embedding model; add a `max_chunk_chars` hard cap.

**Normalisation:** ensures `||v||=1` so cosine similarity reduces to a dot product. Vector databases optimise for dot products (BLAS GEMM); the simplification is ~2x faster per lookup.

**Cosine formula:** `cos(θ) = (a·b) / (||a|| × ||b||)`. After normalisation: `cos(θ) = a·b`.

**Embedding cost:** tokens ≈ chars / 4. Cost = (tokens / 1,000,000) × price_per_million.
