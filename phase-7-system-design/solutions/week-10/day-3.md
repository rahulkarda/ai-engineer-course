# Solutions — Week 10 · Day 3: Communicating Technical Decisions

## Exercise 1 — Technical rewrite for a product manager

**Original (for engineers):**
> "We're switching the embedding layer from a 768-dimensional dense vector to a 1536-dimensional sparse-dense hybrid using BM25 + MRL. The change increases recall@10 by 12 points on our internal eval set and adds approximately 40ms to P95 latency due to the dual-pass retrieval."

**Rewrite for a product manager:**

> We tested a new search approach that finds 12% more relevant results for the same query. The tradeoff: each search takes 40ms longer. Our current search is well within our speed target (80ms budget remaining), so this change is safe to ship and would noticeably improve the quality of results users see. Recommend including in the next release.

**What changed and why:**
- "768-dimensional dense vector," "BM25 + MRL," "dual-pass retrieval" — removed entirely. The PM cannot act on these terms.
- "recall@10" → "finds 12% more relevant results." Same number, human-readable meaning.
- "40ms P95 regression" → "each search takes 40ms longer" plus context on the remaining budget. The PM needs to know whether this is safe, not the raw number.
- Conclusion moved to the front (BLUF) and ends with a recommendation.

---

## Exercise 2 — Analogy constructions

### (a) Vector similarity search

> "Think of vector similarity search like finding songs that 'sound similar' to one you like — not by matching the title or artist, but by comparing a numerical fingerprint of the audio's tone, tempo, and rhythm. Unlike a music app's recommendation, our system's fingerprints capture semantic meaning in text, not audio features."

### (b) Rate limiting

> "Rate limiting is like a restaurant that takes only 50 reservations per dinner service — once they're full, new requests go on a waiting list or are turned away. Unlike a restaurant, our system processes requests in milliseconds, so the 'restaurant' fills and empties thousands of times per second."

### (c) Eventual consistency

> "Eventual consistency is like a Wikipedia article that 10 people are editing simultaneously — if you read it at different data centers around the world in the same second, you might see slightly different versions. Unlike Wikipedia, our system guarantees all versions converge to the same state within a bounded time window, typically under 2 seconds."

---

## Exercise 3 — BLUF rewrite

**Original (buried conclusion):**
> "We looked at three hosting options for the inference API. Cloud A has the lowest cost but poorest latency. Cloud B has the best latency but is 3x more expensive. Cloud C is middle on both dimensions. Given our latency SLA of 200ms and budget of $8K/month, Cloud C appears to be the best fit. Should we proceed?"

**Word count:** ~65 words.

**BLUF rewrite:**
> "Recommend Cloud C for the inference API. It meets our 200ms SLA and fits within the $8K/month budget. Cloud A is cheaper but misses the latency target; Cloud B meets latency but costs 3x more. Ready to proceed on your approval."

**Word count:** ~44 words. 32% shorter. The reader knows the conclusion in the first 6 words.

---

## Exercise 4 — Stakeholder mapping (reference example)

For a project migrating a batch ML pipeline to streaming:

| Stakeholder | Primary concern | Lead with |
|------------|----------------|-----------|
| Engineering team | Correctness, migration complexity | Architecture diagram, interface contracts, testing strategy |
| Data science team | Impact on their pipelines | What changes in the API contract; migration timeline; backward compat guarantee |
| Engineering manager | Headcount and timeline | 6-week timeline, 2-engineer ask, risk mitigation plan |
| Product manager | User-visible outcome | Fraud detection improves in real-time vs. day-old data; estimated dollar impact |
| CTO | Strategic alignment | Why this positions us vs. competitors; total cost of ownership |
