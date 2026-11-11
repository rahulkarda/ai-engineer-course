# Solutions — Week 04 · Day 7 (Week Review)

Day 7 is a review and self-assessment day. This file provides answers to the practice problems from day-7.md.

## Practice problem 1 — Manual BM25 scoring

Query: "fast retrieval"  
Document: "fast information retrieval algorithms" (5 tokens, dl=5)  
N=100, avgdl=8, k₁=1.5, b=0.75

**Setup:**  
- df("fast") = 10, df("retrieval") = 20

**IDF calculations (Lucene variant):**

IDF("fast") = ln((100 - 10 + 0.5) / (10 + 0.5) + 1)  
            = ln(90.5 / 10.5 + 1)  
            = ln(8.619 + 1)  
            = ln(9.619) ≈ **2.263**

IDF("retrieval") = ln((100 - 20 + 0.5) / (20 + 0.5) + 1)  
                 = ln(80.5 / 20.5 + 1)  
                 = ln(3.927 + 1)  
                 = ln(4.927) ≈ **1.595**

**TF-normalised scores:**

For "fast" (f=1, dl=5, avgdl=8):

denominator = 1 + 1.5 × (1 - 0.75 + 0.75 × 5/8)  
            = 1 + 1.5 × (0.25 + 0.469)  
            = 1 + 1.5 × 0.719  
            = 1 + 1.078 = 2.078

TF_norm("fast") = 1 × (1.5 + 1) / 2.078 = 2.5 / 2.078 ≈ **1.203**

Score_fast = 2.263 × 1.203 ≈ **2.723**

For "retrieval" (f=1, same denominator):

Score_retrieval = 1.595 × 1.203 ≈ **1.919**

**Total BM25 score = 2.723 + 1.919 ≈ 4.642**

## Practice problem 2 — Manual RRF

BM25 list: [A=rank1, B=rank2, C=rank3, D=rank4, E=rank5]  
Dense list: [C=rank1, A=rank2, D=rank3, F=rank4, G=rank5]

k=60

| Doc | BM25 rank | Dense rank | BM25 contrib | Dense contrib | RRF score |
|-----|-----------|------------|--------------|---------------|-----------|
| A   | 1         | 2          | 1/61≈0.01639 | 1/62≈0.01613  | **0.03252** |
| B   | 2         | —          | 1/62≈0.01613 | 0             | 0.01613   |
| C   | 3         | 1          | 1/63≈0.01587 | 1/61≈0.01639  | **0.03226** |
| D   | 4         | 3          | 1/64≈0.01563 | 1/63≈0.01587  | **0.03150** |
| E   | 5         | —          | 1/65≈0.01538 | 0             | 0.01538   |
| F   | —         | 4          | 0            | 1/64≈0.01563  | 0.01563   |
| G   | —         | 5          | 0            | 1/65≈0.01538  | 0.01538   |

**Final ranking: A, C, D, B, F, E, G**

A wins (rank 1 in BM25 + rank 2 in dense). C is close (rank 3 + rank 1).

## Practice problem 3 — Filter speedup estimation

10M documents, 10% corpus after filter = 1M documents.

**(a) BM25 with inverted index:**  
Pre-filter: restrict posting-list lookups to the 1M candidate doc_ids → ~10× speedup for score accumulation. Build time unchanged. Speedup ≈ **5–10×** (depends on query term posting list sizes).

**(b) Exact k-NN dense retrieval (flat index):**  
Flat index scans all N vectors. If we rebuild an index over the 1M subset: speedup = 10M/1M = **10×** exactly. But rebuilding at query time is expensive — better to pre-filter doc_ids and use a filtered scan.

**(c) HNSW ANN:**  
HNSW is O(log N). For N=10M: log(10M) ≈ 23. For N=1M: log(1M) ≈ 20. Speedup ≈ 23/20 = **~15%** (much smaller than flat). HNSW is already fast at 10M; filtering gives minimal query speedup but does reduce the index build cost for segment-based approaches.

**Key insight:** Filtering helps BM25 and flat-search proportionally to the filter selectivity. HNSW benefits less because its query complexity is logarithmic — the gain is small unless the filter is very aggressive (< 1% corpus).
