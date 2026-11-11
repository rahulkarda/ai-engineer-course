# Phase 4 · Week 04 · Day 2 — Keyword Retrieval: TF-IDF & BM25 Theory

> Objective: Understand TF-IDF and BM25 from first principles; know when keyword retrieval wins.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

**Keyword (sparse) retrieval** scores documents by the importance of shared terms between the query and document, without neural encoding. It is fast, explainable, zero-shot (no training data needed), and excels on exact-match tasks (product IDs, names, rare technical terms).

### TF-IDF

**Term Frequency (TF):** how often term *t* appears in document *d*.

$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

(Some variants use raw count or log(1 + count).)

**Inverse Document Frequency (IDF):** penalises terms that appear in many documents (common words like "the" are uninformative).

$$\text{IDF}(t) = \log \frac{N}{1 + \text{df}(t)}$$

where *N* = corpus size, df(*t*) = number of docs containing *t*.

**TF-IDF score:**

$$\text{score}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Query score = sum of TF-IDF over query terms present in the document.

#### Limitations of TF-IDF

1. **Document length bias** — longer documents accumulate more term occurrences → higher TF even if content density is the same.
2. **Saturated term frequency** — the 10th occurrence of a term adds less signal than the 2nd, but TF treats them linearly.
3. **No field weighting** built in.

### BM25 (Best Match 25)

BM25 (Robertson & Zaragoza, 2009) addresses TF-IDF's two main weaknesses with two parameters:

$$\text{BM25}(t, d) = \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where:
- **f(t, d)** = raw term frequency in document *d*
- **k₁** (default 1.2–2.0) — TF saturation: as f → ∞, numerator saturates at (k₁+1)·IDF
- **b** (default 0.75) — length normalisation: b=0 disables it; b=1 full normalisation
- **|d|** = document length; **avgdl** = average document length in corpus

**IDF in BM25** (Lucene / Elasticsearch variant):

$$\text{IDF}(t) = \ln\!\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$

The "+1" inside the log keeps IDF positive even when df(*t*) > N/2.

**Full document score:**

$$\text{score}(q, d) = \sum_{t \in q} \text{BM25}(t, d)$$

## How it works internally

BM25 represents each document as a **sparse vector** — a dictionary mapping term → score. For a corpus of vocabulary size V, each document vector has at most V entries but in practice << V (most terms absent → score = 0).

Query time:
1. Tokenize query → term set Q.
2. For each term t ∈ Q, look up its posting list (all docs containing t, pre-computed at index time).
3. Compute BM25(t, d) and accumulate per document.
4. Sort by accumulated score → top-k.

Time complexity: O(|Q| · avgPostingLen) — sub-linear in N for typical queries.

**Why BM25 beats TF-IDF in practice:**
- TF saturation prevents long documents dominating via term repetition.
- Length normalisation corrects for short vs. long document bias.
- IDF formula avoids negative scores.

### Tokenisation choices matter

- **Lowercasing** — standard; be careful with proper nouns.
- **Stop word removal** — reduces index size; may hurt precision on queries like "to be or not to be".
- **Stemming / lemmatisation** — increases recall but may reduce precision on exact-match tasks.

## Worked example

```python
"""
day2_tfidf_bm25_theory.py — TF-IDF and BM25 scoring from scratch (no libraries).

Run: python day2_tfidf_bm25_theory.py
Deps: none (stdlib only)
"""

import logging
import math
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CORPUS = [
    "Python is a high-level interpreted language used for data science",
    "BM25 is a probabilistic ranking function for information retrieval",
    "Dense embeddings encode semantic meaning into fixed-length vectors",
    "Information retrieval systems return relevant documents for a query",
    "Python libraries such as NumPy and SciPy are used in data science",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_tfidf(corpus: list[str]) -> tuple[list[Counter], dict[str, float]]:
    """Return per-doc TF counters and global IDF map."""
    tokenized = [tokenize(d) for d in corpus]
    N = len(corpus)
    # IDF
    df: Counter[str] = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1
    idf = {t: math.log(N / (1 + df[t])) for t in df}
    # TF (normalised by doc length)
    tf_docs = [
        Counter({t: cnt / len(tokens) for t, cnt in Counter(tokens).items()})
        for tokens in tokenized
    ]
    return tf_docs, idf


def tfidf_score(query: str, tf_docs: list[Counter], idf: dict[str, float]) -> list[float]:
    terms = tokenize(query)
    scores = []
    for tf in tf_docs:
        s = sum(tf.get(t, 0.0) * idf.get(t, 0.0) for t in terms)
        scores.append(s)
    return scores


def bm25_score(
    query: str,
    corpus: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    tokenized = [tokenize(d) for d in corpus]
    N = len(corpus)
    avgdl = sum(len(t) for t in tokenized) / N

    # IDF (Lucene variant)
    df: Counter[str] = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    def idf(t: str) -> float:
        return math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)

    query_terms = tokenize(query)
    scores = []
    for tokens in tokenized:
        tf = Counter(tokens)
        dl = len(tokens)
        s = 0.0
        for t in query_terms:
            f = tf.get(t, 0)
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / avgdl)
            s += idf(t) * numerator / denominator if denominator != 0 else 0
        scores.append(s)
    return scores


def rank(scores: list[float], corpus: list[str]) -> list[tuple[float, str]]:
    ranked = sorted(zip(scores, corpus), reverse=True)
    return ranked


def main() -> None:
    query = "Python data science libraries"

    tf_docs, idf = build_tfidf(CORPUS)
    tf_scores = tfidf_score(query, tf_docs, idf)
    bm_scores = bm25_score(query, CORPUS)

    log.info("=== TF-IDF ===")
    for score, doc in rank(tf_scores, CORPUS):
        log.info("  [%.4f] %s", score, doc)

    log.info("=== BM25 ===")
    for score, doc in rank(bm_scores, CORPUS):
        log.info("  [%.4f] %s", score, doc)


if __name__ == "__main__":
    main()
```

### Expected insight

Both methods rank the Python+data-science doc first. BM25 typically separates scores more cleanly because of TF saturation and length normalisation.

## Exercises

1. **Parameter sensitivity.** Vary k₁ across {0.5, 1.2, 2.0} and b across {0, 0.5, 1.0}. Print the top-1 doc and its score for each combination. What happens when b=0? When k₁ is very large?
2. **Length normalisation demo.** Add a very long document (repeat "Python" 50 times with unrelated content). Without BM25 (b=0.75) vs with it, does the long doc unfairly dominate? Explain in a comment.
3. **Stop words.** Add a stop-word list `{"is", "a", "for", "in", "and", "the", "of", "to", "are", "used"}`. Filter them in `tokenize`. Does ranking change? When would removing stop words hurt?
4. **IDF variants.** Implement the classic TF-IDF IDF (`log(N/df)`) vs. the smooth IDF (`log(1 + N/df)`). Compare scores when a term appears in every document.
5. **Unit tests.** Write pytest tests for `tokenize`, `bm25_score`, and `tfidf_score` using the CORPUS. Assert at least: (a) the Python doc ranks first for `query = "Python data science"`; (b) an all-zeros score list when query has no corpus terms.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check

- [ ] I can derive the BM25 formula and explain each parameter.
- [ ] I can articulate why BM25 outperforms raw TF-IDF on long documents.
- [ ] I can implement TF-IDF and BM25 scoring in pure Python without reference.
- [ ] I understand the difference between sparse and dense retrieval representations.
- [ ] I can list two scenarios where keyword retrieval outperforms semantic retrieval.

## Supplementary links

- [The Probabilistic Relevance Framework (Robertson & Zaragoza, 2009)](https://www.staff.city.ac.uk/~sbrp622/IRF.pdf)
- [Elasticsearch BM25 documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/similarity.html)
- [rank-bm25 Python library](https://github.com/dorianbrown/rank_bm25)
