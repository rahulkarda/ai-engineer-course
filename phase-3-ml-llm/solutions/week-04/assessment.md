# Solutions — Week 04 · Assessment

## Conceptual answers

### 1. BoW vs TF-IDF

A BoW vector's dimension `j` holds the raw integer count of term `j` in the document. A TF-IDF vector's dimension `j` holds the product of TF(j, doc) — normalised frequency within the document — and IDF(j, corpus) — the log-inverse of the fraction of documents containing term `j`.

TF-IDF improves classification because it suppresses terms that appear in most documents (e.g. "the", "is") — they carry no discriminative signal — and amplifies rare, topic-specific terms that appear in few documents but many times when they do appear. The result is a more discriminative feature space.

---

### 2. BPE — first three steps on toy corpus

Corpus: `["aab aac", "aab aab", "bbc bbc bbc"]`

Initial character vocab (word → chars → frequency):
```
('a','a','b</w>') : 3
('a','a','c</w>') : 1
('b','b','c</w>') : 3
```

**Step 1 — count pairs:**
```
('a','a'): 3+3+1 = 4   ← from aab×3, aac×1  (note: each "aab" has one 'a','a' pair)
Wait — aab has chars (a,a,b</w>) → pairs: (a,a) and (a,b</w>)
aab appears 3 times → (a,a): 3; (a,b</w>): 3
aac appears 1 time  → (a,a): 1; (a,c</w>): 1
bbc appears 3 times → (b,b): 3; (b,c</w>): 3
```
Most frequent: `('a','a')` freq=4 and `('b','c</w>')` freq=3 and `('a','b</w>')` freq=3 → tie-break by corpus order or alphabetical: pick `('a','a')`.

Merge `('a','a') → 'aa'`:
```
('aa','b</w>') : 3
('aa','c</w>') : 1
('b','b','c</w>') : 3
```

**Step 2 — count pairs:**
```
('aa','b</w>'): 3
('aa','c</w>'): 1
('b','b'): 3
('b','c</w>'): 3
```
Tie: `('aa','b</w>')`, `('b','b')`, `('b','c</w>')` all at 3. Pick `('aa','b</w>')` (alphabetical).

Merge `('aa','b</w>') → 'aab</w>'`:
```
('aab</w>') : 3
('aa','c</w>') : 1
('b','b','c</w>') : 3
```

**Step 3 — count pairs:**
```
('aa','c</w>'): 1
('b','b'): 3
('b','c</w>'): 3
```
Tie: pick `('b','b')`.

Merge `('b','b') → 'bb'`:
```
('aab</w>') : 3
('aa','c</w>') : 1
('bb','c</w>') : 3
```

---

### 3. OOV handling

With a fixed word-level vocabulary, `"tokenizability"` maps to `[UNK]` — all information is lost. The model cannot distinguish `"tokenizability"` from `"asdfghjkl"`.

With BPE, the word is split into known subword pieces. Even if the full string is unseen, its components (`"token"`, `"ize"`, `"ability"`) or individual characters are in the vocabulary. The model receives a meaningful (if imperfect) representation, and can generalise from semantically related sub-units.

---

### 4. Contextual vs static embeddings

Sentence: `"I deposited the cheque at the bank after a walk by the river bank."`

Static word2vec gives a single vector for `"bank"` — typically a blend of both senses. When this vector is used downstream, the model cannot distinguish which sense is intended.

BERT produces two different vectors for the two occurrences of `"bank"`: the first is influenced by tokens like `"deposited"`, `"cheque"` → financial sense; the second is influenced by `"walk"`, `"river"` → geographic sense. Each token vector is computed as a function of all other tokens in the sequence via self-attention.

---

### 5. Cosine vs Euclidean

Document A: `"learning"` appears 100 times. Document B: `"learning"` appears 10 times.

Raw count vectors: `A = [100, ...]`, `B = [10, ...]`. Euclidean distance is large because magnitudes differ by 10×.

Cosine similarity divides by magnitudes: `cos(A, B) = (A · B) / (‖A‖ · ‖B‖)`. If the documents discuss the same topics (just one is longer), their normalised direction is similar → high cosine score. TF-IDF normalisation (built into `TfidfVectorizer`'s L2 normalisation) further ensures length-independent comparison.

---

### 6. WordPiece vs BPE

BPE selects the pair with the highest **raw frequency** in the current symbol sequences.

WordPiece selects the pair that maximises the **language model likelihood gain**: it prefers `(A, B)` if `freq(AB) / (freq(A) * freq(B))` is largest — favouring pairs with high mutual information over mere co-occurrence.

WordPiece would segment `"unwanted"` as `["un", "##want", "##ed"]` rather than merging the most frequent characters blindly. In morphologically rich languages (German, Turkish), WordPiece tends to produce cleaner prefix/suffix splits because it resists merging frequent characters that happen to co-occur by coincidence.

---

### 7. Sentence-BERT

Sentence-BERT uses a **Siamese network**: two identical BERT encoders (shared weights) process two sentences simultaneously. Each encoder produces a sentence vector (via mean pooling). The training objective is:

- **NLI**: classify sentence pairs as entailment, neutral, or contradiction using the concatenation `[u; v; |u-v|]`.
- **STS**: minimise MSE between cosine similarity of the vectors and gold similarity scores.

Plain BERT's `[CLS]` token is pretrained for masked language modelling, not for semantic similarity. The Siamese fine-tuning directly optimises the geometry of the embedding space so that semantically similar sentences are close under cosine similarity — producing a dramatically better semantic similarity search performance.

---

### 8. API mocking

`unittest.mock.patch` replaces a named object (e.g. `openai.OpenAI`) with a `MagicMock` for the duration of the `with` block. The test code runs the function under test; whenever the function calls the patched object, it receives the mock's pre-configured return values instead of making a real API call.

The mock should return a `MagicMock` object whose `.embeddings.create()` method returns an object with a `.data` attribute: a list of objects each having an `.embedding` attribute containing a list of floats matching the expected embedding dimension. This exactly replicates the real OpenAI API response structure.

---

## Coding challenge — reference solutions

### Challenge A — BPE encoder/decoder

See `day-3.md` worked example for the core implementation. Key addition:

```python
def decode(self, tokens: list[str]) -> str:
    words, current = [], []
    for tok in tokens:
        if tok.endswith(END_OF_WORD):
            current.append(tok[:-len(END_OF_WORD)])
            words.append("".join(current))
            current = []
        else:
            current.append(tok)
    if current:
        words.append("".join(current))
    return " ".join(words)
```

Passing assertions verified in `tests/test_bpe_tokenizer.py` from day-6.

### Challenge B — SemanticSearchEngine

See `day-5.md` and `day-6.md` reference implementations. Key pattern:

```python
class SemanticSearchEngine:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._docs: list[str] = []
        self._index: np.ndarray | None = None

    def build_index(self, documents: list[str]) -> None:
        self._docs = list(documents)
        vecs = self._embedder.embed(documents)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._index = vecs / np.where(norms == 0, 1, norms)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Call build_index first.")
        q = self._embedder.embed([query])[0]
        q /= np.linalg.norm(q) or 1
        scores = self._index @ q
        idx = scores.argsort()[::-1][:min(top_k, len(self._docs))]
        return [(self._docs[i], float(scores[i])) for i in idx]
```

---

## Grading notes

| Band | Score | Typical profile |
|---|---|---|
| Distinction | ≥ 90% | Full round-trip + MMR or byte-BPE stretch goal + all 8 conceptual questions precise |
| Pass | 75–89% | BPE + search engine working, ≥ 11 passing tests, 6/8 conceptual questions clear |
| Borderline | 65–74% | One component missing or major edge case unhandled; schedule a 30-min review |
| Fail | < 65% | Re-do days 3 and 5 before re-submitting |
