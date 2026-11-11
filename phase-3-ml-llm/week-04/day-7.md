# Phase 3 · Week 04 · Day 7 (Sunday) — Project + Assessment

> Objective: Submit the week project and complete the assessment.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Sunday structure

| Time block | Activity |
|---|---|
| 0:00 – 0:30 | Review day-1 through day-5 notes; close any open exercises |
| 0:30 – 2:30 | Polish and complete the week project (see `project.md`) |
| 2:30 – 3:00 | Self-grade against the project rubric |
| 3:00 – 4:00 | Complete the assessment (`assessment.md`) |

## Week review: key ideas

### Text preprocessing recap

Raw text → normalised tokens → sparse matrix (BoW / TF-IDF). Classical ML classifiers (Naive Bayes, Logistic Regression, SVM) work directly on TF-IDF vectors. Main limitation: no word order, no semantics.

### Word embeddings recap

Dense vectors where geometry encodes meaning. word2vec (Skip-gram / CBOW) and GloVe both exploit co-occurrence statistics. Cosine similarity measures directional similarity. Embeddings enable analogical arithmetic.

### Tokenisation recap

Subword tokenisation (BPE, WordPiece, SentencePiece) eliminates OOV, keeps vocabulary manageable, and implicitly represents morphology. BPE uses greedy frequency-based merges; WordPiece uses likelihood gain; SentencePiece unigram uses top-down pruning.

### Sentence & contextual embeddings recap

Transformers produce contextual token representations. Sentence-BERT fine-tunes for semantic similarity tasks. Sentence embeddings enable efficient semantic search via cosine similarity + approximate nearest neighbours.

### Testing pattern

Always mock external embedding APIs:
```python
mock_embedder.embed.side_effect = lambda texts: fake_vectors(texts)
```
Never commit API keys. Use `os.getenv("OPENAI_API_KEY")`.

## Reflection prompts

Work through these before the assessment:

1. **Trade-off**: When would you choose TF-IDF + logistic regression over sentence embeddings + nearest-neighbour search? Name one scenario for each.
2. **Debugging**: Your BPE tokenizer produces fewer tokens than expected after loading from disk. What is the most likely cause?
3. **Architecture**: A colleague suggests storing raw API embeddings in a Python dict keyed by document string. What are the risks at scale? What would you use instead?
4. **Complexity**: The BPE training loop is O(V × N × M) where V = vocabulary, N = corpus tokens, M = merge count. Which part is dominant for large corpora and how do production libraries optimise it?
5. **Evaluation**: How would you measure whether your semantic search engine is actually returning relevant results? Name two quantitative metrics.

## Assessment

See `assessment.md` for the graded assessment.

## Self-check
- [ ] I can build a text preprocessing pipeline without reference.
- [ ] I can implement BPE training and inference from scratch.
- [ ] I can write a semantic search engine with a swappable embedding provider.
- [ ] I can mock external APIs correctly so tests never hit the network.
- [ ] I can explain the progression from BoW → static embeddings → contextual embeddings.
