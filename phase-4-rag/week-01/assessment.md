# Week 01 Assessment — RAG Foundations & Document Ingestion

## Conceptual (interview-style)

1. **What is RAG and why does retrieval improve LLM output quality?**
   Explain the retrieval-augmented generation pattern at the architecture level. Why does grounding the LLM on retrieved documents reduce hallucination compared to relying solely on parametric knowledge? What is the role of the vector store in this architecture?

2. **RAG vs fine-tuning vs long context — when do you choose each?**
   A team is building a legal document Q&A system with 200 000 documents that are updated weekly. Explain the trade-offs of each approach along the dimensions of: update latency, inference cost, hallucination risk, and engineering complexity. Which would you recommend and why?

3. **Why does document cleaning matter before chunking and embedding?**
   Give two concrete examples of how dirty text (unicode artifacts, excessive whitespace, boilerplate noise) degrades retrieval quality. Explain the mechanism — how does the noise affect the embedding vector and the cosine similarity score?

4. **How does metadata improve retrieval quality in a RAG system?**
   Describe two ways metadata enables retrieval that pure semantic search cannot: (a) pre-filtering (e.g., date range, source type), (b) post-retrieval attribution (e.g., citing the source document). What fields should every `Document` carry at minimum?

5. **What is pipeline idempotency and how do you implement it?**
   Define idempotency in the context of an ingestion pipeline. Describe a concrete implementation using content hashing. What is stored, where, and how is it checked? What failure mode does idempotency prevent in a production system that runs the pipeline nightly?

6. **Debugging scenario**: You are running a PDF ingestion pipeline and see this traceback:

   ```
   File "loaders/pdf.py", line 18, in load
     reader = pypdf.PdfReader(str(path))
   pypdf.errors.PdfStreamError: Stream has ended unexpectedly
   ```

   (a) What are the two most likely root causes?
   (b) How would you distinguish between them programmatically?
   (c) What should the pipeline do with this document — skip, retry, or fail?
   (d) What metadata flag would you add to signal this failure to a downstream monitoring system?

## Coding challenge

### Spec

Build a `DocumentIngestionPipeline` class in a single file `ingestion.py`.

**Requirements:**

```python
from pathlib import Path
from typing import Protocol
import logging
import os

# (a) Accepts a directory path
pipeline = DocumentIngestionPipeline(input_dir=Path("/some/dir"))

# (b) Discovers files by extension: .md, .csv, .txt only
# (c) Loads each file into a Document (Pydantic model) with:
#     - text: str  — file contents
#     - metadata: DocumentMetadata with at minimum:
#         source_path: str
#         format: str   ("markdown" | "csv" | "text")
# (d) Cleans the text: unicodedata.normalize("NFC", text) + re.sub(r"\s+", " ", text).strip()
# (e) Returns list[Document]
docs: list[Document] = pipeline.run()
```

**Acceptance criteria:**

- `pipeline.run()` returns an empty list (not an exception) when `input_dir` does not exist or contains no supported files.
- Progress is logged at `INFO` level: at minimum, file count discovered and document count returned.
- No secrets hardcoded. Any API key references use `os.getenv("VAR_NAME")`.
- All function and method signatures have type hints.
- Pydantic v2 models for `Document` and `DocumentMetadata`.
- At least **2 pytest tests** using the `tmp_path` fixture:
  - Test 1: pipeline returns correct `Document` objects for a directory with one `.md` and one `.csv` file.
  - Test 2: pipeline returns an empty list when given a nonexistent directory.

**Time box: 90 minutes.**

**What to submit:** `ingestion.py` + `test_ingestion.py` in the same directory.

**Constraint:** stdlib + Pydantic only. Do not install `pypdf`, `python-docx`, or `beautifulsoup4` — this challenge covers only the three text formats.

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation (no reference solution consulted) | 40% |
| Conceptual understanding (interview answers are clear, accurate, precise) | 25% |
| Debugging & problem-solving (Q6 diagnosis is correct and actionable) | 20% |
| Code quality & testing (type hints, logging, Pydantic, ≥2 tests passing) | 15% |

**Scoring guide:**

- Independent implementation: full credit if the coding challenge compiles, passes both required tests, and handles edge cases. Partial credit if tests pass but acceptance criteria are partially met.
- Conceptual: full credit for answers that include mechanism (not just definition). Partial credit for correct-but-shallow answers.
- Debugging: full credit requires identifying both root causes, a correct programmatic distinction, and a justified disposition strategy. Partial credit for identifying one root cause.
- Code quality: full credit requires type hints on all signatures, `logging` not `print`, Pydantic models, no hardcoded secrets, `os.getenv` used. Each missing element reduces score proportionally.

Answer key: ../solutions/week-01/assessment.md
