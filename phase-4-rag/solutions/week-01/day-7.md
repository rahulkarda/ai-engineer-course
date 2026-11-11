# Solutions — Phase 4 · Week 01 · Day 7

> Day 7 is a review and assessment-prep day. See `project.md` for the full week project solution.

---

## Exercise 1 — PDF No Text Layer Trace

**Scenario:** A scanned PDF is passed to `PDFLoader`. What happens?

### Step-by-step trace

1. `PDFLoader.load(path)` opens the file with `pypdf.PdfReader`.
2. For each page, `page.extract_text()` is called.
3. Because the PDF contains only a rasterized image (no embedded text), `extract_text()` returns `""` (or `None`, coerced to `""`).
4. A `Document` is created with `text=""` and `metadata.format="pdf"`.
5. The `TextCleaner` receives the empty string, returns `""`.
6. `CleaningStage` calls `doc.is_empty` → `True`. If `drop_empty=True` the document is silently removed from the batch.
7. The document never reaches the chunker or embedding stage.

### Detection and handling

```python
# In PDFLoader.load — already in the reference implementation:
if not text.strip():
    logger.warning(
        "PDFLoader: page %d of %s has no extractable text "
        "(may be scanned image — consider OCR).",
        page_num,
        path,
    )
```

**Options after detection:**

| Strategy | When to use |
|---|---|
| Log + skip | Acceptable when scanned PDFs are rare and you can flag them for manual review. |
| Flag in metadata | Set `doc.metadata.extra["needs_ocr"] = "true"` so a downstream process can handle them. |
| OCR fallback | Use `pytesseract` + `pdf2image` to rasterize pages and run OCR. Wire as a `FallbackLoaderStage`. |
| Reject loudly | Raise a domain exception so the caller knows the file was not processed. |

**Minimal OCR-fallback pattern:**

```python
import logging
from pathlib import Path
from doc_ingestion.models import Document, DocumentMetadata

logger = logging.getLogger(__name__)


def _ocr_page(image) -> str:  # image: PIL.Image
    try:
        import pytesseract  # type: ignore[import-untyped]
        return pytesseract.image_to_string(image)
    except ImportError:
        logger.error("pytesseract not installed — cannot OCR page.")
        return ""


class PDFLoaderWithOCRFallback:
    """Load PDF; fall back to OCR for pages with no extractable text."""

    def load(self, path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("pypdf required: pip install pypdf") from exc

        try:
            from pdf2image import convert_from_path  # type: ignore[import-untyped]
            _has_pdf2image = True
        except ImportError:
            _has_pdf2image = False

        reader = PdfReader(str(path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                logger.warning("Page %d of %s has no text — attempting OCR.", page_num, path)
                if _has_pdf2image:
                    images = convert_from_path(str(path), first_page=page_num + 1, last_page=page_num + 1)
                    text = _ocr_page(images[0]) if images else ""
                else:
                    logger.warning("pdf2image not installed — skipping OCR for page %d.", page_num)

            documents.append(
                Document(
                    text=text,
                    metadata=DocumentMetadata(
                        source_path=str(path),
                        format="pdf",
                        row_index=page_num,
                        extra={"page": str(page_num), "ocr_used": str(not bool(page.extract_text()))},
                    ),
                )
            )
        return documents
```

---

## Exercise 2 — Retry / Fallback Strategy

### Transient errors — exponential backoff

Use for errors that are likely to resolve on retry: network timeouts, rate limits, temporary file locks.

```python
"""Exponential backoff wrapper for any PipelineStage."""

from __future__ import annotations

import logging
import time
from typing import Callable

from doc_ingestion.models import Document
from doc_ingestion.pipeline import PipelineStage

logger = logging.getLogger(__name__)


class RetryStage:
    """Wrap a PipelineStage with exponential backoff on transient errors.

    Args:
        inner:           The stage to wrap.
        max_attempts:    Maximum number of attempts per document (default 3).
        base_delay_s:    Initial wait in seconds; doubles each retry (default 1.0).
        retryable:       Predicate that returns True for retryable exceptions.
                         Defaults to retrying all exceptions.
    """

    def __init__(
        self,
        inner: PipelineStage,
        max_attempts: int = 3,
        base_delay_s: float = 1.0,
        retryable: Callable[[Exception], bool] | None = None,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._retryable = retryable or (lambda _: True)

    def process(self, documents: list[Document]) -> list[Document]:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._inner.process(documents)
            except Exception as exc:  # noqa: BLE001
                if not self._retryable(exc):
                    raise
                last_exc = exc
                delay = self._base_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "RetryStage: attempt %d/%d failed (%s); retrying in %.1fs.",
                    attempt,
                    self._max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"All {self._max_attempts} attempts failed."
        ) from last_exc
```

### Format failures — fallback loader

Use when a primary loader (e.g. `pypdf`) fails and an alternative exists (e.g. `pdfminer.six`).

```python
from pathlib import Path
from doc_ingestion.loaders import DocumentLoader
from doc_ingestion.models import Document
import logging

logger = logging.getLogger(__name__)


class FallbackLoader:
    """Try loaders in order; use the first one that succeeds.

    Args:
        loaders: Ordered list of DocumentLoader instances.  The first that
                 does not raise is used; the rest are skipped.
    """

    def __init__(self, loaders: list[DocumentLoader]) -> None:
        self._loaders = loaders

    def load(self, path: Path) -> list[Document]:
        last_exc: Exception | None = None
        for loader in self._loaders:
            try:
                return loader.load(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FallbackLoader: %s failed on %s: %s — trying next.",
                    type(loader).__name__,
                    path,
                    exc,
                )
                last_exc = exc
        raise RuntimeError(
            f"All loaders failed for {path}"
        ) from last_exc
```

**Example — PDF with fallback:**

```python
from doc_ingestion.loaders import PDFLoader

# Wrap pypdf with a pdfminer fallback (if installed).
try:
    from pdfminer_loader import PdfMinerLoader  # hypothetical
    pdf_loader = FallbackLoader([PDFLoader(), PdfMinerLoader()])
except ImportError:
    pdf_loader = PDFLoader()
```

---

## Exercise 3 — Assessment Practice Answer

**Scenario:** "Build a customer-support knowledge base from 50,000 support articles updated weekly. The system must answer customer questions accurately."

### Why RAG over fine-tuning?

| Dimension | RAG | Fine-tuning |
|---|---|---|
| Knowledge freshness | Re-index updated articles in minutes | Retrain takes hours/days; deployed model is static |
| Cost per update | Near-zero (re-embed changed docs only) | Compute cost for each training run |
| Factual grounding | Retrieves exact article text; answer cites source | Model may hallucinate; no explicit source tracing |
| Scale | 50k articles fit easily in a vector store | 50k articles would be a small fine-tuning corpus — model still hallucinated |
| Debuggability | Inspect retrieved chunks to understand answers | Hard to trace why the model said something |

### Why RAG over long-context injection?

| Dimension | RAG | Long context (stuff-all-docs) |
|---|---|---|
| Latency | Retrieve 5–20 relevant chunks: fast | 50k × 500 tokens = 25M tokens per query: infeasible |
| Cost | Proportional to retrieved chunks | Proportional to full corpus |
| Relevance | Semantic retrieval surfaces the right article | Diluted by irrelevant articles; model loses focus |

### Concrete architecture choice

1. **Weekly batch job**: crawl updated articles → load with `MarkdownLoader` or `HTMLLoader` → clean → chunk (512-token sliding window, 10% overlap) → embed with `text-embedding-3-small` → upsert into a vector store (Pinecone / pgvector).
2. **Query time**: embed user question → ANN search → return top-5 chunks → LLM generates answer with citations.
3. **Freshness**: detect changed articles by comparing SHA-256 hashes against the `HashDeduplicator` cache; only re-embed changed docs.

---

## Self-check answers

**Q: What are the four stages of an ingestion pipeline?**
A: Load → Clean → Chunk → Embed (with optional Deduplicate between Clean and Chunk).

**Q: Why use a Protocol for PipelineStage instead of an abstract base class?**
A: Protocols enable structural subtyping (duck typing checked by mypy) so any class with a `process` method satisfies the interface without inheritance. This avoids tight coupling and makes third-party stages easy to integrate.

**Q: Why guard optional imports (pypdf, python-docx) with try/except ImportError?**
A: So the package installs and runs without heavy dependencies. Users who only process text/CSV/Markdown do not need to install PDF or DOCX dependencies. The stub raises a clear `ImportError` with install instructions only when the missing feature is actually used.

**Q: What does NFKC normalisation do?**
A: Decomposes and recomposes Unicode characters using compatibility equivalence. Converts ligatures (ﬁ → fi), full-width characters (Ａ → A), and stylistic variants to their canonical forms, improving tokenisation and embedding quality.

**Q: What is the difference between deduplication and idempotency?**
A: Deduplication removes exact content duplicates within a single pipeline run. Idempotency means re-running the pipeline on the same input produces the same output without creating duplicates — achieved by persisting seen hashes across runs (e.g. `CachingPipeline` with shelve).

**Q: Name two ways to handle a scanned PDF with no text layer.**
A: (1) Log a warning and skip (suitable when OCR is not available or cost is prohibitive). (2) Use `pdf2image` + `pytesseract` to OCR each page as a fallback, storing the result in the same `Document` model.
