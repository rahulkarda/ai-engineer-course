# Week 01 Project — Robust Multi-Format Document Ingestion Pipeline

## Overview

Build `doc_ingestion`, a typed Python package that ingests documents from a directory of mixed formats, cleans and normalizes the text, enriches each document with structured metadata, and outputs a JSON summary. This project integrates every concept from Week 01: document loading, text cleaning, metadata modeling, table extraction, and stage-based pipeline architecture.

## Learning goals

- Implement a stage-based ingestion pipeline using a typed `PipelineStage` Protocol with independent, replaceable stages.
- Build format-specific loaders (Markdown, CSV, plain text; optionally PDF and DOCX) that produce a consistent `Document` model.
- Apply text cleaning (unicode normalization, whitespace collapse) and table serialization (CSV rows → Markdown table) as standalone, testable components.
- Design a typed `DocumentMetadata` Pydantic model that carries provenance, content statistics, and a deduplication hash.
- Deliver a CLI-driven tool with error isolation, structured logging, and a JSON summary output.

## Project spec

### Package: `doc_ingestion`

**Required components:**

| Module | Responsibility |
|---|---|
| `models.py` | `Document` and `DocumentMetadata` Pydantic models |
| `loaders/markdown.py` | Load `.md` and `.txt` files |
| `loaders/csv_loader.py` | Load `.csv` files; serialize rows to text |
| `loaders/pdf.py` | Load `.pdf` via `pypdf` (optional — see note) |
| `loaders/docx.py` | Load `.docx` via `python-docx` (optional — see note) |
| `loaders/html.py` | Load `.html` via `html.parser` |
| `dispatcher.py` | `FileDispatcher`: map extension → loader; `LoadError` |
| `cleaning.py` | `TextCleaner`; `CleaningStage` (PipelineStage) |
| `extraction.py` | `TableExtractor`: CSV-like content → Markdown table |
| `enrichment.py` | `MetadataEnricher`: add `ingested_at`, `char_count`, `word_count`, `content_hash` |
| `pipeline.py` | `PipelineStage` Protocol, `Pipeline`, `HashDeduplicator`, `IngestionPipeline` |
| `__main__.py` | CLI entry point (`argparse` or `typer`) |

**Optional heavy dependencies (install only if building the stretch goal):**

```bash
uv add pypdf python-docx
```

Required formats for full credit: `.md`, `.csv`, `.txt`. HTML is a bonus. PDF and DOCX are stretch.

### Required models (`models.py`)

```python
class DocumentMetadata(BaseModel):
    source_path: str          # absolute path to the source file
    format: str               # "markdown" | "csv" | "text" | "html" | "pdf" | "docx"
    ingested_at: str          # ISO 8601 UTC timestamp
    char_count: int           # len(document.text)
    word_count: int           # len(document.text.split())
    content_hash: str         # sha256 of document.text (hex)
    page_count: int | None    # PDF only; None for all other formats
    extra: dict[str, str]     # format-specific extras

class Document(BaseModel):
    text: str
    metadata: DocumentMetadata
```

### File structure to create

```
doc_ingestion/                    # package root
├── __init__.py
├── __main__.py
├── models.py
├── pipeline.py
├── dispatcher.py
├── cleaning.py
├── extraction.py
├── enrichment.py
└── loaders/
    ├── __init__.py
    ├── markdown.py
    ├── csv_loader.py
    ├── html.py
    ├── pdf.py                    # optional — pypdf
    └── docx.py                   # optional — python-docx
tests/
├── conftest.py
├── test_models.py
├── test_loaders.py
├── test_cleaning.py
├── test_pipeline.py
└── test_enrichment.py
pyproject.toml
.env.example
.gitignore
```

### CLI interface

```bash
python -m doc_ingestion --input-dir ./docs --output-json ./summary.json
```

Output JSON schema:

```json
{
  "total_documents": 12,
  "formats": {"markdown": 5, "csv": 3, "text": 4},
  "documents": [
    {
      "source": "/abs/path/to/file.md",
      "format": "markdown",
      "char_count": 1234,
      "content_hash": "a1b2c3d4"
    }
  ]
}
```

### Code constraints (non-negotiable)

- Type hints on every function and method signature.
- `logging` everywhere — no `print` statements in package code.
- Pydantic v2 models for all data crossing module boundaries.
- No hardcoded paths or secrets. Use `os.getenv("VAR_NAME")` for any credentials.
- `pathlib.Path` for all filesystem operations — no `os.path`.

## Deliverables checklist

- [ ] `doc_ingestion/models.py` — `Document` and `DocumentMetadata` Pydantic models.
- [ ] `doc_ingestion/loaders/markdown.py` — loads `.md` and `.txt`.
- [ ] `doc_ingestion/loaders/csv_loader.py` — loads `.csv`, serializes rows.
- [ ] `doc_ingestion/loaders/html.py` — loads `.html` with `html.parser`.
- [ ] `doc_ingestion/cleaning.py` — `TextCleaner` and `CleaningStage`.
- [ ] `doc_ingestion/extraction.py` — `TableExtractor` serializing to Markdown table.
- [ ] `doc_ingestion/enrichment.py` — `MetadataEnricher` adding 4 metadata fields.
- [ ] `doc_ingestion/pipeline.py` — `PipelineStage` Protocol, `Pipeline`, `HashDeduplicator`, `IngestionPipeline`.
- [ ] `doc_ingestion/dispatcher.py` — `FileDispatcher` with per-file error isolation.
- [ ] `doc_ingestion/__main__.py` — CLI with `--input-dir` and `--output-json` flags.
- [ ] `tests/` — at least 5 pytest tests passing.
- [ ] `.env.example` present; `.env` in `.gitignore`.
- [ ] `pyproject.toml` with project metadata and dependencies.

## Testing requirements

Minimum **5 pytest tests** located in `tests/`. Use `tmp_path` for all filesystem fixtures. Mock any external calls (no real network, no real LLM).

Suggested test coverage:

| Test | File | What to assert |
|---|---|---|
| `test_markdown_loader` | `test_loaders.py` | Returns correct `Document` with text and metadata |
| `test_csv_loader` | `test_loaders.py` | Serializes rows; `format == "csv"` |
| `test_text_cleaner` | `test_cleaning.py` | NFC normalization + whitespace collapse |
| `test_metadata_enricher` | `test_enrichment.py` | All 4 fields present and correct types |
| `test_pipeline_dedup` | `test_pipeline.py` | Duplicate documents removed; `content_hash` matches |
| `test_pipeline_bad_file` | `test_pipeline.py` | `LoadError` on unreadable file → empty result, no exception |
| `test_cli_output_json` | `test_pipeline.py` | JSON file written with correct schema |

Mock pattern for `PdfLoader` (if implemented):

```python
from unittest.mock import patch, MagicMock

def test_pdf_loader(tmp_path):
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello world"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("doc_ingestion.loaders.pdf.pypdf.PdfReader", return_value=mock_reader):
        from doc_ingestion.loaders.pdf import PdfLoader
        doc = PdfLoader().load(pdf_file)

    assert doc.text == "Hello world"
    assert doc.metadata.page_count == 1
    assert doc.metadata.format == "pdf"
```

## Stretch goals

1. **DOCX support**: Implement `DocxLoader` using `python-docx`. Handle documents with tables: extract table cells and serialize using `TableExtractor`. Add a `test_docx_loader` test that mocks `python_docx.Document`.

2. **Async loading**: Refactor `IngestionPipeline` to use `asyncio` for I/O-bound loaders. Implement `AsyncPipelineStage` Protocol with `async def process(...)`. Benchmark: measure wall time for 100 files sync vs async.

3. **Dedup by content hash with persistence**: Implement a `PersistentHashDeduplicator` that saves the seen-hashes set to a `shelve` file between runs. Verify that a second pipeline run on the same directory produces zero new documents (all hashes already seen).

## Evaluation against week rubric

| Dimension | Weight | How it is evaluated |
|---|---|---|
| Independent implementation | 40% | Pipeline runs end-to-end on a fresh directory without consulting the reference solution |
| Conceptual understanding | 25% | Day-7 review answers + assessment Q1–Q5 accuracy |
| Debugging & problem-solving | 20% | Assessment Q6 diagnosis; error isolation in `FileDispatcher` |
| Code quality & testing | 15% | Type hints, `logging`, Pydantic models, ≥ 5 tests passing, no `print`, no hardcoded secrets |

Pass threshold: **75%** weighted score.

Full reference solution: solutions/week-01/project.md
