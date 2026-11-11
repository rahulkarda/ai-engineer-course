# Solution: Lesson 5 — Files API

## Exercise 1: Storage inventory

```python
def get_storage_used(client) -> int:
    """Return total bytes across all uploaded files."""
    return sum(f.size_bytes for f in client.beta.files.list())


# tests/test_lesson5.py
from unittest.mock import MagicMock

def make_file(size: int):
    f = MagicMock()
    f.size_bytes = size
    return f

def test_get_storage_used():
    client = MagicMock()
    client.beta.files.list.return_value = [
        make_file(1_000_000),
        make_file(2_500_000),
        make_file(500_000),
    ]
    assert get_storage_used(client) == 4_000_000

def test_get_storage_used_empty():
    client = MagicMock()
    client.beta.files.list.return_value = []
    assert get_storage_used(client) == 0
```

## Exercise 2: Batch + Files design (structure, no real API calls)

```python
"""
Design: upload once, batch 50 questions.

Key structural points:
  - upload_pdf() called ONCE before batch creation
  - Each Request references the same file_id in a document content block
  - betas=["files-api-2025-04-14"] must be in every Request's params
"""

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


def build_pdf_question_batch(
    file_id: str,
    questions: list[str],
    model: str = "claude-haiku-5-5",
) -> list[Request]:
    """Build a batch of questions about a single uploaded PDF."""
    return [
        Request(
            custom_id=f"q-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": file_id},
                        },
                    ],
                }],
                betas=["files-api-2025-04-14"],
            ),
        )
        for i, question in enumerate(questions)
    ]


# Tests
def test_batch_uses_same_file_id():
    questions = [f"Question {i}" for i in range(50)]
    requests = build_pdf_question_batch("file_abc123", questions)

    assert len(requests) == 50

    for req in requests:
        # Find the document content block
        user_content = req.params["messages"][0]["content"]
        doc_blocks = [b for b in user_content if b.get("type") == "document"]
        assert len(doc_blocks) == 1
        assert doc_blocks[0]["source"]["file_id"] == "file_abc123"


def test_batch_custom_ids_are_unique():
    requests = build_pdf_question_batch("file_xyz", ["Q1", "Q2", "Q3"])
    ids = [r.custom_id for r in requests]
    assert len(set(ids)) == 3
```

## Exercise 3: MIME type validation

```python
from pathlib import Path

SUPPORTED_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".html": "text/html",
    ".md":   "text/markdown",
    ".csv":  "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def allowed_mime(path: Path) -> str | None:
    """Return MIME type for the file extension, or None if unsupported."""
    return SUPPORTED_MIME.get(path.suffix.lower())


import pytest

@pytest.mark.parametrize("ext,expected", [
    (".pdf",  "application/pdf"),
    (".png",  "image/png"),
    (".csv",  "text/csv"),
    (".mp3",  None),
    (".exe",  None),
    (".PDF",  "application/pdf"),  # case-insensitive
])
def test_allowed_mime(ext, expected):
    assert allowed_mime(Path(f"file{ext}")) == expected
```

## Exercise 4: Citation extractor

```python
def extract_citations(response) -> list[dict]:
    """
    Return a list of citation dicts from all text blocks in the response.
    Each dict has: cited_text, document_index.
    """
    citations: list[dict] = []
    for block in response.content:
        if block.type == "text" and hasattr(block, "citations") and block.citations:
            for citation in block.citations:
                citations.append({
                    "cited_text": getattr(citation, "cited_text", ""),
                    "document_index": getattr(citation, "document_index", None),
                })
    return citations


def test_extract_citations_with_two():
    cit1 = MagicMock(); cit1.cited_text = "Revenue was $4.2B"; cit1.document_index = 0
    cit2 = MagicMock(); cit2.cited_text = "Founded in 1976"; cit2.document_index = 0

    block = MagicMock()
    block.type = "text"
    block.text = "The company was founded in 1976 and had revenue of $4.2B."
    block.citations = [cit1, cit2]

    resp = MagicMock()
    resp.content = [block]

    result = extract_citations(resp)
    assert len(result) == 2
    assert result[0]["cited_text"] == "Revenue was $4.2B"
    assert result[1]["cited_text"] == "Founded in 1976"


def test_extract_citations_none():
    block = MagicMock()
    block.type = "text"
    block.citations = None
    resp = MagicMock(); resp.content = [block]
    assert extract_citations(resp) == []
```
