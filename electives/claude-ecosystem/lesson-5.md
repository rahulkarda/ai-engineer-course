# Elective: Claude Ecosystem · Lesson 5 — Files API

> Objective: Upload files once and reuse them across multiple API calls for vision and PDF analysis.
> Time: ~1 hr · Prereqs: [[lesson-1.md]]

## Concept

Without the Files API, every request that involves a PDF or image must re-encode and
re-upload the full binary payload as base64. For a 2 MB PDF queried 20 times, that's
40 MB of redundant data in your requests.

The Files API lets you upload once and reference by `file_id` in subsequent messages.
This improves request latency, simplifies code, and is the foundation for the capstone project.

**File storage facts:**
- Max file size: 500 MB per file.
- Total storage: 100 GB per organization.
- Files persist until explicitly deleted.
- Upload/list/delete operations: free. Content used in messages: billed as input tokens.
- Not available on Amazon Bedrock or Google Vertex AI.

**Supported MIME types for documents:**
`application/pdf`, `text/plain`, `text/html`, `text/markdown`, `text/csv`,
`application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`

**Supported MIME types for images:**
`image/jpeg`, `image/png`, `image/gif`, `image/webp`

## How it works internally

Upload returns a `file_id` (e.g. `"file_01AbCdEf..."`). You reference it as a content
block inside a `messages` request:

```python
# For documents (PDF, text, etc.)
{"type": "document", "source": {"type": "file", "file_id": file_id}}

# For images
{"type": "image", "source": {"type": "file", "file_id": file_id}}
```

The content-block `type` must match the file's MIME type — `document` for PDFs and text,
`image` for raster images.

**Citations:** Add `"citations": {"enabled": True}` to a `document` block to get
inline citations pointing back to the source document. Incompatible with structured output.

**Note on SDK namespaces:** The Files API is stable (out of beta). Current SDK versions
may show `client.beta.files` — this refers to the same stable API; treat it as the
correct path until the SDK updates to `client.files`.

## Worked example

```python
# lesson_5_demo.py
import logging
import os
import tempfile
from pathlib import Path

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def upload_pdf(path: Path) -> str:
    """Upload a PDF and return its file_id."""
    with path.open("rb") as f:
        uploaded = client.beta.files.upload(
            file=(path.name, f, "application/pdf"),
        )
    log.info("Uploaded %s → %s (%d bytes)", path.name, uploaded.id, uploaded.size_bytes)
    return uploaded.id


def upload_image(path: Path) -> str:
    """Upload an image and return its file_id."""
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_map.get(path.suffix.lower(), "image/png")

    with path.open("rb") as f:
        uploaded = client.beta.files.upload(
            file=(path.name, f, mime),
        )
    log.info("Uploaded image %s → %s", path.name, uploaded.id)
    return uploaded.id


def query_pdf(file_id: str, question: str) -> str:
    """Ask a question about an uploaded PDF."""
    response = client.beta.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                    "citations": {"enabled": True},
                },
            ],
        }],
        betas=["files-api-2025-04-14"],
    )
    return next(b.text for b in response.content if b.type == "text")


def query_pdf_multi(file_id: str, questions: list[str]) -> list[str]:
    """
    Ask multiple questions about the same uploaded PDF.
    Re-uses the file_id — no re-upload needed.
    """
    answers: list[str] = []
    for question in questions:
        answer = query_pdf(file_id, question)
        answers.append(answer)
        log.info("Q: %s | A: %s", question[:60], answer[:80])
    return answers


def describe_image(file_id: str) -> str:
    """Ask Claude to describe an uploaded image."""
    response = client.beta.messages.create(
        model="claude-haiku-5-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": file_id},
                },
            ],
        }],
        betas=["files-api-2025-04-14"],
    )
    return next(b.text for b in response.content if b.type == "text")


def list_uploaded_files() -> list[dict]:
    """List all files in the organisation's storage."""
    files = []
    for f in client.beta.files.list():
        files.append({
            "id": f.id,
            "filename": f.filename,
            "size_bytes": f.size_bytes,
        })
    return files


def delete_file(file_id: str) -> None:
    """Delete a file from storage."""
    client.beta.files.delete(file_id)
    log.info("Deleted file %s", file_id)


def demo_pdf_pipeline() -> None:
    """Create a temp PDF-like file and run the full upload → query → delete cycle."""
    # In production: use a real PDF from disk.
    # Here we use a text file to demonstrate the API flow without a real PDF binary.
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tmp.write("Annual Revenue Report 2024\nTotal revenue: $4.2 billion\n")
        tmp_path = Path(tmp.name)

    try:
        # Upload once
        file_id = client.beta.files.upload(
            file=(tmp_path.name, tmp_path.open("rb"), "text/plain"),
        ).id
        log.info("file_id: %s", file_id)

        # Query multiple times with the same file_id
        questions = [
            "What is the total revenue mentioned?",
            "What year does this report cover?",
        ]
        for q in questions:
            response = client.beta.messages.create(
                model="claude-haiku-5-5",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {"type": "document", "source": {"type": "file", "file_id": file_id}},
                    ],
                }],
                betas=["files-api-2025-04-14"],
            )
            answer = next(b.text for b in response.content if b.type == "text")
            log.info("Q: %s | A: %s", q, answer)

        # Clean up
        delete_file(file_id)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    demo_pdf_pipeline()
```

Key observations:
- `client.beta.files.upload()` takes a `(filename, fileobj, mime_type)` tuple.
- The `betas=["files-api-2025-04-14"]` header is required in current SDK versions.
- Content-block `type` must match the file's MIME class (`document` vs `image`).
- Delete files when done to stay within storage limits.

## Exercises

1. **File inventory.** Write a function `get_storage_used() -> int` that sums `size_bytes`
   across all uploaded files. Test it with a mocked file list.

2. **Batch + Files.** Design (no real API calls) a function that:
   (a) uploads a PDF once,
   (b) creates a batch of 50 questions about that PDF,
   (c) collects and returns the answers.
   Write the structure with mocked clients; focus on correct content block structure.

3. **MIME type validation.** Write a function `allowed_mime(path: Path) -> str | None`
   that returns the correct MIME type for supported file extensions or `None` for
   unsupported ones. Test with `.pdf`, `.png`, `.csv`, `.mp3` (unsupported).

4. **Citation extractor.** Given a mocked response with `citations` on text blocks,
   write a function `extract_citations(response) -> list[dict]` that returns each
   citation's `cited_text` and `document_index`. Mock a response with two citations.

(Solutions: solutions/lesson-5.md)

## Self-check

- [ ] I can upload a PDF and an image using the Files API.
- [ ] I can reference a `file_id` in both `document` and `image` content blocks.
- [ ] I know the difference in content block type for PDF vs. image.
- [ ] I can list and delete files to manage storage.
- [ ] I understand how the Files API combines with prompt caching and batches.

## Supplementary links

- [Files API guide](https://docs.anthropic.com/en/docs/build-with-claude/files)
- [Vision guide](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [PDF support](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support)
