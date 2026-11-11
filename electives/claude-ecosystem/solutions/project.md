# Solution: Capstone Project — Batch Document-Processing Pipeline

## models.py

```python
# pipeline/models.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    doc_id: str
    doc_type: Literal["contract", "report", "paper", "other"]
    entities: list[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(max_length=100)
    risk_score: int = Field(ge=1, le=5)
    error: str | None = None
```

## uploader.py

```python
# pipeline/uploader.py
import logging
from pathlib import Path
import anthropic

log = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def upload_documents(
    paths: list[Path],
    client: anthropic.Anthropic,
) -> dict[str, str]:
    """
    Upload each PDF; return {filename: file_id}.
    Skips files larger than 50 MB with a warning.
    """
    file_id_map: dict[str, str] = {}

    for path in paths:
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            log.warning(
                "Skipping %s — size %d bytes exceeds 50 MB limit", path.name, size
            )
            continue

        with path.open("rb") as f:
            uploaded = client.beta.files.upload(
                file=(path.name, f, "application/pdf"),
            )
        file_id_map[path.name] = uploaded.id
        log.info("Uploaded %s → %s", path.name, uploaded.id)

    return file_id_map
```

## extractor.py

```python
# pipeline/extractor.py
import json
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .models import DocumentMetadata

EXTRACTION_SYSTEM_PROMPT: list[dict] = [
    {
        "type": "text",
        "text": (
            "You are a precise document metadata extractor. "
            "For each document provided, extract structured metadata and return it "
            "as a JSON object that conforms exactly to the following schema:\n\n"
            + json.dumps(DocumentMetadata.model_json_schema(), indent=2)
            + "\n\nRules:\n"
            "- doc_id: use the value provided in the user message\n"
            "- doc_type: classify as one of: contract, report, paper, other\n"
            "- entities: up to 10 named entities (people, orgs, dates)\n"
            "- summary: one sentence, max 100 characters\n"
            "- risk_score: integer 1 (low) to 5 (high)\n"
            "- error: null unless you cannot extract from the document\n"
            "Return only the JSON object, no prose."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


def build_batch_requests(
    file_id_map: dict[str, str],
    model: str = "claude-haiku-5-5",
) -> list[Request]:
    """Build one Request per document."""
    requests: list[Request] = []

    for filename, file_id in file_id_map.items():
        doc_id = Path(filename).stem if "Path" in dir() else filename.rsplit(".", 1)[0]

        requests.append(
            Request(
                custom_id=doc_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=512,
                    output_config={"effort": "low"},
                    system=EXTRACTION_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Extract metadata from this document. doc_id = {doc_id!r}",
                            },
                            {
                                "type": "document",
                                "source": {"type": "file", "file_id": file_id},
                            },
                        ],
                    }],
                    betas=["files-api-2025-04-14"],
                ),
            )
        )

    return requests


from pathlib import Path  # noqa: E402  (module-level import moved here for clarity)
```

## processor.py

```python
# pipeline/processor.py
import json
import logging
import time

import anthropic

from .models import DocumentMetadata

log = logging.getLogger(__name__)


def submit_and_wait(
    requests: list,
    client: anthropic.Anthropic,
    poll_interval_s: int = 30,
) -> str:
    """Submit a batch; poll until ended; return batch_id."""
    batch = client.messages.batches.create(requests=requests)
    log.info("Batch submitted: %s", batch.id)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        log.info(
            "status=%s  processing=%d  succeeded=%d  errored=%d",
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval_s)

    return batch.id


def collect_results(
    batch_id: str,
    client: anthropic.Anthropic,
) -> tuple[list[DocumentMetadata], list[str]]:
    """
    Stream batch results.
    Returns (successful_results, failed_custom_ids).
    """
    successful: list[DocumentMetadata] = []
    failed: list[str] = []

    for item in client.messages.batches.results(batch_id):
        if item.result.type != "succeeded":
            log.warning("Item %s failed: %s", item.custom_id, item.result.type)
            failed.append(item.custom_id)
            continue

        msg = item.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")

        try:
            data = json.loads(text)
            metadata = DocumentMetadata(**data)
            successful.append(metadata)
        except Exception as exc:
            log.warning("Parse/validation error for %s: %s", item.custom_id, exc)
            failed.append(item.custom_id)

    return successful, failed
```

## tracker.py

```python
# pipeline/tracker.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field

import anthropic

log = logging.getLogger(__name__)

MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"input": 4.00,  "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00,  "output": 10.00},
    "claude-haiku-5-5":  {"input": 0.10,  "output": 0.50},
}


@dataclass
class CostTracker:
    model: str
    total_input_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    _call_costs: list[float] = field(default_factory=list)

    def record(self, usage: anthropic.types.Usage) -> float:
        self.total_input_tokens += usage.input_tokens
        self.total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.total_output_tokens += usage.output_tokens
        self.call_count += 1
        cost = self._calculate_cost(usage)
        self._call_costs.append(cost)
        return cost

    def _calculate_cost(self, usage) -> float:
        prices = MODEL_PRICES.get(self.model, MODEL_PRICES["claude-opus-5-5"])
        m = 1_000_000
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        return (
            usage.input_tokens * prices["input"] / m
            + cache_write * prices["input"] * 1.25 / m
            + cache_read * prices["input"] * 0.10 / m
            + usage.output_tokens * prices["output"] / m
        )

    @property
    def total_cost(self) -> float:
        return sum(self._call_costs)

    @property
    def cache_hit_rate(self) -> float:
        total = (
            self.total_input_tokens
            + self.total_cache_read_tokens
            + self.total_cache_write_tokens
        )
        return (self.total_cache_read_tokens / total) if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "cache_write_tokens": self.total_cache_write_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cache_hit_rate_pct": round(self.cache_hit_rate * 100, 1),
        }


def record_batch_results(
    batch_id: str,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-5-5",
) -> dict:
    """Retrieve batch object; return a cost summary dict."""
    batch = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts

    # Batch API does not return per-request usage in the batch object;
    # estimate based on counts and average token assumptions.
    estimated_input = counts.succeeded * 350   # system prompt (~200) + doc ref + question
    estimated_output = counts.succeeded * 150  # structured JSON output

    tracker = CostTracker(model=model)

    class _FakeUsage:
        def __init__(self):
            self.input_tokens = estimated_input
            self.output_tokens = estimated_output
            self.cache_creation_input_tokens = estimated_input * 3 // 4
            self.cache_read_input_tokens = 0

    if counts.succeeded > 0:
        tracker.record(_FakeUsage())

    return {
        "batch_id": batch_id,
        "total_requests": counts.succeeded + counts.errored + counts.processing,
        "succeeded": counts.succeeded,
        "errored": counts.errored,
        **tracker.summary(),
    }
```

## main.py

```python
# pipeline/main.py
import json
import logging
from pathlib import Path

import anthropic

from .uploader import upload_documents
from .extractor import build_batch_requests
from .processor import submit_and_wait, collect_results
from .tracker import record_batch_results

log = logging.getLogger(__name__)


def run_pipeline(
    doc_dir: Path,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> dict:
    """Full pipeline: discover → upload → batch → collect → write results."""
    pdf_paths = sorted(doc_dir.glob("*.pdf"))
    if not pdf_paths:
        log.warning("No PDF files found in %s", doc_dir)
        return {"error": "No PDFs found"}

    log.info("Found %d PDF files", len(pdf_paths))

    file_id_map = upload_documents(pdf_paths, client)
    requests = build_batch_requests(file_id_map)

    if dry_run:
        log.info("DRY RUN — %d requests would be submitted", len(requests))
        return {"dry_run": True, "request_count": len(requests)}

    batch_id = submit_and_wait(requests, client)
    results, failed = collect_results(batch_id, client)

    output_path = doc_dir / "results.jsonl"
    with output_path.open("w") as f:
        for meta in results:
            f.write(meta.model_dump_json() + "\n")

    log.info(
        "Written %d results to %s (%d failed)", len(results), output_path, len(failed)
    )

    summary = record_batch_results(batch_id, client)
    summary["failed_ids"] = failed
    return summary
```

## tests/conftest.py

```python
# tests/conftest.py
from unittest.mock import MagicMock
import pytest

from pipeline.models import DocumentMetadata


@pytest.fixture
def mock_client():
    client = MagicMock()

    # files.upload returns an object with .id and .size_bytes
    upload_result = MagicMock()
    upload_result.id = "file_test123"
    upload_result.size_bytes = 1_024
    client.beta.files.upload.return_value = upload_result

    # files.list returns empty iterator by default
    client.beta.files.list.return_value = []

    # batches.create returns a batch object
    batch_obj = MagicMock()
    batch_obj.id = "batch_test456"
    batch_obj.processing_status = "ended"
    client.messages.batches.create.return_value = batch_obj

    # batches.retrieve returns ended status
    retrieve_obj = MagicMock()
    retrieve_obj.processing_status = "ended"
    retrieve_obj.request_counts = MagicMock(
        succeeded=1, errored=0, expired=0, processing=0
    )
    client.messages.batches.retrieve.return_value = retrieve_obj

    # batches.results returns empty iterator by default
    client.messages.batches.results.return_value = []

    return client


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        doc_id="report_2024",
        doc_type="report",
        entities=["ACME Corp", "Q4 2024"],
        summary="Quarterly financial report showing 15% revenue growth.",
        risk_score=2,
    )


@pytest.fixture
def sample_batch_result(sample_metadata):
    """A mocked BetaMessageBatchIndividualResponse for a succeeded result."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = sample_metadata.model_dump_json()

    message = MagicMock()
    message.content = [text_block]

    result = MagicMock()
    result.type = "succeeded"
    result.message = message

    item = MagicMock()
    item.custom_id = "report_2024"
    item.result = result

    return item
```

## tests/test_uploader.py

```python
# tests/test_uploader.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import tempfile

from pipeline.uploader import upload_documents, MAX_FILE_SIZE_BYTES


def test_upload_returns_file_id_map(mock_client, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 sample")

    result = upload_documents([pdf], mock_client)

    assert result == {"doc.pdf": "file_test123"}
    mock_client.beta.files.upload.assert_called_once()


def test_upload_skips_large_files(mock_client, tmp_path):
    large_pdf = tmp_path / "huge.pdf"
    # Create file with size just over 50 MB
    large_pdf.write_bytes(b"\x00" * (MAX_FILE_SIZE_BYTES + 1))

    result = upload_documents([large_pdf], mock_client)

    assert result == {}
    mock_client.beta.files.upload.assert_not_called()
```

## tests/test_extractor.py

```python
# tests/test_extractor.py
from pipeline.extractor import build_batch_requests, EXTRACTION_SYSTEM_PROMPT


def test_builds_correct_request_count():
    file_id_map = {"a.pdf": "fid_1", "b.pdf": "fid_2", "c.pdf": "fid_3"}
    requests = build_batch_requests(file_id_map)
    assert len(requests) == 3


def test_cache_control_in_system():
    blocks = EXTRACTION_SYSTEM_PROMPT
    cached = [b for b in blocks if b.get("cache_control") == {"type": "ephemeral"}]
    assert len(cached) == 1


def test_custom_id_is_filename_stem():
    file_id_map = {"report_2024.pdf": "fid_abc"}
    requests = build_batch_requests(file_id_map)
    assert requests[0].custom_id == "report_2024"
```

## tests/test_processor.py

```python
# tests/test_processor.py
import json
from unittest.mock import MagicMock, call

from pipeline.processor import submit_and_wait, collect_results
from pipeline.models import DocumentMetadata


def test_submit_and_wait_polls_until_ended(mock_client):
    in_progress = MagicMock()
    in_progress.processing_status = "in_progress"
    in_progress.request_counts = MagicMock(processing=5, succeeded=0, errored=0)

    ended = MagicMock()
    ended.processing_status = "ended"
    ended.request_counts = MagicMock(processing=0, succeeded=5, errored=0)

    mock_client.messages.batches.retrieve.side_effect = [in_progress, in_progress, ended]

    batch_id = submit_and_wait([], mock_client, poll_interval_s=0)
    assert mock_client.messages.batches.retrieve.call_count == 3


def test_collect_results_parses_json(mock_client, sample_batch_result):
    mock_client.messages.batches.results.return_value = [sample_batch_result]

    results, failed = collect_results("batch_abc", mock_client)

    assert len(results) == 1
    assert isinstance(results[0], DocumentMetadata)
    assert results[0].doc_id == "report_2024"
    assert failed == []


def test_collect_results_handles_invalid_json(mock_client):
    text_block = MagicMock(); text_block.type = "text"; text_block.text = "not json"
    message = MagicMock(); message.content = [text_block]
    result = MagicMock(); result.type = "succeeded"; result.message = message
    item = MagicMock(); item.custom_id = "bad_doc"; item.result = result

    mock_client.messages.batches.results.return_value = [item]

    results, failed = collect_results("batch_abc", mock_client)
    assert results == []
    assert "bad_doc" in failed
```

## tests/test_tracker.py

```python
# tests/test_tracker.py
from unittest.mock import MagicMock
from pipeline.tracker import CostTracker


def make_usage(inp, out, write=0, read=0):
    u = MagicMock()
    u.input_tokens = inp
    u.output_tokens = out
    u.cache_creation_input_tokens = write
    u.cache_read_input_tokens = read
    return u


def test_calculate_cost_haiku():
    tracker = CostTracker(model="claude-haiku-5-5")
    usage = make_usage(1_000, 100)
    cost = tracker.record(usage)
    expected = 1_000 * 0.10 / 1e6 + 100 * 0.50 / 1e6
    assert abs(cost - expected) < 1e-12


def test_cache_hit_rate():
    tracker = CostTracker(model="claude-haiku-5-5")
    usage = make_usage(inp=200, out=50, write=800, read=800)
    tracker.record(usage)
    # cache_read / (input + cache_read + cache_write) = 800 / (200+800+800) = ~44.4%
    expected = 800 / (200 + 800 + 800)
    assert abs(tracker.cache_hit_rate - expected) < 1e-6


def test_summary_keys():
    tracker = CostTracker(model="claude-haiku-5-5")
    tracker.record(make_usage(100, 50))
    s = tracker.summary()
    for key in ("calls", "total_input_tokens", "total_output_tokens",
                "total_cost_usd", "cache_hit_rate_pct"):
        assert key in s, f"Missing key: {key}"
```
