# Elective: Claude Ecosystem · Capstone Project

## Cost-Optimised Batch Document-Processing Pipeline

> Objective: Build a production-grade pipeline that processes large document corpora
> using the Batches API + prompt caching + Files API, with full token/cost tracking.
> Time: ~4 hr · Prereqs: All 7 lessons

---

## Problem statement

Your team has 500 PDF documents (contracts, research papers, incident reports) that need
to be processed nightly to extract structured metadata:

- **Document type** (contract / report / paper / other)
- **Key entities** (people, organisations, dates)
- **One-sentence summary**
- **Risk score** (1–5 integer)

Constraints:
- Processing must cost under $2.00 per nightly run.
- Each run must complete within 2 hours.
- Output must be deterministic — the same document always produces the same schema.
- All API calls must be mockable for CI.

---

## Architecture

```
documents/
  doc_001.pdf
  doc_002.pdf
  ...

pipeline/
  uploader.py      # Upload documents via Files API; return file_id map
  extractor.py     # Build batch requests with caching + structured output
  processor.py     # Submit batch, poll, collect results
  tracker.py       # Accumulate usage + compute cost
  models.py        # Pydantic output schema
  main.py          # Orchestrate the full pipeline

tests/
  conftest.py      # Shared mocks and fixtures
  test_uploader.py
  test_extractor.py
  test_processor.py
  test_tracker.py
```

---

## Specifications

### `models.py` — Output schema

Define a `DocumentMetadata` Pydantic model with:
- `doc_id: str` — matches the `custom_id` from the batch request
- `doc_type: Literal["contract", "report", "paper", "other"]`
- `entities: list[str]` — up to 10 named entities
- `summary: str` — one sentence, max 100 characters
- `risk_score: int` — integer 1–5 (use `Field(ge=1, le=5)`)
- `error: str | None = None` — populated if extraction failed

### `uploader.py`

```python
def upload_documents(paths: list[Path], client: Anthropic) -> dict[str, str]:
    """
    Upload each PDF; return {filename: file_id}.
    Skip files larger than 50 MB.
    Log a warning for skipped files.
    """
```

Implementation requirements:
- Use `client.beta.files.upload()` with `betas=["files-api-2025-04-14"]`.
- Return a `dict[str, str]` mapping `path.name` to `uploaded.id`.
- If a file exceeds 50 MB, log a warning and skip it (don't raise).

### `extractor.py`

```python
EXTRACTION_SYSTEM_PROMPT: str  # Stable shared prompt with cache_control

def build_batch_requests(
    file_id_map: dict[str, str],
    model: str = "claude-haiku-5-5",
) -> list[Request]:
    """
    Build one Request per document.
    custom_id = filename (without extension).
    System prompt carries cache_control for cross-request reuse.
    Output must conform to DocumentMetadata JSON schema.
    """
```

Implementation requirements:
- `EXTRACTION_SYSTEM_PROMPT` must be a list with a single text block carrying
  `"cache_control": {"type": "ephemeral"}`.
- Each `Request` must include:
  - `model="claude-haiku-5-5"` (cheapest adequate tier for extraction)
  - `max_tokens=512`
  - `output_config={"effort": "low"}` (classification task — low effort suffices)
  - A `document` content block referencing `file_id` with `betas=["files-api-2025-04-14"]`
  - A user text block asking for JSON output matching `DocumentMetadata`

### `processor.py`

```python
def submit_and_wait(
    requests: list[Request],
    client: Anthropic,
    poll_interval_s: int = 30,
) -> str:
    """Submit a batch; poll until ended; return batch_id."""

def collect_results(
    batch_id: str,
    client: Anthropic,
) -> tuple[list[DocumentMetadata], list[str]]:
    """
    Stream batch results.
    Returns (successful_results, failed_custom_ids).
    Parse JSON from the text block; validate against DocumentMetadata.
    On parse/validation error: append custom_id to failed list.
    """
```

### `tracker.py`

Reuse/extend `CostTracker` from lesson 6. Add:

```python
def record_batch_results(
    batch_id: str,
    results: list[DocumentMetadata],
    client: Anthropic,
) -> dict:
    """
    Retrieve the batch object; read request_counts and any available usage.
    Return a summary dict with:
      total_requests, succeeded, errored,
      estimated_cost_usd, cache_hit_rate_pct
    """
```

### `main.py`

```python
def run_pipeline(
    doc_dir: Path,
    client: Anthropic,
    dry_run: bool = False,
) -> dict:
    """
    Full pipeline:
      1. Discover all .pdf files in doc_dir.
      2. Upload via uploader.upload_documents().
      3. Build batch requests via extractor.build_batch_requests().
      4. If dry_run: print request count + estimated token count; return early.
      5. Submit and wait via processor.submit_and_wait().
      6. Collect results via processor.collect_results().
      7. Track cost via tracker.record_batch_results().
      8. Write results to doc_dir / "results.jsonl" (one JSON object per line).
      9. Return summary dict from tracker.
    """
```

---

## Tests (all API calls mocked)

### `conftest.py`

Provide pytest fixtures:
- `mock_client` — `MagicMock` with mocked `beta.files.upload`, `messages.batches.create`,
  `messages.batches.retrieve`, `messages.batches.results`.
- `sample_metadata` — a valid `DocumentMetadata` instance for use in other tests.
- `sample_batch_result` — a mocked `BetaMessageBatchIndividualResponse` with `type="succeeded"`.

### `test_uploader.py`

- `test_upload_returns_file_id_map`: mock `client.beta.files.upload` to return an object
  with `.id = "file_abc"`. Call `upload_documents([Path("doc.pdf")], mock_client)`.
  Assert result is `{"doc.pdf": "file_abc"}`.
- `test_upload_skips_large_files`: mock a 60 MB file. Assert it's skipped and the
  result dict is empty.

### `test_extractor.py`

- `test_builds_correct_request_count`: given 3 file IDs, assert `build_batch_requests`
  returns 3 `Request` objects.
- `test_cache_control_in_system`: assert that the system prompt list contains exactly one
  block with `"cache_control": {"type": "ephemeral"}`.
- `test_custom_id_is_filename_stem`: assert `custom_id` for `"report_2024.pdf"` is
  `"report_2024"`.

### `test_processor.py`

- `test_submit_and_wait_polls_until_ended`: mock `retrieve` to return `"in_progress"` twice,
  then `"ended"`. Assert `submit_and_wait` called `retrieve` 3 times.
- `test_collect_results_parses_json`: mock a succeeded result whose text block contains
  valid `DocumentMetadata` JSON. Assert returned list has one `DocumentMetadata` object.
- `test_collect_results_handles_invalid_json`: mock a succeeded result with malformed JSON.
  Assert it goes to `failed_custom_ids`, not the results list.

### `test_tracker.py`

- `test_calculate_cost_haiku`: assert cost for 1000 input + 100 output tokens on Haiku 5.5
  matches the expected value ($0.0001 + $0.00005 = $0.00015).
- `test_cache_hit_rate`: assert `cache_hit_rate` is correct for a mix of cached and uncached tokens.
- `test_summary_keys`: assert `CostTracker.summary()` contains the expected keys.

---

## Acceptance criteria

- [ ] `pytest tests/` passes with all API calls mocked (no real key required in CI).
- [ ] `main.py --dry-run` prints estimated token count and exits without API calls.
- [ ] Results written to `results.jsonl` validate against `DocumentMetadata` for every row.
- [ ] `CostTracker.summary()` shows `cache_hit_rate_pct > 0` when the shared system prompt
      is reused across batch items (verify in the mocked integration test).
- [ ] No hardcoded API keys anywhere. All secrets via `os.getenv("ANTHROPIC_API_KEY")`.
- [ ] `ruff check` passes on all source files.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|-----------|--------|
| Pipeline correctness (all stages integrate; results.jsonl is valid) | 35% |
| Test quality (all mocks, edge cases covered, no real API calls in tests) | 25% |
| Cost design (caching + batch used correctly; estimated savings documented) | 20% |
| Code quality (type hints, logging, Pydantic schemas, no god files) | 20% |

Answer key: solutions/project.md

---

## Extension challenges (optional)

1. **Retry logic.** Add exponential-backoff retry for `errored` batch items (up to 3 attempts).
2. **Incremental processing.** Track which `file_id`s were processed in a previous run
   (persisted in a `state.json`) and skip them on re-runs.
3. **Cost alerting.** If estimated cost exceeds a configurable threshold before submitting
   the batch, log a warning and require `--force` to continue.
4. **Cross-provider deployment sketch.** Write a `make_client(provider: str) -> Anthropic`
   factory function that returns:
   - `Anthropic()` for `"anthropic"`,
   - `AnthropicBedrockMantle(aws_region=os.getenv("AWS_REGION"))` for `"bedrock"`,
   - `AnthropicVertex(project_id=..., region=...)` for `"vertex"`.
   Note any features (Files API, Batches API) unavailable on each provider.
