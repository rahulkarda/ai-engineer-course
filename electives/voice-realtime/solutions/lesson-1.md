# Solutions: Lesson 1 — Pipeline Anatomy & Latency Budgets

## Exercise 1 — Interim STT transcripts

```python
async def mock_stt_stage_with_interim(
    speech_in: asyncio.Queue[AudioChunk],
    text_out: asyncio.Queue[Transcript],
) -> None:
    """Emit interim transcripts every 2 frames, final every 6."""
    buffer: list[AudioChunk] = []
    utterance = 0
    interim_count = 0
    while True:
        chunk = await speech_in.get()
        buffer.append(chunk)
        if len(buffer) % 2 == 0 and len(buffer) < 6:
            interim_count += 1
            await text_out.put(
                Transcript(text=f"interim {interim_count}", is_final=False)
            )
        if len(buffer) >= 6:
            utterance += 1
            await text_out.put(
                Transcript(text=f"mock utterance {utterance}", is_final=True)
            )
            buffer.clear()
        speech_in.task_done()


async def mock_llm_stage_final_only(
    text_in: asyncio.Queue[Transcript],
    token_out: asyncio.Queue[LLMToken],
) -> None:
    """Only process final transcripts."""
    while True:
        transcript = await text_in.get()
        if transcript.is_final:
            words = ["This", " is", " a", " mock", " response."]
            for i, word in enumerate(words):
                await asyncio.sleep(0.02)
                await token_out.put(LLMToken(text=word, is_final=(i == len(words) - 1)))
        text_in.task_done()
```

## Exercise 2 — Latency tracker

```python
async def latency_tracker(audio_out: asyncio.Queue[AudioOutput]) -> None:
    while True:
        audio = await audio_out.get()
        # The 'first frame' timestamp is embedded in AudioOutput.timestamp
        # In a real system, propagate the first AudioChunk.timestamp through the chain.
        log.info(
            "E2E latency: %.0f ms (audio ts=%.3f)",
            (time.monotonic() - audio.timestamp) * 1000,
            audio.timestamp,
        )
        audio_out.task_done()
```

Note: for accurate E2E measurement, the timestamp must be captured at the first
audio frame (in `audio_source`) and propagated through all dataclasses to `AudioOutput`.

## Exercise 3 — Back-pressure observation

With `q_tokens = asyncio.Queue(maxsize=2)` and TTS sleep `0.001 s`:
- TTS finishes fast → queue stays near-empty → no back-pressure.
- Swap: increase TTS sleep to `0.1 s`, keep `maxsize=2`:
  - `mock_tts_stage` is slow → `q_tokens` fills to 2 → `mock_llm_stage` blocks on
    `await token_out.put(...)` → LLM pauses.
- This is desirable: prevents OOM from unbounded queue growth during slow consumers.

## Exercise 4 — PipelineMetrics

```python
from pydantic import BaseModel

class PipelineMetrics(BaseModel):
    total_utterances: int = 0
    avg_ttfa_ms: float = 0.0
    max_ttfa_ms: float = 0.0

    def update(self, ttfa_ms: float) -> None:
        self.max_ttfa_ms = max(self.max_ttfa_ms, ttfa_ms)
        # Running average
        self.avg_ttfa_ms = (
            (self.avg_ttfa_ms * self.total_utterances + ttfa_ms)
            / (self.total_utterances + 1)
        )
        self.total_utterances += 1


async def run_pipeline_with_metrics(n_chunks: int = 20) -> PipelineMetrics:
    metrics = PipelineMetrics()
    # ... (same queue setup as run_pipeline) ...
    # In audio_player, compute TTFA using propagated timestamps and call metrics.update(ttfa_ms)
    return metrics
```
