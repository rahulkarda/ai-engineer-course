# Solutions: Lesson 6 — WebSockets, Realtime APIs, and Evaluation

## Exercise 1 — WebSocket server test

```python
import asyncio
import json
import pytest
import websockets


@pytest.mark.asyncio
async def test_ws_server_emits_transcript():
    """Start server on random port, connect client, send audio, assert transcript."""
    import socket
    from voice_pipeline.ws_server import run_server

    # Find a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]

    server_task = asyncio.create_task(run_server("localhost", port))
    await asyncio.sleep(0.1)  # let server start

    received: list[str] = []
    try:
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            # Send 10 audio frames (640 bytes each = 20 ms × 10 = 200 ms audio)
            for _ in range(10):
                await ws.send(b"\x00" * 640)
            await asyncio.sleep(0.3)  # let pipeline process
            await ws.send(json.dumps({"type": "session_end"}))
            try:
                async with asyncio.timeout(1.0):
                    async for msg in ws:
                        if isinstance(msg, str):
                            received.append(msg)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    transcripts = [json.loads(m) for m in received if "transcript" in m]
    assert any(t.get("type") == "transcript" for t in transcripts), (
        f"No transcript received. Got: {received}"
    )
```

## Exercise 2 — compute_wer tests

```python
import pytest
from voice_pipeline.ws_server import compute_wer


def test_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_one_substitution():
    assert compute_wer("hello world", "hello earth") == pytest.approx(0.5)


def test_wer_completely_different():
    wer = compute_wer("hello world", "foo bar baz qux")
    assert wer > 0.9


def test_wer_empty_reference():
    assert compute_wer("", "") == 0.0


def test_wer_deletion():
    # "hello world" vs "hello" — one deletion
    assert compute_wer("hello world", "hello") == pytest.approx(0.5)
```

## Exercise 3 — session_stats control message

```python
# In voice_session(), add to the control message handler:
elif ctrl.get("type") == "get_stats":
    await ws.send(json.dumps({
        "type": "stats",
        "ttfa_ms": record.ttfa_ms,
        "turn_count": session_turn_count,  # add counter to session
    }))
```

Add `session_turn_count: int = 0` to the session scope and increment after each
`handle_turn`.

## Exercise 4 — VoiceAgentBenchmark

```python
from pydantic import BaseModel
from voice_pipeline.ws_server import compute_wer, evaluate_latencies, LatencyRecord


class BenchmarkResult(BaseModel):
    total_turns: int
    mean_wer: float
    ttfa_p50_ms: float
    ttfa_p95_ms: float
    ttfa_max_ms: float


class VoiceAgentBenchmark:
    def __init__(
        self,
        cases: list[tuple[bytes, str]],  # (audio_bytes, expected_transcript)
        stt_provider,
        llm_provider,
        tts_provider,
    ) -> None:
        self._cases = cases
        self._stt = stt_provider
        self._llm = llm_provider
        self._tts = tts_provider

    async def run(self) -> BenchmarkResult:
        import time
        records: list[LatencyRecord] = []
        wers: list[float] = []

        for audio, expected in self._cases:
            t0 = time.monotonic()
            transcript = await self._stt.transcribe(audio)
            stt_done = time.monotonic()

            # Stream LLM
            from voice_pipeline.llm_tts import SentenceBuffer
            buf = SentenceBuffer()
            first_token_ts = None
            first_chunk_ts = None
            response_text = ""
            async for delta in self._llm.stream_tokens():
                if first_token_ts is None:
                    first_token_ts = time.monotonic()
                response_text += delta
                for sentence in buf.feed(delta):
                    chunk = await self._tts.synthesize(sentence)
                    if first_chunk_ts is None:
                        first_chunk_ts = time.monotonic()
            remainder = buf.flush()
            if remainder:
                chunk = await self._tts.synthesize(remainder)
                if first_chunk_ts is None:
                    first_chunk_ts = time.monotonic()

            t_end = time.monotonic()
            rec = LatencyRecord(
                session_id=f"bench_{len(records)}",
                speech_end_ts=t0,
                first_audio_ts=first_chunk_ts,
            )
            rec.compute_ttfa()
            records.append(rec)
            wers.append(compute_wer(expected, transcript.text))

        stats = evaluate_latencies(records)
        return BenchmarkResult(
            total_turns=len(self._cases),
            mean_wer=sum(wers) / len(wers) if wers else 0.0,
            ttfa_p50_ms=stats.get("p50_ms", 0.0),
            ttfa_p95_ms=stats.get("p95_ms", 0.0),
            ttfa_max_ms=stats.get("max_ms", 0.0),
        )
```
