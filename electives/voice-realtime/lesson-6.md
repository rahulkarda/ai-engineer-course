# Elective: Voice & Realtime — Lesson 6
# WebSockets, Realtime APIs, and Evaluation

> Objective: Connect the voice pipeline to a WebSocket transport; measure and evaluate latency and quality.  
> Time: ~60 min  
> Prereqs: [[lesson-5.md]]

---

## Concept

### Why WebSockets?

HTTP is request-response: one side sends, the other replies. Voice agents need
**bidirectional, low-latency, persistent** connections — audio flows from client
to server AND server to client simultaneously. WebSockets provide exactly this:

- Single TCP connection, full-duplex.
- Binary frames for efficient audio transport.
- Text frames for control messages (JSON).
- ~0.1 ms framing overhead vs. HTTP headers.

### WebSocket voice protocol design

A minimal protocol for a voice agent WebSocket server:

```
Client → Server (binary):  raw PCM audio frames (16-bit, 16 kHz)
Client → Server (text):    JSON control {"type": "interrupt"} | {"type": "session_end"}
Server → Client (binary):  TTS audio chunks (PCM or MP3)
Server → Client (text):    JSON events:
    {"type": "transcript", "text": "...", "is_final": true}
    {"type": "llm_token", "text": "..."}
    {"type": "speaking_start"}
    {"type": "speaking_end"}
    {"type": "error", "message": "..."}
```

### Realtime API landscape (2025)

Several providers now offer end-to-end voice WebSocket APIs that skip the
STT→LLM→TTS pipeline you build yourself:

| Provider | API | Notes |
|----------|-----|-------|
| OpenAI Realtime API | `wss://api.openai.com/v1/realtime` | GPT-4o-realtime; audio in/out |
| Google Multimodal Live | `wss://generativelanguage.googleapis.com/ws/...` | Gemini; audio/video |
| Ultravox (Fixie AI) | gRPC + WebSocket | open-weight; self-hostable |

Trade-offs vs. building your own:
- **Provider API**: lower latency (~300 ms TTFA possible), less control, higher cost per minute, vendor lock-in.
- **Custom pipeline**: full control over STT/LLM/TTS choices, auditable, locally testable, more moving parts.

---

## How it works internally

### websockets library

```python
import websockets

# Server
async def handler(ws):
    async for message in ws:
        if isinstance(message, bytes):
            # audio frame
            await ws.send(b"audio_response")
        else:
            # control JSON
            data = json.loads(message)

async with websockets.serve(handler, "localhost", 8765):
    await asyncio.Future()  # run forever

# Client
async with websockets.connect("ws://localhost:8765") as ws:
    await ws.send(b"\x00" * 640)  # send audio frame
    response = await ws.recv()
```

### Framing strategy

Audio frames should be sent at the capture rate — don't buffer 1 second of audio
before sending, or you defeat the streaming purpose. At 20 ms frames:
640 bytes × 50 frames/sec = 32 KB/sec upstream. Well within typical WebSocket limits.

### Evaluation metrics

| Metric | How to measure | Target |
|--------|---------------|--------|
| TTFA (Time To First Audio) | `t(first_audio_chunk) - t(speech_end)` | ≤ 1 s |
| Word Error Rate (WER) | `jiwer.wer(reference, hypothesis)` | ≤ 5% for clear speech |
| Turn detection accuracy | precision/recall on turn-end events | ≥ 95% |
| Barge-in latency | `t(barge_in_detected) - t(speech_start)` | ≤ 500 ms |
| LLM first-token latency | stream event timing | ≤ 600 ms |
| TTS naturalness | MOS (Mean Opinion Score) 1–5, human eval | ≥ 3.5 |

---

## Worked example

```python
"""
voice_pipeline/ws_server.py
WebSocket voice-agent server using the mock pipeline.
No real audio/API required — connects mock STT, LLM, TTS.
"""
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import websockets
from websockets.server import WebSocketServerProtocol

from voice_pipeline.barge_in import BargInCoordinator, PipelineConfig
from voice_pipeline.llm_tts import MockLLMProvider, MockTTSProvider, SentenceBuffer
from voice_pipeline.stt import MockSTTProvider, Transcript
from voice_pipeline.vad import MockVADBackend, TurnDetector

log = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    session_id: str
    speech_end_ts: float
    first_audio_ts: float | None = None
    ttfa_ms: float | None = None
    timestamps: dict[str, float] = field(default_factory=dict)

    def mark(self, event: str) -> None:
        self.timestamps[event] = time.monotonic()

    def compute_ttfa(self) -> None:
        if self.first_audio_ts:
            self.ttfa_ms = (self.first_audio_ts - self.speech_end_ts) * 1000


@dataclass
class SessionConfig:
    session_id: str
    echo_suppression: bool = True
    max_response_tokens: int = 200


async def voice_session(
    ws: WebSocketServerProtocol,
    config: SessionConfig,
) -> None:
    """Handle one WebSocket voice session."""
    coord = BargInCoordinator(PipelineConfig(echo_suppression=config.echo_suppression))
    vad = TurnDetector(MockVADBackend(), silence_end_ms=600)
    stt = MockSTTProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider(latency_ms=30)
    record = LatencyRecord(
        session_id=config.session_id,
        speech_end_ts=time.monotonic(),
    )

    log.info("Session %s started", config.session_id)
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # Audio frame from client
                segments = vad.feed(message)
                for segment in segments:
                    # Turn detected — run STT
                    record.speech_end_ts = time.monotonic()
                    record.mark("speech_end")
                    transcript = await stt.transcribe(segment.as_bytes())
                    if not transcript.text.strip():
                        continue

                    record.mark("stt_done")
                    await ws.send(json.dumps({
                        "type": "transcript",
                        "text": transcript.text,
                        "is_final": True,
                    }))

                    # Check for barge-in signal from client before LLM
                    if coord.barge_in_event.is_set():
                        coord.reset_for_new_turn()
                        continue

                    # Stream LLM response → TTS → client
                    coord.agent_started_speaking()
                    buf = SentenceBuffer()
                    first_audio = True
                    async for delta in llm.stream_tokens():
                        if coord.barge_in_event.is_set():
                            log.info("Session %s: barge-in during LLM", config.session_id)
                            break
                        for sentence in buf.feed(delta):
                            chunk = await tts.synthesize(sentence)
                            if first_audio:
                                record.first_audio_ts = time.monotonic()
                                record.compute_ttfa()
                                first_audio = False
                                await ws.send(json.dumps({"type": "speaking_start"}))
                            await ws.send(chunk.data)  # binary audio

                    remainder = buf.flush()
                    if remainder and not coord.barge_in_event.is_set():
                        chunk = await tts.synthesize(remainder)
                        if first_audio:
                            record.first_audio_ts = time.monotonic()
                            record.compute_ttfa()
                        await ws.send(chunk.data)

                    coord.agent_stopped_speaking()
                    coord.reset_for_new_turn()
                    await ws.send(json.dumps({"type": "speaking_end"}))
                    log.info(
                        "Session %s: TTFA=%.0f ms",
                        config.session_id,
                        record.ttfa_ms or -1,
                    )

            elif isinstance(message, str):
                # Control message
                try:
                    ctrl = json.loads(message)
                    if ctrl.get("type") == "interrupt":
                        coord.barge_in_event.set()
                        log.info("Session %s: client interrupt", config.session_id)
                    elif ctrl.get("type") == "session_end":
                        log.info("Session %s: client ended session", config.session_id)
                        break
                except json.JSONDecodeError:
                    log.warning("Session %s: invalid control message", config.session_id)
    except websockets.ConnectionClosed:
        log.info("Session %s: connection closed", config.session_id)
    finally:
        log.info("Session %s: finished", config.session_id)


async def run_server(host: str = "localhost", port: int = 8765) -> None:
    session_counter = 0

    async def handler(ws: WebSocketServerProtocol) -> None:
        nonlocal session_counter
        session_counter += 1
        config = SessionConfig(session_id=f"s{session_counter}")
        await voice_session(ws, config)

    log.info("Voice agent WebSocket server on ws://%s:%d", host, port)
    async with websockets.serve(handler, host, port):
        await asyncio.Future()


# ── Evaluation utilities ──────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate without external library.
    WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n = len(ref)
    if n == 0:
        return 0.0
    # Dynamic programming (Levenshtein on word level)
    dp = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        prev = dp[:]
        dp[0] = i + 1
        for j, h in enumerate(hyp):
            dp[j + 1] = min(
                prev[j] + (0 if r == h else 1),  # substitution
                dp[j] + 1,                        # insertion
                prev[j + 1] + 1,                  # deletion
            )
    return dp[len(hyp)] / n


def evaluate_latencies(records: list[LatencyRecord]) -> dict[str, float]:
    """Compute p50/p95/max TTFA from a list of LatencyRecord objects."""
    ttfas = sorted(r.ttfa_ms for r in records if r.ttfa_ms is not None)
    if not ttfas:
        return {}
    n = len(ttfas)
    return {
        "count": n,
        "p50_ms": ttfas[n // 2],
        "p95_ms": ttfas[int(n * 0.95)],
        "max_ms": ttfas[-1],
        "mean_ms": sum(ttfas) / n,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
```

---

## Exercises

1. Write a pytest test that creates a WebSocket server on a random port, connects
   a client, sends 5 × 640-byte mock audio frames, and asserts the client receives
   a `{"type": "transcript"}` JSON message. Use `websockets.connect` in the test.

2. The `compute_wer` function is a simple Levenshtein. Test it with:
   - Identical strings → WER = 0.0
   - One word different → WER = 1/N
   - Completely different → WER close to 1.0

3. Add a `session_stats` endpoint: when the client sends
   `{"type": "get_stats"}`, the server responds with
   `{"type": "stats", "ttfa_ms": <last_ttfa>, "turn_count": <N>}`.

4. Implement a `VoiceAgentBenchmark` class that:
   - Runs the pipeline with a list of `(audio_bytes, expected_transcript)` pairs.
   - Collects `LatencyRecord` objects.
   - Returns a `BenchmarkResult` Pydantic model with WER, p50/p95 TTFA.

*(Solutions: [solutions/lesson-6.md](solutions/lesson-6.md))*

---

## Self-check

- [ ] I can design a binary+text WebSocket protocol for voice agents
- [ ] I can implement a WebSocket server that runs the mock pipeline
- [ ] I can compute TTFA, WER, and summarize p50/p95 latencies
- [ ] I understand the trade-offs between building a custom pipeline vs. using a provider realtime API
