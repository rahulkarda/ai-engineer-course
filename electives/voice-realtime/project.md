# Elective: Voice & Realtime — Capstone Project
# Voice-Agent Orchestration Loop

> Estimated time: ~4 hours  
> Prereqs: All six lessons complete

---

## Overview

Build a **production-quality voice-agent orchestration loop** — a single cohesive
system that integrates VAD, STT, LLM, TTS, barge-in, and WebSocket transport.
Everything is mockable; tests must never require audio hardware or live API keys
(other than `ANTHROPIC_API_KEY` which may be mocked too).

---

## Specification

### Directory layout

```
voice_agent/
├── __init__.py
├── config.py          # PipelineConfig, AgentConfig (pydantic-settings)
├── vad.py             # TurnDetector, VADBackend protocol, MockVADBackend, WebRTCVADBackend
├── stt.py             # STTProvider protocol, MockSTTProvider, WhisperSTTProvider stub
├── llm.py             # LLMProvider protocol, AnthropicLLMProvider, MockLLMProvider
├── tts.py             # TTSProvider protocol, MockTTSProvider, ElevenLabsTTSProvider stub
├── barge_in.py        # BargInCoordinator, AgentState
├── orchestrator.py    # VoiceOrchestrator (main loop)
├── ws_server.py       # WebSocket server wrapping the orchestrator
└── evaluation.py      # LatencyRecord, BenchmarkRunner, compute_wer, evaluate_latencies

tests/
├── test_vad.py
├── test_stt.py
├── test_llm.py
├── test_tts.py
├── test_barge_in.py
├── test_orchestrator.py
├── test_ws_server.py
└── test_evaluation.py
```

### Core class: `VoiceOrchestrator`

```python
class VoiceOrchestrator:
    def __init__(
        self,
        vad_backend: VADBackend,
        stt_provider: STTProvider,
        llm_provider: LLMProvider,
        tts_provider: TTSProvider,
        config: AgentConfig,
    ) -> None: ...

    async def run(self, audio_in: asyncio.Queue[bytes]) -> None:
        """Main loop: process audio until sentinel (b"") received."""
        ...

    async def handle_turn(self, segment: SpeechSegment) -> TurnResult: ...

    @property
    def metrics(self) -> list[LatencyRecord]: ...
```

### `LLMProvider` protocol

```python
class LLMProvider(Protocol):
    async def stream_response(
        self,
        transcript: str,
        history: list[ConversationTurn],
    ) -> AsyncIterator[str]: ...
```

`AnthropicLLMProvider` must:
- Use `os.getenv("ANTHROPIC_API_KEY")`.
- Use `claude-haiku-5-5` by default; accept `claude-sonnet-5-5` or `claude-opus-5-5` via config.
- Pass conversation history as Anthropic `messages` format.
- Stream tokens via `client.messages.stream(...)`.

### Barge-in requirements

- `VoiceOrchestrator.handle_turn()` must cancel TTS playback if `BargInCoordinator.barge_in_event` fires.
- After cancellation, the orchestrator must immediately re-enter LISTENING state.
- Turn result must include an `interrupted: bool` field.

### Metrics requirements

`VoiceOrchestrator.metrics` returns a list of `LatencyRecord` objects, one per turn.
Each record must contain:
- `turn_id` (int, incrementing)
- `speech_end_ts` (float, monotonic)
- `stt_done_ts` (float)
- `llm_first_token_ts` (float)
- `tts_first_chunk_ts` (float)
- `agent_stopped_ts` (float)
- `interrupted` (bool)

`evaluation.py` must expose:
- `compute_wer(reference: str, hypothesis: str) -> float`
- `evaluate_latencies(records: list[LatencyRecord]) -> LatencyStats`

```python
class LatencyStats(BaseModel):
    count: int
    ttfa_p50_ms: float
    ttfa_p95_ms: float
    ttfa_max_ms: float
    ttfa_mean_ms: float
    stt_mean_ms: float
    llm_first_token_mean_ms: float
    tts_first_chunk_mean_ms: float
    interrupted_count: int
```

### WebSocket server requirements

`ws_server.py` must expose a `run_server(host, port)` coroutine.  
The protocol (binary audio frames + JSON control messages) must match the design
from Lesson 6.

---

## Acceptance criteria

### Functional

- [ ] `VoiceOrchestrator` processes 5 mock audio segments end-to-end without errors.
- [ ] After processing, `orchestrator.metrics` contains 5 `LatencyRecord` entries.
- [ ] Barge-in mid-turn sets `TurnResult.interrupted = True` and the orchestrator
      continues to the next turn.
- [ ] Conversation history is passed to the LLM on every turn (assert with MockLLMProvider).
- [ ] `AnthropicLLMProvider` constructs correct `messages` format (unit test with monkeypatched client).

### WebSocket

- [ ] A client can connect, send 10 audio frames, and receive at least one
      `{"type": "transcript"}` JSON message.
- [ ] Sending `{"type": "interrupt"}` mid-turn cancels TTS (assert
      `speaking_end` event arrives within 200 ms).

### Metrics & evaluation

- [ ] `compute_wer("hello world", "hello world") == 0.0`
- [ ] `compute_wer("hello world", "hello earth") == pytest.approx(0.5)`
- [ ] `evaluate_latencies([...])` returns correct p50/p95 from a known list.

### Code quality

- [ ] All public functions/classes have type hints and docstrings.
- [ ] No secrets hardcoded; all API keys via `os.getenv`.
- [ ] `logging` used throughout (no `print`).
- [ ] Tests use `pytest.mark` to categorize: `unit`, `integration`.
- [ ] All external calls (Anthropic, Deepgram, ElevenLabs) mocked in tests.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|-----------|--------|
| Independent implementation (all modules present, tests pass) | 40% |
| Conceptual correctness (correct pipeline ordering, latency tracking) | 25% |
| Debugging & problem-solving (barge-in correctly cancels, history works) | 20% |
| Code quality & testing (types, logging, mocks, pytest marks) | 15% |

---

## Extension challenges (optional)

1. **Adaptive model selection**: if the user's last 3 turns had low WER and short
   utterances, downgrade to `claude-haiku-5-5`; if complex, upgrade to
   `claude-sonnet-5-5`. Unit-test the selection logic.

2. **Reconnect logic**: if the WebSocket drops, the orchestrator should attempt
   3 reconnections with exponential backoff before giving up.

3. **Turn timeout**: if the user starts speaking but doesn't complete a turn
   within 10 seconds, emit a timeout event and ask the user to repeat.

4. **Live transcription display**: stream interim transcripts (is_final=False)
   to the client as they arrive, not just the final result.

---

Answer key: [solutions/project.md](solutions/project.md)
