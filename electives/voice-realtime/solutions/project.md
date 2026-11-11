# Solutions: Capstone Project — Voice-Agent Orchestration Loop

## Complete reference implementation

### `voice_agent/config.py`

```python
"""
voice_agent/config.py
Pydantic-settings based configuration for the voice agent pipeline.
"""
import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """
    Runtime configuration. All values can be overridden via environment variables.
    Example: VOICE_AGENT_LLM_MODEL=claude-sonnet-5-5
    """
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 300
    llm_system_prompt: str = (
        "You are a helpful voice assistant. Keep responses concise and conversational."
    )
    stt_silence_end_ms: int = 600
    stt_min_speech_ms: int = 250
    tts_latency_ms: float = 50.0
    echo_suppression: bool = True
    min_barge_in_ms: float = 200.0
    ws_host: str = "localhost"
    ws_port: int = 8765

    class Config:
        env_prefix = "VOICE_AGENT_"
```

### `voice_agent/llm.py`

```python
"""
voice_agent/llm.py
LLM provider protocol and implementations.
"""
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    async def stream_response(
        self,
        transcript: str,
        history: list[ConversationTurn],
    ) -> AsyncIterator[str]: ...


class MockLLMProvider:
    """
    Mock: streams a fixed response as space-separated word tokens.
    Never calls Anthropic or any external API.
    """
    def __init__(
        self,
        response: str = "This is a mock response for testing.",
        token_delay_s: float = 0.01,
    ) -> None:
        self._response = response
        self._delay = token_delay_s
        self.call_count = 0
        self.last_history: list[ConversationTurn] = []

    async def stream_response(
        self,
        transcript: str,
        history: list[ConversationTurn],
    ) -> AsyncIterator[str]:
        self.call_count += 1
        self.last_history = list(history)

        async def _gen() -> AsyncIterator[str]:
            words = self._response.split()
            for i, word in enumerate(words):
                await asyncio.sleep(self._delay)
                yield word + (" " if i < len(words) - 1 else "")

        return _gen()


class AnthropicLLMProvider:
    """
    Anthropic streaming LLM.
    Model IDs: claude-haiku-5-5 | claude-sonnet-5-5 | claude-opus-5-5
    Requires ANTHROPIC_API_KEY.
    """
    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        max_tokens: int = 300,
        system_prompt: str = "You are a helpful voice assistant.",
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        import anthropic  # noqa: PLC0415
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._system = system_prompt

    async def stream_response(
        self,
        transcript: str,
        history: list[ConversationTurn],
    ) -> AsyncIterator[str]:
        messages = [{"role": t.role, "content": t.content} for t in history]
        messages.append({"role": "user", "content": transcript})

        async def _gen() -> AsyncIterator[str]:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system,
                messages=messages,
            ) as stream:
                async for delta in stream.text_stream:
                    yield delta

        return _gen()
```

### `voice_agent/orchestrator.py`

```python
"""
voice_agent/orchestrator.py
VoiceOrchestrator: main async turn-processing loop.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field

from voice_agent.barge_in import AgentState, BargInCoordinator, PipelineConfig
from voice_agent.config import AgentConfig
from voice_agent.llm import AnthropicLLMProvider, ConversationTurn, LLMProvider
from voice_agent.stt import MockSTTProvider, STTProvider
from voice_agent.tts import MockTTSProvider, TTSProvider
from voice_agent.vad import MockVADBackend, SpeechSegment, TurnDetector, VADBackend

log = logging.getLogger(__name__)

SENTINEL = b""  # empty bytes signals end of audio stream


@dataclass
class LatencyRecord:
    turn_id: int
    speech_end_ts: float
    stt_done_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    tts_first_chunk_ts: float = 0.0
    agent_stopped_ts: float = 0.0
    interrupted: bool = False

    @property
    def ttfa_ms(self) -> float:
        if self.tts_first_chunk_ts and self.speech_end_ts:
            return (self.tts_first_chunk_ts - self.speech_end_ts) * 1000
        return -1.0

    @property
    def stt_ms(self) -> float:
        return (self.stt_done_ts - self.speech_end_ts) * 1000

    @property
    def llm_first_token_ms(self) -> float:
        return (self.llm_first_token_ts - self.stt_done_ts) * 1000


@dataclass
class TurnResult:
    turn_id: int
    transcript: str
    response: str
    interrupted: bool
    record: LatencyRecord


class VoiceOrchestrator:
    """
    Async voice-agent orchestration loop.

    Wires VAD → STT → LLM → TTS → playback with barge-in support.
    All providers are injected (default to mocks if not supplied).
    """
    def __init__(
        self,
        vad_backend: VADBackend | None = None,
        stt_provider: STTProvider | None = None,
        llm_provider: LLMProvider | None = None,
        tts_provider: TTSProvider | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._vad = TurnDetector(
            vad_backend or MockVADBackend(),
            silence_end_ms=self._config.stt_silence_end_ms,
            min_speech_ms=self._config.stt_min_speech_ms,
        )
        self._stt = stt_provider or MockSTTProvider()
        self._llm = llm_provider or MockLLMProvider()
        self._tts = tts_provider or MockTTSProvider(latency_ms=self._config.tts_latency_ms)
        self._barge_in = BargInCoordinator(
            PipelineConfig(
                echo_suppression=self._config.echo_suppression,
                min_barge_in_ms=self._config.min_barge_in_ms,
            )
        )
        self._history: list[ConversationTurn] = []
        self._metrics: list[LatencyRecord] = []
        self._turn_count = 0

    @property
    def metrics(self) -> list[LatencyRecord]:
        return list(self._metrics)

    async def run(self, audio_in: asyncio.Queue[bytes]) -> None:
        """
        Main loop: reads audio bytes, processes turns until sentinel received.
        """
        log.info("Orchestrator: starting")
        while True:
            raw = await audio_in.get()
            if raw == SENTINEL:
                log.info("Orchestrator: sentinel received, stopping")
                audio_in.task_done()
                break
            segments = self._vad.feed(raw)
            for segment in segments:
                result = await self.handle_turn(segment)
                log.info(
                    "Turn %d: '%s' → TTFA=%.0f ms interrupted=%s",
                    result.turn_id,
                    result.transcript[:40],
                    result.record.ttfa_ms,
                    result.interrupted,
                )
            audio_in.task_done()

    async def handle_turn(self, segment: SpeechSegment) -> TurnResult:
        """Process one speech segment end-to-end."""
        self._turn_count += 1
        turn_id = self._turn_count
        record = LatencyRecord(turn_id=turn_id, speech_end_ts=time.monotonic())
        self._barge_in.reset_for_new_turn()

        # STT
        transcript = await self._stt.transcribe(segment.as_bytes())
        record.stt_done_ts = time.monotonic()
        if not transcript.text.strip():
            return TurnResult(
                turn_id=turn_id,
                transcript="",
                response="",
                interrupted=False,
                record=record,
            )
        self._history.append(ConversationTurn(role="user", content=transcript.text))

        # LLM + TTS with barge-in support
        response_parts: list[str] = []
        interrupted = False

        stop_event = asyncio.Event()

        async def _llm_tts() -> None:
            nonlocal interrupted
            from voice_agent.tts import SentenceBuffer  # noqa: PLC0415
            buf = SentenceBuffer()
            first_token = True
            stream = await self._llm.stream_response(transcript.text, self._history[:-1])
            async for delta in stream:
                if stop_event.is_set():
                    interrupted = True
                    break
                if first_token:
                    record.llm_first_token_ts = time.monotonic()
                    first_token = False
                for sentence in buf.feed(delta):
                    if stop_event.is_set():
                        break
                    chunk = await self._tts.synthesize(sentence)
                    if not record.tts_first_chunk_ts:
                        record.tts_first_chunk_ts = time.monotonic()
                    response_parts.append(sentence)
            if not stop_event.is_set():
                rem = buf.flush()
                if rem:
                    chunk = await self._tts.synthesize(rem)
                    if not record.tts_first_chunk_ts:
                        record.tts_first_chunk_ts = time.monotonic()
                    response_parts.append(rem)

        self._barge_in.agent_started_speaking()
        llm_tts_task = asyncio.create_task(_llm_tts())
        barge_in_task = asyncio.create_task(self._barge_in.barge_in_event.wait())

        done, pending = await asyncio.wait(
            [llm_tts_task, barge_in_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if barge_in_task in done:
            interrupted = True
            stop_event.set()

        self._barge_in.agent_stopped_speaking()
        record.agent_stopped_ts = time.monotonic()
        record.interrupted = interrupted

        assembled = " ".join(response_parts)
        if assembled:
            self._history.append(ConversationTurn(role="assistant", content=assembled))
        self._metrics.append(record)

        return TurnResult(
            turn_id=turn_id,
            transcript=transcript.text,
            response=assembled,
            interrupted=interrupted,
            record=record,
        )
```

### `voice_agent/evaluation.py`

```python
"""
voice_agent/evaluation.py
Latency statistics and WER computation.
"""
from pydantic import BaseModel
from voice_agent.orchestrator import LatencyRecord


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


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n = len(ref)
    if n == 0:
        return 0.0
    dp = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        prev = dp[:]
        dp[0] = i + 1
        for j, h in enumerate(hyp):
            dp[j + 1] = min(
                prev[j] + (0 if r == h else 1),
                dp[j] + 1,
                prev[j + 1] + 1,
            )
    return dp[len(hyp)] / n


def evaluate_latencies(records: list[LatencyRecord]) -> LatencyStats:
    ttfas = sorted(r.ttfa_ms for r in records if r.tts_first_chunk_ts)
    stts = [r.stt_ms for r in records if r.stt_done_ts]
    llm_fts = [r.llm_first_token_ms for r in records if r.llm_first_token_ts]
    tts_fcs = [
        (r.tts_first_chunk_ts - r.llm_first_token_ts) * 1000
        for r in records
        if r.tts_first_chunk_ts and r.llm_first_token_ts
    ]
    n = len(ttfas)

    def _percentile(lst: list[float], p: float) -> float:
        if not lst:
            return 0.0
        return lst[int(len(lst) * p)]

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return LatencyStats(
        count=len(records),
        ttfa_p50_ms=_percentile(ttfas, 0.5),
        ttfa_p95_ms=_percentile(ttfas, 0.95),
        ttfa_max_ms=ttfas[-1] if ttfas else 0.0,
        ttfa_mean_ms=_mean(ttfas),
        stt_mean_ms=_mean(stts),
        llm_first_token_mean_ms=_mean(llm_fts),
        tts_first_chunk_mean_ms=_mean(tts_fcs),
        interrupted_count=sum(1 for r in records if r.interrupted),
    )
```

### `tests/test_orchestrator.py` (key tests)

```python
import asyncio
import pytest
from voice_agent.config import AgentConfig
from voice_agent.orchestrator import SENTINEL, VoiceOrchestrator
from voice_agent.vad import FRAME_BYTES


@pytest.mark.asyncio
@pytest.mark.unit
async def test_orchestrator_processes_five_segments():
    orch = VoiceOrchestrator()
    q: asyncio.Queue[bytes] = asyncio.Queue()
    # 5 turns: each = 15 speech frames + 30 silence frames
    for _ in range(5):
        for _ in range(45):
            await q.put(b"\x00" * FRAME_BYTES)
    await q.put(SENTINEL)
    await orch.run(q)
    assert len(orch.metrics) == 5


@pytest.mark.asyncio
@pytest.mark.unit
async def test_orchestrator_barge_in_sets_interrupted():
    from voice_agent.llm import MockLLMProvider
    # Slow LLM so barge-in can fire mid-turn
    llm = MockLLMProvider(
        response="This is a very long response that will be interrupted.",
        token_delay_s=0.05,
    )
    orch = VoiceOrchestrator(llm_provider=llm)
    # Inject barge-in after 100 ms
    async def inject_barge_in():
        await asyncio.sleep(0.1)
        orch._barge_in.barge_in_event.set()

    from voice_agent.vad import SpeechSegment, FRAME_BYTES, FRAME_DURATION_MS
    seg = SpeechSegment(
        frames=[b"\x00" * FRAME_BYTES] * 15,
        speech_frame_count=15,
    )
    barge_task = asyncio.create_task(inject_barge_in())
    result = await orch.handle_turn(seg)
    await barge_task
    assert result.interrupted is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_orchestrator_passes_history_to_llm():
    from voice_agent.llm import MockLLMProvider
    llm = MockLLMProvider()
    orch = VoiceOrchestrator(llm_provider=llm)
    from voice_agent.vad import SpeechSegment, FRAME_BYTES
    seg = SpeechSegment(
        frames=[b"\x00" * FRAME_BYTES] * 15,
        speech_frame_count=15,
    )
    # First turn
    await orch.handle_turn(seg)
    assert llm.call_count == 1
    # Second turn — history should have 1 pair (user + assistant from turn 1)
    await orch.handle_turn(seg)
    assert llm.call_count == 2
    # History passed on second call should include turn 1
    assert any(t.role == "user" for t in llm.last_history)
```

### Key design decisions

1. `SentenceBuffer` is imported from `voice_agent.tts` in the orchestrator
   (co-locate with TTS since it drives TTS dispatch).
2. `SENTINEL = b""` is the clean shutdown signal — no `None` in typed queues.
3. `asyncio.wait(..., return_when=FIRST_COMPLETED)` is preferred over
   `asyncio.create_task` + manual event polling — it is the idiomatic asyncio
   pattern for "whichever happens first."
4. `ConversationTurn` is in `llm.py` (not a separate models file) to keep the
   boundary clear: history is an LLM concern.
5. `LatencyRecord` uses absolute `time.monotonic()` timestamps rather than
   deltas so post-hoc analysis can reconstruct the full timeline.
