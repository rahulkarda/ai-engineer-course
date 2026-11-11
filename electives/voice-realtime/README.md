# Elective: Voice & Realtime Agents

> Module overview · Self-contained · Estimated time: 8–10 hours

## What this module teaches

Voice agents connect spoken language to LLM reasoning in real time. The pipeline
is deceptively simple — microphone → VAD → STT → LLM → TTS → speaker — but
making it *feel* conversational requires tight latency budgets, barge-in handling,
streaming at every stage, and careful orchestration.

After this module you will be able to:

- [ ] Explain the full voice agent pipeline and the latency budget at each stage
- [ ] Implement VAD-gated STT routing (turn detection from audio)
- [ ] Stream LLM tokens directly into a TTS queue to minimize first-word latency
- [ ] Handle barge-in (user interrupts the agent mid-speech)
- [ ] Build a fully mockable voice-agent orchestration loop with asyncio
- [ ] Measure end-to-end latency and identify the dominant bottleneck
- [ ] Evaluate STT/TTS quality and select the right provider for a use case
- [ ] Connect an orchestration loop to a WebSocket transport layer

## Prerequisites

- Python 3.12+
- Phase 5 (Agentic engineering) complete, or equivalent async/LLM experience
- Comfortable with `asyncio`, `typing`, Pydantic, pytest
- `ANTHROPIC_API_KEY` set in environment

## Dependencies

```bash
# Core async + HTTP
uv add websockets httpx

# Anthropic SDK (LLM step)
uv add anthropic

# Audio I/O — add only when running on real hardware
uv add webrtcvad          # VAD
uv add sounddevice        # microphone / speaker
uv add openai-whisper     # local STT (also needs ffmpeg)

# Cloud STT/TTS (optional — swap with mocks in tests)
uv add deepgram-sdk       # Deepgram STT streaming
uv add elevenlabs         # ElevenLabs TTS
```

> All lessons and tests use **mock** implementations — no audio hardware, no API
> keys beyond `ANTHROPIC_API_KEY`, required. Real audio deps are listed for
> reference; install them only when running on actual hardware.

## Lesson index

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [lesson-1.md](lesson-1.md) | Pipeline anatomy & latency budgets | ~45 min |
| 2 | [lesson-2.md](lesson-2.md) | VAD & turn detection | ~60 min |
| 3 | [lesson-3.md](lesson-3.md) | STT options & streaming transcription | ~60 min |
| 4 | [lesson-4.md](lesson-4.md) | LLM streaming + TTS pipeline | ~60 min |
| 5 | [lesson-5.md](lesson-5.md) | Barge-in, interruption, duplex audio | ~60 min |
| 6 | [lesson-6.md](lesson-6.md) | WebSockets, realtime APIs, evaluation | ~60 min |
| P | [project.md](project.md) | Capstone: voice-agent orchestration loop | ~4 hr |

## Solutions

All exercise answers live under [solutions/](solutions/). Open them only after
attempting the exercises yourself.

## Supplementary links

- OpenAI Realtime API (WebSocket voice): <https://platform.openai.com/docs/guides/realtime>
- Deepgram streaming STT: <https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio>
- ElevenLabs streaming TTS: <https://elevenlabs.io/docs/api-reference/streaming>
- webrtcvad PyPI: <https://pypi.org/project/webrtcvad/>
- Whisper GitHub: <https://github.com/openai/whisper>
- Anthropic streaming docs: <https://docs.anthropic.com/en/api/messages-streaming>
- asyncio docs (Python 3.12): <https://docs.python.org/3.12/library/asyncio.html>
- websockets library: <https://websockets.readthedocs.io/>
