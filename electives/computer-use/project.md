# Elective: Computer Use — Capstone Project

> Build a sandboxed browser agent that performs a multi-step web task with a full
> safety stack: action allowlist, human-approval gate for sensitive actions, and
> post-action verification. Browser and LLM are mocked in all tests.

**Estimated time:** 4–5 hours  
**Prereqs:** All six lessons complete

---

## Overview

You will build `SandboxedWebAgent` — a production-grade computer-use agent that:

1. Uses the Anthropic computer-use tool API (mocked in tests) to reason about browser state.
2. Executes actions through a typed `ActionDispatcher` (Playwright, mocked in tests).
3. Enforces a layered safety stack before every action:
   - Action allowlist (default-deny).
   - Sensitive-action detection with human-approval gate.
   - Post-action verification with retry.
4. Logs every action to an audit log.
5. Exposes metrics: turns used, violations blocked, verifications failed.

**Safety is non-negotiable.** Every action must pass through the safety stack. Tests must
cover both the happy path and the safety-blocked path.

---

## Deliverables

```
project/
├── src/
│   └── sandboxed_agent/
│       ├── __init__.py
│       ├── actions.py          # Typed action space (Pydantic)
│       ├── browser_tools.py    # BrowserTools wrapper (Playwright)
│       ├── safety.py           # Allowlist + sensitivity + approval gate + audit
│       ├── verification.py     # Post-action verification
│       ├── agent_loop.py       # Main agentic loop (LLM + dispatch + safety + verify)
│       └── config.py           # Config via pydantic-settings
├── tests/
│   ├── test_actions.py
│   ├── test_safety.py
│   ├── test_verification.py
│   └── test_agent_loop.py
├── pyproject.toml
└── .env.example
```

---

## Specification

### `config.py`

Use `pydantic-settings` (`BaseSettings`). Fields:

```python
class AgentConfig(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = "claude-sonnet-5-5"
    display_width: int = 1280
    display_height: int = 800
    max_turns: int = 20
    max_history_images: int = 3
    sandbox_base_url: str = "https://sandbox.example.com"
    approval_timeout_s: float = 30.0
```

### `actions.py`

Define the full typed action space from Lesson 4:
- `ClickCoordinateAction`, `ClickElementAction`, `TypeAction`, `ScrollAction`,
  `NavigateAction`, `ScreenshotAction`, `DoneAction`
- `AgentAction` discriminated union
- `parse_action(raw: dict) -> AgentAction` — raises `ValidationError` on bad input
- `scale_coordinate(x, y, model_w, model_h, browser_w, browser_h) -> (int, int)`

All Pydantic validators in place: URL scheme check, coordinate bounds, text null-byte check.

### `browser_tools.py`

`BrowserTools` class from Lesson 3, async, with all methods:
- `screenshot() -> ScreenshotResult`
- `navigate(url, timeout_ms) -> NavigateResult`
- `click_coordinate(x, y) -> ClickResult`
- `click_selector(selector, timeout_ms) -> ClickResult`
- `type_text(selector, text, clear_first) -> TypeResult`
- `scroll(x, y, direction, clicks)`
- `get_element_info(selector) -> ElementInfo`
- `wait_for_selector(selector, timeout_ms) -> bool`

`IsolatedBrowserSession` async context manager.

### `safety.py`

Full five-layer safety stack from Lesson 5:
- `AuditLog` with `record(entry)`, `entries`, `violation_count()`
- `ActionAllowlist` with `is_allowed(action_type, url)`, `sandbox_only(base_url)` classmethod
- `is_sensitive_action(action_type, params, current_url) -> (bool, reason)`
- `ApprovalGate` Protocol
- `AutoApproveGate`, `AutoRejectGate` (test stubs)
- `CLIApprovalGate` (production, stdin-based)
- `SafetyLayer.check_and_execute(...)` — runs all layers, returns status dict

### `verification.py`

From Lesson 6:
- `VerificationCondition` dataclass with `description` and async `check` callable
- `verify_with_retry(conditions, max_attempts, base_wait_s)` — raises `VerificationError`
- `with_retry(action_fn, label, max_attempts, base_wait_s, retryable_exceptions)`
- `build_navigation_conditions(page, expected_url_fragment) -> list[VerificationCondition]`
- `build_form_submit_conditions(page, success_selector) -> list[VerificationCondition]`

### `agent_loop.py`

The main `SandboxedWebAgent` class:

```python
class SandboxedWebAgent:
    def __init__(
        self,
        config: AgentConfig,
        browser_tools: BrowserTools,
        safety: SafetyLayer,
        llm_client: Any,   # real Anthropic client or mock
    ) -> None: ...

    async def run(self, task: str) -> AgentRunResult: ...
```

`AgentRunResult` Pydantic model:
```python
class AgentRunResult(BaseModel):
    status: Literal["done", "blocked", "timeout", "error"]
    turns_used: int
    safety_violations: int
    verification_failures: int
    final_message: str
    audit_entries: list[dict]
```

The `run` method:
1. Sends initial `task` to LLM with `COMPUTER_TOOL` definition.
2. On each `tool_use` response: runs safety check, executes action (if cleared),
   runs post-action verification, appends tool_result to messages.
3. On `end_turn`: extracts final text, returns `AgentRunResult`.
4. On `max_turns` exceeded: returns `status="timeout"`.
5. Trims message history to keep only `max_history_images` screenshots.

---

## Acceptance Criteria

### Functional

- [ ] `SandboxedWebAgent.run()` completes a 4-turn mocked task and returns `status="done"`.
- [ ] With `AutoRejectGate`, a task that hits a sensitive action returns `status="blocked"`.
- [ ] Actions not in the allowlist return `status="blocked"` without calling the browser.
- [ ] `AgentRunResult.safety_violations` increments for each blocked action.
- [ ] Post-action verification runs after every state-changing action.
- [ ] When verification fails after max retries, the agent returns `status="error"`.
- [ ] History trimming keeps only the last N screenshot images in messages.

### Safety (all must pass)

- [ ] `NavigateAction` with `javascript:` URL raises `ValidationError` — never reaches browser.
- [ ] `TypeAction` with null bytes raises `ValidationError`.
- [ ] `navigate` to a non-allowlisted domain is blocked by `ActionAllowlist`.
- [ ] Sensitive URL (containing `/delete`) triggers the approval gate.
- [ ] `AutoRejectGate` causes the action to be logged and skipped, not executed.
- [ ] Every action (allowed, blocked, or rejected) appears in `AuditLog`.

### Test coverage

- [ ] `test_actions.py`: all action types parse correctly; invalid inputs raise `ValidationError`.
- [ ] `test_safety.py`: allowlist, sensitivity detection, approve/reject paths all tested.
- [ ] `test_verification.py`: verify_with_retry passes on first try, retries on flaky check, raises on permanent fail.
- [ ] `test_agent_loop.py`: full loop with mock LLM and mock browser; at least four test cases.
- [ ] No test makes a real HTTP request or real LLM call. All external calls mocked.

---

## Implementation hints

**Mock LLM client pattern** (from Lesson 2):
```python
class MockLLMClient:
    """Deterministic mock; drives the agent through a scripted sequence."""
    def __init__(self, responses: list[MagicMock]) -> None:
        self._responses = iter(responses)

    def create(self, **kwargs: Any) -> MagicMock:
        return next(self._responses)
```

**Mock browser page pattern** (from Lesson 3):
```python
def make_mock_page(url: str = "https://sandbox.example.com") -> MagicMock:
    page = MagicMock()
    page.url = url
    page.viewport_size = {"width": 1280, "height": 800}
    page.screenshot = AsyncMock(return_value=MINIMAL_PNG)
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Test Page")
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.locator = MagicMock(return_value=MagicMock(
        click=AsyncMock(), fill=AsyncMock(), clear=AsyncMock(),
        is_visible=AsyncMock(return_value=True),
        text_content=AsyncMock(return_value=""),
        bounding_box=AsyncMock(return_value=None),
    ))
    page.wait_for_selector = AsyncMock()
    return page
```

**Coordinate scaling in the loop.** When the LLM emits a `click_coordinate` action,
always scale before dispatching:
```python
bw = tools._page.viewport_size["width"]
bh = tools._page.viewport_size["height"]
sx, sy = scale_coordinate(action.x, action.y, config.display_width, config.display_height, bw, bh)
```

**History trimming.** To keep only the last N screenshots, walk `messages` in reverse
and count `image` content blocks; set older ones to empty `content: []`.

---

## Extension challenges (optional)

1. **Persistent audit log.** Write `AuditLog` entries to a JSONL file via `pathlib.Path`.
   Add a `load(path) -> AuditLog` classmethod. Test with a temp file.

2. **Network sandboxing.** Add a `setup_network_sandbox(page, allowed_hosts)` function
   that uses `page.route(...)` to block all requests to non-allowlisted hosts.
   Test by mocking `page.route`.

3. **Evaluation harness.** Write a `run_eval(agent, tasks: list[TaskSpec]) -> EvalSummary`
   function using the `AgentEvaluator` from Lesson 6. Include at least five `TaskSpec`
   objects covering: search, navigation, form fill, sensitive action (rejected), and
   a timeout scenario.

4. **DOM hybrid grounding.** Integrate `extract_interactive_elements` from Lesson 4.
   After each navigation, extract elements, format them as context for the LLM, and
   use `ClickElementAction` when the LLM picks an element by ID.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | All modules written; no copy-paste from lessons without adaptation |
| Safety stack correctness | 25% | All safety acceptance criteria pass; no path bypasses the stack |
| Test coverage & quality | 20% | All test files present; every module has meaningful tests; no real API calls |
| Code quality | 15% | Type hints throughout; logging not print; Pydantic models for boundaries; ruff clean |

Answer key: [solutions/project.md](solutions/project.md)

---

## Supplementary links

- [Anthropic computer use docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
- [Playwright Python async API](https://playwright.dev/python/docs/api/class-page)
- [Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
