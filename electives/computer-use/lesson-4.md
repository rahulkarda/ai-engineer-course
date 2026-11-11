# Elective: Computer Use · Lesson 4 — DOM vs. Vision Grounding & Action Spaces

> Objective: Choose the right grounding strategy and define a well-typed action space.
> Time: ~60 min · Prereqs: [[lesson-3.md]]

## Concept

**Grounding** is the process of mapping a high-level intent ("click the Search button")
to a concrete, executable operation (click at coordinate `(320, 240)` or click CSS
selector `button[type=submit]`). The grounding strategy determines how fragile or
robust your agent is.

Two dominant approaches:

| Approach | How it grounds | Strengths | Weaknesses |
|---|---|---|---|
| **Vision-based** | LLM interprets screenshot → emits `(x, y)` | Works on any UI including canvas/custom widgets | Coordinate errors; fragile to resize/zoom |
| **DOM-based** | LLM or extractor reads accessibility tree/HTML → emits CSS selector or element ID | Precise, replayable | Breaks on dynamic IDs; not all content in DOM |

Most production agents use a **hybrid**: DOM grounding for structured interactive
elements (inputs, buttons, links) and vision fallback for everything else.

**Action space** is the complete set of operations an agent can execute. A smaller,
well-typed action space has two benefits:
1. Fewer ways for the LLM to hallucinate invalid actions.
2. Easier to enforce safety (allowlist of permitted actions).

## How it works internally

**DOM grounding pipeline:**
1. After each navigation, extract the accessibility tree (or lightweight DOM snapshot).
2. Flatten into a list of interactive elements: `{id, role, label, selector, bbox}`.
3. Pass this list to the LLM as structured context alongside the screenshot.
4. The LLM picks an element ID from the list — no coordinate estimation needed.

```
Screenshot (image) ──┐
Element list (text) ──┤─▶ LLM ──▶ {element_id: "e42", action: "click"}
Task + history ───────┘
```

**Coordinate scaling.** The Anthropic computer-use tool outputs coordinates in the
space of the `display_width_px` / `display_height_px` you specified. If your actual
browser viewport is different, you must **scale** the coordinates before sending to
the browser. Failure to scale is a common silent bug.

```python
def scale_coordinate(
    x: int, y: int,
    model_width: int, model_height: int,
    browser_width: int, browser_height: int,
) -> tuple[int, int]:
    scaled_x = round(x * browser_width / model_width)
    scaled_y = round(y * browser_height / model_height)
    return scaled_x, scaled_y
```

**Constrained action space.** Rather than exposing raw Playwright, define a closed set
of typed actions as a Pydantic discriminated union. The LLM can only emit these; your
dispatcher handles them. Unrecognized actions raise `ValidationError` and are blocked.

## Worked example

```python
# lesson_4_demo.py
"""
Typed action space with DOM element extraction and coordinate scaling.
No real browser; all examples are self-contained with mocks.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Literal
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed action space (discriminated union)
# ---------------------------------------------------------------------------

class ClickCoordinateAction(BaseModel):
    """Vision-grounded click at (x, y) in model coordinate space."""
    type: Literal["click_coordinate"] = "click_coordinate"
    x: int = Field(ge=0, le=4096)
    y: int = Field(ge=0, le=4096)


class ClickElementAction(BaseModel):
    """DOM-grounded click by element ID from the element list."""
    type: Literal["click_element"] = "click_element"
    element_id: str


class TypeAction(BaseModel):
    """Type text into the currently focused / target element."""
    type: Literal["type"] = "type"
    text: str = Field(min_length=1, max_length=4096)
    target_selector: str | None = None  # click before typing if provided

    @field_validator("text")
    @classmethod
    def no_null_bytes(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("text must not contain null bytes")
        return v


class ScrollAction(BaseModel):
    """Scroll in a direction."""
    type: Literal["scroll"] = "scroll"
    x: int = Field(default=640, ge=0, le=4096)
    y: int = Field(default=400, ge=0, le=4096)
    direction: Literal["up", "down", "left", "right"] = "down"
    clicks: int = Field(default=3, ge=1, le=20)


class NavigateAction(BaseModel):
    """Navigate to a URL."""
    type: Literal["navigate"] = "navigate"
    url: str

    @field_validator("url")
    @classmethod
    def require_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs allowed, got: {v!r}")
        return v


class ScreenshotAction(BaseModel):
    """Take a screenshot and return it."""
    type: Literal["screenshot"] = "screenshot"


class DoneAction(BaseModel):
    """Signal task completion."""
    type: Literal["done"] = "done"
    result: str = ""


# Discriminated union — Pydantic picks the right model from `type`
AgentAction = Annotated[
    ClickCoordinateAction
    | ClickElementAction
    | TypeAction
    | ScrollAction
    | NavigateAction
    | ScreenshotAction
    | DoneAction,
    Field(discriminator="type"),
]


def parse_action(raw: dict[str, Any]) -> AgentAction:
    """
    Parse a raw dict from the LLM into a typed AgentAction.
    Raises ValidationError if the action is malformed or unknown.
    """
    from pydantic import TypeAdapter
    adapter: Any = TypeAdapter(AgentAction)
    return adapter.validate_python(raw)


# ---------------------------------------------------------------------------
# DOM element extraction
# ---------------------------------------------------------------------------

class DOMElement(BaseModel):
    """Lightweight representation of an interactive DOM element."""
    element_id: str         # stable short ID for this snapshot
    role: str               # "button", "link", "textbox", "checkbox", etc.
    label: str | None       # accessible name / aria-label / text content
    selector: str           # CSS selector for Playwright
    bounding_box: dict[str, float] | None  # {x, y, width, height}
    center_x: int | None    # pre-computed click target
    center_y: int | None


def extract_interactive_elements(accessibility_tree: dict) -> list[DOMElement]:
    """
    Walk accessibility tree and extract interactive elements.
    In production: call page.accessibility.snapshot() then walk the tree.
    Here: processes a mock tree structure.
    """
    interactive_roles = {
        "button", "link", "textbox", "checkbox", "radio",
        "combobox", "listbox", "option", "menuitem", "tab",
        "searchbox", "spinbutton", "slider",
    }
    elements: list[DOMElement] = []
    _walk_tree(accessibility_tree, interactive_roles, elements, counter=[0])
    log.info("extracted %d interactive elements", len(elements))
    return elements


def _walk_tree(
    node: dict,
    roles: set[str],
    out: list[DOMElement],
    counter: list[int],
) -> None:
    role = node.get("role", "")
    if role in roles:
        counter[0] += 1
        eid = f"e{counter[0]:03d}"
        label = node.get("name") or node.get("value")
        # Bounding box comes from a separate page.evaluate call in production
        bbox = node.get("bbox")
        cx = round(bbox["x"] + bbox["width"] / 2) if bbox else None
        cy = round(bbox["y"] + bbox["height"] / 2) if bbox else None
        out.append(DOMElement(
            element_id=eid,
            role=role,
            label=label,
            selector=node.get("selector", f"[aria-label='{label}']"),
            bounding_box=bbox,
            center_x=cx,
            center_y=cy,
        ))
    for child in node.get("children", []):
        _walk_tree(child, roles, out, counter)


# ---------------------------------------------------------------------------
# Coordinate scaling
# ---------------------------------------------------------------------------

def scale_coordinate(
    x: int,
    y: int,
    model_width: int,
    model_height: int,
    browser_width: int,
    browser_height: int,
) -> tuple[int, int]:
    """
    Scale (x, y) from model coordinate space to browser viewport coordinates.
    Always apply this before sending coordinates to Playwright.
    """
    scaled_x = round(x * browser_width / model_width)
    scaled_y = round(y * browser_height / model_height)
    return scaled_x, scaled_y


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:
    """
    Dispatch typed AgentActions to the browser.
    All safety checks happen here (see lesson 5 for full gates).
    """

    def __init__(
        self,
        page: Any,
        model_width: int = 1280,
        model_height: int = 800,
    ) -> None:
        self._page = page
        self._model_width = model_width
        self._model_height = model_height

    async def dispatch(self, action: AgentAction) -> dict[str, Any]:
        """Execute an action; return a result dict."""
        match action:
            case ClickCoordinateAction(x=x, y=y):
                bw = (self._page.viewport_size or {}).get("width", 1280)
                bh = (self._page.viewport_size or {}).get("height", 800)
                sx, sy = scale_coordinate(
                    x, y, self._model_width, self._model_height, bw, bh
                )
                await self._page.mouse.click(sx, sy)
                log.info("click_coordinate  model=(%d,%d)  browser=(%d,%d)", x, y, sx, sy)
                return {"clicked": True, "browser_x": sx, "browser_y": sy}

            case ClickElementAction(element_id=eid):
                # In production: look up selector from element list
                log.info("click_element  id=%s", eid)
                await self._page.mouse.click(0, 0)  # placeholder
                return {"clicked_element": eid}

            case TypeAction(text=text, target_selector=sel):
                if sel:
                    await self._page.locator(sel).fill(text)
                else:
                    await self._page.keyboard.type(text)
                log.info("typed  len=%d", len(text))
                return {"typed": len(text)}

            case NavigateAction(url=url):
                await self._page.goto(url, wait_until="networkidle")
                return {"navigated": url}

            case ScrollAction(x=x, y=y, direction=d, clicks=c):
                dy = c * 100 * (1 if d == "down" else -1 if d == "up" else 0)
                dx = c * 100 * (1 if d == "right" else -1 if d == "left" else 0)
                await self._page.mouse.wheel(dx, dy)
                return {"scrolled": d}

            case ScreenshotAction():
                raw = await self._page.screenshot(full_page=False)
                import base64
                return {"screenshot_b64": base64.standard_b64encode(raw).decode()}

            case DoneAction(result=r):
                log.info("done  result=%r", r)
                return {"done": True, "result": r}

        raise ValueError(f"Unhandled action type: {type(action)}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo() -> None:
    import asyncio

    # Mock page
    page = MagicMock()
    page.url = "https://example.com"
    page.viewport_size = {"width": 1280, "height": 800}
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.type = AsyncMock()
    page.locator = MagicMock(return_value=MagicMock(fill=AsyncMock()))
    page.goto = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n...")

    dispatcher = ActionDispatcher(page)

    # Vision-grounded click
    action = parse_action({"type": "click_coordinate", "x": 320, "y": 240})
    result = await dispatcher.dispatch(action)
    log.info("click result: %s", result)

    # DOM-grounded type
    action = parse_action({"type": "type", "text": "hello world", "target_selector": "input#q"})
    result = await dispatcher.dispatch(action)
    log.info("type result: %s", result)

    # Invalid action — should raise ValidationError
    try:
        parse_action({"type": "navigate", "url": "javascript:alert(1)"})
    except Exception as e:
        log.info("blocked bad URL: %s", e)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
```

Key observations:
- `parse_action` with a discriminated union means every LLM output is validated at the
  boundary — invalid actions never reach the browser.
- `scale_coordinate` must be applied before any vision-grounded click.
- The `TypeAction` validator rejects null bytes — a simple injection defense.
- `NavigateAction` blocks `javascript:` and `file://` at the type level.

## Exercises

1. **Element list context.** Write a function `format_element_list(elements: list[DOMElement]) -> str`
   that formats the element list as a compact string for inclusion in the LLM prompt,
   e.g. `"e001 button[Submit] e002 textbox[Search] e003 link[Home]"`.
   The LLM can then emit `{"type": "click_element", "element_id": "e002"}`.

2. **Coordinate clamping.** Add a `clamp_coordinate(x, y, width, height)` function
   that clamps coordinates to the viewport bounds (0–width, 0–height).
   Add this to `ClickCoordinateAction` as a model validator.
   Test the clamping with out-of-bounds values.

3. **Schema generation.** Call `AgentAction.__class__.__name__` and use Pydantic's
   `model_json_schema()` on each leaf model to generate a human-readable action spec.
   Print the action names and their required fields. This is what you'd pass to
   the LLM as a tool definition.

4. **Hybrid grounding.** Modify the dispatcher so that `ClickCoordinateAction`
   first attempts to find a DOM element whose bounding box center is within
   `tolerance` pixels of `(x, y)`. If found, use `click_selector` for reliability;
   if not found, fall back to coordinate click.

(Solutions: solutions/lesson-4.md)

## Self-check

- [ ] I can describe the two grounding strategies and the hybrid approach.
- [ ] I can write a Pydantic discriminated union for an action space.
- [ ] I can implement coordinate scaling and explain when it's required.
- [ ] I can extract interactive elements from an accessibility tree structure.
- [ ] I understand why a closed, typed action space is a safety property.

## Supplementary links

- [Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)
- [Playwright accessibility API](https://playwright.dev/python/docs/api/class-accessibility)
- [WebArena element grounding evaluation](https://webarena.dev/)
- [ARIA roles reference](https://www.w3.org/TR/wai-aria-1.2/#role_definitions)
