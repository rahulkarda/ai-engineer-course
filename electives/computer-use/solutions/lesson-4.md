# Solutions: Lesson 4 — DOM vs. Vision Grounding & Action Spaces

## Exercise 1 — Element list context

```python
def format_element_list(elements: list[DOMElement]) -> str:
    """
    Format interactive elements as a compact string for LLM context.
    
    Output format: "e001 button[Submit] e002 textbox[Search] e003 link[Home]"
    
    Design notes:
    - Short IDs (e001) are unambiguous and token-efficient.
    - role[label] gives the model both semantic type and text content.
    - Elements without labels fall back to selector truncated to 20 chars.
    """
    parts: list[str] = []
    for el in elements:
        label = el.label or el.selector[:20]
        parts.append(f"{el.element_id} {el.role}[{label}]")
    return "  ".join(parts)


# Tests
def test_format_element_list_basic():
    elements = [
        DOMElement(
            element_id="e001", role="button", label="Submit",
            selector="button[type=submit]", bounding_box=None,
            center_x=None, center_y=None,
        ),
        DOMElement(
            element_id="e002", role="textbox", label="Search",
            selector="input#q", bounding_box=None,
            center_x=None, center_y=None,
        ),
    ]
    result = format_element_list(elements)
    assert "e001 button[Submit]" in result
    assert "e002 textbox[Search]" in result

def test_format_element_list_no_label():
    elements = [
        DOMElement(
            element_id="e001", role="button", label=None,
            selector="div.complex-class-name-here > button",
            bounding_box=None, center_x=None, center_y=None,
        ),
    ]
    result = format_element_list(elements)
    assert "e001 button[" in result
    # Label falls back to truncated selector
    assert len(result) < 80
```

---

## Exercise 2 — Coordinate clamping

```python
from pydantic import model_validator

# Updated ClickCoordinateAction with clamping:
class ClickCoordinateAction(BaseModel):
    type: Literal["click_coordinate"] = "click_coordinate"
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    viewport_width: int = Field(default=1280, ge=1)
    viewport_height: int = Field(default=800, ge=1)

    @model_validator(mode="after")
    def clamp_to_viewport(self) -> "ClickCoordinateAction":
        self.x = max(0, min(self.x, self.viewport_width - 1))
        self.y = max(0, min(self.y, self.viewport_height - 1))
        return self


def clamp_coordinate(
    x: int, y: int, width: int, height: int
) -> tuple[int, int]:
    """Standalone clamp for use outside of Pydantic models."""
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


# Tests
def test_clamp_within_bounds():
    assert clamp_coordinate(500, 400, 1280, 800) == (500, 400)

def test_clamp_exceeds_width():
    assert clamp_coordinate(2000, 400, 1280, 800) == (1279, 400)

def test_clamp_negative():
    assert clamp_coordinate(-10, -5, 1280, 800) == (0, 0)

def test_action_clamps_on_construction():
    action = ClickCoordinateAction(x=9999, y=9999, viewport_width=1280, viewport_height=800)
    assert action.x == 1279
    assert action.y == 799
```

---

## Exercise 3 — Schema generation

```python
from pydantic import TypeAdapter
import json

# Action leaf models
ACTION_MODELS = [
    ClickCoordinateAction,
    ClickElementAction,
    TypeAction,
    ScrollAction,
    NavigateAction,
    ScreenshotAction,
    DoneAction,
]

def generate_action_spec() -> str:
    """Generate a human-readable action specification for the LLM."""
    lines = ["Available actions:\n"]
    for model in ACTION_MODELS:
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        # Filter out the 'type' literal field
        fields = [
            f"{k}({'required' if k in required else 'optional'})"
            for k in props if k != "type"
        ]
        action_name = schema.get("title", model.__name__)
        lines.append(f"  {action_name}: {', '.join(fields) if fields else 'no params'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_action_spec())
```

Example output:
```
Available actions:

  ClickCoordinateAction: x(required), y(required)
  ClickElementAction: element_id(required)
  TypeAction: text(required), target_selector(optional)
  ScrollAction: x(optional), y(optional), direction(optional), clicks(optional)
  NavigateAction: url(required)
  ScreenshotAction: no params
  DoneAction: result(optional)
```

---

## Exercise 4 — Hybrid grounding

```python
async def dispatch_hybrid_click(
    page,
    x: int,
    y: int,
    elements: list[DOMElement],
    model_width: int = 1280,
    model_height: int = 800,
    tolerance: int = 20,
) -> dict:
    """
    Try DOM grounding first (more reliable), fall back to coordinate click.
    """
    # Scale coordinates to browser viewport
    bw = page.viewport_size["width"]
    bh = page.viewport_size["height"]
    sx, sy = scale_coordinate(x, y, model_width, model_height, bw, bh)

    # Find DOM element whose center is within tolerance of (sx, sy)
    best: DOMElement | None = None
    best_dist = float("inf")
    for el in elements:
        if el.center_x is None or el.center_y is None:
            continue
        dist = ((el.center_x - sx) ** 2 + (el.center_y - sy) ** 2) ** 0.5
        if dist < tolerance and dist < best_dist:
            best = el
            best_dist = dist

    if best is not None:
        import logging
        logging.getLogger(__name__).info(
            "hybrid_click: using DOM  selector=%r  dist=%.1f", best.selector, best_dist
        )
        await page.locator(best.selector).click()
        return {"method": "dom", "selector": best.selector, "dist": best_dist}
    else:
        logging.getLogger(__name__).info(
            "hybrid_click: fallback to coordinate  (%d,%d)", sx, sy
        )
        await page.mouse.click(sx, sy)
        return {"method": "coordinate", "x": sx, "y": sy}
```
