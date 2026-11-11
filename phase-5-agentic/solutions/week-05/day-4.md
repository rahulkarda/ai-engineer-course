# Solutions — Week 05 · Day 4: Reflexion

## Exercise 1: Structured `ReflectionEntry`

```python
from pydantic import BaseModel


class ReflectionEntry(BaseModel):
    lesson: str
    root_cause: str
    fix_strategy: str


def _reflect_structured(self, task: str, code: str, test_output: str) -> ReflectionEntry:
    import json
    prompt = (
        f"Task: {task}\n\nCode:\n```python\n{code}\n```\n\nTest failures:\n{test_output}\n\n"
        "Respond as JSON: {\"lesson\": str, \"root_cause\": str, \"fix_strategy\": str}\n"
        "Each field: 1 sentence max. Be specific to this failure."
    )
    resp = self.client.messages.create(
        model=self.model, max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return ReflectionEntry(**json.loads(raw.strip()))
```

Updated context builder:

```python
def _build_actor_system(self) -> str:
    base = "You are an expert Python programmer."
    if self._reflection_entries:
        entries = self._reflection_entries[-self.max_reflections:]
        formatted = "\n".join(
            f"Attempt {i+1}: Lesson={e.lesson} | Root cause={e.root_cause} | Fix={e.fix_strategy}"
            for i, e in enumerate(entries)
        )
        base += f"\n\nPast failure analysis:\n{formatted}"
    return base
```

---

## Exercise 2: Reflection deduplication

```python
def _is_duplicate(self, new_lesson: str, existing: list[str]) -> bool:
    if not existing:
        return False
    joined = "; ".join(existing)
    prompt = (
        f"Existing lessons: {joined}\n\nNew lesson: {new_lesson}\n\n"
        "Is the new lesson substantially similar to any existing one? Reply only YES or NO."
    )
    resp = self.client.messages.create(
        model=self.model, max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().upper() == "YES"

def _add_reflection(self, lesson: str) -> None:
    if self._is_duplicate(lesson, self._reflections):
        log.info("Duplicate reflection skipped: %s", lesson[:60])
        return
    self._reflections.append(lesson)
    log.info("Reflection added (total=%d)", len(self._reflections))
```

Test:

```python
def test_dedup_fires_on_similar(monkeypatch):
    from day4_reflexion import ReflexionAgent
    agent = ReflexionAgent()
    agent._reflections = ["I should validate input is sorted before processing."]

    # Mock LLM to say YES (duplicate)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="YES")]
    )
    agent.client = mock_client

    agent._add_reflection("I need to check that the list is sorted first.")
    # Duplicate → not added
    assert len(agent._reflections) == 1
```

---

## Exercise 3: Factual evaluator with fuzzy matching

```python
import difflib


def _evaluate_factual(self, answer: str, gold: str) -> tuple[bool, str]:
    """Fuzzy-match evaluator for short factual answers."""
    # Normalise
    a_norm = answer.strip().lower()
    g_norm = gold.strip().lower()

    ratio = difflib.SequenceMatcher(None, a_norm, g_norm).ratio()
    passed = ratio >= 0.85
    detail = f"Similarity={ratio:.2f} (threshold=0.85)"
    log.info("Factual eval: %s — %s", "PASS" if passed else "FAIL", detail)
    return passed, detail
```

---

## Exercise 4: `ReflexionMemoryStore` (cross-session persistence)

```python
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel


class StoredReflections(BaseModel):
    task_hash: str
    reflections: list[str]


class ReflexionMemoryStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self._data: dict[str, StoredReflections] = {}
        self._load()

    def _task_hash(self, task: str) -> str:
        return hashlib.sha256(task.encode()).hexdigest()[:12]

    def _load(self) -> None:
        if self.store_path.exists():
            raw = json.loads(self.store_path.read_text())
            self._data = {k: StoredReflections(**v) for k, v in raw.items()}
            log.info("Loaded %d task records from %s", len(self._data), self.store_path)

    def save(self) -> None:
        self.store_path.write_text(
            json.dumps({k: v.model_dump() for k, v in self._data.items()}, indent=2)
        )

    def get(self, task: str) -> list[str]:
        key = self._task_hash(task)
        return self._data.get(key, StoredReflections(task_hash=key, reflections=[])).reflections

    def add(self, task: str, lesson: str) -> None:
        key = self._task_hash(task)
        if key not in self._data:
            self._data[key] = StoredReflections(task_hash=key, reflections=[])
        self._data[key].reflections.append(lesson)
        self.save()
```

Test with `tmp_path`:

```python
def test_memory_store_persists_and_loads(tmp_path):
    store_file = tmp_path / "reflections.json"
    store = ReflexionMemoryStore(store_file)
    task = "Write a sort function"

    store.add(task, "I should handle empty lists.")
    store.add(task, "I should not mutate the input.")

    # Reload from disk
    store2 = ReflexionMemoryStore(store_file)
    loaded = store2.get(task)
    assert len(loaded) == 2
    assert "empty lists" in loaded[0]
```
