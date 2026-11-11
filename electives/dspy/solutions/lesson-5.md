# Solutions — Lesson 5: Teleprompters & Advanced Patterns

---

## Exercise 1 — Pydantic structured output

```python
"""lesson5_ex1_pydantic.py"""
from pydantic import BaseModel
import dspy
from typing import Literal


class InvoiceFields(BaseModel):
    vendor: str
    amount: float
    date: str


class ExtractInvoice(dspy.Signature):
    """Extract invoice fields from raw invoice text."""
    invoice_text: str = dspy.InputField()
    result: InvoiceFields = dspy.OutputField()


def test_pydantic_output() -> None:
    """Unit test: mock the predictor, assert Pydantic model is returned."""
    from unittest.mock import MagicMock, patch

    mock_result = MagicMock()
    mock_result.result = InvoiceFields(vendor="Acme Corp", amount=1250.00, date="2026-01-15")

    with patch.object(dspy.TypedPredictor, "__call__", return_value=mock_result):
        predictor = dspy.TypedPredictor(ExtractInvoice)
        result = predictor(invoice_text="Invoice from Acme Corp: $1250.00 on Jan 15, 2026")

    assert isinstance(result.result, InvoiceFields)
    assert result.result.vendor == "Acme Corp"
    assert result.result.amount == 1250.00
```

Note: `dspy.TypedPredictor` handles Pydantic model output fields. For
`Literal`-only signatures, plain `dspy.ChainOfThought` works fine.

---

## Exercise 2 — Assert retry behavior

```python
import dspy
import logging

logger = logging.getLogger(__name__)

class BriefAnswer(dspy.Signature):
    """Answer concisely in at most 15 words."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class BriefQA(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(BriefAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.predict(question=question)
        word_count = len(pred.answer.split())
        if word_count > 15:
            logger.warning(
                "Answer too long (%d words); Assert will trigger retry: %s",
                word_count,
                pred.answer[:60],
            )
        dspy.Assert(
            word_count <= 15,
            f"Answer is {word_count} words; must be ≤ 15",
        )
        return pred
```

Configure retries globally:

```python
dspy.settings.configure(backtrack_to=3)   # up to 3 retry attempts
```

On retry, DSPy appends the assertion message to the prompt:
`"Previous answer violated constraint: Answer is 22 words; must be ≤ 15. Try again."`

---

## Exercise 3 — Suggest vs Assert

Replace `dspy.Assert` with `dspy.Suggest`:

```python
dspy.Suggest(
    len(pred.answer.split()) <= 15,
    f"Answer is {len(pred.answer.split())} words; prefer ≤ 15",
)
```

With `Suggest`:
- During **plain inference**: no error raised; program continues even if violated
- During **optimization**: the optimizer sees the soft penalty and penalizes programs
  that violate it, nudging demo selection away from long-answer traces

`Assert` raises `dspy.AssertionError` and triggers a retry; `Suggest` only influences
the optimizer's scoring during compilation.

---

## Exercise 4 — KNNFewShot unit test

```python
"""pipeline_knn.py"""
from dspy.teleprompt import KNNFewShot
import dspy


def compile_knn(
    program: dspy.Module,
    trainset: list[dspy.Example],
) -> dspy.Module:
    optimizer = KNNFewShot(k=2, trainset=trainset)
    return optimizer.compile(program)
```

```python
"""test_knn.py"""
from unittest.mock import MagicMock, patch
import dspy
from pipeline_knn import compile_knn


def test_compile_knn_called_with_program() -> None:
    mock_compiled = MagicMock()

    with patch("pipeline_knn.KNNFewShot") as MockKNN:
        instance = MagicMock()
        instance.compile.return_value = mock_compiled
        MockKNN.return_value = instance

        program = MagicMock(spec=dspy.Module)
        trainset = [MagicMock(spec=dspy.Example)]

        result = compile_knn(program, trainset)

    MockKNN.assert_called_once_with(k=2, trainset=trainset)
    instance.compile.assert_called_once_with(program)
    assert result is mock_compiled
```

---

## Exercise 5 — Ensemble sketch

```python
"""pipeline_ensemble.py"""
import dspy
from dspy.teleprompt import BootstrapFewShot, Ensemble
from lesson3_metrics import SentimentProgram, exact_match_metric


def build_ensemble(trainset: list[dspy.Example]) -> dspy.Module:
    variants: list[dspy.Module] = []
    for seed in range(3):
        program = SentimentProgram()
        optimizer = BootstrapFewShot(
            metric=exact_match_metric,
            max_bootstrapped_demos=3,
        )
        # Note: BootstrapFewShot doesn't natively support a seed parameter in all
        # versions; vary the trainset order as a proxy for diversity.
        import random
        shuffled = trainset[:]
        random.seed(seed)
        random.shuffle(shuffled)
        compiled = optimizer.compile(program, trainset=shuffled)
        variants.append(compiled)

    ensemble_opt = Ensemble(reduce_fn=dspy.majority)
    return ensemble_opt.compile(variants)
```

Test:

```python
"""test_ensemble.py"""
from unittest.mock import MagicMock, patch
import dspy
from pipeline_ensemble import build_ensemble


def test_build_ensemble_returns_compiled() -> None:
    mock_ensemble_program = MagicMock()

    with (
        patch("pipeline_ensemble.BootstrapFewShot") as MockBSFS,
        patch("pipeline_ensemble.Ensemble") as MockEnsemble,
    ):
        bs_instance = MagicMock()
        bs_instance.compile.return_value = MagicMock(spec=dspy.Module)
        MockBSFS.return_value = bs_instance

        ens_instance = MagicMock()
        ens_instance.compile.return_value = mock_ensemble_program
        MockEnsemble.return_value = ens_instance

        trainset = [MagicMock(spec=dspy.Example) for _ in range(6)]
        result = build_ensemble(trainset)

    assert result is mock_ensemble_program
    assert MockBSFS.call_count == 3    # one optimizer per variant
    assert ens_instance.compile.call_count == 1
```
