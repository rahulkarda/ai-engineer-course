# Week 8 Day 7 — Solutions: Review Exercises (Assessment Prep)

## Exercise 1: Zero-Shot vs Few-Shot — Clear Explanation

### Zero-Shot Prompting

Give the model a task description only — no examples of the desired input/output pattern.

```
System: Extract the sentiment from the following text. Return one of: positive, neutral, negative.
User: "The new product update is decent but the old version was faster."
```

**When to use zero-shot:**
- The task is conceptually simple and the model has seen it during training (sentiment,
  translation, summarisation, basic classification)
- You are prototyping and want to measure baseline performance before investing in examples
- Prompt real estate is expensive (long documents fill context quickly)
- The output format is self-evident (single label, yes/no, simple JSON)

**Limitation:** For tasks with idiosyncratic or domain-specific output formats, the model
may produce valid-looking but structurally wrong responses. A model asked to output a
custom JSON schema zero-shot may invent plausible-but-wrong field names.

---

### Few-Shot Prompting

Include 2-5 worked examples (input → output pairs) before the real task. The model
pattern-matches to the demonstrated format.

```
System: Extract the event title and date from the text.

Examples:

Text: "We have a board meeting on December 3rd at 2pm."
Output: {"title": "board meeting", "date": "2025-12-03"}

Text: "Reminder: product launch is scheduled for the 15th of next month."
Output: {"title": "product launch", "date": "2025-09-15"}

Now extract from:
Text: "Join us for the holiday party on January 10th."
Output:
```

**When to use few-shot:**
- The output has a strict or unusual format the model has not been trained to produce
  natively (proprietary schemas, domain-specific codes, non-standard JSON structures)
- Zero-shot produces inconsistent structure (varies across calls)
- The task requires subtle judgment calls best demonstrated by example (medical coding,
  legal entity tagging, financial classification)
- Edge cases need to be covered — a few-shot example can show how to handle nulls,
  abbreviations, or ambiguous input

**Cost vs quality trade-off:** Few-shot examples cost input tokens. For bulk processing,
use prompt caching (Exercise 3, Day 6) so the examples are cached after the first call.

**Rule of thumb:** Start zero-shot. If error rate > 5% or output format is inconsistent
after 20 test cases, add few-shot examples targeting the failure modes you observed.

---

## Exercise 2: Debug the Broken Repair Loop

### Broken Code

```python
# BUG: The error message is never passed to the LLM — it only knows
# "please fix" but not WHAT to fix. The model will often repeat the
# same mistake because it has no information about what went wrong.

def extract_with_repair(client: LLMClient, text: str) -> dict:
    system = "Extract event info as JSON."
    user = f"Text: {text}"
    
    for attempt in range(3):
        response = client.complete(system=system, user=user)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # BUG IS HERE: we reassign `user` but we lose the error details.
            # The model only sees "please fix your response" — no error context.
            user = "Please fix your response."   # <-- broken line
    
    raise RuntimeError("Failed after 3 attempts")
```

**The bug in one sentence:** `user` is replaced with a vague instruction that discards
both the original text *and* the specific parse error, so the model cannot know what to
correct.

### Fixed Code

```python
import json
import logging

from week08.protocol import LLMClient

logger = logging.getLogger(__name__)


def extract_with_repair(client: LLMClient, text: str) -> dict:
    """Extract JSON from text with automatic repair on parse failure."""
    system = "Extract event info as JSON. Return ONLY valid JSON, no markdown."
    original_user = f"Text: {text}"

    for attempt in range(3):
        # On first attempt: use the original prompt.
        # On retry: include both the original request AND the specific error.
        if attempt == 0:
            user = original_user
        else:
            # FIX: pass (1) what we originally asked, (2) the exact error.
            user = (
                f"{original_user}\n\n"
                f"Your previous response could not be parsed as JSON.\n"
                f"Error: {last_error}\n\n"  # <-- this is the key fix
                "Please return only valid JSON, with no extra text or code fences."
            )

        logger.debug("extract_with_repair attempt %d/3", attempt + 1)
        response = client.complete(system=system, user=user)

        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError as exc:
            last_error = str(exc)  # capture for next iteration
            logger.warning("attempt %d: parse error: %s", attempt + 1, last_error)

    raise RuntimeError(f"Failed after 3 attempts. Last error: {last_error}")
```

**Two additional improvements in the fix:**
1. `response.content.strip()` — models sometimes add leading/trailing whitespace or a
   trailing newline that breaks `json.loads`.
2. The system prompt explicitly says "no markdown" — models often wrap JSON in backtick
   fences, which is the most common cause of `JSONDecodeError` in extraction pipelines.

---

## Exercise 3: Medical Document Extractor System Prompt

A production system prompt must cover: task definition, output schema, safety constraints,
PHI handling, confidence scoring, and graceful handling of edge cases.

```
You are a medical document extraction assistant. Your sole purpose is to extract
structured clinical data from medical records and discharge summaries.

## Output Format

Return ONLY valid JSON conforming exactly to this schema. Do not include markdown
fences, explanations, or commentary.

{
  "patient_age": integer or null,
  "patient_sex": "M" | "F" | "other" | null,
  "primary_diagnosis": {
    "code": "ICD-10 code string or null",
    "description": "plain text description"
  },
  "secondary_diagnoses": [
    {"code": "ICD-10 or null", "description": "string"}
  ],
  "medications": [
    {"name": "string", "dose": "string or null", "frequency": "string or null"}
  ],
  "procedures": ["list of procedure names"],
  "lab_values": [
    {"test": "string", "value": "string", "unit": "string or null", "flag": "H" | "L" | "N" | null}
  ],
  "admission_date": "YYYY-MM-DD or null",
  "discharge_date": "YYYY-MM-DD or null",
  "confidence": 0.0-1.0,
  "extraction_warnings": ["list of strings describing ambiguities or missing fields"]
}

## PHI Handling

CRITICAL: Do NOT include in the output:
- Patient name, initials, or nickname
- Date of birth (use age only)
- Social Security Number, MRN, or any government ID
- Street address, city, state, or ZIP
- Phone numbers or email addresses
- Names of treating physicians beyond "attending physician" / "specialist"
- Any other 18 HIPAA identifiers

If the source document contains PHI beyond age/sex, acknowledge its presence in
extraction_warnings but do not copy the value into any field.

## Confidence Scoring

Set confidence based on document clarity:
- 0.90-1.00: Complete document, all fields clearly stated, no ambiguity
- 0.70-0.89: Most fields present; 1-2 fields inferred from context
- 0.50-0.69: Significant gaps or ambiguous terminology; key fields inferred
- 0.00-0.49: Document is fragmentary, illegible, or off-domain; flag in warnings

## Edge Cases

- If a field is absent or illegible: use null, do not guess
- If ICD-10 code is not stated: set code to null, fill description from text
- If the document is not a medical record: set confidence to 0.0 and add
  "Document does not appear to be a medical record" to extraction_warnings
- If medication dosage is unclear: record what is written verbatim in the
  dose field; add a warning

## Safety Constraints

You must NEVER:
- Provide medical advice, diagnosis interpretation, or treatment recommendations
- Modify, correct, or second-guess clinical values (record them as written)
- Invent or estimate values not present in the source document
- Include speculative fields not in the schema

If asked to do any of the above, respond with:
{"error": "Request outside extraction scope", "confidence": 0.0}
```

---

## Exercise 4: Cost Calculation

**Scenario:** Process 1000 documents. Each document requires one LLM call:
- Input: 2000 tokens (system prompt + document text)
- Output: 500 tokens (extracted JSON)

**Pricing (Anthropic, mid-2025):**
- claude-opus-5-5 input:  $15.00 per 1 million tokens
- claude-opus-5-5 output: $75.00 per 1 million tokens

**Calculation:**

```
Input cost:
  1,000 docs × 2,000 tokens/doc = 2,000,000 input tokens
  2,000,000 tokens × $15.00 / 1,000,000 = $30.00

Output cost:
  1,000 docs × 500 tokens/doc = 500,000 output tokens
  500,000 tokens × $75.00 / 1,000,000 = $37.50

Total cost:
  $30.00 + $37.50 = $67.50
```

**With prompt caching (same system prompt for all 1000 docs):**

Assume the system prompt is 500 tokens (out of the 2000 input tokens).
After the first call, those 500 tokens are cached.

```
Call 1 (cache write):
  2,000 tokens × $15.00/1M × 1.25 (write surcharge) = $0.0375  ← write costs 25% more
  500 tokens × $75.00/1M = $0.0375
  Call 1 total: $0.075

Calls 2-1000 (cache read for 500 tokens, uncached for 1500 tokens):
  Cached read:   500 tokens × $15.00/1M × 0.10 = $0.00075 per call
  Uncached input: 1500 tokens × $15.00/1M = $0.0225 per call
  Output:         500 tokens × $75.00/1M = $0.0375 per call
  Per call total: $0.06075
  999 calls: $60.69

Total with caching: $0.075 + $60.69 = $60.76

Savings vs uncached: $67.50 - $60.76 = $6.74 (~10% saving)
```

**Takeaway:** Prompt caching gives roughly 10% savings when the cached block is 25%
of total input tokens. Savings scale with the fraction of input that is static. For
a 2000-token system prompt reused with short 100-token user messages, savings can
exceed 80%.

**Code to verify:**
```python
def calculate_cost(
    num_docs: int,
    input_tokens_per_doc: int,
    output_tokens_per_doc: int,
    input_price_per_million: float = 15.0,
    output_price_per_million: float = 75.0,
) -> float:
    total_input = num_docs * input_tokens_per_doc
    total_output = num_docs * output_tokens_per_doc
    cost = (
        total_input * input_price_per_million / 1_000_000
        + total_output * output_price_per_million / 1_000_000
    )
    return round(cost, 2)


assert calculate_cost(1000, 2000, 500) == 67.50
print(f"Total cost: ${calculate_cost(1000, 2000, 500):.2f}")
# Total cost: $67.50
```
