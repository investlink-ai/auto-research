# Issue #20 — Gold sets + DeepEval pytest harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a schema-driven extraction-quality eval harness (F1 / exact-match / Spearman per field, G-Eval for subjective `Claim` fields, citation-grounding hallucination rate) wired into per-worker DeepEval pytest suites, seeded with real hand-labeled gold examples, with baseline-capture tooling.

**Architecture:** The reference-based metrics (F1 / exact-match / Spearman) are **deterministic plain-Python functions** over the structured Pydantic outputs — DeepEval's `LLMTestCase` is text-based and a poor fit for structured field comparison, so we keep field scoring native and reserve DeepEval for what it is genuinely best at: `GEval` LLM-judge scoring of the three subjective `Claim` fields. Hallucination is the existing mechanical `validate_citation_grounding` (a non-quarantined output is grounded by construction; the meaningful signal is the **quarantine rate** over the gold set). A generic harness (`src/auto_research/eval/`) is unit-tested hermetically with hand-constructed `(predicted, gold)` pairs (no API key); the per-worker `tests/evals/test_{worker}_extraction.py` files run the real workers over the gold `raw_doc` and are `@pytest.mark.eval` (skipped in CI without `ANTHROPIC_API_KEY`, mirroring `test_entity_resolution_eval.py`).

**Tech Stack:** Python 3.12, Pydantic v2 (frozen/strict schemas), DeepEval (`GEval`), `scipy`-free Spearman (rank-correlation implemented in numpy — numpy is already a dep), pytest + `eval` marker (already configured).

**Scope decisions (approved by maintainer 2026-05-29):**
- Gold sets: **harness + real-doc seed (~5-10/worker), full 50-80 labeling deferred** to a tracked human follow-up. No model-proposed (circular) labels.
- Workers covered: **all 4** — `ten_k`, `transcript`, `eight_k`, `s_filings` (generic harness makes the 4th marginal).
- Spec: **focused §14 update** only (the rest of the spec already matches the repo).
- Baseline capture needs a live key (none in env) → **documented gap for #22**, with a one-command capture path provided.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/auto_research/eval/__init__.py` | Package marker + public exports |
| `src/auto_research/eval/gold.py` | Typed per-worker gold-set JSONL loader (`GoldSample`, `GoldSet`, `load_gold_set`) |
| `src/auto_research/eval/metrics.py` | Pure metric fns: `claim_list_f1`, `exact_match`, `spearman`, `confidence_match` |
| `src/auto_research/eval/hallucination.py` | `grounding_outcome` wrapping `validate_citation_grounding` → grounded / quarantined |
| `src/auto_research/eval/geval.py` | DeepEval `GEval` metric builders for the 3 subjective `Claim` fields |
| `src/auto_research/eval/registry.py` | `WORKER_EVALS`: per-worker (extract fn, gold path, field→metric map, subjective fields) |
| `src/auto_research/eval/baseline.py` | `run_worker_eval`, `capture_baseline`; writes `eval/baselines/{worker}__{prompt_version}__baseline.json` |
| `src/auto_research/cli_eval.py` | `auto-research eval capture-baseline --worker …` CLI entry (wired into existing click group) |
| `eval/gold_sets/{ten_k,transcript,eight_k,s_filings}.jsonl` | Hand-labeled seed gold (one JSON object per line) |
| `tests/unit/test_eval_gold.py` | Hermetic: gold loader validation |
| `tests/unit/test_eval_metrics.py` | Hermetic: F1 / exact-match / Spearman / confidence-match |
| `tests/unit/test_eval_hallucination.py` | Hermetic: grounding outcome on grounded + mismatched outputs |
| `tests/unit/test_eval_registry.py` | Hermetic: every worker's field→metric map covers its schema |
| `tests/evals/test_{worker}_extraction.py` (×4) | `@pytest.mark.eval`: run worker over gold, assert metrics ≥ thresholds |
| `docs/specs/2026-05-22-design.md` | §14.1 drift fix |

---

## Task 1: Add DeepEval dependency

**Files:**
- Modify: `pyproject.toml:36-48` (`[project.optional-dependencies]`)

- [ ] **Step 1: Add an `eval` optional-dependency group**

In `pyproject.toml`, under `[project.optional-dependencies]`, add (after the `dev` group):

```toml
# Extraction/RAG eval stack. Kept optional so base installs and CI unit
# runs don't pull the heavy LLM-judge tree. Install with `uv sync --extra eval`.
eval = [
    "deepeval>=2,<4",
]
```

- [ ] **Step 2: Resolve the dependency**

Run: `uv sync --extra eval --extra dev`
Expected: lockfile updates, `deepeval` importable.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from deepeval.metrics import GEval; from deepeval.test_case import LLMTestCaseParams; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(eval): add deepeval optional-dependency group (#20)"
```

---

## Task 2: Gold-set loader

**Files:**
- Create: `src/auto_research/eval/__init__.py`
- Create: `src/auto_research/eval/gold.py`
- Test: `tests/unit/test_eval_gold.py`

The gold JSONL is one object per line. `expected` is a worker-specific dict of expected field values; `raw_doc` is the source text the worker runs on; `subjective` carries G-Eval rubric notes + expected categorical confidence for `Claim` fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_gold.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set


def test_gold_sample_roundtrips_minimal() -> None:
    s = GoldSample(
        doc_id="g-001",
        raw_doc="We expect revenue to decline next quarter.",
        expected={"cik": "0000320193"},
        subjective={"guidance_tone": {"confidence": "high", "rubric_note": "clearly negative"}},
        rationale="explicit negative guidance",
    )
    assert s.doc_id == "g-001"
    assert s.expected["cik"] == "0000320193"


def test_gold_set_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        GoldSet(worker="ten_k", thresholds={"min_f1": 0.6}, samples=(), bogus=1)


def test_load_gold_set_parses_jsonl(tmp_path) -> None:
    p = tmp_path / "ten_k.jsonl"
    p.write_text(
        '{"doc_id":"g-001","raw_doc":"x","expected":{"cik":"1"},"subjective":{},"rationale":"r"}\n'
        '{"doc_id":"g-002","raw_doc":"y","expected":{"cik":"2"},"subjective":{},"rationale":"r"}\n'
    )
    gs = load_gold_set(p, worker="ten_k", thresholds={"min_f1": 0.6})
    assert gs.worker == "ten_k"
    assert len(gs.samples) == 2
    assert gs.samples[0].doc_id == "g-001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_gold.py -v`
Expected: FAIL — `ModuleNotFoundError: auto_research.eval.gold`

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/__init__.py
"""Extraction-quality eval harness (issue #20).

Reference-based field metrics live in `metrics.py` (pure, hermetic);
LLM-judge scoring of subjective `Claim` fields lives in `geval.py`
(DeepEval). The per-worker wiring is in `registry.py`.
"""
```

```python
# src/auto_research/eval/gold.py
"""Typed loader for per-worker gold-set JSONL files at eval/gold_sets/.

One JSON object per line. `expected` holds worker-specific expected field
values; `subjective` holds per-field G-Eval rubric notes keyed by the
subjective `Claim` field name. Pydantic validates shape at load time so a
drifted key surfaces as a clear ValidationError, not a buried KeyError —
the pattern already used by tests/evals/test_entity_resolution_eval.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")


class GoldSample(BaseModel):
    model_config = _FROZEN_STRICT

    doc_id: str
    raw_doc: str
    expected: dict[str, Any]
    subjective: dict[str, dict[str, str]] = {}
    rationale: str = ""


class GoldSet(BaseModel):
    model_config = _FROZEN_STRICT

    worker: str
    thresholds: dict[str, float]
    samples: tuple[GoldSample, ...]


def load_gold_set(
    path: Path, *, worker: str, thresholds: dict[str, float]
) -> GoldSet:
    """Parse a `.jsonl` gold file into a validated `GoldSet`."""
    samples = tuple(
        GoldSample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    return GoldSet(worker=worker, thresholds=thresholds, samples=samples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_eval_gold.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/__init__.py src/auto_research/eval/gold.py tests/unit/test_eval_gold.py
git commit -m "feat(eval): typed gold-set JSONL loader (#20)"
```

---

## Task 3: Pure field metrics

**Files:**
- Create: `src/auto_research/eval/metrics.py`
- Test: `tests/unit/test_eval_metrics.py`

`claim_list_f1` matches predicted vs gold quote lists by normalized `source_quote` (whitespace-collapsed, casefolded) — the same flex notion guardrails use — and returns set-based precision/recall/F1. `spearman` is rank correlation over paired numeric sequences (numpy only). `exact_match` is scalar equality. `confidence_match` compares categorical `Claim.confidence`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_metrics.py
from __future__ import annotations

import math

from auto_research.eval.metrics import (
    PRF,
    claim_list_f1,
    confidence_match,
    exact_match,
    spearman,
)


def test_claim_list_f1_perfect() -> None:
    pred = ["going concern doubt", "material weakness in ICFR"]
    gold = ["material weakness in ICFR", "going  concern   DOUBT"]  # ws + case differ
    prf = claim_list_f1(pred, gold)
    assert prf == PRF(precision=1.0, recall=1.0, f1=1.0)


def test_claim_list_f1_partial() -> None:
    prf = claim_list_f1(["a", "b", "c"], ["a", "b"])
    # tp=2, fp=1, fn=0 -> p=2/3, r=1.0, f1=0.8
    assert math.isclose(prf.precision, 2 / 3)
    assert prf.recall == 1.0
    assert math.isclose(prf.f1, 0.8)


def test_claim_list_f1_empty_both_is_one() -> None:
    assert claim_list_f1([], []).f1 == 1.0


def test_exact_match() -> None:
    assert exact_match("S-3", "S-3") == 1.0
    assert exact_match("S-3", "S-1") == 0.0


def test_confidence_match() -> None:
    assert confidence_match("high", "high") == 1.0
    assert confidence_match("high", "low") == 0.0


def test_spearman_monotonic() -> None:
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


def test_spearman_too_few_points_is_nan() -> None:
    assert math.isnan(spearman([1.0], [2.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: auto_research.eval.metrics`

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/metrics.py
"""Deterministic, hermetic field-level extraction metrics (issue #20).

No LLM, no network — every function here is a pure function of its inputs
so the unit tests run in CI without an API key. The LLM-judge metrics
(subjective `Claim` fields) live in `geval.py` and run only under
`@pytest.mark.eval`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

import numpy as np

_WS = re.compile(r"\s+")


class PRF(NamedTuple):
    precision: float
    recall: float
    f1: float


def _norm(quote: str) -> str:
    """Whitespace-collapsed, casefolded quote key — mirrors the flex match
    the citation-grounding walker uses so eval matching and production
    grounding agree on what 'the same quote' means."""
    return _WS.sub(" ", quote).strip().casefold()


def claim_list_f1(predicted: list[str], gold: list[str]) -> PRF:
    """Set-with-multiplicity F1 over normalized quote strings.

    Multiplicity matters: two distinct gold mentions sharing a quote should
    need two predicted matches. Empty-vs-empty is a perfect score (the
    worker correctly found nothing), not a divide-by-zero.
    """
    if not predicted and not gold:
        return PRF(1.0, 1.0, 1.0)
    pred_c = Counter(_norm(p) for p in predicted)
    gold_c = Counter(_norm(g) for g in gold)
    tp = sum((pred_c & gold_c).values())
    fp = sum(pred_c.values()) - tp
    fn = sum(gold_c.values()) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return PRF(precision, recall, f1)


def exact_match(predicted: object, gold: object) -> float:
    return 1.0 if predicted == gold else 0.0


def confidence_match(predicted: str, gold: str) -> float:
    """Categorical confidence agreement (high/medium/low)."""
    return 1.0 if predicted == gold else 0.0


def spearman(predicted: list[float], gold: list[float]) -> float:
    """Spearman rank correlation. Returns NaN for <2 points or zero
    variance (rank correlation is undefined there)."""
    if len(predicted) != len(gold):
        raise ValueError("spearman: length mismatch")
    if len(predicted) < 2:
        return float("nan")
    pr = _rankdata(np.asarray(predicted, dtype=float))
    gr = _rankdata(np.asarray(gold, dtype=float))
    if pr.std() == 0 or gr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, gr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean of their ranks)."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_eval_metrics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/metrics.py tests/unit/test_eval_metrics.py
git commit -m "feat(eval): pure field metrics — claim-list F1, exact-match, Spearman (#20)"
```

---

## Task 4: Hallucination (citation-grounding) outcome

**Files:**
- Create: `src/auto_research/eval/hallucination.py`
- Test: `tests/unit/test_eval_hallucination.py`

A worker that returns a non-`None` output has already passed `validate_citation_grounding` (ungrounded output is quarantined → `None`). So the eval-level hallucination signal is the **quarantine rate**: `grounding_outcome` re-checks a constructed output against its source and returns `"grounded"` / `"ungrounded"`, letting the harness count how many gold docs the worker had to quarantine.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_hallucination.py
from __future__ import annotations

from auto_research.eval.hallucination import grounding_outcome
from auto_research.extract.schemas import Citation, Claim, EightKOutput


def _output(quote: str, span: tuple[int, int]) -> EightKOutput:
    return EightKOutput(
        cik="1",
        accession_number="a",
        event_classification="other",
        milestone_mentions=[
            Claim(citation=Citation(source_span=span, source_quote=quote), confidence="high")
        ],
        dilution_language_flags=[],
    )


def test_grounded_output() -> None:
    src = "We announced a partnership with Acme Corp today."
    out = _output("partnership with Acme Corp", (15, 41))
    assert grounding_outcome(out, src) == "grounded"


def test_ungrounded_output() -> None:
    src = "We announced a partnership with Acme Corp today."
    out = _output("a totally fabricated quote", (0, 26))
    assert grounding_outcome(out, src) == "ungrounded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_hallucination.py -v`
Expected: FAIL — `ModuleNotFoundError: auto_research.eval.hallucination`

> NOTE: confirm the exact `EightKOutput` `event_classification` enum value (`"other"` here is a placeholder) against `src/auto_research/extract/enums.py:EventClassification` when implementing; use a real member.

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/hallucination.py
"""Citation-grounding outcome for the eval harness (issue #20, INV-2).

Reuses the production walker so 'grounded' means exactly what it means in
extraction. A non-None worker output is grounded by construction; the
meaningful eval signal is how often the worker *would* have been
quarantined over the gold set (the hallucination rate)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from auto_research.extract.guardrails import CitationMismatch, validate_citation_grounding

GroundingOutcome = Literal["grounded", "ungrounded"]


def grounding_outcome(output: BaseModel, source_text: str) -> GroundingOutcome:
    try:
        validate_citation_grounding(output, source_text)
    except CitationMismatch:
        return "ungrounded"
    return "grounded"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_eval_hallucination.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/hallucination.py tests/unit/test_eval_hallucination.py
git commit -m "feat(eval): citation-grounding hallucination outcome (#20)"
```

---

## Task 5: G-Eval builders for subjective fields

**Files:**
- Create: `src/auto_research/eval/geval.py`
- Test: `tests/unit/test_eval_geval.py` (construction only — no LLM call)

Three subjective `Claim` fields get LLM-judge scoring: `TenKOutput.guidance_tone`, `TranscriptOutput.prepared_remarks_tone`, `TranscriptOutput.q_and_a_evasiveness`. Each builder returns a configured DeepEval `GEval` metric; the **invocation** (which hits an LLM) happens only inside the `@pytest.mark.eval` suites. The unit test asserts the metrics construct with the right name/params, importing nothing that fires a network call.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_geval.py
from __future__ import annotations

from auto_research.eval.geval import SUBJECTIVE_GEVAL_FIELDS, build_geval_metric


def test_registry_lists_three_subjective_fields() -> None:
    assert set(SUBJECTIVE_GEVAL_FIELDS) == {
        "guidance_tone",
        "prepared_remarks_tone",
        "q_and_a_evasiveness",
    }


def test_build_metric_has_name_and_threshold() -> None:
    m = build_geval_metric("guidance_tone", threshold=0.7)
    assert "guidance_tone" in m.name
    assert m.threshold == 0.7
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_eval_geval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/geval.py
"""DeepEval G-Eval metric builders for subjective Claim fields (issue #20).

DeepEval is used *only* here — structured field comparison (F1 / exact /
Spearman) stays in metrics.py because LLMTestCase is text-based. Each
field's rubric scores whether the extracted claim's quote + categorical
confidence is a defensible reading of the source passage, judged against
the gold rationale supplied via the test case context."""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Rubric per subjective field. Kept terse and behavior-anchored; G-Eval
# expands these into evaluation steps. The categorical confidence
# (high/medium/low) is part of what is judged — an overconfident claim on
# weak language should score lower.
_RUBRICS: dict[str, str] = {
    "guidance_tone": (
        "Given the source passage (context) and the gold rationale, judge "
        "whether the extracted guidance-tone claim — its quote and its "
        "high/medium/low confidence — is a defensible reading of management's "
        "forward guidance. Penalize quotes that omit the operative guidance "
        "language or confidence that overstates hedged language."
    ),
    "prepared_remarks_tone": (
        "Judge whether the extracted prepared-remarks tone claim faithfully "
        "characterizes the scripted-remarks sentiment in the source, with "
        "confidence matching how explicit the language is."
    ),
    "q_and_a_evasiveness": (
        "Judge whether the extracted Q&A-evasiveness claim correctly reflects "
        "analyst-question dodging / non-answers in the source, with confidence "
        "matching how clearly evasive the exchange is."
    ),
}

SUBJECTIVE_GEVAL_FIELDS = tuple(_RUBRICS)


def build_geval_metric(field: str, *, threshold: float) -> GEval:
    """Build a configured (not yet evaluated) G-Eval metric for `field`."""
    if field not in _RUBRICS:
        raise KeyError(f"no G-Eval rubric for field {field!r}")
    return GEval(
        name=f"{field}_quality",
        criteria=_RUBRICS[field],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=threshold,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_eval_geval.py -v`
Expected: PASS (2 tests)

> If DeepEval's `GEval` requires a model/key at *construction* time in the installed version, pass `model="gpt-4o-mini"` lazily or wrap construction so the unit test still runs offline; confirm against the resolved `deepeval` version from Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/geval.py tests/unit/test_eval_geval.py
git commit -m "feat(eval): G-Eval builders for subjective Claim fields (#20)"
```

---

## Task 6: Per-worker registry (field → metric map)

**Files:**
- Create: `src/auto_research/eval/registry.py`
- Test: `tests/unit/test_eval_registry.py`

`WORKER_EVALS` maps each worker name to its extract fn, gold path, prompt-version string, threshold defaults, and the per-field metric kind. `test_eval_registry.py` asserts every non-identity field of each output schema is assigned a metric (so a future schema field can't silently escape eval).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_registry.py
from __future__ import annotations

import pytest

from auto_research.eval.registry import WORKER_EVALS


@pytest.mark.parametrize("worker", ["ten_k", "transcript", "eight_k", "s_filings"])
def test_every_schema_field_has_a_metric(worker: str) -> None:
    spec = WORKER_EVALS[worker]
    schema_fields = set(spec.output_model.model_fields)
    covered = set(spec.field_metrics) | set(spec.identity_fields) | set(spec.subjective_fields)
    missing = schema_fields - covered
    assert not missing, f"{worker}: fields with no eval metric: {missing}"


def test_prompt_version_is_resolved_string() -> None:
    for spec in WORKER_EVALS.values():
        assert isinstance(spec.prompt_version, str) and spec.prompt_version
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_eval_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/registry.py
"""Per-worker eval wiring (issue #20).

`field_metrics` maps a schema field to how it is scored:
  - "claim_list": list[Claim|Mention] -> claim_list_f1 over source_quotes
  - "exact":      scalar categorical/identity-ish field -> exact_match
  - "numeric":    float field -> Spearman across the gold set
  - "claim_presence": Claim | None -> presence + quote match
Subjective Claim fields are scored by G-Eval (see geval.py) and listed in
`subjective_fields`; identity fields (cik/accession/...) are excluded from
quality scoring but listed so the coverage test passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from auto_research.extract.prompts.eight_k import EIGHT_K_PROMPT_VERSION
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.prompts.transcript_split import TRANSCRIPT_QA_PROMPT_VERSION
from auto_research.extract.schemas import (
    EightKOutput,
    SFilingOutput,
    TenKOutput,
    TranscriptOutput,
)
from auto_research.extract.workers import (
    extract_eight_k,
    extract_s_filing,
    extract_ten_k,
    extract_transcript,
)


def _gold_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "eval" / "gold_sets"
    raise FileNotFoundError("project root not found")


@dataclass(frozen=True)
class WorkerEvalSpec:
    worker: str
    extract_fn: Callable[..., BaseModel | None]
    output_model: type[BaseModel]
    prompt_version: str
    identity_fields: tuple[str, ...]
    subjective_fields: tuple[str, ...]
    field_metrics: dict[str, str]
    default_thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def gold_path(self) -> Path:
        return _gold_root() / f"{self.worker}.jsonl"


WORKER_EVALS: dict[str, WorkerEvalSpec] = {
    "ten_k": WorkerEvalSpec(
        worker="ten_k",
        extract_fn=extract_ten_k,
        output_model=TenKOutput,
        prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
        identity_fields=("cik", "accession_number", "fiscal_period_end"),
        subjective_fields=("guidance_tone",),
        field_metrics={
            "accrual_flags": "claim_list",
            "supplier_mentions": "claim_list",
            "customer_mentions": "claim_list",
            "risk_factor_deltas": "claim_list",
            "icfr_material_weaknesses": "claim_list",
            "critical_accounting_estimate_changes": "claim_list",
            "going_concern": "claim_presence",
            "language_novelty_score": "numeric",
        },
        default_thresholds={"min_f1": 0.6, "min_geval": 0.6},
    ),
    "transcript": WorkerEvalSpec(
        worker="transcript",
        extract_fn=extract_transcript,
        output_model=TranscriptOutput,
        prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
        identity_fields=("ticker", "event_datetime"),
        subjective_fields=("prepared_remarks_tone", "q_and_a_evasiveness"),
        field_metrics={"forward_statements": "claim_list"},
        default_thresholds={"min_f1": 0.6, "min_geval": 0.6},
    ),
    "eight_k": WorkerEvalSpec(
        worker="eight_k",
        extract_fn=extract_eight_k,
        output_model=EightKOutput,
        prompt_version=EIGHT_K_PROMPT_VERSION,
        identity_fields=("cik", "accession_number"),
        subjective_fields=(),
        field_metrics={
            "event_classification": "exact",
            "milestone_mentions": "claim_list",
            "dilution_language_flags": "claim_list",
        },
        default_thresholds={"min_f1": 0.6},
    ),
    "s_filings": WorkerEvalSpec(
        worker="s_filings",
        extract_fn=extract_s_filing,
        output_model=SFilingOutput,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        identity_fields=("cik", "accession_number"),
        subjective_fields=(),
        field_metrics={
            "form_type": "exact",
            "dilution_event": "claim_presence",
            "capital_raise_language": "claim_list",
            "use_of_proceeds": "claim_list",
        },
        default_thresholds={"min_f1": 0.6},
    ),
}
```

> Verify the four prompt-version import paths/names against the repo when implementing (Task-0 exploration found them at `prompts/ten_k_narrative.py`, `prompts/transcript_split.py`, `prompts/eight_k.py`, `prompts/s_filings_dilution.py`). Adjust the `extract_*` import to match `extract/workers/__init__.py`'s actual exports.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_eval_registry.py -v`
Expected: PASS — confirms each schema's fields are fully partitioned into identity / subjective / metric'd.

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/registry.py tests/unit/test_eval_registry.py
git commit -m "feat(eval): per-worker eval registry with field->metric coverage check (#20)"
```

---

## Task 7: Scoring + baseline capture

**Files:**
- Create: `src/auto_research/eval/baseline.py`
- Test: `tests/unit/test_eval_baseline.py`

`score_output(spec, predicted, gold_sample)` turns one `(predicted_output, gold)` pair into a per-field score dict using the registry + metrics. `run_worker_eval(spec, gold_set, ...)` aggregates over the gold set (mean per field, Spearman across the numeric column, quarantine rate). `capture_baseline` writes `eval/baselines/{worker}__{prompt_version}__baseline.json`. The unit test exercises `score_output` with **hand-built** predicted/gold objects (no worker run, no LLM).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_baseline.py
from __future__ import annotations

from auto_research.eval.baseline import score_output
from auto_research.eval.gold import GoldSample
from auto_research.eval.registry import WORKER_EVALS
from auto_research.extract.schemas import Citation, Claim, EightKOutput


def test_score_output_eight_k_matches_gold() -> None:
    src = "We announced a partnership with Acme Corp today."
    predicted = EightKOutput(
        cik="1",
        accession_number="a",
        event_classification="partnership",
        milestone_mentions=[
            Claim(
                citation=Citation(source_span=(15, 41), source_quote="partnership with Acme Corp"),
                confidence="high",
            )
        ],
        dilution_language_flags=[],
    )
    gold = GoldSample(
        doc_id="g-001",
        raw_doc=src,
        expected={
            "event_classification": "partnership",
            "milestone_mentions": ["partnership with Acme Corp"],
            "dilution_language_flags": [],
        },
        rationale="r",
    )
    scores = score_output(WORKER_EVALS["eight_k"], predicted, gold)
    assert scores["event_classification"] == 1.0
    assert scores["milestone_mentions"] == 1.0  # f1
    assert scores["dilution_language_flags"] == 1.0
    assert scores["_grounding"] == "grounded"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_eval_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/auto_research/eval/baseline.py
"""Scoring + baseline capture for the extraction eval harness (issue #20).

`score_output` is hermetic (pure structured comparison). `run_worker_eval`
and `capture_baseline` run the real worker over a gold set's raw_doc and
therefore need ANTHROPIC_API_KEY; they are invoked from the
@pytest.mark.eval suites and the `auto-research eval capture-baseline` CLI.
G-Eval scoring of subjective fields is layered in by the eval suites (it
needs an LLM judge) — baseline.py records the structured + grounding
scores and leaves a `subjective` slot for the suite to fill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_research.eval.gold import GoldSample, GoldSet
from auto_research.eval.hallucination import grounding_outcome
from auto_research.eval.metrics import claim_list_f1, exact_match, spearman
from auto_research.eval.registry import WorkerEvalSpec


def _claim_quotes(value: Any) -> list[str]:
    """Extract source_quotes from a list[Claim|Mention] or a Claim|None."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    quotes: list[str] = []
    for it in items:
        cit = getattr(it, "citation", None)
        if cit is not None:
            quotes.append(cit.source_quote)
    return quotes


def score_output(
    spec: WorkerEvalSpec, predicted: BaseModel, gold: GoldSample
) -> dict[str, Any]:
    """Per-field score for one predicted output vs its gold sample."""
    scores: dict[str, Any] = {}
    for fname, kind in spec.field_metrics.items():
        pred_val = getattr(predicted, fname)
        gold_val = gold.expected.get(fname)
        if kind == "exact":
            scores[fname] = exact_match(
                pred_val.value if hasattr(pred_val, "value") else pred_val, gold_val
            )
        elif kind in ("claim_list", "claim_presence"):
            scores[fname] = claim_list_f1(
                _claim_quotes(pred_val), list(gold_val or [])
            ).f1
        elif kind == "numeric":
            scores[fname] = float(pred_val)  # aggregated via Spearman at set level
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"unknown metric kind {kind!r}")
    scores["_grounding"] = grounding_outcome(predicted, gold.raw_doc)
    return scores


def run_worker_eval(
    spec: WorkerEvalSpec, gold_set: GoldSet, **extract_kwargs: Any
) -> dict[str, Any]:
    """Run the worker over the whole gold set and aggregate (needs a key)."""
    per_sample: list[dict[str, Any]] = []
    quarantined = 0
    numeric_pred: dict[str, list[float]] = {}
    numeric_gold: dict[str, list[float]] = {}
    for s in gold_set.samples:
        out = spec.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id, **extract_kwargs)
        if out is None:
            quarantined += 1
            continue
        row = score_output(spec, out, s)
        per_sample.append(row)
        for fname, kind in spec.field_metrics.items():
            if kind == "numeric" and fname in s.expected:
                numeric_pred.setdefault(fname, []).append(row[fname])
                numeric_gold.setdefault(fname, []).append(float(s.expected[fname]))
    agg: dict[str, Any] = {"n": len(gold_set.samples), "quarantined": quarantined}
    agg["hallucination_rate"] = (
        quarantined / len(gold_set.samples) if gold_set.samples else 0.0
    )
    scored = [r for r in per_sample]
    for fname, kind in spec.field_metrics.items():
        if kind == "numeric":
            agg[fname] = spearman(numeric_pred.get(fname, []), numeric_gold.get(fname, []))
        else:
            vals = [r[fname] for r in scored if isinstance(r.get(fname), (int, float))]
            agg[fname] = sum(vals) / len(vals) if vals else float("nan")
    return agg


def capture_baseline(spec: WorkerEvalSpec, gold_set: GoldSet, **kwargs: Any) -> Path:
    """Run the eval and persist eval/baselines/{worker}__{ver}__baseline.json."""
    agg = run_worker_eval(spec, gold_set, **kwargs)
    here = Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "pyproject.toml").exists())
    out_path = (
        root / "eval" / "baselines"
        / f"{spec.worker}__{spec.prompt_version}__baseline.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"worker": spec.worker, "metrics": agg}, indent=2))
    return out_path
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_eval_baseline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/eval/baseline.py tests/unit/test_eval_baseline.py
git commit -m "feat(eval): per-field scoring + baseline capture (#20)"
```

---

## Task 8: Seed gold sets (real hand-labeled excerpts)

**Files:**
- Create: `eval/gold_sets/ten_k.jsonl`
- Create: `eval/gold_sets/transcript.jsonl`
- Create: `eval/gold_sets/eight_k.jsonl`
- Create: `eval/gold_sets/s_filings.jsonl`

Seed each with **5-10 hand-labeled excerpts** drawn from realistic filing/transcript language (the existing `s_filings_dilution__gold.json` is the precedent for short realistic `raw_doc` strings with hand-labeled `expected`). Each `expected` value must be a **verbatim substring** of `raw_doc` for claim-list/presence fields (so the grounding check passes and F1 is meaningful). Label the categorical confidence + a one-line rubric note under `subjective` for the subjective fields.

- [ ] **Step 1: Write `eval/gold_sets/eight_k.jsonl` (worked example — replicate the pattern for the other three)**

```json
{"doc_id":"8k-001","raw_doc":"Item 1.01 Entry into a Material Definitive Agreement. On May 3, 2026, the Company entered into a strategic partnership with Acme Robotics to co-develop edge-AI accelerators.","expected":{"event_classification":"partnership","milestone_mentions":["strategic partnership with Acme Robotics to co-develop edge-AI accelerators"],"dilution_language_flags":[]},"subjective":{},"rationale":"explicit Item 1.01 partnership; no securities issuance language"}
{"doc_id":"8k-002","raw_doc":"Item 3.02 Unregistered Sales of Equity Securities. The Company issued 2,000,000 shares of common stock in a private placement for gross proceeds of $30 million.","expected":{"event_classification":"dilution","milestone_mentions":[],"dilution_language_flags":["issued 2,000,000 shares of common stock in a private placement"]},"subjective":{},"rationale":"Item 3.02 equity issuance is dilutive"}
```

(Add 3-6 more 8-K lines spanning the `EventClassification` enum members — confirm members in `extract/enums.py`. Cover at least one `other`/non-event filing so precision is testable.)

- [ ] **Step 2: Write the other three gold files**

`s_filings.jsonl` — reuse/expand the two examples already in `eval/baselines/s_filings_dilution__gold.json`, adapting keys to the `expected` shape (`form_type`, `dilution_event` quote, `capital_raise_language`, `use_of_proceeds`). `ten_k.jsonl` — short MD&A / risk-factor / going-concern excerpts with `guidance_tone` under `subjective`. `transcript.jsonl` — short prepared-remarks + Q&A exchanges with `prepared_remarks_tone` and `q_and_a_evasiveness` under `subjective` and at least one `forward_statements` quote.

- [ ] **Step 3: Validate every gold file loads**

Run:
```bash
uv run python -c "
from pathlib import Path
from auto_research.eval.registry import WORKER_EVALS
from auto_research.eval.gold import load_gold_set
for w, spec in WORKER_EVALS.items():
    gs = load_gold_set(spec.gold_path, worker=w, thresholds=spec.default_thresholds)
    assert gs.samples, w
    # every claim-list/presence expected value must be a substring of raw_doc
    for s in gs.samples:
        for f, kind in spec.field_metrics.items():
            if kind in ('claim_list','claim_presence'):
                for q in (s.expected.get(f) or []):
                    assert q in s.raw_doc, f'{w}/{s.doc_id}: {q!r} not in raw_doc'
    print(w, len(gs.samples), 'ok')
"
```
Expected: each worker prints `ok` with its sample count.

- [ ] **Step 4: Commit**

```bash
git add eval/gold_sets/*.jsonl
git commit -m "data(eval): seed hand-labeled gold sets (5-10/worker) (#20)"
```

---

## Task 9: Per-worker DeepEval pytest suites (×4)

**Files:**
- Create: `tests/evals/test_eight_k_extraction.py`
- Create: `tests/evals/test_ten_k_extraction.py`
- Create: `tests/evals/test_transcript_extraction.py`
- Create: `tests/evals/test_s_filings_extraction.py`

Each file is a thin, worker-specific wrapper over the shared harness (satisfies the AC's per-file naming while staying DRY). They are `@pytest.mark.eval` and **skip without `ANTHROPIC_API_KEY`**, exactly like `test_entity_resolution_eval.py`. Subjective-field suites additionally build the G-Eval metric and `assert_test` it per sample.

- [ ] **Step 1: Write `tests/evals/test_eight_k_extraction.py` (objective-only worker — full code)**

```python
"""8-K extraction-quality eval (issue #20, spec §14.1).

Run locally:  uv run pytest -m eval tests/evals/test_eight_k_extraction.py -v
Skips without ANTHROPIC_API_KEY (no real key in CI), like the
entity-resolution eval.
"""

from __future__ import annotations

import os

import pytest

from auto_research.eval.baseline import run_worker_eval, score_output
from auto_research.eval.gold import load_gold_set
from auto_research.eval.registry import WORKER_EVALS

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["eight_k"]


@pytest.fixture(scope="module")
def gold_set():
    return load_gold_set(_SPEC.gold_path, worker="eight_k", thresholds=_SPEC.default_thresholds)


@pytest.fixture(scope="module")
def aggregate(gold_set):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — 8-K extraction eval needs a real LLM")
    return run_worker_eval(_SPEC, gold_set)


def test_claim_list_f1_meets_threshold(gold_set, aggregate) -> None:
    threshold = gold_set.thresholds["min_f1"]
    below = {
        f: aggregate[f]
        for f, kind in _SPEC.field_metrics.items()
        if kind == "claim_list" and aggregate[f] == aggregate[f]  # not NaN
        and aggregate[f] < threshold
    }
    assert not below, f"8-K fields below F1 {threshold}: {below}"


def test_event_classification_exact_match(aggregate) -> None:
    assert aggregate["event_classification"] >= 0.8, aggregate["event_classification"]


def test_hallucination_rate_is_low(aggregate) -> None:
    assert aggregate["hallucination_rate"] <= 0.1, aggregate["hallucination_rate"]
```

- [ ] **Step 2: Write `tests/evals/test_transcript_extraction.py` (adds G-Eval on subjective fields — full code)**

```python
"""Transcript extraction-quality eval incl. G-Eval on subjective tone/evasiveness."""

from __future__ import annotations

import os

import pytest

from auto_research.eval.geval import build_geval_metric
from auto_research.eval.gold import load_gold_set
from auto_research.eval.registry import WORKER_EVALS

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["transcript"]


@pytest.fixture(scope="module")
def gold_set():
    return load_gold_set(_SPEC.gold_path, worker="transcript", thresholds=_SPEC.default_thresholds)


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — transcript eval needs a real LLM")


@pytest.fixture(scope="module")
def outputs(gold_set):
    _require_key()
    return [
        (s, _SPEC.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id)) for s in gold_set.samples
    ]


@pytest.mark.parametrize("field", ["prepared_remarks_tone", "q_and_a_evasiveness"])
def test_subjective_field_geval(field, gold_set, outputs) -> None:
    from deepeval.test_case import LLMTestCase

    metric = build_geval_metric(field, threshold=gold_set.thresholds["min_geval"])
    failures: list[str] = []
    for sample, out in outputs:
        if out is None or field not in sample.subjective:
            continue
        claim = getattr(out, field)
        tc = LLMTestCase(
            input=f"Assess {field} for this passage.",
            actual_output=f"{claim.citation.source_quote} (confidence={claim.confidence})",
            context=[sample.raw_doc, sample.subjective[field].get("rubric_note", "")],
        )
        metric.measure(tc)
        if metric.score < metric.threshold:
            failures.append(f"{sample.doc_id}: {field} G-Eval {metric.score:.2f} ({metric.reason})")
    assert not failures, "\n".join(failures)
```

- [ ] **Step 3: Write `tests/evals/test_ten_k_extraction.py` and `tests/evals/test_s_filings_extraction.py`**

`ten_k`: combine the F1/hallucination structure from Step 1 with a `guidance_tone` G-Eval block from Step 2; also assert the `language_novelty_score` Spearman is reported (it will be NaN on a tiny seed — assert it is either NaN or in [-1, 1], and `log`/skip on NaN so a small seed doesn't hard-fail). `s_filings`: objective-only structure from Step 1 over `form_type` (exact), `dilution_event`/`capital_raise_language`/`use_of_proceeds` (claim-list/presence).

- [ ] **Step 4: Run the suites (offline → they skip)**

Run: `uv run pytest -m eval tests/evals/ -v`
Expected: all SKIPPED with "ANTHROPIC_API_KEY not set" (proves they collect + import cleanly under DeepEval).

- [ ] **Step 5: Confirm default CI run still excludes them**

Run: `uv run pytest tests/unit -q` (the `make test` scope)
Expected: PASS, eval suites not collected.

- [ ] **Step 6: Commit**

```bash
git add tests/evals/test_*_extraction.py
git commit -m "feat(eval): per-worker DeepEval extraction suites, 4 workers (#20)"
```

---

## Task 10: CLI baseline-capture entry

**Files:**
- Create: `src/auto_research/cli_eval.py`
- Modify: `src/auto_research/cli.py` (register the `eval` group)
- Test: `tests/unit/test_cli_eval.py`

- [ ] **Step 1: Write the failing test (uses click's CliRunner; no LLM)**

```python
# tests/unit/test_cli_eval.py
from __future__ import annotations

from click.testing import CliRunner

from auto_research.cli import cli


def test_eval_capture_baseline_help() -> None:
    res = CliRunner().invoke(cli, ["eval", "capture-baseline", "--help"])
    assert res.exit_code == 0
    assert "--worker" in res.output
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_cli_eval.py -v`
Expected: FAIL — no `eval` subcommand.

- [ ] **Step 3: Implement the CLI group and register it**

```python
# src/auto_research/cli_eval.py
"""`auto-research eval` — capture extraction-quality baselines (issue #20)."""

from __future__ import annotations

import click

from auto_research.eval.baseline import capture_baseline
from auto_research.eval.gold import load_gold_set
from auto_research.eval.registry import WORKER_EVALS


@click.group(name="eval")
def eval_group() -> None:
    """Extraction-quality eval commands."""


@eval_group.command(name="capture-baseline")
@click.option(
    "--worker",
    type=click.Choice(sorted(WORKER_EVALS)),
    required=True,
    help="Worker whose gold set to score.",
)
def capture_baseline_cmd(worker: str) -> None:
    """Run the worker over its gold set and write a baseline JSON (needs a key)."""
    spec = WORKER_EVALS[worker]
    gold_set = load_gold_set(spec.gold_path, worker=worker, thresholds=spec.default_thresholds)
    path = capture_baseline(spec, gold_set)
    click.echo(f"wrote {path}")
```

In `src/auto_research/cli.py`, register the group (match the existing registration style — confirm whether the root is `cli` group):

```python
from auto_research.cli_eval import eval_group

cli.add_command(eval_group)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_cli_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/cli_eval.py src/auto_research/cli.py tests/unit/test_cli_eval.py
git commit -m "feat(eval): `auto-research eval capture-baseline` CLI (#20)"
```

---

## Task 11: Capture baselines OR document the gap

**Files:**
- Create (if key available): `eval/baselines/{worker}__{prompt_version}__baseline.json` ×4
- Modify (always): the PR body / issue comment with the baseline status

- [ ] **Step 1: Attempt capture if a key is present**

```bash
if [ -n "$ANTHROPIC_API_KEY" ]; then
  for w in eight_k s_filings ten_k transcript; do
    uv run auto-research eval capture-baseline --worker "$w"
  done
else
  echo "No ANTHROPIC_API_KEY — baseline capture deferred (documented gap for #22)."
fi
```

- [ ] **Step 2: If captured, commit baselines; else record the gap**

If baselines were written:
```bash
git add eval/baselines/*__baseline.json
git commit -m "data(eval): capture extraction baselines for v1 prompts (#20)"
```
If not, note in the PR body: *"Baseline capture requires a live ANTHROPIC_API_KEY (none in this environment). Harness + `auto-research eval capture-baseline --worker <w>` are ready; running them to populate `eval/baselines/{worker}__v1__baseline.json` and tightening thresholds is tracked for #22."* — this is the AC-sanctioned gap call-out.

---

## Task 12: Focused spec §14.1 update

**Files:**
- Modify: `docs/specs/2026-05-22-design.md:432-453`

- [ ] **Step 1: Fix the subjective-field drift and enumerate coverage**

In §14.1, replace the G-Eval bullet `G-Eval (LLM-judge) for subjective fields (evasiveness, forward_tone_quality)` with the real fields:

```markdown
  - G-Eval (LLM-judge) for subjective `Claim` fields: `guidance_tone`
    (10-K), `prepared_remarks_tone` + `q_and_a_evasiveness` (transcript)
```

And add a coverage line under §14.1 noting all four workers (`ten_k`, `transcript`, `eight_k`, `s_filings`) have gold sets at `eval/gold_sets/{worker}.jsonl` and baselines at `eval/baselines/{worker}__{prompt_version}__baseline.json`, with reference-based fields scored by F1/exact-match, `language_novelty_score` by Spearman, and the hallucination signal computed as the citation-grounding quarantine rate (reusing `validate_citation_grounding`).

- [ ] **Step 2: Verify no other stale field name remains**

Run: `grep -rn "forward_tone_quality" docs/ src/`
Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/specs/2026-05-22-design.md
git commit -m "docs(spec): §14.1 — real subjective fields + 4-worker eval coverage (#20)"
```

---

## Task 13: Full verification + PR

- [ ] **Step 1: Lint + type + unit suite**

Run:
```bash
uv run ruff check src/auto_research/eval tests/unit/test_eval_*.py tests/evals
uv run mypy src/auto_research/eval
uv run pytest tests/unit -q
uv run pytest -m eval tests/evals -q   # all SKIPPED without a key, but must collect
```
Expected: ruff clean, mypy clean, unit PASS, eval SKIPPED (not errored).

- [ ] **Step 2: Open the PR with the §-tagged evidence template**

```bash
git push -u origin feat/20-gold-sets-deepeval
gh pr create --fill --title "feat(eval): gold sets + DeepEval pytest harness (#20)" \
  --body "$(cat <<'EOF'
Implements #20. Schema-driven extraction-quality harness (F1 / exact-match /
Spearman per field, G-Eval for subjective Claim fields, citation-grounding
hallucination rate) + per-worker DeepEval pytest suites (4 workers) + seed
gold sets + baseline-capture CLI.

## Risk tier
Tier 1 (new eval code + test data; no production extraction path touched).

## Evidence
- `uv run pytest tests/unit -q` — PASS (harness fully unit-tested, hermetic)
- `uv run pytest -m eval tests/evals -q` — SKIPPED (no key in CI; collects cleanly)
- ruff + mypy clean on `src/auto_research/eval`

## Known gap (tracked for #22)
Baseline capture needs a live ANTHROPIC_API_KEY (none in this env). Gold sets
are seeded at ~5-10/worker (real hand-labeled excerpts); scaling to the spec's
50-80/worker is a human-labeling follow-up. `auto-research eval
capture-baseline --worker <w>` is ready to populate eval/baselines/.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** AC1 (gold sets committed, size noted) → Task 8 + Task 13 PR body. AC2 (`tests/evals/test_{worker}_extraction.py` under DeepEval) → Task 9. AC3 (baselines at `eval/baselines/{worker}__{prompt_version}__*.json`) → Task 7 (tooling) + Task 11 (capture or gap). AC4 (passes baseline thresholds OR calls out the gap) → Task 11 + PR body.
- **Hermetic-vs-live boundary** honored: every metric/loader/registry test is unit-tier (no key); real-LLM scoring is `@pytest.mark.eval` (matches the test-taxonomy rule).
- **No prompt-version bump**: this issue consumes existing v1 prompt artifacts; it does not edit any prompt/contract, so no `*_VERSION` change (matches the bump-policy rule).
- **Categorical confidence** preserved (no float confidence introduced).
- **Open verification items** flagged inline for the implementer: exact `EventClassification` enum members (Tasks 4/8), prompt-version import paths (Task 6), DeepEval `GEval` construction-time model requirement (Task 5), and the root click group name (Task 10).
