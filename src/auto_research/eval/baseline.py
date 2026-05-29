"""Scoring + baseline capture for the extraction eval harness.

`score_output` is hermetic (pure structured comparison). `run_worker_eval`
and `capture_baseline` run the real worker over a gold set's raw_doc and
therefore need ANTHROPIC_API_KEY; they are invoked from the
@pytest.mark.eval suites and the `auto-research eval capture-baseline` CLI.
G-Eval scoring of subjective fields is layered in by the eval suites (it
needs an LLM judge) — baseline.py records the structured + grounding
scores and leaves subjective scoring to the suite.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_research._io import atomic_write_text, project_root
from auto_research.eval.gold import GoldSample, GoldSet
from auto_research.eval.hallucination import grounding_outcome
from auto_research.eval.metrics import claim_list_f1, exact_match, spearman
from auto_research.eval.registry import WorkerEvalSpec

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Map non-finite floats (NaN/Inf) to None so baselines are valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _claim_quotes(value: Any) -> list[str]:
    """Extract source_quotes from a list[Claim|Mention], a single Claim, or None."""
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
    """Score `predicted` against `gold` for all fields in `spec.field_metrics`.

    Returns a dict mapping each field name to its score plus a special
    ``_grounding`` key with the citation-grounding outcome.
    """
    scores: dict[str, Any] = {}
    for fname, kind in spec.field_metrics.items():
        pred_val = getattr(predicted, fname)
        gold_val = gold.expected.get(fname)
        if kind == "exact":
            # StrEnum compares equal to its str value, so `.value` is not
            # strictly required for the current enum fields — but extracting
            # it keeps the comparison a str-vs-str one for any Enum (incl.
            # IntEnum, whose members would otherwise never equal a str gold).
            scores[fname] = exact_match(
                pred_val.value if hasattr(pred_val, "value") else pred_val, gold_val
            )
        elif kind in ("claim_list", "claim_presence"):
            if gold_val is not None and not isinstance(gold_val, list):
                raise ValueError(
                    f"gold field {fname!r} (kind {kind}) must be a list of "
                    f"expected quote strings, got {type(gold_val).__name__}: "
                    f"{gold_val!r}"
                )
            scores[fname] = claim_list_f1(_claim_quotes(pred_val), gold_val or []).f1
        elif kind == "numeric":
            # A None here means the worker omitted an optional numeric field;
            # treat it as unscoreable (NaN) rather than crashing the whole run
            # on float(None). run_worker_eval drops NaN samples from Spearman.
            scores[fname] = float(pred_val) if pred_val is not None else float("nan")
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"unknown metric kind {kind!r}")
    scores["_grounding"] = grounding_outcome(predicted, gold.raw_doc)
    return scores


def run_worker_eval(
    spec: WorkerEvalSpec, gold_set: GoldSet, **extract_kwargs: Any
) -> dict[str, Any]:
    """Run the real worker over every sample in `gold_set` and aggregate scores.

    Requires ANTHROPIC_API_KEY (or provider equivalent). Not unit-tested
    directly — called from @pytest.mark.eval suites and the CLI.
    """
    per_sample: list[dict[str, Any]] = []
    quarantined = 0
    numeric_pred: dict[str, list[float]] = {}
    numeric_gold: dict[str, list[float]] = {}
    _numeric_fields = [f for f, k in spec.field_metrics.items() if k == "numeric"]
    for s in gold_set.samples:
        out = spec.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id, **extract_kwargs)
        if out is None:
            quarantined += 1
            continue
        row = score_output(spec, out, s)
        per_sample.append(row)
        for fname in _numeric_fields:
            # Pair predicted+gold only when both are present and finite — a
            # sample whose gold omits the field, or whose prediction was None
            # (NaN from score_output), can't contribute to a rank correlation.
            if fname in s.expected and math.isfinite(row[fname]):
                numeric_pred.setdefault(fname, []).append(row[fname])
                numeric_gold.setdefault(fname, []).append(float(s.expected[fname]))
    agg: dict[str, Any] = {"n": len(gold_set.samples), "quarantined": quarantined}
    agg["hallucination_rate"] = (
        quarantined / len(gold_set.samples) if gold_set.samples else 0.0
    )
    for fname, kind in spec.field_metrics.items():
        if kind == "numeric":
            agg[fname] = spearman(numeric_pred.get(fname, []), numeric_gold.get(fname, []))
        else:
            vals = [
                r[fname] for r in per_sample if isinstance(r.get(fname), (int, float))
            ]
            if vals:
                agg[fname] = sum(vals) / len(vals)
            else:
                logger.warning(
                    "field %r has zero scoreable samples; aggregated value will be NaN",
                    fname,
                )
                agg[fname] = float("nan")
    return agg


def capture_baseline(
    spec: WorkerEvalSpec,
    gold_set: GoldSet,
    *,
    baselines_root: Path | None = None,
    **kwargs: Any,
) -> Path:
    """Run `run_worker_eval` and write results to eval/baselines/.

    ``baselines_root`` overrides the default project-rooted output directory;
    pass a ``tmp_path`` in tests to avoid writing into the repository tree.
    """
    agg = run_worker_eval(spec, gold_set, **kwargs)
    if baselines_root is None:
        baselines_root = project_root() / "eval" / "baselines"
    out_path = baselines_root / f"{spec.worker}__{spec.prompt_version}__baseline.json"
    metrics = {k: _json_safe(v) for k, v in agg.items()}
    # Atomic write: a crash mid-write must not leave a truncated baseline that
    # a later regression check would misread as a real (low) score.
    atomic_write_text(out_path, json.dumps({"worker": spec.worker, "metrics": metrics}, indent=2))
    return out_path
