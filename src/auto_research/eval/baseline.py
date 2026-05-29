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
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_research.eval.gold import GoldSample, GoldSet
from auto_research.eval.hallucination import grounding_outcome
from auto_research.eval.metrics import claim_list_f1, exact_match, spearman
from auto_research.eval.registry import WorkerEvalSpec


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
            scores[fname] = exact_match(
                pred_val.value if hasattr(pred_val, "value") else pred_val, gold_val
            )
        elif kind in ("claim_list", "claim_presence"):
            scores[fname] = claim_list_f1(
                _claim_quotes(pred_val), list(gold_val or [])
            ).f1
        elif kind == "numeric":
            scores[fname] = float(pred_val)
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
    for fname, kind in spec.field_metrics.items():
        if kind == "numeric":
            agg[fname] = spearman(numeric_pred.get(fname, []), numeric_gold.get(fname, []))
        else:
            vals = [
                r[fname] for r in per_sample if isinstance(r.get(fname), (int, float))
            ]
            agg[fname] = sum(vals) / len(vals) if vals else float("nan")
    return agg


def capture_baseline(spec: WorkerEvalSpec, gold_set: GoldSet, **kwargs: Any) -> Path:
    """Run `run_worker_eval` and write results to eval/baselines/."""
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
