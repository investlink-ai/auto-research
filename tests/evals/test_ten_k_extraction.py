"""10-K extraction-quality eval incl. G-Eval on guidance_tone (spec section 14.1).

Run locally:  uv run pytest -m eval tests/evals/test_ten_k_extraction.py -v
Skips without ANTHROPIC_API_KEY (no real key in CI), like the
entity-resolution eval.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
from pydantic import BaseModel

from auto_research.eval.baseline import run_worker_eval
from auto_research.eval.geval import build_geval_metric
from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["ten_k"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(_SPEC.gold_path, worker="ten_k", thresholds=_SPEC.default_thresholds)


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — 10-K extraction eval needs a real LLM")


@pytest.fixture(scope="module")
def aggregate(gold_set: GoldSet) -> dict[str, Any]:
    _require_key()
    return run_worker_eval(_SPEC, gold_set)


@pytest.fixture(scope="module")
def outputs(gold_set: GoldSet) -> list[tuple[GoldSample, BaseModel | None]]:
    _require_key()
    return [
        (s, _SPEC.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id)) for s in gold_set.samples
    ]


def test_claim_list_f1_meets_threshold(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    threshold = gold_set.thresholds["min_f1"]
    below = {
        f: aggregate[f]
        for f, kind in _SPEC.field_metrics.items()
        if kind == "claim_list"
        and aggregate[f] == aggregate[f]  # not NaN
        and aggregate[f] < threshold
    }
    assert not below, f"10-K fields below F1 {threshold}: {below}"


def test_hallucination_rate_is_low(aggregate: dict[str, Any]) -> None:
    assert aggregate["hallucination_rate"] <= 0.1, aggregate["hallucination_rate"]


@pytest.mark.parametrize("field", ["guidance_tone"])
def test_subjective_field_geval(
    field: str,
    gold_set: GoldSet,
    outputs: list[tuple[GoldSample, BaseModel | None]],
) -> None:
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
        score = metric.score
        if score is not None and score < metric.threshold:
            failures.append(
                f"{sample.doc_id}: {field} G-Eval {score:.2f} ({metric.reason})"
            )
    assert not failures, "\n".join(failures)


def test_language_novelty_score_spearman_reported(aggregate: dict[str, Any]) -> None:
    v = aggregate["language_novelty_score"]
    assert math.isnan(v) or -1.0 <= v <= 1.0, v
