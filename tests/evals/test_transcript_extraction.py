"""Transcript extraction-quality eval incl. G-Eval on subjective tone/evasiveness."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from auto_research.eval.geval import build_geval_metric
from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["transcript"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(
        _SPEC.gold_path, worker="transcript", thresholds=_SPEC.default_thresholds
    )


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — transcript eval needs a real LLM")


@pytest.fixture(scope="module")
def outputs(gold_set: GoldSet) -> list[tuple[GoldSample, BaseModel | None]]:
    _require_key()
    return [
        (s, _SPEC.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id)) for s in gold_set.samples
    ]


@pytest.mark.parametrize("field", ["prepared_remarks_tone", "q_and_a_evasiveness"])
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
