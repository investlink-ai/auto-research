"""Shared assertion helpers for the per-worker extraction-eval suites.

The four `test_{worker}_extraction.py` suites are thin wrappers over the
same scoring harness; without these helpers each re-implemented the
key-skip, NaN guard, threshold lookup, and G-Eval loop, so a fix to any
one (e.g. the NaN guard) had to be applied four times. Thresholds are read
from the gold set's `thresholds` dict (sourced from
`WorkerEvalSpec.default_thresholds`) so the registry is the single source
of truth — no magic numbers in the suite bodies.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest

from auto_research.eval.geval import build_geval_metric
from auto_research.eval.gold import GoldSample, GoldSet
from auto_research.eval.registry import WorkerEvalSpec


def require_anthropic_key(worker: str) -> None:
    """Skip the calling eval test when no real LLM key is configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip(f"ANTHROPIC_API_KEY not set — {worker} extraction eval needs a real LLM")


def assert_claim_list_f1(
    spec: WorkerEvalSpec, aggregate: dict[str, Any], gold_set: GoldSet
) -> None:
    """Every claim_list field's mean F1 must meet the gold set's min_f1.

    NaN aggregates (a field with zero scoreable samples) are skipped here —
    `assert_hallucination_rate` is the gate that catches an all-quarantined
    worker, so this assertion does not double-report it as an F1 failure.
    """
    threshold = gold_set.thresholds["min_f1"]
    below = {
        f: aggregate[f]
        for f, kind in spec.field_metrics.items()
        if kind == "claim_list" and math.isfinite(aggregate[f]) and aggregate[f] < threshold
    }
    assert not below, f"{spec.worker} claim_list fields below F1 {threshold}: {below}"


def assert_exact_field(aggregate: dict[str, Any], field: str, gold_set: GoldSet) -> None:
    """An exact-match field must meet min_exact, with a clear message when the
    aggregate is NaN because every sample was quarantined (rather than a
    confusing `nan >= 0.8` failure)."""
    threshold = gold_set.thresholds["min_exact"]
    value = aggregate[field]
    assert math.isfinite(value), (
        f"{field} aggregate is NaN — every scored sample was quarantined "
        f"({aggregate.get('quarantined')}/{aggregate.get('n')}); no exact-match signal"
    )
    assert value >= threshold, f"{field} exact-match {value} < {threshold}"


def assert_hallucination_rate(aggregate: dict[str, Any], gold_set: GoldSet) -> None:
    rate = aggregate["hallucination_rate"]
    threshold = gold_set.thresholds["max_hallucination_rate"]
    assert rate <= threshold, f"hallucination (quarantine) rate {rate} > {threshold}"


def assert_geval_fields(
    outputs: list[tuple[GoldSample, Any]], gold_set: GoldSet, fields: list[str]
) -> None:
    """Run G-Eval over each subjective `field` for every gold sample that
    carries a rubric note for it. Skips samples whose output was quarantined
    (None) or whose subjective Claim field is absent/None."""
    from deepeval.test_case import LLMTestCase

    threshold = gold_set.thresholds["min_geval"]
    failures: list[str] = []
    for field in fields:
        metric = build_geval_metric(field, threshold=threshold)
        for sample, out in outputs:
            if out is None or field not in sample.subjective:
                continue
            claim = getattr(out, field, None)
            if claim is None:  # non-mandatory Claim field absent on this output
                continue
            tc = LLMTestCase(
                input=f"Assess {field} for this passage.",
                actual_output=f"{claim.citation.source_quote} (confidence={claim.confidence})",
                context=[sample.raw_doc, sample.subjective[field].get("rubric_note", "")],
            )
            metric.measure(tc)
            score = metric.score
            if score is not None and score < metric.threshold:
                failures.append(f"{sample.doc_id}: {field} G-Eval {score:.2f} ({metric.reason})")
    assert not failures, "\n".join(failures)
