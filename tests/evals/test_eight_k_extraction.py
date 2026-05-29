"""8-K extraction-quality eval (spec section 14.1).

Run locally:  uv run pytest -m eval tests/evals/test_eight_k_extraction.py -v
Skips without ANTHROPIC_API_KEY (no real key in CI), like the
entity-resolution eval.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from auto_research.eval.baseline import run_worker_eval
from auto_research.eval.gold import GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["eight_k"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(_SPEC.gold_path, worker="eight_k", thresholds=_SPEC.default_thresholds)


@pytest.fixture(scope="module")
def aggregate(gold_set: GoldSet) -> dict[str, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — 8-K extraction eval needs a real LLM")
    return run_worker_eval(_SPEC, gold_set)


def test_claim_list_f1_meets_threshold(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    threshold = gold_set.thresholds["min_f1"]
    below = {
        f: aggregate[f]
        for f, kind in _SPEC.field_metrics.items()
        if kind == "claim_list"
        and aggregate[f] == aggregate[f]  # not NaN
        and aggregate[f] < threshold
    }
    assert not below, f"8-K fields below F1 {threshold}: {below}"


def test_event_classification_exact_match(aggregate: dict[str, Any]) -> None:
    assert aggregate["event_classification"] >= 0.8, aggregate["event_classification"]


def test_hallucination_rate_is_low(aggregate: dict[str, Any]) -> None:
    assert aggregate["hallucination_rate"] <= 0.1, aggregate["hallucination_rate"]
