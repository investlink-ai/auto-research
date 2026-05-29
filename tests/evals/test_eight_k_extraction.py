"""8-K extraction-quality eval (spec section 14.1).

Run locally:  uv run pytest -m eval tests/evals/test_eight_k_extraction.py -v
Skips without ANTHROPIC_API_KEY (no real key in CI), like the
entity-resolution eval.
"""

from __future__ import annotations

from typing import Any

import pytest

from auto_research.eval.baseline import run_worker_eval
from auto_research.eval.gold import GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS
from tests.evals._eval_asserts import (
    assert_claim_list_f1,
    assert_exact_field,
    assert_hallucination_rate,
    require_anthropic_key,
)

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["eight_k"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(_SPEC.gold_path, worker="eight_k", thresholds=_SPEC.default_thresholds)


@pytest.fixture(scope="module")
def aggregate(gold_set: GoldSet) -> dict[str, Any]:
    require_anthropic_key("8-K")
    return run_worker_eval(_SPEC, gold_set)


def test_claim_list_f1_meets_threshold(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    assert_claim_list_f1(_SPEC, aggregate, gold_set)


def test_event_classification_exact_match(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    assert_exact_field(aggregate, "event_classification", gold_set)


def test_hallucination_rate_is_low(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    assert_hallucination_rate(aggregate, gold_set)
