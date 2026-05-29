"""10-K extraction-quality eval incl. G-Eval on guidance_tone (spec section 14.1).

Run locally:  uv run pytest -m eval tests/evals/test_ten_k_extraction.py -v
Skips without ANTHROPIC_API_KEY (no real key in CI), like the
entity-resolution eval.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import BaseModel

from auto_research.eval.baseline import run_worker_eval
from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS
from tests.evals._eval_asserts import (
    assert_claim_list_f1,
    assert_geval_fields,
    assert_hallucination_rate,
    require_anthropic_key,
)

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["ten_k"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(_SPEC.gold_path, worker="ten_k", thresholds=_SPEC.default_thresholds)


@pytest.fixture(scope="module")
def aggregate(gold_set: GoldSet) -> dict[str, Any]:
    require_anthropic_key("10-K")
    return run_worker_eval(_SPEC, gold_set)


@pytest.fixture(scope="module")
def outputs(gold_set: GoldSet) -> list[tuple[GoldSample, BaseModel | None]]:
    require_anthropic_key("10-K")
    return [
        (s, _SPEC.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id)) for s in gold_set.samples
    ]


def test_claim_list_f1_meets_threshold(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    assert_claim_list_f1(_SPEC, aggregate, gold_set)


def test_hallucination_rate_is_low(gold_set: GoldSet, aggregate: dict[str, Any]) -> None:
    assert_hallucination_rate(aggregate, gold_set)


def test_subjective_field_geval(
    gold_set: GoldSet,
    outputs: list[tuple[GoldSample, BaseModel | None]],
) -> None:
    assert_geval_fields(outputs, gold_set, list(_SPEC.subjective_fields))


def test_language_novelty_score_spearman_reported(aggregate: dict[str, Any]) -> None:
    v = aggregate["language_novelty_score"]
    assert math.isnan(v) or -1.0 <= v <= 1.0, v
