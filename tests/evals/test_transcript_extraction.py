"""Transcript extraction-quality eval incl. G-Eval on subjective tone/evasiveness."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set
from auto_research.eval.registry import WORKER_EVALS
from tests.evals._eval_asserts import assert_geval_fields, require_anthropic_key

pytestmark = pytest.mark.eval

_SPEC = WORKER_EVALS["transcript"]


@pytest.fixture(scope="module")
def gold_set() -> GoldSet:
    return load_gold_set(
        _SPEC.gold_path, worker="transcript", thresholds=_SPEC.default_thresholds
    )


@pytest.fixture(scope="module")
def outputs(gold_set: GoldSet) -> list[tuple[GoldSample, BaseModel | None]]:
    require_anthropic_key("transcript")
    return [
        (s, _SPEC.extract_fn(raw_doc=s.raw_doc, doc_id=s.doc_id)) for s in gold_set.samples
    ]


def test_subjective_field_geval(
    gold_set: GoldSet,
    outputs: list[tuple[GoldSample, BaseModel | None]],
) -> None:
    assert_geval_fields(outputs, gold_set, list(_SPEC.subjective_fields))
