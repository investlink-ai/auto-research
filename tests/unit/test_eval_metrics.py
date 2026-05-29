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
