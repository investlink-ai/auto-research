"""Deterministic, hermetic field-level extraction metrics.

No LLM, no network — every function here is a pure function of its inputs
so the unit tests run in CI without an API key. The LLM-judge metrics
(subjective `Claim` fields) live in `geval.py` and run only under
`@pytest.mark.eval`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

import numpy as np

_WS = re.compile(r"\s+")


class PRF(NamedTuple):
    precision: float
    recall: float
    f1: float


def _norm(quote: str) -> str:
    """Whitespace-collapsed, casefolded quote key — mirrors the flex match
    the citation-grounding walker uses so eval matching and production
    grounding agree on what 'the same quote' means."""
    return _WS.sub(" ", quote).strip().casefold()


def claim_list_f1(predicted: list[str], gold: list[str]) -> PRF:
    """Set-with-multiplicity F1 over normalized quote strings.

    Multiplicity matters: two distinct gold mentions sharing a quote should
    need two predicted matches. Empty-vs-empty is a perfect score (the
    worker correctly found nothing), not a divide-by-zero.
    """
    if not predicted and not gold:
        return PRF(1.0, 1.0, 1.0)
    pred_c = Counter(_norm(p) for p in predicted)
    gold_c = Counter(_norm(g) for g in gold)
    tp = sum((pred_c & gold_c).values())
    fp = sum(pred_c.values()) - tp
    fn = sum(gold_c.values()) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return PRF(precision, recall, f1)


def exact_match(predicted: object, gold: object) -> float:
    return 1.0 if predicted == gold else 0.0


def spearman(predicted: list[float], gold: list[float]) -> float:
    """Spearman rank correlation. Returns NaN for <2 points or zero
    variance (rank correlation is undefined there)."""
    if len(predicted) != len(gold):
        raise ValueError("spearman: length mismatch")
    if len(predicted) < 2:
        return float("nan")
    pr = _rankdata(np.asarray(predicted, dtype=float))
    gr = _rankdata(np.asarray(gold, dtype=float))
    if pr.std() == 0 or gr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, gr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean of their ranks)."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
