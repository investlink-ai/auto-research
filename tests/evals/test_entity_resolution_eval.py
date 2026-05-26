"""Entity-resolution F1 eval against the hand-built gold set.

Marked `@pytest.mark.eval` — excluded from default CI runs (no real
Anthropic key in CI). Run locally via:

    pytest -m eval tests/evals/test_entity_resolution_eval.py -v

Embeddings use BGE (in-process, deterministic) so the eval result is
attributable to the disambiguator's behavior rather than to upstream
embedding flakiness. Voyage in production may shift the candidate
slate slightly; the equivalent live eval (when added) will re-measure
F1 on the real backend.

Acceptance gate: micro-accuracy >= `min_f1` in the gold set's thresholds
(currently 0.85). For single-label classification with one prediction
per sample, micro-accuracy = micro-F1.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from auto_research.extract.embeddings import EmbeddingAdapter
from auto_research.extract.entity_resolution import (
    EntityResolution,
    EntityResolver,
)

pytestmark = pytest.mark.eval


def _gold_path() -> Path:
    """Anchor the gold set on the project root via pyproject.toml walk."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "eval" / "baselines" / "entity_resolution__gold.json"
    raise FileNotFoundError("project root with pyproject.toml not found")


@pytest.fixture(scope="module")
def gold_set() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_gold_path().read_text())
    return data


@pytest.fixture(scope="module")
def resolver() -> EntityResolver:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — entity-resolution eval needs a real LLM")
    return EntityResolver(
        adapter=EmbeddingAdapter(backend="bge"),
        top_k=3,
        usd_cap=5.0,
    )


@pytest.fixture(scope="module")
def resolutions(
    resolver: EntityResolver, gold_set: dict[str, Any]
) -> list[tuple[dict[str, Any], EntityResolution]]:
    samples: list[dict[str, Any]] = gold_set["samples"]
    return [(s, resolver.resolve(s["mention_text"])) for s in samples]


def test_resolution_accuracy_meets_threshold(
    gold_set: dict[str, Any],
    resolutions: list[tuple[dict[str, Any], EntityResolution]],
) -> None:
    threshold = float(gold_set["thresholds"]["min_f1"])
    correct = 0
    wrong: list[str] = []
    for sample, result in resolutions:
        expected = sample["expected_ticker"]  # None for "unknown"
        if expected == result.resolved_ticker:
            correct += 1
        else:
            wrong.append(
                f"  {sample['mention_id']}: expected={expected!r} "
                f"got={result.resolved_ticker!r} reasoning={result.reasoning!r}"
            )
    accuracy = correct / len(resolutions)
    assert accuracy >= threshold, (
        f"entity-resolution accuracy {accuracy:.3f} below threshold {threshold} "
        f"({correct}/{len(resolutions)} correct). Misses:\n"
        + "\n".join(wrong)
    )


def test_every_resolution_carries_audit_reasoning(
    resolutions: list[tuple[dict[str, Any], EntityResolution]],
) -> None:
    """Acceptance: 'Disambiguator stores reasoning per resolution for audit'."""
    missing = [
        sample["mention_id"]
        for sample, result in resolutions
        if not result.reasoning.strip()
    ]
    assert not missing, f"resolutions with empty reasoning: {missing}"


def test_unknowns_do_not_false_confidently_match(
    resolutions: list[tuple[dict[str, Any], EntityResolution]],
) -> None:
    """Acceptance: '`unknown` is an allowed output (no false-confident matches)'.

    Soft gate: at least half of the gold samples whose expected_ticker is
    None must resolve to None. False-confident matches on `unknown` gold
    is the failure mode the issue is explicitly defending against.
    """
    unknown_gold = [
        (s, r) for s, r in resolutions if s["expected_ticker"] is None
    ]
    if not unknown_gold:
        pytest.skip("no `unknown` samples in gold set")
    correctly_unknown = sum(
        1 for _, r in unknown_gold if r.resolved_ticker is None
    )
    rate = correctly_unknown / len(unknown_gold)
    assert rate >= 0.5, (
        f"only {correctly_unknown}/{len(unknown_gold)} 'unknown' gold mentions "
        f"resolved to None ({rate:.2%}); disambiguator is making false-"
        "confident matches"
    )


def test_candidate_slate_size_is_top_k(
    resolutions: list[tuple[dict[str, Any], EntityResolution]],
) -> None:
    """Every non-empty mention should produce exactly top_k candidates."""
    sizes = Counter(len(r.considered) for _, r in resolutions)
    assert sizes == Counter({3: len(resolutions)}), (
        f"unexpected candidate-slate sizes: {sizes}"
    )
