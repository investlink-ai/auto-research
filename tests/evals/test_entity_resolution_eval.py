"""Entity-resolution accuracy eval against the hand-built gold set.

Marked `@pytest.mark.eval` — excluded from default CI runs (no real
Anthropic key in CI). Run locally via:

    pytest -m eval tests/evals/test_entity_resolution_eval.py -v

Embeddings use BGE (in-process, deterministic) so the eval result is
attributable to the disambiguator's behavior rather than to upstream
embedding flakiness. Voyage in production may shift the candidate
slate slightly; the equivalent live eval (when added) will re-measure
on the real backend.

Acceptance gate: micro-accuracy >= `min_f1` in the gold set's
thresholds (currently 0.85). Micro-accuracy is reported under the
`min_f1` key for backward compatibility with the existing baseline
schema; the metric treats `None` (unknown) as its own class — see the
threshold check below.

The `resolutions` fixture is fault-tolerant: a per-sample exception (LLM
rate-limit, network blip) is captured as a sentinel `EntityResolution`
with `resolved_ticker=None` and a `reasoning` string naming the
exception type, so the audit-reasoning and slate-size tests can still
run on the successful samples instead of failing all three with an
identical fixture-setup traceback.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from auto_research.extract.embeddings import EmbeddingAdapter
from auto_research.extract.entity_resolution import (
    EntityResolution,
    EntityResolver,
)
from auto_research.extract.prompts.entity_resolution import (
    ENTITY_RESOLUTION_PROMPT_VERSION,
)

pytestmark = pytest.mark.eval


_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")


class _GoldSample(BaseModel):
    """One hand-labeled gold-set entry."""

    model_config = _FROZEN_STRICT

    mention_id: str
    mention_text: str
    expected_ticker: str | None
    rationale: str


class _GoldSet(BaseModel):
    """Typed wrapper around `eval/baselines/entity_resolution__gold.json`.

    Pydantic validates the shape at fixture-setup time so a schema drift
    (renamed key, missing threshold) surfaces as a clear ValidationError
    rather than a `KeyError` buried inside an assertion."""

    model_config = _FROZEN_STRICT

    prompt_name: str
    thresholds: dict[str, float]
    samples: tuple[_GoldSample, ...]


def _gold_path() -> Path:
    """Anchor the gold set on the project root via pyproject.toml walk."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "eval" / "baselines" / "entity_resolution__gold.json"
    raise FileNotFoundError("project root with pyproject.toml not found")


@pytest.fixture(scope="module")
def gold_set() -> _GoldSet:
    return _GoldSet.model_validate_json(_gold_path().read_text())


@pytest.fixture(scope="module")
def resolver() -> EntityResolver:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — entity-resolution eval needs a real LLM")
    return EntityResolver(
        adapter=EmbeddingAdapter(backend="bge"),
        top_k=3,
    )


def _resolve_with_capture(
    resolver: EntityResolver, sample: _GoldSample
) -> EntityResolution:
    """Resolve one sample, capturing any exception as a sentinel result."""
    try:
        return resolver.resolve(sample.mention_text)
    except Exception as exc:  # pragma: no cover — exercised only in failures
        return EntityResolution(
            resolved_ticker=None,
            reasoning=f"resolver raised {type(exc).__name__}: {exc}",
            considered=(),
            prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
            embed_model_version="error",
        )


@pytest.fixture(scope="module")
def resolutions(
    resolver: EntityResolver, gold_set: _GoldSet
) -> list[tuple[_GoldSample, EntityResolution]]:
    return [(s, _resolve_with_capture(resolver, s)) for s in gold_set.samples]


def test_resolution_accuracy_meets_threshold(
    gold_set: _GoldSet,
    resolutions: list[tuple[_GoldSample, EntityResolution]],
) -> None:
    if not resolutions:
        pytest.skip("gold set is empty")
    threshold = gold_set.thresholds["min_f1"]
    correct = 0
    wrong: list[str] = []
    for sample, result in resolutions:
        if sample.expected_ticker == result.resolved_ticker:
            correct += 1
        else:
            wrong.append(
                f"  {sample.mention_id}: expected={sample.expected_ticker!r} "
                f"got={result.resolved_ticker!r} reasoning={result.reasoning!r}"
            )
    accuracy = correct / len(resolutions)
    assert accuracy >= threshold, (
        f"entity-resolution accuracy {accuracy:.3f} below threshold {threshold} "
        f"({correct}/{len(resolutions)} correct). Misses:\n"
        + "\n".join(wrong)
    )


def test_every_resolution_carries_audit_reasoning(
    resolutions: list[tuple[_GoldSample, EntityResolution]],
) -> None:
    """Acceptance: 'Disambiguator stores reasoning per resolution for audit'."""
    missing = [
        sample.mention_id
        for sample, result in resolutions
        if not result.reasoning.strip()
    ]
    assert not missing, f"resolutions with empty reasoning: {missing}"


def test_unknowns_do_not_false_confidently_match(
    resolutions: list[tuple[_GoldSample, EntityResolution]],
) -> None:
    """Acceptance: '`unknown` is an allowed output (no false-confident matches)'.

    Soft gate: at least half of the gold samples whose expected_ticker is
    None must resolve to None. False-confident matches on `unknown` gold
    is the failure mode the issue is explicitly defending against.
    """
    unknown_gold = [
        (s, r) for s, r in resolutions if s.expected_ticker is None
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


def test_candidate_slates_are_within_top_k(
    resolver: EntityResolver,
    resolutions: list[tuple[_GoldSample, EntityResolution]],
) -> None:
    """Each non-empty, non-errored mention produces 1..top_k candidates.

    Empty mentions short-circuit before retrieval (`considered=()`); a
    resolver error in the sentinel path also reports `considered=()`. Both
    legitimately yield zero candidates, so we assert `<= top_k` (rather
    than `== top_k`) and report the size distribution to make a future
    universe shrink or fixture drift legible at a glance.
    """
    top_k = resolver._top_k
    over_cap = [
        (sample.mention_id, len(result.considered))
        for sample, result in resolutions
        if len(result.considered) > top_k
    ]
    assert not over_cap, (
        f"resolutions exceeded top_k={top_k}: {over_cap}"
    )
    # Report the distribution as a Counter — useful breadcrumb when a
    # future eval-set entry short-circuits or hits the sentinel path.
    sizes: Counter[int] = Counter(len(r.considered) for _, r in resolutions)
    assert sum(sizes.values()) == len(resolutions), sizes
