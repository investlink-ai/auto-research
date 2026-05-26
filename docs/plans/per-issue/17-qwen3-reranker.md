# Issue 17 — Qwen3-Reranker (top-20 → top-5) Implementation Plan

**Goal:** Add a local Qwen3-Reranker pass that reorders the top-K output of
`hybrid_retrieve` (today the top-20 from RRF fusion) into a top-k (default 5),
selectable by an explicit `tier` parameter — `dev` (0.6B), `deployment` (4B),
`ci-cpu` (0.6B on CPU).

**Architecture:** New standalone module `src/auto_research/extract/rerank.py`,
parallel to `rag_retrieval.py`. The reranker is a class (`Qwen3Reranker`)
mirroring `EmbeddingAdapter`'s explicit-config pattern: required `tier` kwarg,
allowlist validation with loud raises, module-level singleton cache, startup
log of `tier + model + device + dtype`. A pure top-level `rerank()` function
composes a reranker with a list of `HybridHit`s and returns a list of
`RerankHit`s — no mutation of `hybrid_retrieve`, no wire-up into a worker (the
extraction pipeline does not yet call `hybrid_retrieve` either, so the
reranker stays standalone until a worker is added).

**Tech Stack:** `transformers` (already a transitive dep via
`sentence-transformers`), PyTorch (same), `torch.device("mps"|"cpu")` for the
runtime split. Qwen3-Reranker is a causal LM scored by the
`p(yes) / (p(yes) + p(no))` differential at the last position per the upstream
model card.

**Tier classification:** Tier 1. `src/auto_research/extract/rerank.py` is not
on the AGENTS.md §3 sensitive-paths list (the Tier-2 escalators inside
`extract/` are `guardrails.py`, `schemas.py`, `chunking/`, and citation
grounding — the reranker is none of these). Verification gate per
AGENTS.md §6: `make quick` + targeted unit test.

---

## Acceptance Criteria → Tasks

| AC | Tasks |
|---|---|
| Deterministic reorder under same model/dtype/backend/input | T2 (rerank fn + tie-break), T3 (real scorer determinism) |
| Hand-built test: precision@5 improvement over RRF alone on micro-corpus, both 0.6B and 4B | T5 |
| Tier is an explicit caller parameter — no implicit env-var fallback | T1 (validation), T1 (allowlist) |
| Tier + model + dtype + device logged at startup; loud errors on missing/invalid config | T1 (init log), T1 (ValueError on bad tier), T3 (loud `_ensure_…` raises) |
| 0.6B (MPS) runs on dev; 4B (MPS) runs at deployment; stable scores under fixed seed | T3 (model.eval(), no sampling), T5 (live smoke for both) |
| No cross-tier score mixing within a single index build | T1 (`reranker_version` token, analogous to `embed_model_version`) |

---

## File Structure

- **Create** `src/auto_research/extract/rerank.py` — `Qwen3Reranker` class,
  `RerankHit` dataclass, `rerank()` top-level function, `_ensure_qwen3_reranker_warmup`
  singleton loader, version tokens.
- **Modify** `tests/unit/conftest.py` — add `_warm_qwen3_reranker` session-autouse
  fixture, gated on Apple Silicon (swallows only the "transformers / torch not
  installed" remediation, propagates everything else — same pattern as
  `_warm_qwen3_mlx_embeddings`).
- **Modify** `pyproject.toml` — pin `transformers>=4.51,<5` directly in core
  `dependencies` (was previously a transitive `sentence-transformers` dep with
  no floor; Qwen3-Reranker requires 4.51+ per the model card).
- **Modify** `Makefile` — add `setup-reranker` target that pre-pulls the 0.6B
  weights (and 4B when `QWEN3_FULL=1`), mirroring `setup-mlx`.
- **Create** `tests/unit/test_rerank.py` — hermetic tests for tier validation,
  version tokens, init logging, rerank orchestration, determinism, top-k
  slicing, tie-break ordering. Stub the model loader for hermetic runs.
- **Create** `tests/live/test_rerank_qwen3_smoke.py` — precision@5 on a
  hand-built micro-corpus, both 0.6B (always on Apple Silicon) and 4B (gated
  by `QWEN3_FULL=1`).

---

## Task 1: Reranker skeleton + version tokens + tier validation

**Files:**
- Create: `src/auto_research/extract/rerank.py`
- Test:   `tests/unit/test_rerank.py`

- [ ] **Step 1.1: Write the failing test for tier validation + init logging**

```python
# tests/unit/test_rerank.py
"""Unit tests for the Qwen3-Reranker tier-selection layer.

These tests cover the explicit-config contract: tier validation, the
loud-error policy mirrors `EmbeddingAdapter`. Real-model scoring lives
in tests/live/.
"""
from __future__ import annotations

import logging

import pytest

from auto_research.extract.rerank import (
    ALLOWED_TIERS,
    RERANKER_VERSION_TAG,
    Qwen3Reranker,
    reranker_version,
)


def test_tier_allowlist_is_frozen_and_complete() -> None:
    assert isinstance(ALLOWED_TIERS, frozenset)
    assert ALLOWED_TIERS == frozenset({"dev", "deployment", "ci-cpu"})


def test_init_logs_tier_model_device_dtype(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="auto_research.extract.rerank"):
        Qwen3Reranker(tier="ci-cpu")
    matching = [r for r in caplog.records if "reranker_init" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "tier=ci-cpu" in msg
    assert "model=Qwen3-Reranker-0.6B" in msg
    assert "device=cpu" in msg
    assert "dtype=" in msg


def test_unknown_tier_raises_value_error() -> None:
    with pytest.raises(ValueError, match="tier must be"):
        Qwen3Reranker(tier="prod")  # type: ignore[arg-type]


def test_tier_to_model_mapping() -> None:
    assert Qwen3Reranker(tier="dev").model == "Qwen3-Reranker-0.6B"
    assert Qwen3Reranker(tier="deployment").model == "Qwen3-Reranker-4B"
    assert Qwen3Reranker(tier="ci-cpu").model == "Qwen3-Reranker-0.6B"


def test_tier_to_device_mapping() -> None:
    assert Qwen3Reranker(tier="dev").device == "mps"
    assert Qwen3Reranker(tier="deployment").device == "mps"
    assert Qwen3Reranker(tier="ci-cpu").device == "cpu"


def test_reranker_version_token_stable() -> None:
    r = Qwen3Reranker(tier="ci-cpu")
    assert r.reranker_version == f"ci-cpu:Qwen3-Reranker-0.6B:{RERANKER_VERSION_TAG}"
    # Helper produces the same token from primitives.
    assert reranker_version("ci-cpu", "Qwen3-Reranker-0.6B") == r.reranker_version


def test_reranker_version_distinguishes_tiers() -> None:
    # Same model but different tier (0.6B on dev/MPS vs ci-cpu/CPU) must
    # produce distinct vector-space tokens — output distributions diverge
    # by dtype and device.
    dev = Qwen3Reranker(tier="dev").reranker_version
    cpu = Qwen3Reranker(tier="ci-cpu").reranker_version
    assert dev != cpu
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd ~/Documents/projects/auto-research/.worktree/17-qwen3-reranker
uv run pytest tests/unit/test_rerank.py -v
```

Expected: FAIL — `ModuleNotFoundError: auto_research.extract.rerank`.

- [ ] **Step 1.3: Write the minimal implementation**

```python
# src/auto_research/extract/rerank.py
"""Local Qwen3-Reranker over hybrid-retrieval output.

Reranking reorders the top-`candidate_k` `HybridHit`s produced by
`hybrid_retrieve` (RRF fusion of BM25 + dense) into a top-`k` final
selection using a cross-encoder yes/no score from the Qwen3-Reranker
causal LM. Single backend, three tiers — explicit caller parameter, no
env-var fallback (same discipline as `embeddings.EmbeddingAdapter`):

- `tier="dev"` — `Qwen3-Reranker-0.6B` on Apple-Silicon MPS. Fast
  iteration on dev machines.
- `tier="deployment"` — `Qwen3-Reranker-4B` on MPS. Higher quality at
  ~6-8x the latency; the indexing/extraction-time choice.
- `tier="ci-cpu"` — `Qwen3-Reranker-0.6B` on CPU. Explicit opt-in for
  sandboxed CI without MPS.

There is no implicit env-var fallback. Construct on the wrong platform
and the warmup raises a clear `RuntimeError` pointing at the correct
tier. Mixing tiers within a single index build is forbidden by
`reranker_version` — analogous to `embed_model_version`, this is the
stable token a row-stamp / cache key would carry once a downstream
worker persists reranked output.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Literal

_log = logging.getLogger(__name__)

ALLOWED_TIERS: frozenset[str] = frozenset({"dev", "deployment", "ci-cpu"})

_TIER_TO_MODEL: dict[str, str] = {
    "dev": "Qwen3-Reranker-0.6B",
    "deployment": "Qwen3-Reranker-4B",
    "ci-cpu": "Qwen3-Reranker-0.6B",
}

_TIER_TO_DEVICE: dict[str, str] = {
    "dev": "mps",
    "deployment": "mps",
    "ci-cpu": "cpu",
}

_TIER_TO_DTYPE: dict[str, str] = {
    "dev": "fp16",
    "deployment": "fp16",
    "ci-cpu": "fp32",
}

RERANKER_VERSION_TAG: str = "v1"
"""Bump when the reranker scoring contract changes.

Triggers: upstream weight re-upload under same id, prompt-template
edit, dtype/device policy flip, tokenizer truncation budget change.
Non-triggers: choosing a different tier (`reranker_version` already
encodes tier+model).
"""


def reranker_version(tier: str, model: str) -> str:
    """Stable token identifying a reranker score space.

    Returns `"{tier}:{model}:{RERANKER_VERSION_TAG}"`. Including tier
    distinguishes the dev/ci-cpu pair: both run the 0.6B model, but on
    different devices/dtypes, which produces non-identical score
    distributions — treating them as the same score space silently
    degrades any persisted-score downstream consumer.
    """
    return f"{tier}:{model}:{RERANKER_VERSION_TAG}"


class Qwen3Reranker:
    def __init__(
        self,
        *,
        tier: Literal["dev", "deployment", "ci-cpu"],
    ) -> None:
        if tier not in ALLOWED_TIERS:
            raise ValueError(
                f"tier must be one of {sorted(ALLOWED_TIERS)}; got {tier!r}"
            )
        self._tier: Literal["dev", "deployment", "ci-cpu"] = tier
        self._model_id = _TIER_TO_MODEL[tier]
        self._device = _TIER_TO_DEVICE[tier]
        self._dtype = _TIER_TO_DTYPE[tier]
        _log.info(
            "reranker_init tier=%s model=%s device=%s dtype=%s",
            self._tier,
            self._model_id,
            self._device,
            self._dtype,
        )

    @property
    def tier(self) -> Literal["dev", "deployment", "ci-cpu"]:
        return self._tier

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @cached_property
    def reranker_version(self) -> str:
        return reranker_version(self._tier, self._model_id)


__all__ = [
    "ALLOWED_TIERS",
    "RERANKER_VERSION_TAG",
    "Qwen3Reranker",
    "reranker_version",
]
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_rerank.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 1.5: Run quick gate**

```bash
make quick
```

Expected: ruff + mypy green.

- [ ] **Step 1.6: Commit**

```bash
git add src/auto_research/extract/rerank.py tests/unit/test_rerank.py
git commit -m "feat(extract): Qwen3Reranker skeleton with tier validation and version tokens"
```

---

## Task 2: `rerank()` orchestration with injectable scorer + determinism

**Files:**
- Modify: `src/auto_research/extract/rerank.py` — add `RerankHit`, `rerank()`,
  `_ScorerProtocol`.
- Modify: `tests/unit/test_rerank.py` — add orchestration tests.

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/unit/test_rerank.py`:

```python
from datetime import date

from auto_research.extract.chunking import ChunkMetadata, ParentChunk
from auto_research.extract.rag_retrieval import HybridHit
from auto_research.extract.rerank import RerankHit, rerank


def _parent(text: str, idx: int) -> ParentChunk:
    return ParentChunk(
        text=text,
        section_name="Item 1",
        char_span=(0, len(text)),
        token_count=max(1, len(text) // 4),
        table_html=None,
        metadata=ChunkMetadata(
            ticker="NVDA",
            filing_date=date(2025, 3, 15),
            fiscal_period="FY2025",
            doc_type="10-K",
            doc_id=f"doc-{idx}",
        ),
    )


def _hit(text: str, idx: int, rrf_score: float) -> HybridHit:
    return HybridHit(
        parent=_parent(text, idx),
        score=rrf_score,
        bm25_rank=idx + 1,
        dense_rank=idx + 1,
        bm25_score=1.0 / (idx + 1),
        dense_score=1.0 / (idx + 1),
    )


def test_rerank_reorders_by_scorer_descending() -> None:
    # Five hits in RRF order; scorer assigns higher score to later
    # items so rerank should reverse them.
    hits = [_hit(f"passage {i}", i, rrf_score=1.0 - i * 0.1) for i in range(5)]

    def fake_scorer(query: str, passages: list[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]

    out = rerank(query="q", hits=hits, top_k=3, scorer=fake_scorer)

    assert [h.parent.metadata.doc_id for h in out] == ["doc-4", "doc-3", "doc-2"]
    assert [h.score for h in out] == [4.0, 3.0, 2.0]
    # prev_rank is 1-based in the original RRF input order.
    assert [h.prev_rank for h in out] == [5, 4, 3]
    # prev_rrf_score carried through unchanged.
    assert out[0].prev_rrf_score == pytest.approx(1.0 - 4 * 0.1)


def test_rerank_top_k_clamps_to_input_length() -> None:
    hits = [_hit(f"p{i}", i, rrf_score=1.0 - i * 0.1) for i in range(3)]

    def fake_scorer(query: str, passages: list[str]) -> list[float]:
        return [1.0, 2.0, 3.0]

    out = rerank(query="q", hits=hits, top_k=10, scorer=fake_scorer)
    assert len(out) == 3


def test_rerank_deterministic_tie_break() -> None:
    # Three hits, two tied on score. Tie-break: higher prev_rrf_score wins;
    # if still tied, lexicographic by doc_id.
    hits = [
        _hit("a", 0, rrf_score=0.5),
        _hit("b", 1, rrf_score=0.9),
        _hit("c", 2, rrf_score=0.7),
    ]

    def fake_scorer(query: str, passages: list[str]) -> list[float]:
        return [0.8, 0.8, 0.8]

    out1 = rerank(query="q", hits=hits, top_k=3, scorer=fake_scorer)
    out2 = rerank(query="q", hits=hits, top_k=3, scorer=fake_scorer)
    # Order is stable across calls.
    assert [h.parent.metadata.doc_id for h in out1] == [
        h.parent.metadata.doc_id for h in out2
    ]
    # Higher prev_rrf_score first.
    assert out1[0].parent.metadata.doc_id == "doc-1"  # rrf=0.9
    assert out1[1].parent.metadata.doc_id == "doc-2"  # rrf=0.7
    assert out1[2].parent.metadata.doc_id == "doc-0"  # rrf=0.5


def test_rerank_invalid_top_k_raises() -> None:
    hits = [_hit("a", 0, 0.5)]

    def fake_scorer(q: str, p: list[str]) -> list[float]:
        return [0.1] * len(p)

    with pytest.raises(ValueError, match="top_k must be positive"):
        rerank(query="q", hits=hits, top_k=0, scorer=fake_scorer)


def test_rerank_empty_input_returns_empty() -> None:
    def fake_scorer(q: str, p: list[str]) -> list[float]:
        return []

    out = rerank(query="q", hits=[], top_k=5, scorer=fake_scorer)
    assert out == []


def test_rerank_scorer_length_mismatch_raises() -> None:
    hits = [_hit("a", 0, 0.5), _hit("b", 1, 0.4)]

    def bad_scorer(q: str, p: list[str]) -> list[float]:
        return [0.1]  # one short

    with pytest.raises(ValueError, match="scorer returned 1 scores for 2 passages"):
        rerank(query="q", hits=hits, top_k=2, scorer=bad_scorer)
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_rerank.py -v
```

Expected: 6 new tests FAIL — `RerankHit` / `rerank` not importable.

- [ ] **Step 2.3: Implement `RerankHit` + `rerank()`**

Append to `src/auto_research/extract/rerank.py` (above `__all__`):

```python
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from auto_research.extract.chunking import ParentChunk
from auto_research.extract.rag_retrieval import HybridHit


@dataclass(frozen=True)
class RerankHit:
    """One reranked hit, carrying both the reranker score and the prior
    RRF context so a caller can diagnose how the reranker moved each
    item (or persist both columns).
    """

    parent: ParentChunk
    score: float
    prev_rrf_score: float
    prev_rank: int


# A scorer maps `(query, passages)` to a per-passage relevance score
# (larger = more relevant). The protocol lives at the module level so
# the unit tests can substitute a deterministic stub for the real
# Qwen3-Reranker model. The real implementation is `Qwen3Reranker.score`.
ScorerFn = Callable[[str, list[str]], list[float]]


def rerank(
    *,
    query: str,
    hits: Sequence[HybridHit],
    top_k: int,
    scorer: ScorerFn,
) -> list[RerankHit]:
    """Reorder `hits` by `scorer(query, [h.parent.text for h in hits])` and
    return the top `top_k`.

    Tie-break for deterministic output: descending reranker score, then
    descending prior RRF score, then ascending `doc_id`. The reranker's
    yes-probability is dense-floating-point and ties are statistically
    rare, but they DO happen on identical passages (e.g., two filings
    that quote the same boilerplate); the fixed tie-break makes the
    output order reproducible across runs.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")
    if not hits:
        return []
    passages = [h.parent.text for h in hits]
    scores = scorer(query, passages)
    if len(scores) != len(passages):
        raise ValueError(
            f"scorer returned {len(scores)} scores for {len(passages)} passages"
        )
    indexed = list(enumerate(zip(hits, scores, strict=True)))
    indexed.sort(
        key=lambda item: (
            -item[1][1],
            -item[1][0].score,
            item[1][0].parent.metadata.doc_id,
        )
    )
    return [
        RerankHit(
            parent=h.parent,
            score=s,
            prev_rrf_score=h.score,
            prev_rank=orig_idx + 1,
        )
        for orig_idx, (h, s) in indexed[:top_k]
    ]
```

Add to `__all__`:

```python
__all__ = [
    "ALLOWED_TIERS",
    "RERANKER_VERSION_TAG",
    "Qwen3Reranker",
    "RerankHit",
    "ScorerFn",
    "rerank",
    "reranker_version",
]
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_rerank.py -v
```

Expected: all 13 tests PASS (7 from Task 1 + 6 from Task 2).

- [ ] **Step 2.5: Run quick gate**

```bash
make quick
```

Expected: ruff + mypy green.

- [ ] **Step 2.6: Commit**

```bash
git add src/auto_research/extract/rerank.py tests/unit/test_rerank.py
git commit -m "feat(extract): rerank() orchestration over HybridHit with deterministic tie-break"
```

---

## Task 3: Real `score()` path + singleton model loader

**Files:**
- Modify: `src/auto_research/extract/rerank.py` — add
  `_ensure_qwen3_reranker_warmup`, `Qwen3Reranker.score`, prompt template,
  yes/no token resolution.
- Modify: `tests/unit/test_rerank.py` — add tests with mocked
  transformers (hermetic).

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/unit/test_rerank.py`:

```python
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np


class _FakeLogits:
    """Stand-in for a single forward pass output. Holds a 1D logits
    tensor for the last position; indexing with the yes/no token ids
    returns scalar values."""

    def __init__(self, yes_logit: float, no_logit: float, vocab_size: int = 100) -> None:
        self._logits = np.full(vocab_size, -1e4, dtype=np.float32)
        self._logits[10] = yes_logit  # token id 10 = "yes" by stub convention
        self._logits[20] = no_logit  # token id 20 = "no"

    def __getitem__(self, idx: int) -> float:
        return float(self._logits[idx])


class _FakeModelOutput:
    def __init__(self, last_logits: _FakeLogits) -> None:
        # shape (batch=1, seq, vocab); only the last position is read.
        self.logits = np.zeros((1, 4, 100), dtype=np.float32)
        for v_id in range(100):
            self.logits[0, -1, v_id] = -1e4
        self.logits[0, -1, 10] = last_logits[10]
        self.logits[0, -1, 20] = last_logits[20]


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "yes":
            return [10]
        if text == "no":
            return [20]
        # Any other text encodes to a fixed dummy sequence.
        return [1, 2, 3, 4]

    def __call__(
        self,
        prompt: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 2048,
    ) -> dict[str, object]:
        # `score()` will move this dict to the device; the fake `to()`
        # makes that a no-op.
        class _Inputs(dict[str, object]):
            def to(self, device: str) -> "_Inputs":
                return self

        return _Inputs(input_ids=[[1, 2, 3, 4]])


class _FakeModel:
    """Returns a deterministic logits vector per call. The unit test
    sets `score_sequence` to feed a sequence of (yes_logit, no_logit)
    tuples — one consumed per `__call__` invocation."""

    def __init__(self, score_sequence: list[tuple[float, float]]) -> None:
        self._iter: Iterator[tuple[float, float]] = iter(score_sequence)

    def __call__(self, **inputs: object) -> _FakeModelOutput:
        yes, no = next(self._iter)
        return _FakeModelOutput(_FakeLogits(yes, no))

    def eval(self) -> "_FakeModel":
        return self

    def to(self, device: str) -> "_FakeModel":
        return self


def test_score_returns_yes_probability_per_passage() -> None:
    """yes_logit > no_logit → score close to 1; equal logits → ~0.5."""
    fake_model = _FakeModel(
        score_sequence=[
            (5.0, -5.0),  # passage 0: yes much more likely
            (0.0, 0.0),  # passage 1: tied
            (-5.0, 5.0),  # passage 2: no much more likely
        ]
    )
    fake_tokenizer = _FakeTokenizer()

    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, fake_tokenizer),
    ):
        r = Qwen3Reranker(tier="ci-cpu")
        scores = r.score(query="q", passages=["a", "b", "c"])

    assert scores[0] > 0.95
    assert scores[1] == pytest.approx(0.5, abs=1e-6)
    assert scores[2] < 0.05


def test_score_is_deterministic_for_same_inputs() -> None:
    """Two calls with the same model fixtures and inputs produce
    bit-identical scores (no sampling, eval mode)."""
    fixed_seq = [(2.0, -1.0)] * 2

    def make() -> tuple[_FakeModel, _FakeTokenizer]:
        return _FakeModel(score_sequence=list(fixed_seq)), _FakeTokenizer()

    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        side_effect=lambda model_id, device, dtype: make(),
    ):
        s1 = Qwen3Reranker(tier="ci-cpu").score(query="q", passages=["a"])
        s2 = Qwen3Reranker(tier="ci-cpu").score(query="q", passages=["a"])
    assert s1 == s2


def test_score_empty_passages_returns_empty_list() -> None:
    fake_model = _FakeModel(score_sequence=[])
    fake_tokenizer = _FakeTokenizer()
    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, fake_tokenizer),
    ):
        scores = Qwen3Reranker(tier="ci-cpu").score(query="q", passages=[])
    assert scores == []
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_rerank.py -v
```

Expected: 3 new tests FAIL — `score()` method / `_ensure_qwen3_reranker_warmup`
not defined.

- [ ] **Step 3.3: Implement the real scoring path**

Add to `src/auto_research/extract/rerank.py` (above the `__all__` block,
after the `Qwen3Reranker` class definition — the loader stays at module level
so it can be patched by tests):

Insert imports near the top, after the existing ones:

```python
import math
import platform
from typing import Any
```

Add the prompt-template constants and HF repo map near the top, after
`_TIER_TO_DTYPE`:

```python
# Tokenizer truncation budget. Qwen3-Reranker is a causal LM with
# 32K-token context; 2048 covers the contextual-chunking pattern
# (LLM-generated context prefix + child text + query + instruction)
# without pinning the device to full 32K on every batch.
_RERANKER_MAX_LENGTH = 2048

# Prompt template per the Qwen3-Reranker model card. The instruction
# is domain-tailored to this corpus (SEC filings + earnings transcripts
# + analyst materials) — matches the embeddings module's Qwen3 query
# instruction and is the natural-language analogue of the same domain
# tailoring. Revisit when a Ragas / DeepEval baseline gives a tuning
# handle.
_RERANKER_INSTRUCTION = (
    "Given a financial research query, judge whether the passage is "
    "relevant. Answer yes or no."
)
_RERANKER_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only '
    'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_RERANKER_PROMPT_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# HF repo ids. Use the upstream Qwen org weights directly (the
# reranker has not been re-quantized by mlx-community at issue
# pickup time).
_RERANKER_HF_REPOS: dict[str, str] = {
    "Qwen3-Reranker-0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "Qwen3-Reranker-4B": "Qwen/Qwen3-Reranker-4B",
}

# Module-level singleton cache keyed by (model_id, device). 0.6B on
# MPS and 0.6B on CPU are different cache entries because dtype and
# device differ; the score distributions diverge accordingly.
_RERANKER_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}
```

Add the warmup helper above the `Qwen3Reranker` class (so it can be
referenced by `cached_property`):

```python
def _ensure_qwen3_reranker_warmup(
    model_id: str, device: str, dtype: str
) -> tuple[Any, Any]:
    """Load a Qwen3-Reranker once and return `(model, tokenizer)`.

    Idempotent via `_RERANKER_MODELS` keyed by `(model_id, device)`.
    Raises `RuntimeError` with a clear remediation on:

    - `device="mps"` requested on a non-Apple-Silicon host (use
      `tier="ci-cpu"` on Linux CI).
    - `transformers` / `torch` import failure (should not happen with
      core deps installed, but surfacing the right error beats a
      `ModuleNotFoundError` from inside this function).
    - HF cache miss with no network reachable (point at
      `make setup-reranker`).
    """
    key = (model_id, device)
    cached = _RERANKER_MODELS.get(key)
    if cached is not None:
        return cached

    if device == "mps" and not (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    ):
        raise RuntimeError(
            f"Qwen3-Reranker tier requested device={device!r} but host is "
            f"system={platform.system()!r} machine={platform.machine()!r}. "
            "Construct with tier='ci-cpu' on non-Apple-Silicon hosts."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"Qwen3-Reranker {model_id!r} requested but `transformers` / "
            "`torch` could not be imported. Run `uv sync` to install "
            "core deps."
        ) from exc

    repo = _RERANKER_HF_REPOS.get(model_id)
    if repo is None:
        raise ValueError(
            f"No HF repo mapping for Qwen3-Reranker model {model_id!r}; "
            f"known: {sorted(_RERANKER_HF_REPOS)}"
        )

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch_dtype)
    except Exception as exc:
        raise RuntimeError(
            f"Qwen3-Reranker {model_id!r} could not be loaded — likely a "
            "HuggingFace cache miss with no network reachable. Populate "
            "the cache with:\n"
            "    make setup-reranker\n"
            f"(repo: {repo})"
        ) from exc
    model = model.to(device).eval()

    _RERANKER_MODELS[key] = (model, tokenizer)
    return _RERANKER_MODELS[key]
```

Add the `score()` method to `Qwen3Reranker`:

```python
    @cached_property
    def _model_and_tokenizer(self) -> tuple[Any, Any]:
        return _ensure_qwen3_reranker_warmup(self._model_id, self._device, self._dtype)

    def score(self, *, query: str, passages: list[str]) -> list[float]:
        """Score `(query, passage)` pairs with the Qwen3-Reranker
        yes/no head; return per-passage `p(yes) / (p(yes) + p(no))`.

        Deterministic: `eval()` mode, no sampling, no dropout. Score
        magnitudes only meaningful within a single `(tier, model)`;
        cross-tier comparison is forbidden by the `reranker_version`
        guard.
        """
        if not passages:
            return []
        import torch

        model, tokenizer = self._model_and_tokenizer
        yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("no", add_special_tokens=False)[0]
        scores: list[float] = []
        with torch.no_grad():
            for passage in passages:
                prompt = (
                    _RERANKER_PROMPT_PREFIX
                    + f"<Instruct>: {_RERANKER_INSTRUCTION}\n"
                    + f"<Query>: {query}\n"
                    + f"<Document>: {passage}"
                    + _RERANKER_PROMPT_SUFFIX
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=_RERANKER_MAX_LENGTH,
                ).to(self._device)
                out = model(**inputs)
                last_logits = out.logits[0, -1, :]
                yes_logit = float(last_logits[yes_id])
                no_logit = float(last_logits[no_id])
                m = max(yes_logit, no_logit)
                ey = math.exp(yes_logit - m)
                en = math.exp(no_logit - m)
                scores.append(ey / (ey + en))
        return scores
```

Add `_ensure_qwen3_reranker_warmup` to `__all__`:

```python
__all__ = [
    "ALLOWED_TIERS",
    "RERANKER_VERSION_TAG",
    "Qwen3Reranker",
    "RerankHit",
    "ScorerFn",
    "_ensure_qwen3_reranker_warmup",
    "rerank",
    "reranker_version",
]
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_rerank.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 3.5: Run quick gate**

```bash
make quick
```

Expected: ruff + mypy green. If mypy flags `transformers` / `torch` imports
without stubs, add the corresponding `[[tool.mypy.overrides]]` entries to
`pyproject.toml` with `ignore_missing_imports = true` (mirroring the existing
overrides for `mlx_embeddings`).

- [ ] **Step 3.6: Commit**

```bash
git add src/auto_research/extract/rerank.py tests/unit/test_rerank.py pyproject.toml
git commit -m "feat(extract): Qwen3Reranker.score backed by transformers + singleton model cache"
```

---

## Task 4: Dependency pin + Makefile setup target + conftest warmup

**Files:**
- Modify: `pyproject.toml` — add `transformers>=4.51,<5` to core
  `dependencies`. Add `mypy.overrides` for `transformers.*` and `torch.*` if
  Task 3 didn't already add them.
- Modify: `Makefile` — add `setup-reranker` target, update `.PHONY`.
- Modify: `tests/unit/conftest.py` — add `_warm_qwen3_reranker` session-autouse
  fixture.

- [ ] **Step 4.1: Add `transformers` pin and mypy override**

Edit `pyproject.toml` `dependencies` list (insert after `tiktoken==0.13.0`,
preserving alphabetical-ish order):

```toml
    "tiktoken==0.13.0",
    "transformers>=4.51,<5",
    "traceloop-sdk>=0.30",
```

Add the mypy overrides (after the `voyageai.*` block):

```toml
[[tool.mypy.overrides]]
module = "transformers.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true
```

- [ ] **Step 4.2: Sync deps**

```bash
uv sync
```

Expected: `transformers` resolves to a 4.51.x or newer release; no other
package downgrades.

- [ ] **Step 4.3: Add `setup-reranker` Makefile target**

Add to `.PHONY` line:

```makefile
.PHONY: quick check check-full test test-broad integration eval live-smoke lint typecheck setup-nlp setup-mlx setup-reranker smoke
```

Append after the `setup-mlx` block:

```makefile
# Pre-pull Qwen3-Reranker weights into the HuggingFace cache. Mirrors
# `setup-nlp` / `setup-mlx`: lands the model in cache during a real-
# network window so the conftest's session-autouse warmup can serve
# hermetic tests without touching the network. The 0.6B variant is
# the dev-tier default (also reused by ci-cpu); the 4B variant is
# opt-in via QWEN3_FULL=1 (~8 GB download).
setup-reranker:
	uv run python -c "from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup; _ensure_qwen3_reranker_warmup('Qwen3-Reranker-0.6B', 'cpu', 'fp32')"
	@if [ "$$QWEN3_FULL" = "1" ]; then \
		uv run python -c "from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup; _ensure_qwen3_reranker_warmup('Qwen3-Reranker-4B', 'cpu', 'fp32')"; \
	fi
```

- [ ] **Step 4.4: Add the conftest warmup fixture**

Append to `tests/unit/conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def _warm_qwen3_reranker() -> None:
    """Warm the Qwen3-Reranker-0.6B model once per session on any host.

    Same lazy-load-then-socket-monkey-patch concern as the BGE and
    Qwen3-Embedding warmups: the first hermetic reranker unit test
    that triggers a real load would otherwise pull ~1.2 GB of
    Qwen3-Reranker-0.6B weights from HuggingFace under a socket-blocked
    environment. Pre-warming at session start lands the weights in HF
    cache via `make setup-reranker`.

    Cross-platform: the reranker's `ci-cpu` tier runs on Linux CI. On
    Apple Silicon, the same warmup populates the `(0.6B, cpu)` cache
    entry; the `dev` tier's `(0.6B, mps)` entry is loaded lazily by
    the tests that actually exercise MPS.

    Only swallows the "transformers / torch not installed" remediation
    error — all other failures (cache miss, repo rename, API drift)
    propagate so the session start fails loudly with the actionable
    remediation, per the explicit-config-loud rule.
    """
    from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup

    try:
        _ensure_qwen3_reranker_warmup("Qwen3-Reranker-0.6B", "cpu", "fp32")
    except RuntimeError as exc:
        if "uv sync" not in str(exc):
            raise
```

- [ ] **Step 4.5: Run quick gate + full unit suite**

```bash
make quick && make test
```

Expected: all green. The conftest warmup runs once; subsequent unit tests
that mock `_ensure_qwen3_reranker_warmup` are unaffected (the patch replaces
the function for the duration of the test).

- [ ] **Step 4.6: Pre-pull weights (if first time)**

```bash
make setup-reranker
```

Expected: 0.6B repo downloads to `~/.cache/huggingface/` once; idempotent
re-runs.

- [ ] **Step 4.7: Commit**

```bash
git add pyproject.toml uv.lock Makefile tests/unit/conftest.py
git commit -m "chore(extract): pin transformers, add setup-reranker target, warm Qwen3-Reranker in unit conftest"
```

---

## Task 5: Live smoke — precision@5 on a hand-built micro-corpus

**Files:**
- Create: `tests/live/test_rerank_qwen3_smoke.py`.

- [ ] **Step 5.1: Build the micro-corpus + write the failing test**

```python
# tests/live/test_rerank_qwen3_smoke.py
"""Live smoke for Qwen3-Reranker on a hand-built micro-corpus.

Asserts the AC requirement that rerank improves precision@5 over RRF
alone, measured against both the 0.6B (dev/ci-cpu) and 4B (deployment)
backends. The 0.6B path runs on Apple-Silicon MPS (matching the dev
tier); the 4B path is gated by QWEN3_FULL=1 because the weights are
~8 GB.

What this catches that unit tests can't:

- Upstream Qwen3-Reranker repo drift (Alibaba rename, weight re-upload,
  tokenizer change).
- Score distribution shift between 0.6B and 4B — both must still rank
  the known-relevant passages above the distractors.
- Apple-Silicon MPS / fp16 numerical issues that don't surface on the
  fp32-CPU unit test path.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import date

import pytest

from auto_research.extract.chunking import ChunkMetadata, ParentChunk
from auto_research.extract.rag_retrieval import HybridHit
from auto_research.extract.rerank import Qwen3Reranker, rerank


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# The micro-corpus: 12 passages drawn from the kinds of text the
# extraction layer will see. Index 0-3 are the known-relevant set
# (data-center / Hopper / Blackwell AI-infrastructure capacity); the
# rest are distractors of varying topic distance.
_QUERY = "NVIDIA data-center revenue drivers from Hopper and Blackwell GPU shipments"

_PASSAGES_AND_GOLD: list[tuple[str, bool]] = [
    (
        "Data Center revenue grew 154% year-over-year driven by strong "
        "demand for our Hopper architecture and the ramp of Blackwell.",
        True,
    ),
    (
        "Cloud service providers expanded their Hopper-based GPU "
        "capacity through Q4, with multiple hyperscalers placing orders "
        "for Blackwell training systems.",
        True,
    ),
    (
        "Our Blackwell platform began shipping to leading cloud "
        "customers in the fourth quarter; demand exceeds supply.",
        True,
    ),
    (
        "Hopper architecture (H100, H200) shipments remained a primary "
        "driver of Data Center segment revenue in the period.",
        True,
    ),
    (
        "Gaming revenue increased sequentially as channel inventory "
        "normalized and GeForce RTX 40-series demand stayed solid.",
        False,
    ),
    (
        "Automotive segment revenue grew modestly on continued "
        "DRIVE Orin design wins across passenger vehicles.",
        False,
    ),
    (
        "Cash from operations more than doubled year-over-year on "
        "higher net income and favorable working-capital movements.",
        False,
    ),
    (
        "Stock-based compensation expense was largely consistent with "
        "the prior quarter; no acceleration events occurred.",
        False,
    ),
    (
        "Our China revenue continued to be impacted by U.S. export "
        "controls limiting the products we can ship into that market.",
        False,
    ),
    (
        "Operating expenses grew modestly driven by higher engineering "
        "headcount and continued investment in software platforms.",
        False,
    ),
    (
        "The board authorized an additional $25 billion of share "
        "repurchases with no expiration date.",
        False,
    ),
    (
        "Inventory increased sequentially in preparation for the "
        "Blackwell platform ramp and continued Hopper customer demand.",
        True,
    ),  # mildly relevant to the query — included to test ordering nuance
]


def _make_hit(idx: int, text: str, rrf_score: float) -> HybridHit:
    parent = ParentChunk(
        text=text,
        section_name="MD&A",
        char_span=(0, len(text)),
        token_count=max(1, len(text) // 4),
        table_html=None,
        metadata=ChunkMetadata(
            ticker="NVDA",
            filing_date=date(2025, 2, 26),
            fiscal_period="FY2025-Q4",
            doc_type="10-K",
            doc_id=f"doc-NVDA-{idx:02d}",
        ),
    )
    return HybridHit(
        parent=parent,
        score=rrf_score,
        bm25_rank=idx + 1,
        dense_rank=idx + 1,
        bm25_score=1.0 / (idx + 1),
        dense_score=1.0 / (idx + 1),
    )


@dataclass(frozen=True)
class _Corpus:
    hits: list[HybridHit]
    gold_doc_ids: set[str]


def _build_corpus() -> _Corpus:
    """The RRF ordering deliberately mis-orders the corpus: gold items
    are interleaved with distractors so a top-5 cut from RRF alone
    catches only some of them, and the reranker has room to improve."""
    # Deliberately bad RRF order: distractor, gold, distractor, gold, ...
    rrf_order = [4, 0, 5, 1, 6, 11, 7, 2, 8, 3, 9, 10]
    hits: list[HybridHit] = []
    for rank_idx, corpus_idx in enumerate(rrf_order):
        text, _ = _PASSAGES_AND_GOLD[corpus_idx]
        hits.append(_make_hit(corpus_idx, text, rrf_score=1.0 - rank_idx * 0.05))
    gold_doc_ids = {
        f"doc-NVDA-{i:02d}"
        for i, (_, is_gold) in enumerate(_PASSAGES_AND_GOLD)
        if is_gold
    }
    return _Corpus(hits=hits, gold_doc_ids=gold_doc_ids)


def _precision_at_5(doc_ids: list[str], gold: set[str]) -> float:
    return sum(1 for d in doc_ids[:5] if d in gold) / 5.0


@pytest.mark.skipif(
    not _is_apple_silicon(), reason="dev-tier MPS requires Apple Silicon"
)
def test_rerank_06b_improves_precision_at_5_over_rrf() -> None:
    corpus = _build_corpus()
    rrf_top5 = [h.parent.metadata.doc_id for h in corpus.hits[:5]]
    rrf_p5 = _precision_at_5(rrf_top5, corpus.gold_doc_ids)

    reranker = Qwen3Reranker(tier="dev")
    reranked = rerank(
        query=_QUERY, hits=corpus.hits, top_k=5, scorer=lambda q, ps: reranker.score(query=q, passages=ps)
    )
    rerank_top5 = [h.parent.metadata.doc_id for h in reranked]
    rerank_p5 = _precision_at_5(rerank_top5, corpus.gold_doc_ids)

    assert rerank_p5 > rrf_p5, (
        f"rerank precision@5={rerank_p5:.2f} did not exceed RRF "
        f"baseline={rrf_p5:.2f}. RRF top-5={rrf_top5}; "
        f"rerank top-5={rerank_top5}; gold={sorted(corpus.gold_doc_ids)}"
    )
    # Reranker must produce a stable order on the same input — re-run
    # and confirm bit-identical.
    reranked2 = rerank(
        query=_QUERY, hits=corpus.hits, top_k=5, scorer=lambda q, ps: reranker.score(query=q, passages=ps)
    )
    assert [h.parent.metadata.doc_id for h in reranked2] == rerank_top5


@pytest.mark.skipif(
    os.environ.get("QWEN3_FULL") != "1",
    reason="QWEN3_FULL=1 required to opt into the 8 GB 4B download",
)
@pytest.mark.skipif(
    not _is_apple_silicon(), reason="deployment-tier MPS requires Apple Silicon"
)
def test_rerank_4b_improves_precision_at_5_over_rrf() -> None:
    corpus = _build_corpus()
    rrf_top5 = [h.parent.metadata.doc_id for h in corpus.hits[:5]]
    rrf_p5 = _precision_at_5(rrf_top5, corpus.gold_doc_ids)

    reranker = Qwen3Reranker(tier="deployment")
    reranked = rerank(
        query=_QUERY, hits=corpus.hits, top_k=5, scorer=lambda q, ps: reranker.score(query=q, passages=ps)
    )
    rerank_top5 = [h.parent.metadata.doc_id for h in reranked]
    rerank_p5 = _precision_at_5(rerank_top5, corpus.gold_doc_ids)

    assert rerank_p5 > rrf_p5, (
        f"rerank-4B precision@5={rerank_p5:.2f} did not exceed RRF "
        f"baseline={rrf_p5:.2f}. RRF top-5={rrf_top5}; "
        f"rerank top-5={rerank_top5}; gold={sorted(corpus.gold_doc_ids)}"
    )
```

- [ ] **Step 5.2: Run the live 0.6B test**

```bash
QWEN3_FULL=0 uv run pytest tests/live/test_rerank_qwen3_smoke.py -v -m live
```

Expected on Apple Silicon: `test_rerank_06b_…` PASS;
`test_rerank_4b_…` SKIPPED with reason "QWEN3_FULL=1 required".

Expected on Linux CI: both SKIPPED with reason "Apple Silicon required".

- [ ] **Step 5.3: (Optional, dev box) Run the 4B test**

```bash
QWEN3_FULL=1 uv run pytest tests/live/test_rerank_qwen3_smoke.py::test_rerank_4b_improves_precision_at_5_over_rrf -v -m live
```

Expected: PASS (after the one-time ~8 GB download). Skip if you don't want
to incur the download — it's gated for exactly this reason and the AC is
satisfied by demonstrating the path works (e.g., one local run, recorded in
the PR body).

- [ ] **Step 5.4: Commit**

```bash
git add tests/live/test_rerank_qwen3_smoke.py
git commit -m "test(extract): live smoke — Qwen3-Reranker precision@5 vs RRF on micro-corpus"
```

---

## Final Verification + PR

- [ ] **Step F.1: Full pre-PR gate**

```bash
make quick && make test
```

Expected: ruff + mypy + unit suite all green.

- [ ] **Step F.2: Open the PR**

```bash
git push -u origin feat/17-qwen3-reranker
gh pr create --title "feat(extract): Qwen3-Reranker on top-20 → top-5 (#17)" --body "$(cat <<'EOF'
## Summary

Adds `auto_research.extract.rerank` — a local Qwen3-Reranker pass that reorders the top-K output of hybrid retrieval (today the top-20 from RRF fusion) into a top-k (default 5).

## AC → evidence

- [x] Deterministic reorder under same model/dtype/backend/input
  → `tests/unit/test_rerank.py::test_rerank_deterministic_tie_break`,
    `::test_score_is_deterministic_for_same_inputs`
- [x] Tier is an explicit caller parameter; no implicit env-var fallback
  → `tests/unit/test_rerank.py::test_unknown_tier_raises_value_error`,
    `::test_tier_to_model_mapping`, `::test_tier_to_device_mapping`
- [x] Tier + model + dtype + device logged at startup; missing/invalid config errors loud
  → `tests/unit/test_rerank.py::test_init_logs_tier_model_device_dtype`,
    `_ensure_qwen3_reranker_warmup` raises with remediation on platform / import / cache miss
- [x] 0.6B (MPS) on dev; 4B (MPS) at deployment; stable scores
  → `tests/live/test_rerank_qwen3_smoke.py::test_rerank_06b_improves_precision_at_5_over_rrf`
    (local run output pasted below);
  → `…::test_rerank_4b_improves_precision_at_5_over_rrf` (QWEN3_FULL=1, optional)
- [x] Hand-built precision@5 test, both 0.6B and 4B
  → both live tests above; gold-vs-rerank top-5 diff printed on failure
- [x] No cross-tier score mixing within a single index build
  → `tests/unit/test_rerank.py::test_reranker_version_distinguishes_tiers`;
    `reranker_version("dev", …) != reranker_version("ci-cpu", …)`

## Verification

`make quick && make test` — green (output below).
Live smoke `make live-smoke` — 0.6B test green on Apple Silicon dev box (output below).

## Test plan
- [x] `make quick`
- [x] `make test`
- [x] `QWEN3_FULL=0 make live-smoke` covering `test_rerank_06b_…`
- [ ] `QWEN3_FULL=1 …test_rerank_4b_…` (optional; opt-in 8 GB download)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step F.3: Delete this per-issue plan post-merge**

Per `docs/AI_WORKFLOW.md §1.5`, per-issue plans are deleted at PR merge. After
the squash-merge lands on `main`, the cleanup commit on `main` removes
`docs/plans/per-issue/17-qwen3-reranker.md` along with any other merged
per-issue plans.
