"""Unit tests for the Qwen3-Reranker tier-selection layer.

These tests cover the explicit-config contract: tier validation, the
loud-error policy mirrors `EmbeddingAdapter`. Real-model scoring lives
in tests/live/.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from auto_research.extract.chunking import ChunkMetadata, ParentChunk
from auto_research.extract.rag_retrieval import HybridHit
from auto_research.extract.rerank import (
    ALLOWED_TIERS,
    RERANKER_VERSION_TAG,
    Qwen3Reranker,
    RerankHit,
    rerank,
    reranker_version,
)


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


def test_tier_allowlist_is_frozen_and_complete() -> None:
    assert isinstance(ALLOWED_TIERS, frozenset)
    assert frozenset({"dev", "deployment", "ci-cpu"}) == ALLOWED_TIERS


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
    assert reranker_version("ci-cpu", "Qwen3-Reranker-0.6B") == r.reranker_version


def test_reranker_version_distinguishes_tiers() -> None:
    # Same model but different tier (0.6B on dev/MPS vs ci-cpu/CPU) must
    # produce distinct vector-space tokens — output distributions diverge
    # by dtype and device.
    dev = Qwen3Reranker(tier="dev").reranker_version
    cpu = Qwen3Reranker(tier="ci-cpu").reranker_version
    assert dev != cpu


def test_rerank_reorders_by_scorer_descending() -> None:
    # Five hits in RRF order; scorer assigns higher score to later
    # items so rerank should reverse them.
    hits = [_hit(f"passage {i}", i, rrf_score=1.0 - i * 0.1) for i in range(5)]

    def fake_scorer(query: str, passages: list[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]

    out = rerank(query="q", hits=hits, top_k=3, scorer=fake_scorer)

    assert [h.parent.metadata.doc_id for h in out] == ["doc-4", "doc-3", "doc-2"]
    assert [h.score for h in out] == [4.0, 3.0, 2.0]
    assert [h.prev_rank for h in out] == [5, 4, 3]
    assert out[0].prev_rrf_score == pytest.approx(1.0 - 4 * 0.1)
    assert isinstance(out[0], RerankHit)


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


# ---------------------------------------------------------------------
# Real-scoring path tests — hermetic via mocked `_ensure_qwen3_reranker_warmup`.
# ---------------------------------------------------------------------
_YES_TOKEN_ID = 10
_NO_TOKEN_ID = 20


class _FakeModelOutput:
    """Mimics `transformers` CausalLM output: `.logits` is a tensor of
    shape (batch, seq, vocab). Only the last-position slice is read by
    `Qwen3Reranker.score`, so the other positions are zero-filled."""

    def __init__(self, yes_logit: float, no_logit: float, vocab_size: int = 100) -> None:
        last = np.full(vocab_size, -1e4, dtype=np.float32)
        last[_YES_TOKEN_ID] = yes_logit
        last[_NO_TOKEN_ID] = no_logit
        self.logits = np.zeros((1, 4, vocab_size), dtype=np.float32)
        self.logits[0, -1, :] = last


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "yes":
            return [_YES_TOKEN_ID]
        if text == "no":
            return [_NO_TOKEN_ID]
        return [1, 2, 3, 4]

    def __call__(
        self,
        prompt: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 2048,
    ) -> dict[str, Any]:
        class _Inputs(dict[str, Any]):
            def to(self, device: str) -> _Inputs:
                return self

        return _Inputs(input_ids=[[1, 2, 3, 4]])


class _FakeModel:
    """Yields the next pre-canned (yes_logit, no_logit) pair per call."""

    def __init__(self, score_sequence: list[tuple[float, float]]) -> None:
        self._iter: Iterator[tuple[float, float]] = iter(score_sequence)

    def __call__(self, **inputs: object) -> _FakeModelOutput:
        yes, no = next(self._iter)
        return _FakeModelOutput(yes, no)

    def eval(self) -> _FakeModel:
        return self

    def to(self, device: str) -> _FakeModel:
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
