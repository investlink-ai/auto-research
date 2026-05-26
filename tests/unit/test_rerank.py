"""Unit tests for the Qwen3-Reranker tier-selection layer.

These tests cover the explicit-config contract: tier validation, the
loud-error policy mirrors `EmbeddingAdapter`. Real-model scoring lives
in tests/live/.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any
from unittest.mock import patch

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
from tests._otel_helpers import SpanRecorder


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


class _FakeTokenizer:
    """Stand-in for a HuggingFace tokenizer.

    Supports the surface `Qwen3Reranker.score` actually touches:
    `convert_tokens_to_ids`, `encode` (single-string, for prefix/suffix
    pre-tokenization + single-token assertion), and call-form on a list
    of bodies returning `{"input_ids": list[list[int]]}`. Exposes
    `pad_token_id` / `eos_token_id` so the padding path inside
    `score()` can look them up.
    """

    pad_token_id = 0
    eos_token_id = 99

    def convert_tokens_to_ids(self, text: str) -> int:
        if text == "yes":
            return _YES_TOKEN_ID
        if text == "no":
            return _NO_TOKEN_ID
        return 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "yes":
            return [_YES_TOKEN_ID]
        if text == "no":
            return [_NO_TOKEN_ID]
        return [1, 2, 3, 4]

    def __call__(
        self,
        text_input: list[str] | str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = True,
        max_length: int = 2048,
        return_attention_mask: bool = True,
    ) -> dict[str, list[list[int]]]:
        if isinstance(text_input, str):
            text_input = [text_input]
        # One body row per input; content is irrelevant — the fake model
        # reads the configured logit table by row index, not by ids.
        return {"input_ids": [[5, 6, 7, 8] for _ in text_input]}


class _FakeModel:
    """Returns logits with caller-specified (yes, no) values at the last
    position of each row. One row per batch input."""

    def __init__(self, score_sequence: list[tuple[float, float]]) -> None:
        self._score_sequence = list(score_sequence)

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any = None,
        **_: Any,
    ) -> Any:
        import torch

        batch = input_ids.shape[0]
        seq = input_ids.shape[1]
        vocab = 100
        logits = torch.full((batch, seq, vocab), -1e4, dtype=torch.float32)
        for i in range(batch):
            yes_logit, no_logit = self._score_sequence[i]
            logits[i, -1, _YES_TOKEN_ID] = yes_logit
            logits[i, -1, _NO_TOKEN_ID] = no_logit

        class _Out:
            def __init__(self, logits: Any) -> None:
                self.logits = logits

        return _Out(logits)

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
    fixed_seq = [(2.0, -1.0)]

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


def test_rerank_emits_extract_rerank_span(span_recorder: SpanRecorder) -> None:
    """`rerank()` wraps its orchestration in an `extract.rerank` span
    carrying input/output counts and the score-space token. Mirrors
    `extract.hybrid_retrieve` from `rag_retrieval.py`."""
    hits = [_hit(f"p{i}", i, rrf_score=1.0 - i * 0.1) for i in range(3)]

    def fake_scorer(q: str, ps: list[str]) -> list[float]:
        return [0.1, 0.2, 0.3]

    rerank(
        query="q",
        hits=hits,
        top_k=2,
        scorer=fake_scorer,
        reranker_version="ci-cpu:Qwen3-Reranker-0.6B:v1",
    )
    attrs = span_recorder.attrs("extract.rerank")
    assert attrs["extract.worker"] == "reranker"
    assert attrs["rerank.candidate_count"] == 3
    assert attrs["rerank.top_k"] == 2
    assert attrs["rerank.hits_count"] == 2
    assert attrs["rerank.reranker_version"] == "ci-cpu:Qwen3-Reranker-0.6B:v1"
    assert attrs["extract.outcome"] == "success"


def test_rerank_span_marks_error_outcome_on_scorer_failure(
    span_recorder: SpanRecorder,
) -> None:
    """A scorer raising propagates, and the span records outcome=error
    with a status description so traces bucket failures correctly."""
    hits = [_hit("p", 0, 0.5)]

    def boom(q: str, ps: list[str]) -> list[float]:
        raise RuntimeError("scorer broke")

    with pytest.raises(RuntimeError, match="scorer broke"):
        rerank(query="q", hits=hits, top_k=1, scorer=boom)
    attrs = span_recorder.attrs("extract.rerank")
    assert attrs["extract.outcome"] == "error"


def test_rerank_span_omits_reranker_version_when_not_provided(
    span_recorder: SpanRecorder,
) -> None:
    """Default `reranker_version=None` should not emit the attribute —
    keeps the span attribute set tight when callers don't stamp the
    score-space token."""
    hits = [_hit("p", 0, 0.5)]

    def fake_scorer(q: str, ps: list[str]) -> list[float]:
        return [0.5]

    rerank(query="q", hits=hits, top_k=1, scorer=fake_scorer)
    attrs = span_recorder.attrs("extract.rerank")
    assert "rerank.reranker_version" not in attrs


def test_score_emits_extract_reranker_score_span(
    span_recorder: SpanRecorder,
) -> None:
    """`Qwen3Reranker.score` emits a nested `extract.reranker.score` span
    carrying tier / model / device / dtype / batch_size so model
    inference latency can be observed independently of the orchestration
    overhead."""
    fake_model = _FakeModel(score_sequence=[(1.0, 0.0), (0.5, 0.5)])
    fake_tokenizer = _FakeTokenizer()
    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, fake_tokenizer),
    ):
        Qwen3Reranker(tier="ci-cpu").score(query="q", passages=["a", "b"])
    attrs = span_recorder.attrs("extract.reranker.score")
    assert attrs["extract.worker"] == "reranker"
    assert attrs["reranker.tier"] == "ci-cpu"
    assert attrs["reranker.model"] == "Qwen3-Reranker-0.6B"
    assert attrs["reranker.device"] == "cpu"
    assert attrs["reranker.dtype"] == "fp32"
    assert attrs["reranker.batch_size"] == 2
    assert attrs["extract.outcome"] == "success"


def test_score_runs_one_forward_per_batch_not_per_passage() -> None:
    """All N passages must be scored in a single model forward pass —
    the batching optimization is the headline perf win. If `score()`
    loops, this fake will record N>1 calls."""
    call_count = 0

    class _CountingFakeModel(_FakeModel):
        def __call__(
            self,
            *,
            input_ids: Any,
            attention_mask: Any = None,
            **kwargs: Any,
        ) -> Any:
            nonlocal call_count
            call_count += 1
            return super().__call__(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    fake_model = _CountingFakeModel(
        score_sequence=[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    )
    fake_tokenizer = _FakeTokenizer()
    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, fake_tokenizer),
    ):
        scores = Qwen3Reranker(tier="ci-cpu").score(
            query="q", passages=["a", "b", "c", "d"]
        )
    assert len(scores) == 4
    assert call_count == 1, f"expected single batched forward; got {call_count}"


# ---------------------------------------------------------------------
# Code-review fixes — exercised contracts post-review.
# ---------------------------------------------------------------------

def test_score_callable_positionally_without_lambda_wrapper() -> None:
    """`scorer=reranker.score` must bind directly; rerank() invokes the
    scorer positionally as `scorer(query, passages)`."""
    fake_model = _FakeModel(score_sequence=[(2.0, -1.0)])
    fake_tokenizer = _FakeTokenizer()
    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, fake_tokenizer),
    ):
        reranker = Qwen3Reranker(tier="ci-cpu")
        out = rerank(
            query="q",
            hits=[_hit("p", 0, 0.5)],
            top_k=1,
            scorer=reranker.score,
        )
    assert len(out) == 1


def test_rerank_stamps_reranker_version_on_each_hit() -> None:
    hits = [_hit(f"p{i}", i, 0.5 - i * 0.1) for i in range(3)]

    def fake_scorer(q: str, ps: list[str]) -> list[float]:
        return [1.0, 2.0, 3.0]

    out = rerank(
        query="q",
        hits=hits,
        top_k=3,
        scorer=fake_scorer,
        reranker_version="ci-cpu:Qwen3-Reranker-0.6B:v1",
    )
    assert all(h.reranker_version == "ci-cpu:Qwen3-Reranker-0.6B:v1" for h in out)


def test_rerank_default_reranker_version_is_none() -> None:
    hits = [_hit("p", 0, 0.5)]

    def fake_scorer(q: str, ps: list[str]) -> list[float]:
        return [1.0]

    out = rerank(query="q", hits=hits, top_k=1, scorer=fake_scorer)
    assert out[0].reranker_version is None


def test_unknown_device_raises_loud_error() -> None:
    """`_ensure_qwen3_reranker_warmup` rejects devices outside the
    {mps, cpu} allowlist before attempting any model load."""
    from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup

    with pytest.raises(ValueError, match="device must be"):
        _ensure_qwen3_reranker_warmup("Qwen3-Reranker-0.6B", "cuda", "fp32")


def test_warmup_cache_key_includes_dtype() -> None:
    """Conftest populates the cache with a 3-tuple key including dtype.
    The bare `(model_id, device)` key would collide if two tiers shared
    the same `(model, device)` but used different dtypes."""
    from auto_research.extract.rerank import _RERANKER_MODELS

    assert ("Qwen3-Reranker-0.6B", "cpu", "fp32") in _RERANKER_MODELS


def test_assert_single_token_raises_on_multi_token_yes() -> None:
    """The yes/no scoring takes `tokenizer.encode(...)[0]` — if the
    tokenizer splits 'yes' or 'no' across tokens the [0] slice silently
    grabs the wrong id. The warmup helper must assert single-token
    encoding loudly."""
    from auto_research.extract.rerank import _assert_yes_no_single_token

    class _MultiTokTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            if text == "yes":
                return [1, 2]  # split across two tokens — wrong
            if text == "no":
                return [3]
            return [4]

    with pytest.raises(RuntimeError, match="expected single tokens"):
        _assert_yes_no_single_token(_MultiTokTokenizer())


def test_truncation_is_handled_inside_batched_tokenizer_call() -> None:
    """Truncation moves into the batched `tokenizer(...)` call with
    `max_length=body_budget`; no separate `_truncate_passage_to_budget`
    helper is needed. This test pins the contract that the tokenizer is
    invoked with `truncation=True` and a finite `max_length`."""
    captured: dict[str, object] = {}

    class _RecordingTokenizer(_FakeTokenizer):
        def __call__(
            self,
            text_input: list[str] | str,
            *,
            add_special_tokens: bool = False,
            truncation: bool = True,
            max_length: int = 2048,
            return_attention_mask: bool = True,
        ) -> dict[str, list[list[int]]]:
            # Only the batched-bodies call records — the prefix/suffix
            # pre-tokenization uses `encode()`, a different code path.
            if isinstance(text_input, list):
                captured["truncation"] = truncation
                captured["max_length"] = max_length
                captured["batch_size"] = len(text_input)
            return super().__call__(
                text_input,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                return_attention_mask=return_attention_mask,
            )

    fake_model = _FakeModel(score_sequence=[(1.0, 0.0), (0.5, 0.5)])
    with patch(
        "auto_research.extract.rerank._ensure_qwen3_reranker_warmup",
        return_value=(fake_model, _RecordingTokenizer()),
    ):
        Qwen3Reranker(tier="ci-cpu").score(query="q", passages=["a", "b"])
    assert captured["truncation"] is True
    assert isinstance(captured["max_length"], int)
    assert captured["max_length"] > 0
    assert captured["batch_size"] == 2
