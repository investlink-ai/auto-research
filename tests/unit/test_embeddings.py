from datetime import date
from pathlib import Path

import pytest

from auto_research.extract.chunking import ChildChunk, ChunkMetadata
from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.extract.embeddings import EmbeddingAdapter, FallbackDecision


def test_unknown_voyage_model_fails_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-totally-fake")
    with pytest.raises(ValueError, match="voyage-totally-fake"):
        EmbeddingAdapter(rag_root=tmp_path)


def test_default_voyage_model_is_voyage_finance_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    monkeypatch.delenv("VOYAGE_MODEL", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    assert adapter.decision.backend == "voyage"
    assert adapter.decision.model == "voyage-finance-2"
    assert adapter.decision.reason == "voyage_used"


def test_fallback_no_key_logs_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    caplog.set_level("INFO", logger="auto_research.extract.embeddings")
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    assert adapter.decision == FallbackDecision("bge", "bge-small-en-v1.5", "no_key")
    assert any("reason=no_key" in r.message for r in caplog.records)


def test_force_local_overrides_voyage_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    caplog.set_level("INFO", logger="auto_research.extract.embeddings")
    adapter = EmbeddingAdapter(rag_root=tmp_path, force_local=True)
    assert adapter.decision.reason == "explicit_override"
    assert any("reason=explicit_override" in r.message for r in caplog.records)


def _make_child(
    text: str,
    *,
    ticker: str = "NVDA",
    doc_type: str = "10-K",
    doc_id: str = "doc-1",
    filing_date: date = date(2025, 3, 15),
) -> ChildChunk:
    md = ChunkMetadata(
        ticker=ticker,
        filing_date=filing_date,
        fiscal_period="FY2025",
        doc_type=doc_type,
        doc_id=doc_id,
    )
    return ChildChunk(
        text=text,
        char_span=(0, len(text)),
        token_count=len(text.split()),
        parent_id=f"{doc_id}:0:{len(text)}",
        section_name="Item 7",
        from_table=False,
        metadata=md,
    )


def _wrap(child: ChildChunk, context: str = "") -> ContextualChildChunk:
    return ContextualChildChunk(child=child, context=context)


def test_embed_bge_writes_both_stores_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    chunks = [
        _wrap(_make_child(f"NVDA China export controls passage {i}", doc_id="doc-A"))
        for i in range(3)
    ]
    adapter.embed(chunks)

    per_doc = tmp_path / "doc-A.lance"
    corpus = tmp_path / "_corpus_narrative.lance"
    assert per_doc.exists(), "per-doc store missing after embed()"
    assert corpus.exists(), "per-corpus narrative store missing after embed()"

    hits_doc = adapter.query("export controls", k=3, store="per_doc", doc_id="doc-A")
    hits_corpus = adapter.query("export controls", k=3, store="corpus_narrative")
    assert len(hits_doc) == 3
    assert len(hits_corpus) == 3


def test_embed_query_is_deterministic_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    chunks = [
        _wrap(_make_child("supply chain disruption in Taiwan", doc_id="doc-D")),
        _wrap(_make_child("export controls on advanced GPUs", doc_id="doc-D")),
        _wrap(_make_child("share buyback authorization", doc_id="doc-D")),
        _wrap(_make_child("revenue grew 14% year over year", doc_id="doc-D")),
        _wrap(_make_child("data center demand strong", doc_id="doc-D")),
    ]
    adapter.embed(chunks)
    a = adapter.query("China chip export", k=3, store="per_doc", doc_id="doc-D")
    b = adapter.query("China chip export", k=3, store="per_doc", doc_id="doc-D")
    assert [h.parent_id for h in a] == [h.parent_id for h in b]
    assert [h.text for h in a] == [h.text for h in b]


def _shrink_voyage_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the module's retry waits to near-zero for fast tests.
    Tenacity reads these at Retrying() construction inside _encode, so
    monkeypatching before the embed call takes effect.
    """
    monkeypatch.setattr(
        "auto_research.extract.embeddings._VOYAGE_RETRY_WAIT_INITIAL", 0.001
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings._VOYAGE_RETRY_WAIT_MAX", 0.01
    )


def test_voyage_rate_limit_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voyage 429s trigger tenacity exponential-jitter retries; the embed
    completes if the endpoint recovers within the retry budget. Decision
    stays on voyage — no silent BGE swap.
    """
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    _shrink_voyage_retry(monkeypatch)

    from voyageai.error import RateLimitError

    class _QuotaError(RateLimitError):
        def __init__(self) -> None:
            super().__init__("simulated 429")  # type: ignore[no-untyped-call]

    class _FlakyVoyage:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str], model: str, input_type: str) -> object:
            self.calls += 1
            if self.calls < 3:
                raise _QuotaError()
            return type(
                "Resp", (), {"embeddings": [[0.0] * 1024 for _ in texts]}
            )()

    adapter = EmbeddingAdapter(rag_root=tmp_path)
    fake = _FlakyVoyage()
    adapter.__dict__["_voyage_client"] = fake

    adapter.embed([_wrap(_make_child("retry me", doc_id="doc-R"))])

    assert fake.calls == 3
    assert adapter.decision.backend == "voyage"


def test_voyage_rate_limit_error_propagates_after_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent 429s exhaust the retry budget; RateLimitError propagates
    to the caller. The adapter does NOT silently switch to BGE — mixing
    Voyage and BGE vectors in one corpus produces an incoherent space.
    """
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    _shrink_voyage_retry(monkeypatch)
    monkeypatch.setattr(
        "auto_research.extract.embeddings._VOYAGE_RETRY_ATTEMPTS", 2
    )

    from voyageai.error import RateLimitError

    class _QuotaError(RateLimitError):
        def __init__(self) -> None:
            super().__init__("simulated 429")  # type: ignore[no-untyped-call]

    class _AlwaysQuota:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str], model: str, input_type: str) -> object:
            self.calls += 1
            raise _QuotaError()

    adapter = EmbeddingAdapter(rag_root=tmp_path)
    assert adapter.decision.reason == "voyage_used"
    fake = _AlwaysQuota()
    adapter.__dict__["_voyage_client"] = fake

    with pytest.raises(RateLimitError):
        adapter.embed([_wrap(_make_child("data center revenue", doc_id="doc-Q"))])

    # All attempts consumed; decision unchanged.
    assert fake.calls == 2
    assert adapter.decision == FallbackDecision(
        "voyage", "voyage-finance-2", "voyage_used"
    )


def test_query_filter_ticker_and_filing_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    adapter.embed([
        _wrap(_make_child(
            "AMD's MI300 ramps in data center",
            ticker="AMD",
            doc_id="doc-AMD",
            filing_date=date(2024, 6, 1),
        )),
    ])
    adapter.embed([
        _wrap(_make_child(
            "NVDA H100 supply tight through Q2",
            ticker="NVDA",
            doc_id="doc-NVDA-2024",
            filing_date=date(2024, 12, 1),
        )),
    ])
    adapter.embed([
        _wrap(_make_child(
            "NVDA Blackwell architecture launches",
            ticker="NVDA",
            doc_id="doc-NVDA-2025",
            filing_date=date(2025, 3, 15),
        )),
    ])
    hits = adapter.query(
        "GPU demand",
        k=5,
        store="corpus_narrative",
        where="ticker = 'NVDA' AND filing_date >= '2025-01-01'",
    )
    assert {h.doc_id for h in hits} == {"doc-NVDA-2025"}
    assert all(h.ticker == "NVDA" for h in hits)
    assert all(h.filing_date >= date(2025, 1, 1) for h in hits)


def test_bm25_query_ranks_lexical_match_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FTS index built at embed-time backs `bm25_query`; the
    lexical-strongest doc out-ranks distractors. Verifies the BM25 half
    of the hybrid contract — the dense half is exercised by the existing
    `test_embed_query_is_deterministic_top_k`.
    """
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    chunks = [
        _wrap(_make_child("export controls limit GPU shipments", doc_id="doc-FTS")),
        _wrap(_make_child("quarterly dividend declaration", doc_id="doc-FTS")),
        _wrap(_make_child("free cash flow disclosure", doc_id="doc-FTS")),
    ]
    adapter.embed(chunks)
    hits = adapter.bm25_query("export controls", k=3, store="per_doc", doc_id="doc-FTS")
    assert hits, "bm25_query must return at least one hit"
    assert hits[0].text.startswith("export controls"), (
        f"top hit should be the lexical match; got {hits[0].text!r}"
    )
    # FTS scores are positive; the top score is strictly highest.
    assert hits[0].score > 0
    if len(hits) > 1:
        assert hits[0].score >= hits[1].score


def test_bm25_query_filter_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR D7: the same `where` filter that scopes dense retrieval also
    scopes BM25, so callers don't have to filter twice.
    """
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    adapter.embed([
        _wrap(_make_child(
            "NVDA export controls passage",
            ticker="NVDA",
            doc_id="doc-NVDA-FTS",
            filing_date=date(2025, 3, 15),
        )),
    ])
    adapter.embed([
        _wrap(_make_child(
            "AMD export controls passage",
            ticker="AMD",
            doc_id="doc-AMD-FTS",
            filing_date=date(2025, 3, 15),
        )),
    ])
    hits = adapter.bm25_query(
        "export controls",
        k=5,
        store="corpus_narrative",
        where="ticker = 'NVDA'",
    )
    assert {h.ticker for h in hits} == {"NVDA"}


def test_query_uses_query_input_type_for_voyage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voyage's asymmetric encoder expects input_type='query' on the query
    side and 'document' on the corpus side. Mixing them weakens ranking.
    """
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")

    class _CapturingVoyage:
        def __init__(self) -> None:
            self.input_types: list[str] = []

        def embed(self, texts: list[str], model: str, input_type: str) -> object:
            self.input_types.append(input_type)
            return type(
                "Resp", (), {"embeddings": [[0.0] * 1024 for _ in texts]}
            )()

    adapter = EmbeddingAdapter(rag_root=tmp_path)
    fake = _CapturingVoyage()
    adapter.__dict__["_voyage_client"] = fake

    adapter.embed([_wrap(_make_child("corpus passage", doc_id="doc-IT"))])
    adapter.query("user search text", k=1, store="per_doc", doc_id="doc-IT")

    assert fake.input_types == ["document", "query"]
