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
