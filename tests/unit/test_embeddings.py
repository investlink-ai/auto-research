from pathlib import Path

import pytest

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
