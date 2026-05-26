from datetime import date
from pathlib import Path

import pytest

from auto_research.extract.chunking import ChildChunk, ChunkMetadata
from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.extract.embeddings import (
    ALLOWED_MLX_QWEN3_MODELS,
    BGE_MODEL_ID,
    DEFAULT_MLX_QWEN3_MODEL,
    EMBED_MODEL_VERSION_TAG,
    EmbeddingAdapter,
    embed_model_version,
    resolve_backend_from_env,
)

# ---- Explicit backend selection contract ---------------------------------


def test_voyage_backend_with_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    monkeypatch.delenv("VOYAGE_MODEL", raising=False)
    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    assert adapter.backend == "voyage"
    assert adapter.model == "voyage-finance-2"


def test_voyage_backend_rejects_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-totally-fake")
    with pytest.raises(ValueError, match="voyage-totally-fake"):
        EmbeddingAdapter(backend="voyage", rag_root=tmp_path)


def test_bge_backend_explicit(tmp_path: Path) -> None:
    """`backend="bge"` selects the in-process model regardless of whether
    `VOYAGE_API_KEY` is set — selection is explicit, not env-derived.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    assert adapter.backend == "bge"
    assert adapter.model == BGE_MODEL_ID


def test_bge_backend_rejects_voyage_model_kwarg(tmp_path: Path) -> None:
    """`voyage_model` paired with `backend="bge"` is incoherent — the caller's
    intent is ambiguous, so reject loudly at init.
    """
    with pytest.raises(ValueError, match="voyage_model is only valid"):
        EmbeddingAdapter(
            backend="bge", rag_root=tmp_path, voyage_model="voyage-finance-2"
        )


def test_unknown_backend_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        EmbeddingAdapter(backend="other", rag_root=tmp_path)  # type: ignore[arg-type]


def test_voyage_without_api_key_raises_at_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing with `backend="voyage"` succeeds even without a key,
    but the first embed call must raise a clear `RuntimeError` instead of
    silently falling back to BGE.
    """
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY is not set"):
        adapter.embed([_wrap(_make_child("anything", doc_id="doc-NK"))])


def test_resolve_backend_from_env_voyage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "voyage")
    assert resolve_backend_from_env() == "voyage"


def test_resolve_backend_from_env_bge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge")
    assert resolve_backend_from_env() == "bge"


def test_resolve_backend_from_env_unset_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default — missing `EMBEDDING_BACKEND` must error, not silently
    pick a backend.
    """
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_BACKEND env var must be set"):
        resolve_backend_from_env()


def test_resolve_backend_from_env_invalid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    with pytest.raises(RuntimeError) as excinfo:
        resolve_backend_from_env()
    # The error must (a) echo back the invalid value the user supplied,
    # AND (b) enumerate every canonical valid backend so the typo is
    # self-remediating without a docs trip. Locking both halves
    # prevents a future refactor from accidentally dropping any of the
    # three names from the message.
    msg = str(excinfo.value)
    assert "openai" in msg
    for valid in ("voyage", "bge", "qwen3-mlx"):
        assert valid in msg, f"valid backend {valid!r} missing from error: {msg!r}"


# ---- Shared helpers ------------------------------------------------------


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


# ---- BGE-backed end-to-end -----------------------------------------------


def test_embed_bge_writes_both_stores_atomically(tmp_path: Path) -> None:
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
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


def test_embed_stamps_version_columns_in_rows(tmp_path: Path) -> None:
    """Each LanceDB row carries the three pure-function-contract versions
    that produced it (chunker_version, contextual_prompt_version,
    embed_model_version). Write-only audit metadata at present; backs
    the materialization-versioned-tables follow-up.
    """
    import lancedb

    from auto_research.extract.chunking import CHUNKER_VERSION
    from auto_research.extract.prompts.contextual_chunk import (
        CONTEXTUAL_CHUNK_PROMPT_VERSION,
    )

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("stamped row passage", doc_id="doc-STAMP"))])

    df = lancedb.connect(tmp_path).open_table("doc-STAMP").to_pandas()
    assert len(df) == 1
    assert df["chunker_version"].iloc[0] == CHUNKER_VERSION
    assert df["contextual_prompt_version"].iloc[0] == CONTEXTUAL_CHUNK_PROMPT_VERSION
    assert df["embed_model_version"].iloc[0] == f"bge:{BGE_MODEL_ID}:{EMBED_MODEL_VERSION_TAG}"

    # And the per-corpus narrative store carries the same stamps.
    corpus_df = lancedb.connect(tmp_path).open_table("_corpus_narrative").to_pandas()
    assert corpus_df["chunker_version"].iloc[0] == CHUNKER_VERSION


def test_embed_model_version_helper_composes_backend_model_tag() -> None:
    """`embed_model_version(backend, model)` returns the stable token used
    to invalidate downstream caches transitively. Backend is part of the
    token (different backends serving the same model id can diverge on
    quantization)."""
    assert embed_model_version("bge", "bge-small-en-v1.5") == (
        f"bge:bge-small-en-v1.5:{EMBED_MODEL_VERSION_TAG}"
    )
    assert embed_model_version("voyage", "voyage-finance-2") == (
        f"voyage:voyage-finance-2:{EMBED_MODEL_VERSION_TAG}"
    )


def test_embed_model_version_property_matches_helper(tmp_path: Path) -> None:
    """The adapter property delegates to the module-level helper so callers
    that hold an adapter (workers, embed-once test scaffolding) and callers
    that compose tokens for cache keys see the same value."""
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    assert adapter.embed_model_version == embed_model_version("bge", adapter.model)


def test_embed_query_is_deterministic_top_k(tmp_path: Path) -> None:
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
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


def test_query_filter_ticker_and_filing_date(tmp_path: Path) -> None:
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
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


def test_bm25_query_ranks_lexical_match_first(tmp_path: Path) -> None:
    """The FTS index built at embed-time backs `bm25_query`; the
    lexical-strongest doc out-ranks distractors. Verifies the BM25 half
    of the hybrid contract — the dense half is exercised by the existing
    `test_embed_query_is_deterministic_top_k`.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
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
    assert hits[0].score > 0
    if len(hits) > 1:
        assert hits[0].score >= hits[1].score


def test_bm25_query_filter_composes(tmp_path: Path) -> None:
    """ADR D7: the same `where` filter that scopes dense retrieval also
    scopes BM25, so callers don't have to filter twice.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
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


# ---- Voyage path (mocked client) -----------------------------------------


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

    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    fake = _FlakyVoyage()
    adapter.__dict__["_voyage_client"] = fake

    adapter.embed([_wrap(_make_child("retry me", doc_id="doc-R"))])

    assert fake.calls == 3
    assert adapter.backend == "voyage"


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

    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    fake = _AlwaysQuota()
    adapter.__dict__["_voyage_client"] = fake

    with pytest.raises(RateLimitError):
        adapter.embed([_wrap(_make_child("data center revenue", doc_id="doc-Q"))])

    # All attempts consumed; backend unchanged.
    assert fake.calls == 2
    assert adapter.backend == "voyage"
    assert adapter.model == "voyage-finance-2"


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

    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    fake = _CapturingVoyage()
    adapter.__dict__["_voyage_client"] = fake

    adapter.embed([_wrap(_make_child("corpus passage", doc_id="doc-IT"))])
    adapter.query("user search text", k=1, store="per_doc", doc_id="doc-IT")

    assert fake.input_types == ["document", "query"]


# ---- BGE warmup (Issue #64) -----------------------------------------------


def test_bge_embed_makes_no_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hermetic guarantee — `EmbeddingAdapter.embed()` in BGE mode must
    not touch the network. The session-autouse conftest fixture has
    already warmed BGE via `_ensure_bge_warmup`, so the socket
    monkey-patch can't trigger a lazy HuggingFace download.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    assert adapter.backend == "bge"

    import socket
    from typing import Any

    def _no_socket(*args: Any, **kwargs: Any) -> None:
        raise OSError("network access forbidden during embed")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    monkeypatch.setattr(socket, "getaddrinfo", _no_socket)

    adapter.embed([_wrap(_make_child("hermetic test passage", doc_id="doc-HRM"))])
    hits = adapter.query("hermetic", k=1, store="per_doc", doc_id="doc-HRM")
    assert hits, "BGE embed/query must work end-to-end under socket lockdown"


def test_ensure_bge_warmup_raises_with_remediation_on_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the HuggingFace cache miss can't be filled (no network /
    airgapped runner), `_ensure_bge_warmup` must surface a
    `RuntimeError` naming `make setup-nlp` as the fix — mirroring the
    spaCy warmup pattern in `_nlp_warmup.py`.
    """
    from auto_research.extract import embeddings as emb

    monkeypatch.setattr(emb, "_BGE_MODEL", None)

    class _BrokenST:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("simulated no-network HF cache miss")

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _BrokenST)

    with pytest.raises(RuntimeError, match="make setup-nlp"):
        emb._ensure_bge_warmup()


# ---- Qwen3-MLX selection + validation (no MLX install required) ----------


def test_qwen3_mlx_backend_with_default_model(tmp_path: Path) -> None:
    """`backend="qwen3-mlx"` selects Qwen3-Embedding-0.6B by default —
    the small, dev-scoped variant. No env-var fallback; the choice is
    explicit at the kwarg level."""
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    assert adapter.backend == "qwen3-mlx"
    assert adapter.model == DEFAULT_MLX_QWEN3_MODEL
    assert adapter.model == "Qwen3-Embedding-0.6B"


def test_qwen3_mlx_backend_with_explicit_4b_model(tmp_path: Path) -> None:
    """The 4B variant is selectable by passing `mlx_qwen3_model` —
    deployment-grade, opt-in."""
    adapter = EmbeddingAdapter(
        backend="qwen3-mlx", rag_root=tmp_path, mlx_qwen3_model="Qwen3-Embedding-4B"
    )
    assert adapter.backend == "qwen3-mlx"
    assert adapter.model == "Qwen3-Embedding-4B"


def test_qwen3_mlx_backend_rejects_unknown_model(tmp_path: Path) -> None:
    """Unlisted MLX model raises with the allowlist in the error
    message — mirrors the existing Voyage pattern at the same call
    site."""
    with pytest.raises(ValueError, match="Qwen3-totally-fake"):
        EmbeddingAdapter(
            backend="qwen3-mlx",
            rag_root=tmp_path,
            mlx_qwen3_model="Qwen3-totally-fake",
        )


def test_allowed_mlx_qwen3_models_contains_both_variants() -> None:
    """The allowlist is the single source of truth for which MLX
    Qwen3 weights this adapter accepts — both dev (0.6B) and
    deployment (4B) variants are present."""
    assert "Qwen3-Embedding-0.6B" in ALLOWED_MLX_QWEN3_MODELS
    assert "Qwen3-Embedding-4B" in ALLOWED_MLX_QWEN3_MODELS


def test_voyage_backend_rejects_mlx_qwen3_model_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mlx_qwen3_model` paired with `backend="voyage"` is incoherent —
    the caller's intent is ambiguous, so reject loudly at init.
    Mirrors the existing `voyage_model` + `backend="bge"` check."""
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-test")
    with pytest.raises(ValueError, match="mlx_qwen3_model is only valid"):
        EmbeddingAdapter(
            backend="voyage",
            rag_root=tmp_path,
            mlx_qwen3_model="Qwen3-Embedding-0.6B",
        )


def test_bge_backend_rejects_mlx_qwen3_model_kwarg(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mlx_qwen3_model is only valid"):
        EmbeddingAdapter(
            backend="bge",
            rag_root=tmp_path,
            mlx_qwen3_model="Qwen3-Embedding-0.6B",
        )


def test_qwen3_mlx_backend_rejects_voyage_model_kwarg(tmp_path: Path) -> None:
    """Symmetric to the bge↔voyage_model rejection — `voyage_model`
    paired with the MLX backend is incoherent."""
    with pytest.raises(ValueError, match="voyage_model is only valid"):
        EmbeddingAdapter(
            backend="qwen3-mlx",
            rag_root=tmp_path,
            voyage_model="voyage-finance-2",
        )


def test_qwen3_mlx_on_non_darwin_raises_at_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing with `backend="qwen3-mlx"` succeeds even on Linux,
    but the first embed call must raise a clear `RuntimeError` pointing
    at `backend="voyage"` or `backend="bge"` as the remediation.
    Same lazy-failure pattern as Voyage without `VOYAGE_API_KEY`.
    """
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    # Reset the singleton so the platform check re-runs even on a Mac
    # runner where prior tests may have warmed it.
    from auto_research.extract import embeddings as emb

    monkeypatch.setattr(emb, "_QWEN3_MODELS", {}, raising=False)

    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    with pytest.raises(RuntimeError, match=r"voyage.*bge|bge.*voyage"):
        adapter.embed([_wrap(_make_child("anything", doc_id="doc-QWLNX"))])


def test_qwen3_mlx_platform_check_fires_before_mlx_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the ordering bug where `_encode`'s qwen3-mlx
    branch imported `mlx_embeddings` BEFORE accessing `self._qwen3`
    (which triggers the platform check). On a Linux host without the
    `[mlx]` extra installed, the import would fail with
    `ModuleNotFoundError` instead of the friendly platform-check
    `RuntimeError` naming the cross-platform backends.

    Simulates the missing-module case by sticking `None` into
    `sys.modules['mlx_embeddings']` (which makes `import` raise) and
    confirms the platform check still wins on a non-Apple-Silicon
    host. The test passes on both Linux CI (real ImportError absent
    the extra) and Mac dev boxes (extra installed; sys.modules
    override forces the import-error mode synthetically)."""
    import platform
    import sys

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)

    from auto_research.extract import embeddings as emb

    monkeypatch.setattr(emb, "_QWEN3_MODELS", {}, raising=False)

    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        adapter.embed([_wrap(_make_child("anything", doc_id="doc-QWORD"))])


def test_qwen3_mlx_on_intel_mac_raises_at_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apple Silicon means `arm64`; an Intel Mac (`x86_64`) is still
    Darwin but MLX won't run on it. The platform check rejects both
    non-Darwin AND non-arm64 hosts."""
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    from auto_research.extract import embeddings as emb

    monkeypatch.setattr(emb, "_QWEN3_MODELS", {}, raising=False)

    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        adapter.embed([_wrap(_make_child("anything", doc_id="doc-QWINT"))])


def test_resolve_backend_from_env_qwen3_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "qwen3-mlx")
    assert resolve_backend_from_env() == "qwen3-mlx"


def test_resolve_backend_from_env_invalid_lists_all_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error message must enumerate all three valid values so a
    user setting `EMBEDDING_BACKEND=mlx` (typo) sees the canonical
    spelling. Asserts each name appears individually."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "mlx")
    with pytest.raises(RuntimeError) as excinfo:
        resolve_backend_from_env()
    msg = str(excinfo.value)
    for valid in ("voyage", "bge", "qwen3-mlx"):
        assert valid in msg, f"valid backend {valid!r} missing from error: {msg!r}"


def test_qwen3_mlx_constants_kept_in_sync() -> None:
    """Three parallel sources of truth — `ALLOWED_MLX_QWEN3_MODELS`
    (the constructor allowlist), `_QWEN3_MLX_HF_REPOS` (the load-time
    HF repo map), and `_MODEL_DIM` (the LanceDB schema dim) — must
    agree. Drift between them is a silent failure mode (constructor
    accepts a model that crashes at first use). Lock all three here
    so adding a new variant requires updating all three or this
    test fails loudly."""
    from auto_research.extract.embeddings import _MODEL_DIM, _QWEN3_MLX_HF_REPOS

    assert set(_QWEN3_MLX_HF_REPOS.keys()) == set(ALLOWED_MLX_QWEN3_MODELS), (
        "ALLOWED_MLX_QWEN3_MODELS and _QWEN3_MLX_HF_REPOS keys disagree — "
        "every allowlisted model needs an HF repo mapping"
    )
    for model_id in ALLOWED_MLX_QWEN3_MODELS:
        assert model_id in _MODEL_DIM, (
            f"allowlisted MLX model {model_id!r} missing from _MODEL_DIM — "
            "the LanceDB schema won't know its native vector dim"
        )


# ---- Qwen3-MLX encode + embed (mlx_embeddings.load monkey-patched) -------


class _StubQwen3Generate:
    """Records `generate(model, tokenizer, texts)` calls and emits
    a deterministic L2-normalized output with `.text_embeds`.

    Mirrors the real `mlx_embeddings.generate` surface: takes the
    `(model, tokenizer)` returned by `mlx_embeddings.load(...)` plus
    a list of texts, returns an object whose `.text_embeds` is a
    `(batch, dim)`-shaped array-like.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.generate_calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: object,
        tokenizer: object,
        texts: list[str],
        **kwargs: object,
    ) -> object:
        # `**kwargs` absorbs `max_length` (and any future tunables the
        # adapter passes through) so the stub keeps tracking the
        # production call shape without per-kwarg fixture updates.
        self.generate_calls.append({
            "model": model,
            "tokenizer": tokenizer,
            "texts": list(texts),
            "kwargs": dict(kwargs),
        })
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            vec = [0.0] * self.dim
            vec[i % self.dim] = 1.0
            out.append(vec)

        class _Out:
            text_embeds = out

        return _Out()


def _install_fake_mlx(
    monkeypatch: pytest.MonkeyPatch, dim: int
) -> _StubQwen3Generate:
    """Install a fake `mlx_embeddings` module + force Apple-Silicon
    platform values so the warmup path runs cross-platform under test.
    `load(...)` returns a `(model_sentinel, tokenizer_sentinel)` tuple
    but only after validating the repo string is one the production
    `_QWEN3_MLX_HF_REPOS` actually publishes — that way a typo in the
    constant surfaces here at unit-test time instead of slipping
    through to the Mac live smoke. `generate(...)` records calls and
    emits deterministic output."""
    import platform
    import sys
    import types

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    from auto_research.extract import embeddings as emb

    monkeypatch.setattr(emb, "_QWEN3_MODELS", {}, raising=False)

    known_repos = frozenset(emb._QWEN3_MLX_HF_REPOS.values())

    def _fake_load(repo: str) -> tuple[str, str]:
        assert repo in known_repos, (
            f"production _QWEN3_MLX_HF_REPOS resolved to {repo!r}, "
            f"which is not one of the published mlx-community repos "
            f"{sorted(known_repos)} — typo or out-of-date constant?"
        )
        return ("model_sentinel", "tokenizer_sentinel")

    stub = _StubQwen3Generate(dim=dim)
    fake = types.ModuleType("mlx_embeddings")
    fake.load = _fake_load  # type: ignore[attr-defined]
    fake.generate = stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_embeddings", fake)
    return stub


def test_qwen3_mlx_encode_document_side_omits_query_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asymmetric encoder: the corpus side passes texts verbatim.
    No `Instruct: ...\\nQuery:` prefix when input_type='document'."""
    stub = _install_fake_mlx(monkeypatch, dim=1024)
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("corpus passage one", doc_id="doc-QD"))])

    assert stub.generate_calls, "mlx_embeddings.generate was not called"
    first = stub.generate_calls[0]
    assert first["texts"] == ["corpus passage one"]


def test_qwen3_mlx_encode_query_side_prepends_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asymmetric encoder: the query side prepends the Qwen3-Embedding
    instruction prefix per the model card. Without it, ranking quality
    degrades on the asymmetric corpus."""
    stub = _install_fake_mlx(monkeypatch, dim=1024)
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("indexed text", doc_id="doc-QQ"))])
    adapter.query("user search", k=1, store="per_doc", doc_id="doc-QQ")

    # Last generate call is the query side.
    last = stub.generate_calls[-1]
    texts = last["texts"]
    assert isinstance(texts, list) and len(texts) == 1
    only_text = texts[0]
    assert isinstance(only_text, str)
    assert only_text.startswith("Instruct: "), (
        f"query text must start with the Qwen3 instruction prefix; "
        f"got {only_text!r}"
    )
    assert "\nQuery: user search" in only_text


def test_qwen3_mlx_embed_writes_rows_with_correct_vector_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`embed()` materializes a per-doc LanceDB table whose vector
    column matches the Qwen3-0.6B native dim (1024), plus the standard
    metadata columns. Mirrors the Voyage row-shape contract."""
    _install_fake_mlx(monkeypatch, dim=1024)
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    chunks = [
        _wrap(_make_child(f"qwen3 passage {i}", doc_id="doc-QDIM"))
        for i in range(3)
    ]
    adapter.embed(chunks)

    import lancedb

    db = lancedb.connect(tmp_path)
    tbl = db.open_table("doc-QDIM")
    df = tbl.to_pandas()
    assert len(df) == 3
    # LanceDB stores the fixed-size vector column as an object dtype of
    # numpy arrays. Length per row should equal the native model dim.
    assert all(len(v) == 1024 for v in df["vector"])
    # Metadata columns preserved.
    for col in (
        "text",
        "ticker",
        "filing_date",
        "fiscal_period",
        "doc_type",
        "doc_id",
        "parent_id",
        "section_name",
    ):
        assert col in df.columns


def test_qwen3_mlx_4b_uses_2560_dim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 4B variant materializes 2560-dim vectors — the native dim
    advertised by the upstream model card."""
    _install_fake_mlx(monkeypatch, dim=2560)
    adapter = EmbeddingAdapter(
        backend="qwen3-mlx",
        rag_root=tmp_path,
        mlx_qwen3_model="Qwen3-Embedding-4B",
    )
    adapter.embed([_wrap(_make_child("four-billion passage", doc_id="doc-Q4B"))])

    import lancedb

    df = lancedb.connect(tmp_path).open_table("doc-Q4B").to_pandas()
    assert len(df) == 1
    assert len(df["vector"].iloc[0]) == 2560


# ---- Qwen3-MLX real-inference end-to-end (Apple-Silicon-only) ------------


def _is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="Qwen3-MLX backend is Apple-Silicon-only; "
    "real-inference tests skipped on Linux / Intel-Mac",
)
def test_qwen3_mlx_0_6b_real_inference_end_to_end(tmp_path: Path) -> None:
    """Mac-only smoke: real Qwen3-Embedding-0.6B in-process.
    `make setup-mlx` populated the HF cache; the session-autouse
    `_warm_qwen3_mlx_embeddings` fixture pre-loaded the singleton.

    Asserts the vector dim matches the model card (1024) and that
    embed → query round-trips through the LanceDB per-doc store.
    """
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    assert adapter.backend == "qwen3-mlx"
    assert adapter.model == "Qwen3-Embedding-0.6B"

    chunks = [
        _wrap(_make_child(
            f"NVDA data center revenue passage {i}", doc_id="doc-QE2E"
        ))
        for i in range(3)
    ]
    adapter.embed(chunks)

    import lancedb

    df = lancedb.connect(tmp_path).open_table("doc-QE2E").to_pandas()
    assert len(df) == 3
    assert all(len(v) == 1024 for v in df["vector"])

    hits = adapter.query(
        "data center revenue", k=3, store="per_doc", doc_id="doc-QE2E"
    )
    assert len(hits) == 3
    assert all(h.doc_id == "doc-QE2E" for h in hits)


# ---- reembed_doc / reembed_corpus (encoder-only re-encode) ----------------


def test_reembed_doc_preserves_metadata_byte_for_byte(tmp_path: Path) -> None:
    """Encoder-only: every metadata column survives reembed unchanged.

    The vector column is re-encoded (and may differ depending on
    encoder determinism), but `text` + `ticker` + `filing_date` + ... +
    `chunker_version` + `contextual_prompt_version` MUST round-trip
    byte-for-byte. Only `embed_model_version` re-stamps to the current
    adapter's identity — but in this test we reembed against the same
    backend/model so that string is unchanged too.
    """
    import lancedb

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    chunks = [
        _wrap(_make_child(f"reembed passage {i}", doc_id="doc-REE"))
        for i in range(3)
    ]
    adapter.embed(chunks)

    before = lancedb.connect(tmp_path).open_table("doc-REE").to_pandas()
    before_sorted = before.sort_values("parent_id").reset_index(drop=True)

    n = adapter.reembed_doc("doc-REE")
    assert n == 3

    after = lancedb.connect(tmp_path).open_table("doc-REE").to_pandas()
    after_sorted = after.sort_values("parent_id").reset_index(drop=True)

    for col in (
        "text",
        "ticker",
        "filing_date",
        "fiscal_period",
        "doc_type",
        "doc_id",
        "parent_id",
        "section_name",
        "chunker_version",
        "contextual_prompt_version",
        "embed_model_version",
    ):
        assert list(before_sorted[col]) == list(after_sorted[col]), (
            f"reembed clobbered column {col!r}"
        )


def test_reembed_doc_preserves_upstream_version_stamps_when_module_constants_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the module-level `CHUNKER_VERSION` or `CONTEXTUAL_CHUNK_PROMPT_VERSION`
    advances between original embed and reembed (e.g., another PR bumped them
    but the chunker output for this doc has NOT been regenerated), the reembed
    must NOT silently re-stamp rows to the new versions — that would lie about
    which chunker / prompt contract actually produced the underlying text.
    Encoder-only path preserves the original stamps verbatim.
    """
    import lancedb

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("first-version passage", doc_id="doc-STAMPED"))])

    monkeypatch.setattr(
        "auto_research.extract.embeddings.CHUNKER_VERSION", "v999",
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings.CONTEXTUAL_CHUNK_PROMPT_VERSION", "v999",
    )

    adapter.reembed_doc("doc-STAMPED")
    df = lancedb.connect(tmp_path).open_table("doc-STAMPED").to_pandas()
    assert df["chunker_version"].iloc[0] == "v1"
    assert df["contextual_prompt_version"].iloc[0] == "v1"


def test_reembed_doc_makes_no_anthropic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reembed path is encoder-only — by definition it must not invoke
    contextual-chunking. Monkeypatching `contextualize_chunks` to raise
    `AssertionError` proves it's never imported / called by reembed.
    """
    import auto_research.extract.chunking_contextual as cc

    def _exploding(**kwargs: object) -> object:
        raise AssertionError(
            "reembed must not invoke contextualize_chunks (Anthropic call)"
        )

    monkeypatch.setattr(cc, "contextualize_chunks", _exploding)

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("no anthropic here", doc_id="doc-NOAN"))])
    # If reembed touched contextualize_chunks at all, this would AssertionError.
    n = adapter.reembed_doc("doc-NOAN")
    assert n == 1


def test_reembed_doc_dim_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing 1024-dim table re-embedded by a 2560-dim encoder is a new
    vector space — refuse loudly. The error message must name both dims so
    the operator can diagnose without reading the adapter source.
    """
    _install_fake_mlx(monkeypatch, dim=1024)
    write_adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    write_adapter.embed([_wrap(_make_child("dim-mismatch source", doc_id="doc-DIM"))])

    _install_fake_mlx(monkeypatch, dim=2560)
    swap_adapter = EmbeddingAdapter(
        backend="qwen3-mlx",
        rag_root=tmp_path,
        mlx_qwen3_model="Qwen3-Embedding-4B",
    )
    with pytest.raises(RuntimeError) as excinfo:
        swap_adapter.reembed_doc("doc-DIM")
    msg = str(excinfo.value)
    assert "1024" in msg and "2560" in msg


def test_reembed_doc_rebuilds_fts_index(tmp_path: Path) -> None:
    """The reembed path overwrites the per-doc table, which drops any
    existing FTS index. The implementation MUST rebuild it with the same
    `_FTS_INDEX_KWARGS` so BM25 query parity is preserved end-to-end.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([
        _wrap(_make_child("export controls passage", doc_id="doc-FTSRE")),
        _wrap(_make_child("dividend declaration", doc_id="doc-FTSRE")),
    ])
    adapter.reembed_doc("doc-FTSRE")

    hits = adapter.bm25_query(
        "export controls", k=2, store="per_doc", doc_id="doc-FTSRE"
    )
    assert hits, "BM25 query returned no hits — FTS index was not rebuilt"
    assert hits[0].text.startswith("export controls")


def test_reembed_corpus_processes_corpus_table(tmp_path: Path) -> None:
    """`reembed_corpus()` re-encodes the `_corpus_narrative` table in place;
    row count preserved, embed_model_version preserved when reembedding
    against the same backend/model.
    """
    import lancedb

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([
        _wrap(_make_child("nvda passage", doc_id="doc-CR1", doc_type="10-K"))
    ])
    adapter.embed([
        _wrap(_make_child("amd passage", doc_id="doc-CR2", doc_type="10-K"))
    ])

    before_n = len(
        lancedb.connect(tmp_path).open_table("_corpus_narrative").to_pandas()
    )
    assert before_n == 2

    n = adapter.reembed_corpus()
    assert n == 2

    after = lancedb.connect(tmp_path).open_table("_corpus_narrative").to_pandas()
    assert len(after) == 2
    assert set(after["doc_id"]) == {"doc-CR1", "doc-CR2"}


def test_reembed_doc_empty_table_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty per-doc table must not invoke the encoder (`self._encode`
    against zero texts is wasted work and a few backends crash on empty
    batch). Returns 0 instead.
    """
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema

    db = lancedb.connect(tmp_path)
    schema = _schema(384)
    db.create_table("doc-EMPTY", data=pa.Table.from_pylist([], schema=schema))

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)

    encode_calls: list[object] = []
    original_encode = adapter._encode

    def _track(texts: list[str], *, input_type: str = "document") -> object:
        encode_calls.append(texts)
        return original_encode(texts, input_type=input_type)

    monkeypatch.setattr(adapter, "_encode", _track)

    n = adapter.reembed_doc("doc-EMPTY")
    assert n == 0
    assert encode_calls == [], "encoder must not be called for an empty table"
