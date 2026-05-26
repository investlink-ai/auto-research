from datetime import date
from pathlib import Path
from typing import Any

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


def _versioned_lance_path(rag_root: Path, base: str, version: str) -> Path:
    """Path to the on-disk LanceDB table for `(base, materialization version)`.

    The `{base}__{version}.lance` shape comes from
    `auto_research.extract.materialization.versioned_table_name`; helper
    here keeps tests readable when they assert directly against the
    filesystem.
    """
    from auto_research.extract.materialization import versioned_table_name

    return rag_root / f"{versioned_table_name(base, version)}.lance"


def _open_adapter_table(adapter: EmbeddingAdapter, base: str) -> Any:
    """Open the LanceDB table that `adapter.embed()` writes for `base` and
    return a pandas DataFrame. Tests use this when they need to inspect
    actual column values written under the adapter's own materialization
    version."""
    import lancedb

    from auto_research.extract.materialization import versioned_table_name

    name = versioned_table_name(base, adapter.materialization_version)
    return lancedb.connect(adapter._rag_root).open_table(name).to_pandas()


def _promote_and_bump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[ContextualChildChunk],
    *,
    backend: str = "bge",
    new_tag: str = "v-new",
) -> tuple["EmbeddingAdapter", "EmbeddingAdapter"]:
    """Embed `chunks` under the current `EMBED_MODEL_VERSION_TAG`, promote
    that materialization, then monkeypatch the tag and return a freshly-
    constructed adapter at the new namespace.

    The returned `(old, new)` pair lets reembed-flow tests act as the
    operator does: build the initial materialization, promote it, then
    construct a new adapter (here simulated by a tag bump rather than an
    actual model swap) whose `materialization_version` differs and whose
    `reembed_doc()` reads the just-promoted active source.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    old = EmbeddingAdapter(backend=backend, rag_root=tmp_path)  # type: ignore[arg-type]
    old.embed(chunks)
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=old.materialization_version,
            embed_model_version=old.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=len({c.child.metadata.doc_id for c in chunks}),
        ),
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", new_tag
    )
    new = EmbeddingAdapter(backend=backend, rag_root=tmp_path)  # type: ignore[arg-type]
    return old, new


# ---- BGE-backed end-to-end -----------------------------------------------


def test_embed_bge_writes_both_stores_atomically(tmp_path: Path) -> None:
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    chunks = [
        _wrap(_make_child(f"NVDA China export controls passage {i}", doc_id="doc-A"))
        for i in range(3)
    ]
    adapter.embed(chunks)

    per_doc = _versioned_lance_path(tmp_path, "doc-A", adapter.materialization_version)
    corpus = _versioned_lance_path(
        tmp_path, "_corpus_narrative", adapter.materialization_version
    )
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

    from auto_research.extract.chunking import CHUNKER_VERSION
    from auto_research.extract.prompts.contextual_chunk import (
        CONTEXTUAL_CHUNK_PROMPT_VERSION,
    )

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("stamped row passage", doc_id="doc-STAMP"))])

    df = _open_adapter_table(adapter, "doc-STAMP")
    assert len(df) == 1
    assert df["chunker_version"].iloc[0] == CHUNKER_VERSION
    assert df["contextual_prompt_version"].iloc[0] == CONTEXTUAL_CHUNK_PROMPT_VERSION
    assert df["embed_model_version"].iloc[0] == f"bge:{BGE_MODEL_ID}:{EMBED_MODEL_VERSION_TAG}"

    # And the per-corpus narrative store carries the same stamps.
    corpus_df = _open_adapter_table(adapter, "_corpus_narrative")
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

    df = _open_adapter_table(adapter, "doc-QDIM")
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

    df = _open_adapter_table(adapter, "doc-Q4B")
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

    df = _open_adapter_table(adapter, "doc-QE2E")
    assert len(df) == 3
    assert all(len(v) == 1024 for v in df["vector"])

    hits = adapter.query(
        "data center revenue", k=3, store="per_doc", doc_id="doc-QE2E"
    )
    assert len(hits) == 3
    assert all(h.doc_id == "doc-QE2E" for h in hits)


# ---- reembed_doc / reembed_corpus (encoder-only re-encode) ----------------


def test_reembed_doc_preserves_metadata_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encoder-only: every metadata column survives reembed unchanged.

    `text` + `ticker` + `filing_date` + ... + `chunker_version` +
    `contextual_prompt_version` round-trip byte-for-byte from the source
    materialization's rows. `embed_model_version` is the one column the
    reembed path re-stamps to the new adapter's identity — by design,
    since the new namespace's whole point is to record that it WAS
    re-encoded by a new model — so it's excluded from the round-trip
    list.
    """
    chunks = [
        _wrap(_make_child(f"reembed passage {i}", doc_id="doc-REE"))
        for i in range(3)
    ]
    old, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    before = _open_adapter_table(old, "doc-REE")
    before_sorted = before.sort_values("parent_id").reset_index(drop=True)

    n = new.reembed_doc("doc-REE")
    assert n == 3

    after = _open_adapter_table(new, "doc-REE")
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
    ):
        assert list(before_sorted[col]) == list(after_sorted[col]), (
            f"reembed clobbered column {col!r}"
        )
    # `embed_model_version` re-stamps to the new adapter's tag — that's the
    # whole point of the cross-namespace reembed.
    assert (
        after_sorted["embed_model_version"].iloc[0]
        == new.embed_model_version
    )


def test_reembed_doc_preserves_upstream_version_stamps_when_module_constants_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `CHUNKER_VERSION` / `CONTEXTUAL_CHUNK_PROMPT_VERSION` advance
    between original embed and reembed (e.g., another PR bumped them but
    the chunker output for this doc has NOT been regenerated), the
    reembed must NOT silently re-stamp rows to the new versions — that
    would lie about which chunker / prompt contract actually produced
    the underlying text. Encoder-only path preserves the original stamps
    verbatim.

    Uses `_promote_and_bump` to land the source rows in the active
    namespace; the post-promotion module-constant bump happens
    independently of the materialization_version flip done by the helper.
    """
    chunks = [_wrap(_make_child("first-version passage", doc_id="doc-STAMPED"))]
    _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    # Bump the upstream module constants AFTER the helper's tag bump. The
    # source rows on disk still carry the original stamps; reembed must
    # preserve them rather than re-stamp from the live module values.
    monkeypatch.setattr(
        "auto_research.extract.embeddings.CHUNKER_VERSION", "v999",
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings.CONTEXTUAL_CHUNK_PROMPT_VERSION", "v999",
    )

    new.reembed_doc("doc-STAMPED")
    df = _open_adapter_table(new, "doc-STAMPED")
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

    chunks = [_wrap(_make_child("no anthropic here", doc_id="doc-NOAN"))]
    _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    def _exploding(**kwargs: object) -> object:
        raise AssertionError(
            "reembed must not invoke contextualize_chunks (Anthropic call)"
        )

    monkeypatch.setattr(cc, "contextualize_chunks", _exploding)

    # If reembed touched contextualize_chunks at all, this would AssertionError.
    n = new.reembed_doc("doc-NOAN")
    assert n == 1


def test_reembed_doc_dim_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing 1024-dim table re-embedded by a 2560-dim encoder is a new
    vector space — refuse loudly. The error message must name both dims so
    the operator can diagnose without reading the adapter source.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    _install_fake_mlx(monkeypatch, dim=1024)
    write_adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    write_adapter.embed([_wrap(_make_child("dim-mismatch source", doc_id="doc-DIM"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=write_adapter.materialization_version,
            embed_model_version=write_adapter.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

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


def test_reembed_doc_rebuilds_fts_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reembed path writes a fresh per-doc table at the destination
    materialization, which has no FTS index until the implementation
    rebuilds it. Rebuilding with the same `_FTS_INDEX_KWARGS` is what
    keeps BM25 query parity across a reembed.

    Promotes BOTH materializations in sequence so the BM25 query at the
    end reads from the destination namespace via the active pointer.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    chunks = [
        _wrap(_make_child("export controls passage", doc_id="doc-FTSRE")),
        _wrap(_make_child("dividend declaration", doc_id="doc-FTSRE")),
    ]
    _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)
    new.reembed_doc("doc-FTSRE")

    # Promote the destination materialization so `bm25_query` reads from it.
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=new.materialization_version,
            embed_model_version=new.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

    hits = new.bm25_query(
        "export controls", k=2, store="per_doc", doc_id="doc-FTSRE"
    )
    assert hits, "BM25 query returned no hits — FTS index was not rebuilt"
    assert hits[0].text.startswith("export controls")


def test_reembed_corpus_processes_corpus_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reembed_corpus()` reads the source corpus rows from the active
    materialization, re-encodes against the adapter's backend/model, and
    writes them into the adapter's own namespace; row count preserved,
    doc_id set preserved.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    # Two narrative docs embedded under v1; embed() requires one doc per
    # call so we land them as two separate calls into the same adapter.
    old = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    old.embed([_wrap(_make_child("nvda passage", doc_id="doc-CR1", doc_type="10-K"))])
    old.embed([_wrap(_make_child("amd passage", doc_id="doc-CR2", doc_type="10-K"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=old.materialization_version,
            embed_model_version=old.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=2,
        ),
    )

    before = _open_adapter_table(old, "_corpus_narrative")
    assert len(before) == 2

    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-reembed-corpus"
    )
    new = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    n = new.reembed_corpus()
    assert n == 2

    after = _open_adapter_table(new, "_corpus_narrative")
    assert len(after) == 2
    assert set(after["doc_id"]) == {"doc-CR1", "doc-CR2"}


def test_reembed_doc_empty_table_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty per-doc table at the active namespace must not invoke the
    encoder (`self._encode` against zero texts is wasted work and a few
    backends crash on empty batch). Returns 0 instead.

    We write an empty source table directly under the versioned namespace
    of the (pre-bump) adapter, promote it, then bump the tag so reembed's
    source/dest namespaces differ and the active-pointer guard is
    satisfied.
    """
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        versioned_table_name,
        write_active_materialization,
    )

    db = lancedb.connect(tmp_path)
    schema = _schema(384)
    # Construct an adapter for its materialization_version, then write the
    # empty source table at that namespace so the reembed source resolves.
    old = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    source_name = versioned_table_name("doc-EMPTY", old.materialization_version)
    db.create_table(source_name, data=pa.Table.from_pylist([], schema=schema))
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=old.materialization_version,
            embed_model_version=old.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-empty-bump"
    )
    new = EmbeddingAdapter(backend="bge", rag_root=tmp_path)

    encode_calls: list[object] = []
    original_encode = new._encode

    def _track(texts: list[str], *, input_type: str = "document") -> object:
        encode_calls.append(texts)
        return original_encode(texts, input_type=input_type)

    monkeypatch.setattr(new, "_encode", _track)

    n = new.reembed_doc("doc-EMPTY")
    assert n == 0
    assert encode_calls == [], "encoder must not be called for an empty table"


def test_reembed_doc_propagates_new_vectors_into_corpus_narrative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After reembed_doc(X), the destination materialization's per-corpus
    narrative rows for doc X must carry the new encoder's vectors and
    stamp. Without this propagation, store='corpus_narrative' queries
    against the new materialization would silently return only rows from
    the original embed sweep — the mixed-vector-space failure mode the
    module preamble warns against.

    For this propagation to apply, the destination corpus table must
    already exist (the embed → reembed sequence in this test creates it
    when the first per-doc reembed for the new materialization writes;
    here we exercise that via two narrative docs and assert both flow
    into the destination corpus table).
    """
    chunks = [
        _wrap(_make_child("narrative passage one", doc_id="doc-PROP")),
        _wrap(_make_child("narrative passage two", doc_id="doc-PROP")),
    ]
    old, new = _promote_and_bump(
        tmp_path, monkeypatch, chunks, new_tag="v999"
    )

    before_corpus = _open_adapter_table(old, "_corpus_narrative")
    assert len(before_corpus) == 2

    new.reembed_doc("doc-PROP")

    # Per-doc table at the new materialization carries the new stamp.
    after_doc = _open_adapter_table(new, "doc-PROP")
    for stamp in after_doc["embed_model_version"]:
        assert stamp.endswith(":v999")
    # Corpus narrative at the destination materialization was created by
    # the propagation path during reembed_doc and carries the new stamp
    # too — proving the vector-copy fired and the dest corpus actually
    # holds the new vectors, not stale ones.
    after_corpus = _open_adapter_table(new, "_corpus_narrative")
    assert len(after_corpus) == 2
    for stamp in after_corpus["embed_model_version"]:
        assert stamp.endswith(":v999"), (
            f"corpus row stamp {stamp!r} was not updated by reembed_doc; "
            "vector-copy into _corpus_narrative did not fire"
        )


def test_reembed_doc_non_narrative_doc_does_not_touch_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-narrative doc_types (S-1, S-3, 8-K, DEF 14A) are never written
    to `_corpus_narrative__*` by embed(). reembed_doc on such a doc must
    not create or write to a corpus table at either the source OR the
    destination materialization — the vector-copy hook must no-op for
    non-narrative-eligible rows.
    """
    import lancedb

    from auto_research.extract.materialization import versioned_table_name

    chunks = [_wrap(_make_child("s-filing passage", doc_id="doc-S1", doc_type="S-1"))]
    old, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    db = lancedb.connect(tmp_path)
    source_corpus = versioned_table_name(
        "_corpus_narrative", old.materialization_version
    )
    dest_corpus = versioned_table_name(
        "_corpus_narrative", new.materialization_version
    )
    # embed() did not create the corpus table for a non-narrative doc.
    assert source_corpus not in db.table_names()

    new.reembed_doc("doc-S1")

    # reembed_doc did not create one at the destination either.
    assert dest_corpus not in db.table_names()


def test_reembed_doc_does_not_re_encode_for_corpus_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus-propagation path must REUSE the vectors computed during
    the per-doc re-encode rather than calling the encoder a second time.
    Otherwise --all reverts to the 2x-cost regime review finding #2
    flagged.
    """
    chunks = [
        _wrap(_make_child(f"prop-cost passage {i}", doc_id="doc-COST"))
        for i in range(3)
    ]
    _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    encode_call_count = 0
    original_encode = new._encode

    def _count(texts: list[str], *, input_type: str = "document") -> object:
        nonlocal encode_call_count
        encode_call_count += 1
        return original_encode(texts, input_type=input_type)

    monkeypatch.setattr(new, "_encode", _count)

    new.reembed_doc("doc-COST")

    assert encode_call_count == 1, (
        f"reembed_doc invoked the encoder {encode_call_count} times; expected "
        "exactly 1 (per-doc only — corpus propagates via vector-copy)"
    )


def test_reembed_corpus_missing_table_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rag_root whose active materialization is populated solely from
    non-narrative doc_types never produces a `_corpus_narrative__*` table.
    reembed_corpus must return 0 silently rather than raise
    FileNotFoundError, so `extract reembed --all` over a non-narrative
    corpus succeeds.
    """
    chunks = [
        _wrap(_make_child("s-filing only", doc_id="doc-NONARR", doc_type="S-1")),
    ]
    _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)

    n = new.reembed_corpus()
    assert n == 0


def test_reembed_doc_dim_check_runs_only_for_non_empty_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty-table short-circuit must precede the dim-mismatch check.
    An empty placeholder table with a 384-dim schema reembedded by a
    1024-dim adapter has zero rows to corrupt — the operation should
    no-op, not raise.

    Constructs the empty 384-dim placeholder directly under a "BGE-shaped"
    materialization namespace, promotes that namespace as active, then
    flips to a qwen3-mlx adapter producing 1024-dim vectors. reembed_doc
    against the empty source must short-circuit to 0 BEFORE the
    cross-dim guard fires.
    """
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        versioned_table_name,
        write_active_materialization,
    )

    # Construct a BGE adapter, write an empty 384-dim table under its
    # own materialization namespace, and promote that namespace.
    bge = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    db = lancedb.connect(tmp_path)
    schema_384 = _schema(384)
    empty_name = versioned_table_name(
        "doc-EMPTY-DIM", bge.materialization_version
    )
    db.create_table(empty_name, data=pa.Table.from_pylist([], schema=schema_384))
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=bge.materialization_version,
            embed_model_version=bge.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

    # Now construct a qwen3-mlx adapter producing 1024-dim vectors.
    _install_fake_mlx(monkeypatch, dim=1024)
    adapter = EmbeddingAdapter(backend="qwen3-mlx", rag_root=tmp_path)
    n = adapter.reembed_doc("doc-EMPTY-DIM")
    assert n == 0


def test_reembed_doc_emits_otel_narrative_count_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OTel `extract.reembed` span carries `embedding.narrative_count`
    so dashboards joining the embed and reembed span sources see
    consistent schema and can track the narrative-eligible fraction of
    reembed work. The materialization_version attribute is also asserted
    so cross-issue dashboards can route by version cleanly.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Rebind the embeddings module's tracer at the source so the span
    # produced inside _reembed_table flows to our exporter.
    import auto_research.extract.embeddings as emb

    original_tracer = emb._tracer
    emb._tracer = trace.get_tracer(__name__, tracer_provider=provider)
    try:
        chunks = [_wrap(_make_child("narrative one", doc_id="doc-OTEL"))]
        _, new = _promote_and_bump(tmp_path, monkeypatch, chunks)
        exporter.clear()
        new.reembed_doc("doc-OTEL")
    finally:
        emb._tracer = original_tracer

    spans = exporter.get_finished_spans()
    reembed_spans = [s for s in spans if s.name == "extract.reembed"]
    assert reembed_spans, "no extract.reembed span emitted"
    attrs = reembed_spans[0].attributes or {}
    assert "embedding.narrative_count" in attrs
    assert attrs["embedding.narrative_count"] == 1
    assert (
        attrs["embedding.materialization_version"] == new.materialization_version
    )


# ---- materialization-versioned read path ---------------------------------


def test_embed_to_inactive_namespace_does_not_perturb_active_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC criterion: building a new materialization (embed to a different
    namespace than the active one) must NOT change what `query()` returns
    from the active namespace. The two materializations are isolated on
    disk; promotion is the only way to flip which the read path uses.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    v1_text = "ACTIVE-V1: original passage that must remain queryable"
    v2_text = "INACTIVE-V2: silent shadow embed that must not surface"

    adapter_v1 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v1.embed([_wrap(_make_child(v1_text, doc_id="doc-INACT"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=adapter_v1.materialization_version,
            embed_model_version=adapter_v1.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-inact-unit"
    )
    adapter_v2 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v2.embed([_wrap(_make_child(v2_text, doc_id="doc-INACT"))])

    hits = adapter_v1.query(
        "original passage", k=1, store="per_doc", doc_id="doc-INACT"
    )
    assert len(hits) == 1
    assert hits[0].text == v1_text


def test_query_raises_on_active_pointer_embed_model_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter constructed with a different `embed_model_version` than
    the active pointer's must NOT silently degrade — read-path queries
    raise loudly. The error names both `embed_model_version` strings so
    operators can identify the misconfiguration without source
    spelunking.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    adapter_v1 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v1.embed([_wrap(_make_child("mismatch passage", doc_id="doc-MM"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=adapter_v1.materialization_version,
            embed_model_version=adapter_v1.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-different"
    )
    adapter_v2 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        adapter_v2.query("anything", k=1, store="per_doc", doc_id="doc-MM")
    msg = str(excinfo.value)
    assert adapter_v1.embed_model_version in msg
    assert adapter_v2.embed_model_version in msg


def test_query_falls_back_to_adapter_own_namespace_when_no_active_pointer(
    tmp_path: Path,
) -> None:
    """Fresh installs and tests embed+query in a single adapter session
    without an explicit promote step. The read path falls back to the
    adapter's own materialization_version when `active_materialization.json`
    is absent.
    """
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("fallback passage", doc_id="doc-FB"))])
    # No write_active_materialization() — pointer file is absent.
    hits = adapter.query(
        "fallback passage", k=1, store="per_doc", doc_id="doc-FB"
    )
    assert len(hits) == 1


def test_query_span_carries_materialization_version_attribute(
    tmp_path: Path,
) -> None:
    """AC: `embedding.materialization_version` on `extract.embed_query`
    spans so dashboards can route by materialization. Sister assertion
    to the reembed-side test; pinned here for the query surface
    independently."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    import auto_research.extract.embeddings as emb

    original_tracer = emb._tracer
    emb._tracer = trace.get_tracer(__name__, tracer_provider=provider)
    try:
        adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
        adapter.embed([_wrap(_make_child("query span text", doc_id="doc-QS"))])
        exporter.clear()
        adapter.query("query span text", k=1, store="per_doc", doc_id="doc-QS")
    finally:
        emb._tracer = original_tracer

    spans = exporter.get_finished_spans()
    query_spans = [s for s in spans if s.name == "extract.embed_query"]
    assert query_spans
    attrs = query_spans[0].attributes or {}
    assert (
        attrs["embedding.materialization_version"]
        == adapter.materialization_version
    )


def test_reembed_doc_raises_when_no_active_materialization(
    tmp_path: Path,
) -> None:
    """Without an active pointer there is no source to re-encode from;
    reembed_doc must raise with a remediation pointing at the embed /
    migration path."""
    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    with pytest.raises(RuntimeError, match="no active materialization"):
        adapter.reembed_doc("doc-NO-ACTIVE")


def test_reembed_doc_raises_when_active_matches_own_version(
    tmp_path: Path,
) -> None:
    """If the active pointer already names this adapter's materialization,
    reembed would re-write into the same namespace — a no-op masquerading
    as an operation. Raise so the operator bumps a version constant first.
    """
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter.embed([_wrap(_make_child("self-reembed test", doc_id="doc-SR"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=adapter.materialization_version,
            embed_model_version=adapter.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )
    with pytest.raises(RuntimeError, match="already matches"):
        adapter.reembed_doc("doc-SR")


# ---- OTel coverage for error paths --------------------------------------


def _make_exporter() -> tuple[Any, Any]:
    """Build an in-memory OTel span exporter and a TracerProvider wired to
    it; return both so callers can install the provider via emb._tracer
    rebinding and read finished spans after the test action."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, trace.get_tracer(__name__, tracer_provider=provider)


def test_query_span_carries_materialization_version_even_on_mismatch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `_resolve_read_active` raises due to embed_model_version
    mismatch, the OTel `extract.embed_query` span must STILL carry
    `embedding.materialization_version` so dashboards bucketing error
    rates by version don't lose the routing key."""
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    adapter_v1 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v1.embed([_wrap(_make_child("query span mismatch", doc_id="doc-QM"))])
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=adapter_v1.materialization_version,
            embed_model_version=adapter_v1.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )
    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-mismatch"
    )
    adapter_v2 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)

    exporter, fake_tracer = _make_exporter()
    import auto_research.extract.embeddings as emb

    original_tracer = emb._tracer
    emb._tracer = fake_tracer
    try:
        with pytest.raises(RuntimeError):
            adapter_v2.query("anything", k=1, store="per_doc", doc_id="doc-QM")
    finally:
        emb._tracer = original_tracer

    spans = exporter.get_finished_spans()
    query_spans = [s for s in spans if s.name == "extract.embed_query"]
    assert query_spans
    attrs = query_spans[0].attributes or {}
    assert (
        attrs["embedding.materialization_version"]
        == adapter_v2.materialization_version
    )


def test_reembed_doc_emits_precondition_error_span_when_no_active_pointer(
    tmp_path: Path,
) -> None:
    """`_resolve_reembed_source_version` raises BEFORE `_reembed_table`
    creates its OTel span; the wrapper code must emit a one-shot
    `extract.reembed` error span so traces still bucket the failure."""
    exporter, fake_tracer = _make_exporter()
    import auto_research.extract.embeddings as emb

    original_tracer = emb._tracer
    emb._tracer = fake_tracer
    try:
        adapter = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
        with pytest.raises(RuntimeError, match="no active materialization"):
            adapter.reembed_doc("doc-NOPE")
    finally:
        emb._tracer = original_tracer

    spans = exporter.get_finished_spans()
    reembed_spans = [s for s in spans if s.name == "extract.reembed"]
    assert reembed_spans, "precondition-error reembed span missing"
    attrs = reembed_spans[0].attributes or {}
    assert attrs["extract.outcome"] == "error"
    assert "embedding.materialization_version" in attrs
