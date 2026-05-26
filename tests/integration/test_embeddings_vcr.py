"""VCR-recorded Voyage embedding call for the EmbeddingAdapter.

Cassette captures one POST /v1/embeddings response against voyage-finance-2.
Replay is offline; regenerate by deleting the cassette and re-running with
VOYAGE_API_KEY set (vcrpy record_mode="once" records on absence).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
import vcr

from auto_research.extract.chunking import ChildChunk, ChunkMetadata
from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.extract.embeddings import EmbeddingAdapter

CASSETTE_PATH = (
    Path(__file__).parent / "cassettes" / "test_embeddings"
    / "voyage_embed_finance_v2.yaml"
)

REEMBED_CASSETTE_PATH = (
    Path(__file__).parent / "cassettes" / "test_embeddings"
    / "voyage_reembed_finance_v2.yaml"
)


def _build_vcr() -> vcr.VCR:
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_PATH.parent),
        record_mode="once",
        filter_headers=[("authorization", "REDACTED"), ("x-api-key", "REDACTED")],
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )


@pytest.fixture
def voyage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_MODEL", raising=False)
    # On replay (any cassette this module references exists), supply a
    # dummy key so the adapter picks the voyage backend; the live header
    # is redacted by `filter_headers` in the cassette anyway. On record,
    # leave the shell-provided key alone — overriding it would 401
    # against the live endpoint.
    #
    # Review finding #13: pre-fix this only checked CASSETTE_PATH (the
    # embed cassette), so a session with the reembed cassette but not
    # the embed cassette would fail to inject the dummy key and the
    # reembed test would crash in _voyage_client looking for a real
    # VOYAGE_API_KEY. Now any cassette this module owns is sufficient.
    any_cassette = CASSETTE_PATH.exists() or REEMBED_CASSETTE_PATH.exists()
    if any_cassette and not os.environ.get("VOYAGE_API_KEY"):
        monkeypatch.setenv("VOYAGE_API_KEY", "vk-test-not-a-real-key")


def _chunk(
    text: str, *, doc_id: str = "doc-vcr", doc_type: str = "10-K"
) -> ContextualChildChunk:
    md = ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 3, 15),
        fiscal_period="FY2025",
        doc_type=doc_type,
        doc_id=doc_id,
    )
    child = ChildChunk(
        text=text,
        char_span=(0, len(text)),
        token_count=len(text.split()),
        parent_id=f"{doc_id}:0:{len(text)}",
        section_name="Item 7",
        from_table=False,
        metadata=md,
    )
    return ContextualChildChunk(child=child, context="")


def test_voyage_embed_round_trip_against_recorded_response(
    tmp_path: Path, voyage_env: None
) -> None:
    if not CASSETTE_PATH.exists() and not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip(
            f"VCR cassette missing at {CASSETTE_PATH} and no VOYAGE_API_KEY set. "
            "Record with VOYAGE_API_KEY set: "
            "`pytest tests/integration/test_embeddings_vcr.py`."
        )
    adapter = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    assert adapter.backend == "voyage"
    assert adapter.model == "voyage-finance-2"
    chunks = [_chunk("NVDA China export controls Q4 commentary")]
    with _build_vcr().use_cassette(CASSETTE_PATH.name):
        adapter.embed(chunks)
    from auto_research.extract.materialization import versioned_table_name

    per_doc = tmp_path / f"{versioned_table_name('doc-vcr', adapter.materialization_version)}.lance"
    corpus = tmp_path / f"{versioned_table_name('_corpus_narrative', adapter.materialization_version)}.lance"
    assert per_doc.exists()
    assert corpus.exists()


def test_voyage_reembed_doc_round_trip_against_recorded_response(
    tmp_path: Path, voyage_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reembed against Voyage produces a per-doc table at a DIFFERENT
    materialization namespace than the original embed, with the expected
    1024-dim vector column and the original row count, having made
    exactly two POST /v1/embeddings calls (the initial embed + the
    reembed).

    Skip policy is stricter than the sibling embed round-trip: this test
    skips whenever the cassette is missing, regardless of `VOYAGE_API_KEY`
    presence. The auto-record-on-key path is opt-in via
    `VOYAGE_RECORD_CASSETTES=1`. A `.env`-loaded but stale / invalid key
    must not silently trigger a 401-burning live call when an operator
    runs the integration suite from a checkout.
    """
    if not REEMBED_CASSETTE_PATH.exists() and not os.environ.get(
        "VOYAGE_RECORD_CASSETTES"
    ):
        pytest.skip(
            f"VCR cassette missing at {REEMBED_CASSETTE_PATH}. Record with "
            "a valid VOYAGE_API_KEY and `VOYAGE_RECORD_CASSETTES=1`: "
            "`VOYAGE_RECORD_CASSETTES=1 pytest "
            "tests/integration/test_embeddings_vcr.py "
            "-k voyage_reembed_doc_round_trip`."
        )
    import lancedb

    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        versioned_table_name,
        write_active_materialization,
    )

    adapter_v1 = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
    chunks = [_chunk("Reembed round-trip text against voyage-finance-2")]
    with _build_vcr().use_cassette(REEMBED_CASSETTE_PATH.name) as cassette:
        adapter_v1.embed(chunks)
        # Promote v1 and bump the tag so adapter_v2 has a different
        # materialization namespace to write reembed output into. Both
        # adapters still hit voyage-finance-2 so the cassette responses
        # remain compatible (the request body is determined by model +
        # text, not by adapter materialization_version).
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
            "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-reembed-vcr"
        )
        adapter_v2 = EmbeddingAdapter(backend="voyage", rag_root=tmp_path)
        n = adapter_v2.reembed_doc("doc-vcr")
        # The encoder-only reembed must hit Voyage exactly once for the
        # re-encode (after embed's one call) — a regression where
        # reembed_doc accidentally re-invokes embed() internally
        # (doubling Voyage spend and reintroducing Anthropic calls)
        # would otherwise pass the dim/text/count round-trip below.
        # The corpus-propagation vector-copy is a metadata-only op and
        # must NOT add a third request.
        assert len(cassette.requests) == 2, (
            f"expected exactly 2 POST /v1/embeddings calls (1 embed + "
            f"1 reembed_doc); got {len(cassette.requests)}"
        )

    assert n == 1
    dest_table = versioned_table_name("doc-vcr", adapter_v2.materialization_version)
    df = lancedb.connect(tmp_path).open_table(dest_table).to_pandas()
    assert len(df) == 1
    assert len(df["vector"].iloc[0]) == 1024  # voyage-finance-2 native dim
    assert df["text"].iloc[0] == "Reembed round-trip text against voyage-finance-2"


def test_embed_promote_query_loop_returns_vectors_from_promoted_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end materialization flow: embed under v1, then under v2,
    promote v2, query — the hit must come from the v2 namespace.

    Uses BGE (hermetic, no VCR) so the integration suite can exercise
    the full embed → promote → query loop without network access. The
    same flow applies to Voyage in production; the materialization
    layout is backend-agnostic.

    Asserts the AC criterion verbatim ("assert vectors come from the
    promoted version") by writing two different texts under the two
    materializations and confirming the query result text is the one
    associated with the promoted-version embed.
    """

    from auto_research.extract.embeddings import EmbeddingAdapter
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        versioned_table_name,
        write_active_materialization,
    )

    v1_text = "VERSION-ONE: NVDA Hopper data center revenue commentary"
    v2_text = "VERSION-TWO: NVDA Blackwell architecture launch commentary"

    adapter_v1 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v1.embed([_chunk(v1_text, doc_id="doc-LOOP", doc_type="10-K")])

    monkeypatch.setattr(
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-loop"
    )
    adapter_v2 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v2.embed([_chunk(v2_text, doc_id="doc-LOOP", doc_type="10-K")])

    # Both materializations now exist on disk; nothing is promoted yet.
    assert (
        tmp_path / f"{versioned_table_name('doc-LOOP', adapter_v1.materialization_version)}.lance"
    ).exists()
    assert (
        tmp_path / f"{versioned_table_name('doc-LOOP', adapter_v2.materialization_version)}.lance"
    ).exists()

    # Promote v2 — atomic flip, no row rewrites.
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version=adapter_v2.materialization_version,
            embed_model_version=adapter_v2.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )

    # Query from adapter_v2 (matches active embed_model_version stamp,
    # passes the read-path guard). The hit must be the v2 text because
    # the active pointer routes the query to v2's namespace.
    hits = adapter_v2.query(
        "NVDA data center commentary",
        k=1,
        store="per_doc",
        doc_id="doc-LOOP",
    )
    assert len(hits) == 1
    assert hits[0].text == v2_text
    assert hits[0].text != v1_text


def test_embed_to_inactive_namespace_does_not_perturb_active_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: 'embed() to an inactive namespace does not perturb query
    results from the active namespace'.

    Embed-and-promote v1, then embed v2 (writes to inactive namespace).
    A subsequent query against the v1-matched adapter must still return
    the v1 text. The build path's writes to v2 are silent to live reads,
    which is the whole reason the materialization-versioned layout
    exists.
    """
    from auto_research.extract.embeddings import EmbeddingAdapter
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    v1_text = "INACTIVE-TEST: original V1 text"
    v2_text = "INACTIVE-TEST: silently-written V2 text"

    adapter_v1 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v1.embed([_chunk(v1_text, doc_id="doc-INACT", doc_type="10-K")])
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
        "auto_research.extract.embeddings.EMBED_MODEL_VERSION_TAG", "v-inact"
    )
    adapter_v2 = EmbeddingAdapter(backend="bge", rag_root=tmp_path)
    adapter_v2.embed([_chunk(v2_text, doc_id="doc-INACT", doc_type="10-K")])

    # Query from adapter_v1 — its embed_model_version matches the active
    # pointer, so the read-path mismatch guard is satisfied. The query
    # routes through `versioned_table_name(..., active.version)` =
    # v1's namespace; the v2 namespace is invisible to this read.
    hits = adapter_v1.query(
        "inactive test text", k=1, store="per_doc", doc_id="doc-INACT",
    )
    assert len(hits) == 1
    assert hits[0].text == v1_text
    assert hits[0].text != v2_text
