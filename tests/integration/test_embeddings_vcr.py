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
    # On replay (cassette exists), supply a dummy key so the adapter picks
    # the voyage backend; the live header is redacted by `filter_headers`
    # in the cassette anyway. On record, leave the shell-provided key
    # alone — overriding it would 401 against the live endpoint.
    if CASSETTE_PATH.exists() and not os.environ.get("VOYAGE_API_KEY"):
        monkeypatch.setenv("VOYAGE_API_KEY", "vk-test-not-a-real-key")


def _chunk(text: str) -> ContextualChildChunk:
    md = ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 3, 15),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="doc-vcr",
    )
    child = ChildChunk(
        text=text,
        char_span=(0, len(text)),
        token_count=len(text.split()),
        parent_id="doc-vcr:0:64",
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
    adapter = EmbeddingAdapter(rag_root=tmp_path)
    assert adapter.decision.backend == "voyage"
    assert adapter.decision.model == "voyage-finance-2"
    chunks = [_chunk("NVDA China export controls Q4 commentary")]
    with _build_vcr().use_cassette(CASSETTE_PATH.name):
        adapter.embed(chunks)
    assert (tmp_path / "doc-vcr.lance").exists()
    assert (tmp_path / "_corpus_narrative.lance").exists()
