"""VCR-recorded integration test for contextual chunking (Issue #14 AC).

Acceptance criterion: "Context generation calls use prompt caching
(verified via VCR)."

Cassette captures two `POST /v1/messages` against the same cached system
block (instructions + parent text) but different user contents (two
children of the same parent):

1. First call returns `cache_creation_input_tokens=5000`, `cache_read=0`.
2. Second call returns `cache_creation_input_tokens=0`, `cache_read=5000`.

Replay is offline. To regenerate, delete the cassette and re-run with
`ANTHROPIC_API_KEY` set — vcrpy's `record_mode="once"` records on absence
and replays otherwise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import vcr
from anthropic import Anthropic

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
)
from auto_research.extract.chunking_contextual import contextualize_chunks

CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_chunking_contextual"
    / "two_children_same_parent_cache.yaml"
)


def _build_vcr() -> vcr.VCR:
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_PATH.parent),
        record_mode="once",
        filter_headers=[
            ("x-api-key", "REDACTED"),
            ("authorization", "REDACTED"),
            ("anthropic-organization-id", "REDACTED"),
            ("User-Agent", "auto-research-test"),
        ],
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )


@pytest.fixture
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy key so the SDK constructor is happy on replay."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def _chunkset() -> ChunkSet:
    parent_text = (
        "Item 7. Management's Discussion and Analysis. NVIDIA's Q3-2026 "
        "performance was materially affected by China export controls."
    )
    metadata = ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 11, 19),
        fiscal_period="Q3-2026",
        doc_type="10-Q",
        doc_id="nvda-q3-2026",
    )
    span = (0, len(parent_text))
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 7",
        char_span=span,
        token_count=40,
        table_html=None,
        metadata=metadata,
    )
    parent_id = f"{metadata.doc_id}::{span[0]}-{span[1]}"
    children = (
        ChildChunk(
            text="China export controls reduced H100 revenue by ~$2B in Q3.",
            char_span=(0, 60),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
        ChildChunk(
            text="Mitigation: H20 variant sales ramped to $1.2B in the same period.",
            char_span=(60, 130),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
    )
    return ChunkSet(parents=(parent,), children=children)


def test_contextual_chunking_caches_parent_in_system_block(
    anthropic_api_key: None, tmp_path: Path,
) -> None:
    """Two children of the same parent: first call writes the prompt cache;
    second call reads it. Verifies the Anthropic-side caching that's AC
    bullet 1 ("calls use prompt caching, verified via VCR").

    Spies on the injected SDK to capture the raw `Message.usage` payload —
    `contextualize_chunks` itself returns `ContextualChildChunk`s and doesn't
    surface cache metrics, so the spy is the cleanest way to assert the
    cache_creation / cache_read shape without ambient harness state.
    """
    sdk = Anthropic()
    captured: list[Any] = []
    original_create = sdk.messages.create

    def spy_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)
        captured.append(response)
        return response

    sdk.messages.create = spy_create  # type: ignore[method-assign]

    cassette = _build_vcr()
    with cassette.use_cassette(CASSETTE_PATH.name):
        out = contextualize_chunks(
            chunkset=_chunkset(),
            cache_root=tmp_path,
            anthropic_client=sdk,
        )

    assert len(out) == 2
    # Both children received non-empty generated contexts.
    assert all(c.context for c in out)

    # AC: prompt cache writes on call 1, reads on call 2.
    assert len(captured) == 2
    first, second = captured
    assert first.usage.cache_creation_input_tokens == 5000
    assert (first.usage.cache_read_input_tokens or 0) == 0
    assert second.usage.cache_read_input_tokens == 5000
    assert (second.usage.cache_creation_input_tokens or 0) == 0
