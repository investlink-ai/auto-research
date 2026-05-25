"""VCR-recorded integration test for contextual chunking.

Acceptance criterion: "Context generation calls use prompt caching
(verified via VCR)."

What this test actually verifies (be honest):

1. **End-to-end SDK envelope parsing.** `contextualize_chunks` correctly
   routes through `make_extraction_client`, the SDK serializes the
   request, vcrpy returns a recorded `Message` JSON, the SDK deserializes
   it, and the wrapped client returns a fully-typed `Message` with
   `usage.cache_creation_input_tokens` / `cache_read_input_tokens`
   populated. A regression that drops the cache_control hint, breaks
   `_caching.cached_system_block`, or returns a non-Message shape would
   crash here.

2. **Request shape (assertions on the SDK spy).** Each captured call
   carries the `cache_control: ephemeral` marker on the system block and
   the per-parent document metadata. The cassette is a delivery
   mechanism for the responses; the request-shape contract is enforced
   inline.

What this test does NOT verify (the unit test
`test_contextualize_chunks_passes_metadata_and_parent_in_cached_system_block`
covers it):

- That the cache_control hint reaches the actual Anthropic API on the
  wire. vcrpy's `match_on` does not include `body`, so the cassette
  serves the canned responses regardless of body shape. The inline spy
  assertions are the structural contract.
- That a real Anthropic Haiku call would fire the prompt cache for our
  test-fixture parent text (it would not — Haiku's cache minimum is
  ~2048 tokens; our parent is ~40). The cassette's
  `cache_creation_input_tokens=1200` is illustrative, picked to model
  an actual production parent of ~1100 tokens.

Replay is offline. To regenerate against a real ~1.1K-token parent,
delete the cassette, set `ANTHROPIC_API_KEY`, and re-run; vcrpy's
`record_mode="once"` records on absence and replays otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import vcr
from anthropic import Anthropic

import auto_research.extract.chunking_contextual as chunking_contextual
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


@pytest.fixture(autouse=True)
def _reset_client_singleton() -> Iterator[None]:
    chunking_contextual._CLIENT = None
    yield
    chunking_contextual._CLIENT = None


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
        "Item 7. Management's Discussion and Analysis. CRDO fiscal-2024 "
        "AEC product revenue concentration remained weighted to three "
        "hyperscaler customers."
    )
    metadata = ChunkMetadata(
        ticker="CRDO",
        filing_date=date(2024, 6, 18),
        fiscal_period="FY2024",
        doc_type="10-K",
        doc_id="crdo-fy2024",
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
            text="AEC revenue represented 62% of total product revenue in FY2024.",
            char_span=(0, 65),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
        ChildChunk(
            text="Top three hyperscalers accounted for ~70% of FY2024 revenue.",
            char_span=(65, 130),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
    )
    return ChunkSet(parents=(parent,), children=children)


def test_contextual_chunking_envelope_and_request_shape(
    anthropic_api_key: None, tmp_path: Path,
) -> None:
    """Two children of the same parent flow through `contextualize_chunks`,
    the SDK envelope correctly carries usage metadata across vcrpy
    deserialization, AND each captured request has the cache_control hint
    + parent text + per-parent metadata in the system block (the structural
    contract the cassette can't verify by itself)."""
    sdk = Anthropic()
    captured_responses: list[Any] = []
    captured_calls: list[Any] = []
    original_create = sdk.messages.create

    def spy_create(*args: Any, **kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        response = original_create(*args, **kwargs)
        captured_responses.append(response)
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
    assert all(c.context for c in out)

    # Envelope parsing — usage round-trips through SDK Pydantic models.
    assert len(captured_responses) == 2
    first, second = captured_responses
    assert first.usage.cache_creation_input_tokens == 1200
    assert (first.usage.cache_read_input_tokens or 0) == 0
    assert second.usage.cache_read_input_tokens == 1200
    assert (second.usage.cache_creation_input_tokens or 0) == 0

    # Request shape — these are the structural assertions the cassette
    # can't verify (match_on excludes body). Every call must carry the
    # cache_control marker, the parent text, and the metadata fields the
    # prompt asks the model to name.
    for call_kwargs in captured_calls:
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        block = system[0]["text"]
        assert "ticker: CRDO" in block
        assert "fiscal_period: FY2024" in block
        assert "doc_type: 10-K" in block
        assert "section: Item 7" in block
        assert "<parent_passage>" in block

    # The two calls must have distinct user content (the two children).
    assert (
        captured_calls[0]["messages"][0]["content"]
        != captured_calls[1]["messages"][0]["content"]
    )
