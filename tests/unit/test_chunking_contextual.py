"""Unit tests for `auto_research.extract.chunking_contextual` (Issue #14).

Hermetic: the Anthropic SDK is mocked. Each test exercises one AC bullet
or one behavior (cache hit, prompt-version invalidation, ≤100-token cap,
prepend ordering, prompt-caching block shape).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
    count_tokens,
)
from auto_research.extract.chunking_contextual import (
    ContextualChildChunk,
    contextualize_chunks,
)


def _metadata() -> ChunkMetadata:
    return ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 11, 19),
        fiscal_period="Q3-2026",
        doc_type="10-Q",
        doc_id="nvda-q3-2026",
    )


def _make_chunkset(
    *,
    parent_text: str = "Item 7. MD&A. We expect China export controls to weigh on H100 sales.",
    child_texts: tuple[str, ...] = (
        "China export controls reduced H100 revenue by ~$2B in Q3.",
        "Mitigation: H20 variant sales ramped to $1.2B in the same period.",
    ),
) -> ChunkSet:
    metadata = _metadata()
    parent_span = (0, len(parent_text))
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 7",
        char_span=parent_span,
        token_count=20,
        table_html=None,
        metadata=metadata,
    )
    parent_id = f"{metadata.doc_id}::{parent_span[0]}-{parent_span[1]}"
    children = tuple(
        ChildChunk(
            text=text,
            char_span=(i * 100, i * 100 + len(text)),
            token_count=len(text.split()),
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        )
        for i, text in enumerate(child_texts)
    )
    return ChunkSet(parents=(parent,), children=children)


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _fake_client(*texts: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.side_effect = [_make_response(t) for t in texts]
    return cast(anthropic.Anthropic, fake)


def test_contextualize_chunks_returns_one_per_child(tmp_path: Path) -> None:
    chunkset = _make_chunkset()
    client = _fake_client(
        "This chunk is from NVDA Q3-2026 10-Q MD&A on China export controls.",
        "This chunk is from NVDA Q3-2026 10-Q MD&A on H20 variant ramp.",
    )

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    assert len(out) == 2
    assert all(isinstance(c, ContextualChildChunk) for c in out)
    assert out[0].context.startswith("This chunk is from NVDA Q3-2026")
    # `embedding_text` prepends context with a blank line before the chunk.
    assert out[0].embedding_text.startswith(out[0].context + "\n\n")
    assert out[0].embedding_text.endswith(chunkset.children[0].text)


def test_contextualize_chunks_second_call_is_cache_hit(tmp_path: Path) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    sdk = MagicMock()
    sdk.messages.create.return_value = _make_response(
        "This chunk is from NVDA Q3-2026 10-Q MD&A example."
    )

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )
    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )
    # One SDK call across two invocations — the second came from cache.
    assert sdk.messages.create.call_count == 1


def test_contextualize_chunks_prompt_version_bump_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-6 / ADR D6 evidence: changing the prompt version forces a fresh
    SDK call even though the chunk + parent text are unchanged."""
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    sdk = MagicMock()
    sdk.messages.create.return_value = _make_response(
        "This chunk is from NVDA Q3-2026 10-Q MD&A example."
    )

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )

    # Patch the module's view of the prompt version (mirrors what a real
    # `bump-prompt-version` skill edit would do post-restart).
    monkeypatch.setattr(
        "auto_research.extract.chunking_contextual.CONTEXTUAL_CHUNK_PROMPT_VERSION",
        "v2",
    )

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )

    # Cache key changed with the prompt version → fresh SDK call.
    assert sdk.messages.create.call_count == 2


def test_contextualize_chunks_drops_over_cap_context(tmp_path: Path) -> None:
    """A >100-token generated context falls through to `context=""` rather
    than quarantining — contextual chunking is a retrieval-lift feature,
    not a citation-grounding invariant."""
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    bloated = " ".join(["overlong"] * 200)  # ~200 tokens by cl100k_base
    client = _fake_client(bloated)

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    assert out[0].context == ""
    # Drop → `embedding_text` is the chunk text alone.
    assert out[0].embedding_text == chunkset.children[0].text


def test_contextualize_chunks_enforces_100_token_cap_on_kept_contexts(
    tmp_path: Path,
) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    short = "This chunk is from NVDA Q3-2026 10-Q MD&A on China export controls."
    client = _fake_client(short)

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    assert out[0].context  # non-empty
    assert count_tokens(out[0].context) <= 100


def test_contextualize_chunks_passes_parent_text_in_cached_system_block(
    tmp_path: Path,
) -> None:
    """The Anthropic prompt cache hits only when the system block is stable
    across calls. Parent text MUST live in the cached system block; child
    text MUST live in the user message — so multiple children of the same
    parent share a cached prefix."""
    chunkset = _make_chunkset(
        parent_text="UNIQUE_PARENT_MARKER section text",
        child_texts=("UNIQUE_CHILD_MARKER inner text",),
    )
    client = _fake_client("ctx")

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    fake = cast(MagicMock, client)
    call = fake.messages.create.call_args
    system_blocks = call.kwargs["system"]
    assert isinstance(system_blocks, list)
    # The shared `cached_system_block` helper emits one block with
    # `cache_control: ephemeral`.
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    # Parent text is in the cached system block; child text is in user content.
    assert "UNIQUE_PARENT_MARKER" in system_blocks[0]["text"]
    assert "UNIQUE_PARENT_MARKER" not in call.kwargs["messages"][0]["content"]
    assert call.kwargs["messages"][0]["content"] == "UNIQUE_CHILD_MARKER inner text"
