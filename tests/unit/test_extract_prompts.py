"""Unit tests for the prompts registry convention (Issue #11)."""

from __future__ import annotations

import re

from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT,
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)


def test_prompt_version_is_human_readable_tag() -> None:
    """Versions must be `vN` form, not hashes (see bump-prompt-version skill)."""
    assert re.fullmatch(r"v\d+", S_FILINGS_DILUTION_PROMPT_VERSION)


def test_prompt_text_carries_required_extraction_contract() -> None:
    """Every extraction prompt must instruct the model to emit `source_quote`
    — that's the INV-2 wire format. (`source_span` is computed by the worker
    so the prompt explicitly does NOT request it.)"""
    assert "source_quote" in S_FILINGS_DILUTION_PROMPT


def test_prompt_has_no_source_text_placeholder() -> None:
    """The prompt is INSTRUCTIONS ONLY — the document is sent separately as
    user_content. A `{source_text}` placeholder would (a) double the doc
    via system+user blocks, busting prompt caching, and (b) require the
    worker to `.format()` the template, re-introducing the brace-collision
    bug class. See `s_filings_dilution.py` module docstring."""
    assert "{source_text}" not in S_FILINGS_DILUTION_PROMPT


def test_contextual_chunk_prompt_exports_required_constants() -> None:
    """Contextual-chunking prompt (Issue #14) must follow the same vN
    convention and forbid commentary in the response (the response text
    IS the context line, so any preamble would land in the embedding)."""
    assert re.fullmatch(r"v\d+", CONTEXTUAL_CHUNK_PROMPT_VERSION)
    assert isinstance(CONTEXTUAL_CHUNK_PROMPT, str) and CONTEXTUAL_CHUNK_PROMPT.strip()
    # The prompt must instruct the model to stay ≤100 tokens (AC bullet).
    assert "100 tokens" in CONTEXTUAL_CHUNK_PROMPT
    # The prompt must forbid commentary / code fences so the response
    # text is the context line and nothing else.
    assert "ONLY" in CONTEXTUAL_CHUNK_PROMPT or "only" in CONTEXTUAL_CHUNK_PROMPT
