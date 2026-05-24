"""Unit tests for the prompts registry convention (Issue #11)."""

from __future__ import annotations

import re

from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)


def test_prompt_version_is_human_readable_tag() -> None:
    """Versions must be `vN` form, not hashes (see bump-prompt-version skill)."""
    assert re.fullmatch(r"v\d+", S_FILINGS_DILUTION_PROMPT_VERSION)


def test_prompt_text_carries_required_extraction_contract() -> None:
    """Every extraction prompt must instruct the model to emit `source_span`
    and `source_quote` for every claim — that's the INV-2 wire format."""
    assert "source_span" in S_FILINGS_DILUTION_PROMPT
    assert "source_quote" in S_FILINGS_DILUTION_PROMPT


def test_prompt_text_carries_placeholder() -> None:
    """Prompt is a template with a `{source_text}` placeholder."""
    assert "{source_text}" in S_FILINGS_DILUTION_PROMPT
