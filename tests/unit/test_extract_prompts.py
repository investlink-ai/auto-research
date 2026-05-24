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
