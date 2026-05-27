"""Unit tests for the prompts registry convention."""

from __future__ import annotations

import re

from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT,
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
from auto_research.extract.prompts.eight_k import (
    EIGHT_K_PROMPT,
    EIGHT_K_PROMPT_VERSION,
)
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT,
    TEN_K_FINANCIALS_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT,
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.prompts.transcript import (
    TRANSCRIPT_PROMPT,
    TRANSCRIPT_PROMPT_VERSION,
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
    """Contextual-chunking prompt must follow the same vN convention and
    forbid commentary in the response (the response text IS the context
    line, so any preamble would land in the embedding)."""
    assert re.fullmatch(r"v\d+", CONTEXTUAL_CHUNK_PROMPT_VERSION)
    assert isinstance(CONTEXTUAL_CHUNK_PROMPT, str) and CONTEXTUAL_CHUNK_PROMPT.strip()
    # The prompt must instruct the model to stay ≤100 tokens (AC bullet).
    assert "100 tokens" in CONTEXTUAL_CHUNK_PROMPT
    # The prompt must forbid commentary / code fences so the response
    # text is the context line and nothing else.
    assert "ONLY" in CONTEXTUAL_CHUNK_PROMPT or "only" in CONTEXTUAL_CHUNK_PROMPT


# --- 8-K prompt -------------------------------------------------------------


def test_eight_k_prompt_exports() -> None:
    assert re.fullmatch(r"v\d+", EIGHT_K_PROMPT_VERSION)
    assert "source_quote" in EIGHT_K_PROMPT
    assert "{source_text}" not in EIGHT_K_PROMPT
    # At least one EventClassification enum value present so the model
    # knows the closed-set choices.
    assert "milestone" in EIGHT_K_PROMPT
    assert "partnership" in EIGHT_K_PROMPT


# --- Transcript prompt ------------------------------------------------------


def test_transcript_prompt_exports() -> None:
    assert re.fullmatch(r"v\d+", TRANSCRIPT_PROMPT_VERSION)
    assert "source_quote" in TRANSCRIPT_PROMPT
    assert "{source_text}" not in TRANSCRIPT_PROMPT
    # Must call out the prepared-remarks vs Q&A split so the model
    # produces both fields.
    assert "prepared_remarks_tone" in TRANSCRIPT_PROMPT
    assert "q_and_a_evasiveness" in TRANSCRIPT_PROMPT
    assert "forward_statements" in TRANSCRIPT_PROMPT


# --- 10-K narrative prompt --------------------------------------------------


def test_ten_k_narrative_prompt_exports() -> None:
    assert re.fullmatch(r"v\d+", TEN_K_NARRATIVE_PROMPT_VERSION)
    assert "source_quote" in TEN_K_NARRATIVE_PROMPT
    assert "{source_text}" not in TEN_K_NARRATIVE_PROMPT
    # All narrative TenKOutput fields must be named so the model is told
    # to populate them.
    for field in (
        "guidance_tone",
        "accrual_flags",
        "supplier_mentions",
        "customer_mentions",
        "risk_factor_deltas",
    ):
        assert field in TEN_K_NARRATIVE_PROMPT, f"missing instruction for {field}"
    # Item 8 / financials and language_novelty_score must be explicitly
    # excluded from the narrative prompt — they're handled separately.
    assert "financials" in TEN_K_NARRATIVE_PROMPT  # mentioned as "do not populate"
    assert "language_novelty_score" in TEN_K_NARRATIVE_PROMPT


# --- 10-K Item 8 financials prompt ------------------------------------------


def test_ten_k_financials_prompt_exports() -> None:
    assert re.fullmatch(r"v\d+", TEN_K_FINANCIALS_PROMPT_VERSION)
    assert "source_quote" in TEN_K_FINANCIALS_PROMPT
    assert "value_usd" in TEN_K_FINANCIALS_PROMPT
    # Categorical confidence labels (per user-feedback memory).
    assert "high" in TEN_K_FINANCIALS_PROMPT
    assert "medium" in TEN_K_FINANCIALS_PROMPT
    assert "low" in TEN_K_FINANCIALS_PROMPT
    # Line-item coverage.
    for line in ("revenue", "net_income", "total_assets", "cash_from_operations"):
        assert line in TEN_K_FINANCIALS_PROMPT, f"missing line item {line}"
