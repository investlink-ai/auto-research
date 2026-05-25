"""Token counting and chunk-budget constants.

`cl100k_base` is the closest publicly available tokenizer to Claude's
(Anthropic does not publish a stable public tokenizer). Used here only
for chunk-size budgeting, not for cost estimation, so exact alignment
isn't required.
"""

from __future__ import annotations

from typing import Final

import tiktoken

SINGLE_SHOT_TOKEN_CUTOFF: Final[int] = 100_000
"""Docs below this token count go single-shot; >= goes through RAG (ADR D10)."""

MAX_PARENT_TOKENS: Final[int] = 4_000
MIN_CHILD_TOKENS: Final[int] = 200
MAX_CHILD_TOKENS: Final[int] = 800

# Hard limit on how many tokens an UNBREAKABLE child span can hold
# before we quarantine. Some real-world filings have legitimate
# paragraphs slightly above MAX_CHILD_TOKENS with no boundary tags
# (e.g. iXBRL `<span>`-wrapped sentences without `</p>` inside Item
# 16 boilerplate); failing those would lose the entire ChunkSet. The
# 2x ratio admits these gracefully while still quarantining truly
# pathological inputs (a runaway paragraph > 1600 tokens with no
# boundary is almost always upstream parser breakage, not a real
# filing structure).
MAX_UNBREAKABLE_CHILD_TOKENS: Final[int] = MAX_CHILD_TOKENS * 2

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return cl100k_base token count for `text`."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))
