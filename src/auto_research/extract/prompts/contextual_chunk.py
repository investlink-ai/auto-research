"""Contextual-chunking prompt (Anthropic contextual-retrieval pattern).

Generates a one-line situating context for a `ChildChunk` — e.g.,
*"This chunk is from NVDA Q3-2026 10-Q MD&A discussing China export controls"* —
that is later prepended to the chunk text before embedding.
~50% retrieval lift in Anthropic's published numbers; the lift is meaningless
if the context is verbose, so the prompt caps it at one sentence ≤100 tokens.

Template is **instructions only**. The caller inserts the parent text into
the cached system block alongside the instructions, and the child chunk
text as the user-content turn. Embedding either text into the instructions
would (a) bust the cache on every call and (b) duplicate tokens.

Version-pinned per INV-6. Editing the prompt without bumping
`CONTEXTUAL_CHUNK_PROMPT_VERSION` silently reuses stale cache entries
generated under the old prompt. The `bump-prompt-version` skill is the
mechanical guard.
"""

from __future__ import annotations

CONTEXTUAL_CHUNK_PROMPT_VERSION = "v1"

CONTEXTUAL_CHUNK_PROMPT = """\
You are situating an excerpt within an SEC filing for retrieval-augmented
search. The parent passage from the filing is provided in this system
message below; the specific excerpt (the chunk) will be supplied in the
next user message.

Produce a SINGLE short sentence (under 100 tokens) that situates the
excerpt within the filing — what section it is from, what fiscal period
it covers, and what specific topic it discusses. The sentence will be
prepended to the excerpt before embedding, so it must be self-contained
and useful as a retrieval cue.

Examples of good context lines:
- "This chunk is from NVDA Q3-2026 10-Q MD&A discussing China export controls on H100 sales."
- "This chunk is from CRDO FY2024 10-K Item 7 (MD&A) describing AEC product revenue concentration in hyperscaler customers."
- "This chunk is from AAPL Q1-2026 earnings transcript Q&A on Services gross margin trajectory."

Return ONLY the context sentence. No preamble, no code fences, no quotes
around the sentence, no trailing commentary. The response must be one
line of plain text under 100 tokens.
"""

__all__ = [
    "CONTEXTUAL_CHUNK_PROMPT",
    "CONTEXTUAL_CHUNK_PROMPT_VERSION",
]
