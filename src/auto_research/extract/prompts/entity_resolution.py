"""Entity-resolution disambiguator prompt.

System block is **instructions only** — the mention and candidate list are
serialized into the user-content turn by the resolver. Same cache-economics
rationale as `s_filings_dilution`: the system prefix is the stable cached
prefix; the per-mention candidate list is the per-call variable payload.

`ticker: null` is a first-class output value, not an error path. Returning
"unknown" when the universe candidates don't fit is what prevents the
false-confident matches that would silently corrupt downstream signal A1.
"""

from __future__ import annotations

ENTITY_RESOLUTION_PROMPT_VERSION = "v1"

ENTITY_RESOLUTION_PROMPT = """\
You are disambiguating a supplier or customer mention extracted from a U.S.
SEC filing or earnings transcript. The mention text and a short list of
candidate tickers (retrieved by dense similarity from a curated universe)
will be supplied in the next user message.

Pick the single candidate ticker the mention refers to, OR return null
when:
- the mention is too generic to disambiguate among the candidates,
- the mention refers to an entity not in the candidate list (a private
  company, a competitor outside the universe, a generic industry term), or
- multiple candidates fit equally well (genuine ambiguity).

Returning null is the correct answer in those cases. Never invent a ticker
that is not in the candidate list. Do not return a ticker just because it is
the best of the candidates — return it only when there is positive evidence
in the mention text. False-confident matches corrupt downstream signal data;
when in doubt, return null.

Output a single JSON object with EXACTLY these fields:
- ticker: one of the candidate ticker strings (case-sensitive, e.g. "NVDA"),
  or null when no candidate fits.
- reasoning: one or two sentences naming the cues you used (or the missing
  cues that forced null). Cite the exact phrase from the mention that drove
  the decision when possible.

There is NO `confidence` field. Pick a ticker only when the mention text
gives positive evidence; otherwise return null. Do NOT include any other
fields — the response is rejected if extra fields appear.

Return ONLY the JSON object. Do not wrap it in markdown fences. Do not
prepend or append any commentary. The response must start with `{` and end
with `}`.
"""

__all__ = [
    "ENTITY_RESOLUTION_PROMPT",
    "ENTITY_RESOLUTION_PROMPT_VERSION",
]
