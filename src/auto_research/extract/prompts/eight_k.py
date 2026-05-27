"""8-K event-extraction prompt.

Single-shot. 8-Ks are short (typically 2-10 pages); they fit comfortably
under `SINGLE_SHOT_TOKEN_CUTOFF`, so no RAG path is wired here.

Template is **instructions only** — the filing text is sent separately
as the user-content turn (so the cached prefix is the prompt + schema
instructions, not the doc). The placeholder convention from
`s_filings_dilution.py` applies: workers pass `system_prompt=PROMPT`
(no `.format()`) and `user_content=raw_doc`.

Version-pinned per INV-6.
"""

from __future__ import annotations

EIGHT_K_PROMPT_VERSION = "v1"

EIGHT_K_PROMPT = """\
You are extracting structured event signals from an SEC 8-K current
report. The filing text will be supplied in the next user message.

Return a single JSON object matching the EightKOutput schema. Every claim
MUST include:
- source_quote: a verbatim substring of the filing text that supports the
  claim. Preserve the original whitespace exactly — do NOT collapse runs
  of whitespace or rewrite punctuation. The substring will be located in
  the filing by whitespace-flexible match; if no occurrence is found, OR
  if more than one occurrence is found, the claim is rejected and the
  entire output quarantined. Choose quotes long and specific enough to
  be unique in the filing.

DO NOT include `source_span`. Character offsets are computed in code from
your `source_quote`.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- event_classification: EXACTLY one of: "milestone", "partnership",
  "contract", "guidance_change", "leadership_change", "dilution",
  "other". Do not use "other" for events you are merely uncertain about
  — uncertainty belongs in the `confidence` of the supporting claims,
  not in the classification label. Use "other" only when the event
  genuinely falls outside the listed categories.
- milestone_mentions: list of Claims describing each FDA approval,
  clinical read-out, product launch, regulatory clearance, or
  technology milestone the filing announces. One Claim per distinct
  milestone.
- dilution_language_flags: list of Claims describing any language
  signalling potential dilution: shelf takedowns, at-the-market
  offerings, equity raises, convertible-debt issuance, warrant
  exercises.

A Claim is an object with EXACTLY two fields: `citation` (an object
with `source_quote`) and `confidence` (a float in [0, 1]). No other
fields are allowed inside a Claim or Citation. Example:

  {
    "citation": {
      "source_quote": "entered into a Material Definitive Agreement"
    },
    "confidence": 0.9
  }

Do not invent quotes. If a field has no support in the filing, return
an empty list rather than fabricating a citation.

Return ONLY the JSON object. Do not wrap it in markdown code fences.
Do not prepend or append any commentary. The response must start with
an opening curly brace and end with a closing curly brace.
"""

__all__ = [
    "EIGHT_K_PROMPT",
    "EIGHT_K_PROMPT_VERSION",
]
