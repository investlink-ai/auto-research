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

Return a single JSON object matching the EightKOutput schema.

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
with `source_quote`) and `confidence` (EXACTLY one of "high",
"medium", or "low" — float confidence is rejected). No other fields
are allowed inside a Claim or Citation.

Example of a fully-formed EightKOutput:

  {
    "cik": "0001234567",
    "accession_number": "0001234567-26-000003",
    "event_classification": "contract",
    "milestone_mentions": [],
    "dilution_language_flags": [
      {
        "citation": {
          "source_quote": "entered into a Material Definitive Agreement"
        },
        "confidence": "high"
      }
    ]
  }

Constraints (apply to every field unless noted):
- source_quote MUST be a verbatim substring of the filing text — preserve
  original whitespace and punctuation; do NOT collapse runs of whitespace
  or rewrite phrasing. The substring is located in the filing by
  whitespace-flexible match; ZERO matches or AMBIGUOUS matches (more
  occurrences in the filing than citations sharing the same quote)
  quarantine the entire output. When a quote naturally repeats in the
  filing (e.g., a recurring entity name), emit one citation per textual
  occurrence so the count matches.
- Choose quotes long and specific enough to be unique unless emitting
  per-occurrence multiple citations.
- DO NOT include `source_span`; character offsets are computed in code.
- DO NOT invent quotes. If a field has no support, return an empty list.
- DO NOT wrap the response in markdown fences or any commentary. The
  response MUST start with `{` and end with `}`.
"""

__all__ = [
    "EIGHT_K_PROMPT",
    "EIGHT_K_PROMPT_VERSION",
]
