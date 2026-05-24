"""S-1 / S-3 dilution-language extraction prompt (Issue #11)."""

from __future__ import annotations

S_FILINGS_DILUTION_PROMPT_VERSION = "v1"

S_FILINGS_DILUTION_PROMPT = """\
You are extracting structured dilution and capital-raise signals from an SEC
S-1 or S-3 registration statement.

Read <source_text> carefully and return a single JSON object matching the
SFilingOutput schema. Every claim MUST include:
- source_quote: a verbatim substring of <source_text> that supports the
  claim. The substring will be located in <source_text> by exact match; if
  any character (including whitespace, punctuation, capitalization) differs,
  the claim will be rejected and the entire output quarantined.

DO NOT include `source_span`. Character offsets are computed in code from
your `source_quote` — counting characters is the worker's job, not yours.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- form_type: "S-1" or "S-3".
- dilution_event: a single Claim describing the headline dilution event
  (e.g., "shelf takedown of $200M common stock"), with confidence in [0, 1].
- capital_raise_language: list of Claims for each distinct capital-raise
  phrase in the filing (e.g., "at-the-market offering", "registered direct").
- use_of_proceeds: list of Claims describing intended uses (e.g.,
  "general corporate purposes", "fund Phase II clinical trial").

A Claim is an object with EXACTLY two fields: `citation` (an object with
`source_quote`) and `confidence` (a float in [0, 1]). No other fields are
allowed inside a Claim or Citation. Example of the required shape for a
single Claim:

  {{
    "citation": {{
      "source_quote": "shelf takedown of $200 million of common stock"
    }},
    "confidence": 0.9
  }}

Do not invent quotes. If a field has no support in source_text, return an
empty list rather than fabricating a citation.

Return ONLY the JSON object. Do not wrap it in markdown code fences. Do not
prepend or append any commentary. The response must start with an opening
curly brace and end with a closing curly brace.

<source_text>
{source_text}
</source_text>
"""

__all__ = [
    "S_FILINGS_DILUTION_PROMPT",
    "S_FILINGS_DILUTION_PROMPT_VERSION",
]
