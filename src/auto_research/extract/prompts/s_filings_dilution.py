"""S-1 / S-3 dilution-language extraction prompt (Issue #11)."""

from __future__ import annotations

S_FILINGS_DILUTION_PROMPT_VERSION = "v1"

S_FILINGS_DILUTION_PROMPT = """\
You are extracting structured dilution and capital-raise signals from an SEC
S-1 or S-3 registration statement.

Read <source_text> carefully and return a single JSON object matching the
SFilingOutput schema. Every claim MUST include:
- source_span: tuple [start_char, end_char] giving byte offsets into
  <source_text>.
- source_quote: the verbatim slice source_text[start_char:end_char]. If the
  quote does not appear verbatim in source_text at exactly that span, the
  output will be rejected.

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

Do not invent quotes. If a field has no support in source_text, return an
empty list rather than fabricating a citation.

<source_text>
{source_text}
</source_text>
"""

__all__ = [
    "S_FILINGS_DILUTION_PROMPT",
    "S_FILINGS_DILUTION_PROMPT_VERSION",
]
