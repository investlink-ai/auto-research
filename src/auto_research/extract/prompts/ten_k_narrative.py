"""10-K narrative-extraction prompt.

Drives BOTH the single-shot path (`count_tokens(raw_doc) <
SINGLE_SHOT_TOKEN_CUTOFF`) AND the RAG branch. In the RAG branch the
worker stuffs the top-5 reranked parents per field into the user-content
turn; the prompt itself does not change between branches.

The prompt covers ONLY narrative TenKOutput fields. `financials` (Item 8)
is extracted by a separate worker path from `ParentChunk.table_html` and
has its own prompt + schema (`ten_k_financials.py`). `language_novelty_
score` is NOT requested in the prompt — it is computed downstream from
the supplier/customer/risk-factor text vs the prior year's extraction.

Version-pinned per INV-6.
"""

from __future__ import annotations

TEN_K_NARRATIVE_PROMPT_VERSION = "v1"

TEN_K_NARRATIVE_PROMPT = """\
You are extracting narrative signals from an SEC 10-K annual report. The
10-K text (or, in the RAG branch, the retrieved top passages) will be
supplied in the next user message. Focus on Items 1A (Risk Factors),
7 (MD&A), and 7A (Quantitative and Qualitative Disclosures About Market
Risk) — those sections dominate the language-signal value for downstream
signals.

Return a single JSON object matching the TenKOutput schema's narrative
fields. Every claim MUST include:
- source_quote: a verbatim substring of the supplied text supporting
  the claim. Preserve the original whitespace exactly — do NOT collapse
  runs of whitespace. The substring will be located in the supplied
  text by whitespace-flexible match; if no occurrence is found, OR if
  more than one occurrence is found, the claim is rejected and the
  output quarantined. Choose quotes long and specific enough to be
  unique in the supplied text.

DO NOT include `source_span`. Character offsets are computed in code.
DO NOT populate `financials` (Item 8 is handled by a separate prompt).
DO NOT populate `language_novelty_score` (computed downstream).

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- fiscal_period_end: the period-end date in ISO format (YYYY-MM-DD).
- guidance_tone: a single Claim describing the tone of forward-looking
  language in MD&A (e.g., "cautious; gross-margin headwinds called out
  twice"). Confidence in [0, 1].
- accrual_flags: list of Claims flagging accrual-quality concerns —
  large unbilled receivables, deferred revenue swings, capitalized R&D
  growing faster than revenue, restructuring-charge resets.
- supplier_mentions: list of SupplierMention objects naming specific
  named suppliers (e.g., TSMC, Foxconn, Samsung, ASML). Each
  SupplierMention has:
  - mention_text: the verbatim name as it appears in the filing.
  - citation: {source_quote: "..."}.
  - resolved_ticker, resolver_confidence, resolver_reasoning: ALL null
    — a separate resolver step runs later. Do NOT fabricate or guess.
- customer_mentions: list of CustomerMention objects (same shape as
  SupplierMention). Include named customers — typically hyperscaler or
  enterprise customers explicitly called out (NVDA, MSFT, GOOGL, AMZN,
  META) — NOT vague references like "certain large customers" or
  "our key customer". Same null-resolver-fields discipline as
  supplier mentions.
- risk_factor_deltas: list of RiskFactorDelta objects, each:
  - change_type: EXACTLY one of "added", "removed", or "modified"
    (vs the prior year's 10-K).
  - text: the new (or removed, or modified) risk-factor language.
  - citation: {source_quote: "..."} anchoring the text in THIS filing.
  When the prior year is not available in the supplied text, treat all
  Item 1A risk factors as "added".

A Claim is `{"citation": {"source_quote": "..."}, "confidence":
0.0-1.0}`. A SupplierMention or CustomerMention is `{"mention_text":
"...", "citation": {"source_quote": "..."}, "resolved_ticker": null,
"resolver_confidence": null, "resolver_reasoning": null}`. A
RiskFactorDelta is `{"change_type": "...", "text": "...",
"citation": {"source_quote": "..."}}`. No other fields are allowed
inside any of these objects.

If a field has no support in the supplied text, return an empty list.
Do not fabricate citations.

Return ONLY the JSON object. No markdown code fences. No commentary.
"""

__all__ = [
    "TEN_K_NARRATIVE_PROMPT",
    "TEN_K_NARRATIVE_PROMPT_VERSION",
]
