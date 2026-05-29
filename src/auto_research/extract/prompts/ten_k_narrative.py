"""10-K narrative-extraction prompt.

Drives BOTH the single-shot path (`count_tokens(raw_doc) <
SINGLE_SHOT_TOKEN_CUTOFF`) AND the RAG branch. In the RAG branch the
worker stuffs the top-5 reranked parents per field into the user-content
turn; the prompt itself does not change between branches.

The prompt covers narrative TenKOutput fields. `language_novelty_score`
is NOT requested in the prompt — it is computed downstream from the
supplier/customer/risk-factor text vs the prior year's extraction.

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
fields.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- fiscal_period_end: the period-end date in ISO format (YYYY-MM-DD).
- guidance_tone: a single Claim describing the tone of forward-looking
  language in MD&A (e.g., "cautious; gross-margin headwinds called out
  twice"). Confidence categorical (one of "high", "medium", or "low").
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
- going_concern: a single Claim quoting verbatim the auditor's
  "substantial doubt" sentence from the Item 8 audit report or the
  Item 7 liquidity discussion, or null when the audit report carries
  an unqualified opinion. Do NOT paraphrase — quote the actual
  disclaimer sentence. Confidence categorical
  ("high", "medium", "low").
- icfr_material_weaknesses: a list of Claims, one per distinct
  material weakness disclosed in management's Item 9A internal-
  controls-over-financial-reporting (ICFR) report. Empty list when
  management concludes ICFR is effective with no material
  weaknesses. Quote the verbatim weakness description.
- critical_accounting_estimate_changes: a list of Claims for
  accounting estimates that management flags in Item 7 MD&A "Critical
  Accounting Estimates" or the Item 8 Significant Accounting Policies
  note as requiring significant judgment AND where management
  indicates a change versus the prior year (new estimate, methodology
  change, materially different assumptions). Empty list when no YoY
  change is flagged.

A Claim is `{"citation": {"source_quote": "..."}, "confidence":
"high"|"medium"|"low"}` — float confidence is rejected. A
SupplierMention or CustomerMention is `{"mention_text":
"...", "citation": {"source_quote": "..."}, "resolved_ticker": null,
"resolver_confidence": null, "resolver_reasoning": null}`. A
RiskFactorDelta is `{"change_type": "...", "text": "...",
"citation": {"source_quote": "..."}}`. No other fields are allowed
inside any of these objects.

Example of a fully-formed narrative TenKOutput (language_novelty_score
omitted — see Constraints):

  {
    "cik": "0001045810",
    "accession_number": "0001045810-26-000001",
    "fiscal_period_end": "2026-01-31",
    "guidance_tone": {
      "citation": {"source_quote": "We expect cautious growth in fiscal 2027"},
      "confidence": "high"
    },
    "accrual_flags": [
      {
        "citation": {"source_quote": "capitalized software development costs of $118 million"},
        "confidence": "medium"
      }
    ],
    "supplier_mentions": [
      {
        "mention_text": "Taiwan Semiconductor Manufacturing Company (TSMC)",
        "citation": {"source_quote": "Taiwan Semiconductor Manufacturing Company (TSMC) advanced-node wafer pricing"},
        "resolved_ticker": null,
        "resolver_confidence": null,
        "resolver_reasoning": null
      }
    ],
    "customer_mentions": [],
    "risk_factor_deltas": [],
    "going_concern": null,
    "icfr_material_weaknesses": [],
    "critical_accounting_estimate_changes": []
  }

Constraints (apply to every field unless noted):
- source_quote MUST be a verbatim substring of the supplied text —
  preserve original whitespace and punctuation; do NOT collapse runs of
  whitespace. The substring is located by whitespace-flexible match;
  ZERO matches or AMBIGUOUS matches (more occurrences than citations
  sharing the same quote) quarantine the entire output. When a quote
  naturally repeats (e.g., the same supplier named across Risk
  Factors, MD&A, and Properties), emit one SupplierMention per textual
  occurrence so counts match — the worker pairs them in document order.
- Choose quotes long and specific enough to be unique unless
  intentionally emitting per-occurrence multiple citations.
- DO NOT include `source_span`; character offsets are computed in code.
- DO NOT populate `language_novelty_score` (computed downstream; the
  schema defaults it to 0.0).
- DO NOT invent quotes. If a field has no support, return an empty list.
- DO NOT wrap the response in markdown fences or any commentary.
  The response MUST start with `{` and end with `}`.
"""

__all__ = [
    "TEN_K_NARRATIVE_PROMPT",
    "TEN_K_NARRATIVE_PROMPT_VERSION",
]
