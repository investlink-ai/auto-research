"""Per-field 10-K narrative extraction prompt (Option A / PR-B).

The RAG path runs one Anthropic call per narrative field so each call
uses the model tier the routing table actually declares: most fields
are Haiku-tier (only the cross-doc supplier/customer mentions need
Sonnet). The unified pre-split call
forced everything through `_NARRATIVE_DEFAULT_TASK = "supplier_mentions"`
(Sonnet) — the Haiku-tier fields paid the wrong tier.

Cache-prefix discipline: the parameterized blocks (`{field_name}`,
`{field_description}`) sit at the END of the template so the long
preamble (constraints, identity-field rules, citation discipline) is
the cached prefix across all fields. With the per-worker
ephemeral-cache marker on the system block, this keeps the
~80%-cached prefix economics intact across the field loop.
"""

from __future__ import annotations

from typing import NamedTuple

from auto_research.extract.schemas import (
    TenKAccrualFlagsPartial,
    TenKCriticalAccountingEstimateChangesPartial,
    TenKCustomerMentionsPartial,
    TenKGoingConcernPartial,
    TenKGuidanceTonePartial,
    TenKIcfrMaterialWeaknessesPartial,
    TenKRiskFactorDeltasPartial,
    TenKSupplierMentionsPartial,
)

TEN_K_NARRATIVE_FIELD_PROMPT_VERSION = "v1"


class TenKNarrativeFieldConfig(NamedTuple):
    """Per-field config for the 10-K RAG loop.

    `schema` is the partial pydantic model whose tool_use `input_schema`
    Anthropic validates against; `field_name` is both the JSON key
    (matched by `getattr(partial, field_name)` in the worker) and the
    cache-namespace token. Bundling these here keeps `schema` and
    `field_name` aligned — splitting them across the worker module + a
    prompt config was a footgun where adding a field required edits
    in two places.
    """

    field_name: str
    schema: type
    description: str
    retrieval_query: str


TEN_K_NARRATIVE_FIELD_PROMPT = """\
You are extracting ONE narrative signal from an SEC 10-K annual report.
The retrieved top passages from the filing sections relevant to this
field will be supplied in the next user message.

This call extracts EXACTLY ONE narrative field — do not populate any
other narrative fields. The output schema's `extra="forbid"` enforces
this; emitting another field will fail validation and quarantine the
call.

Always populate these identity fields on every call so downstream can
verify the per-field calls agree on which filing they extracted from:

- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- fiscal_period_end: the period-end date in ISO format (YYYY-MM-DD).

Constraints (apply to every citation in this output):

- source_quote MUST be a verbatim substring of the supplied passages —
  preserve original whitespace and punctuation; do NOT collapse runs of
  whitespace. The substring is located by whitespace-flexible match;
  ZERO matches or AMBIGUOUS matches (more occurrences than citations
  sharing the same quote) quarantine the entire output. When a quote
  naturally repeats (e.g., a recurring entity name across Risk
  Factors, MD&A, and Properties), emit one citation per textual
  occurrence so counts match — the worker pairs them in document
  order.
- Choose quotes long and specific enough to be unique unless
  intentionally emitting per-occurrence multiple citations.
- DO NOT include `source_span`; character offsets are computed in code.
- DO NOT invent quotes. If the field has no support in the retrieved
  passages, emit the empty list (or, for single-Claim fields, you
  MUST still cite — silent omission is not an option; if the
  passages truly carry no signal, cite the strongest available
  hedging language).

Confidence on every Claim is EXACTLY one of "high", "medium", or "low"
— float confidence is rejected.

Now extract the field `{field_name}` from this filing:

{field_description}
"""


# Per-field configuration consumed by the RAG worker loop. Each entry
# is `(field_name, field_description, retrieval_query)`. The
# `retrieval_query` drives the per-field rerank — the prompt is silent
# about retrieval and the worker is silent about the prompt-level
# field semantics, so changing one of the three doesn't require
# coordinated edits across the others.
#
# The order of this list is load-bearing: the RAG worker iterates in
# this order and stages per-field cache writes; reordering changes the
# observable per-field cache namespace. New fields go at the end.
TEN_K_NARRATIVE_FIELD_CONFIGS: tuple[TenKNarrativeFieldConfig, ...] = (
    TenKNarrativeFieldConfig(
        field_name="guidance_tone",
        schema=TenKGuidanceTonePartial,
        description=(
            "A single Claim describing the tone of forward-looking language "
            "in MD&A (e.g., 'cautious; gross-margin headwinds called out "
            "twice'). Quote the passage in MD&A that most strongly carries "
            "the tone, not a generic disclaimer."
        ),
        retrieval_query=(
            "What is management's tone on forward growth, gross margin, and "
            "demand in the MD&A section?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="accrual_flags",
        schema=TenKAccrualFlagsPartial,
        description=(
            "A list of Claims flagging accrual-quality concerns — large "
            "unbilled receivables, deferred revenue swings, capitalized R&D "
            "growing faster than revenue, restructuring-charge resets. One "
            "Claim per distinct concern. Empty list when none surface in "
            "the retrieved passages."
        ),
        retrieval_query=(
            "What are the accrual-quality concerns: unbilled receivables, "
            "deferred revenue swings, capitalized R&D, restructuring resets?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="supplier_mentions",
        schema=TenKSupplierMentionsPartial,
        description=(
            "A list of SupplierMention objects naming specific named "
            "suppliers (e.g., TSMC, Foxconn, Samsung, ASML). Each "
            "SupplierMention has:\n"
            "  - mention_text: the verbatim name as it appears in the "
            "filing.\n"
            "  - citation: {source_quote: '...'}.\n"
            "  - resolved_ticker, resolver_confidence, resolver_reasoning: "
            "ALL null — a separate resolver step runs later. Do NOT "
            "fabricate or guess.\n"
            "Empty list when no specific named supplier is called out."
        ),
        retrieval_query=(
            "Which specific named suppliers (e.g., TSMC, Foxconn, Samsung, "
            "ASML) does the company rely on?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="customer_mentions",
        schema=TenKCustomerMentionsPartial,
        description=(
            "A list of CustomerMention objects (same shape as "
            "SupplierMention). Include named customers — typically "
            "hyperscaler or enterprise customers explicitly called out "
            "(NVDA, MSFT, GOOGL, AMZN, META) — NOT vague references like "
            "'certain large customers' or 'our key customer'. Same "
            "null-resolver-fields discipline as supplier mentions. Empty "
            "list when no specific named customer is called out."
        ),
        retrieval_query=(
            "Which specific named customers — hyperscalers, large "
            "enterprises — are explicitly called out by name?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="risk_factor_deltas",
        schema=TenKRiskFactorDeltasPartial,
        description=(
            "A list of RiskFactorDelta objects, each:\n"
            "  - change_type: EXACTLY one of 'added', 'removed', or "
            "'modified' (vs the prior year's 10-K).\n"
            "  - text: the new (or removed, or modified) risk-factor "
            "language.\n"
            "  - citation: {source_quote: '...'} anchoring the text in "
            "THIS filing.\n"
            "When the prior year is not available in the supplied passages, "
            "treat all Item 1A risk factors as 'added'."
        ),
        retrieval_query=(
            "What new, removed, or modified Item 1A risk factors does this "
            "filing disclose?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="going_concern",
        schema=TenKGoingConcernPartial,
        description=(
            "A single Claim quoting verbatim the auditor's "
            "'substantial doubt' sentence from the Item 8 audit report "
            "or the Item 7 liquidity discussion, or null when the audit "
            "report carries an unqualified opinion. Do NOT paraphrase — "
            "quote the actual disclaimer sentence."
        ),
        retrieval_query=(
            "Does the auditor's report in Item 8 or the liquidity "
            "discussion in Item 7 express substantial doubt about the "
            "company's ability to continue as a going concern?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="icfr_material_weaknesses",
        schema=TenKIcfrMaterialWeaknessesPartial,
        description=(
            "A list of Claims, one per distinct material weakness "
            "disclosed in management's Item 9A internal-controls-over-"
            "financial-reporting (ICFR) report. Empty list when "
            "management concludes ICFR is effective with no material "
            "weaknesses identified. Quote the verbatim weakness "
            "description sentence (e.g., 'we did not maintain "
            "effective controls over X')."
        ),
        retrieval_query=(
            "Does management's Item 9A internal-controls-over-financial-"
            "reporting report identify any material weaknesses in ICFR?"
        ),
    ),
    TenKNarrativeFieldConfig(
        field_name="critical_accounting_estimate_changes",
        schema=TenKCriticalAccountingEstimateChangesPartial,
        description=(
            "A list of Claims for accounting estimates that management "
            "flags in Item 7 MD&A 'Critical Accounting Estimates' or "
            "the Item 8 Significant Accounting Policies note as "
            "requiring significant judgment AND where management "
            "indicates a change versus the prior year (new estimate, "
            "methodology change, materially different assumptions). "
            "Empty list when no YoY change is flagged. Quote the "
            "verbatim change-indicating sentence. "
            "Do NOT flag routine annual updates driven solely by market "
            "input changes (e.g., updated discount rates, commodity "
            "prices) unless management explicitly calls out a "
            "methodological or structural change to the estimate itself."
        ),
        retrieval_query=(
            "Which critical accounting estimates does Item 7 MD&A or "
            "the Item 8 footnotes flag as new, changed, or requiring "
            "materially different assumptions versus the prior year?"
        ),
    ),
)


__all__ = [
    "TEN_K_NARRATIVE_FIELD_CONFIGS",
    "TEN_K_NARRATIVE_FIELD_PROMPT",
    "TEN_K_NARRATIVE_FIELD_PROMPT_VERSION",
    "TenKNarrativeFieldConfig",
]
