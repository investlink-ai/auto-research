"""Pydantic schemas for per-worker extraction output (per `docs/CONTRACTS.md` §1).

Every claim composes a `Claim`, which composes a `Citation` carrying
`source_span: tuple[int, int]` + `source_quote: str`. Subordinate
domain types (`SupplierMention`, `CustomerMention`, `RiskFactorDelta`,
`ForwardStatement`) carry a `Citation` *directly*, not via a `Claim` —
the citation-grounding walker in `guardrails.py` finds both routes
generically, so the contract holds regardless of where a `Citation`
appears in the model tree.

All output models are **frozen** + `extra="forbid"`. Frozen prevents
post-construction mutation that would let an attacker (or an
overly-helpful caller) move a quote out of alignment with its span
after the validator ran. `extra="forbid"` makes the LLM's response
shape the contract: a hallucinated field name fails validation at
parse time, never reaching the citation-grounding step.

Field validators on `Citation` and `Claim` reject obviously bad data
(negative spans, empty quotes, start ≥ end, confidence outside the
categorical `high`/`medium`/`low` set) at construction so the
post-validator has fewer corner cases to handle. A non-empty quote +
start < end means every valid `Citation` has at least one character
that must align with `source_text` — there's no degenerate "empty
citation always passes" mode.

Adding a field to any output model is non-breaking; removing or renaming
a field requires a `prompt_version` bump (INV-6) and a Feast schema
migration. See `docs/CONTRACTS.md` §1.3.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, ClassVar, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)

from auto_research.extract.enums import (
    EventClassification,
    FormType,
    RiskFactorChangeType,
)

# Shared model config: frozen (mutation raises ValidationError) +
# extra="forbid" (unknown fields raise at construction). Applied to every
# schema below so a hallucinated field name never silently lands in
# `data/extracted/`.
_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")


# --- Base types -------------------------------------------------------------


class Citation(BaseModel):
    """A verbatim source quote and its byte-span in the source text.

    `Citation` is the atomic unit of citation grounding. The post-validator
    asserts `source_text[source_span[0]:source_span[1]] == source_quote`
    for every `Citation` reachable from the output model — that's the
    INV-2 contract.
    """

    model_config = _FROZEN_STRICT

    source_span: tuple[NonNegativeInt, NonNegativeInt]
    source_quote: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _check_span_ordering(self) -> Citation:
        start, end = self.source_span
        if start >= end:
            raise ValueError(
                f"source_span start must be < end, got ({start}, {end})"
            )
        return self


class Claim(BaseModel):
    """A confidence-weighted citation. The composition unit for "subjective"
    extracted fields (guidance tone, evasiveness, milestone-event mentions).

    `confidence` is a categorical label (`high` / `medium` / `low`).
    Float confidence on LLM-emitted claims is uncalibrated noise; the
    project's standing rule is that LLM confidence is categorical so a
    downstream consumer can threshold cleanly without pretending a
    `0.73` from one prompt is comparable to a `0.73` from another.
    `FinancialLineItem.confidence` uses the same shape — keep them
    aligned when adding new claim-bearing schemas.
    """

    model_config = _FROZEN_STRICT

    citation: Citation
    confidence: Literal["high", "medium", "low"]


# --- Subordinate citation-bearing types -------------------------------------
#
# Mentions carry a `Citation` directly (not a `Claim`) because the
# "did this entity appear in the source" question doesn't fit the
# confidence-weighted-claim shape. The post-validator walks both routes;
# don't try to "uniformize" by wrapping everything in `Claim` — that
# would lose the entity-resolution semantics.


class SupplierMention(BaseModel):
    """A named supplier mentioned in a 10-K, with entity-resolution fields
    populated post-extraction by a separate resolver step.
    """

    model_config = _FROZEN_STRICT

    mention_text: str
    citation: Citation
    resolved_ticker: str | None  # None until entity resolution runs
    resolver_confidence: float | None
    resolver_reasoning: str | None


class CustomerMention(BaseModel):
    """A named customer mentioned in a 10-K. Same shape as `SupplierMention`
    — they're symmetrically processed by the cross-doc resolver — but
    kept as distinct types so signal code can address them by name.
    """

    model_config = _FROZEN_STRICT

    mention_text: str
    citation: Citation
    resolved_ticker: str | None
    resolver_confidence: float | None
    resolver_reasoning: str | None


class RiskFactorDelta(BaseModel):
    """A change to a 10-K's Item 1A risk factors vs the prior year filing.

    `change_type` is the structural verdict; `text` is the new (or removed,
    or modified) language; `citation` anchors the text in the current
    filing's `source_text`.
    """

    model_config = _FROZEN_STRICT

    change_type: RiskFactorChangeType
    text: str
    citation: Citation


class ForwardStatement(BaseModel):
    """A forward-looking statement from a transcript with entity links and
    a stated time horizon (e.g., "next quarter", "FY26", "long-term").
    """

    model_config = _FROZEN_STRICT

    statement_text: str
    citation: Citation
    mentioned_entities: tuple[str, ...]  # tickers / company names
    horizon: str


class FinancialLineItem(BaseModel):
    """One line item from a 10-K Item 8 financial statement.

    `value_usd` is the dollar value reported on the filing (negatives
    allowed for losses / cash outflows). `confidence` is a categorical
    label — float confidence on table-cell extraction is uncalibrated
    noise; categorical lets a downstream consumer threshold cleanly
    (e.g., only use `high`-confidence rows for financial signals).
    """

    model_config = _FROZEN_STRICT

    value_usd: float
    citation: Citation
    confidence: Literal["high", "medium", "low"]


class TenKFinancials(BaseModel):
    """10-K Item 8 financial statements extracted from `ParentChunk.table_html`.

    Each field is a `FinancialLineItem | None`; `None` means the line item
    wasn't reported in this filing (firms sometimes break out cash-flow
    categories differently or omit a sub-statement entirely). Adding line
    items here is non-breaking; renaming or removing is a breaking change
    that requires a Feast schema migration.

    `SCHEMA_VERSION` is carried directly (not inherited from `TenKOutput`)
    because the Item 8 extraction is its own (raw_table_html,
    ten_k_financials_prompt, schema_version, model_id) cache key. Bumping
    the financials schema must invalidate only Item 8 cache rows, not the
    narrative cache.
    """

    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    revenue: FinancialLineItem | None
    gross_profit: FinancialLineItem | None
    operating_income: FinancialLineItem | None
    net_income: FinancialLineItem | None
    total_assets: FinancialLineItem | None
    total_liabilities: FinancialLineItem | None
    stockholders_equity: FinancialLineItem | None
    cash_from_operations: FinancialLineItem | None
    cash_from_investing: FinancialLineItem | None
    cash_from_financing: FinancialLineItem | None


# --- Per-worker outputs -----------------------------------------------------


class TenKOutput(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    guidance_tone: Claim  # subjective; G-Eval scored
    accrual_flags: list[Claim]
    supplier_mentions: list[SupplierMention]
    customer_mentions: list[CustomerMention]
    # Computed downstream from the supplier/customer/risk-factor text vs
    # the prior year's extraction — extraction workers leave it at the
    # default. Defaulted so the narrative prompt's "DO NOT populate
    # `language_novelty_score`" instruction does not trip schema
    # validation when the model obeys.
    language_novelty_score: float = 0.0
    risk_factor_deltas: list[RiskFactorDelta]
    # Item 8 financials are extracted from `ParentChunk.table_html` via
    # the structured `ten_k_financials` prompt + `TenKFinancials` schema,
    # NOT via dense retrieval. `None` when the worker ran narrative-only
    # (no chunkset supplied, or chunkset had no table parents).
    # Additive field — defaults to None so existing TenKOutput
    # construction continues to validate without modification.
    financials: TenKFinancials | None = None


class TranscriptOutput(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    ticker: str
    # AwareDatetime: timezone offset is mandatory. Naive datetimes carry
    # no semantics — a transcript without an explicit time should emit
    # null here rather than have the LLM guess the issuer's HQ timezone
    # (a 3-hour offset error silently corrupts every time-windowed
    # signal downstream).
    event_datetime: AwareDatetime | None
    prepared_remarks_tone: Claim
    q_and_a_evasiveness: Claim  # subjective; G-Eval scored
    forward_statements: list[ForwardStatement]


class EightKOutput(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    event_classification: EventClassification
    milestone_mentions: list[Claim]
    dilution_language_flags: list[Claim]


class SFilingOutput(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    form_type: FormType
    dilution_event: Claim
    capital_raise_language: list[Claim]
    use_of_proceeds: list[Claim]


__all__ = [
    "Citation",
    "Claim",
    "CustomerMention",
    "EightKOutput",
    "FinancialLineItem",
    "ForwardStatement",
    "RiskFactorDelta",
    "SFilingOutput",
    "SupplierMention",
    "TenKFinancials",
    "TenKOutput",
    "TranscriptOutput",
]
