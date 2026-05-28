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


# Categorical confidence label used on every LLM-emitted confidence
# field. Float confidence is uncalibrated noise; the categorical form
# lets a downstream consumer threshold cleanly. Reuse this alias on
# any new claim-bearing schema so the three values stay aligned across
# the codebase — a `Literal['high','med','low']` typo in a new schema
# would diverge silently if the alias weren't shared.
ConfidenceLevel = Literal["high", "medium", "low"]


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
    """

    model_config = _FROZEN_STRICT

    citation: Citation
    confidence: ConfidenceLevel


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
    # Narrative-only signals XBRL definitionally cannot give. Both
    # narrative paths (single-shot via TEN_K_NARRATIVE_PROMPT, RAG via
    # the per-field config loop) populate these. Spec:
    # docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md.
    going_concern: Claim | None
    icfr_material_weaknesses: list[Claim]
    critical_accounting_estimate_changes: list[Claim]


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


# --- Per-field 10-K narrative partials --------------------------------------
#
# The 10-K RAG path runs one Anthropic call per narrative field so each
# call uses the model tier the routing table actually declares (3 of 5
# narrative fields are Haiku-tier; the unified pre-split call routed
# everything to Sonnet). Each partial carries the identity fields plus
# exactly ONE narrative field — the worker assembles them into a full
# `TenKOutput` at the end of the loop, with a cross-partial identity
# check to catch hallucinated cik / accession_number / fiscal_period_end
# drift between calls.
#
# Each partial has its own `SCHEMA_VERSION` so a future iteration of one
# field's contract (e.g., adding a sub-mention type) invalidates only
# its own cache entries; the orthogonality matches the chunker /
# embed-model version-pinning pattern already in the codebase.


class TenKGuidanceTonePartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    guidance_tone: Claim


class TenKAccrualFlagsPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    accrual_flags: list[Claim]


class TenKSupplierMentionsPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    supplier_mentions: list[SupplierMention]


class TenKCustomerMentionsPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    customer_mentions: list[CustomerMention]


class TenKRiskFactorDeltasPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    risk_factor_deltas: list[RiskFactorDelta]


class TenKGoingConcernPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    going_concern: Claim | None


# --- Transcript partials ----------------------------------------------------
#
# Binary split by routing tier (spec §7.3). Prepared remarks is
# templated/templated tone classification ⇒ Haiku. Q&A evasiveness and
# the forward_statements list both need cross-utterance reasoning ⇒
# Sonnet, so they share one Sonnet call rather than splitting into a
# third. Eliminates the "one Sonnet call for everything" waste on the
# prepared-remarks half.


class TranscriptPreparedRemarksPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    ticker: str
    event_datetime: AwareDatetime | None
    prepared_remarks_tone: Claim


class TranscriptQAPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    ticker: str
    event_datetime: AwareDatetime | None
    q_and_a_evasiveness: Claim
    forward_statements: list[ForwardStatement]


__all__ = [
    "Citation",
    "Claim",
    "ConfidenceLevel",
    "CustomerMention",
    "EightKOutput",
    "ForwardStatement",
    "RiskFactorDelta",
    "SFilingOutput",
    "SupplierMention",
    "TenKAccrualFlagsPartial",
    "TenKCustomerMentionsPartial",
    "TenKGoingConcernPartial",
    "TenKGuidanceTonePartial",
    "TenKOutput",
    "TenKRiskFactorDeltasPartial",
    "TenKSupplierMentionsPartial",
    "TranscriptOutput",
    "TranscriptPreparedRemarksPartial",
    "TranscriptQAPartial",
]
