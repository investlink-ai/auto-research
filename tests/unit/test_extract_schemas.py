"""Schema correctness for `auto_research.extract.schemas` (Issue #9).

INV-2 depends on these models being immutable after construction and on the
`Citation` field validators rejecting obviously bad data at the boundary
— a Pydantic `ValidationError` here means the LLM output never reaches
the citation-grounding validator, which is the right failure mode.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from auto_research.extract.enums import (
    EventClassification,
    FormType,
    RiskFactorChangeType,
)
from auto_research.extract.schemas import (
    Citation,
    Claim,
    CustomerMention,
    EightKOutput,
    ForwardStatement,
    RiskFactorDelta,
    SFilingOutput,
    SupplierMention,
    TenKOutput,
    TranscriptOutput,
)

# --- helpers ----------------------------------------------------------------


def _citation(start: int = 0, end: int = 4, quote: str = "hello") -> Citation:
    return Citation(source_span=(start, end), source_quote=quote)


def _claim(confidence: str = "medium") -> Claim:
    return Claim(citation=_citation(), confidence=confidence)  # type: ignore[arg-type]


# --- Citation ---------------------------------------------------------------


def test_citation_accepts_valid_span_and_quote() -> None:
    c = Citation(source_span=(5, 12), source_quote="lorem  ")
    assert c.source_span == (5, 12)
    assert c.source_quote == "lorem  "


def test_citation_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        Citation(source_span=(-1, 5), source_quote="x")


def test_citation_rejects_negative_end() -> None:
    with pytest.raises(ValidationError):
        Citation(source_span=(0, -1), source_quote="x")


def test_citation_rejects_start_equal_to_end() -> None:
    # An empty span ⇔ an empty quote: that's not a citation. Reject at
    # construction so the walker never has to second-guess.
    with pytest.raises(ValidationError):
        Citation(source_span=(5, 5), source_quote="")


def test_citation_rejects_start_greater_than_end() -> None:
    with pytest.raises(ValidationError):
        Citation(source_span=(10, 5), source_quote="x")


def test_citation_rejects_empty_quote() -> None:
    with pytest.raises(ValidationError):
        Citation(source_span=(0, 5), source_quote="")


def test_citation_is_frozen() -> None:
    c = _citation()
    with pytest.raises(ValidationError):
        c.source_quote = "other"


def test_citation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Citation(source_span=(0, 4), source_quote="hi", note="extra")  # type: ignore[call-arg]


# --- Claim ------------------------------------------------------------------


def test_claim_accepts_categorical_confidence() -> None:
    for level in ("high", "medium", "low"):
        c = Claim(citation=_citation(), confidence=level)
        assert c.confidence == level


def test_claim_rejects_float_confidence() -> None:
    """Float `confidence` must reject — categorical-confidence policy
    (user feedback memory: `LLM confidence is categorical`). Float
    confidence is uncalibrated noise."""
    with pytest.raises(ValidationError):
        Claim(citation=_citation(), confidence=0.7)  # type: ignore[arg-type]


def test_claim_rejects_unknown_categorical_label() -> None:
    with pytest.raises(ValidationError):
        Claim(citation=_citation(), confidence="maybe")  # type: ignore[arg-type]


def test_claim_is_frozen() -> None:
    c = _claim()
    with pytest.raises(ValidationError):
        c.confidence = "high"


# --- Output models — frozen + extra=forbid ----------------------------------
#
# One representative-output test per worker, covering both frozen and
# extra-forbidden — the AC names "all output models frozen; mutation raises."


def _ten_k_output() -> TenKOutput:
    return TenKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        guidance_tone=_claim(),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        language_novelty_score=0.0,
        risk_factor_deltas=[],
    )


def _transcript_output() -> TranscriptOutput:
    return TranscriptOutput(
        ticker="NVDA",
        event_datetime=datetime(2025, 2, 26, 17, 0, tzinfo=UTC),
        prepared_remarks_tone=_claim(),
        q_and_a_evasiveness=_claim(),
        forward_statements=[],
    )


def _eight_k_output() -> EightKOutput:
    return EightKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000002",
        event_classification=EventClassification.MILESTONE,
        milestone_mentions=[],
        dilution_language_flags=[],
    )


def _s_filing_output() -> SFilingOutput:
    return SFilingOutput(
        cik="0001045810",
        accession_number="0001045810-25-000003",
        form_type=FormType.S_3,
        dilution_event=_claim(),
        capital_raise_language=[],
        use_of_proceeds=[],
    )


def test_ten_k_output_is_frozen() -> None:
    out = _ten_k_output()
    with pytest.raises(ValidationError):
        out.cik = "9999"


def test_ten_k_output_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TenKOutput(
            cik="x",
            accession_number="y",
            fiscal_period_end=date(2025, 1, 1),
            guidance_tone=_claim(),
            accrual_flags=[],
            supplier_mentions=[],
            customer_mentions=[],
            language_novelty_score=0.0,
            risk_factor_deltas=[],
            unknown_field=1,  # type: ignore[call-arg]
        )


def test_transcript_output_is_frozen() -> None:
    out = _transcript_output()
    with pytest.raises(ValidationError):
        out.ticker = "OTHER"


def test_transcript_event_datetime_rejects_naive() -> None:
    """Naive datetime carries no semantics — schema must reject so the
    LLM can't silently substitute a guessed timezone (the prompt now
    instructs `null` instead of inventing an HQ offset)."""
    with pytest.raises(ValidationError):
        TranscriptOutput(
            ticker="NVDA",
            event_datetime=datetime(2025, 2, 26, 17, 0),  # naive
            prepared_remarks_tone=_claim(),
            q_and_a_evasiveness=_claim(),
            forward_statements=[],
        )


def test_transcript_event_datetime_accepts_null() -> None:
    """When the transcript lacks an explicit time + timezone, the prompt
    tells the model to emit null; the schema must accept it."""
    out = TranscriptOutput(
        ticker="NVDA",
        event_datetime=None,
        prepared_remarks_tone=_claim(),
        q_and_a_evasiveness=_claim(),
        forward_statements=[],
    )
    assert out.event_datetime is None


def test_transcript_event_datetime_accepts_iso_with_offset() -> None:
    """ISO-8601 with explicit offset (the prompt's mandated format) must
    round-trip cleanly through Pydantic parsing."""
    out = TranscriptOutput.model_validate(
        {
            "ticker": "NVDA",
            "event_datetime": "2026-01-30T17:00:00-05:00",
            "prepared_remarks_tone": _claim().model_dump(),
            "q_and_a_evasiveness": _claim().model_dump(),
            "forward_statements": [],
        }
    )
    assert out.event_datetime is not None
    assert out.event_datetime.tzinfo is not None


def test_eight_k_output_is_frozen() -> None:
    out = _eight_k_output()
    with pytest.raises(ValidationError):
        out.event_classification = EventClassification.OTHER


def test_eight_k_output_rejects_unknown_event_classification() -> None:
    with pytest.raises(ValidationError):
        EightKOutput(
            cik="x",
            accession_number="y",
            event_classification="bogus",  # type: ignore[arg-type]
            milestone_mentions=[],
            dilution_language_flags=[],
        )


def test_s_filing_output_is_frozen() -> None:
    out = _s_filing_output()
    with pytest.raises(ValidationError):
        out.form_type = FormType.S_1


def test_s_filing_output_rejects_unknown_form_type() -> None:
    with pytest.raises(ValidationError):
        SFilingOutput(
            cik="x",
            accession_number="y",
            form_type="10-K",  # type: ignore[arg-type]
            dilution_event=_claim(),
            capital_raise_language=[],
            use_of_proceeds=[],
        )


# --- Subordinate citation-bearing types -------------------------------------
#
# `SupplierMention` and `CustomerMention` carry a `Citation` directly (not a
# `Claim`) — the post-validator's walker must therefore find both routes.
# Covered functionally in `test_extract_guardrails.py`; here we just verify
# they're also frozen.


def test_supplier_mention_is_frozen() -> None:
    m = SupplierMention(
        mention_text="supplier X",
        citation=_citation(),
        resolved_ticker=None,
        resolver_confidence=None,
        resolver_reasoning=None,
    )
    with pytest.raises(ValidationError):
        m.mention_text = "other"


def test_customer_mention_is_frozen() -> None:
    m = CustomerMention(
        mention_text="customer Y",
        citation=_citation(),
        resolved_ticker=None,
        resolver_confidence=None,
        resolver_reasoning=None,
    )
    with pytest.raises(ValidationError):
        m.mention_text = "other"


def test_risk_factor_delta_is_frozen() -> None:
    rfd = RiskFactorDelta(
        change_type=RiskFactorChangeType.ADDED,
        text="new risk",
        citation=_citation(),
    )
    with pytest.raises(ValidationError):
        rfd.text = "other"


def test_forward_statement_is_frozen() -> None:
    fs = ForwardStatement(
        statement_text="will ship next quarter",
        citation=_citation(),
        mentioned_entities=(),
        horizon="next quarter",
    )
    with pytest.raises(ValidationError):
        fs.statement_text = "other"


# --- Enum round-trip + interop ---------------------------------------------
#
# `StrEnum` is chosen so callers can keep using string literals (the natural
# LLM-JSON interop) AND have an importable namespace + IDE auto-complete.
# These tests pin both behaviors so a future "should we drop StrEnum?"
# refactor breaks loudly instead of silently changing the wire format.


def test_event_classification_accepts_string_input() -> None:
    # Pydantic coerces the string to the enum member — the natural
    # LLM-JSON interop path. `# type: ignore[arg-type]` documents that
    # statically-typed callers should pass the enum member; the coercion
    # is for the JSON deserialization path only.
    out = EightKOutput(
        cik="0001045810",
        accession_number="acc-1",
        event_classification="milestone",  # type: ignore[arg-type]
        milestone_mentions=[],
        dilution_language_flags=[],
    )
    assert out.event_classification is EventClassification.MILESTONE
    # `.value` reads the underlying string explicitly. (StrEnum members ARE
    # strings, but mypy narrows `out.event_classification` to the enum's
    # singleton type and flags a string equality as non-overlapping.)
    assert out.event_classification.value == "milestone"


def test_event_classification_accepts_enum_member() -> None:
    out = EightKOutput(
        cik="0001045810",
        accession_number="acc-1",
        event_classification=EventClassification.DILUTION,
        milestone_mentions=[],
        dilution_language_flags=[],
    )
    assert out.event_classification is EventClassification.DILUTION


def test_event_classification_json_round_trip_is_plain_string() -> None:
    out = EightKOutput(
        cik="0001045810",
        accession_number="acc-1",
        event_classification=EventClassification.PARTNERSHIP,
        milestone_mentions=[],
        dilution_language_flags=[],
    )
    dumped = out.model_dump(mode="json")
    # The wire format is the string value, not "<EventClassification.PARTNERSHIP: ...>".
    assert dumped["event_classification"] == "partnership"


def test_form_type_round_trips_with_hyphenated_value() -> None:
    out = SFilingOutput(
        cik="0001045810",
        accession_number="acc-1",
        form_type=FormType.S_1,
        dilution_event=_claim(),
        capital_raise_language=[],
        use_of_proceeds=[],
    )
    assert out.form_type == "S-1"
    assert out.model_dump(mode="json")["form_type"] == "S-1"


def test_risk_factor_change_type_enum_accepted() -> None:
    rfd = RiskFactorDelta(
        change_type=RiskFactorChangeType.REMOVED,
        text="dropped supply-chain risk",
        citation=_citation(),
    )
    assert rfd.change_type == "removed"


# --- SCHEMA_VERSION co-versioning (Issue #11) -------------------------------


def test_every_output_model_carries_schema_version() -> None:
    """Every Pydantic output model exports `SCHEMA_VERSION` as a ClassVar.

    Cache key includes `schema_version`; if a model's field shape changes
    without bumping `SCHEMA_VERSION`, cached rows deserialize wrong on next
    read. See AGENTS.md INV-6.
    """
    for cls in (SFilingOutput, TenKOutput, TranscriptOutput, EightKOutput):
        assert hasattr(cls, "SCHEMA_VERSION"), f"{cls.__name__} missing SCHEMA_VERSION"
        assert isinstance(cls.SCHEMA_VERSION, str)
        assert cls.SCHEMA_VERSION.startswith("v")


def test_schema_version_is_classvar_not_pydantic_field() -> None:
    """`SCHEMA_VERSION` must NOT appear in `model_fields` — it's metadata,
    not data. If it leaks into the field set, dumps include it and
    downstream consumers (Feast, parquet) break."""
    for cls in (SFilingOutput, TenKOutput, TranscriptOutput, EightKOutput):
        assert "SCHEMA_VERSION" not in cls.model_fields


def test_ten_k_output_language_novelty_score_defaults_to_zero() -> None:
    """The narrative prompt instructs the model `DO NOT populate
    language_novelty_score (computed downstream)`. The schema must default
    the field so a prompt-compliant model that omits it does not trip
    Pydantic validation — otherwise every production 10-K extraction
    silently quarantines on the field-required error.
    """
    out = TenKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        guidance_tone=_claim(),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        risk_factor_deltas=[],
    )
    assert out.language_novelty_score == 0.0
