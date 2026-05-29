"""Citation-grounding validator + quarantine routing (Issue #9, INV-2).

Two responsibilities under test:

1. `validate_citation_grounding(output, source_text)` walks the model tree
   and raises `CitationMismatch` (typed; subclasses `ValueError`) on the
   first `Citation` whose `source_text[span] != source_quote`. Crucially,
   the walker reaches `Citation`s reached via `Claim` *and* `Citation`s
   nested directly in subordinate models like `SupplierMention` —
   skipping either route would silently weaken INV-2.

2. `validate_or_quarantine(output, source_text, *, doc_id, worker, ...)`
   is the production routing helper: on success returns the output, on
   `CitationMismatch` writes a `QuarantineRecord` to
   `data/quarantine/<worker>/<doc_id>.json` and returns `None`. The
   skill explicitly forbids any `try/except → log → return output` path
   — this file is the mechanical proof that none exists.

A Hypothesis property test feeds randomized corrupted citations and
asserts both the raise and the quarantine write. Per the
`citation-check` skill's mandate ("Property test exists for the worker
you touched").
"""

from __future__ import annotations

import inspect
import json
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from auto_research.extract import guardrails
from auto_research.extract.enums import EventClassification
from auto_research.extract.guardrails import (
    CitationMismatch,
    QuarantineRecord,
    validate_citation_grounding,
    validate_or_quarantine,
)
from auto_research.extract.schemas import (
    Citation,
    Claim,
    EightKOutput,
    SupplierMention,
    TenKOutput,
)

# --- helpers ----------------------------------------------------------------


def _claim_for(
    start: int, end: int, source_text: str, *, confidence: str = "medium"
) -> Claim:
    return Claim(
        citation=Citation(source_span=(start, end), source_quote=source_text[start:end]),
        confidence=confidence,  # type: ignore[arg-type]
    )


def _minimal_eight_k(claim: Claim) -> EightKOutput:
    return EightKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000002",
        event_classification=EventClassification.MILESTONE,
        milestone_mentions=[claim],
        dilution_language_flags=[],
    )


def _minimal_ten_k_with_supplier(supplier: SupplierMention) -> TenKOutput:
    """Builds a TenKOutput whose only citation reaches via `SupplierMention.citation`
    (i.e., a direct `Citation`, NOT inside a `Claim`). Used to prove the walker
    finds both routes.
    """
    placeholder_quote = "x"  # one valid char; this claim isn't being validated against any source.
    placeholder = Claim(
        citation=Citation(source_span=(0, 1), source_quote=placeholder_quote),
        confidence="medium",
    )
    return TenKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        guidance_tone=placeholder,
        accrual_flags=[],
        supplier_mentions=[supplier],
        customer_mentions=[],
        language_novelty_score=0.0,
        risk_factor_deltas=[],
        going_concern=None,
        icfr_material_weaknesses=[],
        critical_accounting_estimate_changes=[],
    )


# --- validate_citation_grounding -------------------------------------------


def test_validator_passes_for_aligned_citation() -> None:
    source = "Acme Corp grew revenue 30% YoY in Q3."
    output = _minimal_eight_k(_claim_for(0, 9, source))  # "Acme Corp"
    validate_citation_grounding(output, source)  # must not raise


def test_validator_passes_for_output_with_no_citations() -> None:
    """An output whose lists are all empty has nothing to ground — the
    walker yields zero citations and the validator returns. Edge case
    is worth pinning: a regression where the walker spuriously raised
    on absent citations would silently quarantine every empty extraction.
    """
    output = EightKOutput(
        cik="0001045810",
        accession_number="acc-empty",
        event_classification=EventClassification.OTHER,
        milestone_mentions=[],
        dilution_language_flags=[],
    )
    validate_citation_grounding(output, "any text, doesn't matter")


def test_validator_raises_citation_mismatch_for_wrong_quote() -> None:
    source = "Acme Corp grew revenue 30% YoY in Q3."
    bad_claim = Claim(
        citation=Citation(source_span=(0, 9), source_quote="Different"),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    with pytest.raises(CitationMismatch):
        validate_citation_grounding(output, source)


def test_citation_mismatch_subclasses_value_error() -> None:
    source = "Acme Corp"
    bad_claim = Claim(
        citation=Citation(source_span=(0, 4), source_quote="XXXX"),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    with pytest.raises(ValueError) as exc_info:
        validate_citation_grounding(output, source)
    assert isinstance(exc_info.value, CitationMismatch)


def test_validator_walks_direct_citations_not_just_claims() -> None:
    """Supplier/customer mentions carry `Citation` *directly* (not inside a
    `Claim`). The walker MUST visit those — otherwise an LLM could
    hallucinate a `mention_text` with a citation that doesn't match the
    source text and the validator would pass.
    """
    source = "TSMC is a key supplier."
    bad_supplier = SupplierMention(
        mention_text="TSMC",
        citation=Citation(source_span=(0, 4), source_quote="HALLUCINATED"),
        resolved_ticker=None,
        resolver_confidence=None,
        resolver_reasoning=None,
    )
    output = _minimal_ten_k_with_supplier(bad_supplier)
    with pytest.raises(CitationMismatch):
        validate_citation_grounding(output, source)


def test_validator_walks_lists_of_claims() -> None:
    """The first BAD claim in a list trips the validator; clean prefix doesn't
    save the corrupted tail. Order independence is the property under test.
    """
    source = "Revenue rose. Margins fell."
    good = _claim_for(0, 13, source)  # "Revenue rose."
    bad = Claim(
        citation=Citation(source_span=(14, 26), source_quote="HALLUCINATED"),
        confidence="medium",
    )
    output = _minimal_eight_k(good)
    # Manually rebuild with the corrupt claim appended.
    output = EightKOutput(
        cik=output.cik,
        accession_number=output.accession_number,
        event_classification=output.event_classification,
        milestone_mentions=[good, bad],
        dilution_language_flags=[],
    )
    with pytest.raises(CitationMismatch):
        validate_citation_grounding(output, source)


def test_validator_mismatch_message_includes_field_path() -> None:
    source = "Some text."
    bad_claim = Claim(
        citation=Citation(source_span=(0, 4), source_quote="NOPE"),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    with pytest.raises(CitationMismatch) as exc_info:
        validate_citation_grounding(output, source)
    msg = str(exc_info.value)
    # The location helps an investigator find which field hallucinated.
    assert "milestone_mentions" in msg
    assert "NOPE" in msg


# --- validate_or_quarantine -------------------------------------------------


def test_quarantine_returns_output_on_success(tmp_path: Path) -> None:
    source = "Acme Corp grew revenue."
    output = _minimal_eight_k(_claim_for(0, 9, source))
    result = validate_or_quarantine(
        output,
        source,
        doc_id="acc-001",
        worker="eight_k",
        prompt_version="v1",
        quarantine_root=tmp_path,
    )
    assert result is output
    # Nothing was written to quarantine on the success path.
    assert not any(tmp_path.rglob("*"))


def test_quarantine_writes_record_and_returns_none_on_mismatch(tmp_path: Path) -> None:
    source = "Real source text."
    bad_claim = Claim(
        citation=Citation(source_span=(0, 4), source_quote="FAKE"),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    result = validate_or_quarantine(
        output,
        source,
        doc_id="acc-002",
        worker="eight_k",
        prompt_version="v1",
        quarantine_root=tmp_path,
    )
    assert result is None
    qpath = tmp_path / "eight_k" / "acc-002.json"
    assert qpath.exists()
    # The record is parseable, lossless on output, and carries the error message.
    record = json.loads(qpath.read_text())
    assert record["doc_id"] == "acc-002"
    assert record["worker"] == "eight_k"
    assert record["prompt_version"] == "v1"
    assert record["output"]["cik"] == "0001045810"
    assert "FAKE" in record["error"]


def test_quarantine_uses_original_output_when_supplied(tmp_path: Path) -> None:
    """When the worker passes `original_output=parsed_snapshot`, the
    QuarantineRecord captures what the model returned — not the
    worker-rewritten Pydantic dump. This preserves the audit invariant
    'every quarantine path captures what the model actually said' even
    when _resolve_spans snapped source_quote to a raw substring."""
    source = "Real source text."
    bad_claim = Claim(
        citation=Citation(source_span=(0, 4), source_quote="FAKE"),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    original = {
        "milestone_mentions": [
            {"citation": {"source_quote": "model's original quote"}, "confidence": "medium"}
        ],
        "marker": "from-the-model",
    }
    result = validate_or_quarantine(
        output,
        source,
        doc_id="acc-orig",
        worker="eight_k",
        prompt_version="v1",
        quarantine_root=tmp_path,
        original_output=original,
    )
    assert result is None
    record = json.loads((tmp_path / "eight_k" / "acc-orig.json").read_text())
    assert record["output"] == original
    assert record["output"]["marker"] == "from-the-model"


def test_quarantine_record_is_frozen() -> None:
    record = QuarantineRecord(
        doc_id="x",
        worker="eight_k",
        prompt_version="v1",
        output={"field": "value"},
        error="boom",
    )
    with pytest.raises(ValidationError):
        record.doc_id = "y"


def test_no_disabling_flags_on_public_api() -> None:
    """INV-2 has no escape hatch. A `permissive` / `soft_mode` /
    `skip_validation` / `disable_guardrails` kwarg on any public function
    in `guardrails` would be one — fail the suite the moment one appears.
    """
    forbidden = {
        "permissive",
        "soft_mode",
        "skip_validation",
        "disable_guardrails",
        "lenient",
    }
    for name in dir(guardrails):
        if name.startswith("_"):
            continue
        obj = getattr(guardrails, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        offenders = forbidden & set(sig.parameters)
        assert not offenders, f"{name} exposes forbidden kwarg(s): {offenders}"


# --- Property test (citation-check skill mandate) ---------------------------
#
# `@given` strategy yields a (source_text, span_start, span_end, quote) tuple
# where the quote is GUARANTEED to mismatch the slice. The append of a single
# zero-width-space (U+200B) makes the corrupted quote one codepoint longer
# than the slice — they can never be equal, regardless of what Hypothesis
# generated. The downstream assertion is therefore deterministic: the
# validator must raise and the worker stub must quarantine.


@st.composite
def _corrupted_case(draw: st.DrawFn) -> tuple[str, int, int, str]:
    source = draw(st.text(min_size=2, max_size=200))
    span_start = draw(st.integers(min_value=0, max_value=max(0, len(source) - 1)))
    span_end = draw(st.integers(min_value=span_start + 1, max_value=len(source)))
    correct = source[span_start:span_end]
    corrupted_quote = correct + "​"  # zero-width-space; length differs ⇒ guaranteed mismatch.
    return source, span_start, span_end, corrupted_quote


@given(case=_corrupted_case())
def test_property_corrupted_citation_routes_to_quarantine(
    tmp_path_factory: pytest.TempPathFactory,
    case: tuple[str, int, int, str],
) -> None:
    source, start, end, bad_quote = case
    qroot = tmp_path_factory.mktemp("qroot")
    bad_claim = Claim(
        citation=Citation(source_span=(start, end), source_quote=bad_quote),
        confidence="medium",
    )
    output = _minimal_eight_k(bad_claim)
    result = validate_or_quarantine(
        output,
        source,
        doc_id="hyp-doc",
        worker="eight_k",
        prompt_version="vtest",
        quarantine_root=qroot,
    )
    assert result is None
    assert (qroot / "eight_k" / "hyp-doc.json").exists()


@st.composite
def _clean_case(draw: st.DrawFn) -> tuple[str, int, int]:
    source = draw(st.text(min_size=2, max_size=200))
    span_start = draw(st.integers(min_value=0, max_value=max(0, len(source) - 1)))
    span_end = draw(st.integers(min_value=span_start + 1, max_value=len(source)))
    return source, span_start, span_end


@given(case=_clean_case())
def test_property_clean_citation_always_validates(
    tmp_path_factory: pytest.TempPathFactory,
    case: tuple[str, int, int],
) -> None:
    """Dual to the corrupted-case property: any quote that IS `source[span]`
    must pass. Catches a regression where the walker overshoots and
    spuriously raises on aligned claims (false positive).
    """
    source, start, end = case
    qroot = tmp_path_factory.mktemp("qclean")
    output = _minimal_eight_k(_claim_for(start, end, source))
    result = validate_or_quarantine(
        output,
        source,
        doc_id="clean-doc",
        worker="eight_k",
        prompt_version="vtest",
        quarantine_root=qroot,
    )
    assert result is output
    assert not any(qroot.rglob("*"))
