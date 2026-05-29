from __future__ import annotations

from datetime import date

from auto_research.eval.baseline import score_output
from auto_research.eval.gold import GoldSample
from auto_research.eval.registry import WORKER_EVALS
from auto_research.extract.enums import EventClassification, FormType
from auto_research.extract.schemas import (
    Citation,
    Claim,
    EightKOutput,
    SFilingOutput,
    TenKOutput,
)


def _claim(quote: str, span: tuple[int, int], conf: str = "high") -> Claim:
    return Claim(citation=Citation(source_span=span, source_quote=quote), confidence=conf)


def test_score_output_eight_k_matches_gold() -> None:
    src = "We announced a partnership with Acme Corp today."
    # "partnership with Acme Corp" starts at index 15, ends at 41
    quote = "partnership with Acme Corp"
    start = src.index(quote)
    end = start + len(quote)
    predicted = EightKOutput(
        cik="1",
        accession_number="a",
        event_classification=EventClassification.PARTNERSHIP,
        milestone_mentions=[_claim(quote, (start, end))],
        dilution_language_flags=[],
    )
    gold = GoldSample(
        doc_id="g-001",
        raw_doc=src,
        expected={
            "event_classification": "partnership",
            "milestone_mentions": ["partnership with Acme Corp"],
            "dilution_language_flags": [],
        },
        rationale="r",
    )
    scores = score_output(WORKER_EVALS["eight_k"], predicted, gold)
    assert scores["event_classification"] == 1.0
    assert scores["milestone_mentions"] == 1.0
    assert scores["dilution_language_flags"] == 1.0
    assert scores["_grounding"] == "grounded"


def test_score_output_ten_k_going_concern_none() -> None:
    src = "The company has no substantial doubt about its ability to continue as a going concern."
    # guidance_tone grounded in raw_doc
    gt_quote = "no substantial doubt about its ability to continue as a going concern"
    gt_start = src.index(gt_quote)
    gt_end = gt_start + len(gt_quote)

    predicted = TenKOutput(
        cik="2",
        accession_number="b",
        fiscal_period_end=date(2024, 12, 31),
        guidance_tone=_claim(gt_quote, (gt_start, gt_end)),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        risk_factor_deltas=[],
        going_concern=None,
        icfr_material_weaknesses=[],
        critical_accounting_estimate_changes=[],
    )
    gold = GoldSample(
        doc_id="g-002",
        raw_doc=src,
        expected={
            "accrual_flags": [],
            "supplier_mentions": [],
            "customer_mentions": [],
            "risk_factor_deltas": [],
            "icfr_material_weaknesses": [],
            "critical_accounting_estimate_changes": [],
            "going_concern": [],
        },
        rationale="r",
    )
    scores = score_output(WORKER_EVALS["ten_k"], predicted, gold)
    assert scores["going_concern"] == 1.0
    assert scores["accrual_flags"] == 1.0
    assert scores["_grounding"] == "grounded"


def test_score_output_s_filings_mandatory_dilution_event() -> None:
    src = "This S-3 filing relates to the resale of shares issued in a dilutive transaction."
    de_quote = "resale of shares issued in a dilutive transaction"
    de_start = src.index(de_quote)
    de_end = de_start + len(de_quote)

    predicted = SFilingOutput(
        cik="3",
        accession_number="c",
        form_type=FormType.S_3,
        dilution_event=_claim(de_quote, (de_start, de_end)),
        capital_raise_language=[],
        use_of_proceeds=[],
    )
    gold = GoldSample(
        doc_id="g-003",
        raw_doc=src,
        expected={
            "form_type": "S-3",
            "dilution_event": [de_quote],
            "capital_raise_language": [],
            "use_of_proceeds": [],
        },
        rationale="r",
    )
    scores = score_output(WORKER_EVALS["s_filings"], predicted, gold)
    assert scores["form_type"] == 1.0
    assert scores["dilution_event"] == 1.0
    assert scores["capital_raise_language"] == 1.0
    assert scores["use_of_proceeds"] == 1.0
    assert scores["_grounding"] == "grounded"
