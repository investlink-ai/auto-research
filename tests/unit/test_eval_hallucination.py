from __future__ import annotations

from auto_research.eval.hallucination import grounding_outcome
from auto_research.extract.schemas import Citation, Claim, EightKOutput


def _output(quote: str, span: tuple[int, int]) -> EightKOutput:
    return EightKOutput(
        cik="1",
        accession_number="a",
        event_classification="partnership",
        milestone_mentions=[
            Claim(citation=Citation(source_span=span, source_quote=quote), confidence="high")
        ],
        dilution_language_flags=[],
    )


def test_grounded_output() -> None:
    src = "We announced a partnership with Acme Corp today."
    # src[15:41] == "partnership with Acme Corp" (verified via src.index())
    out = _output("partnership with Acme Corp", (15, 41))
    assert grounding_outcome(out, src) == "grounded"


def test_ungrounded_output() -> None:
    src = "We announced a partnership with Acme Corp today."
    out = _output("a totally fabricated quote", (0, 26))
    assert grounding_outcome(out, src) == "ungrounded"
