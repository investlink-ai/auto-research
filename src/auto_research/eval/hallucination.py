"""Citation-grounding outcome for the eval harness (INV-2).

Reuses the production walker so 'grounded' means exactly what it means in
extraction. A non-None worker output is grounded by construction; the
meaningful eval signal is how often the worker *would* have been
quarantined over the gold set (the hallucination rate)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from auto_research.extract.guardrails import CitationMismatch, validate_citation_grounding

GroundingOutcome = Literal["grounded", "ungrounded"]


def grounding_outcome(output: BaseModel, source_text: str) -> GroundingOutcome:
    try:
        validate_citation_grounding(output, source_text)
    except CitationMismatch:
        return "ungrounded"
    return "grounded"
