"""Closed-set value taxonomies for extracted fields.

Separated from `schemas.py` so downstream consumers (signal code, eval
suites, the live critic) can import the enums without pulling in the
heavy output models. `StrEnum` (Python 3.11+) gives:

- importable, IDE-discoverable namespaces
- string-equality round-trips (`out.event_classification == "milestone"`
  keeps working) so LLM-JSON interop is unchanged
- `model_dump(mode="json")` emits plain strings, not enum reprs — the
  wire format stays stable

Adding a member to any enum here is non-breaking on the producer side
(LLM workers will emit it as soon as the prompt covers it) but
**requires a `prompt_version` bump** for the worker that introduces it
(INV-6) so eval baselines stay valid. Removing or renaming a member is
a breaking change to `data/extracted/` and requires a Feast schema
migration.
"""

from __future__ import annotations

from enum import StrEnum


class EventClassification(StrEnum):
    """8-K event categories the worker classifies into.

    `OTHER` is the catch-all; never use it for an event the model is just
    uncertain about (uncertainty lives in the `Claim.confidence` of the
    accompanying mentions, not in the classification label).
    """

    MILESTONE = "milestone"
    PARTNERSHIP = "partnership"
    CONTRACT = "contract"
    GUIDANCE_CHANGE = "guidance_change"
    LEADERSHIP_CHANGE = "leadership_change"
    DILUTION = "dilution"
    OTHER = "other"


class FormType(StrEnum):
    """S-filing form types the worker accepts."""

    S_1 = "S-1"
    S_3 = "S-3"


class RiskFactorChangeType(StrEnum):
    """Direction of a 10-K Item 1A risk-factor change vs the prior year filing."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


__all__ = [
    "EventClassification",
    "FormType",
    "RiskFactorChangeType",
]
