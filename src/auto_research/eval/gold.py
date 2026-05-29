"""Typed loader for per-worker gold-set JSONL files at eval/gold_sets/.

One JSON object per line. `expected` holds worker-specific expected field
values; `subjective` holds per-field G-Eval rubric notes keyed by the
subjective `Claim` field name. Pydantic validates shape at load time so a
drifted key surfaces as a clear ValidationError, not a buried KeyError —
the pattern already used by tests/evals/test_entity_resolution_eval.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")


class GoldSample(BaseModel):
    model_config = _FROZEN_STRICT

    doc_id: str
    raw_doc: str
    expected: dict[str, Any]
    subjective: dict[str, dict[str, str]] = {}
    rationale: str = ""


class GoldSet(BaseModel):
    model_config = _FROZEN_STRICT

    worker: str
    thresholds: dict[str, float]
    samples: tuple[GoldSample, ...]


def load_gold_set(
    path: Path, *, worker: str, thresholds: dict[str, float]
) -> GoldSet:
    """Parse a `.jsonl` gold file into a validated `GoldSet`."""
    samples = tuple(
        GoldSample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    return GoldSet(worker=worker, thresholds=thresholds, samples=samples)
