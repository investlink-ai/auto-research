from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_research.eval.gold import GoldSample, GoldSet, load_gold_set


def test_gold_sample_roundtrips_minimal() -> None:
    s = GoldSample(
        doc_id="g-001",
        raw_doc="We expect revenue to decline next quarter.",
        expected={"cik": "0000320193"},
        subjective={"guidance_tone": {"confidence": "high", "rubric_note": "clearly negative"}},
        rationale="explicit negative guidance",
    )
    assert s.doc_id == "g-001"
    assert s.expected["cik"] == "0000320193"


def test_gold_set_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        GoldSet(worker="ten_k", thresholds={"min_f1": 0.6}, samples=(), bogus=1)  # type: ignore[call-arg]


def test_load_gold_set_parses_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "ten_k.jsonl"
    p.write_text(
        '{"doc_id":"g-001","raw_doc":"x","expected":{"cik":"1"},"subjective":{},"rationale":"r"}\n'
        '{"doc_id":"g-002","raw_doc":"y","expected":{"cik":"2"},"subjective":{},"rationale":"r"}\n'
    )
    gs = load_gold_set(p, worker="ten_k", thresholds={"min_f1": 0.6})
    assert gs.worker == "ten_k"
    assert len(gs.samples) == 2
    assert gs.samples[0].doc_id == "g-001"
