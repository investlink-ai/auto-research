"""Unit tests for `scripts/promote_prompt.py` (Issue #11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from scripts.promote_prompt import PromotionResult, compute_f1, promote


def test_compute_f1_exact_match() -> None:
    expected = {"use_of_proceeds_phrases": ["A", "B"]}
    actual = {
        "use_of_proceeds": [
            {"citation": {"source_quote": "A"}},
            {"citation": {"source_quote": "B"}},
        ]
    }
    assert compute_f1(expected, actual) == pytest.approx(1.0)


def test_compute_f1_partial_match() -> None:
    expected = {"use_of_proceeds_phrases": ["A", "B"]}
    actual = {
        "use_of_proceeds": [
            {"citation": {"source_quote": "A"}},
            {"citation": {"source_quote": "C"}},
        ]
    }
    # 1 TP, 1 FP, 1 FN -> P=0.5, R=0.5, F1=0.5
    assert compute_f1(expected, actual) == pytest.approx(0.5)


def test_promote_refuses_below_f1_threshold(tmp_path: Path) -> None:
    gold = {
        "prompt_name": "s_filings_dilution",
        "thresholds": {"min_f1": 0.9, "max_usd_per_doc": 1.0},
        "samples": [],  # no samples -> mean F1 = 0 -> fail
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))
    client = MagicMock()
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=lambda raw, doc_id: None,  # never called
        langfuse_client=client,
    )
    assert isinstance(result, PromotionResult)
    assert result.promoted is False
    assert "f1 threshold" in result.reason.lower()
    client.update_prompt.assert_not_called()


def test_promote_flips_tag_when_threshold_met(tmp_path: Path) -> None:
    sample = {
        "doc_id": "g1",
        "raw_doc": "x",
        "expected": {
            "form_type": "S-3",
            "dilution_event_quote": "x",
            "use_of_proceeds_phrases": [],
        },
    }
    gold = {
        "prompt_name": "s_filings_dilution",
        "thresholds": {"min_f1": 0.5, "max_usd_per_doc": 1.0},
        "samples": [sample],
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))

    def fake_worker(raw: str, doc_id: str) -> dict[str, Any]:
        return {
            "form_type": "S-3",
            "dilution_event": {"citation": {"source_quote": "x"}},
            "use_of_proceeds": [],
        }

    client = MagicMock()
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=fake_worker,
        langfuse_client=client,
    )
    assert result.promoted is True
    client.update_prompt.assert_called_once()
    assert "production" in client.update_prompt.call_args.kwargs["new_labels"]
