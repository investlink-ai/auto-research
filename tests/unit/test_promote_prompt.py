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


def test_compute_f1_empty_on_both_sides_is_perfect() -> None:
    """A gold sample with no expected fields (e.g., 'no novel dilution
    language') correctly extracted as empty deserves F1=1.0, not 0.0.
    Addresses code-review finding #10."""
    expected = {"dilution_event_quote": None, "use_of_proceeds_phrases": []}
    actual = {"dilution_event": None, "use_of_proceeds": []}
    assert compute_f1(expected, actual) == pytest.approx(1.0)


def test_promote_refuses_below_f1_threshold(tmp_path: Path) -> None:
    gold = {
        "prompt_name": "s_filings_dilution",
        "thresholds": {"min_f1": 0.9, "max_usd_per_doc": 1.0},
        "samples": [],  # no samples -> early-return refusal
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


def test_promote_handles_none_worker_output(tmp_path: Path) -> None:
    """Worker returning None (quarantine signal) must score that sample as
    0.0 and continue, not crash with AttributeError on
    `None.model_dump(...)`. Addresses code-review finding #5."""
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
    client = MagicMock()
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=lambda raw, doc_id: None,
        langfuse_client=client,
    )
    # No crash; sample scored 0.0; mean 0.0 < 0.5 -> refused
    assert result.promoted is False
    assert result.f1 == pytest.approx(0.0)
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
    client.get_prompt.return_value = MagicMock(version=5)
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=fake_worker,
        langfuse_client=client,
    )
    assert result.promoted is True
    # set_prompt_tag's lookup-by-label then update-by-int-version flow
    client.get_prompt.assert_called_once_with("s_filings_dilution", label="v1")
    client.update_prompt.assert_called_once_with(
        name="s_filings_dilution",
        version=5,
        new_labels=["production"],
    )


def test_main_uses_in_code_version_not_a_cli_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI must NOT accept a candidate version. promote_prompt always
    evaluates and promotes the SAME version (the in-code constant);
    accepting separate values invites the bug class where eval runs v1
    while the script labels v2. Addresses code-review finding #3."""
    from scripts.promote_prompt import main

    # Passing a second positional arg must fail at argparse.
    with pytest.raises(SystemExit) as exc:
        main(["s_filings_dilution", "v3"])
    assert exc.value.code == 2  # argparse usage error
