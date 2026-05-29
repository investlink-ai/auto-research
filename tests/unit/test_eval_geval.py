from __future__ import annotations

import pytest

from auto_research.eval.geval import SUBJECTIVE_GEVAL_FIELDS, build_geval_metric


def test_registry_lists_three_subjective_fields() -> None:
    assert set(SUBJECTIVE_GEVAL_FIELDS) == {
        "guidance_tone",
        "prepared_remarks_tone",
        "q_and_a_evasiveness",
    }


def test_build_metric_has_name_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Anthropic judge model validates ANTHROPIC_API_KEY at construction
    # (no network call). Set a dummy value so this construction-only test is
    # hermetic and passes in CI, where no real key is configured.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-for-construction-only")
    m = build_geval_metric("guidance_tone", threshold=0.7)
    assert "guidance_tone" in m.name
    assert m.threshold == 0.7
