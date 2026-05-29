from __future__ import annotations

from auto_research.eval.geval import SUBJECTIVE_GEVAL_FIELDS, build_geval_metric


def test_registry_lists_three_subjective_fields() -> None:
    assert set(SUBJECTIVE_GEVAL_FIELDS) == {
        "guidance_tone",
        "prepared_remarks_tone",
        "q_and_a_evasiveness",
    }


def test_build_metric_has_name_and_threshold() -> None:
    m = build_geval_metric("guidance_tone", threshold=0.7)
    assert "guidance_tone" in m.name
    assert m.threshold == 0.7
