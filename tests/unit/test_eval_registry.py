from __future__ import annotations

import pytest

from auto_research.eval.registry import WORKER_EVALS


@pytest.mark.parametrize("worker", ["ten_k", "transcript", "eight_k", "s_filings"])
def test_every_schema_field_has_a_metric(worker: str) -> None:
    spec = WORKER_EVALS[worker]
    schema_fields = set(spec.output_model.model_fields)
    covered = set(spec.field_metrics) | set(spec.identity_fields) | set(spec.subjective_fields)
    missing = schema_fields - covered
    assert not missing, f"{worker}: fields with no eval metric: {missing}"


def test_prompt_version_is_resolved_string() -> None:
    for spec in WORKER_EVALS.values():
        assert isinstance(spec.prompt_version, str) and spec.prompt_version
