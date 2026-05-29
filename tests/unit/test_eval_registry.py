from __future__ import annotations

import typing

import pytest

from auto_research.eval.registry import WORKER_EVALS

_WORKERS = ["ten_k", "transcript", "eight_k", "s_filings"]


@pytest.mark.parametrize("worker", _WORKERS)
def test_every_schema_field_has_a_metric(worker: str) -> None:
    spec = WORKER_EVALS[worker]
    schema_fields = set(spec.output_model.model_fields)
    covered = set(spec.field_metrics) | set(spec.identity_fields) | set(spec.subjective_fields)
    missing = schema_fields - covered
    assert not missing, f"{worker}: fields with no eval metric: {missing}"


@pytest.mark.parametrize("worker", _WORKERS)
def test_metric_kind_matches_field_type(worker: str) -> None:
    """Coverage alone lets a `list[Claim]` field be mistyped as `exact`,
    which would silently score 0.0 forever. Assert the assigned kind is
    consistent with the field's actual annotation: only list-typed fields
    may be `claim_list`, and `numeric` must annotate a float.
    """
    spec = WORKER_EVALS[worker]
    for fname, kind in spec.field_metrics.items():
        ann = spec.output_model.model_fields[fname].annotation
        is_list = typing.get_origin(ann) is list
        assert is_list == (kind == "claim_list"), (
            f"{worker}.{fname}: kind={kind!r} but the field "
            f"{'is' if is_list else 'is not'} a list type ({ann!r})"
        )
        if kind == "numeric":
            assert ann is float or float in typing.get_args(ann), (
                f"{worker}.{fname}: kind 'numeric' but annotation {ann!r} is not a float"
            )


def test_prompt_version_is_resolved_string() -> None:
    for spec in WORKER_EVALS.values():
        assert isinstance(spec.prompt_version, str) and spec.prompt_version
