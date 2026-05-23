"""MLflow file-backend wrapper.

Thin context-manager API for the experiment tracker used by `backtest/`
and `agents/alpha_library.py`. The file backend at `mlruns/` (configured
via `MLFLOW_TRACKING_URI` in `.env`) stays git-ignored; the run history
is local-only by design — we don't want backtest noise in version
control.

Typical use:

    from auto_research.experiment import start_run
    import mlflow

    with start_run(experiment="backtest", run_name="A2_v1") as run:
        mlflow.log_param("signal_id", "A2")
        mlflow.log_metric("sharpe_net", 0.83)
        mlflow.log_artifact("report.json")

Inspect with:

    uv run mlflow ui   # opens http://localhost:5000
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.entities import Run

_DEFAULT_URI = "file:./mlruns"


@contextmanager
def start_run(
    *,
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Run]:
    """Open a tracked MLflow run.

    Sets the tracking URI from `MLFLOW_TRACKING_URI` (defaults to
    `file:./mlruns`), selects or creates the named experiment, then
    yields the `Run` for the caller to enrich with `mlflow.log_param`,
    `mlflow.log_metric`, `mlflow.log_artifact`, etc.

    The run is auto-closed on exit (success or exception). On exception,
    the run status is set to FAILED — same as `mlflow.start_run`'s
    default contextmanager behavior.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI).strip() or _DEFAULT_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def configured_tracking_uri() -> str:
    """Read-only accessor — handy for diagnostics and tests."""
    raw: Any = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI)
    return str(raw).strip() or _DEFAULT_URI
