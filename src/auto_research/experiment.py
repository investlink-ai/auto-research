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
from pathlib import Path

import mlflow
from mlflow.entities import Run

_DEFAULT_URI = "file:./mlruns"


def _normalize_uri(raw: str) -> str:
    """Resolve relative `file:` URIs to absolute paths.

    Without this, `file:./mlruns` resolves against the current working
    directory — so a run started inside `.worktree/3-mlflow/` writes
    `.worktree/3-mlflow/mlruns/` while `uv run mlflow ui` from the main
    checkout reads an empty `./mlruns/`. Silent data fragmentation.
    Absolute paths and non-file schemes (sqlite://, http://, ...) pass
    through unchanged.
    """
    if not raw.startswith("file:"):
        return raw
    path = Path(raw.removeprefix("file:"))
    return path.resolve().as_uri() if not path.is_absolute() else raw


@contextmanager
def start_run(
    *,
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Run]:
    """Open a tracked MLflow run.

    Sets the tracking URI from `MLFLOW_TRACKING_URI` (defaults to
    `file:./mlruns`, resolved to an absolute path), selects or creates
    the named experiment, then yields the `Run` for the caller to
    enrich with `mlflow.log_param`, `mlflow.log_metric`,
    `mlflow.log_artifact`, etc.

    The run is auto-closed on exit. On exception the run status is set
    to FAILED — same as `mlflow.start_run`'s default contextmanager
    behavior.

    Concurrency: MLflow's active-run stack is `threading.local`. A
    nested call on the same thread (e.g., a backtest sub-run inside a
    research-agent node) must pass `nested=True` to avoid a runtime
    error. Two coroutines on the same event-loop thread share the
    same stack — async callers must serialize their `start_run` calls
    or pass `nested=True`.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI).strip() or _DEFAULT_URI
    mlflow.set_tracking_uri(_normalize_uri(uri))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        yield run


def configured_tracking_uri() -> str:
    """Read-only accessor — handy for diagnostics and tests."""
    raw = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI).strip() or _DEFAULT_URI
    return _normalize_uri(raw)
