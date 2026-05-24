"""Unit tests for the MLflow experiment wrapper.

Uses `tmp_path` to point each test at an isolated `mlruns/` directory,
so tests don't share state with each other or with the developer's
local MLflow store.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from auto_research.experiment import configured_tracking_uri, start_run


@pytest.fixture
def isolated_mlruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets a fresh mlruns/ dir inside tmp_path."""
    mlruns = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{mlruns}")
    return mlruns


def test_configured_tracking_uri_passes_absolute_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/some/abs/path")
    assert configured_tracking_uri() == "file:/some/abs/path"


def test_configured_tracking_uri_resolves_relative_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default `file:./mlruns` resolves to an absolute file URI."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = configured_tracking_uri()
    assert uri.startswith("file://")
    assert uri.endswith("/mlruns")


def test_configured_tracking_uri_stable_across_cwds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Relative URIs anchor on project root — NOT current working directory.

    Defends against worktree-vs-main-checkout data fragmentation: calling
    configured_tracking_uri() from any CWD must return the same store URI.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri_from_repo_root = configured_tracking_uri()

    monkeypatch.chdir(tmp_path)
    uri_from_other_cwd = configured_tracking_uri()

    assert uri_from_repo_root == uri_from_other_cwd
    assert str(tmp_path) not in uri_from_other_cwd


def test_configured_tracking_uri_passes_non_file_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sqlite/http URIs are untouched (only relative file: paths get resolved)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    assert configured_tracking_uri() == "sqlite:///mlflow.db"


def test_start_run_round_trips_param(isolated_mlruns: Path) -> None:
    with start_run(experiment="test-exp", run_name="param-rt") as run:
        mlflow.log_param("signal_id", "A2")
        run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.params["signal_id"] == "A2"
    assert fetched.info.status == "FINISHED"


def test_start_run_round_trips_metric_and_tag(isolated_mlruns: Path) -> None:
    with start_run(
        experiment="test-exp",
        run_name="metric-rt",
        tags={"phase": "smoke"},
    ) as run:
        mlflow.log_metric("sharpe_net", 0.83)
        run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.metrics["sharpe_net"] == pytest.approx(0.83)
    assert fetched.data.tags["phase"] == "smoke"


def test_start_run_logs_artifact(isolated_mlruns: Path, tmp_path: Path) -> None:
    payload = tmp_path / "report.txt"
    payload.write_text("hello mlflow")

    with start_run(experiment="test-exp", run_name="artifact-rt") as run:
        mlflow.log_artifact(str(payload))
        run_id = run.info.run_id

    listed = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert any(a.path == "report.txt" for a in listed)


def test_start_run_marks_failed_on_exception(isolated_mlruns: Path) -> None:
    """MLflow's contextmanager should set status=FAILED if the body raises."""

    class SmokeError(RuntimeError):
        pass

    run_id: str | None = None
    with (
        pytest.raises(SmokeError),
        start_run(experiment="test-exp", run_name="fail-rt") as run,
    ):
        run_id = run.info.run_id
        raise SmokeError("intentional")

    assert run_id is not None
    fetched = mlflow.get_run(run_id)
    assert fetched.info.status == "FAILED"
