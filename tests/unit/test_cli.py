"""Unit tests for the `auto-research` CLI surface.

All tests use Click's CliRunner — no subprocess, no network, no live API
calls. Subcommand wiring tests added in later tasks will additionally
patch the wrapped modules at the `auto_research.cli` boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from auto_research.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_root_help_lists_every_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for expected in ("ingest", "extract", "feast", "eval", "status"):
        assert expected in result.output, f"{expected!r} missing from --help"
    # Verify status is a registered subcommand, not just mentioned in epilog text.
    assert runner.invoke(cli, ["status", "--help"]).exit_code == 0


def test_root_help_documents_required_env_vars(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for var in (
        "ANTHROPIC_API_KEY",
        "SEC_USER_AGENT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "MLFLOW_TRACKING_URI",
    ):
        assert var in result.output, f"{var} missing from root --help epilog"


def test_ingest_edgar_invokes_fetch_filings_for_cik(runner: CliRunner) -> None:
    """The CLI parses --cik / --form-types and forwards to the ingest function."""
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.return_value = []
        result = runner.invoke(
            cli,
            [
                "ingest",
                "edgar",
                "--cik",
                "0001045810",
                "--form-types",
                "S-3,S-1",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_fetch.assert_called_once()
    # CIK is forwarded as a positional argument; form_types split on comma.
    assert mock_fetch.call_args.args[0] == "0001045810"
    assert tuple(mock_fetch.call_args.kwargs["form_types"]) == ("S-3", "S-1")
    # --raw-root and --manifest-path are resolved to absolute paths by Click
    # (`resolve_path=True`), so ingest writes survive a CWD change between
    # invocations. The default values are still rooted at the project tree.
    assert mock_fetch.call_args.kwargs["raw_root"].is_absolute()
    assert mock_fetch.call_args.kwargs["raw_root"].name == "raw"
    assert mock_fetch.call_args.kwargs["manifest_path"].is_absolute()
    assert mock_fetch.call_args.kwargs["manifest_path"].name == "manifest.parquet"


def test_ingest_edgar_default_form_types_is_s_filings(runner: CliRunner) -> None:
    """Smoke default: --form-types omitted -> ('S-3', 'S-1')."""
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.return_value = []
        result = runner.invoke(cli, ["ingest", "edgar", "--cik", "0001045810"])
    assert result.exit_code == 0, result.output
    assert tuple(mock_fetch.call_args.kwargs["form_types"]) == ("S-3", "S-1")


def test_ingest_edgar_normalizes_unpadded_cik(runner: CliRunner) -> None:
    """Operator-friendly: --cik 1045810 must be forwarded as '0001045810'.

    Without this normalization, the manifest's entity_id (always 10-digit
    via edgar.py:_pad_cik) silently mismatches an unpadded extract filter.
    """
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.return_value = []
        result = runner.invoke(cli, ["ingest", "edgar", "--cik", "  1045810  "])
    assert result.exit_code == 0, result.output
    assert mock_fetch.call_args.args[0] == "0001045810"


def test_ingest_edgar_rejects_non_digit_cik(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["ingest", "edgar", "--cik", "NVDA"])
    assert result.exit_code != 0
    assert "--cik must be digits" in result.output


def test_ingest_edgar_rejects_empty_form_types(runner: CliRunner) -> None:
    """Empty / comma-only --form-types must error at parse time, not silently no-op."""
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        result = runner.invoke(
            cli, ["ingest", "edgar", "--cik", "0001045810", "--form-types", ","]
        )
    assert result.exit_code != 0
    assert "at least one" in result.output
    mock_fetch.assert_not_called()


def test_ingest_edgar_catches_edgar_config_error(runner: CliRunner) -> None:
    """A missing SEC_USER_AGENT surfaces as a clean UsageError, not a traceback."""
    from auto_research.ingest.edgar import EdgarConfigError

    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.side_effect = EdgarConfigError("SEC_USER_AGENT is not set")
        result = runner.invoke(cli, ["ingest", "edgar", "--cik", "0001045810"])
    assert result.exit_code != 0
    assert "SEC_USER_AGENT" in result.output


def _write_edgar_manifest_row(
    manifest_path: Path,
    *,
    cik: str,
    accession: str,
    form_type: str,
    raw_path: Path,
) -> None:
    """Write one EDGAR manifest row matching the schema in ingest/manifest.py."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "source": pa.array(["edgar"], type=pa.string()),
            "entity_id": pa.array([cik], type=pa.string()),
            "doc_id": pa.array([accession], type=pa.string()),
            "form_type": pa.array([form_type], type=pa.string()),
            "event_datetime": pa.array(
                [datetime(2024, 1, 2, 12, tzinfo=UTC)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "fetched_at": pa.array(
                [datetime(2024, 1, 2, 13, tzinfo=UTC)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "content_sha256": pa.array(["deadbeef"], type=pa.string()),
            "path": pa.array([str(raw_path)], type=pa.string()),
            "status": pa.array(["ok"], type=pa.string()),
        }
    )
    pq.write_table(table, manifest_path)


def test_extract_s_filings_iterates_manifest_for_cik(
    runner: CliRunner, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_doc = raw_root / "edgar" / "0001045810" / "2024" / "x.txt"
    raw_doc.parent.mkdir(parents=True, exist_ok=True)
    raw_doc.write_text("body of the S-3")
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000001",
        form_type="S-3",
        raw_path=raw_doc,
    )
    out_root = tmp_path / "extracted"

    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        class _Out:
            def model_dump_json(self, *, indent: int = 2) -> str:
                return "{}"

        mock_extract.return_value = _Out()
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(out_root),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_extract.assert_called_once()
    assert mock_extract.call_args.kwargs["raw_doc"] == "body of the S-3"
    assert mock_extract.call_args.kwargs["doc_id"] == "0001045810-24-000001"
    assert (out_root / "s_filings" / "0001045810-24-000001.json").exists()


def test_extract_s_filings_skips_non_s_filings(
    runner: CliRunner, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    raw_doc = raw_root / "edgar" / "0001045810" / "2024" / "y.txt"
    raw_doc.parent.mkdir(parents=True, exist_ok=True)
    raw_doc.write_text("10-K body")
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000002",
        form_type="10-K",
        raw_path=raw_doc,
    )
    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_extract.assert_not_called()


def test_extract_s_filings_logs_and_skips_missing_raw_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A manifest row whose `path` no longer exists on disk must be skipped,
    not crash the batch. Counts as quarantined."""
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000003",
        form_type="S-3",
        raw_path=tmp_path / "raw" / "does_not_exist.txt",
    )
    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_extract.assert_not_called()
    assert "skipping 0001045810-24-000003" in result.output
    # Missing files count as `skipped`, distinct from worker-quarantined.
    assert "skipped=1" in result.output
    assert "quarantined=0" in result.output


def test_extract_s_filings_catches_arbitrary_worker_exceptions(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A reliability primitive (CostCapExceeded, CircuitOpen) or Anthropic 401
    raised by extract_s_filing must not abort the batch; the row is counted
    as `failed` and the loop continues."""
    raw = tmp_path / "raw" / "x.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("body")
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000099",
        form_type="S-3",
        raw_path=raw,
    )
    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        mock_extract.side_effect = RuntimeError("circuit breaker open")
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "extraction failed for 0001045810-24-000099" in result.output
    assert "failed=1" in result.output


def test_extract_s_filings_forwards_quarantine_root(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--quarantine-root must reach extract_s_filing so operators can scope
    the quarantine tree (hermetic tests, alternate audit roots)."""
    raw = tmp_path / "raw" / "x.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("body")
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000100",
        form_type="S-3",
        raw_path=raw,
    )
    quarantine = tmp_path / "alt_quarantine"
    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        mock_extract.return_value = None  # worker quarantined
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
                "--quarantine-root",
                str(quarantine),
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_extract.call_args.kwargs["quarantine_root"] == quarantine


def test_extract_s_filings_normalizes_unpadded_cik(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Operator passes --cik 1045810; the manifest stores '0001045810'.
    Without normalization, the equality filter returns zero candidates."""
    raw = tmp_path / "raw" / "x.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("body")
    manifest_path = tmp_path / "manifest.parquet"
    _write_edgar_manifest_row(
        manifest_path,
        cik="0001045810",
        accession="0001045810-24-000101",
        form_type="S-3",
        raw_path=raw,
    )
    with patch("auto_research.cli.extract_s_filing", autospec=True) as mock_extract:
        class _Out:
            def model_dump_json(self, *, indent: int = 2) -> str:
                return "{}"

        mock_extract.return_value = _Out()
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "1045810",  # unpadded — must normalize to 0001045810
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_extract.assert_called_once()
    assert "candidates=1" in result.output
    assert "persisted=1" in result.output


def test_feast_apply_shells_out_to_feast_cli(runner: CliRunner) -> None:
    with patch("auto_research.cli.subprocess.run", autospec=True) as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.invoke(cli, ["feast", "apply"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["feast", "apply"]
    # cwd is the absolute resolved feast_repo path — relative cwd would let
    # callers from non-root directories silently target the wrong directory.
    assert kwargs["cwd"].is_absolute()
    assert kwargs["cwd"].name == "feast_repo"
    assert kwargs["check"] is False


def test_feast_apply_propagates_nonzero_exit(runner: CliRunner) -> None:
    with patch("auto_research.cli.subprocess.run", autospec=True) as mock_run:
        mock_run.return_value.returncode = 2
        result = runner.invoke(cli, ["feast", "apply"])
    assert result.exit_code == 2


def test_feast_materialize_requires_start_and_end(runner: CliRunner) -> None:
    with patch("auto_research.cli.subprocess.run", autospec=True) as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.invoke(
            cli,
            [
                "feast",
                "materialize",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
            ],
        )
    assert result.exit_code == 0, result.output
    args, _ = mock_run.call_args
    assert args[0] == ["feast", "materialize", "2024-01-01", "2024-01-31"]


def test_feast_materialize_missing_args_errors(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["feast", "materialize"])
    assert result.exit_code != 0
    assert "--start" in result.output


def test_feast_materialize_help_documents_iso_8601_datetime(runner: CliRunner) -> None:
    """The feast CLI requires ISO 8601 *datetime* (not date) for materialize.
    The help text must say so or operators waste a roundtrip diagnosing the
    opaque parse error from feast."""
    result = runner.invoke(cli, ["feast", "materialize", "--help"])
    assert result.exit_code == 0, result.output
    assert "ISO 8601" in result.output


def test_feast_apply_errors_cleanly_when_feast_repo_missing(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invoking `feast apply` from a directory without feast_repo/ must
    emit a clean UsageError instead of a FileNotFoundError traceback."""
    monkeypatch.chdir(tmp_path)
    with patch("auto_research.cli.subprocess.run", autospec=True) as mock_run:
        result = runner.invoke(cli, ["feast", "apply"])
    assert result.exit_code != 0
    assert "feast_repo/ not found" in result.output
    mock_run.assert_not_called()


def test_ingest_fmp_is_not_yet_implemented(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["ingest", "fmp", "--ticker", "NVDA"])
    assert result.exit_code != 0
    assert "not yet implemented" in result.output.lower()


def test_eval_extract_is_not_yet_implemented(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["eval", "extract"])
    assert result.exit_code != 0
    assert "not yet implemented" in result.output.lower()


def test_help_still_lists_fmp_and_eval_subcommands(runner: CliRunner) -> None:
    ingest_help = runner.invoke(cli, ["ingest", "--help"])
    assert "fmp" in ingest_help.output
    eval_help = runner.invoke(cli, ["eval", "--help"])
    assert "extract" in eval_help.output


def test_status_all_ok(runner: CliRunner) -> None:
    from auto_research.cli import CheckResult

    with (
        patch(
            "auto_research.cli._check_langfuse",
            return_value=CheckResult("langfuse", "ok", "http://localhost:3000"),
        ),
        patch(
            "auto_research.cli._check_mlflow",
            return_value=CheckResult("mlflow", "ok", "file:///tmp/mlruns"),
        ),
        patch(
            "auto_research.cli._check_feast",
            return_value=CheckResult("feast", "ok", "3 feature_views"),
        ),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "langfuse" in result.output
    assert "mlflow" in result.output
    assert "feast" in result.output


def test_status_exits_1_when_any_check_errors(runner: CliRunner) -> None:
    from auto_research.cli import CheckResult

    with (
        patch(
            "auto_research.cli._check_langfuse",
            return_value=CheckResult("langfuse", "error", "connection refused"),
        ),
        patch(
            "auto_research.cli._check_mlflow",
            return_value=CheckResult("mlflow", "ok", "ok"),
        ),
        patch(
            "auto_research.cli._check_feast",
            return_value=CheckResult("feast", "ok", "ok"),
        ),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_check_mlflow_reports_configured_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File backend reports the resolved tracking URI; warn if dir is missing."""
    from auto_research.cli import _check_mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    res = _check_mlflow()
    assert res.name == "mlflow"
    assert res.status in {"ok", "warn"}
    assert "mlruns" in res.detail


def test_check_mlflow_single_slash_uri_warns_on_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-slash file: URI must still trigger the dir-existence check."""
    from auto_research.cli import _check_mlflow

    missing = tmp_path / "absent" / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{missing}")
    res = _check_mlflow()
    assert res.name == "mlflow"
    assert res.status == "warn"
    assert "not yet created" in res.detail


def test_check_feast_returns_warn_when_registry_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No registry.db -> warn (not error): 'not applied' is a fixable state."""
    from auto_research.cli import _check_feast

    monkeypatch.chdir(tmp_path)
    res = _check_feast()
    assert res.name == "feast"
    assert res.status == "warn"
    assert "feast_repo" in res.detail.lower()


def test_check_langfuse_warn_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing env vars -> warn (not error): bootstrap state."""
    from auto_research.cli import _check_langfuse

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    res = _check_langfuse()
    assert res.name == "langfuse"
    assert res.status == "warn"


def test_mask_url_credentials_strips_basic_auth() -> None:
    """INV-7: any basic-auth pair in a URL must be masked before display."""
    from auto_research.cli import _mask_url_credentials

    assert _mask_url_credentials("http://admin:s3cr3t@host:3000/path") == (
        "http://***@host:3000/path"
    )
    assert _mask_url_credentials("http://user@host/") == "http://***@host/"
    # No auth -> identity.
    assert _mask_url_credentials("http://localhost:3000") == "http://localhost:3000"


def test_check_langfuse_masks_credentials_in_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGFUSE_HOST with embedded credentials must never echo them in detail."""
    from auto_research.cli import _check_langfuse

    monkeypatch.setenv("LANGFUSE_HOST", "http://admin:s3cr3t@langfuse.internal/")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    # Force the HTTP probe to error so we exercise the non-200 detail path
    # (the most exposure-prone branch) without standing up a real server.
    with patch("auto_research.cli.httpx.get", side_effect=httpx.ConnectError("x")):
        res = _check_langfuse()
    assert res.status == "error"
    assert "s3cr3t" not in res.detail
    assert "admin" not in res.detail
    assert "***@langfuse.internal" in res.detail


def test_check_langfuse_strips_trailing_slash_in_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGFUSE_HOST=http://host:3000/ must not produce //api/public/health."""
    from auto_research.cli import _check_langfuse

    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.local:3000/")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    with patch("auto_research.cli.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        _check_langfuse()
    called_url = mock_get.call_args.args[0]
    assert called_url == "http://langfuse.local:3000/api/public/health"


def test_check_mlflow_returns_error_on_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the experiment module fails to import, status must surface a clean
    error CheckResult — not crash the list-comp mid-build before any line
    prints."""
    import builtins
    from typing import Any

    from auto_research.cli import _check_mlflow

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "auto_research.experiment":
            raise ImportError("simulated broken experiment module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = _check_mlflow()
    assert res.name == "mlflow"
    assert res.status == "error"
    assert "import failed" in res.detail


# --------------------------------------------------------------------------- #
# Telemetry wiring                                                             #
# --------------------------------------------------------------------------- #


def test_ingest_edgar_initializes_telemetry(runner: CliRunner) -> None:
    """ingest edgar must call try_init_telemetry at command start (refs #52)."""
    with (
        patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init,
        patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch,
    ):
        mock_init.return_value = True
        mock_fetch.return_value = []
        result = runner.invoke(
            cli,
            ["ingest", "edgar", "--cik", "0001045810"],
        )
    assert result.exit_code == 0, result.output
    mock_init.assert_called_once()


def test_feast_apply_initializes_telemetry(runner: CliRunner, tmp_path: Path) -> None:
    """feast apply does I/O (subprocess + registry writes) and must
    initialize telemetry per docs/ARCHITECTURE.md §7."""
    feast_repo = tmp_path / "feast_repo"
    feast_repo.mkdir()
    with (
        runner.isolated_filesystem(temp_dir=tmp_path) as cwd,
        patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init,
        patch("auto_research.cli.subprocess.run") as mock_run,
    ):
        Path(cwd, "feast_repo").mkdir()
        mock_init.return_value = True
        mock_run.return_value.returncode = 0
        result = runner.invoke(cli, ["feast", "apply"])
    assert result.exit_code == 0
    mock_init.assert_called_once()


def test_feast_materialize_initializes_telemetry(
    runner: CliRunner, tmp_path: Path
) -> None:
    """feast materialize runs minutes of online-store writes; must
    initialize telemetry."""
    with (
        runner.isolated_filesystem(temp_dir=tmp_path) as cwd,
        patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init,
        patch("auto_research.cli.subprocess.run") as mock_run,
    ):
        Path(cwd, "feast_repo").mkdir()
        mock_init.return_value = True
        mock_run.return_value.returncode = 0
        result = runner.invoke(
            cli,
            [
                "feast",
                "materialize",
                "--start",
                "2024-01-01T00:00:00",
                "--end",
                "2024-01-31T00:00:00",
            ],
        )
    assert result.exit_code == 0
    mock_init.assert_called_once()


def test_extract_s_filings_initializes_telemetry(
    runner: CliRunner, tmp_path: Path
) -> None:
    """extract s-filings must call try_init_telemetry even on an empty manifest."""
    manifest_path = tmp_path / "manifest.parquet"
    empty = pa.table(
        {
            "source": pa.array([], type=pa.string()),
            "entity_id": pa.array([], type=pa.string()),
            "doc_id": pa.array([], type=pa.string()),
            "form_type": pa.array([], type=pa.string()),
            "event_datetime": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "fetched_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "content_sha256": pa.array([], type=pa.string()),
            "path": pa.array([], type=pa.string()),
            "status": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(empty, manifest_path)

    with patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init:
        # Returning False (missing env) is fine — the helper must still
        # have been called, and the command must still succeed.
        mock_init.return_value = False
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest_path),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_init.assert_called_once()


# ---- extract reembed -----------------------------------------------------


def _write_doc(rag_root: Path, doc_id: str, text: str, doc_type: str = "10-K") -> None:
    """Embed one chunk under `doc_id` via BGE and promote that
    materialization so `extract reembed` has both a real LanceDB table
    AND an active pointer to read the source from. Kept tiny — these
    tests exercise CLI wiring, not adapter mechanics (those are covered
    in test_embeddings.py).
    """
    from datetime import date

    from auto_research.extract.chunking import ChildChunk, ChunkMetadata
    from auto_research.extract.chunking_contextual import ContextualChildChunk
    from auto_research.extract.embeddings import EmbeddingAdapter
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        now_utc_iso,
        write_active_materialization,
    )

    md = ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 3, 15),
        fiscal_period="FY2025",
        doc_type=doc_type,
        doc_id=doc_id,
    )
    child = ChildChunk(
        text=text,
        char_span=(0, len(text)),
        token_count=len(text.split()),
        parent_id=f"{doc_id}:0:{len(text)}",
        section_name="Item 7",
        from_table=False,
        metadata=md,
    )
    adapter = EmbeddingAdapter(backend="bge", rag_root=rag_root)
    adapter.embed([ContextualChildChunk(child=child, context="")])
    # Promote the just-embedded materialization so subsequent reembed CLI
    # paths (which require an active pointer) have a valid source. Same
    # materialization_version each call so repeated _write_doc invocations
    # accumulate into the same active namespace.
    write_active_materialization(
        rag_root,
        ActiveMaterialization(
            version=adapter.materialization_version,
            embed_model_version=adapter.embed_model_version,
            promoted_at=now_utc_iso(),
            manifest_count=1,
        ),
    )


def test_extract_reembed_help_advertises_qwen3_default(runner: CliRunner) -> None:
    """`extract reembed --help` must show qwen3-mlx as the default and list
    all three backends; the docstring claim about the default must match
    Click's actual `default=` behavior.
    """
    result = runner.invoke(cli, ["extract", "reembed", "--help"])
    assert result.exit_code == 0, result.output
    assert "[voyage|bge|qwen3-mlx]" in result.output
    # Click line-wraps "[default: qwen3-mlx]" — check both halves independently.
    assert "[default:" in result.output
    assert "qwen3-mlx]" in result.output


def test_extract_reembed_requires_exactly_one_target(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Zero of --doc-id/--corpus/--all => UsageError; two of them => UsageError."""
    none_result = runner.invoke(
        cli, ["extract", "reembed", "--backend", "bge", "--rag-root", str(tmp_path)]
    )
    assert none_result.exit_code != 0
    assert "exactly one of --doc-id, --corpus, --all" in none_result.output

    two_result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(tmp_path),
            "--doc-id", "doc-X",
            "--corpus",
        ],
    )
    assert two_result.exit_code != 0
    assert "exactly one of --doc-id, --corpus, --all" in two_result.output


def test_extract_reembed_dry_run_reports_tokens_and_zero_cost_for_bge(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Dry-run must NOT call the encoder. For BGE it reports $0 (in-process)
    and a token count > 0. Asserts the output line format the operator sees.
    """
    _write_doc(tmp_path, "doc-DR", "a few words of dry run text")
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(tmp_path),
            "--doc-id", "doc-DR",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "reembed dry-run:" in out
    assert "backend=bge" in out
    assert "tables=1" in out
    assert "rows=1" in out
    assert "in-process" in out


def test_extract_reembed_dry_run_voyage_reports_usd_estimate(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voyage dry-run prints a $X.XX estimate derived from the rate dict.
    No `VOYAGE_API_KEY` needed because dry-run never instantiates the
    Voyage client (encoder not called).
    """
    _write_doc(tmp_path, "doc-DVY", "voyage cost estimation text")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "voyage",
            "--rag-root", str(tmp_path),
            "--doc-id", "doc-DVY",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "backend=voyage" in out
    assert "/MTok" in out
    assert "$" in out  # USD cost rendered


def test_extract_reembed_doc_id_invokes_reembed_doc(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--doc-id X forwards to adapter.reembed_doc('X'); --corpus is not called."""
    _write_doc(tmp_path, "doc-RDX", "live path text")
    with patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_doc",
        autospec=True,
        return_value=1,
    ) as mock_doc, patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_corpus",
        autospec=True,
    ) as mock_corpus:
        result = runner.invoke(
            cli,
            [
                "extract", "reembed", "--backend", "bge",
                "--rag-root", str(tmp_path),
                "--doc-id", "doc-RDX",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_doc.call_count == 1
    assert mock_doc.call_args.args[1] == "doc-RDX"
    mock_corpus.assert_not_called()


def test_extract_reembed_all_walks_rag_root_and_does_not_call_reembed_corpus(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--all enumerates `<rag_root>/*.lance` directories minus the corpus
    store and reembeds every per-doc table; it does NOT additionally call
    reembed_corpus because reembed_doc internally propagates fresh vectors
    into the corpus narrative store via vector-copy (review finding #2).
    Calling reembed_corpus on top would re-encode every narrative chunk a
    second time, doubling Voyage spend and risking batch-boundary
    nondeterminism between the per-doc and corpus stores.
    """
    _write_doc(tmp_path, "doc-A1", "alpha text", doc_type="10-K")
    _write_doc(tmp_path, "doc-A2", "beta text", doc_type="10-K")
    with patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_doc",
        autospec=True,
        return_value=1,
    ) as mock_doc, patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_corpus",
        autospec=True,
        return_value=2,
    ) as mock_corpus:
        result = runner.invoke(
            cli,
            [
                "extract", "reembed", "--backend", "bge",
                "--rag-root", str(tmp_path),
                "--all",
            ],
        )
    assert result.exit_code == 0, result.output
    called_doc_ids = sorted(c.args[1] for c in mock_doc.call_args_list)
    assert called_doc_ids == ["doc-A1", "doc-A2"]
    mock_corpus.assert_not_called()


def test_extract_reembed_corpus_only_skips_per_doc_tables(
    runner: CliRunner, tmp_path: Path
) -> None:
    _write_doc(tmp_path, "doc-CX", "corpus only text")
    with patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_doc",
        autospec=True,
    ) as mock_doc, patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_corpus",
        autospec=True,
        return_value=1,
    ) as mock_corpus:
        result = runner.invoke(
            cli,
            [
                "extract", "reembed", "--backend", "bge",
                "--rag-root", str(tmp_path),
                "--corpus",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_doc.assert_not_called()
    assert mock_corpus.call_count == 1


def test_extract_reembed_failure_returns_nonzero_exit(
    runner: CliRunner, tmp_path: Path
) -> None:
    """If reembed_doc raises (e.g., dim mismatch), the CLI catches, logs,
    and exits non-zero so operators wiring this into make targets see the
    failure.
    """
    _write_doc(tmp_path, "doc-FAIL", "boom")
    with patch(
        "auto_research.extract.embeddings.EmbeddingAdapter.reembed_doc",
        autospec=True,
        side_effect=RuntimeError("simulated dim mismatch"),
    ):
        result = runner.invoke(
            cli,
            [
                "extract", "reembed", "--backend", "bge",
                "--rag-root", str(tmp_path),
                "--doc-id", "doc-FAIL",
            ],
        )
    assert result.exit_code != 0
    assert "simulated dim mismatch" in result.output


def test_extract_reembed_doc_id_empty_string_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Review finding #15: `--doc-id ''` is rejected with a clear message
    rather than silently misclassified as 'no target specified'."""
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(tmp_path),
            "--doc-id", "",
        ],
    )
    assert result.exit_code != 0
    assert "--doc-id must be a non-empty string" in result.output


def test_extract_reembed_doc_id_corpus_store_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Review finding #9: `--doc-id _corpus_narrative` would route the
    corpus reembed through `reembed_doc`, polluting OTel `extract.doc_id`
    with a synthetic non-doc id. Reject loudly and point the operator at
    `--corpus`.
    """
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(tmp_path),
            "--doc-id", "_corpus_narrative",
        ],
    )
    assert result.exit_code != 0
    assert "_corpus_narrative" in result.output
    assert "--corpus" in result.output


def test_extract_reembed_live_path_validates_rag_root(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Review finding #7: a typo'd --rag-root on the live path produces
    a clear UsageError up front rather than a misleading per-table
    'reembed_doc() failed' message after lancedb silently creates the
    missing directory.
    """
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(missing),
            "--doc-id", "doc-X",
        ],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_extract_reembed_dry_run_voyage_estimates_to_four_decimals(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding #6: dry-run USD format must surface sub-cent
    estimates. A few-word reembed at $0.06/MTok is ~$0.0001 — formatted
    via :.2f it would render as $0.00 and falsely advertise a free run.
    """
    _write_doc(tmp_path, "doc-CENT", "a tiny doc")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "voyage",
            "--rag-root", str(tmp_path),
            "--doc-id", "doc-CENT",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Four decimal places visible (matches the ":.4f" specifier).
    import re

    match = re.search(r"~\$(\d+\.\d{4})", result.output)
    assert match is not None, (
        f"dry-run output missing 4-decimal USD format: {result.output!r}"
    )


def test_extract_reembed_all_dry_run_does_not_double_count_narrative_rows(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Review finding #2: a narrative chunk lives in both <doc>.lance AND
    _corpus_narrative. The dry-run for `--all` must count it ONCE
    (per-doc only) because per-doc reembed propagates corpus rows via
    vector-copy with no encoder cost. Pre-fix, `--all` dry-run included
    the corpus table in the token sum and inflated USD ~2x for
    narrative-heavy corpora.
    """
    _write_doc(tmp_path, "doc-DD1", "narrative passage one", doc_type="10-K")
    _write_doc(tmp_path, "doc-DD2", "narrative passage two", doc_type="10-K")
    result = runner.invoke(
        cli,
        [
            "extract", "reembed", "--backend", "bge",
            "--rag-root", str(tmp_path),
            "--all",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Only the two per-doc tables are counted; _corpus_narrative is
    # excluded from --all's dry-run target list.
    assert "tables=2" in result.output
    assert "rows=2" in result.output


# ---- extract list-materializations / promote-materialization / gc -------


def _write_manifest(manifest_path: Path, doc_ids: list[str]) -> None:
    """Write a minimal ingest manifest with the given doc_ids under
    `source='edgar'`, `status='ok'` so `promote-materialization` can
    validate namespace completeness against it. Field set matches
    `auto_research.ingest.manifest.SCHEMA` exactly so the parquet
    append succeeds.
    """
    from datetime import UTC, datetime

    from auto_research.ingest import manifest as manifest_mod

    ts = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    rows = [
        {
            "source": "edgar",
            "entity_id": "0001045810",
            "doc_id": doc_id,
            "form_type": "10-K",
            "event_datetime": ts,
            "fetched_at": ts,
            "content_sha256": "x" * 64,
            "path": f"data/raw/{doc_id}.txt",
            "status": "ok",
        }
        for doc_id in doc_ids
    ]
    manifest_mod.append(manifest_path, rows)


def test_list_materializations_empty_rag_root_reports_no_results(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli,
        ["extract", "list-materializations", "--rag-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "no materializations found" in result.output


def test_list_materializations_marks_active_version(
    runner: CliRunner, tmp_path: Path
) -> None:
    _write_doc(tmp_path, "doc-L1", "list-mat passage")
    _write_doc(tmp_path, "doc-L2", "list-mat passage two")
    result = runner.invoke(
        cli,
        ["extract", "list-materializations", "--rag-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # _write_doc embeds + promotes the same materialization for both docs.
    # Expect: 1 materialization line, marked (active), tables=3 (2 per-doc + corpus).
    assert "(active)" in result.output
    assert "tables=3" in result.output


def test_promote_materialization_succeeds_when_namespace_complete(
    runner: CliRunner, tmp_path: Path
) -> None:
    from auto_research.extract.materialization import read_active_materialization

    _write_doc(tmp_path, "doc-P1", "promote-ok one")
    _write_doc(tmp_path, "doc-P2", "promote-ok two")
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest, ["doc-P1", "doc-P2"])

    from auto_research.extract.embeddings import EmbeddingAdapter

    version = EmbeddingAdapter(backend="bge", rag_root=tmp_path).materialization_version
    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "promoted" in result.output
    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.version == version
    assert active.manifest_count == 2


def test_promote_materialization_refuses_incomplete_namespace(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Manifest with 3 doc_ids vs only 2 embedded tables — promotion
    refused with the missing doc_id named so the operator can fix it."""
    _write_doc(tmp_path, "doc-INC1", "incomplete one")
    _write_doc(tmp_path, "doc-INC2", "incomplete two")
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest, ["doc-INC1", "doc-INC2", "doc-INC3-MISSING"])

    from auto_research.extract.embeddings import EmbeddingAdapter

    version = EmbeddingAdapter(backend="bge", rag_root=tmp_path).materialization_version
    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert "incomplete" in result.output.lower()
    assert "doc-INC3-MISSING" in result.output


def test_gc_materialization_keeps_active_and_drops_oldest(
    runner: CliRunner, tmp_path: Path
) -> None:
    """GC sorts by promotion history; keeps active + (--keep-last - 1)
    most-recent previous; removes the rest. Sanity-checks the on-disk
    `.lance` directories actually get removed."""
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        append_promotion_history,
        write_active_materialization,
    )

    for version in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        (tmp_path / f"doc-GC__{version}.lance").mkdir(parents=True)

    for version in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        append_promotion_history(
            tmp_path,
            ActiveMaterialization(
                version=version,
                embed_model_version=f"bge:bge-small-en-v1.5:tag-{version}",
                promoted_at=f"2026-05-{version[0]}0T12:00:00Z",
                manifest_count=1,
            ),
        )
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version="cccccccc",
            embed_model_version="bge:bge-small-en-v1.5:tag-cccccccc",
            promoted_at="2026-05-26T12:00:00Z",
            manifest_count=1,
        ),
    )

    result = runner.invoke(
        cli,
        [
            "extract", "gc-materialization",
            "--keep-last", "2",
            "--rag-root", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "doc-GC__cccccccc.lance").exists()
    assert (tmp_path / "doc-GC__bbbbbbbb.lance").exists()
    assert not (tmp_path / "doc-GC__aaaaaaaa.lance").exists()


def test_gc_materialization_dry_run_makes_no_changes(
    runner: CliRunner, tmp_path: Path
) -> None:
    from auto_research.extract.materialization import (
        ActiveMaterialization,
        append_promotion_history,
        write_active_materialization,
    )

    (tmp_path / "doc-GCD__aaaaaaaa.lance").mkdir(parents=True)
    (tmp_path / "doc-GCD__bbbbbbbb.lance").mkdir(parents=True)
    for version in ("aaaaaaaa", "bbbbbbbb"):
        append_promotion_history(
            tmp_path,
            ActiveMaterialization(
                version=version,
                embed_model_version="bge:bge-small-en-v1.5:tag",
                promoted_at="2026-05-26T12:00:00Z",
                manifest_count=1,
            ),
        )
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version="bbbbbbbb",
            embed_model_version="bge:bge-small-en-v1.5:tag",
            promoted_at="2026-05-26T12:00:00Z",
            manifest_count=1,
        ),
    )

    result = runner.invoke(
        cli,
        [
            "extract", "gc-materialization",
            "--keep-last", "1",
            "--rag-root", str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert (tmp_path / "doc-GCD__aaaaaaaa.lance").exists()
    assert (tmp_path / "doc-GCD__bbbbbbbb.lance").exists()


# ---- code-review fixes ---------------------------------------------------


def test_promote_materialization_refuses_fully_empty_namespace(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A namespace where every table is empty would yield an
    'unknown:unknown:unknown' embed_model_version pointer that no
    adapter could match — the read-path mismatch guard would reject
    every subsequent query. Refuse upfront and name the cause."""
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema
    from auto_research.extract.materialization import versioned_table_name

    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest, ["doc-EMPTY-NS"])

    version = "deadbeef"
    db = lancedb.connect(tmp_path)
    schema = _schema(384)
    db.create_table(
        versioned_table_name("doc-EMPTY-NS", version),
        data=pa.Table.from_pylist([], schema=schema),
        schema=schema,
    )

    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert "no rows" in result.output.lower() or "fully-empty" in result.output


def test_promote_materialization_requires_corpus_when_narrative_present(
    runner: CliRunner, tmp_path: Path
) -> None:
    """If any per-doc table at this version carries a narrative doc_type
    (10-K / 10-Q / transcript), the corpus_narrative table MUST exist —
    otherwise store='corpus_narrative' queries against the promoted
    namespace fail at runtime with FileNotFoundError."""
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema
    from auto_research.extract.materialization import versioned_table_name

    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest, ["doc-10K"])

    version = "cafebabe"
    db = lancedb.connect(tmp_path)
    schema = _schema(384)
    rows = [
        {
            "text": "10-K narrative row",
            "vector": [0.0] * 384,
            "ticker": "NVDA",
            "filing_date": "2025-03-15",
            "fiscal_period": "FY2025",
            "doc_type": "10-K",
            "doc_id": "doc-10K",
            "parent_id": "doc-10K:0:18",
            "section_name": "Item 7",
            "chunker_version": "v1",
            "contextual_prompt_version": "v1",
            "embed_model_version": "bge:bge-small-en-v1.5:v1",
        }
    ]
    db.create_table(
        versioned_table_name("doc-10K", version),
        data=pa.Table.from_pylist(rows, schema=schema),
        schema=schema,
    )

    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert "narrative" in result.output.lower()
    assert "_corpus_narrative" in result.output


def test_promote_materialization_completeness_check_includes_non_edgar_sources(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The completeness check must cover every source in the manifest, not
    just `edgar`. A future FMP doc with no per-doc table must cause the
    promotion to fail; today this is exercised by injecting a manifest
    row with a non-edgar `source`."""
    from datetime import UTC, datetime

    from auto_research.ingest import manifest as manifest_mod

    _write_doc(tmp_path, "doc-EDG", "edgar doc embed", doc_type="10-K")

    manifest = tmp_path / "manifest.parquet"
    ts = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    manifest_mod.append(
        manifest,
        [
            {
                "source": "edgar",
                "entity_id": "0001045810",
                "doc_id": "doc-EDG",
                "form_type": "10-K",
                "event_datetime": ts,
                "fetched_at": ts,
                "content_sha256": "x" * 64,
                "path": "data/raw/doc-EDG.txt",
                "status": "ok",
            },
            {
                "source": "fmp",
                "entity_id": "0001045810",
                "doc_id": "doc-FMP-MISSING",
                "form_type": "earnings_estimate",
                "event_datetime": ts,
                "fetched_at": ts,
                "content_sha256": "y" * 64,
                "path": "data/fmp/doc-FMP-MISSING.json",
                "status": "ok",
            },
        ],
    )

    from auto_research.extract.embeddings import EmbeddingAdapter

    version = EmbeddingAdapter(backend="bge", rag_root=tmp_path).materialization_version
    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert "doc-FMP-MISSING" in result.output


def test_gc_materialization_continues_after_rmtree_failure(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single rmtree failure (locked dir, EIO) must not abort the GC
    sweep — the CLI reports the partial failure, continues deleting other
    targets, and exits non-zero so make-target wiring surfaces it."""
    from typing import Any

    from auto_research.extract.materialization import (
        ActiveMaterialization,
        append_promotion_history,
        write_active_materialization,
    )

    for version in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        (tmp_path / f"doc-RM__{version}.lance").mkdir(parents=True)
    for version in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        append_promotion_history(
            tmp_path,
            ActiveMaterialization(
                version=version,
                embed_model_version="bge:bge-small-en-v1.5:tag",
                promoted_at="2026-05-26T12:00:00Z",
                manifest_count=1,
            ),
        )
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version="cccccccc",
            embed_model_version="bge:bge-small-en-v1.5:tag",
            promoted_at="2026-05-26T12:00:00Z",
            manifest_count=1,
        ),
    )

    import shutil

    original_rmtree = shutil.rmtree
    aaaaa_path = tmp_path / "doc-RM__aaaaaaaa.lance"

    def _selective_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path) == aaaaa_path:
            raise OSError("simulated locked directory")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", _selective_rmtree)

    result = runner.invoke(
        cli,
        [
            "extract", "gc-materialization",
            "--keep-last", "1",
            "--rag-root", str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    # bbbbb removed despite the aaaaa failure.
    assert not (tmp_path / "doc-RM__bbbbbbbb.lance").exists()
    # aaaaa attempted but failed.
    assert (tmp_path / "doc-RM__aaaaaaaa.lance").exists()
    # cccccc (active) kept.
    assert (tmp_path / "doc-RM__cccccccc.lance").exists()
    assert "simulated locked directory" in result.output


# ---- Codex P1: gc refuses without active pointer ------------------------


def test_gc_materialization_refuses_when_no_active_pointer(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A fresh-install or built-but-not-yet-promoted namespace has no
    active pointer. Running gc-materialization in that state used to
    silently wipe every materialization on disk (empty keep set + non-
    empty to_remove set = data loss). The CLI must refuse with a
    UsageError pointing at `promote` as the next step."""
    # Build two materializations on disk but write NO active pointer.
    (tmp_path / "doc-FRESH__aaaaaaaa.lance").mkdir(parents=True)
    (tmp_path / "doc-FRESH__bbbbbbbb.lance").mkdir(parents=True)

    result = runner.invoke(
        cli,
        [
            "extract", "gc-materialization",
            "--keep-last", "1",
            "--rag-root", str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "no active materialization" in result.output
    assert "promote" in result.output.lower()
    # Both directories untouched.
    assert (tmp_path / "doc-FRESH__aaaaaaaa.lance").exists()
    assert (tmp_path / "doc-FRESH__bbbbbbbb.lance").exists()


# ---- Codex P2: promote validates embed_model_version across all rows ----


def test_promote_materialization_refuses_internally_mixed_embed_model_versions(
    runner: CliRunner, tmp_path: Path
) -> None:
    """If a single table contains rows from MULTIPLE distinct
    embed_model_version stamps (a build-path bug or manual LanceDB
    ops), promotion must refuse — landing one stamp in the active
    pointer while the table actually serves two vector spaces would
    defeat the read-path mismatch guard for half the corpus."""
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema
    from auto_research.extract.materialization import versioned_table_name

    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest, ["doc-MIXED"])

    version = "feedface"
    schema = _schema(384)
    # Two rows with different embed_model_version stamps in the SAME table.
    rows = [
        {
            "text": "voyage-stamped row",
            "vector": [0.0] * 384,
            "ticker": "NVDA",
            "filing_date": "2025-03-15",
            "fiscal_period": "FY2025",
            "doc_type": "10-K",
            "doc_id": "doc-MIXED",
            "parent_id": "doc-MIXED:0:20",
            "section_name": "Item 7",
            "chunker_version": "v1",
            "contextual_prompt_version": "v1",
            "embed_model_version": "voyage:voyage-finance-2:v1",
        },
        {
            "text": "bge-stamped row in same table",
            "vector": [1.0] * 384,
            "ticker": "NVDA",
            "filing_date": "2025-03-15",
            "fiscal_period": "FY2025",
            "doc_type": "10-K",
            "doc_id": "doc-MIXED",
            "parent_id": "doc-MIXED:0:29",
            "section_name": "Item 7",
            "chunker_version": "v1",
            "contextual_prompt_version": "v1",
            "embed_model_version": "bge:bge-small-en-v1.5:v1",
        },
    ]
    db = lancedb.connect(tmp_path)
    db.create_table(
        versioned_table_name("doc-MIXED", version),
        data=pa.Table.from_pylist(rows, schema=schema),
        schema=schema,
    )
    # Also create an empty corpus table so the narrative-corpus guard
    # doesn't trip first — the test is about INTERNAL stamp consistency.
    db.create_table(
        versioned_table_name("_corpus_narrative", version),
        data=pa.Table.from_pylist(rows, schema=schema),
        schema=schema,
    )

    result = runner.invoke(
        cli,
        [
            "extract", "promote-materialization",
            "--version", version,
            "--rag-root", str(tmp_path),
            "--manifest-path", str(manifest),
        ],
    )
    assert result.exit_code != 0
    msg = result.output
    assert "NOT uniform" in msg or "internally mixed" in msg.lower()
    assert "voyage:voyage-finance-2:v1" in msg
    assert "bge:bge-small-en-v1.5:v1" in msg
