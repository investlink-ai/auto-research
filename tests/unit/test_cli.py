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
