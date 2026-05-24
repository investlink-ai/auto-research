"""Unit tests for the `auto-research` CLI surface.

All tests use Click's CliRunner — no subprocess, no network, no live API
calls. Subcommand wiring tests added in later tasks will additionally
patch the wrapped modules at the `auto_research.cli` boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from auto_research.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.xfail(
    reason="cumulative - passes once Task 7 registers the status subcommand",
    strict=True,
)
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
    assert mock_fetch.call_args.kwargs["raw_root"] == Path("data/raw")
    assert mock_fetch.call_args.kwargs["manifest_path"] == Path("data/manifest.parquet")


def test_ingest_edgar_default_form_types_is_s_filings(runner: CliRunner) -> None:
    """Smoke default: --form-types omitted -> ('S-3', 'S-1')."""
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.return_value = []
        result = runner.invoke(cli, ["ingest", "edgar", "--cik", "0001045810"])
    assert result.exit_code == 0, result.output
    assert tuple(mock_fetch.call_args.kwargs["form_types"]) == ("S-3", "S-1")


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
    assert "quarantined=1" in result.output


def test_feast_apply_shells_out_to_feast_cli(runner: CliRunner) -> None:
    with patch("auto_research.cli.subprocess.run", autospec=True) as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.invoke(cli, ["feast", "apply"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["feast", "apply"]
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
