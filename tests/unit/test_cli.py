"""Unit tests for the `auto-research` CLI surface.

All tests use Click's CliRunner — no subprocess, no network, no live API
calls. Subcommand wiring tests added in later tasks will additionally
patch the wrapped modules at the `auto_research.cli` boundary.
"""

from __future__ import annotations

from unittest.mock import patch

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


def test_ingest_edgar_default_form_types_is_s_filings(runner: CliRunner) -> None:
    """Smoke default: --form-types omitted -> ('S-3', 'S-1')."""
    with patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch:
        mock_fetch.return_value = []
        result = runner.invoke(cli, ["ingest", "edgar", "--cik", "0001045810"])
    assert result.exit_code == 0, result.output
    assert tuple(mock_fetch.call_args.kwargs["form_types"]) == ("S-3", "S-1")
