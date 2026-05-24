"""Unit tests for the `auto-research` CLI surface.

All tests use Click's CliRunner — no subprocess, no network, no live API
calls. Subcommand wiring tests added in later tasks will additionally
patch the wrapped modules at the `auto_research.cli` boundary.
"""

from __future__ import annotations

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
