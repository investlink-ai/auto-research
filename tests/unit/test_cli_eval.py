"""Unit tests for the `auto-research eval` command group."""

from __future__ import annotations

from click.testing import CliRunner

from auto_research.cli import cli


def test_eval_capture_baseline_help() -> None:
    res = CliRunner().invoke(cli, ["eval", "capture-baseline", "--help"])
    assert res.exit_code == 0
    assert "--worker" in res.output
