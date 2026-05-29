"""`auto-research eval` — capture extraction-quality baselines."""

from __future__ import annotations

import click

from auto_research.eval.baseline import capture_baseline
from auto_research.eval.gold import load_gold_set
from auto_research.eval.registry import WORKER_EVALS


@click.group(name="eval")
def eval_group() -> None:
    """Extraction-quality eval commands."""


@eval_group.command(name="extract", help="DeepEval on extraction outputs (not yet implemented).")
def eval_extract() -> None:
    raise click.UsageError(
        "eval extract is not yet implemented. "
        "DeepEval suite for extraction is planned for the W1 wrap-up; "
        "see docs/plans/2026-05-22-auto-research-implementation.md."
    )


@eval_group.command(name="capture-baseline")
@click.option(
    "--worker",
    type=click.Choice(sorted(WORKER_EVALS)),
    required=True,
    help="Worker whose gold set to score.",
)
def capture_baseline_cmd(worker: str) -> None:
    """Run the worker over its gold set and write a baseline JSON (needs a key)."""
    spec = WORKER_EVALS[worker]
    gold_set = load_gold_set(spec.gold_path, worker=worker, thresholds=spec.default_thresholds)
    path = capture_baseline(spec, gold_set)
    click.echo(f"wrote {path}")
