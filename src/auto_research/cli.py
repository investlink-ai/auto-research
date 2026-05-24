"""`auto-research` console-script entry point.

Thin Click wrapper around the W1 primitives — EDGAR ingest, S-filings
extraction, Feast apply/materialize, and observability health checks.
Backing modules (ingest/edgar, extract/workers/s_filings, experiment,
telemetry, feast_repo/) hold the real logic; this module is wiring +
exit-code discipline.

Required environment for the full W1 smoke (`make smoke`):

    SEC_USER_AGENT       — EDGAR fair-access policy (required at ingest)
    ANTHROPIC_API_KEY    — extraction LLM calls
    OTEL_EXPORTER_OTLP_ENDPOINT, LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY  — telemetry export (status command)
    MLFLOW_TRACKING_URI  — defaults to file:./mlruns if unset
"""

from __future__ import annotations

import click

_ENV_VAR_EPILOG = """
\b
Required environment variables (see .env.example):
  SEC_USER_AGENT             EDGAR fair-access User-Agent (ingest edgar)
  ANTHROPIC_API_KEY          Extraction LLM (extract s-filings)
  OTEL_EXPORTER_OTLP_ENDPOINT
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY        Telemetry export to Langfuse (status)
  MLFLOW_TRACKING_URI        Defaults to file:./mlruns
  FMP_API_KEY                Reserved for ingest fmp (Issue TBD)
"""


@click.group(
    help="auto-research command-line surface.",
    epilog=_ENV_VAR_EPILOG,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """Root group. Subcommands are registered below."""
