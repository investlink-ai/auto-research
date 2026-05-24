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

import subprocess
from pathlib import Path

import click
import pyarrow.parquet as pq

from auto_research.extract.workers.s_filings import extract_s_filing
from auto_research.ingest.edgar import fetch_filings_for_cik

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


_FEAST_REPO_DIR = Path("feast_repo")

_DEFAULT_EDGAR_FORM_TYPES = ("S-3", "S-1")
_DEFAULT_RAW_ROOT = Path("data/raw")
_DEFAULT_MANIFEST = Path("data/manifest.parquet")
_S_FILING_FORM_TYPES = frozenset({"S-1", "S-3"})
_DEFAULT_EXTRACTED_ROOT = Path("data/extracted")


@click.group(
    help="auto-research command-line surface.",
    epilog=_ENV_VAR_EPILOG,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None: ...


@cli.group(help="Ingest raw documents into data/raw/ + manifest.")
def ingest() -> None: ...


@ingest.command("edgar", help="Fetch SEC EDGAR filings for a CIK. Requires SEC_USER_AGENT.")
@click.option(
    "--cik",
    required=True,
    help="Zero-padded 10-digit CIK (e.g., 0001045810 for NVDA).",
)
@click.option(
    "--form-types",
    default=",".join(_DEFAULT_EDGAR_FORM_TYPES),
    show_default=True,
    help="Comma-separated form types (e.g., 'S-3,S-1,10-K').",
)
@click.option(
    "--raw-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_RAW_ROOT,
    show_default=True,
    help="Root directory for raw bytes.",
)
@click.option(
    "--manifest-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="Manifest Parquet file.",
)
def ingest_edgar(
    cik: str,
    form_types: str,
    raw_root: Path,
    manifest_path: Path,
) -> None:
    parsed_forms = tuple(f.strip() for f in form_types.split(",") if f.strip())
    results = fetch_filings_for_cik(
        cik,
        form_types=parsed_forms,
        raw_root=raw_root,
        manifest_path=manifest_path,
    )
    fetched = sum(1 for r in results if not r.cache_hit)
    cached = sum(1 for r in results if r.cache_hit)
    click.echo(
        f"ingest edgar: cik={cik} forms={','.join(parsed_forms)} "
        f"results={len(results)} fetched={fetched} cached={cached}"
    )


@cli.group(help="Run extraction workers over raw documents.")
def extract() -> None: ...


@extract.command(
    "s-filings",
    help="Extract dilution events from every S-1/S-3 in the manifest for --cik.",
)
@click.option("--cik", required=True, help="Zero-padded 10-digit CIK.")
@click.option(
    "--manifest-path",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="Manifest Parquet file.",
)
@click.option(
    "--out-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_EXTRACTED_ROOT,
    show_default=True,
    help="Where to persist SFilingOutput JSON (worker-keyed subdir).",
)
def extract_s_filings(cik: str, manifest_path: Path, out_root: Path) -> None:
    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    candidates = [
        r
        for r in rows
        if r["source"] == "edgar"
        and r["entity_id"] == cik
        and r["form_type"] in _S_FILING_FORM_TYPES
        and r["status"] == "ok"
    ]
    out_dir = out_root / "s_filings"
    persisted = 0
    quarantined = 0
    for row in candidates:
        raw_path = Path(row["path"])
        try:
            raw_doc = raw_path.read_text(errors="replace")
        except OSError as exc:
            click.echo(f"warn: skipping {row['doc_id']}: {exc}", err=True)
            quarantined += 1
            continue
        result = extract_s_filing(raw_doc=raw_doc, doc_id=row["doc_id"])
        if result is None:
            quarantined += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{row['doc_id']}.json").write_text(
            result.model_dump_json(indent=2)
        )
        persisted += 1
    click.echo(
        f"extract s-filings: cik={cik} candidates={len(candidates)} "
        f"persisted={persisted} quarantined={quarantined}"
    )


@cli.group(name="feast", help="Wrap the Feast CLI against feast_repo/.")
def feast_group() -> None: ...


@feast_group.command("apply", help="Run `feast apply` in feast_repo/.")
def feast_apply() -> None:
    proc = subprocess.run(["feast", "apply"], cwd=_FEAST_REPO_DIR, check=False)
    raise SystemExit(proc.returncode)


@feast_group.command(
    "materialize",
    help="Run `feast materialize START END` in feast_repo/. Dates are ISO-8601.",
)
@click.option("--start", required=True, help="Inclusive ISO date (YYYY-MM-DD).")
@click.option("--end", required=True, help="Inclusive ISO date (YYYY-MM-DD).")
def feast_materialize(start: str, end: str) -> None:
    proc = subprocess.run(
        ["feast", "materialize", start, end],
        cwd=_FEAST_REPO_DIR,
        check=False,
    )
    raise SystemExit(proc.returncode)


def _not_implemented(name: str, follow_up: str) -> click.UsageError:
    return click.UsageError(f"{name} is not yet implemented. {follow_up}")


@ingest.command("fmp", help="Fetch from Financial Modeling Prep (not yet implemented).")
@click.option("--ticker", required=False, help="Ticker symbol (e.g., NVDA).")
def ingest_fmp(ticker: str | None) -> None:
    raise _not_implemented(
        "ingest fmp",
        "FMP ingest module is planned for a follow-up issue; "
        "the EDGAR path covers W1 acceptance.",
    )


@cli.group(name="eval", help="Run eval suites against extracted outputs.")
def eval_group() -> None: ...


@eval_group.command("extract", help="DeepEval on extraction outputs (not yet implemented).")
def eval_extract() -> None:
    raise _not_implemented(
        "eval extract",
        "DeepEval suite for extraction is planned for the W1 wrap-up; "
        "see docs/plans/2026-05-22-auto-research-implementation.md.",
    )
